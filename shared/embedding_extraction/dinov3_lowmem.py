# -*- coding: utf-8 -*-
"""
DINOv3 ViT-B/16 embeddings for the 10,442 Isola images, computed with low memory.

extract_embeddings.py (in this folder) holds every image's full activation vector in memory before the
PCA. For DINOv3 that is about 200,000 values per image for each of two layers, about 17 GB for the Isola
set. This script computes the same layers with less memory:

    1. The model, layers, images and preprocessing of extract_dinov3() in extract_embeddings.py:
       DINOv3 ViT-B/16 (LVD-1689M weights, loaded from timm 'vit_base_patch16_dinov3.lvd1689m'),
       images resized to 256 x 256 and ImageNet-normalized, early layer = output of blocks.6 without
       the 5 storage tokens (1 CLS + 4 registers), late layer = output of the final norm (all 261
       tokens), batch size 16.
    2. Each layer's activations are written to a file on disk (a numpy memmap) instead of being held
       in memory.
    3. PCA to 512 components is computed exactly through the image-by-image Gram matrix (N x N,
       N = 10,442): center every feature, accumulate X X^T over chunks of features, take its top 512
       eigenvectors. The principal-component scores are U * sqrt(lambda); the cumulative explained
       variance is sum(top 512 lambda) / trace. extract_embeddings.py uses sklearn
       PCA(svd_solver='auto', random_state=0), which for this size is a randomized SVD that
       approximates the same top 512 components. Scores may differ in sign per component (PCA signs
       are arbitrary), which does not change any distance, correlation or regression prediction
       computed from them.

Runs on Apple MPS or the CPU. Output: <out-dir>/dinov3_early.npz and dinov3_late.npz in the format of
extract_embeddings.py (features float32 N x 512, filenames, network, layer_label, pca_applied,
original_dim, explained_variance_ratio, and pca_method) and an updated extraction_manifest.json.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

HERE = Path(__file__).resolve().parent
# extract_embeddings.py in this folder provides the image loading.
EXTRACTOR = HERE / "extract_embeddings.py"


def load_extractor():
    """Import extract_embeddings.py from this folder, so the images are loaded by the same code."""
    spec = importlib.util.spec_from_file_location("extract_embeddings", EXTRACTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--image-order", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--scratch-dir", type=Path, required=True, help="where the large activation files are written")
    ap.add_argument("--n-components", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--feature-chunk", type=int, default=4096, help="features per chunk when building the Gram matrix")
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    extractor = load_extractor()
    images, order = extractor.load_faces(args.image_order, args.images_dir)
    filenames = order["filename"].tolist()
    number_of_images = len(images)
    print(f"images: {number_of_images}", flush=True)

    # ---- the model, as in extract_dinov3() of extract_embeddings.py ----
    import timm
    model = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, img_size=256)
    if getattr(model, "num_prefix_tokens", 5) != 5:
        sys.exit(f"DINOv3 timm model has {model.num_prefix_tokens} prefix tokens, expected 5 (1 CLS + 4 registers)")
    model.eval().to(device)
    patch = 16
    resize_dim = (256 // patch) * patch
    transform = transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    number_of_storage_tokens = 5
    layer_names = {"early": "blocks.6", "late": "norm"}

    # ---- find each layer's flattened width with a two-image probe ----
    modules = dict(model.named_modules())
    widths = {}
    captured = {}

    def make_hook(label):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if label == "early":
                output = output[:, number_of_storage_tokens:, :]
            captured[label] = output.detach().flatten(1).float().cpu().numpy()
        return hook

    handles = [modules[name].register_forward_hook(make_hook(label)) for label, name in layer_names.items()]
    with torch.no_grad():
        model(torch.stack([transform(images[0]), transform(images[1])]).to(device))
    for label in layer_names:
        widths[label] = captured[label].shape[1]
    captured.clear()
    print(f"layer widths: {widths}", flush=True)

    # ---- forward pass: write each layer's activations to a disk file ----
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    activation_files = {}
    for label, width in widths.items():
        path = args.scratch_dir / f"dinov3_{label}_activations.npy"
        activation_files[label] = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(number_of_images, width))
    start = time.time()
    with torch.no_grad():
        for first in range(0, number_of_images, args.batch_size):
            last = min(first + args.batch_size, number_of_images)
            batch = torch.stack([transform(images[i]) for i in range(first, last)]).to(device)
            model(batch)
            for label in layer_names:
                activation_files[label][first:last] = captured[label]
            captured.clear()
            if (first // args.batch_size) % 50 == 0:
                print(f"  forward {last}/{number_of_images}  {time.time() - start:.0f}s", flush=True)
    for handle in handles:
        handle.remove()
    for label in layer_names:
        activation_files[label].flush()
    del model
    if device == "mps":
        torch.mps.empty_cache()
    print(f"forward pass done in {time.time() - start:.0f}s", flush=True)

    # ---- exact PCA through the image-by-image Gram matrix, one layer at a time ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    for label in layer_names:
        activations = np.load(args.scratch_dir / f"dinov3_{label}_activations.npy", mmap_mode="r")
        number_of_features = activations.shape[1]
        print(f"[{label}] {number_of_images} x {number_of_features}: building the Gram matrix", flush=True)

        # mean of every feature (first sweep over the feature chunks)
        feature_means = np.zeros(number_of_features, dtype=np.float64)
        for chunk_start in range(0, number_of_features, args.feature_chunk):
            chunk = np.asarray(activations[:, chunk_start:chunk_start + args.feature_chunk], dtype=np.float64)
            feature_means[chunk_start:chunk_start + chunk.shape[1]] = chunk.mean(axis=0)

        # Gram matrix of the centered data (second sweep)
        gram = np.zeros((number_of_images, number_of_images), dtype=np.float64)
        for chunk_start in range(0, number_of_features, args.feature_chunk):
            chunk = np.asarray(activations[:, chunk_start:chunk_start + args.feature_chunk], dtype=np.float64)
            chunk -= feature_means[chunk_start:chunk_start + chunk.shape[1]]
            gram += chunk @ chunk.T
        total_variance = float(np.trace(gram))

        # top eigenvectors of the Gram matrix = principal-component directions in image space
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        del gram
        top = np.argsort(eigenvalues)[::-1][:args.n_components]
        top_eigenvalues = np.clip(eigenvalues[top], 0, None)
        scores = eigenvectors[:, top] * np.sqrt(top_eigenvalues)
        explained_variance_ratio = (top_eigenvalues / total_variance).astype(np.float32)
        cumulative = float(explained_variance_ratio.sum())
        print(f"    PCA dinov3/{label}: {number_of_features}D -> {args.n_components}D, "
              f"cumulative EVR={cumulative:.3f}", flush=True)

        out_path = args.out_dir / f"dinov3_{label}.npz"
        np.savez_compressed(
            out_path,
            features=scores.astype(np.float32),
            filenames=np.asarray(filenames, dtype=object),
            network="dinov3",
            layer_label=label,
            pca_applied=True,
            original_dim=number_of_features,
            explained_variance_ratio=explained_variance_ratio,
            pca_method="exact PCA via the image-by-image Gram matrix (dinov3_lowmem.py)",
        )
        print(f"    wrote {out_path.name}  shape={scores.shape}", flush=True)
        manifest_entries.append({"network": "dinov3", "layer_label": label, "shape": list(scores.shape),
                                 "original_dim": number_of_features, "pca_applied": True,
                                 "cumulative_evr": cumulative,
                                 "pca_method": "exact PCA via Gram matrix (low-memory route)"})
        del activations

    manifest_path = args.out_dir / "extraction_manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    keyed = {(m["network"], m["layer_label"]): m for m in existing}
    for entry in manifest_entries:
        keyed[(entry["network"], entry["layer_label"])] = entry
    manifest_path.write_text(json.dumps(list(keyed.values()), indent=2))

    for label in layer_names:
        os.remove(args.scratch_dir / f"dinov3_{label}_activations.npy")
    print("done; activation files removed", flush=True)


if __name__ == "__main__":
    main()
