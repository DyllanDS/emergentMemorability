"""
extract_embeddings.py: early- and late-layer embeddings of a list of images from seven networks
(AlexNet, VGG16-BN, ResNet18, InceptionV3, CORnet-S, DINOv3 ViT-B/16, CLIP ViT-B/32); larger layers
are reduced to 512 principal components.

The THINGS embeddings of shared/embeddings_things/ and the Isola scene embeddings of
shared/embeddings_isola/ come from this script (DINOv3 on the scenes: dinov3_lowmem.py in this
folder). README.md in this folder lists the models, layers, weights and preprocessing, and says which
DINOv3 weight source made which files.

Device: CUDA when available, else Apple MPS (some pooling steps then run on the CPU), else the CPU;
the model, weights and arithmetic are the same. A missing cornet or clip package, or a DINOv3 model
without its 12 blocks or its final norm, stops the run with an error; --only leaves a network out on
purpose. Needs torch, torchvision, timm, scikit-learn, the OpenAI clip package and cornet (install
lines in README.md).

Usage (from the repository root; MPLBACKEND=Agg):
    python shared/embedding_extraction/extract_embeddings.py --images-dir <folder> --image-order <csv> \
        --out-dir <folder> [--only alexnet vgg16 ...]
For THINGS the image list is shared/things/things_images.csv sorted by embedding_row (README.md
writes it out).

When a layer's flattened width exceeds --pca-threshold (default 512), it is reduced to
--pca-components (default 512) principal components, and the explained-variance ratio is saved with
the reduced matrix.

Row order: the rows of every output .npz follow the rows of the image list given with --image-order.
"""

import os
# PyTorch and the numerical libraries can each load their own OpenMP runtime on macOS, which otherwise stops the process
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

# The image folder, the image list and the output folder are required arguments.

PCA_THRESHOLD_DEFAULT = 512
PCA_COMPONENTS_DEFAULT = 512


# ======================================================================
# Data loading
# ======================================================================

class LazyImages:
    """A sequence of images that decodes on access rather than up front.

    Every model here resizes to 256 anyway, so JPEG draft mode decodes
    straight to roughly that scale, which is both faster and smaller than
    decoding at full resolution and then downsampling.
    """

    def __init__(self, paths, target=256):
        self.paths = list(paths)
        self.target = target

    def __len__(self):
        return len(self.paths)

    def _open(self, p):
        im = Image.open(p)
        try:
            im.draft('RGB', (self.target, self.target))
        except Exception:
            pass
        return im.convert('RGB')

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self._open(p) for p in self.paths[i]]
        return self._open(self.paths[i])


def load_faces(image_order_csv, images_dir):
    """Load all images listed in the image list (--image-order), in that exact row order.

    The image list sets the row order of every embeddings .npz, so this function is the only place
    images get opened.
    """
    if not image_order_csv.exists():
        sys.exit(f'ERROR: {image_order_csv} not found '
                  f'(see shared/embedding_extraction/README.md for the image lists).')
    order = pd.read_csv(image_order_csv)
    if 'filename' not in order.columns:
        sys.exit(f'ERROR: {image_order_csv} has no "filename" column.')

    # Some image sets nest images in per-concept subfolders. Where the image list
    # carries a rel_path column it is used directly (a search per file would walk
    # the tree tens of thousands of times).
    # Resolve paths now, decode later. THINGS images are 800x800, so holding
    # 26,107 of them decoded would need about 41 GB; LazyImages opens each one
    # when its batch comes up and lets it go afterwards.
    rels = order['rel_path'] if 'rel_path' in order.columns else None
    paths, missing = [], []
    for i, fname in enumerate(order['filename']):
        path = images_dir / (rels.iloc[i] if rels is not None else fname)
        if not path.exists():
            matches = list(images_dir.rglob(fname))
            if matches:
                path = matches[0]
            else:
                missing.append(fname)
                continue
        paths.append(path)
    images = LazyImages(paths)
    if missing:
        sys.exit(f'ERROR: {len(missing)} image(s) listed in the image list '
                  f'were not found under {images_dir}, e.g. {missing[:5]}')
    return images, order


# ======================================================================
# Feature extraction utilities
# ======================================================================

# Not used by the extractors below (see get_activations_streaming)
_STREAM_EVR = {}


SPATIAL_POOL_THRESHOLD = 65536   # flattened width above which conv maps pool
SPATIAL_POOL_GRID = 8            # pooled to GRID x GRID per channel


def _pool_if_wide(out, threshold=SPATIAL_POOL_THRESHOLD, grid=SPATIAL_POOL_GRID):
    """Average-pool a conv feature map spatially when flattening it whole is
    infeasible.

    A conv feature map wider than `threshold` values (channels x height x width)
    is average-pooled to a `grid` x `grid` map per channel (VGG16-BN's early layer
    has about a million values per image). This keeps the coarse spatial layout
    and every channel. Narrower layers and the late layers are flattened
    unchanged.
    """
    if out.dim() == 4 and out.shape[1:].numel() > threshold:
        h, w = out.shape[-2:]
        if h > grid or w > grid:
            # MPS cannot do adaptive pooling when the input size is not a multiple of the output size;
            # on MPS this step runs on the CPU.
            out = torch.nn.functional.adaptive_avg_pool2d(out.cpu() if out.device.type == 'mps' else out, (grid, grid))
    return out


def get_activations(model, images, layer_names, transform, device,
                    batch_size=32, forward=None, layer_postproc=None):
    """Flattened activations per layer, every layer held in memory (wide conv
    layers are pooled first, see _pool_if_wide).
    """
    return _get_activations_buffered(
        model, images, layer_names, transform, device,
        batch_size=batch_size, forward=forward, layer_postproc=layer_postproc)


def _get_activations_buffered(model, images, layer_names, transform, device,
                              batch_size=32, forward=None, layer_postproc=None):
    """Collects every layer's activations in memory; returns {layer name: (N, D) array}."""
    model.eval().to(device)
    acts = {name: [] for name in layer_names}
    layer_postproc = layer_postproc or {}

    def make_hook(name):
        def hook(_module, _inp, out):
            while isinstance(out, (tuple, list)):
                out = out[0]
            if name in layer_postproc:
                out = layer_postproc[name](out)
            out = _pool_if_wide(out)
            acts[name].append(out.detach().flatten(1).float().cpu())
        return hook

    mod_map = dict(model.named_modules())
    handles = []
    try:
        for name in layer_names:
            if name not in mod_map:
                available = [n for n in mod_map if len(n) > 0][:20]
                raise KeyError(
                    f'Layer {name!r} not found in model. '
                    f'First few available: {available}'
                )
            handles.append(mod_map[name].register_forward_hook(make_hook(name)))

        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size),
                          desc=f'  forward ({len(images)} imgs)'):
                batch = torch.stack([transform(img) for img in images[i:i+batch_size]]).to(device)
                if forward is None:
                    model(batch)
                else:
                    forward(batch)
    finally:
        for h in handles:
            h.remove()

    return {name: torch.cat(acts[name], dim=0).numpy() for name in layer_names}


def get_activations_streaming(model, images, layer_names, transform, device,
                              batch_size=32, forward=None, layer_postproc=None,
                              n_components=512, budget_bytes=6 * 2**30):
    """Not used by the extractors below, which call get_activations().

    A variant of get_activations() that never holds a full high-dimensional
    layer: a layer whose full matrix would exceed `budget_bytes` is reduced with
    an IncrementalPCA fitted across batches (a second forward pass), and the
    narrower layers are collected in the same pass. It does not pool wide conv
    layers (_pool_if_wide).

    Returns {layer_name: (N, D) array}, as get_activations does, and
    {layer_name: cumulative explained-variance ratio, or None}.
    """
    from sklearn.decomposition import IncrementalPCA

    model.eval().to(device)
    layer_postproc = layer_postproc or {}
    mod_map = dict(model.named_modules())
    for name in layer_names:
        if name not in mod_map:
            raise KeyError(f'Layer {name!r} not found in model.')

    def run_pass(collect):
        """collect(name, tensor) is called once per layer per batch."""
        captured = {}

        def make_hook(nm):
            def hook(_m, _i, out):
                while isinstance(out, (tuple, list)):
                    out = out[0]
                if nm in layer_postproc:
                    out = layer_postproc[nm](out)
                captured[nm] = out.detach().flatten(1).float().cpu()
            return hook

        handles = [mod_map[nm].register_forward_hook(make_hook(nm))
                   for nm in layer_names]
        try:
            with torch.no_grad():
                for i in tqdm(range(0, len(images), batch_size),
                              desc=f'  forward ({len(images)} imgs)'):
                    batch = torch.stack(
                        [transform(im) for im in images[i:i+batch_size]]).to(device)
                    if forward is None:
                        model(batch)
                    else:
                        forward(batch)
                    for nm in layer_names:
                        collect(nm, captured[nm])
                    captured.clear()
        finally:
            for h in handles:
                h.remove()

    # ---- probe one batch to learn each layer's flattened width ----
    dims = {}
    probe = torch.stack([transform(im) for im in images[:2]]).to(device)
    cap = {}

    def probe_hook(nm):
        def hook(_m, _i, out):
            while isinstance(out, (tuple, list)):
                out = out[0]
            if nm in layer_postproc:
                out = layer_postproc[nm](out)
            cap[nm] = out.detach().flatten(1).shape[1]
        return hook

    hs = [mod_map[nm].register_forward_hook(probe_hook(nm)) for nm in layer_names]
    try:
        with torch.no_grad():
            model(probe) if forward is None else forward(probe)
    finally:
        for h in hs:
            h.remove()
    dims = dict(cap)

    N = len(images)
    big = {nm for nm, d in dims.items()
           if N * d * 4 > budget_bytes and d > n_components}
    small = [nm for nm in layer_names if nm not in big]
    for nm in big:
        print(f'    {nm}: {dims[nm]}D x {N} would need '
              f'{N*dims[nm]*4/2**30:.1f} GB: streaming through IncrementalPCA',
              flush=True)

    ipca = {nm: IncrementalPCA(n_components=n_components) for nm in big}
    buf = {nm: [] for nm in small}

    # pass 1: fit the big layers, collect the small ones
    def collect1(nm, x):
        if nm in big:
            if x.shape[0] >= n_components:
                ipca[nm].partial_fit(x.numpy())
        else:
            buf[nm].append(x)
    run_pass(collect1)

    out = {nm: torch.cat(buf[nm], 0).numpy() for nm in small}
    evr = {nm: None for nm in small}

    # pass 2: transform the big layers
    if big:
        parts = {nm: [] for nm in big}

        def collect2(nm, x):
            if nm in big:
                parts[nm].append(ipca[nm].transform(x.numpy()).astype(np.float32))
        run_pass(collect2)
        for nm in big:
            out[nm] = np.concatenate(parts[nm], 0)
            evr[nm] = float(ipca[nm].explained_variance_ratio_.sum())
            print(f'    {nm}: IncrementalPCA -> {out[nm].shape[1]}D, '
                  f'cumulative EVR={evr[nm]:.3f}', flush=True)
    return out, evr


class PadToSquare:
    """Pad the shorter side out to a square with white.

    Images of different aspect ratios then stack into one batch after the resize,
    without cropping. A square image is returned unchanged.
    """

    def __call__(self, img):
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        out = Image.new(img.mode, (side, side), 'white')
        out.paste(img, ((side - w) // 2, (side - h) // 2))
        return out


def thingsvision_transform(resize_dim=256):
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((resize_dim, resize_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def save_features(out_dir, network, layer_label, features, filenames,
                   pca_threshold, pca_components):
    """Save an (N, D) feature matrix to <out-dir>/<network>_<layer_label>.npz.

    PCA-reduces to `pca_components` when the raw dimensionality exceeds
    `pca_threshold`; in that case both the reduced matrix and the PCA
    explained_variance_ratio_ are saved, and `pca_applied=True` records it.
    """
    n, d = features.shape
    pca_applied = False
    explained_variance_ratio = None
    original_dim = d

    if d > pca_threshold:
        from sklearn.decomposition import PCA
        n_components = min(pca_components, d, n - 1)
        pca = PCA(n_components=n_components, svd_solver='auto', random_state=0)
        features = pca.fit_transform(features).astype(np.float32)
        explained_variance_ratio = pca.explained_variance_ratio_.astype(np.float32)
        pca_applied = True
        print(f'    PCA {network}/{layer_label}: {original_dim}D -> '
              f'{n_components}D, cumulative EVR={explained_variance_ratio.sum():.3f}')

    out_path = out_dir / f'{network}_{layer_label}.npz'
    kwargs = dict(
        features=features.astype(np.float32),
        filenames=np.asarray(filenames, dtype=object),
        network=network,
        layer_label=layer_label,
        pca_applied=pca_applied,
        original_dim=original_dim,
    )
    if pca_applied:
        kwargs['explained_variance_ratio'] = explained_variance_ratio
    np.savez_compressed(out_path, **kwargs)
    print(f'    wrote {out_path.name}  shape={features.shape}  '
          f'pca_applied={pca_applied}')
    return {
        'network': network, 'layer_label': layer_label,
        'shape': list(features.shape), 'original_dim': original_dim,
        'pca_applied': pca_applied,
        'cumulative_evr': float(explained_variance_ratio.sum()) if pca_applied else None,
    }




# Apple MPS cannot run AdaptiveAvgPool2d when the input size is not a multiple of the output size (AlexNet pools
# 7x7 -> 6x6 and VGG16-BN pools 8x8 -> 7x7 at this 256-pixel input). This wrapper runs only that pooling step on
# the CPU and moves the result back. Same layer, same math; only where it is computed changes.
class _PoolOnCPU(torch.nn.Module):
    def __init__(self, pool):
        super().__init__()
        self.pool = pool

    def forward(self, x):
        return self.pool(x.cpu()).to(x.device)


def _route_avgpool_to_cpu_if_mps(model, device):
    if str(device) == 'mps' and hasattr(model, 'avgpool'):
        model.avgpool = _PoolOnCPU(model.avgpool)
    return model

# ======================================================================
# Model-specific extractors
# ======================================================================

def extract_alexnet(images, filenames, device, out_dir, pca_t, pca_c, manifest):
    print('\n[AlexNet] loading torchvision.models.alexnet (ImageNet)')
    from torchvision.models import alexnet, AlexNet_Weights
    model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    model = _route_avgpool_to_cpu_if_mps(model, device)
    feats = get_activations(model, images, ['features.3', 'avgpool'],
                            thingsvision_transform(), device, batch_size=32)
    manifest.append(save_features(out_dir, 'alexnet', 'early', feats['features.3'], filenames, pca_t, pca_c))
    manifest.append(save_features(out_dir, 'alexnet', 'late', feats['avgpool'], filenames, pca_t, pca_c))


def extract_vgg16(images, filenames, device, out_dir, pca_t, pca_c, manifest):
    print('\n[VGG16-BN] loading torchvision.models.vgg16_bn (ImageNet)')
    from torchvision.models import vgg16_bn, VGG16_BN_Weights
    model = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
    model = _route_avgpool_to_cpu_if_mps(model, device)
    feats = get_activations(model, images, ['features.14', 'avgpool'],
                            thingsvision_transform(), device, batch_size=16)
    manifest.append(save_features(out_dir, 'vgg16', 'early', feats['features.14'], filenames, pca_t, pca_c))
    manifest.append(save_features(out_dir, 'vgg16', 'late', feats['avgpool'], filenames, pca_t, pca_c))


def extract_resnet18(images, filenames, device, out_dir, pca_t, pca_c, manifest):
    print('\n[ResNet18] loading torchvision.models.resnet18 (ImageNet)')
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    feats = get_activations(model, images, ['layer2', 'avgpool'],
                            thingsvision_transform(), device, batch_size=32)
    manifest.append(save_features(out_dir, 'resnet18', 'early', feats['layer2'], filenames, pca_t, pca_c))
    manifest.append(save_features(out_dir, 'resnet18', 'late', feats['avgpool'], filenames, pca_t, pca_c))


def extract_inception(images, filenames, device, out_dir, pca_t, pca_c, manifest):
    print('\n[InceptionV3] loading torchvision.models.inception_v3 (ImageNet)')
    from torchvision.models import inception_v3, Inception_V3_Weights
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    feats = get_activations(model, images, ['Mixed_5b', 'avgpool'],
                            thingsvision_transform(), device, batch_size=32)
    manifest.append(save_features(out_dir, 'inception', 'early', feats['Mixed_5b'], filenames, pca_t, pca_c))
    manifest.append(save_features(out_dir, 'inception', 'late', feats['avgpool'], filenames, pca_t, pca_c))


def extract_cornet(images, filenames, device, out_dir, pca_t, pca_c, manifest):
    print('\n[CORnet-S] loading cornet.cornet_s (pretrained)')
    try:
        import cornet
    except ImportError:
        # A missing package stops the run, so a missing model never passes unnoticed; use --only to
        # leave a network out on purpose.
        sys.exit('ERROR: the `cornet` package is not installed, so CORnet-S cannot be loaded. '
                 'Install with: pip install git+https://github.com/dicarlolab/CORnet '
                 '(or leave CORnet-S out on purpose with --only)')
    if not torch.cuda.is_available():
        import functools
        _orig_load = torch.load
        torch.load = functools.partial(_orig_load, map_location='cpu', weights_only=False)
        try:
            model = cornet.cornet_s(pretrained=True, map_location='cpu')
        except TypeError:
            model = cornet.cornet_s(pretrained=True)
        finally:
            torch.load = _orig_load
    else:
        model = cornet.cornet_s(pretrained=True)
    submods = dict(model.named_modules())
    v2_name = 'module.V2' if 'module.V2' in submods else 'V2'
    it_name = 'module.IT' if 'module.IT' in submods else 'IT'
    feats = get_activations(model, images, [v2_name, it_name],
                            thingsvision_transform(), device, batch_size=16)
    manifest.append(save_features(out_dir, 'cornet', 'early', feats[v2_name], filenames, pca_t, pca_c))
    manifest.append(save_features(out_dir, 'cornet', 'late', feats[it_name], filenames, pca_t, pca_c))


def extract_dinov3(images, filenames, device, out_dir, pca_t, pca_c, manifest):
    # DINOv3 ViT-B/16 with the LVD-1689M weights of timm's port 'vit_base_patch16_dinov3.lvd1689m' (the torch.hub
    # download facebookresearch/dinov3 'dinov3_vitb16' of the same weights is gated). Layers: blocks.6 without the
    # 5 storage tokens, and the final norm; 256-pixel input.
    print('\n[DinoV3 ViT-B/16] loading timm vit_base_patch16_dinov3.lvd1689m')
    import timm
    model = timm.create_model('vit_base_patch16_dinov3.lvd1689m', pretrained=True, img_size=256)
    if getattr(model, 'num_prefix_tokens', 5) != 5:
        sys.exit(f'DINOv3 timm model has {model.num_prefix_tokens} prefix tokens, expected 5 (1 CLS + 4 registers)')
    patch = 16
    resize_dim = (256 // patch) * patch
    tx = transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    blocks = getattr(model, 'blocks', None)
    if blocks is None or len(blocks) < 12:
        # Without 12 blocks the early and late layers cannot be read: stop (see extract_cornet).
        sys.exit(f'ERROR: the DINOv3 model does not have the expected 12 transformer blocks (blocks={blocks}), '
                 'so its early and late layers cannot be read')
    early_name = 'blocks.6'
    # The late layer is the final norm (README.md); a model without it stops the run.
    if not hasattr(model, 'norm'):
        sys.exit('ERROR: the DINOv3 model has no final norm layer, which is the DINOv3 late layer')
    late_name = 'norm'
    # DINOv3 ViTs prepend 1 CLS + 4 register tokens; they are removed from the
    # early layer so that per-patch content (not storage-token variance)
    # dominates the flattened feature vector.
    n_storage = 5
    feats = get_activations(
        model, images, [early_name, late_name],
        tx, device, batch_size=16,
        layer_postproc={
            early_name: lambda t, ns=n_storage: t[:, ns:, :].contiguous(),
        },
    )
    manifest.append(save_features(out_dir, 'dinov3', 'early', feats[early_name], filenames, pca_t, pca_c))
    manifest.append(save_features(out_dir, 'dinov3', 'late', feats[late_name], filenames, pca_t, pca_c))


def extract_clip(images, filenames, device, out_dir, pca_t, pca_c, manifest):
    print('\n[CLIP ViT-B/32] loading OpenAI CLIP')
    try:
        import clip
    except ImportError:
        # A missing package stops the run (see extract_cornet).
        sys.exit('ERROR: the `clip` package is not installed, so CLIP ViT-B/32 cannot be loaded. '
                 'Install with: pip install git+https://github.com/openai/CLIP.git '
                 '(or leave CLIP out on purpose with --only)')
    model, preprocess = clip.load('ViT-B/32', device=device)
    # CLIP's own preprocessing center-crops, which would cut part of a non-square
    # image; padding to a square first makes the crop a no-op
    preprocess = transforms.Compose([PadToSquare(), preprocess])
    model = model.float()
    model_visual = model.visual
    feats = get_activations(
        model_visual, images,
        ['transformer.resblocks.6', 'ln_post'],
        preprocess, device, batch_size=32,
        forward=lambda b: model_visual(b.to(next(model_visual.parameters()).dtype)),
        layer_postproc={
            'transformer.resblocks.6': lambda t: t.transpose(0, 1).contiguous(),
        },
    )
    manifest.append(save_features(out_dir, 'clip', 'early', feats['transformer.resblocks.6'], filenames, pca_t, pca_c))
    manifest.append(save_features(out_dir, 'clip', 'late', feats['ln_post'], filenames, pca_t, pca_c))


EXTRACTORS = {
    'alexnet':   extract_alexnet,
    'vgg16':     extract_vgg16,
    'resnet18':  extract_resnet18,
    'inception': extract_inception,
    'cornet':    extract_cornet,
    'dinov3':    extract_dinov3,
    'clip':      extract_clip,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description='Extract early- and late-layer embeddings of a list of images (seven networks).')
    ap.add_argument('--only', nargs='+', choices=list(EXTRACTORS),
                    help='Run only the named extractors (default: all).')
    ap.add_argument('--images-dir', type=Path, required=True,
                    help='folder of the images (THINGS: the unzipped images folder, with one subfolder per concept)')
    ap.add_argument('--image-order', type=Path, required=True,
                    help='csv with columns filename (and rel_path) in the row order of the embeddings')
    ap.add_argument('--out-dir', type=Path, required=True, help='folder for the <net>_<layer>.npz files')
    ap.add_argument('--pca-threshold', type=int, default=PCA_THRESHOLD_DEFAULT,
                    help='Flattened dims above this get PCA-reduced.')
    ap.add_argument('--pca-components', type=int, default=PCA_COMPONENTS_DEFAULT)
    args = ap.parse_args(argv)

    # CUDA when available, else Apple MPS, else the CPU (same model, weights and math). Four CPU threads.
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    print(f'Loading the images listed in {args.image_order} ...')
    images, order = load_faces(args.image_order, args.images_dir)
    filenames = order['filename'].tolist()
    print(f'Loaded {len(images)} images (first size: {images[0].size})')

    args.out_dir.mkdir(exist_ok=True, parents=True)
    manifest = []
    to_run = args.only if args.only else list(EXTRACTORS)
    for name in to_run:
        EXTRACTORS[name](images, filenames, device, args.out_dir,
                         args.pca_threshold, args.pca_components, manifest)

    manifest_path = args.out_dir / 'extraction_manifest.json'
    existing = []
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
        except Exception:
            existing = []
    keyed = {(m['network'], m['layer_label']): m for m in existing}
    for m in manifest:
        keyed[(m['network'], m['layer_label'])] = m
    manifest_path.write_text(json.dumps(list(keyed.values()), indent=2))

    print(f'\nWrote {len(manifest)} embeddings file(s) this run. '
          f'Manifest ({len(keyed)} total): {manifest_path}')


if __name__ == '__main__':
    main()
