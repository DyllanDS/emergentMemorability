"""
02_crowding_analysis.py (faces): how crowded each face of the 10k US Adult Faces Database is among
all the faces of the database, and how that relates to memorability (SI Appendix, Section 6;
Fig. S15b).

This script runs only when the face database is available (EXTERNAL_INPUTS.md, M4): the 10,168 faces
of Bainbridge, Isola and Oliva (2013) may not be redistributed, so the module holds the saved faces
report (data/provided_faces/faces_crowding.txt) and this code. code/run_module.py calls it through
code/faces/regenerate_faces.py when the archives are in external/bainbridge_faces/.
What it does
    The same computation as code/scenes/02_crowding_analysis.py (see there): the 14 deep-network
    spaces are turned into one z-scored ensemble distance, each face's crowding (its negated mean
    ensemble distance to every other face) is computed over all 10,168 faces, unscored fillers
    included, and correlated (Spearman rho) with the memorability of the 2,222 scored faces. The
    "Limiting case" block of the report holds these whole-sample correlations; Fig. S15b plots the
    ones with the hit rate (rho = -0.32) and the false alarm rate (rho = 0.30).
Inputs (all required on the command line; regenerate_faces.py passes them)
    --embeddings-dir   the 14 face spaces <network>_<layer>.npz (10,168 x 512)
    --image-order      the face list with the scores of the database (00_build_image_order.py)
    --out-dir          for faces_crowding.txt and faces_crowding_meta.json (aggregate reports)
    --per-image-dir    for faces_crowding.npz (per-face values; face-derived, never redistributed)
    The script stops unless there are exactly 14 spaces whose stored file names follow
    image_order.csv row by row.
Seed
    numpy default_rng(0): a 1,500-image sample in each space (spaces in file-name order) for its
    distance scale, as in the scene script.
"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, norm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402

# The folders come from the command line (all required: the face database is not in the repository).
N_SPACES_EXPECTED = 14      # seven networks x early and late layer

CHUNK = 512
RNG = np.random.default_rng(0)


def show(path):
    """A path relative to the repository root when it is inside the repository, for printing."""
    relative = os.path.relpath(str(path), release_paths.release_root())
    return str(path) if relative.startswith('..') else relative


def load_spaces(EMB, order):
    """Row-normalized feature matrices (each row centered and scaled to unit length), one per
    network/layer."""
    spaces = {}
    for f in sorted(EMB.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        # every space must list its images in the row order of image_order.csv
        stored_names = [str(name) for name in z["filenames"]]
        if stored_names != order["filename"].astype(str).tolist():
            raise SystemExit(f"{show(f)}: its rows do not follow the image list (image_order.csv)")
        X = z["features"].astype(np.float32)
        X = X - X.mean(1, keepdims=True)
        n = np.linalg.norm(X, axis=1, keepdims=True)
        spaces[f.stem] = X / np.maximum(n, 1e-8)   # correlation via dot product
    return spaces


def space_scale(X, n_sample=1500):
    """Mean and SD of correlation distance in this space, from a sample."""
    idx = RNG.choice(len(X), size=min(n_sample, len(X)), replace=False)
    S = X[idx] @ X[idx].T
    d = 1.0 - S[np.triu_indices(len(idx), 1)]
    return float(d.mean()), float(d.std() + 1e-8)


def crowding(spaces, scales):
    """One chunked pass over the ensemble space: each face's mean ensemble distance to every other
    face, negated (higher = more crowded)."""
    names = list(spaces)
    N = len(next(iter(spaces.values())))
    meand = np.zeros(N, np.float32)

    for a in range(0, N, CHUNK):
        b = min(a + CHUNK, N)
        # ensemble dissimilarity for these rows: z-scored per space, averaged
        D = np.zeros((b - a, N), np.float32)
        for nm in names:
            X = spaces[nm]
            d = 1.0 - (X[a:b] @ X.T)
            mu, sd = scales[nm]
            D += (d - mu) / sd
        D /= len(names)
        # a face is not its own neighbor
        D[np.arange(b - a), np.arange(a, b)] = np.inf

        finite0 = np.isfinite(D)
        meand[a:b] = -(np.where(finite0, D, 0.0).sum(1) / finite0.sum(1))
        if (a // CHUNK) % 5 == 0:
            print(f"  rows {a}/{N}", flush=True)
    return meand


def rel(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan, np.nan
    return spearmanr(x[ok], y[ok])


def main():
    parser = argparse.ArgumentParser(description='Crowding of the Bainbridge faces in the 14-space ensemble.')
    parser.add_argument('--embeddings-dir', type=Path, required=True)
    parser.add_argument('--image-order', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--per-image-dir', type=Path, required=True)
    args = parser.parse_args()
    EMB, OUT, PER_IMAGE = args.embeddings_dir, args.out_dir, args.per_image_dir
    OUT.mkdir(parents=True, exist_ok=True)
    PER_IMAGE.mkdir(parents=True, exist_ok=True)

    order = pd.read_csv(args.image_order)
    spaces = load_spaces(EMB, order)
    if not spaces:
        raise SystemExit("no embeddings found; run shared/embedding_extraction/extract_embeddings.py first")
    if len(spaces) != N_SPACES_EXPECTED:
        raise SystemExit(f"expected {N_SPACES_EXPECTED} spaces in {show(EMB)}, found {len(spaces)}")
    print(f"{len(spaces)} network/layer spaces: {', '.join(spaces)}")
    N = len(next(iter(spaces.values())))
    if N != len(order):
        raise SystemExit(f"different row counts: embeddings {N}, image_order {len(order)}")

    scales = {nm: space_scale(X) for nm, X in spaces.items()}
    meand = crowding(spaces, scales)

    m = order["has_scores"].to_numpy() == 1
    hr = order["hit_rate"].to_numpy(float)
    fa = order["false_alarm_rate"].to_numpy(float)
    cr = order["corrected_recognition"].to_numpy(float)
    za = lambda p: norm.ppf(np.clip(p, 1e-3, 1 - 1e-3))
    dp = za(hr) - za(fa)

    fh = open(OUT / "faces_crowding.txt", "w", encoding="utf-8")

    def w(s=""):
        print(s)
        fh.write(s + "\n")

    w("REPRESENTATIONAL CROWDING AND FACE MEMORABILITY")
    w("=" * 96)
    w(f"{N:,} faces embedded; {int(m.sum()):,} carry memorability scores.")
    w(f"Ensemble of {len(spaces)} network/layer spaces, each z-scored before "
      f"averaging.")
    w("Crowding is computed over all faces; only scored faces enter the "
      "correlations.")
    w("Spearman rho throughout. Prediction: crowding raises false alarms and "
      "lowers hits.")
    w()

    w("-" * 96)
    w("Limiting case: mean ensemble distance to every other face (negated)")
    w("-" * 96)
    for lab, v in (('hit rate', hr), ('FA rate', fa)):
        r, pp = rel(meand[m], v[m])
        w(f"  {lab:10s} rho = {r:+.3f} (p = {pp:.3g})")
    w()
    w("-" * 96)
    w("NOTES")
    w("-" * 96)
    w("  The block above is summed similarity with equal weights in the limit of")
    w("  a wide kernel: the negated mean ensemble distance from each face to")
    w("  every other face, over the whole sample (Fig. S15b).")
    fh.close()

    # the crowding measure (mean_distance_crowding) and the file names are saved too, so the
    # statistics script can recompute the Fig. S15b correlations at full precision.
    # Per-face values: written to the per-image folder (face-derived data are never redistributed).
    np.savez_compressed(PER_IMAGE / "faces_crowding.npz",
                        mean_distance_crowding=meand,
                        filenames=order["filename"].astype(str).to_numpy(),
                        has_scores=m, cr=cr, hr=hr, fa=fa, dprime=dp)
    json.dump({"spaces": list(spaces), "n_faces": int(N),
               "n_scored": int(m.sum())},
              open(OUT / "faces_crowding_meta.json", "w"), indent=2)
    print(f"\nwrote {show(OUT / 'faces_crowding.txt')}")


if __name__ == "__main__":
    main()
