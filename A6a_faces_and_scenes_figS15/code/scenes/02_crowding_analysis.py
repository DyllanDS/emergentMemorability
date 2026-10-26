"""
02_crowding_analysis.py (scenes): how crowded each Isola scene is among all the images of its
database, and how that relates to memorability (SI Appendix, Section 6; Fig. S15b).

What it does
    The 10,442 images of Isola et al. (2011), 2,400 targets and 8,042 fillers, are embedded in 14
    deep-network spaces (seven networks, an early and a late layer each). In every space each image's
    correlation distance to every other image is computed and z-scored with that space's mean and SD
    (from a 1,500-image sample), and the 14 z-scored distances are averaged into one ensemble
    distance; z-scoring puts the networks on a common scale, since their raw distances differ. Each
    image's crowding is its mean ensemble distance to every other image, negated: summed similarity
    with equal weights in the limit of a wide kernel. Crowding is computed over all 10,442 images,
    because participants saw the unscored fillers as well as the targets and the fillers are part of
    what crowds a target. It is correlated (Spearman rho) with the hit rate and the false alarm rate of
    the 2,222 scored targets of the CVPR 2011 paper.
    The "Limiting case" block of the report holds these whole-sample (not held-out) correlations, which
    Fig. S15b plots (hit rate rho = -0.51, false alarm rate rho = 0.13).
    Prediction, from the mirror effect and the REM account of it: crowding should raise false alarms
    and lower hits.
Inputs (the defaults; folders can be given on the command line)
    shared/embeddings_isola/regenerated_<network>_<layer>.npz   14 spaces, 10,442 x 512
        (shared/embeddings_isola/README.md)
    A6a_faces_and_scenes_figS15/data/scenes/image_order.csv     the row order of the embeddings, with
        the public Isola scores and the flag of the 2,222 targets of the CVPR 2011 paper.
    The script stops unless there are exactly 14 spaces whose stored file names follow
    image_order.csv row by row.
Outputs (A6a_faces_and_scenes_figS15/outputs/scenes/)
    isola_crowding.txt         the report
    isola_crowding_meta.json   the spaces and the image counts
    isola_crowding.npz         per-image values: the crowding measure (mean_distance_crowding), the
                               scores and the file names
Seed
    numpy default_rng(0): a 1,500-image sample in each space (spaces in file-name order) for its
    distance scale.
Run (from the repository root; code/run_module.py runs it)
    python A6a_faces_and_scenes_figS15/code/scenes/02_crowding_analysis.py
    Under a minute and under 1 GB of memory.
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

# The folders come from the command line (defaults: the folders of this repository), see main().
MODULE = Path(release_paths.module_path(__file__))
DEFAULT_EMB = Path(release_paths.shared_path('embeddings_isola'))
DEFAULT_ORDER = MODULE / "data" / "scenes" / "image_order.csv"
DEFAULT_OUT = MODULE / "outputs" / "scenes"
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
    """One chunked pass over the ensemble space: each image's mean ensemble distance to every other
    image, negated (higher = more crowded)."""
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
        # an image is not its own neighbor
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
    parser = argparse.ArgumentParser(description='Crowding of the Isola scenes in the 14-space ensemble.')
    parser.add_argument('--embeddings-dir', type=Path, default=DEFAULT_EMB)
    parser.add_argument('--image-order', type=Path, default=DEFAULT_ORDER)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    EMB, OUT = args.embeddings_dir, args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)

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

    # Every image is context, but only the target repeats (about 100 images
    # after the first showing) give a hit rate comparable to the object and
    # face data. Filler "hits" come from vigilance repeats a few trials later
    # and index a different retention interval, so only targets enter the
    # correlations.
    m = (order["has_scores"].to_numpy() == 1)
    if "kind" in order.columns:
        m &= (order["kind"].to_numpy() == "target")
    if "in_cvpr_2222" in order.columns:
        m &= (order["in_cvpr_2222"].to_numpy() == 1)
    hr = order["hit_rate"].to_numpy(float)
    fa = order["false_alarm_rate"].to_numpy(float)
    cr = order["corrected_recognition"].to_numpy(float)
    za = lambda p: norm.ppf(np.clip(p, 1e-3, 1 - 1e-3))
    dp = za(hr) - za(fa)

    fh = open(OUT / "isola_crowding.txt", "w", encoding="utf-8")

    def w(s=""):
        print(s)
        fh.write(s + "\n")

    w("REPRESENTATIONAL CROWDING AND SCENE MEMORABILITY")
    w("=" * 96)
    w(f"{N:,} images embedded; {int(m.sum()):,} carry memorability scores.")
    w(f"Ensemble of {len(spaces)} network/layer spaces, each z-scored before "
      f"averaging.")
    w("Crowding is computed over all images; only CVPR target images enter the "
      "correlations.")
    w("Spearman rho throughout. Prediction: crowding raises false alarms and "
      "lowers hits.")
    w()

    w("-" * 96)
    w("Limiting case: mean ensemble distance to every other image (negated)")
    w("-" * 96)
    for lab, v in (('hit rate', hr), ('FA rate', fa)):
        r, pp = rel(meand[m], v[m])
        w(f"  {lab:10s} rho = {r:+.3f} (p = {pp:.3g})")
    w()
    w("-" * 96)
    w("NOTES")
    w("-" * 96)
    w("  The block above is summed similarity with equal weights in the limit of")
    w("  a wide kernel: the negated mean ensemble distance from each image to")
    w("  every other image, over the whole sample (Fig. S15b).")
    fh.close()

    # the crowding measure (mean_distance_crowding) and the file names are saved too, so the
    # statistics script can recompute the Fig. S15b correlations at full precision.
    np.savez_compressed(OUT / "isola_crowding.npz",
                        mean_distance_crowding=meand,
                        filenames=order["filename"].astype(str).to_numpy(),
                        has_scores=m, cr=cr, hr=hr, fa=fa, dprime=dp)
    json.dump({"spaces": list(spaces), "n_images": int(N),
               "n_scored": int(m.sum())},
              open(OUT / "isola_crowding_meta.json", "w"), indent=2)
    print(f"\nwrote {show(OUT / 'isola_crowding.txt')}")


if __name__ == "__main__":
    main()
