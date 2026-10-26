"""
bin_by_crowding.py: how crowded each image was in the image set it was tested in (SI Appendix,
Section 3; the crowding values define the pairs of Fig. S8b and c).

WHAT IT DOES
  For each image set, every tested image (cell) gets its crowding in the 49-dimensional space of
  fit_inversion.py, computed against the other images tested in the same set: minus the mean
  correlation distance to its 20 nearest set-mates (the measure of Fig. S8b and c; higher = its most
  similar set-mates are more similar). Correlation distance = 1 minus the Pearson correlation of two
  images' 49 values. paired_crowding.py and stats_figS8.py read the table of cells this script
  writes.

INPUTS (inside the repository)
  outputs/model_inputs/datasets_observed.csv (code/extract_datasets.py)
  outputs/fit_datasets_cells.csv (code/fit_datasets.py)
  shared/embeddings_things/<four late layers>.npz, shared/things/things_image_mapping.csv,
  shared/things/things_images.csv

OUTPUT (in outputs/)
  bin_by_crowding_cells.csv  one row per cell: the observed table (cond, image_nr, n_old, n_new, hit,
                             fa, cr), the embedding row r, the model's pred_cr, the numbers of hits and
                             false alarms (hit_c, fa_c) and the crowding (crowd_near)

SEED 0: the PCA (random_state 0); nothing else is random.

RUN (from the repository root; a few seconds):  python A3a_model_through_own_lists_figS8/code/bin_by_crowding.py
"""
import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (first: it stops Python from writing __pycache__ folders)

PATHS = release_paths.module_paths(__file__)
EMB = Path(release_paths.shared_path('embeddings_things'))            # the four late-layer embedding files
MAPPING = Path(release_paths.shared_path('things', 'things_image_mapping.csv'))
IMAGES = Path(release_paths.shared_path('things', 'things_images.csv'))
OUT = Path(PATHS['outputs'])
MODEL_INPUTS = OUT / "model_inputs"
SELECTION = ["alexnet_late", "vgg16_late", "resnet18_late", "clip_late"]
NDIM = 49
KNN = 20                         # "near" crowding: the 20 nearest set-mates
SEED = 0


def build_space():
    """The 49-d space: standardize each late layer, put the four side by side, PCA to 49."""
    mats = []
    for nm in SELECTION:
        X = np.load(EMB / f"{nm}.npz", allow_pickle=True)["features"].astype(np.float32)
        mats.append((X - X.mean(0)) / (X.std(0) + 1e-8))
    return PCA(n_components=NDIM, random_state=SEED).fit_transform(
        np.concatenate(mats, axis=1)).astype(np.float32)


def crowding_within_set(Z, rows):
    """Crowding of each image within its set (the images of `rows`): minus its mean correlation
    distance to its KNN nearest set-mates."""
    S = Z[rows]
    S = S - S.mean(1, keepdims=True)
    S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-8)
    D = 1.0 - S @ S.T
    np.fill_diagonal(D, np.nan)
    near = -np.sort(np.nan_to_num(D, nan=np.inf), axis=1)[:, :KNN].mean(axis=1)
    return near


def main():
    obs = pd.read_csv(MODEL_INPUTS / "datasets_observed.csv", float_precision="round_trip")
    cells = pd.read_csv(OUT / "fit_datasets_cells.csv", float_precision="round_trip")
    # image_nr indexes the THINGS database; the embedding is in filename order,
    # so the two have to be joined through the mapping table
    mp = pd.read_csv(MAPPING)
    mp["filename"] = mp.orig_fname.str.rsplit("/", n=1).str[-1]
    order = pd.read_csv(IMAGES).sort_values("embedding_row")
    frow = pd.Series(np.arange(len(order)), index=order.filename)
    nr2row = pd.Series(mp.filename.map(frow).to_numpy(), index=mp.image_nr)
    obs["r"] = obs.image_nr.map(nr2row)
    obs = obs.dropna(subset=["r"])
    obs["r"] = obs.r.astype(int)
    obs = obs.merge(cells[["cond", "r", "pred_cr"]], on=["cond", "r"], how="left")
    obs["hit_c"] = obs.hit * obs.n_old
    obs["fa_c"] = obs.fa * obs.n_new

    Z = build_space()
    parts = []
    for c, g in obs.groupby("cond"):
        rows = g.r.to_numpy()
        near = crowding_within_set(Z, rows)
        gg = g.copy()
        gg["crowd_near"] = near
        parts.append(gg)
    d = pd.concat(parts, ignore_index=True)

    d.to_csv(OUT / "bin_by_crowding_cells.csv", index=False)
    print(f"\nwrote outputs/bin_by_crowding_cells.csv")


if __name__ == "__main__":
    main()
