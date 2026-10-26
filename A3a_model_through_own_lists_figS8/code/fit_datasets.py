"""
fit_datasets.py: the continuous-feature REM run through the three image set experiments
(SI Appendix, Section 3; Fig. S8b and c).

WHAT IT DOES
  The same model, space and procedure as fit_inversion.py, run through every block of every one of
  the 450 participants of the image set experiments: 40 studied images, then the block's test images.
  sigma and the criterion are set from the overall hit and false alarm rates: the criterion
  sets the model's false alarm rate equal to the observed one, and sigma is the grid value whose hit
  rate comes closest to the observed hit rate. The sigma sweep uses 900 randomly chosen
  subject-blocks and 10 simulated participants per block; the full pass at the chosen sigma uses
  every subject-block and 20 simulated participants per block. The model's hit and false alarm rates
  for every image in every set (cell) are predictions, set beside the observed ones. The image sets
  overlap, so 1,023 images were tested in more than one set; paired_crowding.py compares each of them
  with itself.

INPUTS (inside the repository)
  shared/embeddings_things/alexnet_late.npz, vgg16_late.npz, resnet18_late.npz, clip_late.npz
  shared/things/things_image_mapping.csv (THINGS image number -> file name) and
      shared/things/things_images.csv (file name -> embedding row)
  outputs/model_inputs/datasets_study.csv, datasets_test.csv, datasets_observed.csv (written by
      code/extract_datasets.py)

OUTPUTS (in outputs/)
  fit_datasets.txt        the sigma sweep and the chosen operating point
  fit_datasets_cells.csv  one row per image-by-set cell: the observed table plus the model's
                          pred_hit, pred_fa, pred_cr
  The script also prints the share of the variance that the 49 components keep.

SEED 0: the PCA (scikit-learn random_state 0) and, with numpy default_rng(0), the 6,000 images of the
  density estimates, the 900 subject-blocks of the sweep (a separate generator, drawn without
  replacement over the subject-blocks in the order of participant and block), and the simulated
  participants (a fresh generator with seed 0 for each sweep value and for the full pass).

RUN (from the repository root; about 3 minutes):  python A3a_model_through_own_lists_figS8/code/fit_datasets.py
"""
import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, gaussian_kde
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
U, C_ENC = 0.30, 0.60            # storage probability per feature; probability a stored feature is copied correctly
SIGMAS = [0.1, 0.2, 0.35, 0.6, 1.0]                 # noise grid, as a fraction of each feature's SD
SEED = 0


def build_space():
    """The 49-d space: standardize each late layer, put the four side by side, PCA to 49."""
    mats = []
    for nm in SELECTION:
        X = np.load(EMB / f"{nm}.npz", allow_pickle=True)["features"].astype(np.float32)
        mats.append((X - X.mean(0)) / (X.std(0) + 1e-8))
    pca = PCA(n_components=NDIM, random_state=SEED)
    Z = pca.fit_transform(np.concatenate(mats, axis=1)).astype(np.float32)
    print(f"PCA of the {sum(m.shape[1] for m in mats):,} standardized columns: {NDIM} components keep "
          f"{pca.explained_variance_ratio_.sum():.4f} of the variance", flush=True)
    return Z


def marginals(Z, ngrid=512):
    """Each feature's distribution over THINGS: a Gaussian kernel density (bandwidth factor 0.2)
    from 6,000 randomly chosen images, tabulated on 512 points."""
    rng = np.random.default_rng(SEED)
    samp = rng.choice(len(Z), 6000, replace=False)
    grids, dens = [], []
    for j in range(Z.shape[1]):
        col = Z[samp, j]
        kde = gaussian_kde(col, bw_method=0.2)
        lo, hi = col.min(), col.max()
        pad = 0.15 * (hi - lo)
        g = np.linspace(lo - pad, hi + pad, ngrid)
        grids.append(g)
        dens.append(np.maximum(kde(g), 1e-9))
    return np.array(grids), np.array(dens)


def familiarity(Z, st, probes_idx, grids, dens, sigma, nsubj, rng):
    """Mean likelihood ratio of each probe against the traces of nsubj simulated participants."""
    n, w = len(st), Z.shape[1]
    stored = rng.random((nsubj, n, w)) < U
    correct = rng.random((nsubj, n, w)) < C_ENC
    T = np.repeat(Z[st][None], nsubj, axis=0) + rng.standard_normal((nsubj, n, w)) * sigma
    for j in range(w):
        bad = ~correct[:, :, j]
        if bad.any():
            T[:, :, j][bad] = Z[rng.integers(0, len(Z), int(bad.sum())), j]
    P = Z[probes_idx]
    logL = np.zeros((len(P), nsubj, n))
    for j in range(w):
        t = T[:, :, j]
        fT = np.maximum(np.interp(t, grids[j], dens[j],
                                  left=dens[j][0], right=dens[j][-1]), 1e-9)
        d = t[None] - P[:, j][:, None, None]
        kern = np.exp(-0.5 * (d / sigma[j]) ** 2) / (sigma[j] * np.sqrt(2 * np.pi))
        logL += np.log(np.maximum(1.0 + C_ENC * (kern / fT[None] - 1.0), 1e-12)) \
            * stored[:, :, j][None]
    return np.exp(logL).mean(axis=2)


def run(Z, blocks, test_by_block, grids, dens, sigma, nsubj, seed=SEED):
    rng = np.random.default_rng(seed)
    F, I, O, C = [], [], [], []
    for key, st in blocks:
        tt = test_by_block.get(key)
        if tt is None:
            continue
        pr, old, cond = tt
        f = familiarity(Z, st, pr, grids, dens, sigma, nsubj, rng)
        F.append(f.ravel())
        I.append(np.repeat(pr[:, None], nsubj, 1).ravel())
        O.append(np.repeat(old[:, None], nsubj, 1).ravel())
        C.append(np.repeat(np.array(cond)[:, None], nsubj, 1).ravel())
    return (np.concatenate(F), np.concatenate(I),
            np.concatenate(O).astype(bool), np.concatenate(C))


def main():
    study = pd.read_csv(MODEL_INPUTS / "datasets_study.csv")
    test = pd.read_csv(MODEL_INPUTS / "datasets_test.csv")
    obs = pd.read_csv(MODEL_INPUTS / "datasets_observed.csv", float_precision="round_trip")
    # embedding row of every file name (things_images.csv sorted by embedding_row is the row order of
    # the embedding files)
    order = pd.read_csv(IMAGES).sort_values("embedding_row")
    mp = pd.read_csv(MAPPING)
    mp["filename"] = mp.orig_fname.str.rsplit("/", n=1).str[-1]
    frow = pd.Series(np.arange(len(order)), index=order.filename)
    nr2row = pd.Series(mp.filename.map(frow).to_numpy(), index=mp.image_nr)

    for df in (study, test, obs):
        df["r"] = df.image_nr.map(nr2row)
    study = study.dropna(subset=["r"]); test = test.dropna(subset=["r"])
    obs = obs.dropna(subset=["r"])
    study["r"] = study.r.astype(int); test["r"] = test.r.astype(int)
    obs["r"] = obs.r.astype(int)

    tgt_hit = test[test.is_old == 1].said_old.mean()
    tgt_fa = test[test.is_old == 0].said_old.mean()
    za = lambda v: norm.ppf(np.clip(v, 1e-3, 1 - 1e-3))
    print(f"observed overall: hit {tgt_hit:.3f}, FA {tgt_fa:.3f}, "
          f"d' {za(tgt_hit)-za(tgt_fa):.2f}", flush=True)

    Z = build_space()
    grids, dens = marginals(Z)
    sd = Z.std(0)

    blocks = [((s, b), g.r.to_numpy()) for (s, b), g in study.groupby(["sub", "block"])]
    tb = {}
    cond_codes = {c: i for i, c in enumerate(sorted(test.cond.unique()))}
    for (s, b), g in test.groupby(["sub", "block"]):
        tb[(s, b)] = (g.r.to_numpy(), g.is_old.to_numpy(),
                      [cond_codes[c] for c in g.cond])
    inv = {v: k for k, v in cond_codes.items()}
    print(f"{len(blocks):,} subject-blocks", flush=True)

    rng = np.random.default_rng(SEED)
    sub_blocks = [blocks[i] for i in rng.choice(len(blocks), 900, replace=False)]

    lines = ["CONTINUOUS REM THROUGH THE THREE IMAGE SET EXPERIMENTS",
             "=" * 96,
             f"{len(blocks):,} subject-blocks, {study['sub'].nunique()} participants, "
             f"three image sets.",
             f"space: {' + '.join(SELECTION)} reduced to {NDIM} dimensions.",
             f"storage {U}, stored-correctly {C_ENC}. sigma and criterion set to the",
             f"observed overall rates (hit {tgt_hit:.3f}, FA {tgt_fa:.3f}); "
             f"everything else is prediction.", "",
             f"{'sigma':>6} {'hit':>7} {'FA':>7} {'d-prime':>8}"]
    print(lines[-1], flush=True)

    best = None
    for frac in SIGMAS:
        sigma = (frac * sd).astype(np.float32)
        F, I, O, C = run(Z, sub_blocks, tb, grids, dens, sigma, 10)
        crit = np.quantile(F[~O], 1 - tgt_fa)
        said = F > crit
        h, f = said[O].mean(), said[~O].mean()
        s = f"{frac:>6.2f} {h:>7.3f} {f:>7.3f} {za(h)-za(f):>8.2f}"
        lines.append(s); print(s, flush=True)
        if best is None or abs(h - tgt_hit) < best[0]:
            best = (abs(h - tgt_hit), frac)

    frac = best[1]
    print(f"\nfull pass at sigma {frac:g} ...", flush=True)
    sigma = (frac * sd).astype(np.float32)
    F, I, O, C = run(Z, blocks, tb, grids, dens, sigma, 20)
    crit = np.quantile(F[~O], 1 - tgt_fa)
    said = F > crit
    lines += ["", f"chosen sigma {frac:g}, criterion {crit:.4g}: "
                  f"model hit {said[O].mean():.3f}, FA {said[~O].mean():.3f}"]

    d = pd.DataFrame({"r": I, "old": O, "cond": [inv[c] for c in C], "said": said})
    mh = d[d.old].groupby(["cond", "r"]).said.mean().rename("pred_hit")
    mf = d[~d.old].groupby(["cond", "r"]).said.mean().rename("pred_fa")
    pred = pd.concat([mh, mf], axis=1).reset_index()
    pred["pred_cr"] = pred.pred_hit - pred.pred_fa
    m = obs.merge(pred, on=["cond", "r"], how="inner")
    print(f"matched {len(m):,} image-by-set cells", flush=True)

    (OUT / "fit_datasets.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    m.to_csv(OUT / "fit_datasets_cells.csv", index=False)
    print(f"\nwrote outputs/fit_datasets.txt and outputs/fit_datasets_cells.csv")


if __name__ == "__main__":
    main()
