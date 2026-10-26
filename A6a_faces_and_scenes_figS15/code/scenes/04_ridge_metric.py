"""
04_ridge_metric.py (scenes): held-out correlation of summed similarity with memorability, with every
network dimension weighted equally or with dimension weights fitted on other images (Fig. S15a).

What it does
    The 14 deep-network spaces of the 10,442 Isola scenes are z-scored per dimension and placed side by
    side (14 x 512 = 7,168 dimensions). The 2,222 scored target images are split into five folds
    (KFold, shuffled, seed 0). For each fold, using the other four folds only:
      - equal weights ("isotropic metric" in the report): every dimension counts the same; the kernel
        width tau of the summed similarity is the one of 22 values (0.05 to 500, geometric) whose
        correlation with memorability is largest in absolute value on the training images;
      - fitted weights ("ridge-weighted metric"): a ridge regression of memorability on the 7,168
        dimensions (penalty alpha chosen from 10, 100, 1,000, 10,000, 100,000 by a 3-fold split of the
        training images) gives each dimension a weight, its squared coefficient rescaled to mean 1;
        each dimension is multiplied by the square root of its weight, summed similarity is recomputed
        in the weighted space and its tau chosen the same way.
    Summed similarity of an image is the log of the sum, over every other image of the 10,442, of
    exp(-(d - offset) / tau), with d the correlation distance in the (weighted) joined space z-scored
    with the mean and SD of a 1,000-image sample, and offset the 1st percentile of that sample's
    z-scored distances. The ridge regression only supplies the dimension weights: the measure stays
    relational, so an image's score depends on its distances to every other image and never on the
    image alone.
    Each version is then correlated (Spearman rho) with corrected recognition (hit rate minus false
    alarm rate) on the held-out fold. Fig. S15a shows the mean of the five held-out fold correlations
    (the two "crowding" rows of the HELD-OUT block of the report). Weights and tau are chosen inside
    each training fold only, and every correlation comes from held-out images. The highest correlation
    these scores allow is set by their split-half reliability (05_noise_ceiling.py).
Inputs (defaults; folders can be given on the command line)
    shared/embeddings_isola/regenerated_<network>_<layer>.npz   (see 02_crowding_analysis.py)
    A6a_faces_and_scenes_figS15/data/scenes/image_order.csv
    The script stops unless there are exactly 14 spaces whose stored file names follow
    image_order.csv row by row.
Outputs (A6a_faces_and_scenes_figS15/outputs/scenes/)
    ridge_metric.txt          the report (fold lines; HELD-OUT block; chosen tau and alpha)
    ridge_metric_folds.csv    the held-out correlation of each fold (plain = equal weights, learned =
                              fitted weights)
    ridge_heldout_scores.csv  for every scored image, its fold and its held-out summed similarity under
                              equal and fitted weights (stats_figS15.py rechecks the folds from it)
Seed
    0 everywhere: KFold(5, shuffle=True, random_state=0) for the folds, KFold(3, ...) inside each
    training set for alpha, and default_rng(0) for the 1,000-image sample that sets the distance scale
    of each summed-similarity computation.
Run (from the repository root; code/run_module.py runs it)
    python A6a_faces_and_scenes_figS15/code/scenes/04_ridge_metric.py
    About 3 minutes and 2 GB of memory.
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, norm
from scipy.special import logsumexp
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402

# The folders come from the command line (defaults: the folders of this repository), see main().
MODULE = Path(release_paths.module_path(__file__))
DEFAULT_EMB = Path(release_paths.shared_path('embeddings_isola'))
DEFAULT_ORDER = MODULE / "data" / "scenes" / "image_order.csv"
DEFAULT_OUT = MODULE / "outputs" / "scenes"
N_SPACES_EXPECTED = 14      # seven networks x early and late layer

CHUNK = 512
TAUS = np.round(np.geomspace(0.05, 500.0, 22), 4)
ALPHAS = [1e1, 1e2, 1e3, 1e4, 1e5]
N_FOLDS = 5
SEED = 0


def show(path):
    """A path relative to the repository root when it is inside the repository, for printing."""
    relative = os.path.relpath(str(path), release_paths.release_root())
    return str(path) if relative.startswith('..') else relative


def load_joint(EMB, order):
    """All spaces concatenated, each z-scored so no network dominates by scale."""
    mats, names = [], []
    for f in sorted(EMB.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        # every space must list its images in the row order of image_order.csv
        stored_names = [str(name) for name in z["filenames"]]
        if stored_names != order["filename"].astype(str).tolist():
            raise SystemExit(f"{show(f)}: its rows do not follow the image list (image_order.csv)")
        X = z["features"].astype(np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        mats.append(X)
        names.append(f.stem)
    if len(names) != N_SPACES_EXPECTED:
        raise SystemExit(f"expected {N_SPACES_EXPECTED} spaces in {show(EMB)}, found {len(names)}")
    return np.concatenate(mats, axis=1), names


def crowd(X, taus, chunk=CHUNK):
    """Kernel crowding (summed similarity, one array per kernel width tau) under the metric X already
    encodes."""
    N = len(X)
    Xn = X - X.mean(1, keepdims=True)
    Xn /= np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-8)
    # scale reference from a sample, so the kernel sits in a usable range
    rs = np.random.default_rng(SEED).choice(N, size=min(1000, N), replace=False)
    S = Xn[rs] @ Xn[rs].T
    dd = 1.0 - S[np.triu_indices(len(rs), 1)]
    mu, sd = float(dd.mean()), float(dd.std() + 1e-8)
    off = float(np.percentile((dd - mu) / sd, 1))

    ksum = {t: np.zeros(N, np.float32) for t in taus}
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        D = (1.0 - (Xn[a:b] @ Xn.T) - mu) / sd
        D[np.arange(b - a), np.arange(a, b)] = np.inf
        fin = np.isfinite(D)
        Dc = np.where(fin, D, np.inf)
        for t in taus:
            z = -(Dc - off) / t
            z[~fin] = -np.inf
            ksum[t][a:b] = logsumexp(z, axis=1)
    return ksum


def rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return spearmanr(x[ok], y[ok]).statistic if ok.sum() > 10 else np.nan


def main():
    parser = argparse.ArgumentParser(description='Held-out crowding with equal and ridge-fitted weights.')
    parser.add_argument('--embeddings-dir', type=Path, default=DEFAULT_EMB)
    parser.add_argument('--image-order', type=Path, default=DEFAULT_ORDER)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--per-image-dir', type=Path, default=None,
                        help='folder for ridge_heldout_scores.csv (default: the --out-dir folder)')
    args = parser.parse_args()
    EMB, OUT = args.embeddings_dir, args.out_dir
    PER_IMAGE = args.per_image_dir if args.per_image_dir is not None else OUT
    OUT.mkdir(parents=True, exist_ok=True)
    PER_IMAGE.mkdir(parents=True, exist_ok=True)

    order = pd.read_csv(args.image_order)
    m = order["has_scores"].to_numpy() == 1
    if "kind" in order.columns:
        m &= order["kind"].to_numpy() == "target"
    if "in_cvpr_2222" in order.columns:
        m &= order["in_cvpr_2222"].to_numpy() == 1
    cr = order["corrected_recognition"].to_numpy(float)
    hr = order["hit_rate"].to_numpy(float)
    fa = order["false_alarm_rate"].to_numpy(float)
    za = lambda p: norm.ppf(np.clip(p, 1e-3, 1 - 1e-3))
    dp = za(hr) - za(fa)

    X, names = load_joint(EMB, order)
    scored = np.flatnonzero(m)
    print(f"{X.shape[0]:,} images x {X.shape[1]:,} dims from {len(names)} spaces; "
          f"{len(scored):,} scored", flush=True)

    fh = open(OUT / "ridge_metric.txt", "w", encoding="utf-8")

    def w(s=""):
        print(s, flush=True)
        fh.write(s + "\n")

    w("CROWDING IN A LEARNED METRIC")
    w("=" * 96)
    w(f"{X.shape[0]:,} images, {X.shape[1]:,} concatenated dimensions "
      f"({len(names)} network/layer spaces, each z-scored).")
    w(f"{len(scored):,} scored items, {N_FOLDS}-fold cross-validation.")
    w("Dimension weights and tau are chosen inside each training fold only; every")
    w("number below is from held-out items.")
    w()

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    res = {k: [] for k in ("plain", "learned")}
    taus_used, alphas_used = [], []
    heldout_rows = []      # one row per held-out image (ridge_heldout_scores.csv)

    # isotropic crowding does not depend on the fold, so compute it once
    ksum_plain = crowd(X, TAUS)

    for fold, (tr, te) in enumerate(kf.split(scored), 1):
        tr_i, te_i = scored[tr], scored[te]

        # --- ridge regression on the embedding (the source of the dimension weights) ---
        best_a, best_r, best_model = None, -np.inf, None
        inner = KFold(n_splits=3, shuffle=True, random_state=SEED)
        for al in ALPHAS:
            sc = []
            for itr, ite in inner.split(tr_i):
                mdl = Ridge(alpha=al).fit(X[tr_i[itr]], cr[tr_i[itr]])
                sc.append(rho(mdl.predict(X[tr_i[ite]]), cr[tr_i[ite]]))
            s = np.nanmean(sc)
            if s > best_r:
                best_r, best_a = s, al
        model = Ridge(alpha=best_a).fit(X[tr_i], cr[tr_i])
        alphas_used.append(best_a)

        # --- the learned metric: dimensions weighted by squared coefficient ---
        wgt = model.coef_ ** 2
        wgt = wgt / (wgt.mean() + 1e-12)
        Xw = X * np.sqrt(wgt, dtype=np.float32)
        ksum_l = crowd(Xw, TAUS)

        # --- tau chosen on the training items only ---
        def pick(ks):
            bt, bv = None, -np.inf
            for t in TAUS:
                v = abs(rho(ks[t][tr_i], cr[tr_i]))
                if np.isfinite(v) and v > bv:
                    bv, bt = v, t
            return bt
        t_p, t_l = pick(ksum_plain), pick(ksum_l)
        taus_used.append((t_p, t_l))

        res["plain"].append(rho(ksum_plain[t_p][te_i], cr[te_i]))
        res["learned"].append(rho(ksum_l[t_l][te_i], cr[te_i]))

        # keep each held-out image's summed similarity under both metrics (at the tau chosen on this
        # fold's training images), so stats_figS15.py can recompute each fold correlation.
        for image_row in te_i:
            heldout_rows.append(dict(image_row=int(image_row),
                                     filename=str(order["filename"].iloc[image_row]),
                                     fold=fold, alpha=float(best_a),
                                     tau_equal_weights=float(t_p), tau_fitted_weights=float(t_l),
                                     summed_similarity_equal_weights=float(ksum_plain[t_p][image_row]),
                                     summed_similarity_fitted_weights=float(ksum_l[t_l][image_row]),
                                     hit_rate=float(hr[image_row]), false_alarm_rate=float(fa[image_row]),
                                     corrected_recognition=float(cr[image_row])))

        w(f"  fold {fold}: crowding plain {res['plain'][-1]:+.3f} (tau {t_p:.2f}) | "
          f"crowding learned {res['learned'][-1]:+.3f} (tau {t_l:.2f}) | "
          f"alpha {best_a:g}")

    w()
    w("-" * 96)
    w("HELD-OUT SPEARMAN RHO WITH CORRECTED RECOGNITION")
    w("-" * 96)
    lab = {"plain": "crowding, isotropic metric",
           "learned": "crowding, ridge-weighted metric"}
    for k in ("plain", "learned"):
        v = np.array(res[k], float)
        w(f"  {lab[k]:34s} {np.nanmean(v):+.3f}  "
          f"(folds {', '.join(f'{x:+.3f}' for x in v)})")
    w()

    w(f"  tau chosen: isotropic {[f'{a:.2f}' for a, _ in taus_used]}, "
      f"learned {[f'{b:.2f}' for _, b in taus_used]}")
    w(f"  ridge alpha chosen: {alphas_used}")
    w()
    w("-" * 96)
    w("NOTES")
    w("-" * 96)
    w("  Each image is scored by its summed similarity to every other image of")
    w("  the database. The isotropic metric weights every dimension equally")
    w("  (Fig. S15a, equal weights); the ridge-weighted metric weights each")
    w("  dimension by its squared ridge coefficient, rescaled to mean 1 (Fig.")
    w("  S15a, fitted weights). The highest correlation these scores allow is")
    w("  about the square root of the Spearman-Brown value in noise_ceiling.txt.")
    fh.close()

    pd.DataFrame(res).to_csv(OUT / "ridge_metric_folds.csv", index=False)
    pd.DataFrame(heldout_rows).to_csv(PER_IMAGE / "ridge_heldout_scores.csv", index=False)
    print(f"\nwrote {show(OUT / 'ridge_metric.txt')}")
    print(f"wrote {show(PER_IMAGE / 'ridge_heldout_scores.csv')}")


if __name__ == "__main__":
    main()
