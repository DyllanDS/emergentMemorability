"""
05_noise_ceiling.py (faces): split-half reliability of the face memorability scores (the 0.66 and
the dashed line of Fig. S15a).

This script runs only when the face database is available (EXTERNAL_INPUTS.md, M4): the module holds
the saved faces report (data/provided_faces/noise_ceiling.txt and .json) and this code.
code/run_module.py calls it through code/faces/regenerate_faces.py when the archives are in
external/bainbridge_faces/.
What it does
    The face database has no per-participant data, so the reliability is computed over trials: each
    scored face's own old and new trials (hit, miss, false alarm and correct-rejection counts) are
    split at random into two halves (25 times, seed 0), each half gives every face a corrected
    recognition score, and the two halves are correlated (Spearman rho) over the 2,222 scored faces.
    The Fig. S15 caption prints the mean, 0.66 (uncorrected); the report also gives the Spearman-Brown
    value, whose square root is about the highest correlation the scores allow (SI Appendix,
    Section 6). Without participant matrices (the default) only the block over trials runs.
Inputs (all required on the command line; regenerate_faces.py passes them)
    --image-order      the face list with the counts of the database (00_build_image_order.py)
    --out-dir          for noise_ceiling.txt, noise_ceiling.json, noise_ceiling_splits.csv (aggregate)
Seed
    numpy default_rng(0): the 25 hypergeometric trial splits.

Every correlation with these memorability scores is bounded by how reliable the scores themselves
are. There are two ways to measure it:

  over participants  Split the participants in half, score every image twice, and correlate. This
                     is what the literature calls human consistency, and it is what Isola et al.
                     report. It needs per-participant response vectors.

  over trials        Split each image's own trials in half instead. It needs only the hit, miss,
                     false alarm and correct-rejection counts, so it works where per-participant
                     data were not released.

The report gives the split-half value with its Spearman-Brown correction to the full sample, since a
half-sample correlation understates the reliability of the whole.
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

# The files come from the command line (all required: the face database is not in the repository).

N_SPLITS = 25          # Isola et al. use 25 random splits
RNG = np.random.default_rng(0)
SPLIT_VALUES = []      # every split's correlations, for noise_ceiling_splits.csv


def show(path):
    """A path relative to the repository root when it is inside the repository, for printing."""
    relative = os.path.relpath(str(path), release_paths.release_root())
    return str(path) if relative.startswith('..') else relative


def sb(r):
    """Spearman-Brown: half-sample reliability -> full-sample."""
    return 2 * r / (1 + r) if np.isfinite(r) else np.nan


def za(p):
    return norm.ppf(np.clip(p, 1e-3, 1 - 1e-3))


def rates_from_counts(hit, n_old, fa, n_new):
    hr = np.divide(hit, n_old, out=np.full_like(hit, np.nan, float), where=n_old > 0)
    far = np.divide(fa, n_new, out=np.full_like(fa, np.nan, float), where=n_new > 0)
    return hr, far


def ceiling_over_subjects(det, falarm, mask_items):
    """det/falarm: (participants x items).

    Coding: 1 is a response, -1 is no response, and 0 means the participant
    never saw that item. So a trial happened wherever the entry is non-zero;
    counting 0 as no response would count every unseen item as a miss and
    lower the reliability. Not used for the faces, which have no
    per-participant data.
    """
    nS = det.shape[0]
    out = {k: [] for k in ("hit", "fa", "cr", "dprime")}
    for _ in range(N_SPLITS):
        perm = RNG.permutation(nS)
        halves = (perm[: nS // 2], perm[nS // 2:])
        got = []
        for h in halves:
            d, f = det[h], falarm[h]
            seen_old = d != 0
            seen_new = f != 0
            hit = np.where(seen_old, d == 1, 0).sum(0).astype(float)
            n_old = seen_old.sum(0).astype(float)
            fa = np.where(seen_new, f == 1, 0).sum(0).astype(float)
            n_new = seen_new.sum(0).astype(float)
            hr, far = rates_from_counts(hit, n_old, fa, n_new)
            got.append((hr, far, hr - far, za(hr) - za(far)))
        for key, i in zip(("hit", "fa", "cr", "dprime"), range(4)):
            a, b = got[0][i][mask_items], got[1][i][mask_items]
            ok = np.isfinite(a) & np.isfinite(b)
            out[key].append(spearmanr(a[ok], b[ok]).statistic)
    SPLIT_VALUES.append(("over participants", out))
    return {k: (float(np.mean(v)), float(sb(np.mean(v)))) for k, v in out.items()}


def ceiling_over_trials(hits, misses, fas, crs, mask_items):
    """Split each image's own trials, for data with no per-participant responses."""
    # unscored images carry NaN counts, and an image with no trials of one kind
    # cannot contribute; restrict before sampling rather than guarding inside
    keep = (mask_items & np.isfinite(hits) & np.isfinite(misses)
            & np.isfinite(fas) & np.isfinite(crs))
    keep &= (hits + misses >= 2) & (fas + crs >= 2)
    hits, misses = hits[keep], misses[keep]
    fas, crs = fas[keep], crs[keep]
    mask_items = np.ones(keep.sum(), bool)
    out = {k: [] for k in ("hit", "fa", "cr", "dprime")}
    n_old = (hits + misses).astype(int)
    n_new = (fas + crs).astype(int)
    for _ in range(N_SPLITS):
        h1 = RNG.hypergeometric(np.maximum(hits, 0).astype(int),
                                np.maximum(misses, 0).astype(int),
                                np.maximum(n_old // 2, 1))
        f1 = RNG.hypergeometric(np.maximum(fas, 0).astype(int),
                                np.maximum(crs, 0).astype(int),
                                np.maximum(n_new // 2, 1))
        n_old_1 = np.maximum(n_old // 2, 1)
        n_new_1 = np.maximum(n_new // 2, 1)
        hr1, far1 = rates_from_counts(h1.astype(float), n_old_1.astype(float),
                                      f1.astype(float), n_new_1.astype(float))
        hr2, far2 = rates_from_counts((hits - h1).astype(float),
                                      np.maximum(n_old - n_old_1, 1).astype(float),
                                      (fas - f1).astype(float),
                                      np.maximum(n_new - n_new_1, 1).astype(float))
        pairs = [(hr1, hr2), (far1, far2), (hr1 - far1, hr2 - far2),
                 (za(hr1) - za(far1), za(hr2) - za(far2))]
        for key, (a, b) in zip(("hit", "fa", "cr", "dprime"), pairs):
            a, b = a[mask_items], b[mask_items]
            ok = np.isfinite(a) & np.isfinite(b)
            out[key].append(spearmanr(a[ok], b[ok]).statistic)
    SPLIT_VALUES.append(("over trials", out))
    return {k: (float(np.mean(v)), float(sb(np.mean(v)))) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description='Split-half reliability of the face memorability scores.')
    parser.add_argument('--image-order', type=Path, required=True)
    parser.add_argument('--subject-matrices', type=Path, default=Path('none'),
                        help='not used for the faces (the database has no per-participant data)')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    OUT = args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)

    order = pd.read_csv(args.image_order)
    m = order["has_scores"].to_numpy() == 1
    if "kind" in order.columns:
        m &= order["kind"].to_numpy() == "target"
    if "in_cvpr_2222" in order.columns:
        m &= order["in_cvpr_2222"].to_numpy() == 1

    fh = open(OUT / "noise_ceiling.txt", "w", encoding="utf-8")

    def w(s=""):
        print(s, flush=True)
        fh.write(s + "\n")

    w("NOISE CEILINGS")
    w("=" * 88)
    w(f"{int(m.sum()):,} scored items, {N_SPLITS} random splits, Spearman rank "
      f"correlation.")
    w()

    results = {}
    subj = args.subject_matrices
    if str(subj).lower() != "none" and subj.exists():
        import h5py
        with h5py.File(subj, "r") as f:
            # MATLAB writes v7.3 transposed
            tdet = np.array(f["tdet"]).T
            tfa = np.array(f["tfa"]).T
        w(f"Per-subject data available: {tdet.shape[0]} subjects x "
          f"{tdet.shape[1]} targets.")
        w("Ceiling computed over participants, which is the convention in this "
          "literature.")
        w()
        mi = m[: tdet.shape[1]] if len(m) >= tdet.shape[1] else m
        # image_order is sorted by filename; the subject matrices follow the
        # order of the public package (index_in_set), so the target mask is
        # rebuilt in that order
        rel = order.copy()
        rel_targets = ((rel["kind"] == "target") & (rel["in_cvpr_2222"] == 1)) \
            if "kind" in rel.columns else (rel["has_scores"] == 1)
        idx = rel.loc[rel_targets, "index_in_set"].to_numpy().astype(int) - 1
        mask = np.zeros(tdet.shape[1], bool)
        mask[idx[idx < tdet.shape[1]]] = True
        w(f"  targets entering the ceiling: {int(mask.sum()):,}")
        results["over participants"] = ceiling_over_subjects(tdet, tfa, mask)
    else:
        w("No per-participant data; reliability computed over trials instead:")
        w("each image's own old and new trials are split in half at random.")
        w()

    if {"hits", "misses", "false_alarms", "correct_rejections"} <= set(order.columns):
        results["over trials"] = ceiling_over_trials(
            order["hits"].to_numpy(float), order["misses"].to_numpy(float),
            order["false_alarms"].to_numpy(float),
            order["correct_rejections"].to_numpy(float), m)

    for how, res in results.items():
        w("-" * 88)
        w(f"Ceiling {how}")
        w("-" * 88)
        w(f"{'measure':22s} {'split-half':>12s} {'Spearman-Brown':>16s}")
        for k, lab in (("cr", "corrected recognition"),):
            half, full = res[k]
            w(f"{lab:22s} {half:12.3f} {full:16.3f}")
        w()

    w("-" * 88)
    w("NOTES")
    w("-" * 88)
    w("  Split-half: the mean Spearman rho between two random halves over the")
    w("  splits. Spearman-Brown: 2r / (1 + r) of that mean, the reliability of the")
    w("  full sample. A correlation with these scores can be at most about the")
    w("  square root of the Spearman-Brown value.")
    w("  Corrected recognition is the hit rate minus the false alarm rate.")
    w("  All splits draw from one numpy generator with seed 0.")
    fh.close()
    json.dump({how: {"cr": res["cr"]} for how, res in results.items()},
              open(OUT / "noise_ceiling.json", "w"), indent=2)

    # the individual split-half correlations (one row per unit, split and measure)
    split_rows = []
    for how, out in SPLIT_VALUES:
        for split_number in range(N_SPLITS):
            for key in ("cr",):
                split_rows.append(dict(unit=how, split=split_number + 1, measure=key,
                                       split_half_rho=float(out[key][split_number])))
    pd.DataFrame(split_rows).to_csv(OUT / "noise_ceiling_splits.csv", index=False)
    print(f"\nwrote {show(OUT / 'noise_ceiling.txt')}")
    print(f"wrote {show(OUT / 'noise_ceiling_splits.csv')}")


if __name__ == "__main__":
    main()
