"""
05_noise_ceiling.py (scenes): split-half reliability of the scene memorability scores (the 0.73 and
the dashed line of Fig. S15a).

What it does
    The 843 usable participant records of Isola et al. (2011) are split at random into two halves
    (25 times, seed 0); each half gives every target image a hit rate, a false alarm rate, corrected
    recognition (hit rate minus false alarm rate) and d'; the two halves are correlated (Spearman rho)
    over the 2,222 scored targets, and the 25 correlations are averaged. The report gives the mean for
    corrected recognition, which the Fig. S15 caption prints (0.73, uncorrected), and its
    Spearman-Brown value for the full sample, 2r / (1 + r), whose square root is about the highest
    correlation the scores allow (SI Appendix, Section 6).
    Participant matrices: 1 = a response, -1 = no response, 0 = the participant never saw the image.
Inputs (defaults; can be given on the command line)
    A6a_faces_and_scenes_figS15/data/scenes/image_order.csv    the scored targets and their columns
                                                                in the participant matrices
    A6a_faces_and_scenes_figS15/data/scenes/subject_matrices.mat   843 x 2,400 participant matrices
        (the public subj_data.mat exported with shared/embedding_extraction/export_subjects.m; MATLAB
        v7.3, read with h5py). With --subject-matrices none the report holds no reliability, and the
        later steps read the scene reliability from data/paper_run_scenes/.
Outputs (A6a_faces_and_scenes_figS15/outputs/scenes/)
    noise_ceiling.txt, noise_ceiling.json   the mean over the 25 splits (raw and Spearman-Brown)
    noise_ceiling_splits.csv                the 25 split-half correlations themselves
Seed
    numpy default_rng(0): the 25 participant permutations.
Run (from the repository root; code/run_module.py runs it)
    python A6a_faces_and_scenes_figS15/code/scenes/05_noise_ceiling.py
    A few seconds.

Every correlation with these memorability scores is bounded by how reliable the scores themselves
are. Splitting the participants in half, scoring every image twice and correlating the two scores is
what the literature calls human consistency, and it is what Isola et al. report. The report gives the
split-half value with its Spearman-Brown correction to the full sample, since a half-sample
correlation understates the reliability of the whole.
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

# The files come from the command line (defaults: the files of this repository), see main().
MODULE = Path(release_paths.module_path(__file__))
DEFAULT_ORDER = MODULE / "data" / "scenes" / "image_order.csv"
DEFAULT_SUBJECTS = MODULE / "data" / "scenes" / "subject_matrices.mat"
DEFAULT_OUT = MODULE / "outputs" / "scenes"

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
    lower the reliability. Hits and false alarms counted this way correlate
    with the released counts at r = 1.0000.
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


def main():
    parser = argparse.ArgumentParser(description='Split-half reliability of the scene memorability scores.')
    parser.add_argument('--image-order', type=Path, default=DEFAULT_ORDER)
    parser.add_argument('--subject-matrices', type=Path, default=DEFAULT_SUBJECTS,
                        help='participant matrices (MATLAB v7.3); use "none" to skip the reliability over participants')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
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
        w("No participant matrices: the scene reliability comes from data/paper_run_scenes/.")
        w()

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
