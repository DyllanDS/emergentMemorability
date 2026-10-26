"""
paired_crowding.py: the same image in two or three image sets, compared with itself (SI Appendix,
Section 3; Fig. S8b and c).

WHAT IT DOES
  For every image tested in two or more image sets (with at least 5 old and 5 new trials in each),
  the cell where the image's 20 nearest set-mates were least similar (least crowded set) is compared
  with the cell where they were most similar (most crowded set): the difference in hit rate, false
  alarm rate and d′ (rates kept between 1/(2N) and 1 - 1/(2N), N = the cell's number of old or new
  trials), computed image by image and averaged over images, with every image counted once. The same
  differences are computed from the model's predicted rates for the same cells. 95% percentile
  intervals: 10,000 resamples of the images (observed and model; the error bars of Fig. S8b and c).
  It also splits the d′ change into the part carried by hits and the part carried by false alarms
  (SI Appendix, Section 3: false alarms contribute most of the change), moving each rate half its
  mean change either side of its mean over the paired cells. Crowding, the cell rates and the
  model's predictions are read from the files the earlier scripts of this module wrote.

INPUTS (in outputs/; written by the earlier scripts of this module)
  bin_by_crowding_cells.csv (crowding of every cell), fit_datasets_cells.csv (the model's rates),
  model_inputs/datasets_test.csv (every kept test trial: the participants of each set, and a check
      that the cells add up to the trials)

OUTPUTS (in outputs/)
  paired_crowding.txt          the report
  paired_crowding_summary.csv  measure, source (observed, model), mean, 95% interval over images
                               (lo_images, hi_images)
  paired_crowding_images.csv   one row per image: both cells, their rates and the differences

SEED 0 (numpy default_rng, one generator for the script), drawn in this order: the 10,000 image
  resamples of the observed differences; 2,000 draws of each set's participants and 10,000 draws of
  the images (section 4 of the code), whose values are not used; and the 10,000 image resamples of
  the model's differences, whose intervals depend on the draws before them.

RUN (from the repository root; a few seconds):  python A3a_model_through_own_lists_figS8/code/paired_crowding.py
"""
import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (first: it stops Python from writing __pycache__ folders)

PATHS = release_paths.module_paths(__file__)
OUT = Path(PATHS['outputs'])
MODEL_INPUTS = OUT / "model_inputs"
CROWD_COLS = ["crowd_near"]
PRIMARY = "crowd_near"   # crowding by the 20 nearest set-mates (bin_by_crowding.py), the measure of Fig. S8b and c
NBOOT_IMG = 10000
NBOOT_SUB = 2000
SEED = 0


def corrected(p, n):
    """Rates clamped by 1/(2N) so a cell at 0 or 1 still yields a finite z."""
    p = np.asarray(p, float)
    n = np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = np.where(n > 0, 1.0 / (2.0 * np.maximum(n, 1e-9)), np.nan)
    return np.clip(p, lo, 1.0 - lo)


def dprime(hit, fa, n_old, n_new):
    return norm.ppf(corrected(hit, n_old)) - norm.ppf(corrected(fa, n_new))


def paired_deltas(hit, fa, n_old, n_new, a, b):
    """Least-crowded cell minus most-crowded cell, one row per image.

    Positive means memory is better when the image sits among less similar company.
    """
    dp = dprime(hit, fa, n_old, n_new)
    return pd.DataFrame({
        "d_hit": hit[a] - hit[b],
        "d_fa": fa[a] - fa[b],
        "d_dprime": dp[a] - dp[b],
    })


def boot_ci_images(deltas, rng, nboot=NBOOT_IMG):
    """Percentile CI from resampling images, the unit the contrast is defined on."""
    n = len(deltas)
    idx = rng.integers(0, n, size=(nboot, n))
    out = {}
    for c in deltas.columns:
        v = deltas[c].to_numpy()
        means = np.nanmean(v[idx], axis=1)
        out[c] = (np.nanmean(v), np.percentile(means, 2.5), np.percentile(means, 97.5))
    return out


def fmt_ci(t):
    return f"{t[0]:+.3f}  [{t[1]:+.3f}, {t[2]:+.3f}]"


def main():
    rng = np.random.default_rng(SEED)
    lines = []

    def w(s=""):
        print(s, flush=True)
        lines.append(s)

    # ---- 1. cells, crowding and model predictions, as bin_by_crowding.py and fit_datasets.py wrote them ----
    d = pd.read_csv(OUT / "bin_by_crowding_cells.csv", float_precision="round_trip")
    fit = pd.read_csv(OUT / "fit_datasets_cells.csv", float_precision="round_trip")
    d = d.merge(fit[["cond", "r", "pred_hit", "pred_fa"]], on=["cond", "r"], how="left")
    d = d.reset_index(drop=True)
    d["cell"] = np.arange(len(d))

    # ---- 2. pairs: the least and most crowded set each image appeared in ----
    counts = d.groupby("r").cond.nunique()
    multi = d.r.map(counts) >= 2
    dm = d[multi].copy()

    pairs = {}
    for col in CROWD_COLS:
        g = dm.sort_values(col).groupby("r")
        p = pd.DataFrame({
            "a": g.cell.first().to_numpy(),          # least crowded set for this image
            "b": g.cell.last().to_numpy(),           # most crowded set for this image
            "r": g.cell.first().index.to_numpy(),
            "k": g.size().to_numpy(),
        })
        p["span"] = d[col].to_numpy()[p.b] - d[col].to_numpy()[p.a]
        pairs[col] = p

    hit = d.hit.to_numpy()
    fa = d.fa.to_numpy()
    n_old = d.n_old.to_numpy()
    n_new = d.n_new.to_numpy()

    P = pairs[PRIMARY]
    A, B = P.a.to_numpy(), P.b.to_numpy()
    obs_deltas = paired_deltas(hit, fa, n_old, n_new, A, B)
    obs_ci = boot_ci_images(obs_deltas, rng)

    # ---- 3. the participants of each set, and the cells rebuilt from the trial table ----
    test = pd.read_csv(MODEL_INPUTS / "datasets_test.csv",
                       usecols=["sub", "cond", "image_nr", "is_old", "said_old"])
    key = pd.Series(d.cell.to_numpy(),
                    index=pd.MultiIndex.from_arrays([d.cond, d.image_nr]))
    test["cell"] = pd.MultiIndex.from_arrays([test.cond, test.image_nr]).map(key)
    test = test.dropna(subset=["cell"])
    test["cell"] = test.cell.astype(int)
    subs = pd.Categorical(test["sub"])
    test["sidx"] = subs.codes
    nsub = len(subs.categories)
    # each participant saw exactly one set
    sub_cond = test.groupby("sidx").cond.first()
    cond_members = {c: g.index.to_numpy() for c, g in sub_cond.groupby(sub_cond)}

    ncell = len(d)
    old_m = test.is_old.to_numpy() == 1
    cidx_o = test.cell.to_numpy()[old_m]
    said_o = test.said_old.to_numpy()[old_m]

    chk_no = np.bincount(cidx_o, minlength=ncell)
    chk_hit = np.bincount(cidx_o, weights=said_o, minlength=ncell)
    recon_ok = (np.abs(chk_no - n_old).max(), np.abs(chk_hit / np.maximum(chk_no, 1) - hit).max())

    # ---- 4. draws whose values are not used ----
    # Before the model's image resamples of section 6, the generator makes 2,000 draws of each set's
    # participants (one multinomial per set, sets in the order changed, maintained, reduced) and then
    # 10,000 draws of the images in two or more sets. The model's intervals in Fig. S8b and c are
    # those of this order of draws.
    for _ in range(NBOOT_SUB):
        for c, members in cond_members.items():
            rng.multinomial(len(members), np.full(len(members), 1 / len(members)))
    rng.integers(0, len(P), size=(NBOOT_IMG, len(P)))

    dp_cells = dprime(hit, fa, n_old, n_new)

    # ---- 5. the d-prime change split into the part the hits carry and the part the false alarms carry ----
    # d-prime is not linear in the rates, so each rate is moved half its mean change
    # either side of its mean over the paired cells (the 1/(2N) limits use the mean
    # numbers of old and new trials of those cells).
    both = np.concatenate([A, B])
    hbar, fbar = hit[both].mean(), fa[both].mean()
    nobar, nnbar = n_old[both].mean(), n_new[both].mean()

    def dprime_parts(bh, bf, h0=hbar, f0=fbar):
        """d-prime change split into the part the hits carry and the part the false alarms carry."""
        hl, hm = corrected(h0 + bh / 2, nobar), corrected(h0 - bh / 2, nobar)
        fl, fm = corrected(f0 + bf / 2, nnbar), corrected(f0 - bf / 2, nnbar)
        return norm.ppf(hl) - norm.ppf(hm), -(norm.ppf(fl) - norm.ppf(fm))

    # ---- 6. the model's own predictions, same pairs, same measures ----
    has_pred = d.pred_hit.notna().all()
    ph, pf = d.pred_hit.to_numpy(), d.pred_fa.to_numpy()
    mod_deltas = paired_deltas(ph, pf, n_old, n_new, A, B)
    mod_ci = boot_ci_images(mod_deltas, rng)
    mh_bar, mf_bar = ph[both].mean(), pf[both].mean()

    # which rate carries the effect, in d-prime units, for data and for model
    obs_parts = dprime_parts(obs_ci["d_hit"][0], obs_ci["d_fa"][0])
    mod_parts = dprime_parts(mod_ci["d_hit"][0], mod_ci["d_fa"][0], mh_bar, mf_bar)

    # ---- 7. the report ----
    two = (P.k == 2).to_numpy()

    w("PAIRED WITHIN-IMAGE CROWDING CONTRAST")
    w("=" * 92)
    w()
    w("WHAT THIS SHOWS")
    w("-" * 92)
    w(f"  Each of the {len(P):,} images measured in two or more sets is compared with itself:")
    w("  the set where it sat among the least similar company versus the set where it sat")
    w("  among the most similar company. Every image contributes exactly one difference and")
    w("  every image counts the same amount.")
    w()
    w(f"  d-prime is higher by {obs_ci['d_dprime'][0]:.3f} "
      f"[{obs_ci['d_dprime'][1]:.3f}, {obs_ci['d_dprime'][2]:.3f}] when the same photograph sits")
    w(f"  in its less crowded set. Hits rise by {obs_ci['d_hit'][0]:.3f} "
      f"[{obs_ci['d_hit'][1]:.3f}, {obs_ci['d_hit'][2]:.3f}] and false alarms fall by")
    w(f"  {-obs_ci['d_fa'][0]:.3f} [{-obs_ci['d_fa'][2]:.3f}, {-obs_ci['d_fa'][1]:.3f}]; in d-prime "
      f"units, {obs_parts[0]:.3f} of the change comes from the hits and")
    w(f"  {obs_parts[1]:.3f} from the false alarms. Section 3 below gives the same contrast for the model's")
    w("  predicted rates.")
    w()
    w(f"  On the same pairs the model predicts {mod_ci['d_dprime'][0]:+.3f} in d-prime, against "
      f"{obs_ci['d_dprime'][0]:+.3f} observed ({mod_ci['d_dprime'][0]/obs_ci['d_dprime'][0]:.0%} of")
    w(f"  the observed change). In the model, false alarms carry {mod_parts[1]:.3f} in d-prime (its "
      f"total change")
    w(f"  is {mod_ci['d_dprime'][0]:.3f}), and it predicts hits "
      f"{'falling' if mod_ci['d_hit'][0] < 0 else 'rising'} in the less crowded set "
      f"({mod_ci['d_hit'][0]:+.3f}), while the")
    w(f"  observed hit rate {'rose' if obs_ci['d_hit'][0] > 0 else 'fell'}.")
    w()

    w("=" * 92)
    w("1. THE PAIRS")
    w("-" * 92)
    w(f"  {len(d):,} image-by-set cells, {d.r.nunique():,} distinct images.")
    w(f"  {len(P):,} images appear in two or more sets: {int(two.sum()):,} in two sets, "
      f"{int((~two).sum()):,} in three.")
    w(f"  Crowding measure: {PRIMARY} (higher = closer neighbors = more crowded).")
    w(f"  Cells rebuilt from the trial files: largest difference in trial count "
      f"{recon_ok[0]:.0f}, in rate {recon_ok[1]:.2e}.")
    w(f"  Participants in datasets_test.csv ({nsub} of them, each in exactly one set).")
    w()

    w("2. PAIRED CONTRAST: least crowded set minus most crowded set, same image")
    w("-" * 92)
    w(f"  Images weighted equally. CIs are 2.5/97.5 percentiles of {NBOOT_IMG:,} image resamples.")
    w()
    w(f"  {'measure':>22} {'mean':>8} {'95% CI (images)':>22}")
    # These labels name the rows of this table and of paired_crowding_summary.csv;
    # write_numbers.py, make_figureS8.py and stats_figS8.py find the rows by them.
    labels = {"d_hit": "hit rate", "d_fa": "false alarm rate", "d_dprime": "d-prime"}
    for c, lab in labels.items():
        w(f"  {lab:>22} {obs_ci[c][0]:>+8.3f} "
          f"{f'[{obs_ci[c][1]:+.3f}, {obs_ci[c][2]:+.3f}]':>22}")
    w()

    w("3. THE MODEL ON THE SAME PAIRS")
    w("-" * 92)
    if not has_pred:
        w("  pred_hit / pred_fa missing for some cells; run fit_datasets.py again.")
    w("  Model predicted rates for the same cells, contrasted over the same pairs, images")
    w("  weighted the same way.")
    w()
    w(f"  {'measure':>22} {'observed':>26} {'model':>26}")
    for c, lab in labels.items():
        w(f"  {lab:>22} "
          f"{f'{obs_ci[c][0]:+.3f} [{obs_ci[c][1]:+.3f}, {obs_ci[c][2]:+.3f}]':>26} "
          f"{f'{mod_ci[c][0]:+.3f} [{mod_ci[c][1]:+.3f}, {mod_ci[c][2]:+.3f}]':>26}")
    w()
    w(f"  Base rates over the paired cells: observed hit {hbar:.3f} FA {fbar:.3f}, "
      f"model hit {mh_bar:.3f} FA {mf_bar:.3f}.")
    w(f"  Split of the paired d-prime effect: observed {obs_parts[0]:.3f} from hits and "
      f"{obs_parts[1]:.3f} from false")
    w(f"  alarms; model {mod_parts[0]:.3f} from hits and {mod_parts[1]:.3f} from false alarms.")
    w()
    w("  The model's rates carry no item-level intrinsic memorability term, so its paired")
    w("  changes come only from the other images in each set. The split of the d-prime")
    w("  change holds each rate at its mean over the paired cells (the base rates above)")
    w("  and moves it half its mean change either side, so the two parts of the split")
    w("  need not add up to the mean of the per-image d-prime changes in the table above")
    w("  (the 1/(2N) limits use the mean numbers of old and new trials of those cells).")
    w()

    (OUT / "paired_crowding.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- 8. per-image and summary tables ----
    per_image = pd.DataFrame({
        "image_nr": d.image_nr.to_numpy()[A],
        "r": P.r.to_numpy(),
        "n_sets": P.k.to_numpy(),
        "cond_least": d.cond.to_numpy()[A],
        "cond_most": d.cond.to_numpy()[B],
        "crowd_least": d[PRIMARY].to_numpy()[A],
        "crowd_most": d[PRIMARY].to_numpy()[B],
        "span": P.span.to_numpy(),
        "hit_least": hit[A], "hit_most": hit[B],
        "fa_least": fa[A], "fa_most": fa[B],
        "n_old_least": n_old[A], "n_old_most": n_old[B],
        "n_new_least": n_new[A], "n_new_most": n_new[B],
        "dprime_least": dp_cells[A], "dprime_most": dp_cells[B],
        "pred_hit_least": ph[A], "pred_hit_most": ph[B],
        "pred_fa_least": pf[A], "pred_fa_most": pf[B],
        "d_hit": obs_deltas.d_hit, "d_fa": obs_deltas.d_fa, "d_dprime": obs_deltas.d_dprime,
        "model_d_hit": mod_deltas.d_hit, "model_d_fa": mod_deltas.d_fa,
        "model_d_dprime": mod_deltas.d_dprime,
    })
    per_image.to_csv(OUT / "paired_crowding_images.csv", index=False)

    rows = []
    for c, lab in labels.items():
        rows.append({"measure": lab, "source": "observed", "mean": obs_ci[c][0],
                     "lo_images": obs_ci[c][1], "hi_images": obs_ci[c][2]})
        rows.append({"measure": lab, "source": "model", "mean": mod_ci[c][0],
                     "lo_images": mod_ci[c][1], "hi_images": mod_ci[c][2]})
    pd.DataFrame(rows).to_csv(OUT / "paired_crowding_summary.csv", index=False)

    print(f"\nwrote {OUT/'paired_crowding.txt'}, paired_crowding_images.csv, "
          f"paired_crowding_summary.csv")


if __name__ == "__main__":
    main()
