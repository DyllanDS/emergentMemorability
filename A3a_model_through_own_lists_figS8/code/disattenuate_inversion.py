"""
disattenuate_inversion.py: the reliability of the observed item scores of the inversion experiment and
the model-to-data correlation corrected for it (SI Appendix, Section 3: "0.31 after correcting for the
reliability of the observed scores").

WHAT IT DOES
  Each critical item's observed corrected recognition (CR, hit rate minus false alarm rate) rests on
  about two dozen old and two dozen new trials, so part of the spread of CR across items is binomial
  sampling noise. Reliability = 1 - (mean over items of the sampling variance of CR) / (variance of
  CR across items, n - 1 in the denominator), with the sampling variance of an item's CR
  h(1 - h) / n_old + f(1 - f) / n_new. The Spearman correlation of observed CR with the model's
  prediction is divided by the square root of that reliability (only observed CR is treated as
  noisy). The report also gives the raw correlations of observed CR and of the model's CR with the
  published score (Kramer et al. 2023).

INPUT   outputs/fit_inversion_items.csv (written by code/fit_inversion.py: per-item hit rate, false
        alarm rate and CR, the model's predictions, the published score, trial counts)
OUTPUT  outputs/disattenuation.txt

RUN (from the repository root; a second):  python A3a_model_through_own_lists_figS8/code/disattenuate_inversion.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (first: it stops Python from writing __pycache__ folders)

PATHS = release_paths.module_paths(__file__)

items = pd.read_csv(os.path.join(PATHS['outputs'], "fit_inversion_items.csv"), float_precision="round_trip")

# ---- 1. per-item corrected recognition and its sampling noise ----
# CR is hit rate minus false alarm rate, each an independent proportion
# estimated from a finite number of old/new trials. Treating each rate as
# a Bernoulli proportion, its sampling variance is p(1-p)/n; the two error
# sources add because hits and false alarms come from disjoint trials.
hit, fa = items["hitRate"], items["faRate"]
n_old, n_new = items["nOld"], items["nNew"]
cr = hit - fa
sampling_var = hit * (1 - hit) / n_old + fa * (1 - fa) / n_new

median_n_old = n_old.median()
median_n_new = n_new.median()
mean_sampling_var = sampling_var.mean()

# ---- 2. reliability = 1 - (noise variance) / (total between-item variance) ----
# Between-item variance in observed CR mixes true differences between items
# with binomial sampling error. 1 minus the share that is sampling error is
# the reliability of the observed CR scores, the quantity that a split-half
# or test-retest reliability coefficient estimates.
observed_var = cr.var(ddof=1)
sampling_share = mean_sampling_var / observed_var
reliability = 1 - sampling_share

# ---- 3. raw item-level correlations (Spearman rho, as the paper reports them) ----
rho_model_obs_cr, _ = spearmanr(items["pred_cr"], cr)
rho_obs_cr_mem, _ = spearmanr(cr, items["origMem"])
rho_model_cr_mem, _ = spearmanr(items["pred_cr"], items["origMem"])

# ---- 4. disattenuation ----
# Only observed CR carries measurement error here (the model's predictions
# are treated as fixed, noise-free values), so the correction divides by the
# square root of the reliability of that one noisy variable.
correction = np.sqrt(reliability)
rho_model_obs_cr_corrected = rho_model_obs_cr / correction

lines = []
lines.append("DISATTENUATED CORRELATION, INVERSION EXPERIMENT")
lines.append("=" * 78)
lines.append(f"{len(items)} critical items, from outputs/fit_inversion_items.csv")
lines.append("")
lines.append("TRIAL COUNTS")
lines.append(f"  median old (studied) trials per item: {median_n_old:g}")
lines.append(f"  median new (lure) trials per item:    {median_n_new:g}")
lines.append("")
lines.append("RELIABILITY OF OBSERVED CORRECTED RECOGNITION")
lines.append(f"  mean per-item binomial sampling variance of CR: {mean_sampling_var:.5f}")
lines.append(f"  between-item variance of observed CR (sample, ddof=1): {observed_var:.5f}")
lines.append(f"  sampling variance as a share of between-item variance: {sampling_share:.3f} ({sampling_share*100:.1f}%)")
lines.append(f"  reliability = 1 - share = {reliability:.4f}")
lines.append("")
lines.append("CORRELATIONS (Spearman rho, item level)")
lines.append(f"  {'':32s}{'raw':>10s}{'disattenuated':>16s}{'model':>10s}")
lines.append(f"  {'CR vs original THINGS memorability':32s}{rho_obs_cr_mem:>10.3f}{'--':>16s}{rho_model_cr_mem:>10.3f}")
lines.append(f"  {'model vs observed CR':32s}{rho_model_obs_cr:>10.3f}{rho_model_obs_cr_corrected:>16.3f}{'--':>10s}")
lines.append("")
lines.append("WHAT THE CORRECTION MEANS")
lines.append(
    "  Each item's observed CR rests on about two dozen old and two dozen new trials, "
    "so part of the spread of observed CR across items is binomial sampling noise. "
    "Dividing the raw correlation of the model's predictions with observed CR by the "
    "square root of the reliability estimates that correlation for observed CR measured "
    "without this noise; the model's predictions are treated as noise-free. The first "
    "row gives the published score (original THINGS memorability) against observed CR "
    "(raw) and against the model's CR (model)."
)

with open(os.path.join(PATHS['outputs'], "disattenuation.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
