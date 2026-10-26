"""
sdt.py: the signal-detection definitions the analyses use, in one place.

The paper's memory measure for an item is its corrected recognition: hit rate minus false alarm
rate (old responses to studied items minus old responses to new items). Some analyses also report
d' and the criterion c, d_a (d' allowing unequal variances, at a fixed z-ROC slope), or work with
binned counts. Where a module uses a definition, the list below names it.

    rates and corrected recognition
        hit_rate(hits, n_old), false_alarm_rate(false_alarms, n_new), corrected_recognition(h, f)
                                                 (module 04b, per face)
    d' and criterion c
        loglinear_rate(k, n)                     (k + 0.5) / (n + 1)  (Hautus 1995)
        dprime_loglinear(h, f, n_old, n_new)     d' and c from log-linear corrected rates, given the
                                                 RATES and trial counts (the face study, through
                                                 face_loader.participant_rates())
        dprime_loglinear_counts(k_h, n_old, k_f, n_new)
                                                 the same from COUNTS (modules A3b and 04b, per item; the
                                                 MATLAB twin sdt_rates.m dprimeLoglinear takes counts)
        clamped_rate(rate, n)                    rate kept inside [1/(2n), 1 - 1/(2n)]
        dprime_clamped(h, f, n_old, n_new)       d' with the 1/(2N) clamp (module A1, per concept)
        criterion_c(z_hit, z_fa)                 -(z_hit + z_fa) / 2
    d_a at a fixed z-ROC slope
        da_fixed_slope(h, f, slope)              sqrt(2 / (1 + s^2)) * (z(h) - s z(f))
        da_from_rating_counts(target_counts, lure_counts, slope=0.8)
                                                 the Python twin of daFromRatingCounts in
                                                 shared/matlab/sdt_rates.m (module A1): counts over the
                                                 six confidence levels from 'sure old' to 'sure new';
                                                 h and f at the middle criterion
    binned SDT for the THINGS continuous-recognition data and the model run through it
        N_BINS, DA_SLOPE, within_image_bin(), cell_counts(), sdt(), used by modules A2c
        (continuous_rem.py, stats_figS4.py) and A5 (sdt_by_context.py and the scripts that use its
        bins). continuous_rem.py sets sdt.N_BINS before calling cell_counts().

Inputs are numbers or numpy arrays; nothing here reads files or draws random numbers.
"""
import numpy as np
from scipy.stats import norm

# ----------------------------------------------------------------------------- rates
def hit_rate(hits, n_old):
    """Proportion of studied (old) test items called old."""
    return np.asarray(hits, dtype=float) / np.asarray(n_old, dtype=float)


def false_alarm_rate(false_alarms, n_new):
    """Proportion of new test items called old."""
    return np.asarray(false_alarms, dtype=float) / np.asarray(n_new, dtype=float)


def corrected_recognition(hit_rate_value, false_alarm_rate_value):
    """The paper's memorability of an item: hit rate minus false alarm rate."""
    return np.asarray(hit_rate_value, dtype=float) - np.asarray(false_alarm_rate_value, dtype=float)


# ----------------------------------------------------------------------------- d' and criterion
def loglinear_rate(count, n):
    """Log-linear corrected proportion (count + 0.5) / (n + 1), so 0 and 1 never occur."""
    return (np.asarray(count, dtype=float) + 0.5) / (np.asarray(n, dtype=float) + 1.0)


def criterion_c(z_hit, z_false_alarm):
    """Criterion c = -(z(H) + z(F)) / 2; positive = conservative (fewer old responses)."""
    return -0.5 * (np.asarray(z_hit, dtype=float) + np.asarray(z_false_alarm, dtype=float))


def dprime_loglinear(hit_rate_value, false_alarm_rate_value, n_old, n_new):
    """d' and c from log-linear corrected rates. Takes rates and the numbers of trials behind them:
    z((h * nh + 0.5) / (nh + 1)) - z((f * nf + 0.5) / (nf + 1)).
    Returns (dprime, c)."""
    z_hit = norm.ppf(loglinear_rate(np.asarray(hit_rate_value, dtype=float) * n_old, n_old))
    z_false_alarm = norm.ppf(loglinear_rate(np.asarray(false_alarm_rate_value, dtype=float) * n_new, n_new))
    return z_hit - z_false_alarm, criterion_c(z_hit, z_false_alarm)


def dprime_loglinear_counts(hits, n_old, false_alarms, n_new):
    """d' and c from counts: z((hits + 0.5) / (n_old + 1)) - z((false_alarms + 0.5) / (n_new + 1)).
    Same numbers as dprime_loglinear() given the rates; this is the form of the MATLAB twin
    sdt_rates.m (dprimeLoglinear takes counts). Returns (dprime, c)."""
    z_hit = norm.ppf(loglinear_rate(hits, n_old))
    z_false_alarm = norm.ppf(loglinear_rate(false_alarms, n_new))
    return z_hit - z_false_alarm, criterion_c(z_hit, z_false_alarm)


def clamped_rate(rate, n):
    """A rate kept inside [1 / (2n), 1 - 1 / (2n)] (the 1/(2N) correction for 0 and 1)."""
    n = np.asarray(n, dtype=float)
    return np.minimum(np.maximum(np.asarray(rate, dtype=float), 1.0 / (2.0 * n)), 1.0 - 1.0 / (2.0 * n))


def dprime_clamped(hit_rate_value, false_alarm_rate_value, n_old, n_new):
    """d' with the 1/(2N) clamp, as module A1 computes it per concept.
    Returns (dprime, c)."""
    z_hit = norm.ppf(clamped_rate(hit_rate_value, n_old))
    z_false_alarm = norm.ppf(clamped_rate(false_alarm_rate_value, n_new))
    return z_hit - z_false_alarm, criterion_c(z_hit, z_false_alarm)


# ----------------------------------------------------------------------------- d_a
def da_fixed_slope(hit_rate_value, false_alarm_rate_value, slope=0.8):
    """d_a at a fixed z-ROC slope s: sqrt(2 / (1 + s^2)) * (z(H) - s * z(F)).
    The paper's fixed slope is 0.8 (Didi-Barnea et al. 2016; SI Appendix, Section 1)."""
    z_hit = norm.ppf(np.asarray(hit_rate_value, dtype=float))
    z_false_alarm = norm.ppf(np.asarray(false_alarm_rate_value, dtype=float))
    return np.sqrt(2.0 / (1.0 + slope ** 2)) * (z_hit - slope * z_false_alarm)


def da_from_rating_counts(target_counts, lure_counts, slope=0.8):
    """Python twin of daFromRatingCounts in shared/matlab/sdt_rates.m.

    target_counts, lure_counts: counts of the six responses ordered from 'sure old' to 'sure new'.
    Cumulative proportions give the ROC points; at least two finite z pairs are required (else NaN);
    h and f are taken at the middle criterion (after the third response category) and d_a is
    computed at the fixed slope."""
    target_counts = np.asarray(target_counts, dtype=float)
    lure_counts = np.asarray(lure_counts, dtype=float)
    number_of_levels = len(target_counts)
    total_targets = target_counts.sum()
    total_lures = lure_counts.sum()
    if total_targets == 0 or total_lures == 0:
        return np.nan
    target_cumulative = np.cumsum(target_counts) / total_targets
    lure_cumulative = np.cumsum(lure_counts) / total_lures
    with np.errstate(divide='ignore'):
        target_z = norm.ppf(target_cumulative[:number_of_levels - 1])
        lure_z = norm.ppf(lure_cumulative[:number_of_levels - 1])
    valid_pairs = np.isfinite(target_z) & np.isfinite(lure_z)
    if valid_pairs.sum() < 2:
        return np.nan
    middle = number_of_levels // 2                     # MATLAB floor(nBins / 2), counted from 1
    hit = target_cumulative[middle - 1]
    false_alarm = lure_cumulative[middle - 1]
    return float(np.sqrt(2.0 / (1.0 + slope ** 2)) * (norm.ppf(hit) - slope * norm.ppf(false_alarm)))


# ----------------------------------------------------------------------------- binned SDT
# Binned SDT of the THINGS continuous-recognition data (SI Appendix, Section 5) and of the model run
# through the same trial sequences (SI Appendix, Fig. S4). The driver code is in module A5
# (sdt_by_context.py).
N_BINS = 20
DA_SLOPE = 0.80   # the z-ROC slope of d_a in the paper (Didi-Barnea et al. 2016)


def within_image_bin(x, img, n_bins=N_BINS):
    """Bin of x within each image (bin 0 = lowest x for that image); -1 where the image has fewer
    trials than bins. x is the context similarity of each trial (NaN for trials not to be binned),
    img the image of each trial. Vectorized numpy."""
    out = np.full(len(x), -1, dtype=np.int8)
    ok = np.isfinite(x)
    idx = np.flatnonzero(ok)
    order = idx[np.lexsort((x[idx], img[idx]))]
    im = img[order]
    _, start, counts = np.unique(im, return_index=True, return_counts=True)
    rank = np.arange(len(order)) - np.repeat(start, counts)
    cnt = np.repeat(counts, counts)
    frac = rank / np.maximum(cnt - 1, 1)
    b = np.minimum((frac * n_bins).astype(np.int8), n_bins - 1)
    b[cnt < n_bins] = -1
    out[order] = b
    return out


def cell_counts(sub, b, y, n_sub):
    """Successes and opportunities per (subject, bin): sub = subject index of each trial (0..n_sub-1),
    b = bin of each trial (-1 = not binned), y = 1 for a success (an old response). Uses the
    module-level N_BINS at call time (continuous_rem.py in module A2c sets it first)."""
    keep = b >= 0
    s, bb, yy = sub[keep], b[keep].astype(np.int64), y[keep]
    flat = s * N_BINS + bb
    n = np.bincount(flat, minlength=n_sub * N_BINS).reshape(n_sub, N_BINS)
    k = np.bincount(flat, weights=yy, minlength=n_sub * N_BINS).reshape(n_sub, N_BINS)
    return k, n


def sdt(hk, hn, fk, fn):
    """Log-linear corrected d', criterion and d_a (at DA_SLOPE) from pooled counts:
    hk hits out of hn old trials, fk false alarms out of fn new trials (arrays per bin).
    Returns (dprime, criterion, d_a, H, F)."""
    H = (hk + 0.5) / (hn + 1.0)
    F = (fk + 0.5) / (fn + 1.0)
    zH, zF = norm.ppf(H), norm.ppf(F)
    dprime = zH - zF
    crit = -0.5 * (zH + zF)
    da = np.sqrt(2 / (1 + DA_SLOPE ** 2)) * (zH - DA_SLOPE * zF)
    return dprime, crit, da, H, F
