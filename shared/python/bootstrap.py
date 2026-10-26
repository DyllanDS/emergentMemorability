"""
bootstrap.py: resampling statistics for the release's statistics layer (statistics/README.md).

Every function that draws random numbers takes the seed as an argument, uses ONE numpy random
generator built from it (numpy.random.default_rng(seed)) and draws in a fixed order, so every
interval reproduces exactly; seeded_per_resample_mean_bootstrap() instead seeds resample i with i.
The default seed is 20261013 (RELEASE_SEED).

    percentile_bootstrap(statistic, n_units, n_resamples, seed)
        resample the units (participants, items or concepts: say which) with replacement,
        recompute the statistic on each resample; returns the resampled statistics
    percentile_interval(resampled, level=0.95)
        the percentile interval of resampled statistics (numpy percentile, linear interpolation)
    matlab_prctile(values, percents)
        MATLAB's prctile, for the intervals that the MATLAB code computes with prctile (Fig. 7d)
    permutation_test(statistic, values_to_shuffle, n_permutations, seed, ...)
        shuffle one variable across units, recompute the statistic; two-sided p
    split_half_reliability(score_for_half, n_units, n_splits, seed)
        random halves of the units (participants), Pearson or Spearman correlation of the two
        halves' item scores, Spearman-Brown corrected; mean and 2.5 to 97.5 percent over splits
    spearman_brown(r)                    the Spearman-Brown step-up of a half-sample correlation
    fisher_z_interval(r, n, level)       the parametric interval of a correlation
    seeded_per_resample_mean_bootstrap(differences, n_resamples)
        the face study's scheme (modules 04 and A3c): one generator per resample, seeded with the
        resample number i, drawing participant indices integers(0, n, n); returns the means

Explicit loops throughout. Where a module needs 10,000 resamples of a heavy statistic it may pass a
statistic function that is vectorized inside (one step, documented where it is used).
"""
import numpy as np
from scipy.stats import norm, pearsonr, spearmanr

RELEASE_SEED = 20261013


def percentile_bootstrap(statistic, n_units, n_resamples=10000, seed=RELEASE_SEED):
    """statistic(indices) -> number (or 1-D array) computed on the units listed in indices
    (a resample with replacement of 0 .. n_units - 1). Returns an array with one row per resample.
    The generator draws rng.integers(0, n_units, n_units) once per resample, in order."""
    generator = np.random.default_rng(seed)
    results = []
    for resample in range(n_resamples):
        indices = generator.integers(0, n_units, n_units)
        results.append(statistic(indices))
    return np.asarray(results, dtype=float)


def percentile_interval(resampled, level=0.95):
    """(low, high) percentile interval of resampled statistics; NaN resamples are ignored.
    Works on a 1-D array, or column by column on a 2-D array."""
    resampled = np.asarray(resampled, dtype=float)
    tail = 100.0 * (1.0 - level) / 2.0
    low = np.nanpercentile(resampled, tail, axis=0)
    high = np.nanpercentile(resampled, 100.0 - tail, axis=0)
    return low, high


def matlab_prctile(values, percents):
    """MATLAB prctile (Statistics and Machine Learning Toolbox): NaN removed; the p-th percentile
    is at position n * p / 100 + 0.5 of the sorted data (counting from 1), linear interpolation
    between neighbors, clamped to the smallest and largest value."""
    values = np.asarray(values, dtype=float)
    sorted_values = np.sort(values[np.isfinite(values)])
    n = len(sorted_values)
    results = []
    for percent in np.atleast_1d(percents):
        position = n * percent / 100.0 + 0.5
        if position <= 1:
            results.append(sorted_values[0])
        elif position >= n:
            results.append(sorted_values[-1])
        else:
            lower = int(np.floor(position))
            fraction = position - lower
            results.append(sorted_values[lower - 1] + fraction * (sorted_values[lower] - sorted_values[lower - 1]))
    return np.array(results)


def permutation_test(statistic, values_to_shuffle, n_permutations=5000, seed=RELEASE_SEED,
                     p_formula='count_plus_one'):
    """Permutation null for statistic(values) -> number, where values_to_shuffle is the variable
    shuffled across units (for example memorability across concepts). The observed statistic uses
    the values as given; each permutation uses generator.permutation(values).
    Two-sided p:
        p_formula='count_plus_one'  (1 + number of |null| >= |observed|) / (1 + n_permutations)
                                    (the default; never exactly 0)
        p_formula='proportion'      number of |null| >= |observed| / n_permutations
    Returns (observed, p, null_values)."""
    generator = np.random.default_rng(seed)
    values_to_shuffle = np.asarray(values_to_shuffle)
    observed = float(statistic(values_to_shuffle))
    null_values = np.empty(n_permutations)
    for permutation in range(n_permutations):
        null_values[permutation] = statistic(generator.permutation(values_to_shuffle))
    at_least_as_extreme = int(np.sum(np.abs(null_values) >= abs(observed)))
    if p_formula == 'count_plus_one':
        p = (1.0 + at_least_as_extreme) / (1.0 + n_permutations)
    elif p_formula == 'proportion':
        p = at_least_as_extreme / float(n_permutations)
    else:
        raise ValueError("p_formula must be 'count_plus_one' or 'proportion'")
    return observed, p, null_values


def spearman_brown(half_correlation):
    """Reliability of the full sample from the correlation between two halves: 2r / (1 + r)."""
    half_correlation = np.asarray(half_correlation, dtype=float)
    return 2.0 * half_correlation / (1.0 + half_correlation)


def split_half_reliability(score_for_half, n_units, n_splits=100, seed=RELEASE_SEED, method='spearman'):
    """Split-half reliability over units (participants).

    score_for_half(indices) -> array of item scores computed from the units in indices.
    Each split: generator.permutation(n_units); first half = the first n_units // 2 of the permuted
    order, second half = the rest. The two halves' item scores are correlated (Spearman by default,
    Pearson with method='pearson'), then Spearman-Brown corrected.
    Returns a dict with the raw half correlations, the corrected values, their means and the 2.5 and
    97.5 percentiles over splits."""
    generator = np.random.default_rng(seed)
    half_correlations = []
    for split in range(n_splits):
        order = generator.permutation(n_units)
        first_half = order[:n_units // 2]
        second_half = order[n_units // 2:]
        scores_first = np.asarray(score_for_half(first_half), dtype=float)
        scores_second = np.asarray(score_for_half(second_half), dtype=float)
        usable = np.isfinite(scores_first) & np.isfinite(scores_second)
        if method == 'spearman':
            correlation = spearmanr(scores_first[usable], scores_second[usable]).statistic
        else:
            correlation = pearsonr(scores_first[usable], scores_second[usable]).statistic
        half_correlations.append(correlation)
    half_correlations = np.asarray(half_correlations, dtype=float)
    corrected = spearman_brown(half_correlations)
    return dict(half_correlations=half_correlations, corrected=corrected,
                mean_half=float(np.mean(half_correlations)), mean_corrected=float(np.mean(corrected)),
                half_low=float(np.percentile(half_correlations, 2.5)),
                half_high=float(np.percentile(half_correlations, 97.5)),
                corrected_low=float(np.percentile(corrected, 2.5)),
                corrected_high=float(np.percentile(corrected, 97.5)),
                n_splits=n_splits, seed=seed, method=method)


def fisher_z_interval(r, n, level=0.95):
    """Parametric interval of a correlation r over n pairs: tanh(atanh(r) -+ z / sqrt(n - 3))."""
    z = np.arctanh(r)
    half_width = norm.ppf(1.0 - (1.0 - level) / 2.0) / np.sqrt(n - 3.0)
    return float(np.tanh(z - half_width)), float(np.tanh(z + half_width))


def seeded_per_resample_mean_bootstrap(differences, n_resamples=10000):
    """The participant bootstrap of the face study (modules 04 and A3c): for resample i a new
    generator numpy.random.default_rng(i) draws integers(0, n, n) participant indices and the mean
    of the paired differences is taken. The result depends on the order of the participants; the
    face data give that order in the column load_order (the order in which the registered analysis
    loads the sessions). Returns the array of resampled means."""
    differences = np.asarray(differences, dtype=float)
    n = len(differences)
    means = np.empty(n_resamples)
    for resample in range(n_resamples):
        indices = np.random.default_rng(resample).integers(0, n, n)
        means[resample] = np.mean(differences[indices])
    return means
