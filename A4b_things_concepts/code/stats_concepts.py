"""
stats_concepts.py: the uncertainty statistics of the THINGS concepts sentence of SI Appendix,
Section 4 (test T46 of statistics/README.md).

WHAT IT DOES
  For each of the four Spearman correlations of that sentence (crowding among the 1,854 THINGS
  concepts at k = 5 against the concept-mean false alarm rate, hit rate, d′ and corrected
  recognition), in the order they appear in the paper, it adds two rows to tests.csv:
  1. rho, n = 1,854 concepts, df = n - 2, the two-sided p at full precision, and a 95 percent
     interval by the Fisher z transform with the Bonett and Wright (2000) standard error for
     Spearman rho, sqrt((1 + rho^2 / 2) / (n - 3)). The p value is also recomputed from rho and n
     (the t approximation scipy uses) and printed as a check.
  2. the percentile bootstrap over concepts (10,000 resamples, seed 20261013; each concept keeps its
     crowding, computed once over all 1,854 concepts, and its memorability). This needs the concept
     memorability, so the interval is computed only when the count table is present
     (EXTERNAL_INPUTS.md M2); without it the row gives the method and no interval.
  Without the count table, rho and p come from the saved results (data/provided/concept_level.json);
  with it, from the module's own run. No correction for multiple comparisons (statistics/README.md).

INPUTS   the values run_module.py passes in
OUTPUTS  rows of outputs/tests.csv (written by run_module.py)
SEED     20261013 (the concept bootstrap, with M2 only)
"""
import os
import sys

sys.dont_write_bytecode = True

import numpy as np
from scipy.stats import norm, t as t_distribution

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import bootstrap      # noqa: E402

import concept_level  # noqa: E402

SCRIPT = 'code/stats_concepts.py'
LOCATION = 'SI Appendix, Section 4, One Measure of Distinctiveness Across Analyses (THINGS concepts)'
NUMBER_OF_BOOTSTRAP_RESAMPLES = 10000

# The four correlations in the order they appear in the paper, with their ledger items.
MEASURES_IN_TEXT_ORDER = ['FA rate', 'hit rate', 'd-prime', 'CR']
LEDGER_ITEM_OF_MEASURE = {'FA rate': 'L9-021', 'd-prime': 'L9-022', 'hit rate': 'L9-023', 'CR': 'L9-024'}
MEASURE_DESCRIPTION = {
    'FA rate': 'the concept-mean false alarm rate',
    'd-prime': "the concept d′ (z of the concept-mean hit rate minus z of the concept-mean false alarm rate)",
    'hit rate': 'the concept-mean hit rate',
    'CR': 'the concept-mean corrected recognition (hit rate minus false alarm rate)',
}


def fisher_z_interval_spearman(rho, n, level=0.95):
    """95 percent interval of a Spearman rho: Fisher z transform with the Bonett and Wright (2000)
    standard error sqrt((1 + rho^2 / 2) / (n - 3))."""
    z = np.arctanh(rho)
    standard_error = np.sqrt((1.0 + rho ** 2 / 2.0) / (n - 3.0))
    critical_value = norm.ppf(1.0 - (1.0 - level) / 2.0)
    return float(np.tanh(z - critical_value * standard_error)), float(np.tanh(z + critical_value * standard_error))


def spearman_p_from_rho(rho, n):
    """Two-sided p of a Spearman rho over n pairs by the t approximation (what scipy's spearmanr
    reports): t = rho * sqrt((n - 2) / ((1 - rho)(1 + rho))), df = n - 2."""
    degrees_of_freedom = n - 2
    t_value = rho * np.sqrt(degrees_of_freedom / ((1.0 - rho) * (1.0 + rho)))
    return float(2.0 * t_distribution.sf(abs(t_value), degrees_of_freedom))


def concept_bootstrap(crowding, memorability, n_resamples, seed=bootstrap.RELEASE_SEED):
    """Percentile bootstrap over concepts of the four Spearman rhos.
    crowding: the k = 5 crowding of every concept; memorability: {measure: concept values}.
    One generator draws 1,854 concept indices with replacement per resample (shared/python/
    bootstrap.py); each resample recomputes the four rhos in MEASURES_IN_TEXT_ORDER.
    Returns {measure: (low, high)}."""
    number_of_concepts = len(crowding)

    def four_correlations(concept_indices):
        correlations = []
        for measure in MEASURES_IN_TEXT_ORDER:
            rho, p = concept_level.spearman(crowding[concept_indices], memorability[measure][concept_indices])
            correlations.append(rho)
        return correlations

    resampled = bootstrap.percentile_bootstrap(four_correlations, number_of_concepts, n_resamples, seed)
    low, high = bootstrap.percentile_interval(resampled)
    intervals = {}
    for position, measure in enumerate(MEASURES_IN_TEXT_ORDER):
        intervals[measure] = (float(low[position]), float(high[position]))
    return intervals


def add_tests(tests, correlations, n, source, crowding=None, memorability=None):
    """Adds the T46 rows to a tests_table.TestsTable.

    correlations: {measure: (rho, p)} at k = 5; n: the number of concepts; source: 'provided' (the
    saved results) or 'regenerated' (this module's run, with crowding and memorability given for the
    concept bootstrap). Returns a list of check lines for the log."""
    check_lines = []
    bootstrap_intervals = None
    n_resamples = NUMBER_OF_BOOTSTRAP_RESAMPLES
    if source == 'regenerated':
        bootstrap_intervals = concept_bootstrap(crowding, memorability, n_resamples)

    for measure in MEASURES_IN_TEXT_ORDER:
        ledger_id = LEDGER_ITEM_OF_MEASURE[measure]
        rho, p = correlations[measure]
        ci_low, ci_high = fisher_z_interval_spearman(rho, n)
        recomputed_p = spearman_p_from_rho(rho, n)
        check_lines.append('%s: rho %+.6f, p %.6g; p recomputed from rho and n = %.6g (relative difference %.1e)'
                           % (measure, rho, p, recomputed_p, abs(recomputed_p - p) / p))
        where_from = ("rho and p from the saved results (data/provided/concept_level.json), used without the "
                      "per-image count table (EXTERNAL_INPUTS.md M2)" if source == 'provided'
                      else "rho and p from this module's run")
        description = ('Spearman rho across the 1,854 THINGS concepts between deep-network crowding (minus the mean '
                       'scaled distance to the five nearest concept centroids, k = 5, in the 14-space ensemble) and %s '
                       'of Kramer et al. (2023)' % MEASURE_DESCRIPTION[measure])
        tests.add(test_id='T46', ledger_ids=ledger_id, location=LOCATION,
                  test='%s; %s; interval from rho and n' % (description, where_from),
                  statistic_name='Spearman rho', statistic=rho, n=n, unit='concepts (26,107 photographs)',
                  df=n - 2, p=p, p_type='parametric two-sided', effect_size_name='rho', effect_size=rho,
                  ci_low=ci_low, ci_high=ci_high,
                  ci_method='Fisher z with the Bonett and Wright (2000) standard error for Spearman rho, '
                            'sqrt((1 + rho^2 / 2) / (n - 3))')

        bootstrap_test = ('%s; percentile bootstrap over the concepts (each concept keeps its crowding, computed once '
                          'over all 1,854 concepts, and its memorability)' % description)
        if bootstrap_intervals is None:
            tests.add(test_id='T46', ledger_ids=ledger_id, location=LOCATION, test=bootstrap_test,
                      statistic_name='Spearman rho', n=n, unit='concepts (26,107 photographs)', p_type='none',
                      effect_size_name='rho',
                      ci_method='percentile bootstrap, %s resamples, seed %d; computed only with the per-image '
                                'count table (EXTERNAL_INPUTS.md M2)' % (format(n_resamples, ','), bootstrap.RELEASE_SEED),
                      resampling_unit='concepts')
        else:
            bootstrap_low, bootstrap_high = bootstrap_intervals[measure]
            tests.add(test_id='T46', ledger_ids=ledger_id, location=LOCATION, test=bootstrap_test,
                      statistic_name='Spearman rho', statistic=rho, n=n, unit='concepts (26,107 photographs)',
                      p_type='none', effect_size_name='rho', effect_size=rho,
                      ci_low=bootstrap_low, ci_high=bootstrap_high, ci_method='percentile bootstrap',
                      resampling_unit='concepts', n_resamples=n_resamples, seed=bootstrap.RELEASE_SEED)
    return check_lines
