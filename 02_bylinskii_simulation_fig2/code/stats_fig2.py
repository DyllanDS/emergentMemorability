"""
stats_fig2.py: the statistics of Fig. 2 (tests T03, T04 and T05 of statistics/all_tests.csv) and the
numbers.csv rows of the four Fig. 2 correlations (L1-034 to L1-041: rho, 95% CI and p).

What it does
    Reads the per-target values of the saved simulation run in data/paper_run/fig2_paper_run.mat
    (the run the paper reports; run_module('full') reproduces it in every field).
    For each of the 1,575 simulated targets the run holds
        memorability (hit rate minus false alarm rate) in Simulation 1 (within category) and in
        Simulation 2 (across categories), and
        summed similarity in each context: minus the mean Euclidean distance to the other 299 items
        of the target's category (Simulation 1) or to all other 6,299 items (Simulation 2), so that
        higher values mean more similar to the other items, that is, less distinctive.
    For each of the four Spearman correlations of Fig. 2 it gives rho, n = 1,575 targets, the
    two-sided p (t approximation with n - 2 degrees of freedom, as scipy computes it), and a 95%
    percentile bootstrap interval over targets: 10,000 resamples of the 1,575 targets with
    replacement (seed 20261013), the same resamples for every correlation.
    The p of the summed-similarity correlation of Fig. 2b (about 1e-682) is smaller than the
    smallest number a computer stores in double precision, so scipy (and MATLAB) return 0. For this
    correlation the same p is computed through its logarithm (see two_sided_p below) and written as
    text (5.81097e-682); the script checks this method against scipy on the three correlations whose
    p scipy can still represent.
        T03  memorability within vs across (Fig. 2a)
        T04  memorability vs summed similarity, within category and across categories (Fig. 2a);
             a negative rho means that the more distinctive targets (lower summed similarity) are
             the more memorable ones
        T05  summed similarity within vs across (Fig. 2b)

    The bootstrap resamples targets, not simulation runs: each target's memorability is already an
    average over its tests in 1,000 runs per context. The interval shows how much a correlation
    depends on which targets the simulated set contains.

Inputs
    data/paper_run/fig2_paper_run.mat
Outputs
    outputs/tests.csv   one row per correlation (columns of statistics/README.md)
    outputs/numbers.csv the rows L1-034 to L1-041, next to the rows that bylinskii_numbers.m writes
                        (run_module runs first; this script replaces only its own rows)
Seed
    20261013 (numpy.random.default_rng), 10,000 resamples
Run (from the repository root, about 15 seconds)
    python 02_bylinskii_simulation_fig2/code/stats_fig2.py
"""
import os
import sys

sys.dont_write_bytecode = True     # keep the release free of __pycache__ folders
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths                 # noqa: E402
import bootstrap                   # noqa: E402
import tests_table                 # noqa: E402
import numbers_table               # noqa: E402

import csv                         # noqa: E402
import math                        # noqa: E402

import numpy as np                 # noqa: E402
import scipy.io                    # noqa: E402
from scipy.special import betaln, hyp2f1                # noqa: E402
from scipy.stats import pearsonr, rankdata, spearmanr   # noqa: E402

SEED = bootstrap.RELEASE_SEED      # 20261013
NUMBER_OF_RESAMPLES = 10000
PAPER_SEED = 47826
SCRIPT = 'code/stats_fig2.py'

FALSE_ALARM_PROCEDURE = ('false alarms: in every run each studied target is probed against 30 newly generated '
                         'and stored lure items and counts as a false alarm when the mean likelihood ratio '
                         'exceeds the criterion')
SUMMED_SIMILARITY_DEFINITION = ('summed similarity = minus the mean Euclidean distance to the other 299 items of '
                                'the target\'s category (within) or to the other 6,299 items (across); higher = '
                                'less distinctive')

# The four correlations of Fig. 2: (name, test id, ledger ids, numbers.csv rows written here, location in the
# paper, x values, y values, statistic, description)
CORRELATIONS = [
    dict(name='memorability within vs across', test_id='T03', ledger_ids='L1-034;L1-038', numbers_ids=('L1-034', 'L1-038'),
         location='Fig. 2a caption', x='memorability_within', y='memorability_across',
         statistic_name='Spearman rho',
         test=('Spearman correlation over the simulated targets between memorability (hit rate minus false alarm '
               'rate) in Simulation 1 (within category, criterion 20) and Simulation 2 (across categories, '
               'criterion 5); seed 47826, 1,000 runs per category within and 1,000 runs across; '
               + FALSE_ALARM_PROCEDURE)),
    dict(name='memorability vs summed similarity, within category', test_id='T04', ledger_ids='L1-035;L1-039', numbers_ids=('L1-035', 'L1-039'),
         location='Fig. 2a caption', x='memorability_within', y='summed_similarity_within',
         statistic_name='Spearman rho',
         test=('Spearman correlation over the simulated targets between memorability in Simulation 1 (within '
               'category) and summed similarity within category; ' + SUMMED_SIMILARITY_DEFINITION
               + '; a negative rho means that the more distinctive targets are the more memorable ones')),
    dict(name='memorability vs summed similarity, across categories', test_id='T04', ledger_ids='L1-036;L1-040', numbers_ids=('L1-036', 'L1-040'),
         location='Fig. 2a caption', x='memorability_across', y='summed_similarity_across',
         statistic_name='Spearman rho',
         test=('Spearman correlation over the simulated targets between memorability in Simulation 2 (across '
               'categories) and summed similarity across categories; ' + SUMMED_SIMILARITY_DEFINITION
               + '; a negative rho means that the more distinctive targets are the more memorable ones')),
    dict(name='summed similarity within vs across (Spearman)', test_id='T05', ledger_ids='L1-037;L1-041;L5-024',
         numbers_ids=('L1-037', 'L1-041'),
         location='Fig. 2b caption', x='summed_similarity_within', y='summed_similarity_across',
         statistic_name='Spearman rho',
         test=('Spearman correlation over the simulated targets between summed similarity within category and '
               'across categories; ' + SUMMED_SIMILARITY_DEFINITION)),
]


def load_run(file_path):
    """The per-target values of one saved run of bylinskiiRemSim.m (targets tested in both contexts)."""
    contents = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    figure_data = contents['figureData']
    tested_in_both = np.asarray(figure_data.validBoth, dtype=bool)
    run = {
        'memorability_within': np.asarray(figure_data.amt1_valid, dtype=float),
        'memorability_across': np.asarray(figure_data.amt2_valid, dtype=float),
        'summed_similarity_within': np.asarray(figure_data.summedSim_amt1, dtype=float)[tested_in_both],
        'summed_similarity_across': np.asarray(figure_data.summedSim_amt2, dtype=float)[tested_in_both],
        'seed': int(figure_data.parameters.seedValue),
        'runs_per_context': int(figure_data.parameters.simLen),
        # the correlations the simulation code saved (MATLAB corr)
        'saved': {
            'memorability within vs across': float(figure_data.rho_spearman),
            'memorability vs summed similarity, within category': float(figure_data.rho_withinMem_sim),
            'memorability vs summed similarity, across categories': float(figure_data.rho_acrossMem_sim),
            'summed similarity within vs across (Spearman)': float(figure_data.rho_sim),
        },
    }
    run['number_of_targets'] = len(run['memorability_within'])
    return run


def correlation_value(x_values, y_values, statistic_name):
    """One correlation: Pearson r, or Spearman rho = the Pearson correlation of the ranks (tied values
    share their average rank), which is how MATLAB corr(..., 'Type', 'Spearman') and scipy's
    spearmanr compute it. Written out because the bootstrap calls it 40,000 times."""
    if statistic_name == 'Spearman rho':
        x_values = rankdata(x_values)
        y_values = rankdata(y_values)
    elif statistic_name != 'Pearson r':
        raise ValueError('unknown statistic %s' % statistic_name)
    return float(np.corrcoef(x_values, y_values)[0, 1])


def correlation_p(x_values, y_values, statistic_name):
    """The two-sided p of a correlation (scipy: t approximation with n - 2 degrees of freedom)."""
    if statistic_name == 'Spearman rho':
        return float(spearmanr(x_values, y_values).pvalue)
    return float(pearsonr(x_values, y_values).pvalue)


def log10_two_sided_p(correlation, number_of_pairs):
    """log10 of the two-sided p of a correlation, from the same t statistic scipy uses
    (t = r sqrt(df / (1 - r^2)), df = n - 2), computed through logarithms so that it also works
    when p itself is too small to store (below about 1e-308).
    The two-sided p of a t statistic is the regularized incomplete beta function I_x(a, b) with
    x = df / (df + t^2), a = df / 2, b = 1 / 2, and
        I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) * F(a + b, 1; a + 1; x)
    (F = the Gauss hypergeometric function; Digital Library of Mathematical Functions 8.17.8).
    Taking the logarithm of each factor keeps every intermediate number in range."""
    degrees_of_freedom = number_of_pairs - 2
    t_squared = correlation ** 2 * degrees_of_freedom / (1.0 - correlation ** 2)
    x = degrees_of_freedom / (degrees_of_freedom + t_squared)
    a = degrees_of_freedom / 2.0
    b = 0.5
    log_p = (a * math.log(x) + b * math.log1p(-x) - math.log(a) - betaln(a, b)
             + math.log(hyp2f1(a + b, 1.0, a + 1.0, x)))
    return log_p / math.log(10.0)


def two_sided_p(correlation, scipy_p, number_of_pairs):
    """The p written to tests.csv: scipy's p when scipy can store it; when scipy returns 0 (the p is
    below the smallest double-precision number), the same p from log10_two_sided_p, written as text
    with 6 significant digits (e.g. '5.81097e-682'). Returns (p for tests.csv, how it was computed)."""
    if scipy_p > 0:
        return scipy_p, 'scipy'
    log10_p = log10_two_sided_p(correlation, number_of_pairs)
    exponent = math.floor(log10_p)
    mantissa = 10.0 ** (log10_p - exponent)
    if mantissa >= 9.999995:          # rounding would print 10.00000
        mantissa = mantissa / 10.0
        exponent = exponent + 1
    return '%.5fe%d' % (mantissa, exponent), 'log space'


def all_correlations(run, target_indices):
    """The four Fig. 2 correlations over the targets listed in target_indices (a bootstrap resample)."""
    values = []
    for entry in CORRELATIONS:
        x_values = run[entry['x']][target_indices]
        y_values = run[entry['y']][target_indices]
        values.append(correlation_value(x_values, y_values, entry['statistic_name']))
    return values


def add_rows_to_numbers_csv(numbers, outputs_folder):
    """Write outputs/numbers.csv: the rows of bylinskii_numbers.m (when run_module has run) and the rows of
    this script, in ledger order. Rows of this script from an earlier run are replaced."""
    numbers_file = os.path.join(outputs_folder, 'numbers.csv')
    own_ids = set(row['ledger_id'] for row in numbers.rows)
    other_rows = []
    if os.path.exists(numbers_file):
        with open(numbers_file, newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                if row['ledger_id'] not in own_ids:
                    other_rows.append(row)
    else:
        print('outputs/numbers.csv does not exist yet (run_module writes the other rows of this module)')
    ledger_order = list(numbers.ledger)

    def ledger_position(row):
        if row['ledger_id'] in numbers.ledger:
            return ledger_order.index(row['ledger_id'])
        return len(ledger_order)

    numbers.rows = sorted(other_rows + numbers.rows, key=ledger_position)
    numbers.write(outputs_folder)


def main():
    paths = release_paths.module_paths(__file__)
    paper_run_file = os.path.join(paths['paper_run'], 'fig2_paper_run.mat')
    paper_run = load_run(paper_run_file)
    number_of_targets = paper_run['number_of_targets']
    print('Fig. 2 statistics from the saved run in data/paper_run/ (seed %d, %d runs per context): %d targets'
          % (paper_run['seed'], paper_run['runs_per_context'], number_of_targets))

    # Step 1: reproduce the correlations the saved run stores (MATLAB corr) from the per-target values,
    # with the two-sided p of each (through logarithms when scipy's p is 0, see two_sided_p)
    observed = {}
    for entry in CORRELATIONS:
        x_values = paper_run[entry['x']]
        y_values = paper_run[entry['y']]
        value = correlation_value(x_values, y_values, entry['statistic_name'])
        scipy_p = correlation_p(x_values, y_values, entry['statistic_name'])
        log10_p = log10_two_sided_p(value, number_of_targets)
        if scipy_p > 0:
            # check of the logarithm route against scipy wherever scipy's p can be stored
            difference_in_log10 = abs(log10_p - math.log10(scipy_p))
            print('  %-55s p = %.6g (scipy); log10 p = %.6f, through logarithms %.6f'
                  % (entry['name'], scipy_p, math.log10(scipy_p), log10_p))
            if difference_in_log10 > 1e-6:
                raise RuntimeError('the logarithm route for p disagrees with scipy: %s' % entry['name'])
        p_for_table, p_method = two_sided_p(value, scipy_p, number_of_targets)
        if p_method == 'log space':
            print('  %-55s p = %s (scipy returns 0; computed through logarithms, log10 p = %.4f)'
                  % (entry['name'], p_for_table, log10_p))
        observed[entry['name']] = (value, p_for_table, p_method, log10_p)
        if entry['name'] in paper_run['saved']:
            saved_value = paper_run['saved'][entry['name']]
            difference = abs(value - saved_value)
            print('  %-55s %s = %.10f (saved by the simulation: %.10f; difference %.1e)'
                  % (entry['name'], entry['statistic_name'], value, saved_value, difference))
            if difference > 1e-12:
                raise RuntimeError('the recomputed correlation differs from the saved one: %s' % entry['name'])
        else:
            print('  %-55s %s = %.10f (not saved by the simulation)' % (entry['name'], entry['statistic_name'], value))

    # Step 2: percentile bootstrap over targets, the same 10,000 resamples for all four correlations
    def correlations_of_resample(target_indices):
        return all_correlations(paper_run, target_indices)

    resampled = bootstrap.percentile_bootstrap(correlations_of_resample, number_of_targets,
                                               n_resamples=NUMBER_OF_RESAMPLES, seed=SEED)
    interval_low, interval_high = bootstrap.percentile_interval(resampled)

    # Step 3: the rows of tests.csv
    tests = tests_table.TestsTable(script=SCRIPT)
    for column, entry in enumerate(CORRELATIONS):
        value, p_for_table, p_method, log10_p = observed[entry['name']]
        test_text = entry['test']
        if p_method == 'log space':
            test_text = (test_text + '; p is below the smallest double-precision number (scipy and MATLAB '
                         'return 0) and is computed through its logarithm, log10 p = %.2f' % log10_p)
        common = dict(test_id=entry['test_id'], ledger_ids=entry['ledger_ids'], location=entry['location'],
                      test=test_text, statistic_name=entry['statistic_name'], statistic=value,
                      n=number_of_targets, unit='simulated targets', df=number_of_targets - 2, p=p_for_table,
                      p_type='parametric two-sided',
                      effect_size_name='rho' if entry['statistic_name'] == 'Spearman rho' else 'r',
                      effect_size=value)
        tests.add(ci_low=float(interval_low[column]), ci_high=float(interval_high[column]),
                  ci_method='percentile bootstrap', resampling_unit='targets (simulated items)',
                  n_resamples=NUMBER_OF_RESAMPLES, seed=SEED, **common)
    tests.write(paths['outputs'])
    print('wrote outputs/tests.csv')

    # Step 4: the numbers.csv rows of the four correlations: rho, its 95% interval and the two-sided p
    numbers = numbers_table.NumbersTable(script=SCRIPT)
    for column, entry in enumerate(CORRELATIONS):
        value, p_for_table, p_method, log10_p = observed[entry['name']]
        computed = [value, float(interval_low[column]), float(interval_high[column]), p_for_table]
        note = ('%s over the %d targets of the saved run in data/paper_run/ (run_module(\'full\') reproduces it '
                'in every field), its 95%% percentile bootstrap interval over targets (%s resamples, seed %d) and '
                'the two-sided p; test %s in outputs/tests.csv'
                % (entry['statistic_name'], number_of_targets, format(NUMBER_OF_RESAMPLES, ','), SEED,
                   entry['test_id']))
        for ledger_id in entry['numbers_ids']:
            match = numbers.add(ledger_id, computed_value=computed, mode='saved_run', note=note)
            print('  %s: %s (paper %s): %s' % (ledger_id, numbers_table.format_computed(computed),
                                              numbers.ledger[ledger_id]['paper_value'], match))
    add_rows_to_numbers_csv(numbers, paths['outputs'])


if __name__ == '__main__':
    main()
