"""
stats_figS3.py: the statistics of the 18 simulations of code/bylinskii_robustness.m (test T26 of
statistics/README.md), written to outputs/tests.csv.

What it computes
----------------
T26 (SI Appendix, Section 2, Category Structure and Decision Criteria; SI Appendix, Fig. S3)
    For each of the 18 simulations (cells) of the grid:
    1. the Spearman correlation, across the target items, between each target's memorability (hit rate
       minus false alarm rate) in the within-category context and in the across-category context, with
       n, its two-sided p (t approximation) and a 95% interval from a bootstrap over the target items;
    2. the Pearson correlation, across the same targets, between each target's summed similarity within
       its category (minus the mean Euclidean distance to the other items of its category) and across the
       whole set (minus the mean Euclidean distance to all items), with the bootstrap interval.
    Then the range and the sign count over the 18 cells, the same for the four cells with equal criteria
    and for the summed-similarity correlation.

The grid (code/bylinskii_robustness.m)
    REM with g = 0.40, copy fidelity 0.70, storage 0.30 and 49 features; foils are targets not studied in
    the list, probed against the list; 250 within-category lists per category and 5,000 across-category
    lists.

How the intervals are made (statistics/README.md)
    Item bootstrap: the target items of a cell are resampled with replacement and the correlation is
    recomputed, 10,000 times; the 95% interval is the 2.5 and 97.5 percentiles (shared/python/bootstrap.py,
    percentile_bootstrap and percentile_interval). Every cell uses the same seed, 20261013, so each cell's
    interval can be reproduced on its own; within a cell the memorability and the summed-similarity
    bootstraps draw the same resamples. Each interval resamples the target items of one simulated item set
    (the cell's category structure stays fixed).
    The p of a Spearman rho is the t approximation with n - 2 degrees of freedom (scipy's spearmanr; the
    same formula as MATLAB's corr in the pRho column of the grid table). A p too small to store as a number
    (below about 1e-308; the summed-similarity correlations of 16 cells) is computed through logarithms with
    the formula of module 02 and written as text (p_for_table).
    The 18 cells are a sweep, reported as ranges and sign counts; no multiplicity correction is applied.

Inputs
    outputs/bylinskii_robustness.csv and outputs/bylinskii_robustness_items.csv (written by
        bylinskii_robustness.m; the per-target file holds every cell's per-target trial counts and summed
        similarities)
    (the tables are read with code/robustness_tables.py)
Output
    outputs/tests.csv (columns of statistics/README.md)
Seed    20261013 (numpy.random.default_rng), printed
Runtime about 3 minutes on an Apple M2 Pro (36 bootstraps of 10,000 correlations over 750 to 3,150 targets)

Run (from the repository root):
    python A2b_category_structure_criteria_figS3/code/stats_figS3.py
"""
import math
import os
import sys
import time

import numpy as np
from scipy import stats
from scipy.special import betaln, hyp2f1

sys.dont_write_bytecode = True   # write no __pycache__ folder into shared/python (set before the import below)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402
import bootstrap  # noqa: E402
import tests_table  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robustness_tables  # noqa: E402  (this module's readers of the grid's tables)

SEED = 20261013
N_RESAMPLES = 10000
SCRIPT = 'code/stats_figS3.py'
GRID_DESCRIPTION = ('The grid of code/bylinskii_robustness.m, rng(11, twister): REM with g 0.40, copy fidelity '
                    '0.70, storage 0.30, 49 features; foils are targets not studied in the list, probed against the '
                    'list')

paths = release_paths.module_paths(__file__)
print('stats_figS3.py: item bootstraps with %d resamples, seed %d' % (N_RESAMPLES, SEED))


# ============================================================================ helpers
def item_bootstrap_interval(x_values, y_values, correlation):
    """95% percentile interval of a correlation from 10,000 resamples of the target items (seed 20261013)."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    def statistic(indices):
        if correlation == 'spearman':
            return stats.spearmanr(x_values[indices], y_values[indices]).statistic
        return stats.pearsonr(x_values[indices], y_values[indices]).statistic

    resampled = bootstrap.percentile_bootstrap(statistic, len(x_values), N_RESAMPLES, SEED)
    low, high = bootstrap.percentile_interval(resampled)
    return float(low), float(high)


def log10_two_sided_p(correlation, number_of_pairs):
    """log10 of the two-sided p of a correlation (t = r sqrt(df / (1 - r^2)), df = n - 2), computed through
    logarithms so that it also works when p itself is too small to store (below about 1e-308). The same
    formula as module 02 (02_bylinskii_simulation_fig2/code/stats_fig2.py): the two-sided p of a t statistic
    is the regularized incomplete beta function I_x(a, b) with x = df / (df + t^2), a = df / 2, b = 1 / 2, and
        I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) * F(a + b, 1; a + 1; x)
    (F = the Gauss hypergeometric function; Digital Library of Mathematical Functions 8.17.8)."""
    degrees_of_freedom = number_of_pairs - 2
    t_squared = correlation ** 2 * degrees_of_freedom / (1.0 - correlation ** 2)
    x = degrees_of_freedom / (degrees_of_freedom + t_squared)
    a = degrees_of_freedom / 2.0
    b = 0.5
    log_p = (a * math.log(x) + b * math.log1p(-x) - math.log(a) - betaln(a, b)
             + math.log(hyp2f1(a + b, 1.0, a + 1.0, x)))
    return log_p / math.log(10.0)


def p_for_table(correlation, scipy_p, number_of_pairs):
    """The p written to tests.csv: scipy's p when it can be stored; when scipy returns 0 (the p is below the
    smallest positive double), the same p from log10_two_sided_p, written as text with 6 significant digits
    (for example '2.17879e-687', the reference cell), as module 02 writes its p values."""
    if scipy_p > 0:
        return scipy_p
    log10_p = log10_two_sided_p(correlation, number_of_pairs)
    exponent = math.floor(log10_p)
    mantissa = 10.0 ** (log10_p - exponent)
    if mantissa >= 9.999995:          # rounding would print 10.00000
        mantissa = mantissa / 10.0
        exponent = exponent + 1
    return '%.5fe%d' % (mantissa, exponent)


def describe_cell(cell_number, cell):
    """One line naming a cell and its settings."""
    return ('cell %d "%s" (criteria %g within / %g across, %d category-defining features, %d categories of %d '
            'items, %d targets per category)'
            % (cell_number, cell['label'], cell['critWithin'], cell['critAcross'], cell['catFeatures'],
               cell['nCat'], cell['nPerCat'], robustness_tables.targets_per_category(cell)))


# ============================================================================ read the 18 cells
table_file = os.path.join(paths['outputs'], 'bylinskii_robustness.csv')
items_file = os.path.join(paths['outputs'], 'bylinskii_robustness_items.csv')
for needed_file in (table_file, items_file):
    if not os.path.exists(needed_file):
        sys.exit('stats_figS3.py: outputs/%s is missing; run code/bylinskii_robustness.m first '
                 '(python A2b_category_structure_criteria_figS3/code/run_module.py does it)'
                 % os.path.basename(needed_file))

cells = robustness_tables.read_cells(table_file)
assert len(cells) == 18, 'expected 18 cells, found %d' % len(cells)
items_by_cell = robustness_tables.read_items(items_file)
assert sorted(items_by_cell) == list(range(1, 19)), 'the per-target file must hold cells 1 to 18'

tests = tests_table.TestsTable(script=SCRIPT)

# ============================================================================ every cell
print('\nT26: the 18 cells (target bootstrap per cell)')
cell_results = []
start_time = time.time()
for cell_index, cell in enumerate(cells):
    cell_number = cell_index + 1
    items = items_by_cell[cell_number]

    # The number of targets: categories x targets per category, all of them tested in both contexts.
    expected_targets = int(cell['nCat']) * robustness_tables.targets_per_category(cell)
    assert items['n_targets'] == expected_targets, 'cell %d: %d targets, expected %d' % (
        cell_number, items['n_targets'], expected_targets)
    n_targets_used = len(items['memorability_within'])

    # 1. memorability within vs across (Spearman); must give back the rho of the MATLAB table
    rho, rho_p = robustness_tables.spearman_rho_and_p(items['memorability_within'], items['memorability_across'])
    assert abs(rho - cell['rho']) < 1e-12, 'cell %d: rho %.15f vs table %.15f' % (cell_number, rho, cell['rho'])
    assert abs(rho_p / cell['pRho'] - 1) < 1e-6, 'cell %d: p %.6g vs table %.6g' % (cell_number, rho_p, cell['pRho'])
    rho_low, rho_high = item_bootstrap_interval(items['memorability_within'], items['memorability_across'],
                                                'spearman')

    # 2. summed similarity within vs across (Pearson); must give back the simCorr of the MATLAB table
    similarity_r, similarity_p = robustness_tables.pearson_r_and_p(items['summed_similarity_within'],
                                                                   items['summed_similarity_across'])
    assert abs(similarity_r - cell['simCorr']) < 1e-12, 'cell %d: simCorr %.15f vs table %.15f' % (
        cell_number, similarity_r, cell['simCorr'])
    similarity_low, similarity_high = item_bootstrap_interval(items['summed_similarity_within'],
                                                              items['summed_similarity_across'], 'pearson')

    cell_results.append({'cell': cell, 'cell_number': cell_number, 'n': n_targets_used,
                         'rho': rho, 'rho_p': rho_p, 'rho_low': rho_low, 'rho_high': rho_high,
                         'similarity_r': similarity_r, 'similarity_p': similarity_p,
                         'similarity_low': similarity_low, 'similarity_high': similarity_high})
    print('  %2d %-32s n %4d  rho %+.4f [%+.4f, %+.4f] p %.2g  |  r(summed sim) %+.4f [%+.4f, %+.4f]  (%.0f s)'
          % (cell_number, cell['label'], n_targets_used, rho, rho_low, rho_high, rho_p, similarity_r,
             similarity_low, similarity_high, time.time() - start_time), flush=True)

# ============================================================================ T26 rows, one per cell
for result in cell_results:
    cell = result['cell']
    ledger_ids = ['L6-038', 'L6-046']
    if robustness_tables.has_equal_criteria(cell):
        ledger_ids += ['L6-039']
    if robustness_tables.has_fig2a_structure(cell):
        ledger_ids += ['L6-043']
    tests.add(test_id='T26', ledger_ids=';'.join(ledger_ids),
              location='SI Appendix, Section 2, Category Structure and Decision Criteria; SI Appendix, Fig. S3',
              test='Bylinskii et al. (2015) context manipulation, %s: Spearman correlation across the target items '
                   'between memorability (hit rate minus false alarm rate) in the within-category context and in '
                   'the across-category context. %s' % (describe_cell(result['cell_number'], cell), GRID_DESCRIPTION),
              statistic_name='Spearman rho', statistic=result['rho'], n=result['n'], unit='simulated target items',
              df=result['n'] - 2, p=result['rho_p'], p_type='parametric two-sided', effect_size_name='rho',
              effect_size=result['rho'], ci_low=result['rho_low'], ci_high=result['rho_high'],
              ci_method='percentile bootstrap', resampling_unit='simulated target items',
              n_resamples=N_RESAMPLES, seed=SEED)

for result in cell_results:
    cell = result['cell']
    similarity_test = ('Bylinskii et al. (2015) context manipulation, %s: Pearson correlation across the target '
                       'items between summed similarity within the category (minus the mean Euclidean distance to '
                       'the other items of the category) and across the whole set (minus the mean Euclidean '
                       'distance to all items)'
                       % describe_cell(result['cell_number'], cell))
    # A p below the smallest positive number a double can hold (about 1e-308) comes back from scipy as 0;
    # p_for_table then computes the same p through logarithms and writes it as text.
    similarity_p_for_table = p_for_table(result['similarity_r'], result['similarity_p'], result['n'])
    if result['similarity_p'] == 0:
        similarity_test += ('; its p is below the smallest positive double (about 1e-308), so it is computed '
                            'through logarithms (the regularized incomplete beta function, as in module 02) '
                            'and written as text')
    tests.add(test_id='T26', ledger_ids='L6-040',
              location='SI Appendix, Section 2, Category Structure and Decision Criteria',
              test=similarity_test + '; percentile bootstrap interval over the targets',
              statistic_name='Pearson r', statistic=result['similarity_r'], n=result['n'],
              unit='simulated target items', df=result['n'] - 2, p=similarity_p_for_table,
              p_type='parametric two-sided', effect_size_name='r', effect_size=result['similarity_r'],
              ci_low=result['similarity_low'], ci_high=result['similarity_high'], ci_method='percentile bootstrap',
              resampling_unit='simulated target items', n_resamples=N_RESAMPLES, seed=SEED)

# ============================================================================ T26 summary rows
grid_seed = 'MATLAB rng(11, twister)'

# all 18 cells
all_rho = [result['rho'] for result in cell_results]
largest_rho_p = max(result['rho_p'] for result in cell_results)
number_positive = sum(1 for rho in all_rho if rho > 0)
lowest_cell = min(cell_results, key=lambda result: result['rho'])
highest_cell = max(cell_results, key=lambda result: result['rho'])
tests.add(test_id='T26', ledger_ids='L6-036;L6-038;L6-042;L6-046',
          location='SI Appendix, Section 2, Category Structure and Decision Criteria',
          test='Bylinskii et al. (2015) context manipulation, all 18 cells: number of cells whose Spearman rho '
               '(memorability within vs across category) is above zero, with the smallest (%s) and largest (%s) '
               'rho and the largest p (no multiplicity correction). %s'
               % (lowest_cell['cell']['label'], highest_cell['cell']['label'], GRID_DESCRIPTION),
          statistic_name='cells with rho above zero', statistic=number_positive, n=len(all_rho), unit='cells',
          df='', p=largest_rho_p, p_type='parametric two-sided', effect_size_name='rho, smallest to largest',
          effect_size='', ci_low=min(all_rho), ci_high=max(all_rho),
          ci_method='range over the 18 cells (not a confidence interval)', resampling_unit='', n_resamples='',
          seed=grid_seed)

# the four cells with equal criteria
equal_results = [result for result in cell_results if robustness_tables.has_equal_criteria(result['cell'])]
equal_rho = [result['rho'] for result in equal_results]
equal_positive = sum(1 for rho in equal_rho if rho > 0)
tests.add(test_id='T26', ledger_ids='L6-039',
          location='SI Appendix, Section 2, Category Structure and Decision Criteria',
          test='Bylinskii et al. (2015) context manipulation, the cells with the same criterion in both contexts '
               '(%s): number of cells whose Spearman rho is above zero, with the smallest and largest rho and the '
               'largest p'
               % ', '.join('%g / %g' % (result['cell']['critWithin'], result['cell']['critAcross'])
                           for result in equal_results),
          statistic_name='cells with rho above zero', statistic=equal_positive, n=len(equal_results), unit='cells',
          df='', p=max(result['rho_p'] for result in equal_results), p_type='parametric two-sided',
          effect_size_name='rho, smallest to largest', effect_size='', ci_low=min(equal_rho), ci_high=max(equal_rho),
          ci_method='range over the 4 cells (not a confidence interval)', resampling_unit='', n_resamples='',
          seed=grid_seed)

# summed similarity within vs across, all 18 cells
all_similarity_r = [result['similarity_r'] for result in cell_results]
similarity_positive = sum(1 for r_value in all_similarity_r if r_value > 0)
lowest_similarity = min(cell_results, key=lambda result: result['similarity_r'])
highest_similarity = max(cell_results, key=lambda result: result['similarity_r'])
tests.add(test_id='T26', ledger_ids='L6-040',
          location='SI Appendix, Section 2, Category Structure and Decision Criteria',
          test='Bylinskii et al. (2015) context manipulation, all 18 cells: number of cells whose Pearson r between '
               'summed similarity within the category and across the whole set is above zero, with the smallest '
               '(%s) and largest (%s) r and the largest p'
               % (lowest_similarity['cell']['label'], highest_similarity['cell']['label']),
          statistic_name='cells with r above zero', statistic=similarity_positive, n=len(all_similarity_r),
          unit='cells', df='', p=max(result['similarity_p'] for result in cell_results),
          p_type='parametric two-sided', effect_size_name='r, smallest to largest', effect_size='',
          ci_low=min(all_similarity_r), ci_high=max(all_similarity_r),
          ci_method='range over the 18 cells (not a confidence interval)', resampling_unit='', n_resamples='',
          seed=grid_seed)

tests.write(paths['outputs'])
print('\nall 18 cells: rho %.4f to %.4f, %d of 18 above zero, largest p %.2g'
      % (min(all_rho), max(all_rho), number_positive, largest_rho_p))
print('equal criteria: rho %.4f to %.4f (%d cells)' % (min(equal_rho), max(equal_rho), len(equal_rho)))
print('summed similarity within vs across: r %.4f to %.4f' % (min(all_similarity_r), max(all_similarity_r)))
