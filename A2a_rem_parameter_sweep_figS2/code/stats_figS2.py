"""
stats_figS2.py: the uncertainty statistics of the REM parameter sweeps (tests T20, T24 and T25 of
statistics/README.md), written to outputs/tests.csv.

What it computes
----------------
T20 (Methods, Basic REM Simulation of Memorability: "rho = -0.52, 95% CI [-0.56, -0.49], rather than
-0.55"; Fig. 1b)
    1. the sweep's Shiffrin and Steyvers cell (rem_parameter_sweep.m): Spearman rho across the 1,600
       simulated items between memorability (hit rate minus false alarm rate) and the negated mean
       Euclidean distance to the other items (the -0.52), and the spread of the six cells that run
       these values (Monte Carlo spread; the "-0.55 to -0.51" of SI Appendix, Section 2);
    2. the saved Fig. 1 run (module 01): the Pearson r between memorability and 1 / mean distance
       (Fig. 1b, -0.56) with its bootstrap interval, and the Spearman rho with the negated mean
       distance (the -0.55 of the Methods).
T24 (SI Appendix, Section 2, REM Parameters: "rho = -0.65 to -0.28, all p < 0.001; Fig. S2a), including
at the standard values (rho = -0.52")
    One row per cell of the 32-cell sweep: Spearman rho, n, p and a 95% interval from a bootstrap
    over the simulated items (the cell with the Fig. 1 values gives the -0.61 of SI Appendix,
    Section 2, A Hybrid-Similarity Exemplar Model). Then the range and sign count over the 32 cells,
    and the feature-count sweep (5 to 120 features, 3 seeds each; "rho = -0.66 to -0.35"): the range
    and sign count over its 54 runs.
T25 (SI Appendix, Section 2, REM Parameters: "r = 0.71, 95% CI [0.49, 0.85], n = 32; Fig. S2b")
    Pearson r between each cell's overall d′ and its rho over the 32 cells, with its p and the
    Fisher z interval.

How the intervals are made (statistics/README.md)
    Item bootstrap: the 1,600 simulated items of a cell are resampled with replacement and the
    correlation is recomputed, 10,000 times; the 95% interval is the 2.5 and 97.5 percentiles
    (shared/python/bootstrap.py, percentile_bootstrap and percentile_interval). Every cell uses the
    same seed, 20261013, so each cell's interval can be reproduced on its own.
    The p of a Spearman rho is the t approximation with n - 2 degrees of freedom (scipy's spearmanr;
    the same formula as the pItem column of rem_parameter_sweep.csv).
    The 32 cells, the 18 feature counts and the 3 seeds are a sweep, reported as ranges and sign
    counts; no multiplicity correction is applied.

Inputs
    outputs/rem_parameter_sweep.csv and outputs/rem_parameter_sweep_items.csv (this run of
        rem_parameter_sweep.m; the per-item file holds every cell's per-item trial counts and
        distances)
    outputs/sweep_rem.csv (this run of feature_count_sweep.py) when it is complete;
        otherwise data/paper_run/sweep_rem.csv (the saved run), and the script says so
    ../01_rem_simulation_fig1/data/paper_run/fig1_paper_run.mat
        (the saved Fig. 1 run, module 01's data) for the two Fig. 1 rows of T20; without it those rows
        are left out with a message
    (the sweep's tables are read with code/sweep_tables.py)
Output
    outputs/tests.csv (columns of statistics/README.md)
Seed    20261013 (numpy.random.default_rng), printed
Runtime about 3 to 4 minutes (34 bootstraps of 10,000 correlations over 1,600 items: one per cell and
        two for the Fig. 1 run)

Run (from the repository root):
    python A2a_rem_parameter_sweep_figS2/code/stats_figS2.py
"""
import csv
import os
import sys
import time

import numpy as np
import scipy.io
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402
import bootstrap  # noqa: E402
import tests_table  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_tables  # noqa: E402  (this module's readers of the sweep's tables)

SEED = 20261013
N_RESAMPLES = 10000
SCRIPT = 'code/stats_figS2.py'
FIG1_RUN_FILE = 'fig1_paper_run.mat'
PAPER_FEATURE_COUNTS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 49, 55, 60, 66, 70, 80, 90, 100, 120]
PAPER_SEEDS = [12345, 777, 20260731]

paths = release_paths.module_paths(__file__)
print('stats_figS2.py: item bootstraps with %d resamples, seed %d' % (N_RESAMPLES, SEED))


# ============================================================================ helpers
def spearman_with_p(x_values, y_values):
    """Spearman rho, its two-sided p (t approximation, n - 2 df) and n."""
    result = stats.spearmanr(x_values, y_values)
    return float(result.statistic), float(result.pvalue), len(x_values)


def item_bootstrap_interval(x_values, y_values, correlation='spearman'):
    """95% percentile interval of a correlation from 10,000 resamples of the items (seed 20261013)."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    def statistic(indices):
        if correlation == 'spearman':
            return stats.spearmanr(x_values[indices], y_values[indices]).statistic
        return stats.pearsonr(x_values[indices], y_values[indices]).statistic

    resampled = bootstrap.percentile_bootstrap(statistic, len(x_values), N_RESAMPLES, SEED)
    low, high = bootstrap.percentile_interval(resampled)
    return float(low), float(high)


# ============================================================================ read the 32 cells
sweep_file = os.path.join(paths['outputs'], 'rem_parameter_sweep.csv')
items_file = os.path.join(paths['outputs'], 'rem_parameter_sweep_items.csv')
for needed_file in (sweep_file, items_file):
    if not os.path.exists(needed_file):
        sys.exit('stats_figS2.py: outputs/%s is missing; run code/rem_parameter_sweep.m first '
                 '(python A2a_rem_parameter_sweep_figS2/code/run_module.py does it)' % os.path.basename(needed_file))

cells = sweep_tables.read_cells(sweep_file)
assert len(cells) == 32, 'expected 32 cells, found %d' % len(cells)
# Per-item memorability and negated mean distance (summed similarity) of every cell (cell number = row of
# the csv, from 1).
items_by_cell = sweep_tables.read_items(items_file)


tests = tests_table.TestsTable(script=SCRIPT)

# ============================================================================ T24: every cell
print('\nT24: the 32 cells (item bootstrap per cell)')
cell_results = []
start_time = time.time()
for cell_index, cell in enumerate(cells):
    cell_number = cell_index + 1
    memorability = items_by_cell[cell_number]['memorability']
    negated_mean_distance = items_by_cell[cell_number]['negated_mean_distance']
    rho, p_value = sweep_tables.spearman_rho_and_p(memorability, negated_mean_distance)
    n_items = len(memorability)
    # The per-item file must give back the rho of the MATLAB table.
    assert abs(rho - cell['rhoItem']) < 1e-12, 'cell %d: rho %.15f vs table %.15f' % (cell_number, rho, cell['rhoItem'])
    ci_low, ci_high = item_bootstrap_interval(memorability, negated_mean_distance)
    cell_results.append({'cell': cell, 'rho': rho, 'p': p_value, 'n': n_items, 'ci_low': ci_low, 'ci_high': ci_high})
    print('  %2d %-34s rho %+.4f  p %.3g  n %d  95%% CI [%+.4f, %+.4f]  (%.0f s)'
          % (cell_number, cell['label'], rho, p_value, n_items, ci_low, ci_high, time.time() - start_time),
          flush=True)

reference_result = cell_results[0]
assert reference_result['cell']['label'] == 'Shiffrin & Steyvers reference'

# The six cells with the standard values (the reference cell, and the standard point of the g x w grid and of
# each one-at-a-time group). They differ only in their random draws, so their spread is the Monte Carlo spread
# of the -0.52.
standard_cells = sweep_tables.standard_value_cells(cells)
standard_rho = []
for result in cell_results:
    for standard_cell in standard_cells:
        if result['cell'] is standard_cell:
            standard_rho.append(result['rho'])
assert len(standard_rho) == 6

# ============================================================================ T20
print('\nT20: the sweep\'s standard cell and the saved Fig. 1 run')
tests.add(test_id='T20', ledger_ids='L5-006;L6-022', location='Methods, Basic REM Simulation of Memorability',
          test='Spearman correlation across the simulated items of the REM parameter sweep at the Shiffrin '
               'and Steyvers values (g 0.40, 20 features, storage 0.30, copy 0.70, list 40, criterion 1.0): '
               'memorability (hit rate minus false alarm rate) vs negated mean Euclidean distance to the '
               'other items (rem_parameter_sweep.m, rng(1, \'twister\'))',
          statistic_name='Spearman rho', statistic=reference_result['rho'], n=reference_result['n'],
          unit='simulated items', df=reference_result['n'] - 2, p=reference_result['p'],
          p_type='parametric two-sided', effect_size_name='rho', effect_size=reference_result['rho'],
          ci_low=reference_result['ci_low'], ci_high=reference_result['ci_high'], ci_method='percentile bootstrap',
          resampling_unit='simulated items', n_resamples=N_RESAMPLES, seed=SEED)
tests.add(test_id='T20', ledger_ids='L5-006;L6-034', location='SI Appendix, Section 2, REM Parameters',
          test='Monte Carlo spread of the same Spearman correlation at the Shiffrin and Steyvers values: the six '
               'cells of the sweep with these values (the reference cell, and the standard point of the g x w grid '
               'and of each one-at-a-time group), which differ only in their random draws; mean and range over '
               'the six runs',
          statistic_name='Spearman rho, mean of 6 runs', statistic=float(np.mean(standard_rho)), n=6, unit='runs',
          df='', p='', p_type='none', effect_size_name='rho', effect_size=float(np.mean(standard_rho)),
          ci_low=min(standard_rho), ci_high=max(standard_rho),
          ci_method='range over 6 runs with identical parameters (Monte Carlo spread, not a confidence interval)',
          resampling_unit='', n_resamples='', seed='MATLAB rng(1, twister)')

fig1_file = os.path.join(release_paths.find_module('01'), 'data', 'paper_run', FIG1_RUN_FILE)
if os.path.exists(fig1_file):
    figure_data = scipy.io.loadmat(fig1_file, squeeze_me=True)['figureData']
    fig1_memorability = np.asarray(figure_data['hitFARate'].item(), dtype=float)
    fig1_mean_distance = np.asarray(figure_data['avgDistances'].item(), dtype=float)
    fig1_similarity = np.asarray(figure_data['similarity'].item(), dtype=float)
    fig1_saved_r = float(figure_data['r_sim'].item())
    has_score = np.isfinite(fig1_memorability)
    fig1_memorability = fig1_memorability[has_score]
    fig1_negated_distance = -fig1_mean_distance[has_score]
    fig1_similarity = fig1_similarity[has_score]

    # Fig. 1b: Pearson r of memorability with similarity = 1 / mean distance (-0.56).
    pearson_result = stats.pearsonr(fig1_similarity, fig1_memorability)
    fig1_r = float(pearson_result.statistic)
    assert abs(fig1_r - fig1_saved_r) < 1e-12, 'Fig. 1 r %.15f vs saved %.15f' % (fig1_r, fig1_saved_r)
    fig1_r_low, fig1_r_high = item_bootstrap_interval(fig1_similarity, fig1_memorability, correlation='pearson')
    # The sweep's measure (Spearman with the negated mean distance) on the Fig. 1 run: the -0.55 of the Methods.
    fig1_rho, fig1_rho_p, fig1_n = spearman_with_p(fig1_memorability, fig1_negated_distance)
    fig1_rho_low, fig1_rho_high = item_bootstrap_interval(fig1_memorability, fig1_negated_distance)
    print('  sweep standard cell (Spearman, negated mean distance): %+.4f [%+.4f, %+.4f]'
          % (reference_result['rho'], reference_result['ci_low'], reference_result['ci_high']))
    print('  Fig. 1 run, Pearson with 1 / mean distance (Fig. 1b): %+.4f [%+.4f, %+.4f]'
          % (fig1_r, fig1_r_low, fig1_r_high))
    print('  Fig. 1 run, Spearman with the negated mean distance (the sweep\'s measure; the -0.55 of the Methods): '
          '%+.4f [%+.4f, %+.4f]' % (fig1_rho, fig1_rho_low, fig1_rho_high))

    fig1_test_text = ('saved Fig. 1 run (module 01, fig1_remSim.m, seed 12345; g 0.65, 49 '
                      'features, storage 0.30, copy 0.60, criterion 1): ')
    tests.add(test_id='T20', ledger_ids='L1-019;L1-022', location='Fig. 1 caption (panel b)',
              test=fig1_test_text + 'Pearson correlation across the simulated items between memorability and '
                   'summed similarity defined as 1 / mean Euclidean distance (Fig. 1b)',
              statistic_name='Pearson r', statistic=fig1_r, n=len(fig1_memorability), unit='simulated items',
              df=len(fig1_memorability) - 2, p=float(pearson_result.pvalue), p_type='parametric two-sided',
              effect_size_name='r', effect_size=fig1_r, ci_low=fig1_r_low, ci_high=fig1_r_high,
              ci_method='percentile bootstrap', resampling_unit='simulated items', n_resamples=N_RESAMPLES,
              seed=SEED)
    tests.add(test_id='T20', ledger_ids='L5-006;L5-007', location='Methods, Basic REM Simulation of Memorability',
              test=fig1_test_text + 'Spearman correlation between memorability and negated mean Euclidean '
                   'distance, the measure of the sweep\'s -0.52 (the -0.55 of the Methods)',
              statistic_name='Spearman rho', statistic=fig1_rho, n=fig1_n, unit='simulated items', df=fig1_n - 2,
              p=fig1_rho_p, p_type='parametric two-sided', effect_size_name='rho', effect_size=fig1_rho,
              ci_low=fig1_rho_low, ci_high=fig1_rho_high, ci_method='percentile bootstrap',
              resampling_unit='simulated items', n_resamples=N_RESAMPLES, seed=SEED)
else:
    print('  NOT COMPUTED: the saved Fig. 1 run is not at 01_rem_simulation_fig1/data/paper_run/%s '
          '(module 01\'s data); the two Fig. 1 rows of T20 are left out of tests.csv' % FIG1_RUN_FILE)

# ============================================================================ T24 rows
for cell_index, result in enumerate(cell_results):
    cell = result['cell']
    is_reference = cell['label'] == 'Shiffrin & Steyvers reference'
    tests.add(test_id='T24', ledger_ids='L6-021;L6-033' + (';L6-022' if is_reference else ''),
              location=('SI Appendix, Section 2, REM Parameters and A Hybrid-Similarity Exemplar Model'
                        if cell['label'].startswith('paper values') else 'SI Appendix, Section 2, REM Parameters'),
              test='REM parameter sweep cell %d "%s" (g %.2f, %d features, storage %.2f, copy %.2f, list %d, '
                   'criterion %.1f): Spearman correlation across the simulated items between memorability and '
                   'negated mean Euclidean distance' % (cell_index + 1, cell['label'], cell['g'], cell['w'],
                                                       cell['u'], cell['c'], cell['nStudy'], cell['crit']),
              statistic_name='Spearman rho', statistic=result['rho'], n=result['n'], unit='simulated items',
              df=result['n'] - 2, p=result['p'], p_type='parametric two-sided', effect_size_name='rho',
              effect_size=result['rho'], ci_low=result['ci_low'], ci_high=result['ci_high'],
              ci_method='percentile bootstrap', resampling_unit='simulated items', n_resamples=N_RESAMPLES,
              seed=SEED)

all_rho = [result['rho'] for result in cell_results]
largest_p = max(result['p'] for result in cell_results)
number_negative = sum(1 for rho in all_rho if rho < 0)
tests.add(test_id='T24', ledger_ids='L1-010;L5-008;L6-021;L6-033', location='SI Appendix, Section 2, REM Parameters',
          test='REM parameter sweep, all 32 cells: number of cells whose Spearman rho is below zero, with the '
               'smallest and largest rho and the largest p (no multiplicity correction)',
          statistic_name='cells with rho below zero', statistic=number_negative, n=len(all_rho), unit='cells',
          df='', p=largest_p, p_type='parametric two-sided', effect_size_name='rho, smallest to largest',
          effect_size='', ci_low=min(all_rho), ci_high=max(all_rho),
          ci_method='range over the 32 cells (not a confidence interval)', resampling_unit='', n_resamples='',
          seed='MATLAB rng(1, twister)')

# ============================================================================ T24: feature-count sweep
fresh_feature_file = os.path.join(paths['outputs'], 'sweep_rem.csv')
paper_feature_file = os.path.join(paths['paper_run'], 'sweep_rem.csv')


def read_feature_sweep(path):
    """The runs of a feature-count table: feature count, seed, Spearman rho and its p."""
    rows = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            rows.append({'w': int(row['w']), 'seed': int(row['seed']), 'rho': float(row['r_sim_spearman']),
                         'p': float(row['p_sim_spearman'])})
    return rows


feature_rows = None
feature_source = None
if os.path.exists(fresh_feature_file):
    candidate_rows = read_feature_sweep(fresh_feature_file)
    runs_present = sorted((row['w'], row['seed']) for row in candidate_rows)
    runs_of_paper = sorted((count, seed) for count in PAPER_FEATURE_COUNTS for seed in PAPER_SEEDS)
    if runs_present == runs_of_paper:
        feature_rows = candidate_rows
        feature_source = 'outputs/sweep_rem.csv (this run)'
    else:
        print('\noutputs/sweep_rem.csv does not hold the paper\'s 54 runs (a test run or an unfinished run)')
if feature_rows is None:
    feature_rows = read_feature_sweep(paper_feature_file)
    feature_source = 'data/paper_run/sweep_rem.csv (the saved run; feature_count_sweep.py was not run)'
print('\nT24: feature-count sweep from %s' % feature_source)

feature_rho = [row['rho'] for row in feature_rows]
feature_negative = sum(1 for rho in feature_rho if rho < 0)
tests.add(test_id='T24', ledger_ids='L6-025;L6-026', location='SI Appendix, Section 2, REM Parameters',
          test='REM feature-count sweep, all 54 runs (18 feature counts from 5 to 120, seeds 12345, 777, 20260731): '
               'number of runs whose Spearman rho is below zero, with the smallest and largest single-run rho and '
               'the largest p; source %s'
               % feature_source,
          statistic_name='runs with rho below zero', statistic=feature_negative, n=len(feature_rho), unit='runs',
          df='', p=max(row['p'] for row in feature_rows), p_type='parametric two-sided',
          effect_size_name='rho, smallest to largest', effect_size='', ci_low=min(feature_rho),
          ci_high=max(feature_rho), ci_method='range over the 54 runs (not a confidence interval)',
          resampling_unit='', n_resamples='', seed='12345; 777; 20260731')
print('  54 runs: rho %.4f to %.4f, %d of 54 below zero, largest p %.2g'
      % (min(feature_rho), max(feature_rho), feature_negative, max(row['p'] for row in feature_rows)))

# ============================================================================ T25
print('\nT25: Pearson r between d′ and rho across the 32 cells (Fisher z interval)')
dprime_values = [cell['dprime'] for cell in cells]
rho_values = [cell['rhoItem'] for cell in cells]
pearson_result = stats.pearsonr(dprime_values, rho_values)
r_value = float(pearson_result.statistic)
fisher_low, fisher_high = bootstrap.fisher_z_interval(r_value, len(cells))
tests.add(test_id='T25', ledger_ids='L6-035;L6-024', location='SI Appendix, Section 2, REM Parameters',
          test='REM parameter sweep, all 32 cells: Pearson correlation between each cell\'s overall d′ and its '
               'Spearman rho',
          statistic_name='Pearson r', statistic=r_value, n=len(cells), unit='cells', df=len(cells) - 2,
          p=float(pearson_result.pvalue), p_type='parametric two-sided', effect_size_name='r squared',
          effect_size=r_value ** 2, ci_low=fisher_low, ci_high=fisher_high, ci_method='Fisher z',
          resampling_unit='', n_resamples='', seed='')
print('  all 32 cells: r = %.4f, p = %.3g, df %d, Fisher z [%.4f, %.4f], r squared %.3f'
      % (r_value, pearson_result.pvalue, len(cells) - 2, fisher_low, fisher_high, r_value ** 2))

tests.write(paths['outputs'])
