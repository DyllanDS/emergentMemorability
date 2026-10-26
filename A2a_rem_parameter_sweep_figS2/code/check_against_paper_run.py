"""
check_against_paper_run.py: compares this run of the two REM sweeps with the saved runs in
data/paper_run/ (the output the paper reports), and writes the module's outputs/numbers.csv (one row
per number of the paper this module reproduces).

What it checks
--------------
1. The 32-cell sweep (rem_parameter_sweep.m). outputs/rem_parameter_sweep.csv against the saved table
   data/paper_run/rem_parameter_sweep.csv, cell by cell and column by column (difference
   allowed: 1e-6; for the tiny p values of the pItem column, 1e-6 of the value), and
   outputs/rem_parameter_sweep.txt against the saved printed report, line by line.
   pItem is the p of each cell's Spearman rho (t approximation with n - 2 degrees of freedom; 4.84e-30
   and smaller). The p values are also recomputed here with the same formula (scipy's spearmanr) from
   the per-item file the sweep writes (outputs/rem_parameter_sweep_items.csv) and compared with the
   saved pItem column; numbers.csv uses the recomputed values. (A tcdf function from another toolbox on
   the MATLAB path, such as FieldTrip's external/stats/tcdf.m, writes 0 for p values below about 1e-16;
   run_module.py starts MATLAB with restoredefaultpath to prevent this, and this script reports it if
   it happens.)
2. The feature-count sweep (feature_count_sweep.py). outputs/sweep_rem.csv against the saved table
   data/paper_run/sweep_rem.csv: the eight columns of the saved table, every one of the 54 runs,
   compared as numbers (difference allowed, as for the 32-cell table: 1e-6, and for the p columns
   1e-6 of the value). Skipped, with a message, when outputs/sweep_rem.csv is absent or does not hold
   the 54 runs (for example run_module.py --skip-long without an earlier full run); the numbers then
   come from the saved table.
3. The comparison of the Methods, printed side by side: the sweep's standard cell (Spearman with the
   negated mean distance, the -0.52), the saved Fig. 1 run of module 01 (Spearman with the negated mean
   distance, the -0.55 of the Methods; Pearson with 1 / mean distance, the -0.56 of Fig. 1b), and the
   sweep's own cell at the Fig. 1 values.

Inputs   outputs/ (rem_parameter_sweep.csv, .txt, _items.csv, accuracy_link.json, sweep_rem.csv,
         tests.csv, figures/, figure_data/), data/paper_run/ (the three saved files), and the saved Fig. 1
         run in ../01_rem_simulation_fig1/data/paper_run/ (module 01's data)
Outputs  outputs/equality_with_paper_run.csv (one row per table column compared, and one for the
         printed report) and
         outputs/numbers.csv (ledger_id, location, quote, paper_value, computed_value, match,
         match_type, script, mode, note)

Run (from the repository root, after the sweep, the statistics and the figure):
    python A2a_rem_parameter_sweep_figS2/code/check_against_paper_run.py
"""
import csv
import json
import os
import re
import sys

import numpy as np
import scipy.io
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402
import numbers_table  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_tables  # noqa: E402  (this module's readers of the sweep's tables)

TOLERANCE = 1e-6
FIG1_RUN_FILE = 'fig1_paper_run.mat'
PAPER_FEATURE_COUNTS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 49, 55, 60, 66, 70, 80, 90, 100, 120]
PAPER_SEEDS = [12345, 777, 20260731]
SWEEP_COLUMNS = ['g', 'w', 'u', 'c', 'nStudy', 'crit', 'rhoItem', 'pItem', 'rFeat', 'pFeat', 'meanHit', 'meanFA',
                 'dprime']
FEATURE_SWEEP_COLUMNS = ['mean_cr', 'r_sim_spearman', 'p_sim_spearman', 'r_dist_pearson', 'r_meta', 'p_meta']

paths = release_paths.module_paths(__file__)
problems = []


def read_csv_rows(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def need(path, how_to_make_it):
    if not os.path.exists(path):
        sys.exit('check_against_paper_run.py: %s is missing; %s' % (os.path.relpath(path, paths['module']),
                                                                   how_to_make_it))


# ============================================================================ 1. the 32-cell sweep
fresh_sweep_file = os.path.join(paths['outputs'], 'rem_parameter_sweep.csv')
fresh_report_file = os.path.join(paths['outputs'], 'rem_parameter_sweep.txt')
items_file = os.path.join(paths['outputs'], 'rem_parameter_sweep_items.csv')
for path in (fresh_sweep_file, fresh_report_file, items_file):
    need(path, 'run code/rem_parameter_sweep.m first (run_module.py does it)')
paper_sweep_file = os.path.join(paths['paper_run'], 'rem_parameter_sweep.csv')
paper_report_file = os.path.join(paths['paper_run'], 'rem_parameter_sweep.txt')

fresh_cells = read_csv_rows(fresh_sweep_file)
paper_cells = read_csv_rows(paper_sweep_file)
assert len(fresh_cells) == 32 and len(paper_cells) == 32
for cell_index in range(32):
    if fresh_cells[cell_index]['label'] != paper_cells[cell_index]['label']:
        sys.exit('cell %d is "%s" here and "%s" in the saved run' % (cell_index + 1, fresh_cells[cell_index]['label'],
                                                                    paper_cells[cell_index]['label']))

# Per-item counts and distances: recompute every cell's rho and its p (t approximation).
items_by_cell = sweep_tables.read_items(items_file)
recomputed_rho = []
recomputed_p = []
items_per_cell = []
experiments_per_cell = []
for cell_number in range(1, 33):
    rho, p_value = sweep_tables.spearman_rho_and_p(items_by_cell[cell_number]['memorability'],
                                                   items_by_cell[cell_number]['negated_mean_distance'])
    recomputed_rho.append(rho)
    recomputed_p.append(p_value)
    items_per_cell.append(items_by_cell[cell_number]['n_items'])
    # every simulated experiment tests 20 studied items (min(20, list length); every list has 40 or more)
    experiments_per_cell.append(items_by_cell[cell_number]['old_trials'] / 20)

equality_rows = []
print('1. The 32-cell sweep: this run against the saved run')
for column in SWEEP_COLUMNS:
    fresh_values = np.array([float(cell[column]) for cell in fresh_cells])
    paper_values = np.array([float(cell[column]) for cell in paper_cells])
    absolute_differences = np.abs(fresh_values - paper_values)
    relative_differences = absolute_differences / np.maximum(np.abs(paper_values), 1e-300)
    identical = int(np.sum(fresh_values == paper_values))
    within = bool(np.all(absolute_differences <= TOLERANCE))
    note = ''
    if column == 'pItem':
        # p values of 4.84e-30 and smaller: compared relative to their size
        within = bool(np.all(relative_differences <= TOLERANCE))
        recomputed = np.array(recomputed_p)
        relative_to_recomputed = np.abs(recomputed - paper_values) / paper_values
        cells_with_zero_p = int(np.sum(fresh_values == 0))
        if cells_with_zero_p:
            note = ('this run has pItem = 0 in %d of 32 cells: MATLAB used a tcdf function from outside its own '
                    'toolboxes (see the warning of rem_parameter_sweep.m; run_module.py avoids this with '
                    'restoredefaultpath); ' % cells_with_zero_p)
            print('  WARNING: pItem = 0 in %d cells of this run; the numbers below use the p values recomputed '
                  'from the per-item file' % cells_with_zero_p)
        else:
            note = ('this run\'s pItem differs from the saved column by at most %.1e relative; '
                    % relative_differences.max())
            if not within:
                problems.append('column pItem differs from the saved run by more than %g relative' % TOLERANCE)
        note += ('the p values recomputed from the per-item file with the same t approximation (scipy spearmanr) '
                 'differ from the saved column by at most %.1e relative; largest p %.3g recomputed and %.3g in the '
                 'saved table' % (relative_to_recomputed.max(), recomputed.max(), paper_values.max()))
        if relative_to_recomputed.max() > TOLERANCE:
            problems.append('pItem recomputed from the items differs from the saved run by more than %g relative'
                            % TOLERANCE)
    elif not within:
        problems.append('column %s differs from the saved run by more than %g' % (column, TOLERANCE))
    if column == 'rhoItem':
        largest_item_difference = float(np.max(np.abs(np.array(recomputed_rho) - fresh_values)))
        note = 'rho recomputed in Python from the per-item file differs from the MATLAB table by at most %.1e' \
               % largest_item_difference
        if largest_item_difference > 1e-12:
            problems.append('rho recomputed from the per-item file differs from the MATLAB table')
    equality_rows.append({'table': 'rem_parameter_sweep.csv (32 cells)', 'column': column, 'compared': 32,
                          'identical': identical, 'largest_abs_difference': '%.3g' % absolute_differences.max(),
                          'largest_relative_difference': '%.3g' % relative_differences.max(),
                          'within_1e-6': 'yes' if within else 'no', 'note': note})
    print('  %-8s identical in %2d of 32 cells; largest difference %.3g%s'
          % (column, identical, absolute_differences.max(), ('  (' + note + ')') if note else ''))

with open(fresh_report_file) as handle:
    fresh_report = handle.read().splitlines()
with open(paper_report_file) as handle:
    paper_report = handle.read().splitlines()
report_lines_differing = 0
for line_index in range(max(len(fresh_report), len(paper_report))):
    fresh_line = fresh_report[line_index] if line_index < len(fresh_report) else ''
    paper_line = paper_report[line_index] if line_index < len(paper_report) else ''
    if fresh_line != paper_line:
        report_lines_differing += 1
print('  printed report rem_parameter_sweep.txt: %d of %d lines differ'
      % (report_lines_differing, len(paper_report)))
if report_lines_differing:
    problems.append('the printed report differs from the saved report in %d lines' % report_lines_differing)
equality_rows.append({'table': 'rem_parameter_sweep.txt (printed report)', 'column': 'every line',
                      'compared': len(paper_report), 'identical': len(paper_report) - report_lines_differing,
                      'largest_abs_difference': '', 'largest_relative_difference': '',
                      'within_1e-6': 'yes' if report_lines_differing == 0 else 'no', 'note': 'text compared line by line'})

# ============================================================================ 2. the feature-count sweep
fresh_feature_file = os.path.join(paths['outputs'], 'sweep_rem.csv')
paper_feature_rows = read_csv_rows(os.path.join(paths['paper_run'], 'sweep_rem.csv'))
feature_rows = paper_feature_rows
feature_mode = 'saved_run'
feature_source = 'data/paper_run/sweep_rem.csv'
print('\n2. The feature-count sweep')
if os.path.exists(fresh_feature_file):
    fresh_feature_rows = read_csv_rows(fresh_feature_file)
    runs_present = sorted((int(row['w']), int(row['seed'])) for row in fresh_feature_rows)
    runs_of_paper = sorted((count, seed) for count in PAPER_FEATURE_COUNTS for seed in PAPER_SEEDS)
    if runs_present == runs_of_paper:
        feature_rows = fresh_feature_rows
        feature_mode = 'rerun'
        feature_source = 'outputs/sweep_rem.csv'
        for column in FEATURE_SWEEP_COLUMNS:
            identical = 0
            largest_difference = 0.0
            largest_relative_difference = 0.0
            for paper_row in paper_feature_rows:
                fresh_row = [row for row in fresh_feature_rows
                             if row['w'] == paper_row['w'] and row['seed'] == paper_row['seed']][0]
                fresh_value = float(fresh_row[column])
                paper_value = float(paper_row[column])
                difference = abs(fresh_value - paper_value)
                largest_difference = max(largest_difference, difference)
                largest_relative_difference = max(largest_relative_difference,
                                                  difference / max(abs(paper_value), 1e-300))
                if fresh_value == paper_value:
                    identical += 1
            # The same allowance as for the 32-cell table: a difference of up to 1e-6, and for the two p
            # columns (p_sim_spearman goes down to about 4e-197) up to 1e-6 of the value. Other versions of
            # numpy or scipy can change the last digits; anything larger is reported as a problem.
            if column.startswith('p_'):
                within = largest_relative_difference <= TOLERANCE
            else:
                within = largest_difference <= TOLERANCE
            equality_rows.append({'table': 'sweep_rem.csv (54 runs)', 'column': column, 'compared': 54,
                                  'identical': identical, 'largest_abs_difference': '%.3g' % largest_difference,
                                  'largest_relative_difference': '%.3g' % largest_relative_difference,
                                  'within_1e-6': 'yes' if within else 'no', 'note': ''})
            print('  %-15s identical in %2d of 54 runs; largest difference %.3g' % (column, identical, largest_difference))
            if not within:
                problems.append('feature-count sweep column %s differs from the saved run by more than %g%s '
                                '(%d of 54 runs identical)' % (column, TOLERANCE,
                                                              ' of the value' if column.startswith('p_') else '',
                                                              identical))
    else:
        print('  outputs/sweep_rem.csv does not hold the paper\'s 54 runs (a test run or an unfinished run); '
              'the numbers below come from data/paper_run/sweep_rem.csv')
else:
    print('  not run here (run_module.py --skip-long, or feature_count_sweep.py not run); the numbers below come '
          'from data/paper_run/sweep_rem.csv, the saved run')

with open(os.path.join(paths['outputs'], 'equality_with_paper_run.csv'), 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=['table', 'column', 'compared', 'identical', 'largest_abs_difference',
                                                'largest_relative_difference', 'within_1e-6', 'note'])
    writer.writeheader()
    for row in equality_rows:
        writer.writerow(row)
print('wrote outputs/equality_with_paper_run.csv')

# ============================================================================ the numbers the paper reports
cells = []
for cell_index, row in enumerate(fresh_cells):
    cell = {'label': row['label']}
    for column in SWEEP_COLUMNS:
        cell[column] = float(row[column])
    cell['p_recomputed'] = recomputed_p[cell_index]
    cells.append(cell)
reference = [cell for cell in cells if cell['label'] == 'Shiffrin & Steyvers reference'][0]
fig1_cell = [cell for cell in cells if cell['label'].startswith('paper values')][0]
near_ceiling = [cell for cell in cells if cell['label'] == 'grid g=0.30 w=49'][0]
list_187 = [cell for cell in cells if cell['label'] == 'list length=187'][0]
all_rho = [cell['rhoItem'] for cell in cells]
all_dprime = [cell['dprime'] for cell in cells]
all_hit = [cell['meanHit'] for cell in cells]
largest_p = max(cell['p_recomputed'] for cell in cells)
number_negative = sum(1 for rho in all_rho if rho < 0)
standard_parameters = (0.40, 20, 0.30, 0.70, 40, 1.0)
standard_reruns = []
for cell in cells:
    parameters = (cell['g'], cell['w'], cell['u'], cell['c'], cell['nStudy'], cell['crit'])
    if cell is not reference and all(abs(a - b) < 1e-9 for a, b in zip(parameters, standard_parameters)):
        standard_reruns.append(cell)
assert len(standard_reruns) == 5
rerun_rho = [cell['rhoItem'] for cell in standard_reruns]
six_rho = rerun_rho + [reference['rhoItem']]

need(os.path.join(paths['outputs'], 'accuracy_link.json'), 'run code/accuracy_link.py first')
with open(os.path.join(paths['outputs'], 'accuracy_link.json')) as handle:
    accuracy_link = json.load(handle)

# The 95% intervals printed with the -0.52 and the r = 0.71, from outputs/tests.csv (stats_figS2.py): the item
# bootstrap of the standard cell (T20) and the Fisher z interval of r(d′, rho) over the 32 cells (T25).
need(os.path.join(paths['outputs'], 'tests.csv'), 'run code/stats_figS2.py first')
tests_rows = read_csv_rows(os.path.join(paths['outputs'], 'tests.csv'))
standard_test = [row for row in tests_rows if row['test_id'] == 'T20' and row['ledger_ids'] == 'L5-006;L6-022'][0]
accuracy_test = [row for row in tests_rows if row['test_id'] == 'T25'][0]
assert abs(float(standard_test['statistic']) - reference['rhoItem']) < 1e-5
assert abs(float(accuracy_test['statistic']) - accuracy_link['r_dprime_rho_all32']) < 1e-5
standard_interval = [float(standard_test['ci_low']), float(standard_test['ci_high'])]
accuracy_interval = [float(accuracy_test['ci_low']), float(accuracy_test['ci_high'])]

dprime_sorted = sorted(all_dprime)
fig1_cell_dprime_rank = 1 + sum(1 for value in all_dprime if value < fig1_cell['dprime'])
feature_counts = sorted(set(int(row['w']) for row in feature_rows))
feature_rho = [float(row['r_sim_spearman']) for row in feature_rows]
feature_p = [float(row['p_sim_spearman']) for row in feature_rows]
feature_negative = sum(1 for rho in feature_rho if rho < 0)

# The design header of the printed report (written by the MATLAB script from its reference settings).
header_match = None
for line in fresh_report:
    header_match = re.search(r'database=(\d+), (\d+) runs per cell', line)
    if header_match:
        break
assert header_match is not None, 'the report has no "database=..., ... runs per cell" line'
database_size = int(header_match.group(1))
runs_per_cell = int(header_match.group(2))
if not (all(n == database_size for n in items_per_cell) and all(n == runs_per_cell for n in experiments_per_cell)):
    problems.append('per-item file does not show %d items and %d experiments in every cell' % (database_size,
                                                                                              runs_per_cell))

# The design of the 32 cells, read back from the table.
grid_cells = [cell for cell in cells if cell['label'].startswith('grid ')]


def joined(values, number_format):
    return '/'.join(number_format % value for value in values)


design_text = ('g %s x w %s (%d cells); storage u %s; copy c %s; list length %s; criterion %s; '
               'standard cell g %.2f w %d u %.2f c %.2f list %d crit %.1f; Fig. 1 values cell g %.2f w %d u %.2f c %.2f'
               % (joined(sorted(set(cell['g'] for cell in grid_cells)), '%.2f'),
                  joined(sorted(set(cell['w'] for cell in grid_cells)), '%d'), len(grid_cells),
                  joined([cell['u'] for cell in cells if cell['label'].startswith('storage')], '%.2f'),
                  joined([cell['c'] for cell in cells if cell['label'].startswith('copy')], '%.2f'),
                  joined([cell['nStudy'] for cell in cells if cell['label'].startswith('list length')], '%d'),
                  joined([cell['crit'] for cell in cells if cell['label'].startswith('criterion')], '%.1f'),
                  reference['g'], reference['w'], reference['u'], reference['c'], reference['nStudy'], reference['crit'],
                  fig1_cell['g'], fig1_cell['w'], fig1_cell['u'], fig1_cell['c']))

# ============================================================================ 3. the Methods comparison
print('\n3. Methods: the sweep\'s standard cell and the saved Fig. 1 run')
print('  -0.52: the sweep\'s Shiffrin and Steyvers cell, Spearman rho with the '
      'negated mean distance = %+.4f' % reference['rhoItem'])
fig1_file = os.path.join(release_paths.find_module('01'), 'data', 'paper_run', FIG1_RUN_FILE)
fig1_note = ''
if os.path.exists(fig1_file):
    figure_data = scipy.io.loadmat(fig1_file, squeeze_me=True)['figureData']
    fig1_memorability = np.asarray(figure_data['hitFARate'].item(), dtype=float)
    fig1_mean_distance = np.asarray(figure_data['avgDistances'].item(), dtype=float)
    fig1_similarity = np.asarray(figure_data['similarity'].item(), dtype=float)
    has_score = np.isfinite(fig1_memorability)
    fig1_pearson = float(stats.pearsonr(fig1_similarity[has_score], fig1_memorability[has_score]).statistic)
    fig1_spearman = float(stats.spearmanr(-fig1_mean_distance[has_score], fig1_memorability[has_score]).statistic)
    print('  -0.56 (Fig. 1b): the saved Fig. 1 run (module 01), Pearson r with 1 / mean '
          'distance = %+.4f' % fig1_pearson)
    print('  -0.55 (Methods): the same Fig. 1 run, Spearman rho with the negated mean distance (the sweep\'s measure) '
          '= %+.4f' % fig1_spearman)
    fig1_note = ('; the saved Fig. 1 run (module 01) gives r = %.4f with 1 / mean distance (Fig. 1b) and rho = %.4f '
                 'with the negated mean distance (the -0.55 of the Methods)' % (fig1_pearson, fig1_spearman))
else:
    print('  the saved Fig. 1 run (module 01 data) is not present; not compared')
print('  the sweep\'s own cell at the Fig. 1 values (g 0.65, 49 features, copy 0.60): Spearman %+.4f, d′ %.2f'
      % (fig1_cell['rhoItem'], fig1_cell['dprime']))

# ============================================================================ 4. numbers.csv
figure_file = os.path.join(paths['figures'], 'FigS2_rem_parameter_sweep.png')
panel_a_file = os.path.join(paths['figure_data'], 'FigS2a.csv')
panel_b_file = os.path.join(paths['figure_data'], 'FigS2b.csv')
figure_present = os.path.exists(figure_file) and os.path.exists(panel_a_file) and os.path.exists(panel_b_file)
if figure_present:
    plotted_a = sorted(float(row['rho_item']) for row in read_csv_rows(panel_a_file))
    plotted_b = sorted((float(row['dprime']), float(row['rho_item'])) for row in read_csv_rows(panel_b_file))
    if plotted_a != sorted(all_rho) or plotted_b != sorted(zip(all_dprime, all_rho)):
        problems.append('the plotted values of Fig. S2 differ from the table')
else:
    problems.append('Fig. S2 has not been drawn (run code/make_figureS2.py)')

table = numbers_table.NumbersTable(script='code/rem_parameter_sweep.m; code/check_against_paper_run.py')
sweep_summary = ('%d of 32 cells negative; rho %.4f to %.4f; standard values %.4f'
                 % (number_negative, min(all_rho), max(all_rho), reference['rhoItem']))
table.add('L1-010', sweep_summary, mode='rerun',
          note='Cross-reference to SI Appendix, Section 2: the number of the 32 parameter-sweep cells whose Spearman '
               'rho between memorability and summed similarity is negative, the range of rho over the cells, and '
               'rho at the standard values of Shiffrin and Steyvers (1997)')
table.add('L1-025', 'this module: %d of 32 sweep cells negative (rho %.4f to %.4f); feature-count sweep %d of %d runs '
                    'negative (rho %.4f to %.4f)' % (number_negative, min(all_rho), max(all_rho), feature_negative,
                                                     len(feature_rho), min(feature_rho), max(feature_rho)),
          mode='rerun', script='code/rem_parameter_sweep.m; code/feature_count_sweep.py',
          note='Cross-reference to SI Appendix, Section 2. This module covers REM\'s parameters: the parameter-sweep '
               'cells and the feature-count runs with a negative Spearman rho, and the range of rho in each sweep; '
               'the decision-criterion, continuous-recognition, list-context and measured-feature simulations are '
               'modules A2b, A2c, A2d and A2e')
table.add('L5-005', [reference['w'], reference['g'], reference['c']], mode='rerun', script='code/rem_parameter_sweep.m',
          note='Number of features, g and copy fidelity (probability of correct encoding) of the sweep cell at the '
               'standard values; the cell also uses storage probability %.2f, list length %d and criterion %.1f'
               % (reference['u'], reference['nStudy'], reference['crit']))
table.add('L5-006', [reference['rhoItem']] + standard_interval, mode='rerun',
          script='code/rem_parameter_sweep.m; code/stats_figS2.py',
          note='Spearman rho across the %s simulated items between memorability (hit rate minus false alarm rate) '
               'and summed similarity (negated mean Euclidean distance to the other items) in the sweep cell at the '
               'standard values, with its 95%% percentile bootstrap interval over the items (tests.csv T20)%s'
               % ('{:,}'.format(database_size), fig1_note))
table.add('L5-008', sweep_summary, mode='rerun',
          note='Cross-reference to SI Appendix, Section 2: the number of the 32 parameter-sweep cells with a negative '
               'Spearman rho, the range of rho over the cells, and rho at the standard values')
table.add('L6-019', len(cells), mode='rerun', script='code/rem_parameter_sweep.m',
          note='Number of simulations (cells) in the parameter sweep: the rows of outputs/rem_parameter_sweep.csv')
table.add('L6-020', design_text, mode='rerun', script='code/rem_parameter_sweep.m',
          note='Design of the 32 cells read back from outputs/rem_parameter_sweep.csv: g crossed with the number of '
               'features (15 cells), storage probability, copy fidelity, list length and criterion varied one at a '
               'time from the standard values, the standard cell, and the cell with the Fig. 1 values')
table.add('L6-021', [number_negative, min(all_rho), max(all_rho)], mode='rerun', script='code/rem_parameter_sweep.m',
          note='Number of the 32 sweep cells with a negative Spearman rho, and the smallest and largest rho (at "%s" '
               'and "%s")' % ([c['label'] for c in cells if c['rhoItem'] == min(all_rho)][0],
                              [c['label'] for c in cells if c['rhoItem'] == max(all_rho)][0]))
table.add('L6-022', [reference['rhoItem']] + standard_interval, mode='rerun',
          script='code/rem_parameter_sweep.m; code/stats_figS2.py',
          note='Spearman rho in the sweep cell at the standard values (g 0.40, 20 features, storage 0.30, copy '
               'fidelity 0.70, list length 40, criterion 1.0), with its 95% percentile bootstrap interval over the '
               'items (tests.csv T20)')
table.add('L6-023', int(list_187['nStudy']), mode='rerun', script='code/rem_parameter_sweep.m',
          note='List length of the sweep cell "list length=187", a fixed number in rem_parameter_sweep.m: the number '
               'of distinct images in a 295-trial session of Kramer et al. (2023), counted in their trial file '
               '(EXTERNAL_INPUTS.md, M1). In this run that cell gives rho %.4f and d′ %.2f'
               % (list_187['rhoItem'], list_187['dprime']))
table.add('L6-024', 'd′ %.3f to %.3f (hit rate %.3f to %.3f) while rho stays %.4f to %.4f; r(d′, rho) = %.4f; weakest '
                    'rho at the most accurate cell (grid g=0.30 w=49: rho %.4f, d′ %.2f, hit rate %.3f)'
                    % (min(all_dprime), max(all_dprime), min(all_hit), max(all_hit), min(all_rho), max(all_rho),
                       accuracy_link['r_dprime_rho_all32'], near_ceiling['rhoItem'], near_ceiling['dprime'],
                       near_ceiling['meanHit']),
          mode='rerun', script='code/rem_parameter_sweep.m; code/accuracy_link.py')
table.add('L6-025', [min(feature_counts), max(feature_counts)], mode=feature_mode, script='code/feature_count_sweep.py',
          note='Smallest and largest number of features in the feature-count sweep: %d feature counts (%s) x %d seeds '
               '= %d runs, from %s'
               % (len(feature_counts), ', '.join(str(count) for count in feature_counts),
                  len(set(row['seed'] for row in feature_rows)), len(feature_rows), feature_source))
table.add('L6-026', [min(feature_rho), max(feature_rho)], mode=feature_mode, script='code/feature_count_sweep.py',
          note='Smallest and largest Spearman rho over the %d single runs of the feature-count sweep (column '
               'r_sim_spearman: memorability against 1 / mean Euclidean distance, the same ranks as the negated mean '
               'distance); %d of %d are negative, largest p %.2g; from %s'
               % (len(feature_rho), feature_negative, len(feature_rho), max(feature_p), feature_source))
table.add('L6-027', 'Fig. 1 values cell: d′ %.4f, number %d of 32 from the lowest; median d′ of the 32 cells %.4f; '
                    'd′ range %.3f to %.3f' % (fig1_cell['dprime'], fig1_cell_dprime_rank,
                                                (dprime_sorted[15] + dprime_sorted[16]) / 2, min(all_dprime),
                                                max(all_dprime)),
          mode='rerun', script='code/rem_parameter_sweep.m; code/accuracy_link.py',
          note='Overall d′ of the sweep cell with the Fig. 1 values (g 0.65, 49 features, copy fidelity 0.60), its '
               'rank among the 32 cells from the lowest d′, the median d′ and the range of d′ over the cells; its rho '
               '%.4f is number %d of 32 from the most negative'
               % (fig1_cell['rhoItem'], accuracy_link['fig1_values_rank_most_negative_of_32']))
table.add('L6-028', 'outputs/figures/FigS2_rem_parameter_sweep.png panel a: the 32 rho values grouped by the parameter '
                    'varied (outputs/figure_data/FigS2a.csv); each group to the right of the standard cell includes '
                    'its rerun of the standard values (%s)' % ', '.join(cell['label'] for cell in standard_reruns),
          mode='rerun', script='code/make_figureS2.py',
          note=('Fig. S2a in outputs/figures/FigS2_rem_parameter_sweep.png: the Spearman rho of the 32 cells grouped '
                'by the parameter varied, with the standard cell and the Fig. 1 values cell in black; plotted values '
                'in outputs/figure_data/FigS2a.csv' if figure_present else 'NOT DRAWN: run code/make_figureS2.py'))
table.add('L6-029', 'outputs/figures/FigS2_rem_parameter_sweep.png panel b: rho against d′ for the 32 cells '
                    '(outputs/figure_data/FigS2b.csv), r = %.4f, p < 0.001, n = 32; labels Standard (d′ %.2f, rho %.3f), '
                    'Fig. 1 Values (d′ %.2f, rho %.3f), Near Ceiling (grid g=0.30 w=49, d′ %.2f, rho %.3f)'
                    % (accuracy_link['r_dprime_rho_all32'], reference['dprime'], reference['rhoItem'],
                       fig1_cell['dprime'], fig1_cell['rhoItem'], near_ceiling['dprime'], near_ceiling['rhoItem']),
          mode='rerun', script='code/make_figureS2.py',
          note=('Fig. S2b in outputs/figures/FigS2_rem_parameter_sweep.png: the Spearman rho of the 32 cells against '
                'overall d′ with the least-squares line; plotted values in outputs/figure_data/FigS2b.csv and '
                'FigS2b_fit_line.csv' if figure_present else 'NOT DRAWN: run code/make_figureS2.py'))
table.add('L6-030', len(cells), mode='rerun', script='code/rem_parameter_sweep.m',
          note='Number of simulations (cells) in the parameter sweep')
table.add('L6-031', [runs_per_cell, database_size], mode='rerun', script='code/rem_parameter_sweep.m',
          note='Simulated recognition experiments per cell and simulated items per cell, from the header of '
               'outputs/rem_parameter_sweep.txt ("database=%d, %d runs per cell"); the per-item file shows %s items '
               'and %s experiments (old trials / 20) in each of the 32 cells' % (database_size, runs_per_cell,
                                                                               '{:,}'.format(items_per_cell[0]),
                                                                               '{:,.0f}'.format(experiments_per_cell[0])))
table.add('L6-032', 'neg_mean_distance = minus the mean Euclidean distance from an item to the other %s items of the '
                    'database (rem_parameter_sweep.m, meanPairwiseDistance, the item itself left out)'
                    % '{:,}'.format(database_size - 1),
          mode='rerun', script='code/rem_parameter_sweep.m')
table.add('L6-033', largest_p, mode='rerun',
          script='code/rem_parameter_sweep.m; code/check_against_paper_run.py',
          note='Largest p of the Spearman rho over the 32 cells (t approximation with n - 2 degrees of freedom, '
               'recomputed from the per-item file; the pItem column gives %.3g)' % max(cell['pItem'] for cell in cells))
table.add('L6-034', [min(rerun_rho), max(rerun_rho)], mode='rerun', script='code/rem_parameter_sweep.m',
          note='Smallest and largest Spearman rho of the five cells that rerun the standard values, the point at the '
               'standard setting in each group to the right of the reference values in Fig. S2a (%s); with the '
               'standard cell, the six runs span %.4f to %.4f'
               % (', '.join('%s %.4f' % (cell['label'], cell['rhoItem']) for cell in standard_reruns),
                  min(six_rho), max(six_rho)))
table.add('L6-035', [accuracy_link['r_dprime_rho_all32']] + accuracy_interval + [len(cells)], mode='rerun',
          script='code/accuracy_link.py; code/stats_figS2.py',
          note='Pearson r between overall d′ and Spearman rho over the 32 cells (r squared %.3f, p %.2g), its 95%% '
               'Fisher z interval (tests.csv T25) and the number of cells' % (accuracy_link['r_squared_all32'],
                                                                              accuracy_link['p_dprime_rho_all32']))
table.add('L6-221', 'd′ %.3f to %.3f over the 32 cells while rho is below zero in %d of 32 (%.4f to %.4f)'
                    % (min(all_dprime), max(all_dprime), number_negative, min(all_rho), max(all_rho)),
          mode='rerun', script='code/rem_parameter_sweep.m',
          note='Overall d′ and the Spearman rho between memorability and summed similarity over the 32 sweep cells')
table.write(paths['outputs'])

matches = {}
for row in table.rows:
    matches[row['match']] = matches.get(row['match'], 0) + 1
not_matching = [row['ledger_id'] for row in table.rows if row['match'] == 'no']
if not_matching:
    problems.append('ledger items that do not match: ' + ', '.join(not_matching))
print('\nnumbers.csv: %s' % matches)
if problems:
    print('\nPROBLEMS:')
    for problem in problems:
        print('  ' + problem)
    sys.exit(1)
print('\nall checks passed')
