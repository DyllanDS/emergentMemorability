"""
make_figureS3.py: SI Appendix, Fig. S3, the 18 REM simulations of the Bylinskii et al. (2015) context
manipulation of Fig. 2a, varied one factor at a time (SI Appendix, Section 2, Category Structure and
Decision Criteria).

What the figure shows
---------------------
One panel. Each row is one of the 18 simulations of code/bylinskii_robustness.m. The dot is the Spearman
correlation, across the same target items, between each item's memorability (hit rate minus false alarm
rate) measured in the within-category context and in the across-category context. Positive = items keep
their memorability rank across contexts.
Rows are grouped by the one factor that was varied:
  Unequal Criteria  (within-category / across-category decision criterion; Fig. 2a uses 20 / 5)
  Equal Criteria    (both contexts use the same criterion)
  Category-Defining Features, Categories, Items per Category
The five black dots are simulations that all use the category structure and criteria of Fig. 2a
(21 categories of 300 items, 10 category-defining features, criteria 20 / 5); they differ only in the
random draw. Their label says "Fig. 2a Design" because they share the design of Fig. 2a (structure and
criteria) and use the REM settings of the grid (g 0.40, copy fidelity 0.70). Every dot lies right of the
dashed zero line.

Data (this module)
------------------
  outputs/bylinskii_robustness.csv          the run of bylinskii_robustness.m (default), or
  data/paper_run/bylinskii_robustness.csv   the saved run, the output the paper reports (--paper-run)
  code/bylinskii_robustness.m               the grid script (read here only to print its fixed settings)

Outputs
-------
  outputs/figures/FigS3_bylinskii_robustness.png (300 dpi)
  outputs/figure_data/FigS3.csv (the plotted numbers, one row per dot, top to bottom)
  (with --paper-run the two names end in _paper_run)

Run (from the repository root):
    python A2b_category_structure_criteria_figS3/code/make_figureS3.py
    python A2b_category_structure_criteria_figS3/code/make_figureS3.py --paper-run
"""
import argparse
import csv
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

sys.dont_write_bytecode = True   # write no __pycache__ folder into shared/python (set before the import below)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402
import figure_style as ls  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robustness_tables  # noqa: E402  (this module's readers of the grid's tables)

parser = argparse.ArgumentParser(description='Fig. S3: the 18 simulations of the Bylinskii context manipulation.')
parser.add_argument('--paper-run', action='store_true',
                    help='draw the saved run in data/paper_run/ instead of the run in outputs/')
arguments = parser.parse_args()
paths = release_paths.module_paths(__file__)
suffix = '_paper_run' if arguments.paper_run else ''


# ============================================================== 1. read the 18 simulations
if arguments.paper_run:
    table_file = os.path.join(paths['paper_run'], 'bylinskii_robustness.csv')
    print('drawing the saved run: data/paper_run/bylinskii_robustness.csv')
else:
    table_file = os.path.join(paths['outputs'], 'bylinskii_robustness.csv')
    print('drawing this run: outputs/bylinskii_robustness.csv')
if not os.path.exists(table_file):
    sys.exit('make_figureS3.py: %s is missing; run code/bylinskii_robustness.m first, or draw the saved run '
             'with --paper-run' % os.path.relpath(table_file, paths['module']))
simulations = robustness_tables.read_cells(table_file)
assert len(simulations) == 18


def simulations_whose_label_starts_with(prefix):
    """All simulations whose label in the csv starts with the given text."""
    found = []
    for simulation in simulations:
        if simulation['label'].startswith(prefix):
            found.append(simulation)
    return found


criteria_simulations = simulations_whose_label_starts_with('reference') + \
    simulations_whose_label_starts_with('crit ')
unequal_criteria = []
equal_criteria = []
for simulation in criteria_simulations:
    if robustness_tables.has_equal_criteria(simulation):
        equal_criteria.append(simulation)
    else:
        unequal_criteria.append(simulation)
unequal_criteria.sort(key=lambda s: s['critWithin'])
equal_criteria.sort(key=lambda s: s['critWithin'])


def criteria_name(simulation):
    return '%g / %g' % (simulation['critWithin'], simulation['critAcross'])


# (heading, list of (row label, simulation)) from top to bottom
groups = [
    ('Unequal Criteria (Within / Across)', [(criteria_name(s), s) for s in unequal_criteria]),
    ('Equal Criteria', [(criteria_name(s), s) for s in equal_criteria]),
    ('Category-Defining Features', [('%d' % s['catFeatures'], s) for s in
                                    sorted(simulations_whose_label_starts_with('category-defining'),
                                           key=lambda s: s['catFeatures'])]),
    ('Categories', [('%d' % s['nCat'], s) for s in
                    sorted(simulations_whose_label_starts_with('categories'), key=lambda s: s['nCat'])]),
    ('Items per Category', [('%d' % s['nPerCat'], s) for s in
                            sorted(simulations_whose_label_starts_with('items per category'),
                                   key=lambda s: s['nPerCat'])]),
]
count = 0
for heading, rows in groups:
    count += len(rows)
    assert heading in robustness_tables.GROUP_NAMES
    for row_label, simulation in rows:
        assert robustness_tables.group_of(simulation) == heading, simulation['label']
assert count == 18, 'every simulation must appear exactly once'
assert len(equal_criteria) == 4


# ============================================================== 2. draw
ls.apply_style('model')            # REM simulation: box and light grid, as in Fig. 1
figure, axes = ls.new_figure(n_panels=1, width='single', height_cm=10.5)
axis = axes[0]

# y positions from the top: one line for each heading, one for each simulation
row_positions = []
row_labels = []
heading_positions = []
plotted_rows = []                  # for outputs/figure_data/FigS3.csv
y_position = 0
first_black_dot = None
for heading, rows in groups:
    y_position -= 1
    heading_positions.append((heading, y_position))
    for row_label, simulation in rows:
        y_position -= 1
        row_positions.append(y_position)
        row_labels.append(row_label)
        if robustness_tables.has_fig2a_structure(simulation):
            ls.highlight_point(axis, simulation['rho'], y_position, label=None)
            if first_black_dot is None:
                first_black_dot = (simulation['rho'], y_position)
            marker = 'black (Fig. 2a design)'
        else:
            axis.plot([simulation['rho']], [y_position], linestyle='none', marker='o',
                      markersize=ls.MARKER_SIZE + 1, markerfacecolor=ls.DATA_BLUE,
                      markeredgecolor=ls.DATA_BLUE, zorder=3)
            marker = 'blue'
        plotted_rows.append({'group': heading, 'row_label': row_label, 'simulation': simulation['label'],
                             'y_position': y_position, 'rho': repr(simulation['rho']),
                             'pRho': repr(simulation['pRho']), 'marker': marker})

for heading, heading_y in heading_positions:
    axis.text(0.02, heading_y, heading, ha='left', va='center', fontsize=ls.FONT_SIZE,
              color='black', bbox=dict(facecolor='white', edgecolor='none', pad=0.5), zorder=5)
ls.direct_label(axis, first_black_dot[0], first_black_dot[1], 'Fig. 2a Design', color=ls.LABEL_GRAY,
                ha='left', va='center', dx_points=8)

ls.zero_line(axis, axis='x')
axis.set_xlim(-0.1, 1)
axis.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
axis.set_ylim(y_position - 0.7, -0.3)
axis.set_yticks(row_positions)
axis.set_yticklabels(row_labels)
axis.tick_params(axis='y', length=0)
axis.grid(False, axis='y')
axis.set_xlabel('Correlation of Memorability\nAcross Contexts')
ls.plain_ticks(axis)
largest_p = max(s['pRho'] for s in simulations)
number_above_zero = sum(1 for s in simulations if s['rho'] > 0)
# the statistic sits in the empty space to the right of the equal-criteria rows
equal_criteria_rows = [y for (label, y) in zip(row_labels, row_positions) if label in ('5 / 5', '20 / 20')]
if number_above_zero == len(simulations):
    count_text = 'All %d simulations above zero' % len(simulations)
else:
    count_text = '%d of %d simulations above zero' % (number_above_zero, len(simulations))
axis.text(0.98, np.mean(equal_criteria_rows), '%s\n%s'
          % (count_text, ls.format_p(largest_p).replace('p', 'all p', 1)), ha='right', va='center',
          fontsize=ls.FONT_SIZE, color='black', linespacing=1.3,
          bbox=dict(facecolor='white', edgecolor='none', pad=1.0), zorder=5)

figure_paths = ls.save_figure(figure, 'FigS3_bylinskii_robustness' + suffix, paths['figures'])

figure_data_file = os.path.join(paths['figure_data'], 'FigS3%s.csv' % suffix)
with open(figure_data_file, 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=['group', 'row_label', 'simulation', 'y_position', 'rho', 'pRho',
                                                'marker'])
    writer.writeheader()
    for plotted_row in plotted_rows:
        writer.writerow(plotted_row)
print('wrote %s' % os.path.relpath(figure_data_file, paths['module']))


# ============================================================== 3. print every number used
print('\n==== the numbers of Fig. S3 (from %s) ====' % os.path.relpath(table_file, paths['module']))
all_rho = [s['rho'] for s in simulations]
print('rho over 18: %.3f to %.3f; largest p %.2e; all positive %s' % (
    min(all_rho), max(all_rho), largest_p, all(r > 0 for r in all_rho)))
lowest = min(simulations, key=lambda s: s['rho'])
highest = max(simulations, key=lambda s: s['rho'])
print('   lowest %s, highest %s' % (lowest['label'], highest['label']))
print('equal criteria: ' + ', '.join('%s %.3f' % (criteria_name(s), s['rho']) for s in equal_criteria)
      + '; largest p %.2e' % max(s['pRho'] for s in equal_criteria))
fig2a_runs = [s for s in simulations if robustness_tables.has_fig2a_structure(s)]
print('Fig. 2a design: %d runs, rho %s' % (len(fig2a_runs), ', '.join('%.3f' % s['rho'] for s in fig2a_runs)))
print('   reference cell (first row) rho %.3f' % simulations[0]['rho'])
hits = [s[c] for s in simulations for c in ('hitWithin', 'hitAcross')]
false_alarms = [s[c] for s in simulations for c in ('faWithin', 'faAcross')]
print('hit rates %.3f to %.3f; false alarm rates %.3f to %.3f' % (min(hits), max(hits),
                                                                min(false_alarms), max(false_alarms)))
sim_corr = [s['simCorr'] for s in simulations]
print('summed similarity within vs across: r %.3f to %.3f' % (min(sim_corr), max(sim_corr)))

# number of target items per simulation: categories x targets per category (75, or a quarter of
# the items per category in the Items per Category rows; bylinskii_robustness.m: round(nPerCat / 4)).
# The Spearman p implied by this n reproduces the csv's pRho, which checks n.
largest_p_similarity = 0
for simulation in simulations:
    n_targets = int(simulation['nCat'] * robustness_tables.targets_per_category(simulation))
    t_rho = simulation['rho'] * np.sqrt((n_targets - 2) / (1 - simulation['rho'] ** 2))
    implied_p = 2 * stats.t.sf(abs(t_rho), n_targets - 2)
    assert abs(implied_p / simulation['pRho'] - 1) < 1e-3, simulation['label']
    t_sim = simulation['simCorr'] * np.sqrt((n_targets - 2) / (1 - simulation['simCorr'] ** 2))
    largest_p_similarity = max(largest_p_similarity, 2 * stats.t.sf(abs(t_sim), n_targets - 2))
print('largest p of the summed-similarity correlations (n = targets): %.2e' % largest_p_similarity)

# fixed settings of the grid (read from the .m file)
grid_source = open(os.path.join(paths['code'], 'bylinskii_robustness.m')).read()
reference_line = re.search(r"ref = struct\((.*?)\);", grid_source, re.S).group(1)
print('grid reference struct (bylinskii_robustness.m): ' + ' '.join(reference_line.split()))
