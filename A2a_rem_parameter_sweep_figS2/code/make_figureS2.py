"""
make_figureS2.py: SI Appendix, Fig. S2, the REM parameter sweep (SI Appendix, Section 2).

What the figure shows
---------------------
Two panels, both about the item-level relationship in the REM simulation: the Spearman correlation,
across 1,600 simulated items, between each item's memorability (hit rate minus false alarm rate) and
its summed similarity to the other items (negated mean Euclidean distance). Negative = distinctive
items are better remembered.

  (a) Every one of the 32 simulations of the sweep as one dot, grouped by the parameter that was
      varied. All dots sit below the dashed zero line. The two large black dots are the standard
      Shiffrin and Steyvers (1997) values ("Standard") and the sweep's cell with the parameter
      values of Fig. 1 ("Fig. 1 Values": g 0.65, 49 features, copy 0.60; a cell of this sweep, not
      the Fig. 1 run of module 01).
  (b) The same 32 correlations against overall d′. The relationship is strongest when accuracy is
      low and weakest near ceiling; gray line = least-squares fit. "Near Ceiling" marks the cell
      g 0.30 with 49 features (the most accurate one).

The feature-count sweep (5 to 120 features) and the list-length rows are reported in the text, not
drawn; this script also prints their numbers.

Data (this module)
------------------
  outputs/rem_parameter_sweep.csv        this run of rem_parameter_sweep.m (default), or
  data/paper_run/rem_parameter_sweep.csv the saved run the paper reports (--paper-run)
  outputs/rem_parameter_sweep_items.csv  per-item counts; the p values of this run are computed from
                                         them (the formula of the pItem column)
  outputs/accuracy_link.json             r(d′, rho) from accuracy_link.py (accuracy_link_paper_run.json
                                         with --paper-run)
  outputs/sweep_rem.csv                  the feature-count sweep, when it has been run (else, and with
                                         --paper-run, data/paper_run/sweep_rem.csv)

Outputs
-------
  outputs/figures/FigS2_rem_parameter_sweep.png (300 dpi)
  outputs/figure_data/FigS2a.csv, FigS2b.csv, FigS2b_fit_line.csv (the plotted numbers)
  (with --paper-run the same names end in _paper_run)

Run (from the repository root):
    python A2a_rem_parameter_sweep_figS2/code/make_figureS2.py
    python A2a_rem_parameter_sweep_figS2/code/make_figureS2.py --paper-run
"""
import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402
import figure_style as ls  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_tables  # noqa: E402  (this module's readers of the sweep's tables)

PAPER_FEATURE_COUNTS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 49, 55, 60, 66, 70, 80, 90, 100, 120]
PAPER_SEEDS = [12345, 777, 20260731]

parser = argparse.ArgumentParser(description='Fig. S2: the REM parameter sweep.')
parser.add_argument('--paper-run', action='store_true',
                    help='draw the saved run in data/paper_run/ instead of this run in outputs/')
arguments = parser.parse_args()
paths = release_paths.module_paths(__file__)
suffix = '_paper_run' if arguments.paper_run else ''

# ============================================================== 1. read the 32 simulations
if arguments.paper_run:
    sweep_file = os.path.join(paths['paper_run'], 'rem_parameter_sweep.csv')
    accuracy_file = os.path.join(paths['outputs'], 'accuracy_link_paper_run.json')
    print('drawing the saved run: data/paper_run/rem_parameter_sweep.csv')
else:
    sweep_file = os.path.join(paths['outputs'], 'rem_parameter_sweep.csv')
    accuracy_file = os.path.join(paths['outputs'], 'accuracy_link.json')
    print('drawing this run: outputs/rem_parameter_sweep.csv')
for needed_file, how in ((sweep_file, 'run code/rem_parameter_sweep.m'),
                         (accuracy_file, 'run code/accuracy_link.py' + (' --paper-run' if arguments.paper_run else ''))):
    if not os.path.exists(needed_file):
        sys.exit('make_figureS2.py: %s is missing; %s first' % (os.path.relpath(needed_file, paths['module']), how))
cells = sweep_tables.read_cells(sweep_file)
assert len(cells) == 32

# The p of every cell's rho (t approximation with n - 2 degrees of freedom). The saved run holds them in
# its pItem column. For a new run they are computed here from the per-item file with the same formula,
# so the figure does not depend on which tcdf function MATLAB used (a tcdf from another toolbox on the
# MATLAB path, such as FieldTrip's, writes 0 for p values below about 1e-16).
if arguments.paper_run:
    for cell in cells:
        cell['p'] = cell['pItem']
else:
    items_by_cell = sweep_tables.read_items(os.path.join(paths['outputs'], 'rem_parameter_sweep_items.csv'))
    for cell_index, cell in enumerate(cells):
        rho, p_value = sweep_tables.spearman_rho_and_p(items_by_cell[cell_index + 1]['memorability'],
                                                       items_by_cell[cell_index + 1]['negated_mean_distance'])
        assert abs(rho - cell['rhoItem']) < 1e-12
        cell['p'] = p_value


def cells_whose_label_starts_with(prefix):
    """All simulations whose label in the csv starts with the given text."""
    found = []
    for cell in cells:
        if cell['label'].startswith(prefix):
            found.append(cell)
    return found


standard_cell = cells_whose_label_starts_with('Shiffrin & Steyvers reference')[0]
fig1_values_cell = cells_whose_label_starts_with('paper values')[0]
near_ceiling_cell = cells_whose_label_starts_with('grid g=0.30 w=49')[0]
assert near_ceiling_cell is max(cells, key=lambda c: c['dprime']), 'the g 0.30, 49-feature cell is the most accurate'

# Groups in the order they appear on the x-axis of panel a. Inside a group the dots are ordered
# by the value of the parameter that was varied (the g x w grid: by number of features, then g).
grid_cells = sorted(cells_whose_label_starts_with('grid '), key=lambda c: (c['w'], c['g']))
groups = [
    ('Reference\nValues', [standard_cell, fig1_values_cell]),
    ('g and\nFeatures', grid_cells),
    ('Storage', sorted(cells_whose_label_starts_with('storage'), key=lambda c: c['u'])),
    ('Copy\nFidelity', sorted(cells_whose_label_starts_with('copy'), key=lambda c: c['c'])),
    ('List\nLength', sorted(cells_whose_label_starts_with('list length'), key=lambda c: c['nStudy'])),
    ('Criterion', sorted(cells_whose_label_starts_with('criterion'), key=lambda c: c['crit'])),
]
number_in_groups = 0
for name, members in groups:
    number_in_groups += len(members)
assert number_in_groups == 32, 'every simulation must appear exactly once'
assert len(grid_cells) == 15

# r(d′, rho) saved by the analysis script
with open(accuracy_file) as handle:
    accuracy_link = json.load(handle)
all_dprime = [cell['dprime'] for cell in cells]
all_rho = [cell['rhoItem'] for cell in cells]
r_all, p_all = stats.pearsonr(all_dprime, all_rho)
assert abs(r_all - accuracy_link['r_dprime_rho_all32']) < 1e-12


# ============================================================== 2. draw
ls.apply_style('model')            # REM simulation: box and light grid, as in Fig. 1
figure_width_in = ls.WIDTHS_CM['double'] * ls.CM
figure, axes = plt.subplots(1, 2, figsize=(figure_width_in, 2.75), layout='constrained',
                            gridspec_kw={'width_ratios': [1.85, 1]})
figure.get_layout_engine().set(w_pad=0.06, h_pad=0.06, wspace=0.08)
axis_a, axis_b = axes

Y_LABEL = 'Correlation of Memorability\nWith Summed Similarity'
Y_LIMITS = (-0.8, 0.4)
Y_TICKS = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4]

# ---- panel a: one dot per simulation, grouped by the parameter varied
SPACING_BETWEEN_DOTS = 0.10           # x distance between neighboring dots inside a group
SPACING_REFERENCE_PAIR = 0.5          # the two black reference dots sit further apart
GAP_BETWEEN_GROUPS = 1.15             # x distance from the last dot of one group to the first of the next
group_centers = []
group_names = []
panel_a_rows = []
x_position = 1.0
previous_width = 0.0
for group_index, (name, members) in enumerate(groups):
    spacing = SPACING_REFERENCE_PAIR if group_index == 0 else SPACING_BETWEEN_DOTS
    group_width = spacing * (len(members) - 1)
    if group_index > 0:
        x_position = x_position + previous_width / 2 + GAP_BETWEEN_GROUPS + group_width / 2
    previous_width = group_width
    group_centers.append(x_position)
    group_names.append(name)
    for member_index, cell in enumerate(members):
        dot_x = x_position - group_width / 2 + member_index * spacing
        if cell is standard_cell:
            ls.highlight_point(axis_a, dot_x, cell['rhoItem'], label=None)
            ls.direct_label(axis_a, dot_x, cell['rhoItem'], 'Standard', color=ls.LABEL_GRAY,
                            ha='center', va='bottom', dx_points=0, dy_points=7)
            marker = 'black, labeled Standard'
        elif cell is fig1_values_cell:
            ls.highlight_point(axis_a, dot_x, cell['rhoItem'], label=None)
            ls.direct_label(axis_a, dot_x, cell['rhoItem'], 'Fig. 1 Values', color=ls.LABEL_GRAY,
                            ha='center', va='top', dx_points=0, dy_points=-7)
            marker = 'black, labeled Fig. 1 Values'
        else:
            axis_a.plot([dot_x], [cell['rhoItem']], linestyle='none', marker='o',
                        markersize=ls.MARKER_SIZE, markerfacecolor=ls.DATA_BLUE,
                        markeredgecolor=ls.DATA_BLUE, alpha=0.85, zorder=3)
            marker = 'blue'
        panel_a_rows.append({'cell_number': cells.index(cell) + 1, 'label': cell['label'],
                             'group': name.replace('\n', ' '), 'x_position': dot_x, 'rho_item': cell['rhoItem'],
                             'p': cell['p'], 'marker': marker})
ls.zero_line(axis_a)
axis_a.set_xticks(group_centers)
axis_a.set_xticklabels(group_names)
axis_a.tick_params(axis='x', length=0)
axis_a.set_xlim(group_centers[0] - 1.0, group_centers[-1] + 0.75)
axis_a.grid(False, axis='x')
axis_a.set_ylim(Y_LIMITS)
axis_a.set_yticks(Y_TICKS)
axis_a.set_ylabel(Y_LABEL)
ls.plain_ticks(axis_a)
largest_p_item = max(cell['p'] for cell in cells)
number_below_zero = sum(1 for cell in cells if cell['rhoItem'] < 0)
if number_below_zero == len(cells):
    count_text = 'All %d simulations below zero' % len(cells)
else:
    count_text = '%d of %d simulations below zero' % (number_below_zero, len(cells))
ls.stat_text(axis_a, '%s\n%s' % (count_text, ls.format_p(largest_p_item).replace('p', 'all p', 1)),
             corner='upper left')
ls.panel_label(axis_a, 'a')

# ---- panel b: correlation against overall d′
for cell in cells:
    if cell is standard_cell or cell is fig1_values_cell:
        continue
    ls.scatter_points(axis_b, [cell['dprime']], [cell['rhoItem']], color=ls.DATA_BLUE)
axis_b.set_xlim(0.5, 4)
fit = ls.fit_line(axis_b, all_dprime, all_rho, over='data')
ls.highlight_point(axis_b, standard_cell['dprime'], standard_cell['rhoItem'], label=None)
ls.highlight_point(axis_b, fig1_values_cell['dprime'], fig1_values_cell['rhoItem'], label=None)
# gray labels for the two black points, pulled away from the cloud with thin gray leader lines
axis_b.annotate('Standard', xy=(standard_cell['dprime'], standard_cell['rhoItem']),
                xytext=(1.0, -0.3), textcoords='data', ha='center', va='bottom',
                fontsize=ls.FONT_SIZE, color=ls.LABEL_GRAY,
                arrowprops=dict(arrowstyle='-', color=ls.LABEL_GRAY, linewidth=0.6, shrinkB=4))
axis_b.annotate('Fig. 1 Values', xy=(fig1_values_cell['dprime'], fig1_values_cell['rhoItem']),
                xytext=(1.6, -0.76), textcoords='data', ha='left', va='bottom',
                fontsize=ls.FONT_SIZE, color=ls.LABEL_GRAY,
                arrowprops=dict(arrowstyle='-', color=ls.LABEL_GRAY, linewidth=0.6, shrinkB=4))
ls.direct_label(axis_b, near_ceiling_cell['dprime'], near_ceiling_cell['rhoItem'], 'Near\nCeiling',
                color=ls.LABEL_GRAY, ha='right', va='bottom', dx_points=-1, dy_points=5)
ls.zero_line(axis_b)
axis_b.set_ylim(Y_LIMITS)
axis_b.set_yticks(Y_TICKS)
axis_b.set_xticks([1, 2, 3, 4])
axis_b.set_xlabel('Overall d′')
axis_b.set_ylabel(Y_LABEL)
ls.plain_ticks(axis_b)
ls.stat_text(axis_b, ls.format_r_p(r_all, p_all, n=len(cells), unit='simulations'), corner='upper left')
ls.panel_label(axis_b, 'b')

ls.save_figure(figure, 'FigS2_rem_parameter_sweep' + suffix, paths['figures'])

# ---- the plotted numbers
with open(os.path.join(paths['figure_data'], 'FigS2a%s.csv' % suffix), 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=['cell_number', 'label', 'group', 'x_position', 'rho_item', 'p',
                                                'marker'])
    writer.writeheader()
    for row in panel_a_rows:
        writer.writerow(row)
with open(os.path.join(paths['figure_data'], 'FigS2b%s.csv' % suffix), 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=['cell_number', 'label', 'dprime', 'rho_item', 'marker', 'text_label'])
    writer.writeheader()
    for cell_index, cell in enumerate(cells):
        if cell is standard_cell:
            marker, text_label = 'black', 'Standard'
        elif cell is fig1_values_cell:
            marker, text_label = 'black', 'Fig. 1 Values'
        elif cell is near_ceiling_cell:
            marker, text_label = 'blue', 'Near Ceiling'
        else:
            marker, text_label = 'blue', ''
        writer.writerow({'cell_number': cell_index + 1, 'label': cell['label'], 'dprime': cell['dprime'],
                         'rho_item': cell['rhoItem'], 'marker': marker, 'text_label': text_label})
line_x, line_y = fit.get_data()
slope, intercept = np.polyfit(all_dprime, all_rho, 1)
with open(os.path.join(paths['figure_data'], 'FigS2b_fit_line%s.csv' % suffix), 'w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['end', 'dprime', 'rho_item', 'slope', 'intercept', 'pearson_r', 'p', 'n_cells'])
    writer.writerow(['left', line_x[0], line_y[0], slope, intercept, r_all, p_all, len(cells)])
    writer.writerow(['right', line_x[1], line_y[1], slope, intercept, r_all, p_all, len(cells)])
print('wrote outputs/figure_data/FigS2a%s.csv, FigS2b%s.csv, FigS2b_fit_line%s.csv' % (suffix, suffix, suffix))


# ============================================================== 3. print every number used
print('\n==== numbers of Fig. S2 and SI Appendix, Section 2 ====')
print('item rho over all 32: %.3f to %.3f; largest p %.2e; all negative: %s' % (
    min(all_rho), max(all_rho), largest_p_item, all(r < 0 for r in all_rho)))
print('standard cell (row "%s"): item rho %.3f, feature r %.3f, d′ %.2f, hit %.3f' % (
    standard_cell['label'], standard_cell['rhoItem'], standard_cell['rFeat'], standard_cell['dprime'],
    standard_cell['meanHit']))
print('six runs at the standard parameters: item rho %.3f to %.3f' % (
    accuracy_link['standard_item_min'], accuracy_link['standard_item_max']))
print('Fig. 1 values cell: item rho %.3f, feature r %.3f, d′ %.2f; rank %d of 32 from most negative' % (
    fig1_values_cell['rhoItem'], fig1_values_cell['rFeat'], fig1_values_cell['dprime'],
    accuracy_link['fig1_values_rank_most_negative_of_32']))
print("r(d′, rho): all 32 %+.3f (p %.2e)" % (r_all, p_all))
print("d′ %.2f to %.2f; hit rate %.3f to %.3f" % (min(all_dprime), max(all_dprime),
      min(c['meanHit'] for c in cells), max(c['meanHit'] for c in cells)))
print('Near Ceiling cell %s: d′ %.2f hit %.3f rho %.3f' % (near_ceiling_cell['label'], near_ceiling_cell['dprime'],
      near_ceiling_cell['meanHit'], near_ceiling_cell['rhoItem']))
feature_level = [cell['rFeat'] for cell in cells]
print('feature-level r over 32: %.3f to %.3f; p < 0.05 in %d of 32' % (
    min(feature_level), max(feature_level), sum(1 for c in cells if c['pFeat'] < 0.05)))
print('one-at-a-time sweeps (storage, copy, list, criterion): item rho %.3f to %.3f' % (
    min(c['rhoItem'] for g in groups[2:6] for c in g[1]), max(c['rhoItem'] for g in groups[2:6] for c in g[1])))
print('list length rows:')
for cell in groups[4][1]:
    print('   list %d: rho %.3f  (rounded %.2f)' % (cell['nStudy'], cell['rhoItem'], cell['rhoItem']))

# feature-count sweep (text only): this run when it holds the paper's 54 runs, else the saved run
fresh_feature_file = os.path.join(paths['outputs'], 'sweep_rem.csv')
feature_file = os.path.join(paths['paper_run'], 'sweep_rem.csv')
if not arguments.paper_run and os.path.exists(fresh_feature_file):
    with open(fresh_feature_file) as handle:
        runs_present = sorted((int(row['w']), int(row['seed'])) for row in csv.DictReader(handle))
    if runs_present == sorted((count, seed) for count in PAPER_FEATURE_COUNTS for seed in PAPER_SEEDS):
        feature_file = fresh_feature_file
feature_rows = []
with open(feature_file) as handle:
    for row in csv.DictReader(handle):
        feature_rows.append({'w': int(row['w']), 'seed': int(row['seed']),
                             'rho': float(row['r_sim_spearman']), 'p': float(row['p_sim_spearman']),
                             'r_meta': float(row['r_meta'])})
feature_counts = sorted(set(r['w'] for r in feature_rows))
print('feature-count sweep (%s): %d runs, %d counts %d to %d; rho %.3f to %.3f; largest p %.1e; all negative %s' % (
    os.path.relpath(feature_file, paths['module']), len(feature_rows), len(feature_counts), feature_counts[0],
    feature_counts[-1], min(r['rho'] for r in feature_rows), max(r['rho'] for r in feature_rows),
    max(r['p'] for r in feature_rows), all(r['rho'] < 0 for r in feature_rows)))
