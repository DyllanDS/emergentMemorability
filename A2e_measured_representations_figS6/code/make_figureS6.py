"""
make_figureS6.py: Fig. S6 of the SI Appendix (REM in the measured SPoSE space).

WHAT THE FIGURE SHOWS
    REM written for continuous features (each feature's rarity judged against that dimension's density
    across the concept set) and run in the measured 49-dimensional SPoSE space of 1,600 THINGS concepts,
    at three values of the one added parameter, the retention noise on a stored feature. Gray bars are
    this model. The white outlined bar at the right is REM with its generated features at the Fig. 1
    parameters ("Fig. 1 Values").
    Change in d′ when the same critical item is studied with 16 close neighbors instead of none
    (nearest neighbor held out as the lure). Negative = the same item is remembered worse when its
    list is crowded.
    Error bars: 95% bootstrap confidence intervals across critical items (2,000 resamples, seed
    20261010, from bootstrap_ci.py). The white bar has no interval.

DATA FILES READ
    outputs/continuous_rem_bootstrap_ci.csv     gray bars and intervals (bootstrap_ci.py, from the item
                                                files of continuous_rem_spose.m)
    data/paper_run/continuous_rem_spose.txt     the saved report of the run the paper reports: the gray
                                                bars are checked against it (3 decimals) and the script
                                                stops if they differ
    ../A2d_list_context_figS5/data/paper_run/list_designed.txt
                                                white bar: CELL 2, line "0 -> 16 similar items", d-prime
WRITES
    outputs/figures/FigS6_continuous_feature_rem.png   (300 dpi)
    outputs/figure_data/FigS6.csv                       the plotted numbers

Run (from the repository root; seconds):
    python A2e_measured_representations_figS6/code/make_figureS6.py
"""
import csv
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402
import figure_style as ls  # noqa: E402

paths = release_paths.module_paths(__file__)
BOOTSTRAP_FILE = os.path.join(paths['outputs'], 'continuous_rem_bootstrap_ci.csv')
SAVED_REPORT_FILE = os.path.join(paths['paper_run'], 'continuous_rem_spose.txt')
REM_DESIGNED_FILE = os.path.join(release_paths.find_module('A2d'), 'data', 'paper_run', 'list_designed.txt')
FIGURE_NAME = 'FigS6_continuous_feature_rem'

# The noise values of the gray bars of Fig. S6 (the simulation also runs 0.08).
NOISE_VALUES_SHOWN = ['0.02', '0.03', '0.05']

for needed in (BOOTSTRAP_FILE, SAVED_REPORT_FILE, REM_DESIGNED_FILE):
    if not os.path.exists(needed):
        sys.exit('make_figureS6.py: %s is missing (run continuous_rem_spose.m and bootstrap_ci.py first; the '
                 'white bar needs the data/paper_run folder of module A2d)' % needed)


# ------------------------------------------------------------------ 1. continuous REM in SPoSE
bootstrap_rows = {}
with open(BOOTSTRAP_FILE) as handle:
    for row in csv.DictReader(handle):
        bootstrap_rows[row['noise_fraction_of_sd']] = row

# Check the gray bars against the saved report in data/paper_run/ (3 decimals, as printed there),
# row by noise label.
original_table = {}
with open(SAVED_REPORT_FILE) as handle:
    for line in handle:
        parts = line.split()
        if len(parts) == 8 and re.match(r'^0\.0\d$', parts[0]):
            original_table[parts[0]] = {'designed_d_prime': float(parts[6])}
for noise in NOISE_VALUES_SHOWN:
    rerun_change = float(bootstrap_rows[noise]['designed_d_prime_change'])
    if round(rerun_change, 3) != original_table[noise]['designed_d_prime']:
        sys.exit('make_figureS6.py: this run does not match data/paper_run/continuous_rem_spose.txt at noise %s'
                 % noise)

spose_changes, spose_change_low, spose_change_high = [], [], []
for noise in NOISE_VALUES_SHOWN:
    row = bootstrap_rows[noise]
    spose_changes.append(float(row['designed_d_prime_change']))
    spose_change_low.append(float(row['designed_ci_low']))
    spose_change_high.append(float(row['designed_ci_high']))

# ------------------------------------------------------------------ 2. REM with generated features
generated_space_change = None
with open(REM_DESIGNED_FILE) as handle:
    inside_cell_2 = False
    for line in handle:
        if line.startswith('CELL 2'):
            inside_cell_2 = True
        if inside_cell_2 and '0 -> 16 similar items' in line:
            generated_space_change = float(re.search(r'd-prime\s+([-+]?\d*\.\d+)', line).group(1))
            break
if generated_space_change is None:
    sys.exit('make_figureS6.py: the white-bar value could not be read from %s' % REM_DESIGNED_FILE)

print('Designed-list d′ change, noise 0.02 / 0.03 / 0.05:',
      ['%.3f' % value for value in spose_changes], ' REM with generated features: %.3f' % generated_space_change)


# ------------------------------------------------------------------ 3. the plotted numbers
def write_panel_data(file_name, measure, spose_values, low_ends, high_ends, generated_value, generated_source):
    """One row per bar: the three gray bars (with their interval) and the white bar (no interval)."""
    path = os.path.join(paths['figure_data'], file_name)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['bar', 'space', 'noise_fraction_of_sd', 'measure', 'value', 'ci_low', 'ci_high',
                         'interval', 'source'])
        for index, noise in enumerate(NOISE_VALUES_SHOWN):
            writer.writerow(['gray', 'measured SPoSE space', noise, measure, repr(spose_values[index]),
                             repr(low_ends[index]), repr(high_ends[index]),
                             '95% percentile bootstrap, 2,000 resamples, seed 20261010',
                             'outputs/continuous_rem_bootstrap_ci.csv'])
        writer.writerow(['white', 'generated features (Fig. 1 values)', '', measure, repr(generated_value), '', '',
                         'none', generated_source])
    print('wrote', os.path.relpath(path, release_paths.release_root()))


write_panel_data('FigS6.csv', 'change in d′, 16 vs 0 close neighbors', spose_changes,
                 spose_change_low, spose_change_high, generated_space_change,
                 'A2d_list_context_figS5/data/paper_run/list_designed.txt (CELL 2, 0 -> 16 similar items)')


# ------------------------------------------------------------------ 4. draw
ls.apply_style('bars')
figure, axes = ls.new_figure(n_panels=1, width='single', height_cm=6.6)
axis = axes[0]

category_names = ['Noise\n0.02', 'Noise\n0.03', 'Noise\n0.05', 'Generated\nFeatures']
positions = ls.category_axis(axis, category_names)
bar_width = 0.6

# Gray bars: continuous-feature REM in the measured SPoSE space, one per noise value.
axis.bar(positions[:3], spose_changes, width=bar_width, color=ls.MODEL_GRAY,
         edgecolor='none', zorder=2)
ls.error_bars(axis, positions[:3], spose_changes, spose_change_low, spose_change_high)

# White outlined bar: REM with its generated features, at the Fig. 1 parameters.
axis.bar([positions[3]], [generated_space_change], width=bar_width, color='white',
         edgecolor='black', linewidth=0.8, zorder=2)

ls.zero_line(axis)
axis.set_ylim(-0.7, 0.15)
axis.set_yticks([-0.6, -0.4, -0.2, 0])
axis.set_ylabel("Change in d′\n(16 vs 0 Close Neighbors)")
ls.plain_ticks(axis)

# Group label over the three gray bars, so the two spaces are named on the plot.
axis.text(positions[1], 0.075, 'Measured SPoSE Space', ha='center', va='center',
          fontsize=ls.FONT_SIZE, color=ls.LABEL_GRAY)
axis.plot([positions[0] - 0.3, positions[2] + 0.3], [0.03, 0.03], color=ls.LABEL_GRAY,
          linewidth=0.6, zorder=2)
# Gray label over the outlined bar: REM at the parameter values of Fig. 1.
axis.text(positions[3], 0.075, 'Fig. 1 Values', ha='center', va='center',
          fontsize=ls.FONT_SIZE, color=ls.LABEL_GRAY)

ls.save_figure(figure, FIGURE_NAME, paths['figures'])
