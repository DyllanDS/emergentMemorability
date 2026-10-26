"""
make_figureS15.py: Fig. S15, summed similarity (crowding) in deep-network spaces predicts memorability
in a face database and a scene database.

What the figure shows
    Panel a. For each database (Bainbridge et al. 2013 faces; Isola et al. 2011 scenes) the held-out
             Spearman correlation between an image's summed similarity to every other image in the
             database and its memorability (hit rate minus false alarm rate), computed two ways:
             every network dimension weighted equally, and dimension weights fitted (ridge) on the
             other images of the same database. Each bar is the mean of five held-out fold
             correlations. Each pair has a black dashed segment at the split-half reliability of
             memorability, sign reversed.
    Panel b. The correlation of summed similarity (equal weights, whole sample, not held out) with the
             hit rate (blue) and with the false alarm rate (red) in each database.
    Error bars: 95% confidence intervals (Fisher transformation with the Bonett-Wright standard error,
    2,222 images), from intervals.py. The panel b statement ("All p < 0.001, n = 2,222 images each") is
    built from the computed p values and image counts.
What it plots
    Faces: the saved faces results in data/provided_faces/ (the face database may not be
    redistributed, EXTERNAL_INPUTS.md M4), or the faces recomputed from the database when its archives
    are in external/bainbridge_faces/.
    Scenes: this run (outputs/scenes/). outputs/figure_data/ also lists the values of the saved runs
    (data/provided_faces/, data/paper_run_scenes/) in its paper_run_* columns.
Inputs
    the reports read through read_results.py, and outputs/intervals_figS15.csv (run intervals.py first)
Outputs
    outputs/figures/FigS15_faces_scenes_generality.png (300 dpi; style from shared/python/figure_style.py)
    outputs/figure_data/FigS15a.csv, FigS15b.csv (the plotted values and their intervals, with the
    values of the saved runs beside them)
Run (from the repository root; code/run_module.py runs it)
    MPLBACKEND=Agg python A6a_faces_and_scenes_figS15/code/make_figureS15.py
"""
import csv
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import read_results  # noqa: E402  (also puts shared/python on the path)
import intervals as interval_settings  # noqa: E402
import release_paths  # noqa: E402
import figure_style as ls  # noqa: E402

PATHS = release_paths.module_paths(__file__)
FIGURE_NAME = 'FigS15_faces_scenes_generality'


# ----------------------------------------------------------------------------- reading the files
def read_intervals():
    """{(panel, source, measure): (rho, low, high, p)} from outputs/intervals_figS15.csv"""
    if not os.path.exists(interval_settings.OUTPUT_FILE):
        raise SystemExit('missing %s: run A6a_faces_and_scenes_figS15/code/intervals.py first'
                         % read_results.relative(interval_settings.OUTPUT_FILE))
    table = {}
    with open(interval_settings.OUTPUT_FILE, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            table[(row['panel'], row['source'], row['measure'])] = (float(row['rho']), float(row['ci_low']),
                                                                    float(row['ci_high']), float(row['p']))
    return table


faces_source = read_results.faces_source()     # prints the provided-output message when the faces are provided
SOURCE_OF = {'Faces': faces_source, 'Scenes': 'scenes_rerun'}               # what the figure plots
PAPER_RUN_OF = {'Faces': 'faces_provided', 'Scenes': 'scenes_paper_run'}    # the saved runs in data/
interval_table = read_intervals()

# panel a values: held-out rho (3 decimals, as printed in the report), its interval, and the reliability
equal_weight = {}
fitted_weight = {}
reliability = {}
for database in ('Faces', 'Scenes'):
    source = SOURCE_OF[database]
    equal_weight[database] = interval_table[('a', source, 'corrected recognition, equal weights')]
    fitted_weight[database] = interval_table[('a', source, 'corrected recognition, fitted weights')]
    if database == 'Scenes':
        reliability[database] = read_results.reliability(read_results.scene_reliability_source(),
                                                         'corrected recognition', 'participants')[0]
    else:
        reliability[database] = read_results.reliability(source, 'corrected recognition', 'trials')[0]

# panel b values
hit = {}
false_alarm = {}
for database in ('Faces', 'Scenes'):
    source = SOURCE_OF[database]
    hit[database] = interval_table[('b', source, 'hit rate')]
    false_alarm[database] = interval_table[('b', source, 'false alarm rate')]

# the statement printed in panel b, from the computed values
panel_b_p_values = [hit[database][3] for database in ('Faces', 'Scenes')] + \
                   [false_alarm[database][3] for database in ('Faces', 'Scenes')]
images_per_database = interval_settings.N_SCORED_IMAGES
images_text = '{:,}'.format(images_per_database)
if max(panel_b_p_values) < 0.001:
    panel_b_statement = 'All p < 0.001\nn = %s images each' % images_text
else:
    panel_b_statement = 'Largest p = %.3f\nn = %s images each' % (max(panel_b_p_values), images_text)


# ----------------------------------------------------------------------------- helpers for this figure
def dashed_segment(ax, x_left, x_right, y_value):
    """A black dashed horizontal segment (reliability level) across one pair of bars."""
    ax.plot([x_left, x_right], [y_value, y_value], color='black', linestyle='--', linewidth=0.8,
            zorder=5)


def bar_with_interval(ax, x_position, value_and_interval, color, width):
    value, low, high = value_and_interval[0], value_and_interval[1], value_and_interval[2]
    ax.bar([x_position], [value], width=width, color=color, edgecolor='none', zorder=2)
    ls.error_bars(ax, [x_position], [value], [low], [high])


# ----------------------------------------------------------------------------- the figure
ls.apply_style('bars')
fig, (panel_a, panel_b) = ls.new_figure(n_panels=2, width='double', height_cm=7.4)

# ---- panel a: equal versus fitted weights, against reliability
bar_width = 0.62
group_centres = {'Faces': 1.5, 'Scenes': 3.9}
tick_positions = []
tick_labels = []
for database in ['Faces', 'Scenes']:
    centre = group_centres[database]
    equal_x = centre - 0.5
    fitted_x = centre + 0.5
    bar_with_interval(panel_a, equal_x, equal_weight[database], ls.MEMORY_DARK, bar_width)
    bar_with_interval(panel_a, fitted_x, fitted_weight[database], ls.MEMORY_DARK, bar_width)
    # reliability of memorability, sign reversed so it sits on the same side as the bars
    dashed_segment(panel_a, equal_x - 0.45, fitted_x + 0.45, -reliability[database])
    ls.direct_label(panel_a, centre, -reliability[database], 'Reliability', color=ls.LABEL_GRAY,
                    ha='center', va='top', dx_points=0, dy_points=-3)
    tick_positions += [equal_x, fitted_x]
    tick_labels += ['Equal\nWeights', 'Fitted\nWeights']
    # database name under its pair of tick labels
    panel_a.annotate(database, xy=(centre, 0), xycoords=('data', 'axes fraction'),
                     xytext=(0, -27), textcoords='offset points', ha='center', va='top',
                     fontsize=ls.FONT_SIZE, color='black', annotation_clip=False)

panel_a.set_xticks(tick_positions)
panel_a.set_xticklabels(tick_labels)
panel_a.tick_params(axis='x', length=0)
panel_a.set_xlim(0.2, 5.2)
panel_a.set_ylim(-0.8, 0)
panel_a.set_yticks([-0.8, -0.6, -0.4, -0.2, 0])
panel_a.set_ylabel('Correlation of Summed Similarity\nWith Memorability')
ls.plain_ticks(panel_a)
ls.panel_label(panel_a, 'a')

# ---- panel b: hits fall and false alarms rise with summed similarity
pair_offset = 0.19
pair_width = 0.36
positions = ls.category_axis(panel_b, ['Faces', 'Scenes'])
for position, database in zip(positions, ['Faces', 'Scenes']):
    bar_with_interval(panel_b, position - pair_offset, hit[database], ls.HIT_BLUE, pair_width)
    bar_with_interval(panel_b, position + pair_offset, false_alarm[database], ls.FA_RED, pair_width)
ls.zero_line(panel_b)
# direct labels on the first pair (Faces): "Hits" under the blue bar, "False Alarms" over the red
ls.direct_label(panel_b, positions[0] - pair_offset, hit['Faces'][1], 'Hits', color=ls.HIT_BLUE,
                ha='center', va='top', dx_points=0, dy_points=-3)
ls.direct_label(panel_b, positions[0] + pair_offset, false_alarm['Faces'][2], 'False\nAlarms', color=ls.FA_RED,
                ha='center', va='bottom', dx_points=0, dy_points=3)
panel_b.set_ylim(-0.6, 0.6)
panel_b.set_yticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
panel_b.set_ylabel('Correlation of Summed Similarity\nWith Hit or False Alarm Rate')
ls.stat_text(panel_b, panel_b_statement, corner='upper right')
ls.plain_ticks(panel_b)
ls.panel_label(panel_b, 'b')

ls.save_figure(fig, FIGURE_NAME, PATHS['figures'])

# ----------------------------------------------------------------------------- the plotted numbers
panel_a_rows = []
for database in ('Faces', 'Scenes'):
    for weighting, values in (('equal weights', equal_weight[database]), ('fitted weights', fitted_weight[database])):
        paper_run = interval_table[('a', PAPER_RUN_OF[database], 'corrected recognition, ' + weighting)]
        panel_a_rows.append(dict(database=database, bar=weighting, source=SOURCE_OF[database],
                                 rho_plotted=values[0], ci_low=values[1], ci_high=values[2],
                                 largest_fold_p=values[3], reliability_line=-reliability[database],
                                 paper_run_rho=paper_run[0], paper_run_ci_low=paper_run[1],
                                 paper_run_ci_high=paper_run[2]))
panel_b_rows = []
for database in ('Faces', 'Scenes'):
    for measure, values in (('hit rate', hit[database]), ('false alarm rate', false_alarm[database])):
        paper_run = interval_table[('b', PAPER_RUN_OF[database], measure)]
        panel_b_rows.append(dict(database=database, bar=measure, source=SOURCE_OF[database],
                                 rho_plotted=values[0], ci_low=values[1], ci_high=values[2], p=values[3],
                                 paper_run_rho=paper_run[0], paper_run_ci_low=paper_run[1],
                                 paper_run_ci_high=paper_run[2]))
for file_name, rows in (('FigS15a.csv', panel_a_rows), ('FigS15b.csv', panel_b_rows)):
    path = os.path.join(PATHS['figure_data'], file_name)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print('wrote', read_results.relative(path))

# ----------------------------------------------------------------------------- the plotted values, printed
print('panel a  equal weights   faces %+.3f  scenes %+.3f (saved scene run %+.3f)' % (
    equal_weight['Faces'][0], equal_weight['Scenes'][0], interval_table[('a', 'scenes_paper_run', 'corrected recognition, equal weights')][0]))
print('panel a  fitted weights  faces %+.3f  scenes %+.3f (saved scene run %+.3f)' % (
    fitted_weight['Faces'][0], fitted_weight['Scenes'][0], interval_table[('a', 'scenes_paper_run', 'corrected recognition, fitted weights')][0]))
print('panel a  reliability     faces %.3f   scenes %.3f' % (reliability['Faces'], reliability['Scenes']))
print('panel b  hits            faces %+.3f  scenes %+.3f (saved scene run %+.3f)' % (
    hit['Faces'][0], hit['Scenes'][0], interval_table[('b', 'scenes_paper_run', 'hit rate')][0]))
print('panel b  false alarms    faces %+.3f  scenes %+.3f (saved scene run %+.3f)' % (
    false_alarm['Faces'][0], false_alarm['Scenes'][0], interval_table[('b', 'scenes_paper_run', 'false alarm rate')][0]))
print('panel b  statement: %s' % panel_b_statement.replace('\n', ', '))
