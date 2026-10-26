"""
make_figure4.py: draws Fig. 4a and 4b and SI Appendix, Fig. S11a (the rare-feature face study).

Every plotted number is read from the outputs of memorability_check.py, and panel a of Fig. 4 is
assembled from the five synthetic faces in data/fig4a_faces/.

FIGURES
  Fig4ab_rare_feature_faces.png: Fig. 4, panels a and b. Panel c (the confidence ROCs) is drawn by
  module A3c_face_study_model_and_sdt_figS10_S11; this page keeps its place free, right of panel b.
    a  The design, drawn with five of the experiment's synthetic faces: three study faces (ordinary,
       rare feature, ordinary), the arithmetic filler task, and three test faces (the studied face
       carrying the rare feature, a new face carrying the same feature, a new face without the
       feature).
    b  Proportion called old for the four kinds of test item (studied faces blue, new faces red),
       mean over the 100 registered participants, with 95% bootstrap intervals over participants.
  FigS11a_memorability_by_condition.png: SI Appendix, Fig. S11, panel a.
    a  Memorability (hit rate minus false alarm rate) in the three conditions, with 95% bootstrap
       intervals over participants, and braces for the two paired comparisons (*** p < 0.001,
       * p < 0.05; the stars are set from the p values in figS11a_comparisons.csv).

INPUTS   outputs/figure_data/fig4b_proportion_called_old.csv, figS11a_memorability.csv,
         figS11a_comparisons.csv (written by memorability_check.py); data/fig4a_faces/*.jpg;
         data/face_design.json (the role of each face of panel a in the stimulus manifest)
OUTPUTS  outputs/figures/Fig4ab_rare_feature_faces.png (300 dpi, 17.8 x 12.7 cm; 17.8 cm is the PNAS
         two-column width)
         outputs/figures/FigS11a_memorability_by_condition.png (300 dpi, 8.7 x 6.7 cm; one PNAS column)
         outputs/figure_data/fig4a_design_faces.csv (which face is shown where, with its label)
SEEDS    none
RUN      python 04_rare_feature_faces_fig4/code/make_figure4.py   (from the repository root)
"""
import sys
sys.dont_write_bytecode = True
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths
import figure_style as ls

PATHS = release_paths.module_paths(__file__)
FACES_FOLDER = os.path.join(PATHS['data'], 'fig4a_faces')

# The colors of Fig. 4b: studied faces (hits) blue, new faces (false alarms) red. The memorability
# points of SI Appendix, Fig. S11a are blue too.
STUDIED_BLUE = '#2f5c9e'
NEW_RED = '#d5322f'
FIGURE_WIDTH_CM = 17.8          # PNAS two-column width
FIGURE_HEIGHT_CM = 12.7
# SI Appendix, Fig. S11a: one panel on a page one PNAS column wide. Its axes have the size of the
# Fig. 4b axes (0.355 of the Fig. 4 page in each direction), so its text and points are drawn at
# the same size as those of Fig. 4.
S11A_WIDTH_CM = 8.7
S11A_HEIGHT_CM = 6.7
S11A_AXES_LEFT_CM = 1.67
S11A_AXES_BOTTOM_CM = 1.33
S11A_AXES_WIDTH_CM = 0.355 * FIGURE_WIDTH_CM
S11A_AXES_HEIGHT_CM = 0.355 * FIGURE_HEIGHT_CM
S11A_LETTER_LEFT_CM = 0.78
S11A_LETTER_TOP_CM = 6.41
TICK_LABEL_SIZE = 8
CATEGORY_LABEL_SIZE = 8
# Bold text (the stage titles and the arithmetic example): matplotlib reads only the regular weight
# of the Helvetica font file on macOS, so bold text is set in Arial (same letter shapes), with
# DejaVu Sans (shipped with matplotlib) where Arial is missing.
BOLD_FONTS = ['Arial', 'DejaVu Sans']


def read_rows(file_name):
    with open(os.path.join(PATHS['figure_data'], file_name), newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


# ----------------------------------------------------------------------------- the plotted numbers
rate_rows = read_rows('fig4b_proportion_called_old.csv')
memorability_rows = read_rows('figS11a_memorability.csv')
comparison_rows = read_rows('figS11a_comparisons.csv')

# category labels as printed under the bars and points (line breaks only; the numbers come from the CSVs)
RATE_TICK_LABELS = {'studied_rare_feature': 'Studied,\nrare feature', 'studied_ordinary': 'Studied,\nordinary',
                    'new_with_feature': 'New,\nwith feature', 'new_without': 'New,\nwithout'}
MEMORABILITY_TICK_LABELS = {'rare_feature_not_carried': 'Rare feature,\nnot carried',
                            'rare_feature_carried_by_new_face': 'Rare feature,\ncarried by\na new face',
                            'ordinary_features': 'Ordinary\nfeatures'}

# The five faces of panel a: (horizontal center, file, label).
STUDY_FACES = [(0.115, 'filler_006.jpg', 'Ordinary\nfeatures'),
               (0.228, 'rare_mohawk_c1_target.jpg', 'Rare feature'),
               (0.341, 'filler_009.jpg', 'Ordinary\nfeatures')]
TEST_FACES = [(0.695, 'rare_mohawk_c1_target.jpg', 'The studied face\ncarrying the rare\nfeature'),
              (0.812, 'rare_mohawk_c1_foil.jpg', 'A new face\ncarrying the same\nfeature'),
              (0.929, 'filler_002.jpg', 'A new face\nwithout the\nfeature')]
FILLER_EXAMPLE = '47 + 68 = ?'      # the example problem drawn in Fig. 4a for the 60 s arithmetic task

# ----------------------------------------------------------------------------- style and page
ls.apply_style('data')
matplotlib.rcParams.update({'xtick.direction': 'out', 'ytick.direction': 'out',
                            'xtick.major.size': 3, 'ytick.major.size': 3})
figure = plt.figure(figsize=(FIGURE_WIDTH_CM * ls.CM, FIGURE_HEIGHT_CM * ls.CM))

# ----------------------------------------------------------------------------- panel a: the design
design_axes = figure.add_axes([0.0, 0.58, 1.0, 0.40])
design_axes.set_xlim(0, 1)
design_axes.set_ylim(0, 1)
design_axes.axis('off')

for left, width, title in ((0.055, 0.345, 'Memorization'), (0.425, 0.165, 'Filler Task'),
                           (0.645, 0.330, 'Memory Test')):
    design_axes.add_patch(FancyBboxPatch((left, 0.80), width, 0.13, boxstyle='round,pad=0,rounding_size=0.02',
                                         facecolor='#f2f2f2', edgecolor='0.35', linewidth=0.8,
                                         transform=design_axes.transAxes))
    design_axes.text(left + width / 2, 0.865, title, ha='center', va='center', fontsize=ls.FONT_SIZE,
                     weight='bold', family=BOLD_FONTS)

FACE_WIDTH = 0.085        # in panel-a axes units (the panel is 1 wide and 1 high)
FACE_BOTTOM = 0.36
# keep each face square on the page: height in axes units = width x page width / panel height
face_height = FACE_WIDTH * figure.get_figwidth() / (figure.get_figheight() * 0.40)

panel_a_rows = []
for stage, faces in (('study', STUDY_FACES), ('test', TEST_FACES)):
    for centre, file_name, label in faces:
        image = mpimg.imread(os.path.join(FACES_FOLDER, file_name))
        design_axes.imshow(image, extent=(centre - FACE_WIDTH / 2, centre + FACE_WIDTH / 2,
                                          FACE_BOTTOM, FACE_BOTTOM + face_height), zorder=2, aspect='auto')
        design_axes.text(centre, FACE_BOTTOM - 0.04, label, ha='center', va='top', fontsize=7, linespacing=1.35)
        panel_a_rows.append((stage, centre, file_name, label.replace('\n', ' ')))
design_axes.text(0.5075, 0.60, FILLER_EXAMPLE, ha='center', va='center', fontsize=11, weight='bold',
                 family=BOLD_FONTS)

# ----------------------------------------------------------------------------- panel b: proportion called old
bar_axes = figure.add_axes([0.085, 0.105, 0.355, 0.355])
bar_positions = np.arange(len(rate_rows))
bar_means = np.array([float(row['mean']) for row in rate_rows])
bar_lows = np.array([float(row['ci_low']) for row in rate_rows])
bar_highs = np.array([float(row['ci_high']) for row in rate_rows])
bar_colours = []
for row in rate_rows:
    if row['bar_group'] == 'studied':
        bar_colours.append(STUDIED_BLUE)
    else:
        bar_colours.append(NEW_RED)
bar_axes.bar(bar_positions, bar_means, width=0.68, color=bar_colours, zorder=3)
bar_axes.errorbar(bar_positions, bar_means, yerr=[bar_means - bar_lows, bar_highs - bar_means], fmt='none',
                  ecolor='black', elinewidth=1.1, capsize=3.2, capthick=1.1, zorder=4)
bar_axes.set_xlim(-0.65, len(rate_rows) - 0.35)
bar_axes.set_ylim(0, 1.0)
bar_axes.set_yticks([0, 0.5, 1.0])
bar_axes.set_xticks(bar_positions)
bar_axes.set_xticklabels([RATE_TICK_LABELS[row['condition']] for row in rate_rows], fontsize=CATEGORY_LABEL_SIZE,
                         linespacing=1.3)
bar_axes.tick_params(axis='x', length=0)
bar_axes.tick_params(axis='y', labelsize=TICK_LABEL_SIZE)
bar_axes.set_ylabel('Proportion Called Old', fontsize=ls.FONT_SIZE)
ls.plain_ticks(bar_axes)

# ----------------------------------------------------------------------------- SI Appendix, Fig. S11a: memorability
memorability_figure = plt.figure(figsize=(S11A_WIDTH_CM * ls.CM, S11A_HEIGHT_CM * ls.CM))
point_axes = memorability_figure.add_axes([S11A_AXES_LEFT_CM / S11A_WIDTH_CM, S11A_AXES_BOTTOM_CM / S11A_HEIGHT_CM,
                                           S11A_AXES_WIDTH_CM / S11A_WIDTH_CM, S11A_AXES_HEIGHT_CM / S11A_HEIGHT_CM])
point_positions = np.arange(len(memorability_rows))
point_means = np.array([float(row['mean']) for row in memorability_rows])
point_lows = np.array([float(row['ci_low']) for row in memorability_rows])
point_highs = np.array([float(row['ci_high']) for row in memorability_rows])
point_axes.errorbar(point_positions, point_means, yerr=[point_means - point_lows, point_highs - point_means],
                    fmt='o', color=STUDIED_BLUE, ecolor='black', elinewidth=1.1, capsize=3.2, capthick=1.1,
                    markersize=6.5, zorder=3)
point_axes.set_xlim(-0.55, len(memorability_rows) - 0.45)
point_axes.set_ylim(0, 0.80)
point_axes.set_yticks([0, 0.25, 0.5, 0.75])
point_axes.set_xticks(point_positions)
point_axes.set_xticklabels([MEMORABILITY_TICK_LABELS[row['condition']] for row in memorability_rows],
                           fontsize=CATEGORY_LABEL_SIZE, linespacing=1.3)
point_axes.tick_params(axis='x', length=0)
point_axes.tick_params(axis='y', labelsize=TICK_LABEL_SIZE)
point_axes.set_ylabel('Memorability', fontsize=ls.FONT_SIZE)
ls.plain_ticks(point_axes)


def draw_brace(axes, left_x, right_x, height, label, depth=0.028):
    """A brace from left_x to right_x at the given height, nub up, with its label above it."""
    middle_x = (left_x + right_x) / 2
    axes.plot([left_x, left_x, middle_x, middle_x, middle_x, right_x, right_x],
              [height - depth, height, height, height + depth, height, height, height - depth],
              color='black', linewidth=1.0, clip_on=False, zorder=4, solid_joinstyle='miter')
    axes.text(middle_x, height + depth + 0.015, label, ha='center', va='bottom', fontsize=12)


# Each brace sits 0.055 above the higher of its two error bars, and a brace never sits lower than
# 0.11 above the brace before it (so the stars of the first brace stay clear of the second).
previous_brace_height = None
for comparison in comparison_rows:
    left_index = int(comparison['from_position']) - 1
    right_index = int(comparison['to_position']) - 1
    brace_height = max(point_highs[left_index], point_highs[right_index]) + 0.055
    if previous_brace_height is not None:
        brace_height = max(brace_height, previous_brace_height + 0.11)
    draw_brace(point_axes, point_positions[left_index], point_positions[right_index], brace_height,
               comparison['stars'])
    previous_brace_height = brace_height

# ----------------------------------------------------------------------------- panel letters and save
figure.text(0.012, 0.975, 'a', fontsize=ls.PANEL_LETTER_SIZE, va='top', ha='left')
figure.text(0.035, 0.505, 'b', fontsize=ls.PANEL_LETTER_SIZE, va='top', ha='left')
memorability_figure.text(S11A_LETTER_LEFT_CM / S11A_WIDTH_CM, S11A_LETTER_TOP_CM / S11A_HEIGHT_CM, 'a',
                         fontsize=ls.PANEL_LETTER_SIZE, va='top', ha='left')

with open(os.path.join(PATHS['data'], 'face_design.json'), encoding='utf-8') as handle:
    manifest = json.load(handle)['stimulus_manifest']
role_of_file = {}
for file_name in manifest['practice']:
    role_of_file[file_name] = 'practice face (study practice only)'
for file_name in manifest['ordinary']:
    role_of_file[file_name] = 'ordinary face'
for pair in manifest['pairs']:
    role_of_file[pair['target']] = 'rare-feature pair %s (%s), member named target' % (pair['cell'], pair['feature'])
    role_of_file[pair['foil']] = 'rare-feature pair %s (%s), member named foil' % (pair['cell'], pair['feature'])
with open(os.path.join(PATHS['figure_data'], 'fig4a_design_faces.csv'), 'w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['stage', 'horizontal_position', 'file', 'label', 'role_in_stimulus_manifest'])
    for stage, centre, file_name, label in panel_a_rows:
        writer.writerow([stage, centre, 'data/fig4a_faces/' + file_name, label, role_of_file.get(file_name, '')])

ls.save_figure(figure, 'Fig4ab_rare_feature_faces', PATHS['figures'])
plt.close(figure)
ls.save_figure(memorability_figure, 'FigS11a_memorability_by_condition', PATHS['figures'])
plt.close(memorability_figure)
for row in rate_rows:
    print('  4b %-24s %.3f [%.3f, %.3f]' % (row['label'], float(row['mean']), float(row['ci_low']), float(row['ci_high'])))
for row in memorability_rows:
    print('  S11a %-38s %.3f [%.3f, %.3f]' % (row['label'], float(row['mean']), float(row['ci_low']), float(row['ci_high'])))
for comparison in comparison_rows:
    print('  brace %-10s %+.3f  %s  (t(%s) = %.2f, p = %.2g)' % (comparison['comparison'], float(comparison['mean_difference']),
                                                              comparison['stars'], comparison['df'],
                                                              float(comparison['t']), float(comparison['p'])))
