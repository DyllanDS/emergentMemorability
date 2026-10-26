"""
make_figureS11.py: Fig. 4c and SI Appendix, Fig. S11b, the preregistered discriminability analysis of
the rare-feature face study (SI Appendix, Section 3; AsPredicted #311640, 100 registered participants).
Each panel is saved as a figure of its own; SI Appendix, Fig. S11a is drawn by module 04.

WHAT THE FIGURES SHOW
  Fig. 4c: confidence ROCs for the same studied rare faces scored two ways, and for ordinary faces.
      Points: the seven empirical operating points (share of trials rated at or above each of the
      seven criteria on the eight-step old/new-by-confidence scale), pooled over the 100 participants:
      false alarm rate (x) against hit rate (y). Lines: the registered unequal-variance fit, with one
      set of criteria shared by every kind of face.
        red:          rare faces against unrelated new faces (never-studied ordinary faces)
        blue, dashed: rare faces against the never-studied partner carrying their feature
        dark gray:    ordinary faces against unrelated new faces
      The blue and dark gray curves lie on top of each other (d_a 1.00 and 1.00). Black dashed: chance.
  SI Appendix, Fig. S11b: the effect of the lure set (1.67 minus 1.00) and the effect of the feature
      itself (rare faces against same-feature lures minus ordinary faces against unrelated new faces),
      each shown as the registered estimate with its 95% bootstrap interval over participants (left
      of each pair; the registered bootstrap, 10,000 refits) and over the six rare features (right;
      1,000 refits).

INPUTS (every plotted number is read, none is typed in)
  outputs/roc_operating_points.csv      the empirical operating points (roc_points.py, from the data)
  data/provided/point_fit.json          the registered fit: mean and SD of each kind of face on the axis
                                        where never-studied ordinary faces are N(0, 1) (saved fit;
                                        refitting needs MemToolbox2)
  outputs/registered_fit_summary.csv    the two contrasts with their intervals over participants and over
                                        features (check_provided.py, from the saved fit and resamples)
OUTPUTS  outputs/figures/Fig4c_confidence_roc.png (300 dpi, 9.9 x 9.6 cm)
         outputs/figures/FigS11b_lure_set_and_feature_effects.png (300 dpi, 7.9 x 9.6 cm)
         outputs/figure_data/fig4c_operating_points.csv, fig4c_fitted_distributions.csv,
         outputs/figure_data/figS11b_contrasts.csv
SEEDS    none
RUNTIME  a few seconds
RUN      python A3c_face_study_model_and_sdt_figS10_S11/code/make_figureS11.py
"""
import csv
import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CODE_FOLDER, '..', '..', 'shared', 'python'))
import release_paths              # noqa: E402
import figure_style as ls      # noqa: E402

PATHS = release_paths.module_paths(__file__)
# Each panel is drawn on a page of its own, 9.6 cm high and as wide as its share of a 17.8 cm PNAS
# double-column page (the ROC panel and the effects panel split it 1.25 to 1).
PAGE_HEIGHT_CM = 9.6
ROC_PAGE_WIDTH_CM = ls.WIDTHS_CM['double'] * 1.25 / 2.25
CONTRAST_PAGE_WIDTH_CM = ls.WIDTHS_CM['double'] * 1.0 / 2.25


# ------------------------------------------------------------------ read
def read_operating_points():
    """For each kind of test face: the share of trials rated 2 or higher, 3 or higher, ..., 8
    (seven points, from the most lenient criterion to the strictest)."""
    operating_points = {}
    with open(os.path.join(PATHS['outputs'], 'roc_operating_points.csv'), newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            operating_points[row['face_kind']] = np.array([float(row['rated_%d_or_higher' % rating])
                                                           for rating in range(2, 9)])
    return operating_points


def read_fitted_distributions():
    """Mean and SD of each kind of face on the fitted axis (point_fit.json, 'cells')."""
    with open(os.path.join(PATHS['provided'], 'point_fit.json'), encoding='utf-8') as handle:
        cells = json.load(handle)['cells']
    return {'studied_rare': tuple(cells['rare_old']),
            'same_feature_lure': tuple(cells['rare_foil']),
            'studied_ordinary': tuple(cells['ordinary_old']),
            'never_studied': (0.0, 1.0)}           # the anchor that defines the axis


def read_contrasts():
    """The two contrasts: estimate, interval over participants, interval over features."""
    contrasts = {}
    with open(os.path.join(PATHS['outputs'], 'registered_fit_summary.csv'), newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            contrasts[row['quantity']] = (float(row['estimate']),
                                          (float(row['participants_ci_low']), float(row['participants_ci_high'])),
                                          (float(row['items_ci_low']), float(row['items_ci_high'])))
    return {'lure_set': contrasts['interaction'], 'feature': contrasts['equivalence']}


def d_a(target, lure):
    """d_a between two normal distributions given as (mean, SD)."""
    return (target[0] - lure[0]) * np.sqrt(2.0 / (target[1] ** 2 + lure[1] ** 2))


def write_rows(file_name, rows):
    with open(os.path.join(PATHS['figure_data'], file_name), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: (repr(value) if isinstance(value, float) else value) for name, value in row.items()})


# ------------------------------------------------------------------ build
def main():
    operating_points = read_operating_points()
    distributions = read_fitted_distributions()
    contrasts = read_contrasts()

    # The three scorings: (target kind, lure kind, color, dashed?, label)
    rare_vs_new = ('studied_rare', 'never_studied', ls.NATURAL_RED, False, 'Rare vs Unrelated New Faces')
    rare_vs_feature_lure = ('studied_rare', 'same_feature_lure', ls.CONTROLLED_BLUE, True, 'Rare vs Same-Feature Lure')
    ordinary_vs_new = ('studied_ordinary', 'never_studied', ls.MEMORY_DARK, False, 'Ordinary vs Unrelated New Faces')

    ls.apply_style('data')
    roc_figure = plt.figure(figsize=(ROC_PAGE_WIDTH_CM * ls.CM, PAGE_HEIGHT_CM * ls.CM), layout='constrained')
    roc_figure.get_layout_engine().set(w_pad=0.06, h_pad=0.06)
    roc_axes = roc_figure.add_subplot(1, 1, 1)
    contrast_figure = plt.figure(figsize=(CONTRAST_PAGE_WIDTH_CM * ls.CM, PAGE_HEIGHT_CM * ls.CM),
                                 layout='constrained')
    contrast_figure.get_layout_engine().set(w_pad=0.06, h_pad=0.06)
    contrast_axes = contrast_figure.add_subplot(1, 1, 1)

    # ---------------- Fig. 4c: ROC
    ax = roc_axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_box_aspect(1)
    ax.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=0.8, zorder=1)  # chance
    criterion_positions = np.linspace(-8, 10, 800)
    drawing_order = [ordinary_vs_new, rare_vs_feature_lure, rare_vs_new]   # dashed blue over dark
    point_rows = []
    for layer, (target_kind, lure_kind, color, dashed, label) in enumerate(drawing_order):
        target_mean, target_sd = distributions[target_kind]
        lure_mean, lure_sd = distributions[lure_kind]
        fitted_hits = 1 - norm.cdf((criterion_positions - target_mean) / target_sd)
        fitted_false_alarms = 1 - norm.cdf((criterion_positions - lure_mean) / lure_sd)
        line_style = dict(color=color, linewidth=1.3, zorder=2 + layer)
        if dashed:
            line_style['dashes'] = (3.5, 2.0)
        ax.plot(fitted_false_alarms, fitted_hits, **line_style)
        ax.plot(operating_points[lure_kind], operating_points[target_kind], linestyle='none',
                marker='o', markersize=ls.MARKER_SIZE - 0.5, markerfacecolor=color,
                markeredgecolor='white', markeredgewidth=0.4, zorder=6 + layer, clip_on=False)
        for criterion_number in range(7):
            point_rows.append({'curve': label, 'target_kind': target_kind, 'lure_kind': lure_kind,
                               'criterion': 'rated %d or higher' % (criterion_number + 2),
                               'false_alarm_rate': float(operating_points[lure_kind][criterion_number]),
                               'hit_rate': float(operating_points[target_kind][criterion_number])})
    # direct labels in each curve's color, stacked in the empty corner below the chance line
    label_rows = [(0.28, rare_vs_new), (0.165, rare_vs_feature_lure), (0.05, ordinary_vs_new)]
    distribution_rows = []
    for y_position, (target_kind, lure_kind, color, dashed, label) in label_rows:
        value = d_a(distributions[target_kind], distributions[lure_kind])
        ax.text(0.37, y_position, '%s\n$d_a$ = %s' % (label, ls.format_stat(value)),
                color=color, fontsize=ls.FONT_SIZE, ha='left', va='center', linespacing=1.1)
        distribution_rows.append({'curve': label, 'target_kind': target_kind, 'target_mean': distributions[target_kind][0],
                                  'target_sd': distributions[target_kind][1], 'lure_kind': lure_kind,
                                  'lure_mean': distributions[lure_kind][0], 'lure_sd': distributions[lure_kind][1],
                                  'd_a': float(value)})
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel('False Alarm Rate')
    ax.set_ylabel('Hit Rate')
    ls.plain_ticks(ax)
    ls.panel_label(ax, 'c')

    # ---------------- SI Appendix, Fig. S11b: the two contrasts, two kinds of interval
    ax = contrast_axes
    centres = ls.category_axis(ax, ['Effect of the\nLure Set', 'Effect of the\nFeature Itself'])
    ax.set_ylim(-0.8, 1.5)
    ax.set_yticks([-0.5, 0, 0.5, 1, 1.5])
    ls.zero_line(ax)
    participant_offset, feature_offset = -0.22, 0.22
    contrast_rows = []
    for centre, key, name in zip(centres, ['lure_set', 'feature'], ['effect of the lure set', 'effect of the feature itself']):
        value, participant_interval, feature_interval = contrasts[key]
        ls.dot_with_interval(ax, [centre + participant_offset], [value], [participant_interval[0]],
                             [participant_interval[1]], color=ls.MEMORY_DARK)
        ls.dot_with_interval(ax, [centre + feature_offset], [value], [feature_interval[0]],
                             [feature_interval[1]], color=ls.MEMORY_DARK)
        ax.annotate('Over\nPeople', xy=(centre + participant_offset, -0.77), xycoords='data',
                    ha='center', va='bottom', fontsize=ls.FONT_SIZE, color=ls.LABEL_GRAY,
                    linespacing=1.1)
        ax.annotate('Over\nFeatures', xy=(centre + feature_offset, -0.77), xycoords='data',
                    ha='center', va='bottom', fontsize=ls.FONT_SIZE, color=ls.LABEL_GRAY,
                    linespacing=1.1)
        contrast_rows.append({'contrast': name, 'resampled': 'participants (10,000 refits, registered)',
                              'x': float(centre + participant_offset), 'estimate': value,
                              'ci_low': participant_interval[0], 'ci_high': participant_interval[1]})
        contrast_rows.append({'contrast': name, 'resampled': 'the six rare features (1,000 refits)',
                              'x': float(centre + feature_offset), 'estimate': value,
                              'ci_low': feature_interval[0], 'ci_high': feature_interval[1]})
    ax.set_ylabel('Difference in $d_a$')
    ls.plain_ticks(ax)
    ls.panel_label(ax, 'b')

    ls.save_figure(roc_figure, 'Fig4c_confidence_roc', PATHS['figures'])
    ls.save_figure(contrast_figure, 'FigS11b_lure_set_and_feature_effects', PATHS['figures'])
    write_rows('fig4c_operating_points.csv', point_rows)
    write_rows('fig4c_fitted_distributions.csv', distribution_rows)
    write_rows('figS11b_contrasts.csv', contrast_rows)

    # ---------------- checks printed for the numbers table
    for row in distribution_rows:
        print('d_a %-34s from point_fit.json %.4f' % (row['curve'], row['d_a']))
    for kind, points in operating_points.items():
        print('operating points %-18s %s' % (kind, ' '.join('%.3f' % point for point in points)))
    for key, (value, participant_interval, feature_interval) in contrasts.items():
        print('%-9s %+.3f participants [%+.3f, %+.3f] features [%+.3f, %+.3f]'
              % ((key, value) + participant_interval + feature_interval))


if __name__ == '__main__':
    main()
