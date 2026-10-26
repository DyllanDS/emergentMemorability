"""
make_figureS10.py: SI Appendix, Fig. S10, REM run through the preregistered rare-feature face design,
beside the data (SI Appendix, Section 3).

WHAT THE FIGURE SHOWS
Before any data were collected, the REM model (Shiffrin & Steyvers, 1997) was run through the face
experiment's design at many parameter settings, and three predictions were registered (AsPredicted
#311640). Gray points: model settings (spread sideways at random so they do not hide one another; the
sideways position means nothing). Colored or dark point: the data, 100 participants, with its 95%
bootstrap interval over participants.
  (a) the two rate predictions, as differences in the proportion of faces called old, one gray point
      per setting of the 729-setting sweep run before data collection:
        hits:          studied rare faces minus studied ordinary faces                     (prediction 1)
        false alarms:  a new face carrying a studied face's rare feature minus a new
                       ordinary face                                                       (prediction 2)
  (b) the two d_a contrasts (d_a: the unequal-variance form of d′), one gray point per setting of the
      243-setting run after data collection, scored as the data were (one shared set of seven
      confidence criteria, unequal-variance fit, 100 simulated participants per setting):
        effect of the lure set:       d_a(rare vs never-studied) minus d_a(rare vs same-feature lure)
                                      (prediction 3)
        effect of the feature itself: d_a(rare vs same-feature lure) minus d_a(ordinary vs never-studied)
                                      (registered as an estimate)
  The labels give the number of settings above zero.

INPUTS (every plotted number is read, none is typed in)
  outputs/rem_729_rates.csv             per-setting rates of the 729-setting sweep (rem_729_rates.py)
  outputs/observed_contrasts.csv        the observed hit and false alarm differences and their intervals
                                        (observed_contrasts.py, from the data)
  data/provided/rem_like_data.jsonl     the fitted d_a contrasts of the 243 settings (fields fitted.interaction
                                        and fitted.equivalence; saved fits, refitting needs MemToolbox2)
  outputs/registered_fit_summary.csv    the observed d_a contrasts (registered fit) and their intervals over
                                        participants (check_provided.py, from the saved fit and resamples)
OUTPUTS  outputs/figures/FigS10_rem_face_design.png (300 dpi, 17.8 cm wide)
         outputs/figure_data/figS10a_model_settings.csv, figS10a_data.csv,
         outputs/figure_data/figS10b_model_settings.csv, figS10b_data.csv
SEEDS    the sideways spread of the gray points: numpy.random.default_rng(1), (2), (3), (4), one per strip
         (appearance only)
RUNTIME  a few seconds
RUN      python A3c_face_study_model_and_sdt_figS10_S11/code/make_figureS10.py
"""
import csv
import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CODE_FOLDER, '..', '..', 'shared', 'python'))
import release_paths              # noqa: E402
import figure_style as ls      # noqa: E402

PATHS = release_paths.module_paths(__file__)


# ------------------------------------------------------------------ read the model settings
def read_model_rate_differences():
    """Hit and false alarm differences for each of the 729 settings (rem_729_rates.py)."""
    settings = []
    with open(os.path.join(PATHS['outputs'], 'rem_729_rates.csv'), newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            settings.append({'setting_number': int(row['setting_number']),
                             'hit_difference': float(row['hit_difference_rare_minus_ordinary']),
                             'false_alarm_difference': float(row['false_alarm_difference_feature_lure_minus_new'])})
    return settings


def read_model_d_a_differences():
    """Lure-set effect (fitted.interaction) and feature effect (fitted.equivalence) for each of the
    243 fitted settings of the saved run. Rows with a fitting error would be dropped (there are none)."""
    settings = []
    with open(os.path.join(PATHS['provided'], 'rem_like_data.jsonl'), encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('err') is not None or not row.get('fitted'):
                continue
            settings.append({'setting_number': row['i'], 'g': row['pars']['g'], 'w': row['pars']['w'],
                             'u': row['pars']['u'], 'c': row['pars']['c'], 'nfeat': row['pars']['nfeat'],
                             'lure_set_effect': row['fitted']['interaction'],
                             'feature_effect': row['fitted']['equivalence']})
    return settings


# ------------------------------------------------------------------ read the observed results
def read_observed():
    observed = {}
    with open(os.path.join(PATHS['outputs'], 'observed_contrasts.csv'), newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            observed[row['contrast']] = (float(row['mean_difference']), float(row['ci_low']), float(row['ci_high']))
    with open(os.path.join(PATHS['outputs'], 'registered_fit_summary.csv'), newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            observed[row['quantity']] = (float(row['estimate']), float(row['participants_ci_low']),
                                         float(row['participants_ci_high']))
    return {'hit': observed['hits'], 'false_alarm': observed['false_alarms'],
            'lure_set': observed['interaction'], 'feature': observed['equivalence']}


# ------------------------------------------------------------------ drawing helpers (this figure only)
MODEL_OFFSET = -0.18      # model points sit left of each category center
DATA_OFFSET = 0.20        # the observed point sits right of it
JITTER_HALF_WIDTH = 0.11


def draw_model_strip(ax, centre, values, seed):
    """One small translucent gray point per model setting, spread sideways at random so they
    do not sit on top of one another (the sideways position means nothing)."""
    random_generator = np.random.default_rng(seed)
    x_positions = centre + MODEL_OFFSET + random_generator.uniform(-JITTER_HALF_WIDTH,
                                                                   JITTER_HALF_WIDTH, len(values))
    ax.scatter(x_positions, values, s=4, facecolors=[ls.MODEL_GRAY + (0.35,)],
               edgecolors='none', zorder=3)
    return x_positions


def draw_observed_point(ax, centre, value, low, high, color):
    ls.dot_with_interval(ax, [centre + DATA_OFFSET], [value], [low], [high], color=color)


def count_label(ax, centre, n_above, n_total):
    """'712 of 729 above zero' written at the top of the plot over the model points."""
    ax.annotate('%d of %d\nabove zero' % (n_above, n_total),
                xy=(centre + MODEL_OFFSET, 0.98), xycoords=('data', 'axes fraction'),
                ha='center', va='top', fontsize=ls.FONT_SIZE, color='black', linespacing=1.15)


def column_names(ax, centre, data_color, y_position):
    """'Model' under the gray points and 'Data' under the observed point."""
    ax.annotate('Model', xy=(centre + MODEL_OFFSET, y_position), xycoords='data',
                ha='center', va='bottom', fontsize=ls.FONT_SIZE, color=ls.LABEL_GRAY)
    ax.annotate('Data', xy=(centre + DATA_OFFSET, y_position), xycoords='data',
                ha='center', va='bottom', fontsize=ls.FONT_SIZE, color=data_color)


def write_rows(file_name, rows):
    with open(os.path.join(PATHS['figure_data'], file_name), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: (repr(value) if isinstance(value, float) else value) for name, value in row.items()})


# ------------------------------------------------------------------ build
def main():
    rate_settings = read_model_rate_differences()
    d_a_settings = read_model_d_a_differences()
    observed = read_observed()
    model_hit_differences = np.array([setting['hit_difference'] for setting in rate_settings])
    model_false_alarm_differences = np.array([setting['false_alarm_difference'] for setting in rate_settings])
    model_lure_set_effects = np.array([setting['lure_set_effect'] for setting in d_a_settings])
    model_feature_effects = np.array([setting['feature_effect'] for setting in d_a_settings])

    # counts quoted in the caption and the SI text
    n_rate_settings = len(model_hit_differences)
    n_hit_above = int((model_hit_differences > 0).sum())
    n_false_alarm_above = int((model_false_alarm_differences > 0).sum())
    n_fit_settings = len(model_lure_set_effects)
    n_lure_set_above = int((model_lure_set_effects > 0).sum())
    n_feature_above = int((model_feature_effects > 0).sum())

    ls.apply_style('model')
    fig, axes = ls.new_figure(n_panels=2, width='double', height_cm=8.2)

    # ---------------- panel a: hit and false alarm differences
    ax = axes[0]
    centres = ls.category_axis(ax, ['Hits:\nRare Minus\nOrdinary Face',
                                    'False Alarms:\nLure With Feature\nMinus New Face'])
    ax.set_ylim(-0.2, 0.95)
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    ls.zero_line(ax)
    hit_x = draw_model_strip(ax, centres[0], model_hit_differences, seed=1)
    false_alarm_x = draw_model_strip(ax, centres[1], model_false_alarm_differences, seed=2)
    draw_observed_point(ax, centres[0], *observed['hit'], color=ls.HIT_BLUE)
    draw_observed_point(ax, centres[1], *observed['false_alarm'], color=ls.FA_RED)
    count_label(ax, centres[0], n_hit_above, n_rate_settings)
    count_label(ax, centres[1], n_false_alarm_above, n_rate_settings)
    column_names(ax, centres[0], ls.HIT_BLUE, -0.18)
    column_names(ax, centres[1], ls.FA_RED, -0.18)
    ax.set_ylabel('Difference in Proportion\nCalled Old')
    ax.grid(axis='x', visible=False)
    ls.plain_ticks(ax)
    ls.panel_label(ax, 'a')

    # ---------------- panel b: d_a differences
    ax = axes[1]
    centres_b = ls.category_axis(ax, ['Effect of the\nLure Set', 'Effect of the\nFeature Itself'])
    # The top of the axis leaves room for the count labels: the highest model setting (3.7)
    # must sit below them, not behind them.
    ax.set_ylim(-2.5, 5.3)
    ax.set_yticks([-2, -1, 0, 1, 2, 3, 4])
    ls.zero_line(ax)
    lure_set_x = draw_model_strip(ax, centres_b[0], model_lure_set_effects, seed=3)
    feature_x = draw_model_strip(ax, centres_b[1], model_feature_effects, seed=4)
    draw_observed_point(ax, centres_b[0], *observed['lure_set'], color=ls.MEMORY_DARK)
    draw_observed_point(ax, centres_b[1], *observed['feature'], color=ls.MEMORY_DARK)
    count_label(ax, centres_b[0], n_lure_set_above, n_fit_settings)
    count_label(ax, centres_b[1], n_feature_above, n_fit_settings)
    column_names(ax, centres_b[0], ls.MEMORY_DARK, -2.42)
    column_names(ax, centres_b[1], ls.MEMORY_DARK, -2.42)
    ax.set_ylabel('Difference in $d_a$')
    ax.grid(axis='x', visible=False)
    ls.plain_ticks(ax)
    ls.panel_label(ax, 'b')

    ls.save_figure(fig, 'FigS10_rem_face_design', PATHS['figures'])

    # ---------------- the plotted numbers
    rows = []
    for index, setting in enumerate(rate_settings):
        rows.append({'setting_number': setting['setting_number'],
                     'hit_difference_rare_minus_ordinary': setting['hit_difference'],
                     'hit_strip_x': float(hit_x[index]),
                     'false_alarm_difference_feature_lure_minus_new': setting['false_alarm_difference'],
                     'false_alarm_strip_x': float(false_alarm_x[index])})
    write_rows('figS10a_model_settings.csv', rows)
    write_rows('figS10a_data.csv', [
        {'difference': 'hits: studied rare minus studied ordinary', 'x': float(centres[0] + DATA_OFFSET),
         'mean': observed['hit'][0], 'ci_low': observed['hit'][1], 'ci_high': observed['hit'][2],
         'settings_above_zero': n_hit_above, 'settings': n_rate_settings},
        {'difference': 'false alarms: new face with the feature minus new ordinary face',
         'x': float(centres[1] + DATA_OFFSET), 'mean': observed['false_alarm'][0], 'ci_low': observed['false_alarm'][1],
         'ci_high': observed['false_alarm'][2], 'settings_above_zero': n_false_alarm_above,
         'settings': n_rate_settings}])
    rows = []
    for index, setting in enumerate(d_a_settings):
        rows.append(dict(setting, lure_set_strip_x=float(lure_set_x[index]), feature_strip_x=float(feature_x[index])))
    write_rows('figS10b_model_settings.csv', rows)
    write_rows('figS10b_data.csv', [
        {'difference': 'effect of the lure set (d_a)', 'x': float(centres_b[0] + DATA_OFFSET),
         'estimate': observed['lure_set'][0], 'ci_low': observed['lure_set'][1], 'ci_high': observed['lure_set'][2],
         'settings_above_zero': n_lure_set_above, 'settings': n_fit_settings},
        {'difference': 'effect of the feature itself (d_a)', 'x': float(centres_b[1] + DATA_OFFSET),
         'estimate': observed['feature'][0], 'ci_low': observed['feature'][1], 'ci_high': observed['feature'][2],
         'settings_above_zero': n_feature_above, 'settings': n_fit_settings}])

    # ---------------- every number the caption and SI text quote, for checking
    print('panel a: %d settings; hits above zero %d, false alarms above zero %d'
          % (n_rate_settings, n_hit_above, n_false_alarm_above))
    print('         model hit difference median %.3f range %.3f to %.3f'
          % (np.median(model_hit_differences), model_hit_differences.min(), model_hit_differences.max()))
    print('         model false alarm difference median %.3f range %.3f to %.3f'
          % (np.median(model_false_alarm_differences), model_false_alarm_differences.min(),
             model_false_alarm_differences.max()))
    print('panel b: %d settings; lure-set effect above zero %d, feature effect above zero %d'
          % (n_fit_settings, n_lure_set_above, n_feature_above))
    print('         model lure-set effect median %.3f range %.3f to %.3f'
          % (np.median(model_lure_set_effects), model_lure_set_effects.min(), model_lure_set_effects.max()))
    print('         model feature effect median %.3f range %.3f to %.3f'
          % (np.median(model_feature_effects), model_feature_effects.min(), model_feature_effects.max()))
    for name, (value, low, high) in observed.items():
        print('observed %-12s %+.3f [%+.3f, %+.3f]' % (name, value, low, high))


if __name__ == '__main__':
    main()
