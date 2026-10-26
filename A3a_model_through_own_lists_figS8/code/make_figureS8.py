"""
make_figureS8.py: Fig. S8, a global matching model run through our own experiments (SI Appendix,
Section 3), drawn from this module's run.

WHAT THE FIGURE SHOWS
  A version of REM with continuous feature values (49 features from four computer vision networks)
  is given the study lists and test trials our participants received. Only two numbers are set from
  the data, the encoding noise and the decision criterion, and they are set from the overall hit and
  false alarm rates alone. Everything plotted for the model is therefore a prediction.

  (a) Memorability inversion experiment (the 50 participants of Fig. 3e). One point = one of the
      40 critical items. x = the model's predicted memorability (hit rate minus false alarm rate),
      y = observed memorability (hit rate minus false alarm rate). Gray line = least-squares fit.
      Printed statistic = Spearman rho over the 40 items.
  (b) Image set experiments (the 450 participants of Figs. 6 and 7), the images tested in more than
      one image set. Each image is compared with itself: its d′ in the set where its most similar
      set-mates were least similar (more distinctive) minus its d′ in the set where they were most
      similar (less distinctive). Blue bar = observed, gray bar = model.
  (c) The same comparison split into hit rate (blue) and false alarm rate (red), observed and model.
  Error bars in (b) and (c): 95% bootstrap confidence intervals over the images (10,000 image
  resamples, seed 0), as computed by paired_crowding.py.

INPUTS (outputs/ of this module, written by the earlier steps of run_module.py)
  fit_inversion_items.csv      one row per critical item: observed hitRate, faRate, cr, nOld, nNew,
                               origMem, and the model's pred_hit, pred_fa, pred_cr
  fit_inversion.txt            the model run (sigma sweep and chosen operating point)
  disattenuation.txt           the reliability of the observed scores
  paired_crowding_summary.csv  the paired same-image contrast, observed and model, with intervals
  paired_crowding.txt          the printed form of the same run (counts, the d′ split)
  paired_crowding_images.csv   (only to count the images)
  model_inputs/inversion_sample.csv, datasets_sample.csv (the kept participants and trials)

OUTPUTS
  outputs/figures/FigS8_model_through_experiments.png (300 dpi)
  outputs/figure_data/FigS8a.csv (the 40 points), FigS8b.csv (two bars), FigS8c.csv (four bars)
  The numbers of Fig. S8 and of the SI text about it are printed at the end (logged by run_module.py).

DRAWING AND CHECKS
  The axis limits of each panel (panel a 0 to 1 on both axes, panel b -0.02 to 0.2, panel c -0.08 to
  0.06) widen in whole steps when a point, bar or interval falls outside them (in panel a an item's
  observed memorability can be below 0). The statistic of panel a goes in the corner with the fewest
  points (upper left first); the labels of panel c sit above a rising bar and below a falling one.
  The figure is saved through shared/python/figure_style.py into outputs/figures/. Before drawing, the
  script checks that no critical item is tested more often than there are kept participants (the
  missing tests are the trials outside 200 to 10,000 ms), that the item counts add up to the kept
  critical trials, that every recomputed rho equals the row of the chosen sigma in fit_inversion.txt,
  and that the counts of images and participants agree across the files it reads.

RUN (from the repository root; a few seconds):  python A3a_model_through_own_lists_figS8/code/make_figureS8.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (first: it stops Python from writing __pycache__ folders)
import figure_style as ls  # noqa: E402

PATHS = release_paths.module_paths(__file__)
DATA_FOLDER = PATHS['outputs']                     # the files the earlier steps of this module wrote
FIGURE_NAME = 'FigS8_model_through_experiments'


def read_text(file_name):
    with open(os.path.join(DATA_FOLDER, file_name)) as handle:
        return handle.read()


def emptiest_corner(axis, x_values, y_values, box_width=0.5, box_height=0.24):
    """The corner for the statistic text with the fewest points under it (first choice upper left).
    Positions are fractions of the axes; set the limits first."""
    x_low, x_high = axis.get_xlim()
    y_low, y_high = axis.get_ylim()
    x_fraction = (np.asarray(x_values, dtype=float) - x_low) / (x_high - x_low)
    y_fraction = (np.asarray(y_values, dtype=float) - y_low) / (y_high - y_low)
    boxes = {'upper left': (0, box_width, 1 - box_height, 1), 'lower right': (1 - box_width, 1, 0, box_height),
             'upper right': (1 - box_width, 1, 1 - box_height, 1), 'lower left': (0, box_width, 0, box_height)}
    best_corner = None
    fewest_points = None
    for corner, (left, right, bottom, top) in boxes.items():
        inside = np.sum((x_fraction >= left) & (x_fraction <= right) & (y_fraction >= bottom) & (y_fraction <= top))
        if fewest_points is None or inside < fewest_points:
            best_corner = corner
            fewest_points = inside
    return best_corner


def axis_limits(values, low_default, high_default, step):
    """The default axis limits, widened in whole steps if any plotted value falls outside them."""
    low = low_default
    high = high_default
    smallest = float(np.nanmin(values))
    largest = float(np.nanmax(values))
    while smallest < low:
        low = low - step
    while largest > high:
        high = high + step
    return low, high


# =============================================================================================
# 1. Load the inversion experiment items and recompute the panel (a) statistic
# =============================================================================================
items = pd.read_csv(os.path.join(DATA_FOLDER, 'fit_inversion_items.csv'), float_precision='round_trip')
assert len(items) == 40, 'expected 40 critical items'

# Every kept participant was tested once on every critical item (as studied or as new); the tests
# that are missing are the trials outside 200 to 10,000 ms, which the analysis drops.
inversion_sample = pd.read_csv(os.path.join(DATA_FOLDER, 'model_inputs', 'inversion_sample.csv')).iloc[0]
number_of_kept_participants = int(inversion_sample['kept_participants'])
tests_per_item = items['nOld'] + items['nNew']
assert tests_per_item.max() <= number_of_kept_participants, 'an item has more tests than there are participants'
critical_tests_kept = int(inversion_sample['critical_test_trials_kept'])
assert int(tests_per_item.sum()) == critical_tests_kept, 'the item counts do not add up to the critical test trials'

predicted_memorability = items['pred_cr'].to_numpy()     # model: hit rate minus false alarm rate
observed_memorability = items['cr'].to_numpy()           # data:  hit rate minus false alarm rate
assert np.allclose(observed_memorability, items['hitRate'] - items['faRate'])

rho_memorability, p_memorability = spearmanr(predicted_memorability, observed_memorability)
rho_hits, p_hits = spearmanr(items['pred_hit'], items['hitRate'])
rho_false_alarms, p_false_alarms = spearmanr(items['pred_fa'], items['faRate'])
rho_model_vs_original, p_model_vs_original = spearmanr(items['pred_cr'], items['origMem'])
rho_observed_vs_original, p_observed_vs_original = spearmanr(items['cr'], items['origMem'])

# Check every recomputed rho against the row the model run printed for the chosen sigma.
model_run_text = read_text('fit_inversion.txt')
chosen = re.search(r'closest to the observed operating point: sigma ([\d.]+), criterion ([-\d.e+]+), '
                   r'hit ([\d.]+), FA ([\d.]+)', model_run_text)
chosen_sigma = float(chosen.group(1))
row = None
for line in model_run_text.splitlines():
    parts = line.split()
    if len(parts) == 9 and re.match(r'^[\d.]+$', parts[0]) and abs(float(parts[0]) - chosen_sigma) < 1e-9:
        row = parts
assert row is not None, 'the sweep row of the chosen sigma was not found in fit_inversion.txt'
printed_model_hit, printed_model_fa = float(row[2]), float(row[3])
printed_rho_cr, printed_rho_hit, printed_rho_fa, printed_rho_orig = (float(row[k]) for k in (5, 6, 7, 8))
for recomputed, printed, name in [(rho_memorability, printed_rho_cr, 'model vs observed memorability'),
                                  (rho_hits, printed_rho_hit, 'model vs observed hits'),
                                  (rho_false_alarms, printed_rho_fa, 'model vs observed false alarms'),
                                  (rho_model_vs_original, printed_rho_orig, 'model vs original THINGS')]:
    assert abs(recomputed - printed) < 5e-4, (name, recomputed, printed)
printed_observed_vs_original = float(re.search(r'observed CR vs original THINGS memorability: ([-+\d.]+)',
                                               model_run_text).group(1))
assert abs(rho_observed_vs_original - printed_observed_vs_original) < 5e-4

reliability_text = read_text('disattenuation.txt')
reliability = float(re.search(r'reliability = 1 - share = ([\d.]+)', reliability_text).group(1))
rho_memorability_disattenuated = float(re.search(r'model vs observed CR\s+[-\d.]+\s+([-\d.]+)',
                                                 reliability_text).group(1))

# =============================================================================================
# 2. Load the paired same-image contrast of the image set experiments
# =============================================================================================
summary = pd.read_csv(os.path.join(DATA_FOLDER, 'paired_crowding_summary.csv'), float_precision='round_trip')


def contrast(measure, source):
    """Mean and 95% image-bootstrap interval of one row of paired_crowding_summary.csv.
    The sign is 'more distinctive set minus less distinctive set' (paired_crowding.py's 'least
    crowded set minus most crowded set')."""
    row = summary[(summary['measure'] == measure) & (summary['source'] == source)].iloc[0]
    return dict(mean=row['mean'], low=row['lo_images'], high=row['hi_images'])


observed_dprime = contrast('d-prime', 'observed')
model_dprime = contrast('d-prime', 'model')
observed_hits = contrast('hit rate', 'observed')
model_hits = contrast('hit rate', 'model')
observed_false_alarms = contrast('false alarm rate', 'observed')
model_false_alarms = contrast('false alarm rate', 'model')

paired_text = read_text('paired_crowding.txt')
assert 'PAIRED CONTRAST: least crowded set minus most crowded set, same image' in paired_text
number_of_images = int(re.search(r'([\d,]+) images appear in two or more sets', paired_text)
                       .group(1).replace(',', ''))
number_of_participants = int(re.search(r'\(([\d,]+) of them, each in exactly', paired_text)
                             .group(1).replace(',', ''))
dprime_row = re.search(r'd-prime\s+([+-][\d.]+)\s+\[', paired_text)
datasets_sample = pd.read_csv(os.path.join(DATA_FOLDER, 'model_inputs', 'datasets_sample.csv'))
datasets_sample = datasets_sample.set_index('image_set').loc['all three sets']
assert number_of_images == len(pd.read_csv(os.path.join(DATA_FOLDER, 'paired_crowding_images.csv')))
assert number_of_images == int(datasets_sample['images_in_two_or_more_sets'])
assert number_of_participants == int(datasets_sample['kept_participants'])
assert abs(observed_dprime['mean'] - float(dprime_row.group(1))) < 5e-4
split_row = re.search(r'observed ([-\d.]+) from hits and ([-\d.]+) from false\s+alarms', paired_text)
dprime_from_hits, dprime_from_false_alarms = float(split_row.group(1)), float(split_row.group(2))

# =============================================================================================
# 3. Draw
# =============================================================================================
ls.apply_style('data')
fig, axes = ls.new_figure(n_panels=3, width='double', height_cm=6.4)
axis_a, axis_b, axis_c = axes
# x label of panels b and c: what each bar is (the same image, more minus less distinctive set)
SAME_IMAGE_LABEL = 'Same Image, More Minus\nLess Distinctive Set'

# ---- (a) model-predicted against observed memorability, 40 critical items
ls.scatter_points(axis_a, predicted_memorability, observed_memorability)
# 0 to 1, widened in steps of 0.2 if an item lies outside (hit rate minus
# false alarm rate can be negative)
panel_a_x_limits = axis_limits(predicted_memorability, 0, 1, 0.2)
panel_a_y_limits = axis_limits(observed_memorability, 0, 1, 0.2)
axis_a.set_xlim(*panel_a_x_limits)
axis_a.set_ylim(*panel_a_y_limits)
axis_a.set_xticks([tick for tick in np.arange(-1, 1.01, 0.2)
                   if panel_a_x_limits[0] - 1e-9 <= tick <= panel_a_x_limits[1] + 1e-9])
axis_a.set_yticks([tick for tick in np.arange(-1, 1.01, 0.2)
                   if panel_a_y_limits[0] - 1e-9 <= tick <= panel_a_y_limits[1] + 1e-9])
ls.fit_line(axis_a, predicted_memorability, observed_memorability, over='axis')
axis_a.set_xlabel('Model-Predicted\nMemorability')
axis_a.set_ylabel('Observed\nMemorability')
stat_string_a = ls.format_r_p(rho_memorability, p_memorability, n=40, unit='items', symbol='ρ')
ls.stat_text(axis_a, stat_string_a, corner=emptiest_corner(axis_a, predicted_memorability, observed_memorability))
ls.plain_ticks(axis_a)
ls.panel_label(axis_a, 'a')

# ---- (b) change in d-prime, observed against model
bar_positions_b = ls.category_axis(axis_b, ['Observed', 'Model'])
dprime_means = [observed_dprime['mean'], model_dprime['mean']]
axis_b.bar(bar_positions_b, dprime_means, width=0.6, color=[ls.DATA_BLUE, ls.MODEL_GRAY],
           edgecolor='none', zorder=2)
ls.error_bars(axis_b, bar_positions_b, dprime_means,
              [observed_dprime['low'], model_dprime['low']],
              [observed_dprime['high'], model_dprime['high']])
ls.zero_line(axis_b)
panel_b_values = [observed_dprime['low'], model_dprime['low'], observed_dprime['high'], model_dprime['high']]
panel_b_limits = axis_limits(panel_b_values, -0.02, 0.2, 0.02)
axis_b.set_ylim(*panel_b_limits)
axis_b.set_yticks([tick for tick in (-0.2, -0.1, 0, 0.1, 0.2, 0.3)
                   if panel_b_limits[0] - 1e-9 <= tick <= panel_b_limits[1] + 1e-9])
axis_b.set_ylabel('Change in d′')
axis_b.set_xlabel(SAME_IMAGE_LABEL)
ls.stat_text(axis_b, 'n = {:,} images'.format(number_of_images), corner='upper right')
ls.plain_ticks(axis_b)
ls.panel_label(axis_b, 'b')

# ---- (c) change in hit and false alarm rate, observed and model
group_positions = ls.category_axis(axis_c, ['Observed', 'Model'])
bar_width = 0.36
hit_positions = group_positions - bar_width / 2
false_alarm_positions = group_positions + bar_width / 2
hit_means = [observed_hits['mean'], model_hits['mean']]
false_alarm_means = [observed_false_alarms['mean'], model_false_alarms['mean']]
axis_c.bar(hit_positions, hit_means, width=bar_width, color=ls.HIT_BLUE, edgecolor='none', zorder=2)
axis_c.bar(false_alarm_positions, false_alarm_means, width=bar_width, color=ls.FA_RED,
           edgecolor='none', zorder=2)
ls.error_bars(axis_c, hit_positions, hit_means,
              [observed_hits['low'], model_hits['low']], [observed_hits['high'], model_hits['high']])
ls.error_bars(axis_c, false_alarm_positions, false_alarm_means,
              [observed_false_alarms['low'], model_false_alarms['low']],
              [observed_false_alarms['high'], model_false_alarms['high']])
ls.zero_line(axis_c)
panel_c_values = [observed_hits['low'], model_hits['low'], observed_hits['high'], model_hits['high'],
                  observed_false_alarms['low'], model_false_alarms['low'],
                  observed_false_alarms['high'], model_false_alarms['high']]
panel_c_limits = axis_limits(panel_c_values, -0.08, 0.06, 0.02)
axis_c.set_ylim(*panel_c_limits)
axis_c.set_yticks([tick for tick in (-0.12, -0.08, -0.04, 0, 0.04, 0.08, 0.12)
                   if panel_c_limits[0] - 1e-9 <= tick <= panel_c_limits[1] + 1e-9])
axis_c.set_ylabel('Change in Hit or\nFalse Alarm Rate')
axis_c.set_xlabel(SAME_IMAGE_LABEL)
# Direct labels name the two bars of the first group (no legend): above a bar that rises, below a bar
# that falls.
if observed_hits['mean'] >= 0:
    ls.direct_label(axis_c, hit_positions[0], observed_hits['high'], 'Hits', ls.HIT_BLUE,
                    ha='center', va='bottom', dx_points=0, dy_points=2)
else:
    ls.direct_label(axis_c, hit_positions[0], observed_hits['low'], 'Hits', ls.HIT_BLUE,
                    ha='center', va='top', dx_points=0, dy_points=-2)
if observed_false_alarms['mean'] <= 0:
    ls.direct_label(axis_c, false_alarm_positions[0], observed_false_alarms['low'], 'False\nAlarms', ls.FA_RED,
                    ha='center', va='top', dx_points=0, dy_points=-2)
else:
    ls.direct_label(axis_c, false_alarm_positions[0], observed_false_alarms['high'], 'False\nAlarms', ls.FA_RED,
                    ha='center', va='bottom', dx_points=0, dy_points=2)
ls.plain_ticks(axis_c)
ls.panel_label(axis_c, 'c')

written = ls.save_figure(fig, FIGURE_NAME, PATHS['figures'])

# =============================================================================================
# 4. The plotted values, one file per panel
# =============================================================================================
pd.DataFrame({'image_name': items['imageName'], 'model_predicted_memorability': predicted_memorability,
              'observed_memorability': observed_memorability}).to_csv(
    os.path.join(PATHS['figure_data'], 'FigS8a.csv'), index=False)
pd.DataFrame({'bar': ['observed', 'model'],
              'change_in_dprime': dprime_means,
              'ci_low_images': [observed_dprime['low'], model_dprime['low']],
              'ci_high_images': [observed_dprime['high'], model_dprime['high']],
              'n_images': [number_of_images, number_of_images]}).to_csv(
    os.path.join(PATHS['figure_data'], 'FigS8b.csv'), index=False)
pd.DataFrame({'group': ['observed', 'observed', 'model', 'model'],
              'rate': ['hit rate', 'false alarm rate', 'hit rate', 'false alarm rate'],
              'change': [observed_hits['mean'], observed_false_alarms['mean'], model_hits['mean'],
                         model_false_alarms['mean']],
              'ci_low_images': [observed_hits['low'], observed_false_alarms['low'], model_hits['low'],
                                model_false_alarms['low']],
              'ci_high_images': [observed_hits['high'], observed_false_alarms['high'], model_hits['high'],
                                 model_false_alarms['high']]}).to_csv(
    os.path.join(PATHS['figure_data'], 'FigS8c.csv'), index=False)
print('wrote outputs/figure_data/FigS8a.csv, FigS8b.csv, FigS8c.csv')

# =============================================================================================
# 5. Print the numbers of Fig. S8 and of the SI text about it, with their sources
# =============================================================================================
print('\nNUMBERS (source in brackets)')
print('  (a) critical items: %d; tests per item %d to %d over %d kept participants  [fit_inversion_items.csv]'
      % (len(items), tests_per_item.min(), tests_per_item.max(), number_of_kept_participants))
print('  (a) Spearman rho model vs observed memorability = %.4f, p = %.4g -> %s  [recomputed; fit_inversion.txt]'
      % (rho_memorability, p_memorability, stat_string_a.replace('\n', ' / ')))
print('  (a) reliability of observed scores = %.4f; disattenuated rho = %.3f  [disattenuation.txt]'
      % (reliability, rho_memorability_disattenuated))
print('  (a) rho model vs observed hits = %.4f (p = %.3g); false alarms = %.4f (p = %.3g)'
      % (rho_hits, p_hits, rho_false_alarms, p_false_alarms))
print('  (a) rho vs original THINGS memorability: model %.4f (p = %.3g), observed %.4f (p = %.3g)'
      % (rho_model_vs_original, p_model_vs_original, rho_observed_vs_original, p_observed_vs_original))
print('  (a) model operating point: sigma %g, hit %.3f, false alarm %.3f  [fit_inversion.txt]'
      % (chosen_sigma, printed_model_hit, printed_model_fa))
print('  (a) critical-item observed rates (trial weighted): hit %.3f, false alarm %.3f'
      % ((items.hitRate * items.nOld).sum() / items.nOld.sum(), (items.faRate * items.nNew).sum() / items.nNew.sum()))
print('  (b, c) images in more than one set: %d; participants: %d  [paired_crowding.txt]'
      % (number_of_images, number_of_participants))
for name, observed, model in [('d-prime', observed_dprime, model_dprime),
                              ('hit rate', observed_hits, model_hits),
                              ('false alarm rate', observed_false_alarms, model_false_alarms)]:
    print('  %-28s observed %+.3f [%+.3f, %+.3f]   model %+.3f [%+.3f, %+.3f]   ratio %.2f'
          % (name, observed['mean'], observed['low'], observed['high'], model['mean'], model['low'], model['high'],
             model['mean'] / observed['mean']))
print('  observed d-prime split: %.3f from hits, %.3f from false alarms  [paired_crowding.txt]'
      % (dprime_from_hits, dprime_from_false_alarms))
