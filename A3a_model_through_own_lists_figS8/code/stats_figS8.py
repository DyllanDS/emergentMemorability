"""
stats_figS8.py: the uncertainty statistics of Fig. S8 (SI Appendix, Section 3), tests T31 and T32, on
this module's run of the model through the participants' own lists.

WHAT IT COMPUTES (rows of outputs/tests.csv; the convention is in statistics/README.md)
  T31  Fig. S8a and SI Appendix, Section 3, the memorability inversion experiment: Spearman rank
       correlations over the 40 critical items (n = 40, df = 38, two-sided p from scipy's spearmanr,
       which uses the t distribution) between
         - the model's predicted and the observed corrected recognition (CR = hit rate minus false
           alarm rate; Fig. S8a), and the same rho corrected for the reliability of the observed
           scores,
         - the model's predicted and the observed hit rates and false alarm rates,
         - the published memorability score (Kramer et al. 2023) and the model's CR, and the
           published score and the observed CR (the rho of Fig. 3e).
       Intervals: 95% percentile intervals over items, the 40 items resampled with replacement (each
       keeps its observed scores, its model prediction and its published score). The reliability
       behind the disattenuated rho (binomial sampling error of the items' CR, as
       disattenuate_inversion.py computes it) with its basis (median old and new trials per item).
  T32  Fig. S8b and c and SI Appendix, Section 3, the image set experiments: the paired same-image
       contrast over the images tested in two or more sets (least crowded minus most crowded set;
       d′ with rates kept between 1/(2N) and 1 - 1/(2N)).
         - The error bars of Fig. S8b and c: the observed and the model's changes in d′, hit rate
           and false alarm rate with the intervals of paired_crowding.py (seed 0, 10,000 image
           resamples), read from outputs/paired_crowding_summary.csv.
         - The observed change in d′: paired t test over images (t, df = images - 1, two-sided p,
           mean change with its t-based 95% interval, Cohen's d_z = mean / SD of the per-image
           changes).
         - With 10,000 image resamples of this script (seed 20261013): the model / observed ratio of
           the d′ change ("about a third") and the part of the observed d′ change carried by false
           alarms (SI Appendix, Section 3, "Similar Items Change Memory Mainly Through False Alarms";
           each rate moved half its mean change either side of its mean over the paired cells, as in
           paired_crowding.py), each with its image interval.
         - The numbers of images in two and in three sets.
  No correction for multiple comparisons (statistics/README.md).

INPUTS (inside the repository)
  outputs/fit_inversion_items.csv, outputs/disattenuation.txt (the inversion fit)
  outputs/paired_crowding_summary.csv, outputs/paired_crowding_images.csv, outputs/paired_crowding.txt (its
      d′ split, which this script checks), outputs/bin_by_crowding_cells.csv (the image set cells)
  03_inversion_experiment_fig3/data through shared/python/inversion_loader.py (the kept trials of the
      inversion experiment, for the participant resamples)
  03_inversion_experiment_fig3/outputs/tests.csv, if present: the T06 intervals; with 10,000 resamples
      the log says whether the intervals of the published score against observed CR here equal them
      (the same draws)

OUTPUT  outputs/tests.csv (T31, T32)

SEED 20261013 (numpy default_rng), one generator, drawn in this order:
  1. the participant resamples of the published score against observed CR (50 participants; the
     same draws as module 03's T06; compared with T06 in the log, not written to tests.csv)
  2. the item resamples of T31 (40 items; the same draws as module 03's T06)
  3. the image resamples of T32

RUN (from the repository root; under a minute):  python A3a_model_through_own_lists_figS8/code/stats_figS8.py
  (--resamples N sets a smaller number for a quick check of the code; the paper uses 10,000)
"""
import argparse
import os
import re
import sys
import time

# Use at most two processor threads (repeated runs on one machine give identical results). Must be set
# before numpy is imported.
for thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(thread_setting, '2')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm, spearmanr, t as t_distribution, ttest_1samp  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (first: it stops Python from writing __pycache__ folders)
import bootstrap  # noqa: E402
import inversion_loader  # noqa: E402
import tests_table  # noqa: E402

PATHS = release_paths.module_paths(__file__)
OUTPUTS = PATHS['outputs']
SCRIPT_NAME = 'code/stats_figS8.py'
SEED = bootstrap.RELEASE_SEED
T31_LEDGER_IDS = 'L7-004;L7-005;L7-006;L7-011;L7-016;L7-017;L3-045'
T32_LEDGER_IDS = 'L7-007;L7-008;L7-009;L7-010;L7-012;L7-013;L7-019;L7-020;L7-022;L7-026'
SECTION_3 = 'SI Appendix, Section 3, The Model Predicts Memory From Participants\' Own Lists'


def spearman_rho(first_values, second_values):
    """Spearman rank correlation (tied values get their average rank); NaN if a value is missing."""
    first_values = np.asarray(first_values, dtype=float)
    second_values = np.asarray(second_values, dtype=float)
    if not (np.all(np.isfinite(first_values)) and np.all(np.isfinite(second_values))):
        return float('nan')
    return float(spearmanr(first_values, second_values).statistic)


def rates_from_counts(old_trials, hits, new_trials, false_alarms):
    """Hit rate, false alarm rate and CR of each item from summed counts (NaN without old or new trials)."""
    hit_rate = np.full(len(old_trials), np.nan)
    false_alarm_rate = np.full(len(new_trials), np.nan)
    for item in range(len(old_trials)):
        if old_trials[item] > 0:
            hit_rate[item] = hits[item] / old_trials[item]
        if new_trials[item] > 0:
            false_alarm_rate[item] = false_alarms[item] / new_trials[item]
    return hit_rate, false_alarm_rate, hit_rate - false_alarm_rate


def inversion_count_matrices(kept_trials, participants, item_names):
    """Per participant (rows) and critical item (columns): old trials, hits, new trials, false alarms."""
    item_position = {name: position for position, name in enumerate(item_names)}
    participant_position = {name: position for position, name in enumerate(participants)}
    shape = (len(participants), len(item_names))
    old_trials = np.zeros(shape)
    hits = np.zeros(shape)
    new_trials = np.zeros(shape)
    false_alarms = np.zeros(shape)
    critical_trials = kept_trials[kept_trials['is_critical']]
    for participant, image_name, is_new, response in zip(critical_trials['participant'], critical_trials['image_name'],
                                                         critical_trials['is_new'], critical_trials['response_1to6']):
        row = participant_position[participant]
        column = item_position[image_name]
        said_old = 1 if response <= 3 else 0
        if is_new == 0:
            old_trials[row, column] += 1
            hits[row, column] += said_old
        else:
            new_trials[row, column] += 1
            false_alarms[row, column] += said_old
    return old_trials, hits, new_trials, false_alarms


def summed_counts(counts, drawn_participants):
    """Add up the item counts of the drawn participants (a participant drawn twice counts twice)."""
    total = np.zeros(counts.shape[1])
    for participant_index in drawn_participants:
        total = total + counts[participant_index]
    return total


def clamped_rate(rate, number_of_trials):
    """A rate kept between 1/(2N) and 1 - 1/(2N) (paired_crowding.py's correction); NaN without trials."""
    rate = np.asarray(rate, dtype=float)
    number_of_trials = np.asarray(number_of_trials, dtype=float)
    lowest = np.where(number_of_trials > 0, 1.0 / (2.0 * np.maximum(number_of_trials, 1e-9)), np.nan)
    return np.clip(rate, lowest, 1.0 - lowest)


def dprime_from_rates(hit_rate, false_alarm_rate, old_trials, new_trials):
    return norm.ppf(clamped_rate(hit_rate, old_trials)) - norm.ppf(clamped_rate(false_alarm_rate, new_trials))


def dprime_split(hit_change, false_alarm_change, base_hit, base_false_alarm, mean_old_trials, mean_new_trials):
    """The d′ change carried by the hits and by the false alarms: each rate moved half its change
    either side of its base rate (paired_crowding.py, dprime_parts)."""
    hits_part = (norm.ppf(clamped_rate(base_hit + hit_change / 2, mean_old_trials))
                 - norm.ppf(clamped_rate(base_hit - hit_change / 2, mean_old_trials)))
    false_alarms_part = -(norm.ppf(clamped_rate(base_false_alarm + false_alarm_change / 2, mean_new_trials))
                          - norm.ppf(clamped_rate(base_false_alarm - false_alarm_change / 2, mean_new_trials)))
    return hits_part, false_alarms_part


def paired_t(values):
    """One-sample t test of per-image changes against 0: t, df, two-sided p, mean, t-based 95%
    interval, Cohen's d_z (mean / SD of the changes)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    result = ttest_1samp(values, 0)
    half_width = t_distribution.ppf(0.975, count - 1) * standard_deviation / np.sqrt(count)
    return dict(t=float(result.statistic), df=count - 1, p=float(result.pvalue), mean=mean,
                low=mean - half_width, high=mean + half_width, d_z=mean / standard_deviation, n=count)


def module_03_t06_intervals():
    """The T06 intervals of module 03 (participants and items), if its tests.csv exists."""
    path = os.path.join(release_paths.find_module('03'), 'outputs', 'tests.csv')
    if not os.path.exists(path):
        return None
    table = pd.read_csv(path)
    rows = table[(table['test_id'] == 'T06') & (table['statistic_name'] == 'Spearman rho')
                 & (table['location'].str.contains('Fig. 3e'))]
    intervals = {}
    for _, row in rows.iterrows():
        if row['resampling_unit'] == 'participants':
            intervals['participants'] = (float(row['ci_low']), float(row['ci_high']))
        elif str(row['resampling_unit']).startswith('items'):
            intervals['items'] = (float(row['ci_low']), float(row['ci_high']))
    return intervals


def main():
    parser = argparse.ArgumentParser(description='Uncertainty statistics of Fig. S8 (T31, T32)')
    parser.add_argument('--resamples', type=int, default=10000, help='bootstrap resamples (10,000 in the paper)')
    arguments = parser.parse_args()
    number_of_resamples = arguments.resamples
    started = time.time()
    print('stats_figS8.py: uncertainty statistics of Fig. S8 (tests T31 and T32)')
    print('seed %d; %d resamples per bootstrap' % (SEED, number_of_resamples))
    if number_of_resamples != 10000:
        print('NOTE: fewer than the 10,000 resamples of the paper; a check of the code only')
    tests = tests_table.TestsTable(script=SCRIPT_NAME)
    generator = np.random.default_rng(SEED)

    # =========================================================================================
    # T31: the inversion experiment, 40 critical items
    # =========================================================================================
    items = pd.read_csv(os.path.join(OUTPUTS, 'fit_inversion_items.csv'), float_precision='round_trip')
    item_names = list(items['imageName'])
    number_of_items = len(item_names)
    observed_hit_rate = items['hitRate'].to_numpy(dtype=float)
    observed_false_alarm_rate = items['faRate'].to_numpy(dtype=float)
    observed_cr = items['cr'].to_numpy(dtype=float)
    predicted_hit_rate = items['pred_hit'].to_numpy(dtype=float)
    predicted_false_alarm_rate = items['pred_fa'].to_numpy(dtype=float)
    predicted_cr = items['pred_cr'].to_numpy(dtype=float)
    published_score = items['origMem'].to_numpy(dtype=float)

    # the five correlations, each with its two-sided p from spearmanr (t distribution, n - 2 = 38 df)
    correlations = [
        ('model vs observed CR', predicted_cr, observed_cr, 'observed'),
        ('model vs observed hit rate', predicted_hit_rate, observed_hit_rate, 'observed'),
        ('model vs observed false alarm rate', predicted_false_alarm_rate, observed_false_alarm_rate, 'observed'),
        ('published score vs model CR', published_score, predicted_cr, 'fixed'),
        ('published score vs observed CR', published_score, observed_cr, 'observed'),
    ]
    point = {}
    for name, first_values, second_values, _ in correlations:
        result = spearmanr(first_values, second_values)
        point[name] = (float(result.statistic), float(result.pvalue))
        print('T31 %-36s rho = %+.4f, p = %.4g (n = %d, df = %d)' % (name, result.statistic, result.pvalue,
                                                                    number_of_items, number_of_items - 2))

    # per-participant counts of the 50 kept participants (the trials behind the observed scores)
    test_trials = inversion_loader.load_trials('test')
    kept_trials, _ = inversion_loader.apply_exclusions(test_trials)
    participants = sorted(kept_trials['participant'].unique())
    number_of_participants = len(participants)
    old_trials, hits, new_trials, false_alarms = inversion_count_matrices(kept_trials, participants, item_names)
    for name, counts, column in (('old trials', old_trials, 'nOld'), ('new trials', new_trials, 'nNew')):
        if not np.array_equal(counts.sum(axis=0), items[column].to_numpy(dtype=float)):
            sys.exit('stats_figS8.py: the per-participant %s do not add up to fit_inversion_items.csv' % name)
    summed_hit_rate, summed_false_alarm_rate, _ = rates_from_counts(old_trials.sum(axis=0), hits.sum(axis=0),
                                                                     new_trials.sum(axis=0), false_alarms.sum(axis=0))
    if not (np.allclose(summed_hit_rate, observed_hit_rate, rtol=0, atol=1e-12)
            and np.allclose(summed_false_alarm_rate, observed_false_alarm_rate, rtol=0, atol=1e-12)):
        sys.exit('stats_figS8.py: the per-participant counts do not reproduce the observed rates')
    print('T31 %d kept participants; their counts add up to the item table' % number_of_participants)

    # draws 1: participants (the observed CR recomputed, the published scores fixed), for the check
    # against module 03's T06 below
    participant_resampled = {'published score vs observed CR': np.empty(number_of_resamples)}
    for resample in range(number_of_resamples):
        drawn_participants = generator.integers(0, number_of_participants, number_of_participants)
        _, _, resampled_cr = rates_from_counts(
            summed_counts(old_trials, drawn_participants), summed_counts(hits, drawn_participants),
            summed_counts(new_trials, drawn_participants), summed_counts(false_alarms, drawn_participants))
        participant_resampled['published score vs observed CR'][resample] = spearman_rho(published_score,
                                                                                         resampled_cr)

    # draws 2: items (each item keeps all of its values)
    item_resampled = {name: np.empty(number_of_resamples) for name, _, _, _ in correlations}
    for resample in range(number_of_resamples):
        drawn_items = generator.integers(0, number_of_items, number_of_items)
        for name, first_values, second_values, _ in correlations:
            item_resampled[name][resample] = spearman_rho(first_values[drawn_items], second_values[drawn_items])

    # the reliability of the observed CR and the disattenuated rho (disattenuate_inversion.py's formula)
    sampling_variance = (observed_hit_rate * (1 - observed_hit_rate) / items['nOld']
                         + observed_false_alarm_rate * (1 - observed_false_alarm_rate) / items['nNew'])
    reliability = 1 - float(np.mean(sampling_variance)) / float(np.var(observed_cr, ddof=1))
    median_old = float(np.median(items['nOld']))
    median_new = float(np.median(items['nNew']))
    disattenuated_rho = point['model vs observed CR'][0] / np.sqrt(reliability)
    disattenuation_text = open(os.path.join(OUTPUTS, 'disattenuation.txt')).read()
    reported_reliability = float(re.search(r'reliability = 1 - share = ([\d.]+)', disattenuation_text).group(1))
    if abs(reported_reliability - reliability) > 5e-5:
        sys.exit('stats_figS8.py: the reliability differs from outputs/disattenuation.txt')
    print('T31 reliability of the observed CR %.4f (median %g old and %g new trials per item); disattenuated '
          'rho %.4f' % (reliability, median_old, median_new, disattenuated_rho))

    # check: the published-score intervals equal module 03's T06 intervals (same draws, same data)
    t06 = module_03_t06_intervals()
    participant_low, participant_high = bootstrap.percentile_interval(
        participant_resampled['published score vs observed CR'])
    item_low, item_high = bootstrap.percentile_interval(item_resampled['published score vs observed CR'])
    if t06 is None or number_of_resamples != 10000:
        print('T31 check against module 03 T06 skipped (no 03 tests.csv, or not 10,000 resamples)')
    else:
        same_participants = ('participants' in t06 and
                             np.allclose(t06['participants'], (participant_low, participant_high), atol=1e-5))
        same_items = 'items' in t06 and np.allclose(t06['items'], (item_low, item_high), atol=1e-5)
        print('T31 published score vs observed CR: participant interval [%.6f, %.6f] %s module 03 T06; item '
              'interval [%.6f, %.6f] %s module 03 T06'
              % (participant_low, participant_high, 'equals' if same_participants else 'DIFFERS FROM',
                 item_low, item_high, 'equals' if same_items else 'DIFFERS FROM'))

    t31_rows = [
        ('model vs observed CR', 'Fig. S8a: Spearman rho over the 40 critical items between the model\'s '
         'predicted and the observed memorability (hit rate minus false alarm rate; the 50 participants of '
         'Fig. 3e); the model\'s noise and criterion were set from the overall hit and false alarm rates only',
         SECTION_3 + ' (Fig. S8a)'),
        ('model vs observed hit rate', 'Spearman rho over the 40 critical items between the model\'s predicted '
         'and the observed hit rate', SECTION_3),
        ('model vs observed false alarm rate', 'Spearman rho over the 40 critical items between the model\'s '
         'predicted and the observed false alarm rate', SECTION_3),
        ('published score vs model CR', 'Spearman rho over the 40 critical items between the published '
         'memorability score (Kramer et al. 2023) and the model\'s predicted memorability (hit rate minus false '
         'alarm rate): the inversion in the model', SECTION_3),
        ('published score vs observed CR', 'Spearman rho over the 40 critical items between the published '
         'score and the observed memorability (hit rate minus false alarm rate) of the same 50 participants: '
         'the rho of Fig. 3e (module 03, test T06)', SECTION_3),
    ]
    for name, description, location in t31_rows:
        rho, p_value = point[name]
        item_low, item_high = bootstrap.percentile_interval(item_resampled[name])
        tests.add(test_id='T31', ledger_ids=T31_LEDGER_IDS, location=location, test=description + '; interval: '
                  'the 40 items resampled', statistic_name='Spearman rho', statistic=rho, n=number_of_items,
                  unit='items (%d participants)' % number_of_participants, df=number_of_items - 2, p=p_value,
                  p_type='parametric two-sided', effect_size_name='rho', effect_size=rho, ci_low=item_low,
                  ci_high=item_high, ci_method='percentile bootstrap', resampling_unit='items',
                  n_resamples=number_of_resamples, seed=SEED)
        print('T31 %-36s item CI [%.4f, %.4f]' % (name, item_low, item_high))
    tests.add(test_id='T31', ledger_ids=T31_LEDGER_IDS, location=SECTION_3 + ' (Fig. S8a)',
              test='Fig. S8a rho disattenuated for the reliability of the observed CR: rho / sqrt(reliability); '
                   'reliability = 1 - mean over items of [h(1 - h)/n_old + f(1 - f)/n_new] / variance of CR over '
                   'items (n - 1); median %g old and %g new trials per item; only the observed CR is treated as '
                   'noisy' % (median_old, median_new),
              statistic_name='disattenuated Spearman rho', statistic=disattenuated_rho, n=number_of_items,
              unit='items (%d participants)' % number_of_participants, df='', p='', p_type='none',
              effect_size_name='reliability of the observed CR (binomial sampling error)', effect_size=reliability,
              ci_low='', ci_high='', ci_method='', resampling_unit='', n_resamples='', seed='')

    # =========================================================================================
    # T32: the image set experiments, the same image in two or three sets
    # =========================================================================================
    images = pd.read_csv(os.path.join(OUTPUTS, 'paired_crowding_images.csv'), float_precision='round_trip')
    summary = pd.read_csv(os.path.join(OUTPUTS, 'paired_crowding_summary.csv'), float_precision='round_trip')
    number_of_images = len(images)
    images_in_two_sets = int(np.sum(images['n_sets'] == 2))
    images_in_three_sets = int(np.sum(images['n_sets'] == 3))
    print('\nT32 %d images in two or more sets (%d in two, %d in three)' % (number_of_images, images_in_two_sets,
                                                                          images_in_three_sets))
    # the observed and the model's per-image changes (columns of paired_crowding_images.csv); the third
    # element of each entry names the rows of paired_crowding_summary.csv (keep it as written there)
    measures = [('d_dprime', 'model_d_dprime', 'd-prime', 'change in d′'),
                ('d_hit', 'model_d_hit', 'hit rate', 'change in hit rate'),
                ('d_fa', 'model_d_fa', 'false alarm rate', 'change in false alarm rate')]

    # (a) the error bars of Fig. S8b and c (paired_crowding.py, seed 0); these rows are read from
    # outputs/paired_crowding_summary.csv, so their script column names paired_crowding.py
    for observed_column, model_column, summary_name, label in measures:
        for source in ('observed', 'model'):
            row = summary[(summary['measure'] == summary_name) & (summary['source'] == source)].iloc[0]
            tests.add(test_id='T32', ledger_ids=T32_LEDGER_IDS, location='SI Appendix, Fig. S8 caption (Fig. S8b, c)',
                      test='Paired same-image contrast, %s, %s (least minus most crowded set); interval: images '
                           'resampled (paired_crowding.py, seed 0), as in the error bars of Fig. S8b and c'
                           % (source, label),
                      statistic_name='mean %s' % label, statistic=float(row['mean']), n=number_of_images,
                      unit='images', df='', p='', p_type='none', effect_size_name='%s' % label,
                      effect_size=float(row['mean']), ci_low=float(row['lo_images']), ci_high=float(row['hi_images']),
                      ci_method='percentile bootstrap (paired_crowding.py)', resampling_unit='images',
                      n_resamples=10000, seed=0, script='code/paired_crowding.py')

    # (b) image resamples from this script's generator (draws 3)
    observed_changes = {column: images[column].to_numpy(dtype=float) for column, _, _, _ in measures}
    model_changes = {column: images[column].to_numpy(dtype=float) for _, column, _, _ in measures}
    image_resampled_observed = {column: np.empty(number_of_resamples) for column in observed_changes}
    image_resampled_model = {column: np.empty(number_of_resamples) for column in model_changes}
    image_resampled_ratio = np.empty(number_of_resamples)
    for resample in range(number_of_resamples):
        drawn_images = generator.integers(0, number_of_images, number_of_images)
        for column, values in observed_changes.items():
            image_resampled_observed[column][resample] = np.nanmean(values[drawn_images])
        for column, values in model_changes.items():
            image_resampled_model[column][resample] = np.nanmean(values[drawn_images])
        image_resampled_ratio[resample] = (image_resampled_model['model_d_dprime'][resample]
                                           / image_resampled_observed['d_dprime'][resample])

    # the cells (crowding and rates) and the pairs: least and most crowded cell per image
    cells = pd.read_csv(os.path.join(OUTPUTS, 'bin_by_crowding_cells.csv'), float_precision='round_trip')
    cell_of = {}
    for position, (set_name, image_number) in enumerate(zip(cells['cond'], cells['image_nr'])):
        cell_of[(set_name, int(image_number))] = position
    least_crowded = np.array([cell_of[(set_name, int(number))] for set_name, number
                              in zip(images['cond_least'], images['image_nr'])])
    most_crowded = np.array([cell_of[(set_name, int(number))] for set_name, number
                             in zip(images['cond_most'], images['image_nr'])])
    cell_hit = cells['hit'].to_numpy(dtype=float)
    cell_false_alarm = cells['fa'].to_numpy(dtype=float)
    cell_old = cells['n_old'].to_numpy(dtype=float)
    cell_new = cells['n_new'].to_numpy(dtype=float)
    check_dprime = (dprime_from_rates(cell_hit[least_crowded], cell_false_alarm[least_crowded],
                                      cell_old[least_crowded], cell_new[least_crowded])
                    - dprime_from_rates(cell_hit[most_crowded], cell_false_alarm[most_crowded],
                                        cell_old[most_crowded], cell_new[most_crowded]))
    if not np.allclose(check_dprime, observed_changes['d_dprime'], rtol=0, atol=1e-12):
        sys.exit('stats_figS8.py: the cells do not reproduce the per-image d′ changes of paired_crowding_images.csv')

    # (c) the observed change in d′: paired t test over images
    observed_mean = float(np.nanmean(observed_changes['d_dprime']))
    t_result = paired_t(observed_changes['d_dprime'])
    tests.add(test_id='T32', ledger_ids=T32_LEDGER_IDS, location=SECTION_3,
              test='Paired t test over images of the observed change in d′ (least minus most crowded set) against '
                   '0, images weighted equally; the interval is the t-based 95% interval of the mean change',
              statistic_name='t', statistic=t_result['t'], n=t_result['n'], unit='images', df=t_result['df'],
              p=t_result['p'], p_type='parametric two-sided', effect_size_name='Cohen\'s d_z',
              effect_size=t_result['d_z'], ci_low=t_result['low'], ci_high=t_result['high'],
              ci_method='t-based 95% CI of the mean paired difference', resampling_unit='', n_resamples='', seed='')
    model_mean = float(np.nanmean(model_changes['model_d_dprime']))
    print('T32 change in d′: observed %+.4f, t(%d) = %.2f, p = %.3g, d_z = %.3f, t-based CI [%+.4f, %+.4f]; '
          'model %+.4f'
          % (observed_mean, t_result['df'], t_result['t'], t_result['p'], t_result['d_z'], t_result['low'],
             t_result['high'], model_mean))

    # the model / observed ratio of the d′ change ("about a third")
    ratio = float(np.nanmean(model_changes['model_d_dprime']) / np.nanmean(observed_changes['d_dprime']))
    ratio_low, ratio_high = bootstrap.percentile_interval(image_resampled_ratio)
    tests.add(test_id='T32', ledger_ids=T32_LEDGER_IDS, location=SECTION_3 + ' (Fig. S8b)',
              test='Ratio of the model\'s to the observed same-image change in d′; interval: images resampled '
                   '(the same resamples for both)',
              statistic_name='model / observed change in d′', statistic=ratio, n=number_of_images, unit='images',
              df='', p='', p_type='none', effect_size_name='ratio', effect_size=ratio, ci_low=ratio_low,
              ci_high=ratio_high, ci_method='percentile bootstrap', resampling_unit='images',
              n_resamples=number_of_resamples, seed=SEED)
    print('T32 model / observed d′ change %.3f [%.3f, %.3f]' % (ratio, ratio_low, ratio_high))

    # the part of the observed d′ change carried by the false alarms (SI Appendix, Section 3); the part
    # carried by the hits is checked with the false alarm part against paired_crowding.txt
    both_cells = np.concatenate([least_crowded, most_crowded])
    base_hit = float(np.mean(cell_hit[both_cells]))
    base_false_alarm = float(np.mean(cell_false_alarm[both_cells]))
    mean_old_trials = float(np.mean(cell_old[both_cells]))
    mean_new_trials = float(np.mean(cell_new[both_cells]))
    paired_text = open(os.path.join(OUTPUTS, 'paired_crowding.txt')).read()
    printed_split = re.search(r'Split of the paired d-prime effect: observed ([-\d.]+) from hits and ([-\d.]+) from '
                              r'false\s+alarms; model ([-\d.]+) from hits and ([-\d.]+) from false alarms', paired_text)
    hits_part, false_alarms_part = dprime_split(float(np.nanmean(images['d_hit'])), float(np.nanmean(images['d_fa'])),
                                                base_hit, base_false_alarm, mean_old_trials, mean_new_trials)
    if (abs(hits_part - float(printed_split.group(1))) > 6e-4
            or abs(false_alarms_part - float(printed_split.group(2))) > 6e-4):
        sys.exit('stats_figS8.py: the d′ split differs from paired_crowding.txt')
    _, resampled_false_alarms_part = dprime_split(image_resampled_observed['d_hit'], image_resampled_observed['d_fa'],
                                                  base_hit, base_false_alarm, mean_old_trials, mean_new_trials)
    low, high = bootstrap.percentile_interval(resampled_false_alarms_part)
    tests.add(test_id='T32', ledger_ids=T32_LEDGER_IDS,
              location='SI Appendix, Section 3, Similar Items Change Memory Mainly Through False Alarms (Fig. S8c)',
              test='The observed same-image change in d′ carried by the false alarms: each rate moved half its mean '
                   'change either side of its mean over the paired cells (hit %.3f, false alarm %.3f), as '
                   'paired_crowding.py does; interval: images resampled, base rates fixed'
                   % (base_hit, base_false_alarm),
              statistic_name='d′ change from false alarms', statistic=false_alarms_part, n=number_of_images,
              unit='images', df='', p='', p_type='none', effect_size_name='d′ change from false alarms',
              effect_size=false_alarms_part, ci_low=low, ci_high=high, ci_method='percentile bootstrap',
              resampling_unit='images', n_resamples=number_of_resamples, seed=SEED)
    print('T32 observed d′ split: %.3f from hits, %.3f from false alarms [%.3f, %.3f]'
          % (hits_part, false_alarms_part, low, high))
    model_hits_wrong_way = np.sign(np.nanmean(images['model_d_hit'])) != np.sign(np.nanmean(images['d_hit']))
    print('T32 the model\'s hit change has %s sign as the observed (model %+.4f, observed %+.4f)'
          % ('the OPPOSITE' if model_hits_wrong_way else 'the same', np.nanmean(images['model_d_hit']),
             np.nanmean(images['d_hit'])))

    tests.add(test_id='T32', ledger_ids=T32_LEDGER_IDS,
              location='SI Appendix, Section 3, The Model Predicts Memory From Participants\' Own Lists; Fig. S8 '
                       'caption (Fig. S8b)',
              test='Images tested in two or more image sets with at least 5 old and 5 new trials in each: %d in two '
                   'sets, %d in three' % (images_in_two_sets, images_in_three_sets),
              statistic_name='images', statistic=number_of_images, n=number_of_images, unit='images', df='', p='',
              p_type='none', effect_size_name='', effect_size='', ci_low='', ci_high='', ci_method='',
              resampling_unit='', n_resamples='', seed='')

    tests.write(OUTPUTS)
    print('stats_figS8.py finished in %.0f s' % (time.time() - started))


if __name__ == '__main__':
    main()
