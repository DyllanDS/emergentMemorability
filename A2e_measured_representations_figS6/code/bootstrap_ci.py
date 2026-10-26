"""
bootstrap_ci.py: the 95% bootstrap confidence intervals drawn as the error bars of Fig. S6.

WHAT THIS DOES
    Computes the two results of the continuous-feature REM run in the SPoSE space from the item-level
    results that continuous_rem_spose.m saves, and the 95% bootstrap intervals of the designed lists.

    1. Between items: the Spearman correlation between each concept's memorability (hit rate minus
       false alarm rate, over 2,500 simulated study-test experiments) and its crowding (minus the mean
       correlation distance to its 20 nearest concepts), with its two-sided p.
    2. Designed lists: the change in d′ when a critical concept is studied with 16 close neighbors
       rather than with none (120 critical concepts, 100 simulated participants per list). The
       critical concepts are resampled with replacement 2,000 times and the d′ change is recomputed
       from the resampled mean hit and false alarm rates, with the rate limits of the simulation (rates
       kept between 1/200 and 1 - 1/200).

    Random numbers: numpy default_rng(20261010), one generator for the whole script, drawing in this
    order: for each noise level (0.02, 0.03, 0.05, 0.08), first 2,000 resamples of the concepts, which
    no output uses, then the 2,000 resamples of the critical concepts. The intervals from 10,000
    resamples (seed 20261013) in tests.csv are computed by stats_figS6.py.

READS   outputs/continuous_rem_items_sigma<noise>.csv      (written by continuous_rem_spose.m)
        outputs/continuous_rem_designed_sigma<noise>.csv
WRITES  outputs/continuous_rem_bootstrap_ci.csv   one row per noise level (point values, p, intervals,
                                                  mean rates)
        outputs/continuous_rem_bootstrap_ci.txt   the same, readable

Run (from the repository root; seconds):
    python A2e_measured_representations_figS6/code/bootstrap_ci.py
"""
import os
import sys
import csv

import numpy as np
from scipy.stats import norm, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402

paths = release_paths.module_paths(__file__)
OUTPUT_FOLDER = paths['outputs']

NOISE_VALUES = ['0.02', '0.03', '0.05', '0.08']
NUMBER_OF_RESAMPLES = 2000
SEED = 20261010
SIMULATED_PARTICIPANTS_PER_LIST = 100          # participants per designed list in continuous_rem_spose.m


def read_csv_columns(path):
    """Read a csv file into a dictionary: column name -> numpy array of numbers."""
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    columns = {}
    for name in rows[0].keys():
        columns[name] = np.array([float(row[name]) for row in rows])
    return columns


def z_score_of_rate(rate):
    """Inverse normal of a rate, with the rate limits of the simulation's designed lists."""
    lowest = 1.0 / (2 * SIMULATED_PARTICIPANTS_PER_LIST)
    clipped = min(max(rate, lowest), 1 - lowest)
    return norm.ppf(clipped)


def designed_d_prime_change(hit_0, fa_0, hit_16, fa_16):
    """d′ with 16 close neighbors minus d′ with none, from the mean rates across critical concepts."""
    d_prime_0 = z_score_of_rate(np.mean(hit_0)) - z_score_of_rate(np.mean(fa_0))
    d_prime_16 = z_score_of_rate(np.mean(hit_16)) - z_score_of_rate(np.mean(fa_16))
    return d_prime_16 - d_prime_0


print('Bootstrap intervals of Fig. S6 (its error bars): %d resamples, numpy default_rng(%d)'
      % (NUMBER_OF_RESAMPLES, SEED))
random_generator = np.random.default_rng(SEED)
result_rows = []
for noise in NOISE_VALUES:
    # ---------------- between items ----------------
    items = read_csv_columns(os.path.join(OUTPUT_FOLDER, 'continuous_rem_items_sigma%s.csv' % noise))
    tested_both_ways = (items['old_tests'] > 0) & (items['new_tests'] > 0)   # as in the simulation
    hit_rate = items['hit_count'][tested_both_ways] / items['old_tests'][tested_both_ways]
    false_alarm_rate = items['false_alarm_count'][tested_both_ways] / items['new_tests'][tested_both_ways]
    memorability = hit_rate - false_alarm_rate
    crowding = items['crowding_local20'][tested_both_ways]
    number_of_items = len(memorability)

    correlation, p_value = spearmanr(memorability, crowding)
    # 2,000 resamples of the concepts, drawn at this point of the random stream (no output uses them)
    for resample in range(NUMBER_OF_RESAMPLES):
        random_generator.integers(0, number_of_items, number_of_items)

    # ---------------- designed lists ----------------
    designed = read_csv_columns(os.path.join(OUTPUT_FOLDER, 'continuous_rem_designed_sigma%s.csv' % noise))
    hit_0 = designed['hit_rate_0_neighbors']
    fa_0 = designed['false_alarm_rate_0_neighbors']
    hit_16 = designed['hit_rate_16_neighbors']
    fa_16 = designed['false_alarm_rate_16_neighbors']
    number_of_critical_items = len(hit_0)

    change = designed_d_prime_change(hit_0, fa_0, hit_16, fa_16)
    resampled_changes = []
    for resample in range(NUMBER_OF_RESAMPLES):
        chosen = random_generator.integers(0, number_of_critical_items, number_of_critical_items)
        resampled_changes.append(designed_d_prime_change(hit_0[chosen], fa_0[chosen],
                                                         hit_16[chosen], fa_16[chosen]))
    change_low, change_high = np.percentile(resampled_changes, [2.5, 97.5])

    result_rows.append({
        'noise_fraction_of_sd': noise,
        'mean_hit_rate': np.mean(hit_rate),
        'mean_false_alarm_rate': np.mean(false_alarm_rate),
        'n_items': number_of_items,
        'spearman_memorability_vs_local_crowding': correlation,
        'spearman_p': p_value,
        'n_critical_items': number_of_critical_items,
        'hit_rate_0_neighbors': np.mean(hit_0), 'false_alarm_rate_0_neighbors': np.mean(fa_0),
        'hit_rate_16_neighbors': np.mean(hit_16), 'false_alarm_rate_16_neighbors': np.mean(fa_16),
        'designed_d_prime_change': change,
        'designed_ci_low': change_low,
        'designed_ci_high': change_high,
    })

column_names = list(result_rows[0].keys())
with open(os.path.join(OUTPUT_FOLDER, 'continuous_rem_bootstrap_ci.csv'), 'w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=column_names)
    writer.writeheader()
    for row in result_rows:
        writer.writerow(row)

lines = ['CONTINUOUS-FEATURE REM IN SPoSE: 95% BOOTSTRAP CONFIDENCE INTERVALS',
         '%d resamples of the critical items of the designed lists.' % NUMBER_OF_RESAMPLES,
         '',
         ' noise   hit     FA    n items   rho (local)  p              n crit   d\' change   [95% CI]']
for row in result_rows:
    lines.append(' %s  %.3f  %.3f  %6d     %+.3f        %.2e  %6d     %+.3f    [%+.3f, %+.3f]' % (
        row['noise_fraction_of_sd'], row['mean_hit_rate'], row['mean_false_alarm_rate'],
        row['n_items'], row['spearman_memorability_vs_local_crowding'], row['spearman_p'],
        row['n_critical_items'], row['designed_d_prime_change'], row['designed_ci_low'],
        row['designed_ci_high']))
lines.append('')
lines.append('Designed lists, mean rates (0 neighbors -> 16 neighbors):')
for row in result_rows:
    lines.append(' noise %s: hit %.3f -> %.3f, false alarm %.3f -> %.3f' % (
        row['noise_fraction_of_sd'], row['hit_rate_0_neighbors'], row['hit_rate_16_neighbors'],
        row['false_alarm_rate_0_neighbors'], row['false_alarm_rate_16_neighbors']))
text = '\n'.join(lines) + '\n'
with open(os.path.join(OUTPUT_FOLDER, 'continuous_rem_bootstrap_ci.txt'), 'w') as handle:
    handle.write(text)
print(text)
print('wrote outputs/continuous_rem_bootstrap_ci.csv and outputs/continuous_rem_bootstrap_ci.txt')
