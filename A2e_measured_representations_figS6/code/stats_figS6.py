"""
stats_figS6.py: the uncertainty statistics of Fig. S6 (test T29 of statistics/README.md) -> outputs/tests.csv

WHAT THIS DOES
    For REM run in the measured SPoSE space (SI Appendix, Section 2, Measured Representations; Fig. S6),
    in the convention of statistics/README.md:
    1. The error bars of Fig. S6 (bootstrap_ci.py: 2,000 resamples, numpy default_rng(20261010)), read
       from outputs/continuous_rem_bootstrap_ci.csv.
    2. Intervals from 10,000 percentile-bootstrap resamples, seed 20261013, for
       - the Spearman correlation between memorability and local crowding (n = 1,600 concepts, the
         concepts resampled), at noise 0.02, 0.03 and 0.05;
       - the designed-list d′ change, 16 minus 0 close neighbors (n = 120 critical concepts, the
         critical concepts resampled), and the changes in the false alarm rate to the lure and in the
         hit rate behind it (SI Appendix, Section 2: "because false alarms to its nearest neighbor
         rose"; the lure is the nearest neighbor).
       Each interval uses its own generator numpy default_rng(20261013), so the three noise levels are
       resampled with the same 10,000 sets of concepts (and of critical concepts).
    3. The white bar (REM with generated features at the Fig. 1 parameters): its value is read from the
       saved run of module A2d (d′ change -0.13, designed lists CELL 2). Its item-bootstrap interval
       is test T28 (module A2d) and is not recomputed here.
    4. The comparison in SI Appendix, Section 2: does each gray-bar interval lie below the white bar's
       value ("more than with REM's generated features")?
    The two-sided Spearman p is listed with each correlation (SI Appendix, Section 2: all p < 0.001);
    the d′ and rate changes carry no p. No correction for multiple comparisons is applied.
    The words that describe the model in tests.csv are built from the settings of the run
    (outputs/continuous_rem_settings.csv).

READS   outputs/continuous_rem_items_sigma<noise>.csv, outputs/continuous_rem_designed_sigma<noise>.csv,
        outputs/continuous_rem_settings.csv          (continuous_rem_spose.m)
        outputs/continuous_rem_bootstrap_ci.csv      (bootstrap_ci.py)
        ../A2d_list_context_figS5/data/paper_run/list_designed.txt               (CELL 2)
WRITES  outputs/tests.csv
SEED    20261013 (the 10,000-resample intervals); the error bars of Fig. S6 use 20261010 (bootstrap_ci.py)

Run (from the repository root; about 25 seconds):
    python A2e_measured_representations_figS6/code/stats_figS6.py
"""
import csv
import os
import re
import sys

import numpy as np
from scipy.stats import norm, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402
import bootstrap  # noqa: E402
import tests_table  # noqa: E402

paths = release_paths.module_paths(__file__)
OUTPUT_FOLDER = paths['outputs']
REM_DESIGNED_FILE = os.path.join(release_paths.find_module('A2d'), 'data', 'paper_run', 'list_designed.txt')

SCRIPT = 'code/stats_figS6.py'
SEED = bootstrap.RELEASE_SEED                     # 20261013
NUMBER_OF_RESAMPLES = 10000
PAPER_SEED = 20261010                             # bootstrap_ci.py, the error bars of Fig. S6
PAPER_RESAMPLES = 2000
NOISE_VALUES_SHOWN = ['0.02', '0.03', '0.05']     # the gray bars of Fig. S6 (the simulation also runs 0.08)

TEXT_LOCATION = 'SI Appendix, Section 2, Measured Representations'
RHO_LEDGER_IDS = 'L6-087;L6-095'
CHANGE_LEDGER_IDS = 'L6-088;L6-100'
RHO_LOCATION = TEXT_LOCATION
CHANGE_LOCATION = TEXT_LOCATION + '; Fig. S6'
D_PRIME_NAME = 'change in d′ (16 minus 0 close neighbors)'
PAPER_METHOD = 'percentile bootstrap (the error bars of Fig. S6)'
RELEASE_METHOD = 'percentile bootstrap'


# ------------------------------------------------------------------------- reading the run
def read_settings():
    """outputs/continuous_rem_settings.csv (the settings of the run, written by continuous_rem_spose.m):
    {setting: list of numbers}."""
    settings = {}
    with open(os.path.join(OUTPUT_FOLDER, 'continuous_rem_settings.csv'), newline='') as handle:
        for row in csv.DictReader(handle):
            settings[row['setting']] = [float(value) for value in row['value'].split()]
    return settings


def describe_run(settings):
    """The words of the tests.csv rows that describe the model and the units, from the settings as run."""
    concepts = '{:,}'.format(int(settings['n_concepts_used'][0]))
    model = ('REM with continuous features in the measured SPoSE space (%s THINGS concepts, %d SPoSE dimensions; '
             'storage %.2f, copy %.2f, %d studied, %d old and %d new tested, criterion %.1f; rarity = density of '
             'the value across the %s concepts; simulation seed MATLAB rng(%d)), retention noise %%s of each '
             'dimension\'s SD'
             % (concepts, settings['n_features'][0], settings['storage_u'][0], settings['copy_c'][0],
                settings['n_studied'][0], settings['n_tested_old'][0], settings['n_tested_new'][0],
                settings['criterion'][0], concepts, settings['seed'][0]))
    neighbours = int(settings['n_close_neighbors_designed'][-1])
    designed_lists = ('designed lists, each critical concept studied on a list with %d close neighbors (its 2nd '
                      'to %dth nearest concepts) and on a list with none (the rest of the %d from the farther half '
                      'of the concepts), its nearest concept tested as the lure'
                      % (neighbours, neighbours + 1, settings['n_studied'][0]))
    crowding = ('local crowding (minus the mean correlation distance to the %d nearest concepts)'
                % settings['n_nearest_neighbors_crowding'][0])
    rho_unit = ('concepts (each with its hit and false alarm rate over {:,} simulated experiments)'
                .format(int(settings['n_experiments_between_items'][0])))
    change_unit = ('critical concepts (%d simulated participants per list)'
                   % settings['n_participants_per_list_designed'][0])
    return {'model': model, 'designed_lists': designed_lists, 'crowding': crowding, 'rho_unit': rho_unit,
            'change_unit': change_unit}


def read_csv_columns(path):
    """Read a csv file into a dictionary: column name -> numpy array of numbers."""
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    columns = {}
    for name in rows[0].keys():
        columns[name] = np.array([float(row[name]) for row in rows])
    return columns


def between_item_measures(folder, noise):
    """Memorability (hit rate minus false alarm rate) and local crowding of each concept tested at
    least once as old and once as new (all 1,600 with seed 1), as in continuous_rem_spose.m."""
    items = read_csv_columns(os.path.join(folder, 'continuous_rem_items_sigma%s.csv' % noise))
    tested_both_ways = (items['old_tests'] > 0) & (items['new_tests'] > 0)
    hit_rate = items['hit_count'][tested_both_ways] / items['old_tests'][tested_both_ways]
    false_alarm_rate = items['false_alarm_count'][tested_both_ways] / items['new_tests'][tested_both_ways]
    memorability = hit_rate - false_alarm_rate
    crowding = items['crowding_local20'][tested_both_ways]
    return memorability, crowding


def designed_rates(folder, noise):
    """Hit and false alarm rates of each critical concept with 0 and with 16 close neighbors."""
    designed = read_csv_columns(os.path.join(folder, 'continuous_rem_designed_sigma%s.csv' % noise))
    return (designed['hit_rate_0_neighbors'], designed['false_alarm_rate_0_neighbors'],
            designed['hit_rate_16_neighbors'], designed['false_alarm_rate_16_neighbors'])


def z_score_of_rate(rate, participants_per_list):
    """Inverse normal of a rate kept between 1/(2N) and 1 - 1/(2N), N = participants per list."""
    lowest = 1.0 / (2 * participants_per_list)
    clipped = min(max(rate, lowest), 1 - lowest)
    return norm.ppf(clipped)


def designed_changes(hit_0, fa_0, hit_16, fa_16, participants_per_list):
    """[d′ change, false alarm rate change, hit rate change], 16 minus 0 close neighbors, from the mean
    rates across the critical concepts (the d′ as in continuous_rem_spose.m)."""
    d_prime_0 = (z_score_of_rate(np.mean(hit_0), participants_per_list)
                 - z_score_of_rate(np.mean(fa_0), participants_per_list))
    d_prime_16 = (z_score_of_rate(np.mean(hit_16), participants_per_list)
                  - z_score_of_rate(np.mean(fa_16), participants_per_list))
    false_alarm_change = np.mean(fa_16) - np.mean(fa_0)
    hit_change = np.mean(hit_16) - np.mean(hit_0)
    return np.array([d_prime_16 - d_prime_0, false_alarm_change, hit_change])


def read_paper_run_intervals():
    """The rows of outputs/continuous_rem_bootstrap_ci.csv (the 2,000-resample error bars of Fig. S6), by
    noise."""
    rows = {}
    with open(os.path.join(OUTPUT_FOLDER, 'continuous_rem_bootstrap_ci.csv')) as handle:
        for row in csv.DictReader(handle):
            rows[row['noise_fraction_of_sd']] = row
    return rows


# ------------------------------------------------------------------------- the white bar
def read_white_bar_designed():
    """REM with generated features at the Fig. 1 values on designed lists: CELL 2 of list_designed.txt
    (the saved run of module A2d), read by its row labels: the d′ change from 0 to 16 similar items,
    the number of critical items and of simulated participants per list."""
    inside_cell_2 = False
    change = None
    critical_items = None
    participants = None
    with open(REM_DESIGNED_FILE) as handle:
        for line in handle:
            if line.startswith('CELL 2'):
                inside_cell_2 = True
                continue
            if line.startswith('CELL 3') or line.strip() == 'READING THIS':
                inside_cell_2 = False
            if not inside_cell_2:
                continue
            design = re.search(r'(\d+) critical items x (\d+) conditions x (\d+) participants', line)
            if design:
                critical_items = int(design.group(1))
                participants = int(design.group(3))
            if '0 -> 16 similar items' in line:
                change = float(re.search(r'd-prime\s+([-+]?\d*\.\d+)', line).group(1))
    if change is None or critical_items is None:
        sys.exit('stats_figS6.py: CELL 2 of %s could not be read' % REM_DESIGNED_FILE)
    return change, critical_items, participants


# ------------------------------------------------------------------------- the correlation
def add_correlation_rows(tests, words):
    """SI Appendix, Section 2: at each noise level, the correlation with its interval from 10,000
    resamples of the concepts."""
    for noise in NOISE_VALUES_SHOWN:
        memorability, crowding = between_item_measures(OUTPUT_FOLDER, noise)
        number_of_concepts = len(memorability)
        rho, p_value = spearmanr(memorability, crowding)
        rho = float(rho)
        p_value = float(p_value)
        test_text = (words['model'] % noise + ': Spearman correlation across concepts between memorability (hit '
                     'rate minus false alarm rate) and ' + words['crowding'])

        def resampled_rho(indices):
            return spearmanr(memorability[indices], crowding[indices])[0]
        resampled = bootstrap.percentile_bootstrap(resampled_rho, number_of_concepts, NUMBER_OF_RESAMPLES, SEED)
        low, high = bootstrap.percentile_interval(resampled)
        tests.add(test_id='T29', ledger_ids=RHO_LEDGER_IDS, location=RHO_LOCATION, test=test_text,
                  statistic_name='Spearman rho', statistic=rho, n=number_of_concepts, unit=words['rho_unit'],
                  df=number_of_concepts - 2, p=p_value, p_type='parametric two-sided', effect_size_name='rho',
                  effect_size=rho, ci_low=float(low), ci_high=float(high), ci_method=RELEASE_METHOD,
                  resampling_unit='concepts', n_resamples=NUMBER_OF_RESAMPLES, seed=SEED)
        print('noise %s: rho %.4f, p %.3g, 10,000 resamples [%.3f, %.3f]' % (noise, rho, p_value, low, high))


# ------------------------------------------------------------------------- Fig. S6
def add_designed_list_rows(tests, words, paper_run, participants_per_list):
    """Fig. S6: at each noise level, the d′ interval drawn in the figure and the intervals from 10,000
    resamples of the d′, false alarm and hit rate changes. Returns both d′ intervals, by noise."""
    paper_intervals = {}
    release_intervals = {}
    for noise in NOISE_VALUES_SHOWN:
        hit_0, fa_0, hit_16, fa_16 = designed_rates(OUTPUT_FOLDER, noise)
        number_of_critical_items = len(hit_0)
        changes = designed_changes(hit_0, fa_0, hit_16, fa_16, participants_per_list)
        d_prime_change = float(changes[0])
        false_alarm_change = float(changes[1])
        hit_change = float(changes[2])
        test_text = words['model'] % noise + ': ' + words['designed_lists']
        d_prime_text = test_text + ': change in d′ (16 minus 0 close neighbors) from the mean rates'

        # 1. the error bars of Fig. S6 (bootstrap_ci.py)
        paper_row = paper_run[noise]
        paper_change = float(paper_row['designed_d_prime_change'])
        paper_low = float(paper_row['designed_ci_low'])
        paper_high = float(paper_row['designed_ci_high'])
        paper_intervals[noise] = (paper_low, paper_high)
        tests.add(test_id='T29', ledger_ids=CHANGE_LEDGER_IDS, location=CHANGE_LOCATION, test=d_prime_text,
                  statistic_name=D_PRIME_NAME, statistic=paper_change, n=int(paper_row['n_critical_items']),
                  unit=words['change_unit'], df='', p='', p_type='none', effect_size_name='d′ difference',
                  effect_size=paper_change, ci_low=paper_low, ci_high=paper_high, ci_method=PAPER_METHOD,
                  resampling_unit='critical concepts', n_resamples=PAPER_RESAMPLES, seed=PAPER_SEED)

        # 2. 10,000 resamples of the critical concepts: d′, false alarm and hit rate changes on the same
        #    resamples
        def resampled_changes(indices):
            return designed_changes(hit_0[indices], fa_0[indices], hit_16[indices], fa_16[indices],
                                    participants_per_list)
        resampled = bootstrap.percentile_bootstrap(resampled_changes, number_of_critical_items,
                                                   NUMBER_OF_RESAMPLES, SEED)
        lows, highs = bootstrap.percentile_interval(resampled)
        release_intervals[noise] = (float(lows[0]), float(highs[0]))
        tests.add(test_id='T29', ledger_ids=CHANGE_LEDGER_IDS, location=CHANGE_LOCATION, test=d_prime_text,
                  statistic_name=D_PRIME_NAME, statistic=d_prime_change, n=number_of_critical_items,
                  unit=words['change_unit'], df='', p='', p_type='none', effect_size_name='d′ difference',
                  effect_size=d_prime_change, ci_low=float(lows[0]), ci_high=float(highs[0]),
                  ci_method=RELEASE_METHOD, resampling_unit='critical concepts', n_resamples=NUMBER_OF_RESAMPLES,
                  seed=SEED)

        false_alarm_text = (test_text + ': change in the false alarm rate to the lure (16 minus 0 close '
                            'neighbors; mean %.3f -> %.3f)' % (np.mean(fa_0), np.mean(fa_16)))
        tests.add(test_id='T29', ledger_ids='L6-099', location=TEXT_LOCATION,
                  test=false_alarm_text,
                  statistic_name='change in false alarm rate (16 minus 0 close neighbors)',
                  statistic=false_alarm_change, n=number_of_critical_items, unit=words['change_unit'], df='',
                  p='', p_type='none', effect_size_name='difference in proportion', effect_size=false_alarm_change,
                  ci_low=float(lows[1]), ci_high=float(highs[1]), ci_method=RELEASE_METHOD,
                  resampling_unit='critical concepts', n_resamples=NUMBER_OF_RESAMPLES, seed=SEED)

        hit_text = (test_text + ': change in the hit rate of the critical concept (16 minus 0 close neighbors; '
                    'mean %.3f -> %.3f)' % (np.mean(hit_0), np.mean(hit_16)))
        tests.add(test_id='T29', ledger_ids='L6-099', location=TEXT_LOCATION,
                  test=hit_text,
                  statistic_name='change in hit rate (16 minus 0 close neighbors)', statistic=hit_change,
                  n=number_of_critical_items, unit=words['change_unit'], df='', p='', p_type='none',
                  effect_size_name='difference in proportion', effect_size=hit_change,
                  ci_low=float(lows[2]), ci_high=float(highs[2]), ci_method=RELEASE_METHOD,
                  resampling_unit='critical concepts', n_resamples=NUMBER_OF_RESAMPLES, seed=SEED)
        print('noise %s: d′ change %.4f, Fig. S6 interval [%.3f, %.3f], 10,000 resamples [%.3f, %.3f]; '
              'false alarms %+.3f [%.3f, %.3f]; hits %+.3f [%.3f, %.3f]'
              % (noise, d_prime_change, paper_low, paper_high, lows[0], highs[0], false_alarm_change, lows[1],
                 highs[1], hit_change, lows[2], highs[2]))
    return paper_intervals, release_intervals


# ------------------------------------------------------------------------- main
def main():
    if not os.path.exists(REM_DESIGNED_FILE):
        sys.exit('stats_figS6.py: %s is missing. This module reads the saved REM run of module A2d for the white '
                 'bar of Fig. S6; copy that data/paper_run folder with it.' % REM_DESIGNED_FILE)

    tests = tests_table.TestsTable(script=SCRIPT)
    settings = read_settings()
    words = describe_run(settings)
    participants_per_list = settings['n_participants_per_list_designed'][0]
    paper_run = read_paper_run_intervals()
    white_change, white_critical_items, white_participants = read_white_bar_designed()

    print('T29: %d resamples, numpy default_rng(%d); the error bars of Fig. S6: %d resamples, '
          'default_rng(%d)' % (NUMBER_OF_RESAMPLES, SEED, PAPER_RESAMPLES, PAPER_SEED))

    add_correlation_rows(tests, words)
    paper_change_intervals, release_change_intervals = add_designed_list_rows(
        tests, words, paper_run, participants_per_list)

    # ---------------- the white bar (read from the saved run of module A2d) ----------------
    white_change_text = ('White bar of Fig. S6: REM with generated features at the Fig. 1 parameters on designed '
                         'lists (CELL 2 of list_designed.txt, the saved run of module A2d): change in d′ from 0 to '
                         '16 similar items on the study list, from the mean rates over '
                         + str(white_critical_items) + ' critical items with ' + str(white_participants)
                         + ' simulated participants per list')
    tests.add(test_id='T29', ledger_ids='L6-089',
              location=TEXT_LOCATION + '; Fig. S6 (white bar)',
              test=white_change_text, statistic_name='change in d′ (16 minus 0 similar items)',
              statistic=white_change, n=white_critical_items,
              unit='critical items (' + str(white_participants) + ' simulated participants per list)',
              df='', p='', p_type='none', effect_size_name='d′ difference', effect_size=white_change,
              ci_method=('none (the white bar has no interval; the item bootstrap of this contrast is test T28 of '
                         'module A2d)'),
              resampling_unit='critical items')

    # ---------------- the comparison with the white bar ----------------
    # the gray bars are more negative than the white bar; an interval excludes the white bar's
    # value when its upper end lies below it
    release_below = [noise for noise in NOISE_VALUES_SHOWN if release_change_intervals[noise][1] < white_change]
    paper_below = [noise for noise in NOISE_VALUES_SHOWN if paper_change_intervals[noise][1] < white_change]
    upper_ends_text = ', '.join('noise %s %.3f' % (noise, release_change_intervals[noise][1])
                                for noise in NOISE_VALUES_SHOWN)
    tests.add(test_id='T29', ledger_ids='L6-088;L6-089', location=TEXT_LOCATION,
              test=('Is every gray-bar interval of Fig. S6 below the white bar\'s d′ change of %.3f ("more than with '
                    'REM\'s generated features")? Count of the three noise levels whose 95%% interval lies entirely '
                    'below it, with the 10,000-resample intervals (with the 2,000-resample intervals drawn in '
                    'Fig. S6: %d of 3). Upper ends: %s' % (white_change, len(paper_below), upper_ends_text)),
              statistic_name='gray-bar intervals below the white bar', statistic=len(release_below),
              n=len(NOISE_VALUES_SHOWN), unit='noise levels', df='', p='', p_type='none',
              ci_method='percentile bootstrap (the 10,000-resample intervals of the rows above)',
              resampling_unit='critical concepts', n_resamples=NUMBER_OF_RESAMPLES, seed=SEED)

    tests.write(OUTPUT_FOLDER)


if __name__ == '__main__':
    main()
