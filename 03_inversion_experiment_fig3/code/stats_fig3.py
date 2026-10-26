"""
stats_fig3.py: the uncertainty statistics of Fig. 3 (the memorability inversion experiment), tests T06
and T54, and the numbers.csv rows that print the Fig. 3e interval.

WHAT IT COMPUTES (rows of outputs/tests.csv; the conventions are in statistics/README.md)
  T06  Fig. 3e: the Spearman rank correlation, over the 40 critical items, between each item's
       published memorability score (Kramer et al. 2023) and its corrected recognition in this task
       (CR = hit rate minus false alarm rate over the 50 kept participants; "old" = responses 1 to 3
       on the 6-point scale), with its two-sided p from the t approximation (38 df). rho and p are
       compared with the saved run in data/paper_run/. Two 95% percentile bootstrap intervals of rho
       follow, each from 10,000 resamples:
         - over the 40 items, the interval the paper reports: the items are resampled with
           replacement, each keeping its published score and CR
         - over participants, an additional interval: the 50 kept participants are resampled with
           replacement, every item's hit rate, false alarm rate and CR are recomputed from the
           resampled participants' trials (a participant drawn twice counts twice), then rho; the
           40 items and their published scores stay fixed
  T54  The participant flow (Methods, Experimental Procedures and Analysis): sessions, the
       exclusions by rule, kept participants.
  No correction for multiple comparisons is applied (every test is reported on its own).

INPUTS (all in this repository)
  data/inversion_trials.csv, data/inversion_design.csv, through shared/python/inversion_loader.py
      (which applies the exclusion rules of the Methods)
  outputs/inversion_items_50.csv, outputs/participant_flow.csv and outputs/numbers.csv, written by
      run_module.m (run it first). Each item's published score is read from inversion_items_50.csv:
      it is the mean, over the item's test trials, of the score of the tested image. Several items
      share a published score, and these full-precision values (those of the saved run, which
      inversion_analysis.m reads from data/paper_run/published_scores.csv) fix the order of their
      ranks in the rank correlation.
  data/paper_run/paperStats_inversion.mat (the saved run of this analysis, the output the paper
      reports)

OUTPUTS: outputs/tests.csv, outputs/logs/stats_fig3_log.txt, and the rows L2-014 and L2-015 of
  outputs/numbers.csv (rho, the item interval and p of Fig. 3e, judged against the printed values);
  run_module.m writes the other rows of that file

SEED: 20261013 (numpy.random.default_rng). One generator for the whole script, drawn in this order:
  1. the participant resamples (T06 participant interval)
  2. the item resamples (T06 item interval)
  The item interval comes from the second block of draws, so the participant resamples must be drawn
  first.

RUN (from the repository root; about 10 seconds):
  python 03_inversion_experiment_fig3/code/stats_fig3.py
  (--resamples N sets a smaller number for a quick check of the code; the default, 10,000, is the
  number used for the paper)
"""
import argparse
import csv
import os
import sys
import time

# Limit the numeric libraries to two threads; this must be set before numpy is imported.
for thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(thread_setting, '2')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.io  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CODE_FOLDER, '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (first: it stops Python from writing __pycache__ folders)
import bootstrap  # noqa: E402
import inversion_loader  # noqa: E402
import numbers_table  # noqa: E402
import tests_table  # noqa: E402

PATHS = release_paths.module_paths(__file__)
SCRIPT_NAME = 'code/stats_fig3.py'
SEED = bootstrap.RELEASE_SEED
T06_LEDGER_IDS = 'L2-009;L2-014;L2-015;L5-033'
T54_LEDGER_IDS = 'ADD-011a'
NUMBERS_LEDGER_IDS = ('L2-014', 'L2-015')    # the numbers.csv rows this script writes

LOG_LINES = []


def report(text=''):
    """Print a line and keep it for the log file."""
    print(text)
    LOG_LINES.append(text)


def spearman_rho(first_values, second_values):
    """Spearman rank correlation (tied values get their average rank); NaN if a value is missing."""
    first_values = np.asarray(first_values, dtype=float)
    second_values = np.asarray(second_values, dtype=float)
    if not (np.all(np.isfinite(first_values)) and np.all(np.isfinite(second_values))):
        return float('nan')
    return float(spearmanr(first_values, second_values).statistic)


def rates_from_counts(old_trials, hits, new_trials, false_alarms):
    """Hit rate, false alarm rate and CR of each item from summed counts (NaN where an item has no
    old or no new trial)."""
    hit_rate = np.full(len(old_trials), np.nan)
    false_alarm_rate = np.full(len(new_trials), np.nan)
    for item in range(len(old_trials)):
        if old_trials[item] > 0:
            hit_rate[item] = hits[item] / old_trials[item]
        if new_trials[item] > 0:
            false_alarm_rate[item] = false_alarms[item] / new_trials[item]
    return hit_rate, false_alarm_rate, hit_rate - false_alarm_rate


def read_matlab_items():
    """The per-item table written by inversion_analysis.m (full precision)."""
    path = os.path.join(PATHS['outputs'], 'inversion_items_50.csv')
    if not os.path.exists(path):
        raise SystemExit('outputs/inversion_items_50.csv is missing: run run_module.m first (see README.md)')
    return pd.read_csv(path, float_precision='round_trip')


def check_items_tables(matlab_items, python_items):
    """The shared Python loader's item table must equal the MATLAB table: counts and rates exactly,
    the published score to within 1e-12 (the MATLAB table holds the mean of the score over the item's
    test trials, as inversion_analysis.m computes it; the loader gives the score itself)."""
    if list(matlab_items['image_name']) != list(python_items['image_name']):
        raise SystemExit('the item order of outputs/inversion_items_50.csv differs from inversion_loader.items_table()')
    exact_columns = ['block', 'memorability_quintile', 'n_similar_items', 'n_old', 'hits', 'n_new', 'false_alarms',
                     'hit_rate', 'fa_rate', 'cr']
    for column in exact_columns:
        if not np.array_equal(matlab_items[column].to_numpy(dtype=float), python_items[column].to_numpy(dtype=float)):
            raise SystemExit('column %s of outputs/inversion_items_50.csv differs from inversion_loader.items_table()' % column)
    largest_score_difference = float(np.max(np.abs(matlab_items['kramer_cr'] - python_items['kramer_cr'])))
    if largest_score_difference > 1e-12:
        raise SystemExit('kramer_cr of outputs/inversion_items_50.csv differs from the published scores')
    report('outputs/inversion_items_50.csv (MATLAB) equals shared/python/inversion_loader.items_table(): '
           'columns %s identical; kramer_cr within %.1e of the published score' % (', '.join(exact_columns),
                                                                                  largest_score_difference))


def count_matrices(kept_trials, participants, item_names):
    """Per participant (rows) and critical item (columns): number of old trials, hits, new trials and
    false alarms, over the participant's kept test trials."""
    item_position = {name: position for position, name in enumerate(item_names)}
    shape = (len(participants), len(item_names))
    old_trials = np.zeros(shape)
    hits = np.zeros(shape)
    new_trials = np.zeros(shape)
    false_alarms = np.zeros(shape)
    critical_trials = kept_trials[kept_trials['is_critical']]
    for participant_index, participant in enumerate(participants):
        own_trials = critical_trials[critical_trials['participant'] == participant]
        for _, trial in own_trials.iterrows():
            item = item_position[trial['image_name']]
            said_old = trial['response_1to6'] <= 3
            if trial['is_new'] == 0:
                old_trials[participant_index, item] += 1
                hits[participant_index, item] += said_old
            else:
                new_trials[participant_index, item] += 1
                false_alarms[participant_index, item] += said_old
    return old_trials, hits, new_trials, false_alarms


def summed_counts(counts, drawn_participants):
    """Add up the item counts of the drawn participants (a participant drawn twice counts twice)."""
    total = np.zeros(counts.shape[1])
    for participant_index in drawn_participants:
        total = total + counts[participant_index]
    return total


def merge_into_numbers_csv(numbers):
    """Put this script's rows (NUMBERS_LEDGER_IDS) into outputs/numbers.csv, which run_module.m writes
    with the module's other rows; the rows keep the order of the ledger."""
    path = os.path.join(PATHS['outputs'], 'numbers.csv')
    if not os.path.exists(path):
        raise SystemExit('outputs/numbers.csv is missing: run run_module.m first (see README.md)')
    with open(path, newline='', encoding='utf-8') as handle:
        rows = [row for row in csv.DictReader(handle) if row['ledger_id'] not in NUMBERS_LEDGER_IDS]
    rows.extend(numbers.rows)
    ledger_order = {ledger_id: position for position, ledger_id in enumerate(numbers.ledger)}
    rows.sort(key=lambda row: ledger_order.get(row['ledger_id'], len(ledger_order)))
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=numbers_table.COLUMNS, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    matches = ', '.join('%s %s' % (row['ledger_id'], row['match']) for row in numbers.rows)
    report('wrote rows %s of outputs/numbers.csv (%d rows in all): %s' % (', '.join(NUMBERS_LEDGER_IDS), len(rows),
                                                                            matches))


def main():
    parser = argparse.ArgumentParser(description='Uncertainty statistics of Fig. 3 (tests T06 and T54)')
    parser.add_argument('--resamples', type=int, default=10000,
                        help='bootstrap resamples (default 10,000, the number used for the paper)')
    arguments = parser.parse_args()
    number_of_resamples = arguments.resamples
    started = time.time()
    report('stats_fig3.py: uncertainty statistics of Fig. 3 (memorability inversion experiment)')
    report('seed %d; %d bootstrap resamples' % (SEED, number_of_resamples))
    if number_of_resamples != 10000:
        report('NOTE: not the 10,000 resamples used for the paper; a check of the code only')

    # ---------------------------------------------------------------- 1. the data
    test_trials = inversion_loader.load_trials('test')
    kept_trials, flags = inversion_loader.apply_exclusions(test_trials)
    participants = sorted(kept_trials['participant'].unique())
    number_of_participants = len(participants)
    matlab_items = read_matlab_items()
    python_items = inversion_loader.items_table()
    check_items_tables(matlab_items, python_items)
    item_names = list(matlab_items['image_name'])
    number_of_items = len(item_names)
    published_score = matlab_items['kramer_cr'].to_numpy(dtype=float)    # full precision, from inversion_analysis.m
    corrected_recognition = matlab_items['cr'].to_numpy(dtype=float)
    report('%d critical items, %d kept participants (of %d sessions)' % (number_of_items, number_of_participants,
                                                                         len(flags)))

    # ---------------------------------------------------------------- 2. rho and p, compared with the saved run
    observed = spearmanr(published_score, corrected_recognition)
    rho = float(observed.statistic)
    p_value = float(observed.pvalue)
    paper_stats = scipy.io.loadmat(os.path.join(PATHS['paper_run'], 'paperStats_inversion.mat'),
                                   squeeze_me=True, struct_as_record=False)['paperStats']
    paper_rho = float(paper_stats.r_memVsCR)
    paper_p = float(paper_stats.p_memVsCR)
    report('Fig. 3e: Spearman rho = %.10f, p = %.10g (two-sided, t approximation with %d df); saved run: rho = %.10f, '
           'p = %.10g' % (rho, p_value, number_of_items - 2, paper_rho, paper_p))
    # Stop only when a printed value would change (rho to two decimals, p at three decimals and against
    # the printed bound 0.01); a difference in the last digits is reported as a warning.
    printed_values_differ = (round(rho, 2) != round(paper_rho, 2) or round(p_value, 3) != round(paper_p, 3)
                             or (p_value < 0.01) != (paper_p < 0.01))
    if printed_values_differ:
        raise SystemExit('rho or p differs from the saved run in data/paper_run/ at the printed precision')
    if abs(rho - paper_rho) > 1e-12 or abs(p_value - paper_p) > 1e-9 * paper_p:
        report('WARNING: rho or p differs from the saved run in data/paper_run/ in the last digits (rho %.3g, '
               'p %.3g); every printed value is the same' % (rho - paper_rho, p_value - paper_p))

    # ---------------------------------------------------------------- 3. per-participant counts
    old_trials, hits, new_trials, false_alarms = count_matrices(kept_trials, participants, item_names)
    for name, counts in (('n_old', old_trials), ('hits', hits), ('n_new', new_trials), ('false_alarms', false_alarms)):
        if not np.array_equal(counts.sum(axis=0), matlab_items[name].to_numpy(dtype=float)):
            raise SystemExit('the per-participant counts do not add up to column %s of inversion_items_50.csv' % name)
    report('per-participant counts add up to the item table (n_old, hits, n_new, false_alarms)')

    generator = np.random.default_rng(SEED)

    # ---------------------------------------------------------------- 4. participant bootstrap (draws 1)
    # T06 additional interval; drawn first, before the item resamples
    resampled_rho = np.empty(number_of_resamples)
    for resample in range(number_of_resamples):
        drawn_participants = generator.integers(0, number_of_participants, number_of_participants)
        _, _, resampled_cr = rates_from_counts(
            summed_counts(old_trials, drawn_participants), summed_counts(hits, drawn_participants),
            summed_counts(new_trials, drawn_participants), summed_counts(false_alarms, drawn_participants))
        resampled_rho[resample] = spearman_rho(published_score, resampled_cr)
    missing_resamples = int(np.sum(~np.isfinite(resampled_rho)))
    participant_low, participant_high = bootstrap.percentile_interval(resampled_rho)
    report('participant bootstrap (%d resamples; %d with an item lacking old or new trials, left out): rho 95%% CI '
           '[%.4f, %.4f]' % (number_of_resamples, missing_resamples, participant_low, participant_high))

    # ---------------------------------------------------------------- 5. item bootstrap (draws 2)
    # T06 interval the paper reports; it depends on the participant resamples being drawn first
    resampled_rho_items = np.empty(number_of_resamples)
    for resample in range(number_of_resamples):
        drawn_items = generator.integers(0, number_of_items, number_of_items)
        resampled_rho_items[resample] = spearman_rho(published_score[drawn_items], corrected_recognition[drawn_items])
    item_low, item_high = bootstrap.percentile_interval(resampled_rho_items)
    report('item bootstrap (%d resamples): rho 95%% CI [%.4f, %.4f]' % (number_of_resamples, item_low, item_high))

    # ---------------------------------------------------------------- 6. participant flow (T54)
    flow_path = os.path.join(PATHS['outputs'], 'participant_flow.csv')
    if not os.path.exists(flow_path):
        raise SystemExit('outputs/participant_flow.csv is missing: run run_module.m first (see README.md)')
    with open(flow_path, newline='', encoding='utf-8') as handle:
        flow = list(csv.DictReader(handle))[0]
    python_counts = dict(processed=len(flags), excluded_same_key=int(flags['same_key'].sum()),
                         excluded_fast=int(flags['too_fast'].sum()), excluded_slow=int(flags['too_slow'].sum()),
                         excluded_low_cr=int(flags['low_cr'].sum()), excluded_total=int(flags['excluded'].sum()),
                         kept=number_of_participants)
    rules_met = flags['same_key'].astype(int) + flags['too_fast'].astype(int) + flags['too_slow'].astype(int) + \
        flags['low_cr'].astype(int)
    python_counts['excluded_more_than_one_rule'] = int(np.sum(rules_met > 1))
    for name, value in python_counts.items():
        if int(flow[name]) != value:
            raise SystemExit('participant_flow.csv (MATLAB) and the Python loader differ in %s' % name)
    report('participant flow (outputs/participant_flow.csv, equal to the Python loader): %s sessions before '
           'exclusion, %s excluded, %s kept' % (flow['processed'], flow['excluded_total'], flow['kept']))

    # ---------------------------------------------------------------- 7. tests.csv
    # T06: the first row carries p and the participant interval, the second row the item interval (the
    # interval the paper reports). Module A3a reads these two rows (location containing "Fig. 3e";
    # resampling_unit "participants" or starting with "items") and compares them with its own resamples,
    # so keep these texts.
    tests = tests_table.TestsTable(script=SCRIPT_NAME)
    unit = 'items (%d participants)' % number_of_participants
    tests.add(test_id='T06', ledger_ids=T06_LEDGER_IDS,
              location='Fig. 3 caption (Fig. 3e); Inverting Memorability Through Direct Manipulation of Distinctiveness',
              test=('Fig. 3e: Spearman rank correlation across the %d critical items between the published memorability '
                    'score (Kramer et al. 2023) and corrected recognition in this task (hit rate minus false alarm rate '
                    'over %d participants); additional interval with the participants resampled, every item\'s hit '
                    'rate, false alarm rate and CR recomputed, the items and their published scores fixed')
                   % (number_of_items, number_of_participants),
              statistic_name='Spearman rho', statistic=rho, n=number_of_items, unit=unit, df=number_of_items - 2,
              p=p_value, p_type='parametric two-sided', effect_size_name='rho', effect_size=rho,
              ci_low=participant_low, ci_high=participant_high, ci_method='percentile bootstrap',
              resampling_unit='participants', n_resamples=number_of_resamples, seed=SEED)
    tests.add(test_id='T06', ledger_ids=T06_LEDGER_IDS,
              location='Fig. 3 caption (Fig. 3e); Inverting Memorability Through Direct Manipulation of Distinctiveness',
              test='Fig. 3e: the same Spearman rho; the interval the paper reports, with the %d items resampled (each '
                   'item keeps its published score and CR)' % number_of_items,
              statistic_name='Spearman rho', statistic=rho, n=number_of_items, unit=unit, p_type='none',
              effect_size_name='rho', effect_size=rho, ci_low=item_low, ci_high=item_high,
              ci_method='percentile bootstrap', resampling_unit='items', n_resamples=number_of_resamples,
              seed=SEED)
    flow_text = ('Participant flow, memorability inversion experiment: %s sessions before exclusion; excluded %s (one '
                 'key on more than 80 percent of test trials %s, more than 20 percent of test trials under 200 ms %s, '
                 'more than 20 percent over 7,500 ms %s, hit rate minus false alarm rate below 0.05 %s; %s meet more '
                 'than one rule); kept %s (the cap of 50 removes %s)') % (
        flow['processed'], flow['excluded_total'], flow['excluded_same_key'], flow['excluded_fast'],
        flow['excluded_slow'], flow['excluded_low_cr'], flow['excluded_more_than_one_rule'], flow['kept'],
        flow['removed_by_cap'])
    tests.add(test_id='T54', ledger_ids=T54_LEDGER_IDS, location='Methods, Experimental Procedures and Analysis',
              test=flow_text, statistic_name='participants kept', statistic=int(flow['kept']),
              n=int(flow['processed']), unit='sessions before exclusion', p_type='none', effect_size_name='',
              ci_method='none')

    # ---------------------------------------------------------------- 8. write
    tests.write(PATHS['outputs'])
    report('wrote outputs/tests.csv (%d rows)' % len(tests.rows))

    # ---------------------------------------------------------------- 9. numbers.csv rows of the Fig. 3e interval
    numbers = numbers_table.NumbersTable(script=SCRIPT_NAME)
    numbers_note = ('Spearman rho over the %d critical items (%d participants), its 95%% percentile bootstrap interval '
                    'with the items resampled (%d resamples, seed %d; T06) and its two-sided p (t approximation)'
                    % (number_of_items, number_of_participants, number_of_resamples, SEED))
    numbers.add('L2-014', computed_value=[rho, item_low, item_high, number_of_items, p_value], mode='rerun',
                note=numbers_note)
    numbers.add('L2-015', computed_value=[rho, item_low, item_high, p_value], mode='rerun', note=numbers_note)
    merge_into_numbers_csv(numbers)

    report('finished in %.1f minutes' % ((time.time() - started) / 60))
    with open(os.path.join(PATHS['logs'], 'stats_fig3_log.txt'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(LOG_LINES) + '\n')


if __name__ == '__main__':
    main()
