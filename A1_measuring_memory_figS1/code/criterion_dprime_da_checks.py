"""
criterion_dprime_da_checks.py: SI Appendix, Section 1 (Fig. S1) recomputed in Python, the values the
MATLAB script does not compute, and the module's outputs/numbers.csv.

WHAT IT DOES
PART 1, the three image-set experiments of Figs. 6 and 7. The same participants as
criterion_dprime_da.m (602 sessions; the exclusions in Methods keep 150 per set, 450 in all) and one
score per concept per image set, computed exactly as criterion_dprime_da.m computes them:
    hit rate, false alarm rate   proportion of "old" responses (1 to 3 on the six-point scale) to the
                                 concept's studied images and to its new images
    corrected recognition        hit rate minus false alarm rate (the paper's memorability score)
    d'                           z(hit rate) - z(false alarm rate), each rate kept inside
                                 [1/(2n), 1 - 1/(2n)] (n = the concept's old or new test trials)
    d_a                          sqrt(2 / (1 + s^2)) * (z(H) - s * z(F)), the z-ROC slope s fixed at
                                 0.80, H and F the cumulative "old" rates at the middle confidence
                                 boundary (responses 1 to 3)
    criterion c                  -(z(H) + z(F)) / 2 from the mean of the per-concept hit rates and the
                                 mean of the per-concept false alarm rates (c = 0.29)
It adds what the MATLAB script does not compute: the z-ROC slope the confidence ratings themselves
give (0.59) and d_a at that slope (r = 0.98 in SI Appendix, Section 1), the share of old items at
test in each experiment (Methods), and the trial counts behind each concept score.
PART 2, the REM simulation of Fig. 1 (the simulation side of the main-text statement that the
analyses hold with d' or d_a): per item hit and false alarm rates at the criterion of 1, d', and
d_a with the item's own z-ROC slope over the criteria 0.5, 0.75, 1, 1.5 and 2, from the saved
per-trial odds of the Fig. 1 run (module 01).
PART 3 compares the values with the MATLAB run (outputs/paperStats_criterionDpDa.mat, written by
criterion_dprime_da.m) and with the saved run (data/paper_run/paperStats_criterionDpDa.mat). PART 4
writes the outputs, among them outputs/numbers.csv, one row per ledger item of this module except
the four rows that print a 95% interval (L5-080, L6-012, L6-013, L6-015), which
stats_figS1.py adds. A number the MATLAB script computes is taken from the MATLAB run, and the Python
value must equal it (to 1e-9). The script writes its outputs and then stops with an error when a value differs from the
MATLAB run or from the saved run by more than 1e-9, or when a row of numbers.csv does not agree with
the paper.

INPUTS (relative to the repository root)
    06_image_set_experiments_fig6_fig7/data/maintained_trials.csv, changed_trials.csv,
        reduced_trials.csv (through shared/python/imageset_loader.py: the 602 sessions the paper reports)
    03_inversion_experiment_fig3/data/inversion_trials.csv and
        04_rare_feature_faces_fig4/data/face_trials.csv (only for the share of old items at test)
    01_rem_simulation_fig1/data/paper_run/fig1_paper_run.mat
    A1_measuring_memory_figS1/outputs/paperStats_criterionDpDa.mat (the MATLAB run)
    A1_measuring_memory_figS1/data/paper_run/paperStats_criterionDpDa.mat (the saved run)
    shared/things/concept_table.csv (the THINGS concept names, through shared/python/concept_tables.py)
    shared/ledger/paper_numbers.csv (the printed values, through shared/python/numbers_table.py)
OUTPUTS (A1_measuring_memory_figS1/outputs/)
    concept_scores.csv       the 2,453 usable concept-by-set scores (Fig. S1 and the statistics)
    criterion_table.csv      hit rate, false alarm rate, d' and c from the mean concept rates, overall
                             and per set
    rem_item_scores.csv      the 1,600 simulated items of the Fig. 1 run (Part 2)
    criterion_dprime_da.txt  every number of this script, as text
    numbers.csv              the module's ledger rows
RUN (from the repository root)
    python A1_measuring_memory_figS1/code/criterion_dprime_da_checks.py
    python A1_measuring_memory_figS1/code/criterion_dprime_da_checks.py --without-matlab
        (when criterion_dprime_da.m was not run: every number then comes from this script's own
        recomputation, and numbers.csv says so)
No random numbers. Runtime about 10 seconds; memory about 1 GB (the REM run).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (imported first: keeps the shared folder free of __pycache__)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.io as sio  # noqa: E402
from scipy import stats  # noqa: E402

import concept_tables  # noqa: E402
import face_loader  # noqa: E402
import imageset_loader  # noqa: E402
import inversion_loader  # noqa: E402
import numbers_table  # noqa: E402
import sdt  # noqa: E402

paths = release_paths.module_paths(__file__)
SCRIPT = 'code/criterion_dprime_da_checks.py'
MATLAB_SCRIPT = 'code/criterion_dprime_da.m'

IMAGE_SETS = list(imageset_loader.SET_NAMES)              # maintained, changed, reduced
FIXED_SLOPE = 0.80                                         # the fixed z-ROC slope of d_a (Methods; Fig. S1)
SCORE_COLUMNS = ['corrected_recognition', 'd_prime', 'd_a_slope080']   # the three scores of each concept
RESPONSE_LEVELS = [1, 2, 3, 4, 5, 6]                       # 1 = sure old ... 6 = sure new
OLD_RESPONSES_UP_TO = 3                                    # responses 1 to 3 count as "old"
NUMBER_OF_BOUNDARIES = 5                                   # cumulative boundaries of a six-level scale
CRITERION_CLAMP_TRIALS = 40                                # mean rates kept inside [0.5/40, 1 - 0.5/40]

REM_RUN_NAME = 'fig1_paper_run.mat'
REM_RUN_FILE = os.path.join(release_paths.find_module('01'), 'data', 'paper_run', REM_RUN_NAME)
REM_CRITERION = 1.0                                        # the criterion of the Fig. 1 simulation
REM_ROC_CRITERIA = [0.5, 0.75, 1.0, 1.5, 2.0]              # criteria of each item's z-ROC

MATLAB_RUN_FILE = os.path.join(paths['outputs'], 'paperStats_criterionDpDa.mat')
SAVED_RUN_FILE = os.path.join(paths['paper_run'], 'paperStats_criterionDpDa.mat')
EQUALITY_TOLERANCE = 1e-9

z = stats.norm.ppf
report_lines = []


def say(text=''):
    """Print a line and keep it for outputs/criterion_dprime_da.txt."""
    print(text)
    report_lines.append(text)


def correlation_line(name, x, y):
    """Pearson r (with p) and Spearman rho of two columns, printed on one line. Returns (r, p, rho)."""
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho = stats.spearmanr(x, y).statistic
    say('  %-46s Pearson r = %.4f (p = %.3g), Spearman rho = %.4f, n = %d'
        % (name, pearson_r, pearson_p, spearman_rho, len(x)))
    return float(pearson_r), float(pearson_p), float(spearman_rho)


def rate_kept_inside(rate, number_of_trials):
    """A rate kept inside [1/(2n), 1 - 1/(2n)] (n = the number of trials behind it)."""
    return min(max(rate, 1.0 / (2.0 * number_of_trials)), 1.0 - 1.0 / (2.0 * number_of_trials))


def criterion_from_mean_rates(mean_hit_rate, mean_false_alarm_rate):
    """c and d' from a mean hit rate and a mean false alarm rate, as criterion_dprime_da.m:
    each rate kept inside [0.5/40, 1 - 0.5/40] first (no effect at these rates)."""
    lowest = 0.5 / CRITERION_CLAMP_TRIALS
    z_hit = z(max(lowest, min(1 - lowest, mean_hit_rate)))
    z_false_alarm = z(max(lowest, min(1 - lowest, mean_false_alarm_rate)))
    return -0.5 * (z_hit + z_false_alarm), z_hit - z_false_alarm


def response_counts(trials):
    """Counts of the six responses (1 = sure old ... 6 = sure new) among some trials."""
    counts = []
    for level in RESPONSE_LEVELS:
        counts.append(np.sum(trials['response_1to6'] == level))
    return np.array(counts, dtype=float)


def old_items_per_unit(test_trials, is_old, unit_columns):
    """For each session (or session and block): the number of old (studied) items at test and the
    number of test items. is_old is True for a studied item at test. Returns two lists."""
    old_counts = []
    test_counts = []
    for _, unit_trials in test_trials.assign(is_old=np.asarray(is_old, dtype=bool)).groupby(unit_columns):
        old_counts.append(int(np.sum(unit_trials['is_old'])))
        test_counts.append(len(unit_trials))
    return old_counts, test_counts


def with_commas(count):
    """A whole number with thousands separators: 2453 -> '2,453'."""
    return '{:,}'.format(int(count))


def describe_old_items(old_counts, test_counts, unit_name):
    """'320 of 640 in every one of the 602 sessions', or the ranges when units differ."""
    if min(old_counts) == max(old_counts) and min(test_counts) == max(test_counts):
        return '%d of %d in every one of the %s %s' % (old_counts[0], test_counts[0], with_commas(len(old_counts)),
                                                        unit_name)
    return '%d to %d of %d to %d per %s (%d %s)' % (min(old_counts), max(old_counts), min(test_counts),
                                                     max(test_counts), unit_name.rstrip('s'), len(old_counts),
                                                     unit_name)


def describe_count(counts):
    """'40' when every value is 40, else '39 to 40'."""
    if min(counts) == max(counts):
        return '%d' % min(counts)
    return '%d to %d' % (min(counts), max(counts))


def zroc_slope(old_counts, new_counts):
    """Least-squares slope (and intercept) of z(hit) on z(false alarm) over the five cumulative
    boundaries of six-level confidence counts. Boundaries with a rate of 0 or 1 are left out."""
    with np.errstate(divide='ignore'):
        z_hits = z(np.cumsum(old_counts)[:NUMBER_OF_BOUNDARIES] / old_counts.sum())
        z_false_alarms = z(np.cumsum(new_counts)[:NUMBER_OF_BOUNDARIES] / new_counts.sum())
    finite = np.isfinite(z_hits) & np.isfinite(z_false_alarms)
    slope, intercept = np.polyfit(z_false_alarms[finite], z_hits[finite], 1)
    return float(slope), float(intercept), int(finite.sum())


def main():
    parser = argparse.ArgumentParser(description='Module A1: criterion, d\' and d_a (SI Appendix, Section 1).')
    parser.add_argument('--without-matlab', action='store_true',
                        help='criterion_dprime_da.m was not run: take every number from this script')
    arguments = parser.parse_args()
    problems = []

    # the inputs that are not read through the shared loaders, checked before any work
    if not os.path.exists(REM_RUN_FILE):
        sys.exit('criterion_dprime_da_checks.py: the Fig. 1 REM run is missing: %s (it ships in module 01)'
                 % REM_RUN_FILE)
    if not os.path.exists(MATLAB_RUN_FILE) and not arguments.without_matlab:
        sys.exit('criterion_dprime_da_checks.py: outputs/paperStats_criterionDpDa.mat is missing. Run '
                 'criterion_dprime_da.m first (run_module.py does), or pass --without-matlab to take every '
                 'number from this script.')

    # ======================================================================== PART 1: image sets
    say("SI APPENDIX, SECTION 1 (FIG. S1): CRITERION, d' AND d_a IN THE THREE IMAGE-SET EXPERIMENTS")
    say('module A1, code/criterion_dprime_da_checks.py')
    say('')

    # ---- 1.1 the 602 sessions, the exclusions in Methods, the trial filter and the cap
    all_trials = imageset_loader.load_trials(IMAGE_SETS, with_summed_similarity=False)
    kept, flags = imageset_loader.apply_exclusions(all_trials)
    number_of_sessions = all_trials['participant'].nunique()
    number_excluded = int(flags['excluded'].sum())
    kept_count_by_set = {}
    excluded_count_by_set = {}
    sessions_by_set = {}
    for set_name in IMAGE_SETS:
        kept_count_by_set[set_name] = kept.loc[kept['image_set'] == set_name, 'participant'].nunique()
        excluded_count_by_set[set_name] = int(flags.loc[flags['image_set'] == set_name, 'excluded'].sum())
        sessions_by_set[set_name] = all_trials.loc[all_trials['image_set'] == set_name, 'participant'].nunique()
    number_kept = kept['participant'].nunique()
    say('Participants: %d sessions, %d excluded, %d kept (%s); test trials after the 200 to 10,000 ms '
        'filter: %d' % (number_of_sessions, number_excluded, number_kept,
                        ', '.join('%s %d of %d' % (s, kept_count_by_set[s], sessions_by_set[s]) for s in IMAGE_SETS),
                        len(kept)))

    # ---- 1.2 share of old items at test (Methods: 50% of the test items old), in every session
    #          before exclusions: the image sets, the inversion experiment, the face study
    image_set_old_percent = 100.0 * np.mean(all_trials['is_new'] == 0)
    image_set_old, image_set_tests = old_items_per_unit(all_trials, all_trials['is_new'] == 0, ['participant'])
    block_old, block_tests = old_items_per_unit(all_trials, all_trials['is_new'] == 0, ['participant', 'block'])
    inversion_test = inversion_loader.load_trials('test')
    inversion_old_percent = 100.0 * np.mean(inversion_test['is_new'] == 0)
    inversion_old, inversion_tests = old_items_per_unit(inversion_test, inversion_test['is_new'] == 0,
                                                        ['participant'])
    face_recognition = face_loader.load_trials('recognition')
    face_old_percent = 100.0 * np.mean(face_recognition['is_old'] == 1)
    face_old, face_tests = old_items_per_unit(face_recognition, face_recognition['is_old'] == 1, ['participant'])
    image_set_old_text = describe_old_items(image_set_old, image_set_tests, 'sessions')
    image_set_block_text = describe_old_items(block_old, block_tests, 'test blocks')
    inversion_old_text = describe_old_items(inversion_old, inversion_tests, 'sessions')
    face_old_text = describe_old_items(face_old, face_tests, 'sessions')
    say('Old items at test (all sessions before exclusions):')
    say('  image sets %.1f%%: %s; %s' % (image_set_old_percent, image_set_old_text, image_set_block_text))
    say('  inversion experiment %.1f%%: %s' % (inversion_old_percent, inversion_old_text))
    say('  face study %.1f%%: %s' % (face_old_percent, face_old_text))
    # the study phase of each block comes before its test phase (SI Appendix, Section 1: separate study
    # and test phases)
    study_tables = []
    for set_name in IMAGE_SETS:
        set_rows = imageset_loader.read_set_file(set_name)
        study_tables.append(set_rows[set_rows['phase'] == 'study'])
    all_study_rows = pd.concat(study_tables, ignore_index=True)
    blocks_per_session = all_trials.groupby('participant')['block'].nunique().tolist()
    study_items_per_block = all_study_rows.groupby(['participant', 'block']).size().tolist()
    test_items_per_block = all_trials.groupby(['participant', 'block']).size().tolist()
    study_test_text = ('%s blocks per session, each with %s study images and then %s test items'
                       % (describe_count(blocks_per_session), describe_count(study_items_per_block),
                          describe_count(test_items_per_block)))
    say('  image-set design: %s' % study_test_text)
    response_levels_used = sorted(int(level) for level in all_trials['response_1to6'].unique())
    say('Response levels in the image-set test trials: %s' % response_levels_used)

    # ---- 1.3 one score per concept per image set
    concept_name_of_number = {}
    concept_table = concept_tables.load_concept_table()
    for number, name in zip(concept_table['concept_number'], concept_table['concept']):
        concept_name_of_number[int(number)] = name

    rows = []
    for set_name in IMAGE_SETS:
        set_trials = kept[kept['image_set'] == set_name]
        for concept_number, concept_trials in set_trials.groupby('concept_number'):
            old_trials = concept_trials[concept_trials['is_new'] == 0]
            new_trials = concept_trials[concept_trials['is_new'] == 1]
            if len(old_trials) == 0 or len(new_trials) == 0:
                continue                                   # no score: not tested both ways in this set
            old_counts = response_counts(old_trials)
            new_counts = response_counts(new_trials)
            number_old = len(old_trials)
            number_new = len(new_trials)
            hit_rate = float(np.mean(old_trials['response_1to6'] <= OLD_RESPONSES_UP_TO))
            false_alarm_rate = float(np.mean(new_trials['response_1to6'] <= OLD_RESPONSES_UP_TO))
            if number_old < 2 or number_new < 2:
                d_prime = np.nan
            else:
                d_prime = float(sdt.dprime_clamped(hit_rate, false_alarm_rate, number_old, number_new)[0])
            with np.errstate(divide='ignore', invalid='ignore'):
                d_a = sdt.da_from_rating_counts(old_counts, new_counts, slope=FIXED_SLOPE)
            row = dict(image_set=set_name, concept_number=int(concept_number),
                       concept=concept_name_of_number[int(concept_number)],
                       old_trials=number_old, new_trials=number_new, hit_rate=hit_rate,
                       false_alarm_rate=false_alarm_rate, corrected_recognition=hit_rate - false_alarm_rate,
                       d_prime=d_prime, d_a_slope080=d_a)
            for level in RESPONSE_LEVELS:
                row['old_resp%d' % level] = int(old_counts[level - 1])
            for level in RESPONSE_LEVELS:
                row['new_resp%d' % level] = int(new_counts[level - 1])
            rows.append(row)
    scores = pd.DataFrame(rows)
    usable_mask = np.isfinite(scores['d_prime']) & np.isfinite(scores['d_a_slope080'])
    usable = scores[usable_mask].reset_index(drop=True)
    dropped = scores[~usable_mask]
    dropped_no_dprime = int(np.sum(~np.isfinite(dropped['d_prime'])))
    dropped_da_missing = int(np.sum(np.isnan(dropped['d_a_slope080'])))
    dropped_da_infinite = int(np.sum(np.isinf(dropped['d_a_slope080'])))
    say('')
    say('Concept-by-set scores: %d computed, %d usable (finite d\' and d_a), %d dropped (d\' missing: %d; '
        'd_a missing, fewer than two finite z pairs: %d; d_a infinite, hit rate 1 or false alarm rate 0 '
        'at the middle boundary: %d)' % (len(scores), len(usable), len(dropped), dropped_no_dprime,
                                         dropped_da_missing, dropped_da_infinite))
    usable_by_set = {}
    for set_name in IMAGE_SETS:
        usable_by_set[set_name] = int(np.sum(usable['image_set'] == set_name))
    say('  usable per set: %s' % ', '.join('%s %d' % (s, usable_by_set[s]) for s in IMAGE_SETS))
    say('  trials per usable score: old median %g (%d to %d), new median %g (%d to %d)'
        % (usable['old_trials'].median(), usable['old_trials'].min(), usable['old_trials'].max(),
           usable['new_trials'].median(), usable['new_trials'].min(), usable['new_trials'].max()))

    # ---- 1.4 correlations of corrected recognition with d' and d_a
    corrected_recognition = usable['corrected_recognition'].to_numpy()
    say('')
    say('Corrected recognition against d\' and d_a, all three sets pooled:')
    r_dprime, p_dprime, rho_dprime = correlation_line("vs d'", corrected_recognition, usable['d_prime'].to_numpy())
    r_da, p_da, rho_da = correlation_line('vs d_a (fixed slope 0.80)', corrected_recognition,
                                          usable['d_a_slope080'].to_numpy())
    say('By image set:')
    r_dprime_by_set = {}
    r_da_by_set = {}
    for set_name in IMAGE_SETS:
        one_set = usable[usable['image_set'] == set_name]
        r_dprime_by_set[set_name] = correlation_line("%s: vs d'" % set_name, one_set['corrected_recognition'].to_numpy(),
                                                     one_set['d_prime'].to_numpy())[0]
        r_da_by_set[set_name] = correlation_line('%s: vs d_a (0.80)' % set_name,
                                                 one_set['corrected_recognition'].to_numpy(),
                                                 one_set['d_a_slope080'].to_numpy())[0]

    # ---- 1.5 the criterion from the mean concept rates, overall and per set (criterion_table.csv)
    say('')
    say('Criterion c = -(z(H) + z(F)) / 2 from the mean concept rates (positive = conservative):')
    criterion_rows = []
    groups = [('all three sets', IMAGE_SETS)] + [(set_name, [set_name]) for set_name in IMAGE_SETS]
    # the paper's definition: the mean of the per-concept hit and false alarm rates of the usable scores
    for group_name, group_sets in groups:
        in_group = usable['image_set'].isin(group_sets)
        mean_hit_rate = float(usable.loc[in_group, 'hit_rate'].mean())
        mean_false_alarm_rate = float(usable.loc[in_group, 'false_alarm_rate'].mean())
        criterion, d_prime = criterion_from_mean_rates(mean_hit_rate, mean_false_alarm_rate)
        criterion_rows.append(dict(definition='mean of concept rates (paper)', image_set=group_name,
                                   participants=int(kept.loc[kept['image_set'].isin(group_sets), 'participant'].nunique()),
                                   units=int(np.sum(in_group)), unit='concept-by-set scores',
                                   hit_rate=mean_hit_rate, false_alarm_rate=mean_false_alarm_rate,
                                   d_prime=d_prime, criterion_c=criterion))
    criterion_table = pd.DataFrame(criterion_rows)
    for _, row in criterion_table.iterrows():
        say('  %-30s %-15s hit %.4f, false alarm %.4f, d\' %.4f, c = %+.4f'
            % (row['definition'], row['image_set'], row['hit_rate'], row['false_alarm_rate'], row['d_prime'],
               row['criterion_c']))
    paper_definition = criterion_table[(criterion_table['definition'] == 'mean of concept rates (paper)')
                                       & (criterion_table['image_set'] == 'all three sets')].iloc[0]

    # ---- 1.6 the z-ROC slope the confidence ratings give, and d_a at that slope
    say('')
    say('z-ROC from the pooled confidence counts (all kept trials; slope of z(hit) on z(false alarm)):')
    pooled_trials = kept[kept['image_set'].isin(IMAGE_SETS)]
    old_counts = response_counts(pooled_trials[pooled_trials['is_new'] == 0])
    new_counts = response_counts(pooled_trials[pooled_trials['is_new'] == 1])
    pooled_zroc_slope, pooled_zroc_intercept, boundaries = zroc_slope(old_counts, new_counts)
    say('  %-15s slope %.3f, intercept %.3f (%d boundaries)'
        % ('all three sets', pooled_zroc_slope, pooled_zroc_intercept, boundaries))
    # d_a at the measured slope of the three sets, rounded to two decimals (0.59)
    measured_slope = round(pooled_zroc_slope, 2)
    d_a_measured = []
    for _, row in usable.iterrows():
        old_counts = np.array([row['old_resp%d' % level] for level in RESPONSE_LEVELS], dtype=float)
        new_counts = np.array([row['new_resp%d' % level] for level in RESPONSE_LEVELS], dtype=float)
        d_a_measured.append(sdt.da_from_rating_counts(old_counts, new_counts, slope=measured_slope))
    usable['d_a_measured_slope'] = d_a_measured
    r_da_measured = correlation_line('vs d_a at the measured slope %.2f' % measured_slope, corrected_recognition,
                                     usable['d_a_measured_slope'].to_numpy())[0]

    # ======================================================================== PART 2: REM
    say('')
    say('REM, the Fig. 1 run (module 01, data/paper_run/%s):' % REM_RUN_NAME)
    figure_data = sio.loadmat(REM_RUN_FILE, squeeze_me=True, struct_as_record=False)['figureData']
    hit_odds = np.asarray(figure_data.simHitOdds, dtype=float)       # items x simulations
    false_alarm_odds = np.asarray(figure_data.simFaOdds, dtype=float)
    hit_mask = np.asarray(figure_data.simHitMask).astype(bool)        # tested as a studied item
    false_alarm_mask = np.asarray(figure_data.simFaMask).astype(bool) # tested as a new item
    figure_one_r = float(figure_data.r_sim)
    del figure_data
    item_rows = []
    for item in range(hit_odds.shape[0]):
        odds_when_old = hit_odds[item, hit_mask[item]]
        odds_when_new = false_alarm_odds[item, false_alarm_mask[item]]
        number_old = len(odds_when_old)
        number_new = len(odds_when_new)
        if number_old < 2 or number_new < 2:
            continue
        hit_rate = rate_kept_inside(np.mean(odds_when_old > REM_CRITERION), number_old)
        false_alarm_rate = rate_kept_inside(np.mean(odds_when_new > REM_CRITERION), number_new)
        z_hits = []
        z_false_alarms = []
        for criterion in REM_ROC_CRITERIA:
            z_hits.append(z(rate_kept_inside(np.mean(odds_when_old > criterion), number_old)))
            z_false_alarms.append(z(rate_kept_inside(np.mean(odds_when_new > criterion), number_new)))
        item_slope = np.polyfit(z_false_alarms, z_hits, 1)[0]
        item_rows.append(dict(item=item + 1, tested_old=number_old, tested_new=number_new, hit_rate=hit_rate,
                              false_alarm_rate=false_alarm_rate, corrected_recognition=hit_rate - false_alarm_rate,
                              d_prime=z(hit_rate) - z(false_alarm_rate), zroc_slope=item_slope,
                              d_a=np.sqrt(2 / (1 + item_slope ** 2)) * (z(hit_rate) - item_slope * z(false_alarm_rate))))
    del hit_odds, false_alarm_odds, hit_mask, false_alarm_mask
    items = pd.DataFrame(item_rows)
    say('  %d items (the Fig. 1b r between memorability and similarity, 1 / mean distance, stored in the run: %.3f)'
        % (len(items), figure_one_r))
    rem_r_dprime, _, rem_rho_dprime = correlation_line("REM: vs d'", items['corrected_recognition'].to_numpy(),
                                                       items['d_prime'].to_numpy())
    rem_r_da, _, rem_rho_da = correlation_line('REM: vs d_a (per-item slope)', items['corrected_recognition'].to_numpy(),
                                               items['d_a'].to_numpy())
    rem_mean_hit = float(items['hit_rate'].mean())
    rem_mean_false_alarm = float(items['false_alarm_rate'].mean())
    rem_median_slope = float(items['zroc_slope'].median())
    say('  mean hit rate %.4f, mean false alarm rate %.4f, median per-item z-ROC slope %.3f'
        % (rem_mean_hit, rem_mean_false_alarm, rem_median_slope))

    # ======================================================================== PART 3: checks
    python_values = {
        'criterion_c': float(paper_definition['criterion_c']),
        'dprime_overall': float(paper_definition['d_prime']),
        'overallHR': float(paper_definition['hit_rate']),
        'overallFAR': float(paper_definition['false_alarm_rate']),
        'r_hitFA_dp': r_dprime,
        'r_hitFA_da': r_da,
        'r_hitFA_dp_rem': rem_r_dprime,
        'r_hitFA_da_rem': rem_r_da,
        'rho_hitFA_dp_rem': rem_rho_dprime,
        'rho_hitFA_da_rem': rem_rho_da,
        'remMeanHitRate': rem_mean_hit,
        'remMeanFaRate': rem_mean_false_alarm,
        'remMedianZrocSlope': rem_median_slope,
        'nUsable': float(len(usable)),
        'nComputed': float(len(scores)),
        'nKept': float(number_kept),
        'nExcluded': float(number_excluded),
        'nSessionsBeforeExclusion': float(number_of_sessions),
        'nKeptTrials': float(len(kept)),
        'remItemsUsed': float(len(items)),
        'daSlope': FIXED_SLOPE,
    }
    say('')
    if os.path.exists(MATLAB_RUN_FILE):
        matlab_stats = sio.loadmat(MATLAB_RUN_FILE, squeeze_me=True, struct_as_record=False)['paperStats']
        say('CHECK against the MATLAB run (outputs/paperStats_criterionDpDa.mat):')
        say('  %-24s %20s %20s %10s' % ('field', 'MATLAB', 'Python', 'difference'))
        for name, python_value in python_values.items():
            matlab_value = float(getattr(matlab_stats, name))
            difference = abs(matlab_value - python_value)
            say('  %-24s %20.12f %20.12f %10.2g' % (name, matlab_value, python_value, difference))
            if not difference <= EQUALITY_TOLERANCE:
                problems.append('%s: MATLAB %.12g, Python %.12g' % (name, matlab_value, python_value))
        for set_index, set_name in enumerate(IMAGE_SETS):
            matlab_criterion = float(np.atleast_1d(matlab_stats.criterionBySet)[set_index])
            python_criterion = float(criterion_table[(criterion_table['definition'] == 'mean of concept rates (paper)')
                                                     & (criterion_table['image_set'] == set_name)]['criterion_c'].iloc[0])
            matlab_r_da = float(np.atleast_1d(matlab_stats.r_hitFA_da_bySet)[set_index])
            say('  %-24s %20.12f %20.12f %10.2g' % ('c, ' + set_name, matlab_criterion, python_criterion,
                                                    abs(matlab_criterion - python_criterion)))
            say('  %-24s %20.12f %20.12f %10.2g' % ('r(CR, d_a), ' + set_name, matlab_r_da, r_da_by_set[set_name],
                                                    abs(matlab_r_da - r_da_by_set[set_name])))
            if abs(matlab_criterion - python_criterion) > EQUALITY_TOLERANCE:
                problems.append('criterion of the %s set differs between MATLAB and Python' % set_name)
            if abs(matlab_r_da - r_da_by_set[set_name]) > EQUALITY_TOLERANCE:
                problems.append('r(CR, d_a) of the %s set differs between MATLAB and Python' % set_name)
        use_matlab = not arguments.without_matlab
    else:
        # (only reached with --without-matlab: the check at the start stops the script otherwise)
        matlab_stats = None
        use_matlab = False
        say('MATLAB run not found and --without-matlab given: every number below is this script\'s own.')

    # the six image-set values against the saved run (data/paper_run/), to 1e-9
    saved_stats = sio.loadmat(SAVED_RUN_FILE, squeeze_me=True, struct_as_record=False)['paperStats']
    say('')
    say('CHECK against the saved run (data/paper_run/paperStats_criterionDpDa.mat):')
    say('  %-24s %20s %20s %10s' % ('field', 'saved run', 'this run (Python)', 'difference'))
    for name in ('criterion_c', 'dprime_overall', 'overallHR', 'overallFAR', 'r_hitFA_dp', 'r_hitFA_da'):
        saved_value = float(getattr(saved_stats, name))
        difference = abs(saved_value - python_values[name])
        say('  %-24s %20.12f %20.12f %10.2g' % (name, saved_value, python_values[name], difference))
        if not difference <= EQUALITY_TOLERANCE:
            problems.append('%s differs from the saved run: saved %.12g, this run %.12g' % (name, saved_value,
                                                                                           python_values[name]))

    # values for numbers.csv: those criterion_dprime_da.m computes come from the MATLAB run
    def value(name):
        if use_matlab:
            return float(getattr(matlab_stats, name))
        return python_values[name]
    matlab_script = MATLAB_SCRIPT if use_matlab else SCRIPT
    matlab_note = '' if use_matlab else 'MATLAB step not run: Python recomputation of criterion_dprime_da.m. '

    # ======================================================================== PART 4: outputs
    usable_columns = (['image_set', 'concept_number', 'concept', 'old_trials', 'new_trials', 'hit_rate',
                       'false_alarm_rate', 'corrected_recognition', 'd_prime', 'd_a_slope080', 'd_a_measured_slope']
                      + ['old_resp%d' % level for level in RESPONSE_LEVELS]
                      + ['new_resp%d' % level for level in RESPONSE_LEVELS])
    usable[usable_columns].to_csv(os.path.join(paths['outputs'], 'concept_scores.csv'), index=False)
    criterion_table.to_csv(os.path.join(paths['outputs'], 'criterion_table.csv'), index=False)
    items.to_csv(os.path.join(paths['outputs'], 'rem_item_scores.csv'), index=False)

    criterion = value('criterion_c')
    r_da_value = value('r_hitFA_da')
    r_dprime_value = value('r_hitFA_dp')
    criterion_definition = ('Criterion c = -(z(H) + z(F)) / 2, with H (%.4f) and F (%.4f) the mean hit rate and the '
                            'mean false alarm rate of the %s concept-by-set scores of the three image-set experiments '
                            '(%d participants)' % (value('overallHR'), value('overallFAR'), with_commas(len(usable)),
                                                   number_kept))
    table = numbers_table.NumbersTable(script=SCRIPT)
    table.add('L1-013', "r(CR, d_a 0.80) = %.4f and r(CR, d') = %.4f over %s concept scores (%d participants); "
                        "REM: r(CR, d') = %.4f and r(CR, d_a) = %.4f over %s items"
              % (r_da_value, r_dprime_value, with_commas(len(usable)), number_kept, value('r_hitFA_dp_rem'),
                 value('r_hitFA_da_rem'), with_commas(len(items))),
              mode='rerun', script=matlab_script,
              note=matlab_note + 'Pearson r of corrected recognition with d_a (z-ROC slope 0.80) and with d\' across '
                   'the concept-by-set scores of the three image-set experiments, and the same two correlations '
                   'across the items of the Fig. 1 REM simulation (d_a at each item\'s own z-ROC slope)')
    table.add('L4-018', 'c = %.4f; r(CR, d_a) = %.4f; r(CR, d\') = %.4f' % (criterion, r_da_value, r_dprime_value),
              mode='restatement', script=matlab_script,
              note=matlab_note + 'Criterion c from the mean concept hit and false alarm rates, and Pearson r of '
                   'corrected recognition with d_a (z-ROC slope 0.80) and with d\', over the concept-by-set scores of '
                   'the three image-set experiments')
    table.add('L5-077', image_set_old_percent, mode='rerun',
              note='Percent of test items that were old, in every session before exclusions: image sets %s (%s); '
                   'inversion experiment %.0f%%, %s; face study %.0f%%, %s'
                   % (image_set_old_text, image_set_block_text, inversion_old_percent,
                      inversion_old_text, face_old_percent, face_old_text))
    table.add('L5-078', criterion, mode='rerun', script=matlab_script, note=matlab_note + criterion_definition)
    table.add('L5-081', value('daSlope'), mode='rerun', script=matlab_script,
              note=matlab_note + 'The fixed z-ROC slope of d_a (shared/matlab/sdt_rates.m, daFromRatingCounts; '
                                 'FIXED_SLOPE in the Python scripts)')
    table.add('L6-001', number_kept, mode='rerun',
              note='Participants kept after the exclusions in Methods: %d sessions, %d excluded (%s), %s per image set'
                   % (number_of_sessions, number_excluded,
                      ', '.join('%s %d' % (s, excluded_count_by_set[s]) for s in IMAGE_SETS),
                      describe_count([kept_count_by_set[s] for s in IMAGE_SETS])))
    table.add('L6-002', criterion, mode='rerun', script=matlab_script,
              note=matlab_note + 'Criterion c from the mean hit rate and the mean false alarm rate of the '
                                 'concept-by-set scores of the three image-set experiments')
    table.add('L6-003', 'old at test: %s; %s (%.0f%%)' % (image_set_old_text, image_set_block_text,
                                                         image_set_old_percent), mode='rerun',
              note='Image-set design: %s, in all %d sessions before exclusions'
                   % (study_test_text, number_of_sessions))
    table.add('L6-004', len(SCORE_COLUMNS), mode='rerun',
              note='Scores per concept and image set (corrected recognition, d\' and d_a): %s concept-by-set scores '
                   'computed, %s with a finite d\' and d_a analyzed, %d left out (d_a missing: %d; d_a infinite: %d; '
                   'd\' missing: %d)' % (with_commas(len(scores)), with_commas(len(usable)), len(dropped),
                                          dropped_da_missing,
                                          dropped_da_infinite, dropped_no_dprime))
    table.add('L6-005', value('daSlope'), mode='rerun', script=matlab_script,
              note=matlab_note + 'The fixed z-ROC slope of d_a')
    table.add('L6-006', 'old trials per concept score: median %g (%d to %d); new: median %g (%d to %d)'
              % (usable['old_trials'].median(), usable['old_trials'].min(), usable['old_trials'].max(),
                 usable['new_trials'].median(), usable['new_trials'].min(), usable['new_trials'].max()),
              mode='rerun', note='Old and new test trials behind each of the %s concept-by-set scores (median, '
                                 'minimum and maximum)' % with_commas(len(usable)))
    table.add('L6-007', 'r(CR, d_a) = %.4f at the measured z-ROC slope %.2f and %.4f at the fixed slope 0.80'
              % (r_da_measured, measured_slope, r_da_value),
              mode='rerun', note='Pearson r of corrected recognition with d_a across the %s concept-by-set scores, at '
                                 'the z-ROC slope of the six-point confidence counts of all kept test trials (%.3f, '
                                 'rounded to %.2f) and at the fixed slope 0.80'
                                 % (with_commas(len(usable)), pooled_zroc_slope, measured_slope))
    table.add('L6-008', "Pearson r %.4f (d_a) and %.4f (d'); Spearman rho %.4f and %.4f"
              % (r_da_value, r_dprime_value, rho_da, rho_dprime), mode='rerun', script=matlab_script,
              note=matlab_note + 'Pearson r (criterion_dprime_da.m) and Spearman rho (criterion_dprime_da_checks.py) '
                                 'of corrected recognition with d_a (z-ROC slope 0.80) and with d\' across the '
                                 'concept-by-set scores')
    table.add('L6-009', 'Fig. S1: %s points, r = %.4f' % (with_commas(len(usable)), r_da_value), mode='rerun',
              script='code/make_figureS1.py',
              note='Fig. S1, drawn by code/make_figureS1.py from outputs/concept_scores.csv into '
                   'outputs/figures/FigS1_criterion_dprime_da.png: the concept-by-set scores, the least-squares line '
                   'and the Pearson r, p and n printed in the panel')
    table.add('L6-010', [kept['image_set'].nunique(), number_kept], mode='rerun',
              note='Image-set experiments, and participants kept after the exclusions')
    table.add('L6-011', value('daSlope'), mode='rerun', script=matlab_script,
              note=matlab_note + 'The fixed z-ROC slope of d_a')
    table.add('L6-014', pooled_zroc_slope, mode='rerun',
              note='Least-squares slope of z(hit rate) on z(false alarm rate) over the five cumulative boundaries of '
                   'the six-point confidence counts of all kept test trials of the three image sets')
    table.add('L6-016', len(response_levels_used), mode='rerun',
              note='Confidence levels in the image-set test trials: %s (1 = sure old to 6 = sure new; responses 1 to 3 '
                   'count as old)' % ', '.join(str(level) for level in response_levels_used))
    table.add('L6-201', [len(dropped), len(scores)], mode='rerun',
              note='Concept-by-set scores left out and computed: %d left out, %d with d_a missing (fewer than two '
                   'finite z pairs) and %d with d_a infinite (hit rate 1 or false alarm rate 0 at the middle '
                   'boundary); d\' missing: %d' % (len(dropped), dropped_da_missing, dropped_da_infinite,
                                                   dropped_no_dprime))
    table.add('L6-202', "c = %.4f; Pearson r %.4f (d_a) and %.4f (d'); Spearman rho %.4f and %.4f"
              % (criterion, r_da_value, r_dprime_value, rho_da, rho_dprime), mode='rerun', script=matlab_script,
              note=matlab_note + 'Criterion c from the mean concept hit and false alarm rates, and the Pearson r '
                                 '(criterion_dprime_da.m) and Spearman rho (criterion_dprime_da_checks.py) of '
                                 'corrected recognition with d_a (z-ROC slope 0.80) and with d\' across the '
                                 'concept-by-set scores')
    table.add('L6-203', 'c = %.4f; mean false alarm rate %.4f' % (criterion, value('overallFAR')), mode='rerun',
              script=matlab_script,
              note=matlab_note + 'Criterion c (from the mean concept hit and false alarm rates) and the mean concept '
                                 'false alarm rate, over the %s concept-by-set scores of the three image-set '
                                 'experiments' % with_commas(len(usable)))
    table.write(paths['outputs'])

    say('')
    say('numbers.csv: %d rows; %s' % (len(table.rows), ', '.join('%s %s' % (row['ledger_id'], row['match'])
                                                                  for row in table.rows)))
    with open(os.path.join(paths['outputs'], 'criterion_dprime_da.txt'), 'w') as handle:
        handle.write('\n'.join(report_lines) + '\n')
    print('wrote outputs/concept_scores.csv, criterion_table.csv, rem_item_scores.csv, criterion_dprime_da.txt')

    mismatches = [row['ledger_id'] for row in table.rows if row['match'] == 'no']
    if mismatches:
        problems.append('numbers.csv rows that do not match the paper: %s' % ', '.join(mismatches))
    if problems:
        for problem in problems:
            print('PROBLEM: ' + problem)
        sys.exit('criterion_dprime_da_checks.py: %d check(s) failed (see above)' % len(problems))
    print('All checks passed.')


if __name__ == '__main__':
    main()
