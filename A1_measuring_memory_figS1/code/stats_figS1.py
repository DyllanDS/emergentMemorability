"""
stats_figS1.py: the uncertainty statistics of SI Appendix, Section 1 (tests T21 and T22) -> outputs/tests.csv,
and the rows of outputs/numbers.csv that print a 95% interval (L5-080, L6-012, L6-013, L6-015)

WHAT IT COMPUTES
T21, the response criterion c of the three image-set experiments (Methods; SI Appendix, Section 1:
    c = 0.29). c is -(z(H) + z(F)) / 2, with H the mean of the per-concept-per-set hit rates
    and F the mean of the per-concept-per-set false alarm rates over the 2,453 analyzed concept scores
    (hit rate 0.7174, false alarm rate 0.1229), with a 95% participant-bootstrap interval. c is a bias
    measure, so it is its own effect size; it is an estimate, not a test (no p value).
T22, the agreement of corrected recognition with d_a and d' (Methods: r = 0.98; SI Appendix, Section 1:
    r = 0.98, n = 2,453 concept scores; r = 0.96 with d'; r = 0.98 at the measured slope 0.59). Pearson r
    over the 2,453 concept-by-set scores, df = n - 2, and the parametric two-sided p (it underflows to 0
    in double precision; the row gives the t value), with a 95% bootstrap interval over the
    concept-by-set scores (the unit the correlation is computed over).
    Also the measured z-ROC slope (0.59) with its participant-bootstrap interval. Concept scores with a
    non-finite d' or d_a (101 of 2,554) are left out.

HOW THE RESAMPLING WORKS (statistics/README.md: percentile intervals, 10,000 resamples, seed
20261013, two-sided, no correction for multiple tests)
    Participant bootstrap (the criterion of T21 and the z-ROC slope): one random generator,
    numpy default_rng(20261013). In every resample, for each set in the order maintained, changed,
    reduced, 150 participants are drawn with replacement (generator.integers(0, 150, 150)); a
    participant drawn twice counts twice. The 2,453 analyzed concept scores are the fixed design
    (concepts are not resampled): their hit and false alarm rates are recomputed from the drawn
    participants' response counts. The criterion needs only the rates, so every one of the 2,453
    enters it in every resample. The scores of all concepts of a set are computed at once from count
    arrays (concepts x old/new x six responses) with the formulas of criterion_dprime_da.m; the script
    checks them against outputs/concept_scores.csv before resampling.
    Concept-score bootstrap (the correlations of T22): shared/python/bootstrap.py percentile_bootstrap
    (its own generator, default_rng(20261013)): the 2,453 scores drawn with replacement, the same draws
    for the three correlations.

INPUTS (relative to the repository root)
    06_image_set_experiments_fig6_fig7/data/*_trials.csv (through shared/python/imageset_loader.py)
    A1_measuring_memory_figS1/outputs/concept_scores.csv and numbers.csv (written by
        criterion_dprime_da_checks.py; the point values here must equal the concept scores)
    shared/ledger/paper_numbers.csv (the printed values, through shared/python/numbers_table.py)
OUTPUTS
    A1_measuring_memory_figS1/outputs/tests.csv (T21, T22; columns of statistics/README.md)
    A1_measuring_memory_figS1/outputs/numbers.csv (rows L5-080, L6-012, L6-013 and L6-015 added to the
        rows of criterion_dprime_da_checks.py, in ledger order; the script stops with an error after
        writing it when a row does not agree with the paper at the default 10,000 resamples)
    A1_measuring_memory_figS1/outputs/stats_figS1.txt (the point values and the participant-bootstrap
        means and intervals)
RUN (from the repository root)
    python A1_measuring_memory_figS1/code/stats_figS1.py
    python A1_measuring_memory_figS1/code/stats_figS1.py --resamples 500   (a quick check; the
        tables use the default 10,000)
Seed 20261013. Runtime about 25 seconds for 10,000 resamples on an Apple M2 Pro.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (imported first: keeps the shared folder free of __pycache__)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.special import ndtri  # noqa: E402  (the inverse normal; scipy's norm.ppf computes it)

import bootstrap  # noqa: E402
import imageset_loader  # noqa: E402
import numbers_table  # noqa: E402
import tests_table  # noqa: E402

paths = release_paths.module_paths(__file__)
SCRIPT = 'code/stats_figS1.py'
SEED = bootstrap.RELEASE_SEED                             # 20261013
IMAGE_SETS = list(imageset_loader.SET_NAMES)              # maintained, changed, reduced
FIXED_SLOPE = 0.80                                         # the paper's fixed z-ROC slope for d_a
OLD_RESPONSES = 3                                          # responses 1 to 3 of 6 count as "old"
NUMBER_OF_LEVELS = 6
NUMBER_OF_BOUNDARIES = 5
CRITERION_CLAMP_TRIALS = 40                                # mean rates kept inside [0.5/40, 1 - 0.5/40]
POINT_TOLERANCE = 1e-12
DEFAULT_RESAMPLES = 10000                                  # the resamples behind the printed intervals

lines = []


def say(text=''):
    print(text)
    lines.append(text)


# ----------------------------------------------------------------------------- count arrays
def build_count_arrays(kept):
    """For each set: the participants (id order), the concepts tested (concept number order), and
    counts[participant, concept, role, response]: how often the participant gave each response
    (1 to 6) to the concept's studied images (role 0) and to its new images (role 1)."""
    arrays = {}
    for set_name in IMAGE_SETS:
        set_trials = kept[kept['image_set'] == set_name]
        participants = sorted(set_trials['participant'].unique())
        concepts = sorted(set_trials['concept_number'].unique())
        participant_position = {participant: position for position, participant in enumerate(participants)}
        concept_position = {concept: position for position, concept in enumerate(concepts)}
        counts = np.zeros((len(participants), len(concepts), 2, NUMBER_OF_LEVELS))
        for participant, concept, is_new, response in zip(set_trials['participant'], set_trials['concept_number'],
                                                          set_trials['is_new'], set_trials['response_1to6']):
            counts[participant_position[participant], concept_position[concept], int(is_new), int(response) - 1] += 1
        arrays[set_name] = dict(participants=participants, concepts=concepts, counts=counts,
                                counts_flat=counts.reshape(len(participants), -1),
                                participant_totals=counts.sum(axis=1))      # participants x role x response
    return arrays


# ----------------------------------------------------------------------------- the scores
def score_concepts(concept_counts, measured_slope):
    """The concept scores of one set from its counts (concepts x role x response), all concepts at once.
    Same formulas as criterion_dprime_da.m and sdt.py:
    hit rate and false alarm rate (responses 1 to 3 = old), corrected recognition, d' with each rate
    kept inside [1/(2n), 1 - 1/(2n)] (missing with fewer than 2 old or new trials), and d_a at a fixed
    slope from the cumulative rates at the middle boundary (missing unless at least two of the five
    boundaries give finite z values for both). Returns a dict of arrays (one value per concept)."""
    old_counts = concept_counts[:, 0, :]
    new_counts = concept_counts[:, 1, :]
    number_old = old_counts.sum(axis=1)
    number_new = new_counts.sum(axis=1)
    computed = (number_old > 0) & (number_new > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        hit_rate = old_counts[:, :OLD_RESPONSES].sum(axis=1) / number_old
        false_alarm_rate = new_counts[:, :OLD_RESPONSES].sum(axis=1) / number_new
        hit_inside = np.minimum(np.maximum(hit_rate, 1.0 / (2.0 * number_old)), 1.0 - 1.0 / (2.0 * number_old))
        false_alarm_inside = np.minimum(np.maximum(false_alarm_rate, 1.0 / (2.0 * number_new)),
                                        1.0 - 1.0 / (2.0 * number_new))
        d_prime = ndtri(hit_inside) - ndtri(false_alarm_inside)
        d_prime[(number_old < 2) | (number_new < 2)] = np.nan
        old_cumulative = np.cumsum(old_counts, axis=1) / number_old[:, None]
        new_cumulative = np.cumsum(new_counts, axis=1) / number_new[:, None]
        z_old = ndtri(old_cumulative[:, :NUMBER_OF_BOUNDARIES])
        z_new = ndtri(new_cumulative[:, :NUMBER_OF_BOUNDARIES])
        finite_pairs = np.sum(np.isfinite(z_old) & np.isfinite(z_new), axis=1)
        z_hit_middle = z_old[:, OLD_RESPONSES - 1]
        z_false_alarm_middle = z_new[:, OLD_RESPONSES - 1]
        d_a_fixed = np.sqrt(2.0 / (1.0 + FIXED_SLOPE ** 2)) * (z_hit_middle - FIXED_SLOPE * z_false_alarm_middle)
        d_a_measured = (np.sqrt(2.0 / (1.0 + measured_slope ** 2))
                        * (z_hit_middle - measured_slope * z_false_alarm_middle))
    d_a_fixed[finite_pairs < 2] = np.nan
    d_a_measured[finite_pairs < 2] = np.nan
    usable = computed & np.isfinite(d_prime) & np.isfinite(d_a_fixed)
    return dict(computed=computed, usable=usable, number_old=number_old, number_new=number_new,
                hit_rate=hit_rate, false_alarm_rate=false_alarm_rate,
                corrected_recognition=hit_rate - false_alarm_rate, d_prime=d_prime,
                d_a_fixed=d_a_fixed, d_a_measured=d_a_measured)


def criterion_from_mean_rates(mean_hit_rate, mean_false_alarm_rate):
    """c and d' from mean rates (each kept inside [0.5/40, 1 - 0.5/40] first), as in criterion_dprime_da.m."""
    lowest = 0.5 / CRITERION_CLAMP_TRIALS
    z_hit = ndtri(min(max(mean_hit_rate, lowest), 1 - lowest))
    z_false_alarm = ndtri(min(max(mean_false_alarm_rate, lowest), 1 - lowest))
    return -0.5 * (z_hit + z_false_alarm), z_hit - z_false_alarm


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def zroc_slope(old_counts, new_counts):
    """Slope of z(hit) on z(false alarm) over the five cumulative boundaries (finite ones only)."""
    with np.errstate(divide='ignore'):
        z_hits = ndtri(np.cumsum(old_counts)[:NUMBER_OF_BOUNDARIES] / old_counts.sum())
        z_false_alarms = ndtri(np.cumsum(new_counts)[:NUMBER_OF_BOUNDARIES] / new_counts.sum())
    finite = np.isfinite(z_hits) & np.isfinite(z_false_alarms)
    return float(np.polyfit(z_false_alarms[finite], z_hits[finite], 1)[0])


# The statistics the participant bootstrap keeps from every resample (the criterion of T21 and the
# z-ROC slope of T22).
STATISTIC_NAMES = ['c_paper_all', 'zroc_slope_all', 'scores_in_criterion']


def all_statistics(arrays, weights_by_set, measured_slope, analysed_by_set, return_scores=False):
    """The criterion of T21, the correlations of T22 and the z-ROC slope, from participant
    weights (how often each participant is counted: 1 for everyone gives the point values; a bootstrap
    resample gives 0, 1, 2, ...).

    analysed_by_set: for each set, which of its concepts are analyzed (the 2,453 concept scores with a
    finite d' and d_a when everyone is counted once). They are the fixed design: in a resample their
    scores are recomputed from the drawn participants. A concept enters the criterion when it has a hit
    and a false alarm rate (c needs nothing else) and enters the correlations when its d' and d_a are
    finite (with everyone counted once, all 2,453)."""
    values = {}
    pooled_scores = {name: [] for name in ('corrected_recognition', 'd_prime', 'd_a_fixed', 'd_a_measured')}
    pooled_rates = {name: [] for name in ('hit_rate', 'false_alarm_rate')}
    scores_by_set = {}
    total_old_counts = np.zeros(NUMBER_OF_LEVELS)
    total_new_counts = np.zeros(NUMBER_OF_LEVELS)
    for set_name in IMAGE_SETS:
        one_set = arrays[set_name]
        weights = weights_by_set[set_name]
        concept_counts = (weights @ one_set['counts_flat']).reshape(len(one_set['concepts']), 2, NUMBER_OF_LEVELS)
        scores = score_concepts(concept_counts, measured_slope)
        scores_by_set[set_name] = scores
        analysed = analysed_by_set[set_name]
        in_criterion = analysed & scores['computed']
        in_correlations = analysed & scores['usable']
        for name in pooled_scores:
            pooled_scores[name].append(scores[name][in_correlations])
        for name in pooled_rates:
            pooled_rates[name].append(scores[name][in_criterion])
        # the response counts of all trials of the set, summed over the drawn participants (z-ROC slope)
        set_totals = weights @ one_set['participant_totals'].reshape(len(weights), -1)
        set_totals = set_totals.reshape(2, NUMBER_OF_LEVELS)
        total_old_counts = total_old_counts + set_totals[0]
        total_new_counts = total_new_counts + set_totals[1]
    for name in pooled_scores:
        pooled_scores[name] = np.concatenate(pooled_scores[name])
    for name in pooled_rates:
        pooled_rates[name] = np.concatenate(pooled_rates[name])
    values['c_paper_all'] = criterion_from_mean_rates(np.mean(pooled_rates['hit_rate']),
                                                      np.mean(pooled_rates['false_alarm_rate']))[0]
    values['r_cr_da080'] = pearson(pooled_scores['corrected_recognition'], pooled_scores['d_a_fixed'])
    values['r_cr_dprime'] = pearson(pooled_scores['corrected_recognition'], pooled_scores['d_prime'])
    values['r_cr_da_measured'] = pearson(pooled_scores['corrected_recognition'], pooled_scores['d_a_measured'])
    values['zroc_slope_all'] = zroc_slope(total_old_counts, total_new_counts)
    values['scores_in_correlations'] = len(pooled_scores['corrected_recognition'])
    values['scores_in_criterion'] = len(pooled_rates['hit_rate'])
    if return_scores:
        return values, scores_by_set
    return values


def add_rows_to_numbers_csv(numbers):
    """Rewrite outputs/numbers.csv with the rows of criterion_dprime_da_checks.py and the rows of this
    script, in ledger order. Rows of this script from an earlier run are replaced."""
    numbers_file = os.path.join(paths['outputs'], 'numbers.csv')
    own_ids = set(row['ledger_id'] for row in numbers.rows)
    with open(numbers_file, newline='', encoding='utf-8') as handle:
        other_rows = [row for row in csv.DictReader(handle) if row['ledger_id'] not in own_ids]
    ledger_order = list(numbers.ledger)

    def ledger_position(row):
        if row['ledger_id'] in numbers.ledger:
            return ledger_order.index(row['ledger_id'])
        return len(ledger_order)

    numbers.rows = sorted(other_rows + numbers.rows, key=ledger_position)
    numbers.write(paths['outputs'])


def main():
    parser = argparse.ArgumentParser(description='Module A1: uncertainty statistics T21 and T22.')
    parser.add_argument('--resamples', type=int, default=DEFAULT_RESAMPLES,
                        help='bootstrap resamples (default 10,000)')
    arguments = parser.parse_args()
    number_of_resamples = arguments.resamples

    for needed in ('concept_scores.csv', 'numbers.csv'):
        if not os.path.exists(os.path.join(paths['outputs'], needed)):
            sys.exit('stats_figS1.py: outputs/%s is missing; run criterion_dprime_da_checks.py first' % needed)
    say('stats_figS1.py: uncertainty statistics T21 and T22 (SI Appendix, Section 1, Fig. S1)')
    say('seed %d, %d resamples per bootstrap' % (SEED, number_of_resamples))

    # ---- data: the 450 kept participants and their kept test trials
    all_trials = imageset_loader.load_trials(IMAGE_SETS, with_summed_similarity=False)
    kept, flags = imageset_loader.apply_exclusions(all_trials)
    arrays = build_count_arrays(kept)
    number_of_participants = {set_name: len(arrays[set_name]['participants']) for set_name in IMAGE_SETS}
    total_participants = sum(number_of_participants.values())
    say('participants kept: %s (%d); test trials kept: %d'
        % (', '.join('%s %d' % (s, number_of_participants[s]) for s in IMAGE_SETS), total_participants, len(kept)))

    # ---- the measured z-ROC slope of the data (rounded to 2 decimals, as in the checks script)
    total_old_counts = np.zeros(NUMBER_OF_LEVELS)
    total_new_counts = np.zeros(NUMBER_OF_LEVELS)
    for set_name in IMAGE_SETS:
        totals = arrays[set_name]['participant_totals'].sum(axis=0)
        total_old_counts = total_old_counts + totals[0]
        total_new_counts = total_new_counts + totals[1]
    measured_slope = round(zroc_slope(total_old_counts, total_new_counts), 2)

    # ---- the point values (every participant counted once), checked against concept_scores.csv
    everyone = {set_name: np.ones(number_of_participants[set_name]) for set_name in IMAGE_SETS}
    # the analyzed concepts: the concept scores with a finite d' and d_a
    analysed_by_set = {}
    for set_name in IMAGE_SETS:
        one_set = arrays[set_name]
        concept_counts = (everyone[set_name] @ one_set['counts_flat']).reshape(len(one_set['concepts']), 2,
                                                                                NUMBER_OF_LEVELS)
        analysed_by_set[set_name] = score_concepts(concept_counts, measured_slope)['usable']
    point, scores_by_set = all_statistics(arrays, everyone, measured_slope, analysed_by_set, return_scores=True)
    saved_scores = pd.read_csv(os.path.join(paths['outputs'], 'concept_scores.csv'), float_precision='round_trip')
    largest_difference = 0.0
    for set_name in IMAGE_SETS:
        scores = scores_by_set[set_name]
        usable_concepts = np.array(arrays[set_name]['concepts'])[scores['usable']]
        saved = saved_scores[saved_scores['image_set'] == set_name]
        if not np.array_equal(usable_concepts, saved['concept_number'].to_numpy()):
            sys.exit('stats_figS1.py: the usable concepts of the %s set differ from outputs/concept_scores.csv'
                     % set_name)
        for name, column in (('hit_rate', 'hit_rate'), ('false_alarm_rate', 'false_alarm_rate'),
                             ('corrected_recognition', 'corrected_recognition'), ('d_prime', 'd_prime'),
                             ('d_a_fixed', 'd_a_slope080'), ('d_a_measured', 'd_a_measured_slope')):
            difference = np.max(np.abs(scores[name][scores['usable']] - saved[column].to_numpy()))
            largest_difference = max(largest_difference, float(difference))
    if not largest_difference <= POINT_TOLERANCE:
        sys.exit('stats_figS1.py: the recomputed concept scores differ from outputs/concept_scores.csv by %.3g'
                 % largest_difference)
    number_analysed = point['scores_in_correlations']
    if point['scores_in_criterion'] != number_analysed:
        sys.exit('stats_figS1.py: the criterion and the correlations use different concept scores')
    say('point values: %d usable concept scores, identical to outputs/concept_scores.csv (largest difference '
        '%.2g); measured z-ROC slope %.4f (rounded %.2f)' % (number_analysed, largest_difference,
                                                           point['zroc_slope_all'], measured_slope))

    # ---- participant bootstrap: participants resampled within each set; the analyzed concepts are the
    #      fixed design and their scores are recomputed from the drawn participants
    generator = np.random.default_rng(SEED)
    participant_bootstrap = np.empty((number_of_resamples, len(STATISTIC_NAMES)))
    for resample in range(number_of_resamples):
        weights_by_set = {}
        for set_name in IMAGE_SETS:
            drawn = generator.integers(0, number_of_participants[set_name], number_of_participants[set_name])
            weights_by_set[set_name] = np.bincount(drawn, minlength=number_of_participants[set_name]).astype(float)
        values = all_statistics(arrays, weights_by_set, measured_slope, analysed_by_set)
        for column, name in enumerate(STATISTIC_NAMES):
            participant_bootstrap[resample, column] = values[name]
    participant_low, participant_high = bootstrap.percentile_interval(participant_bootstrap)
    participant_interval = {}
    participant_mean = {}
    for column, name in enumerate(STATISTIC_NAMES):
        participant_interval[name] = (float(participant_low[column]), float(participant_high[column]))
        participant_mean[name] = float(np.mean(participant_bootstrap[:, column]))
    scores_lost_criterion = number_analysed - participant_bootstrap[:, STATISTIC_NAMES.index('scores_in_criterion')]
    say('participant bootstrap (%d resamples): of the %d analyzed concept scores, a resample leaves out of the '
        'criterion %d to %d (no hit or no false alarm rate)'
        % (number_of_resamples, number_analysed, scores_lost_criterion.min(), scores_lost_criterion.max()))
    say('  %-24s %12s %16s %12s %12s' % ('statistic', 'estimate', 'bootstrap mean', '2.5%', '97.5%'))
    for name in STATISTIC_NAMES:
        say('  %-24s %12.6f %16.6f %12.6f %12.6f' % (name, point[name], participant_mean[name],
                                                    participant_interval[name][0], participant_interval[name][1]))

    # ---- concept-score bootstrap of the correlations (the same draws for all three)
    pooled = saved_scores
    corrected_recognition = pooled['corrected_recognition'].to_numpy()
    d_a_fixed = pooled['d_a_slope080'].to_numpy()
    d_prime = pooled['d_prime'].to_numpy()
    d_a_measured = pooled['d_a_measured_slope'].to_numpy()
    number_of_scores = len(pooled)

    def three_correlations(indices):
        return [pearson(corrected_recognition[indices], d_a_fixed[indices]),
                pearson(corrected_recognition[indices], d_prime[indices]),
                pearson(corrected_recognition[indices], d_a_measured[indices])]
    score_resamples = bootstrap.percentile_bootstrap(three_correlations, number_of_scores, number_of_resamples, SEED)
    score_low, score_high = bootstrap.percentile_interval(score_resamples)

    # ---- the tests table
    tests = tests_table.TestsTable(script=SCRIPT)
    participants_text = '%d participants, 150 per set' % total_participants
    common_participant = dict(ci_method='percentile bootstrap', resampling_unit='participants (within each set)',
                              n_resamples=number_of_resamples, seed=SEED)

    # T21: the criterion
    mean_hit = float(pooled['hit_rate'].mean())
    mean_false_alarm = float(pooled['false_alarm_rate'].mean())
    dprime_value = criterion_from_mean_rates(mean_hit, mean_false_alarm)[1]
    low, high = participant_interval['c_paper_all']
    tests.add(test_id='T21', ledger_ids='L5-078;L6-002', location='SI Appendix, Section 1',
              test='Response criterion c of the three image-set experiments: '
                   'c = -(z(H) + z(F)) / 2 from the mean per-concept-per-set hit rate H (%.4f) and false alarm '
                   'rate F (%.4f) of the %s analyzed concept-by-set scores (d\' %.3f); interval: participants '
                   'resampled within each set, the analyzed concept scores kept as the fixed design and their hit '
                   'and false alarm rates recomputed from the drawn participants (bootstrap mean %.4f)'
                   % (mean_hit, mean_false_alarm, '{:,}'.format(len(pooled)), dprime_value,
                      participant_mean['c_paper_all']),
              statistic_name='c', statistic=point['c_paper_all'], n=total_participants,
              unit='participants (%s concept-by-set scores)' % '{:,}'.format(len(pooled)), df='', p='',
              p_type='none',
              effect_size_name='c', effect_size=point['c_paper_all'], ci_low=low, ci_high=high, **common_participant)

    # T22: the agreement of corrected recognition with d_a and d'
    number_dropped = 0
    for set_name in IMAGE_SETS:
        number_dropped += int(np.sum(scores_by_set[set_name]['computed'] & ~scores_by_set[set_name]['usable']))
    number_computed = number_analysed + number_dropped
    unit_text = ('concept-by-set scores (%s; %s computed, %d left out for a non-finite d\' or d_a)'
                 % (participants_text, '{:,}'.format(number_computed), number_dropped))
    correlations = [
        ('r_cr_da080', 0, 'L5-080;L6-012;L1-013;L6-008', 'SI Appendix, Section 1',
         'Pearson r between corrected recognition and d_a (z-ROC slope fixed at 0.80) across the concept-by-set '
         'scores of the three image-set experiments'),
        ('r_cr_dprime', 1, 'L6-013;L6-008', 'SI Appendix, Section 1',
         'Pearson r between corrected recognition and d\' (rates kept inside [1/(2n), 1 - 1/(2n)]) across the '
         'concept-by-set scores'),
        ('r_cr_da_measured', 2, 'L6-015;L6-014', 'SI Appendix, Section 1',
         'Pearson r between corrected recognition and d_a at the z-ROC slope the confidence ratings give '
         '(%.2f, held fixed in the resamples) across the concept-by-set scores' % measured_slope),
    ]
    correlation_results = {}
    for statistic_name_key, column, ledger_ids, location, description in correlations:
        r_value = point[statistic_name_key]
        degrees_of_freedom = number_analysed - 2
        t_value = r_value * np.sqrt(degrees_of_freedom / (1.0 - r_value ** 2))
        p_value = float(2.0 * stats.t.sf(abs(t_value), degrees_of_freedom))
        p_text = ('parametric p underflows to 0 in double precision (t(%d) = %.1f)'
                  % (degrees_of_freedom, t_value)) if p_value == 0.0 else 'parametric p'
        correlation_results[statistic_name_key] = dict(r=r_value, p=p_value, low=float(score_low[column]),
                                                       high=float(score_high[column]))
        tests.add(test_id='T22', ledger_ids=ledger_ids, location=location, statistic_name='Pearson r',
                  test='%s; %s; interval: the concept-by-set scores resampled' % (description, p_text),
                  statistic=r_value, n=number_analysed, unit=unit_text, df=degrees_of_freedom, p=p_value,
                  p_type='parametric two-sided', effect_size_name='r', effect_size=r_value,
                  ci_low=float(score_low[column]), ci_high=float(score_high[column]), ci_method='percentile bootstrap',
                  resampling_unit='concept-by-set scores', n_resamples=number_of_resamples, seed=SEED)
    low, high = participant_interval['zroc_slope_all']
    tests.add(test_id='T22', ledger_ids='L6-014;L6-007', location='SI Appendix, Section 1',
              test='z-ROC slope the confidence ratings give: least-squares slope of z(hit rate) on z(false alarm '
                   'rate) over the five cumulative boundaries of the six-point confidence counts of all kept test '
                   'trials of the three sets; participants resampled within each set',
              statistic_name='z-ROC slope', statistic=point['zroc_slope_all'], n=total_participants,
              unit='participants (%s test trials)' % '{:,}'.format(len(kept)), df='', p='', p_type='none',
              effect_size_name='z-ROC slope', effect_size=point['zroc_slope_all'], ci_low=low, ci_high=high,
              **common_participant)

    # ---- the rows of numbers.csv that print a 95% interval: r, the concept-score bootstrap interval, p
    numbers = numbers_table.NumbersTable(script=SCRIPT)
    interval_note = ('Pearson r between corrected recognition and %s across the %s concept-by-set scores, its 95%% '
                     'percentile bootstrap interval (the scores resampled, %d resamples, seed %d) and its two-sided '
                     'parametric p (computed as 0, below the smallest double-precision number)')
    da_fixed = correlation_results['r_cr_da080']
    numbers.add('L5-080', [da_fixed['r'], da_fixed['low'], da_fixed['high'], da_fixed['p']], mode='rerun',
                note=interval_note % ('d_a (z-ROC slope 0.80)', '{:,}'.format(number_analysed), number_of_resamples,
                                      SEED))
    numbers.add('L6-012', [da_fixed['r'], da_fixed['low'], da_fixed['high'], da_fixed['p'], number_analysed],
                mode='rerun', note=interval_note % ('d_a (z-ROC slope 0.80)', '{:,}'.format(number_analysed),
                                                    number_of_resamples, SEED) + ', and the number of scores')
    dprime_result = correlation_results['r_cr_dprime']
    numbers.add('L6-013', [dprime_result['r'], dprime_result['low'], dprime_result['high'], dprime_result['p']],
                mode='rerun', note=interval_note % ("d'", '{:,}'.format(number_analysed), number_of_resamples, SEED))
    da_measured = correlation_results['r_cr_da_measured']
    numbers.add('L6-015', [da_measured['r'], da_measured['low'], da_measured['high']], mode='rerun',
                note='Pearson r between corrected recognition and d_a at the measured z-ROC slope rounded to %.2f, '
                     'across the %s concept-by-set scores, and its 95%% percentile bootstrap interval (the scores '
                     'resampled, %d resamples, seed %d)'
                     % (measured_slope, '{:,}'.format(number_analysed), number_of_resamples, SEED))
    add_rows_to_numbers_csv(numbers)

    tests.write(paths['outputs'])
    with open(os.path.join(paths['outputs'], 'stats_figS1.txt'), 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    print('wrote outputs/tests.csv and outputs/stats_figS1.txt')
    mismatches = [row['ledger_id'] for row in numbers.rows if row['match'] == 'no']
    if mismatches and number_of_resamples == DEFAULT_RESAMPLES:
        sys.exit('stats_figS1.py: numbers.csv rows that do not match the paper: %s' % ', '.join(mismatches))
    if mismatches:
        print('numbers.csv rows that do not match the paper with %d resamples: %s (the paper uses %d)'
              % (number_of_resamples, ', '.join(mismatches), DEFAULT_RESAMPLES))


if __name__ == '__main__':
    main()
