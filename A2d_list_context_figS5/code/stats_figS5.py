"""
stats_figS5.py: the uncertainty statistics of SI Appendix, Fig. S5 (test T28 of statistics/README.md):
the change in hit rate, false alarm rate, hit rate minus false alarm rate and d′ when the same item is
studied on a crowded rather than an uncrowded list, in REM at the parameter values of Fig. 1, with 95%
bootstrap confidence intervals.

WHAT IT DOES
  1. The intervals of Fig. S5: reads outputs/list_bootstrap.json (written by list_bootstrap.py:
     2,000 resamples, seed 20261010, the error bars of Fig. S5) and writes their rows.
  2. The same percentile bootstrap with 10,000 resamples, seed 20261013 (one generator: the designed
     lists first, then the random lists), the intervals printed in SI Appendix, Section 2, List
     Context. The point estimates are those of step 1.
  Resampled units: the critical item for the designed lists (the five lists of an item stay
  together) and the list for the random lists (all test items of a list stay together). These are
  simulations, so no p values are given; the effect size is the difference itself (in rate or in
  d′). No correction for multiple comparisons is applied.

READS   outputs/list_bootstrap.json (list_bootstrap.py)
        outputs/list_designed_cell2.mat, outputs/list_binned_cell2.mat (list_designed.m, list_binned.m)
WRITES  outputs/tests.csv (16 rows: 2 list types x 4 measures x 2 bootstraps)

Run (from the repository root):
    python A2d_list_context_figS5/code/stats_figS5.py
Seed: numpy.random.default_rng(20261013), 10,000 resamples per list type. Runtime: about 10 seconds.
"""
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import release_paths  # noqa: E402
import tests_table  # noqa: E402
import list_bootstrap  # noqa: E402  (this module's bootstrap functions)

RELEASE_SEED = 20261013                 # the bootstrap of step 2
RELEASE_NUMBER_OF_BOOTSTRAPS = 10000

paths = release_paths.module_paths(__file__)
bootstrap_file = os.path.join(paths['outputs'], 'list_bootstrap.json')
designed_file = os.path.join(paths['outputs'], 'list_designed_cell2.mat')
random_file = os.path.join(paths['outputs'], 'list_binned_cell2.mat')
for needed, how in ((bootstrap_file, 'run code/list_bootstrap.py'),
                    (designed_file, 'run code/list_designed.m'),
                    (random_file, 'run code/list_binned.m')):
    if not os.path.exists(needed):
        sys.exit('stats_figS5.py: %s is missing; %s first' % (os.path.relpath(needed, paths['module']), how))

# =============================================================================== the simulated items
designed = list_bootstrap.load_designed_items(designed_file)
random_lists = list_bootstrap.load_random_list_items(random_file)
number_of_critical_items = designed['hit_rate_by_item'].shape[0]
number_of_lists = random_lists['number_of_lists']
designed_parameters = designed['parameters']
random_parameters = random_lists['parameters']

# test items in the two bins the random-list contrast compares (for the unit column)
studied_with_none = int(np.sum(random_lists['studied_count'] == 0))
studied_with_three_or_more = int(np.sum(random_lists['studied_count'] >= 3))
lures_with_none = int(np.sum(random_lists['lure_count'] == 0))
lures_with_three_or_more = int(np.sum(random_lists['lure_count'] >= 3))


def with_commas(count):
    """6825 -> '6,825'."""
    return '{:,}'.format(count)


# =============================================================================== 1. the intervals of Fig. S5
with open(bootstrap_file, encoding='utf-8') as handle:
    paper_run = json.load(handle)
assert paper_run['seed'] == list_bootstrap.PAPER_RANDOM_SEED
assert paper_run['bootstraps'] == list_bootstrap.PAPER_NUMBER_OF_BOOTSTRAPS

# =============================================================================== 2. 10,000 resamples
print('bootstrap with %d resamples per list type, seed %d' % (RELEASE_NUMBER_OF_BOOTSTRAPS, RELEASE_SEED))
random_generator = np.random.default_rng(RELEASE_SEED)
designed_point = list_bootstrap.designed_list_changes(designed, np.arange(number_of_critical_items))
designed_draws = list_bootstrap.bootstrap_designed_lists(designed, RELEASE_NUMBER_OF_BOOTSTRAPS, random_generator)
random_point = list_bootstrap.random_list_changes(random_lists, np.arange(1, number_of_lists + 1))
random_draws = list_bootstrap.bootstrap_random_lists(random_lists, RELEASE_NUMBER_OF_BOOTSTRAPS, random_generator)
release_standard = {'designed': {}, 'random': {}}
for list_type, point, draws in (('designed', designed_point, designed_draws), ('random', random_point, random_draws)):
    for key in list_bootstrap.CHANGE_NAMES:
        low, high = list_bootstrap.percentile_interval(draws[key])
        release_standard[list_type][key] = {'change': float(point[key]), 'low': low, 'high': high}
        # the point estimate is the same computation as in list_bootstrap.py
        assert abs(point[key] - paper_run[list_type][key]['change']) < 1e-15, (list_type, key)

# =============================================================================== rows of tests.csv
COMMON_SETTINGS = ('REM at the parameter values of Fig. 1 (g %.2f, %d features, storage %.2f, copy fidelity %.2f, '
                   'criterion %.1f), the same item on a crowded and an uncrowded %d-item study list'
                   % (designed_parameters['g'], designed_parameters['w'], designed_parameters['u'],
                      designed_parameters['c'], designed_parameters['crit'], designed_parameters['nStudy']))
LIST_TYPES = {
    'designed': dict(
        description=('designed lists: the critical item studied with %d vs 0 of its next most similar items '
                     '(neighbor ranks 2 to %d) on its list, the rest of the list filled from the far half of its '
                     'neighbor ranking; lure = its most similar item, never studied'
                     % (designed['similar_item_levels'][-1], designed['similar_item_levels'][-1] + 1)),
        n=number_of_critical_items,
        unit=('critical items (each on %d lists, %d simulated participants per list)'
              % (len(designed['similar_item_levels']), designed['participants_per_list'])),
        resampling_unit='critical items (the %d lists of an item kept together)' % len(designed['similar_item_levels']),
        dprime_rule=('d′ = z(mean hit rate) minus z(mean false alarm rate) over the %d critical items, rates kept '
                     'within 1/%d and 1 - 1/%d' % (number_of_critical_items, 2 * designed['participants_per_list'],
                                                   2 * designed['participants_per_list']))),
    'random': dict(
        description=('random lists (%d studied items and %d lures drawn at random per list): test items with 3 or '
                     'more vs 0 close neighbors on their list (close = among the item\'s %d most similar items '
                     'in the %s-item database)'
                     % (random_parameters['nStudy'], random_parameters['nLure'], random_parameters['nNeighbour'],
                        with_commas(int(random_parameters['nDatabase'])))),
        n=number_of_lists,
        unit=('lists (%d simulated participants per list); %s studied items and %s lures with 0 close '
              'neighbors, %s and %s with 3 or more'
              % (random_parameters['nSubjPerList'], with_commas(studied_with_none), with_commas(lures_with_none),
                 with_commas(studied_with_three_or_more), with_commas(lures_with_three_or_more))),
        resampling_unit='lists (all test items of a list kept together)',
        dprime_rule=('d′ = z(mean hit rate) minus z(mean false alarm rate) over the test items of a bin, rates '
                     'kept within 0.0001 and 0.9999')),
}
MEASURES = {
    'hit': dict(statistic_name='change in hit rate (crowded minus uncrowded)',
                effect_size_name='hit rate difference'),
    'false_alarm': dict(statistic_name='change in false alarm rate (crowded minus uncrowded)',
                        effect_size_name='false alarm rate difference'),
    'hits_minus_false_alarms': dict(statistic_name='change in hit rate minus false alarm rate (crowded minus uncrowded)',
                                    effect_size_name='difference in hit rate minus false alarm rate'),
    'dprime': dict(statistic_name='change in d′ (crowded minus uncrowded)', effect_size_name='d′ difference'),
}
# The ledger items (shared/ledger/paper_numbers.csv) and the place in the paper of each row. The intervals
# from 2,000 resamples are the error bars of Fig. S5; those from 10,000 resamples are the intervals of the
# text, which prints the two falls in d′ and the designed-list changes in hits and false alarms (L6-072,
# L6-080) and describes the other changes (L6-073).
FIGURE_LEDGER_IDS = {'hit': 'L6-074;L6-082', 'false_alarm': 'L6-074;L6-082',
                     'hits_minus_false_alarms': 'L6-074;L6-082', 'dprime': 'L6-075;L6-082'}
FIGURE_LOCATIONS = {'hit': 'SI Appendix, Fig. S5a', 'false_alarm': 'SI Appendix, Fig. S5a',
                    'hits_minus_false_alarms': 'SI Appendix, Fig. S5a', 'dprime': 'SI Appendix, Fig. S5b'}
TEXT_LEDGER_IDS = {('designed', 'hit'): 'L6-080;L6-073', ('designed', 'false_alarm'): 'L6-080;L6-073',
                   ('designed', 'dprime'): 'L6-072', ('random', 'dprime'): 'L6-072'}   # the others: L6-073
TEXT_LOCATION = 'SI Appendix, Section 2, List Context'
RUNS = [
    ('paper', paper_run, list_bootstrap.PAPER_NUMBER_OF_BOOTSTRAPS, list_bootstrap.PAPER_RANDOM_SEED,
     'code/list_bootstrap.py', 'percentile bootstrap (the error bars of Fig. S5)'),
    ('release', release_standard, RELEASE_NUMBER_OF_BOOTSTRAPS, RELEASE_SEED, 'code/stats_figS5.py',
     'percentile bootstrap'),
]

tests = tests_table.TestsTable(script='code/stats_figS5.py')
for run_name, run_results, number_of_draws, seed, script, ci_method in RUNS:
    for list_type in ('random', 'designed'):
        for measure in list_bootstrap.CHANGE_NAMES:
            entry = run_results[list_type][measure]
            test_text = '%s; %s; %s' % (COMMON_SETTINGS, LIST_TYPES[list_type]['description'],
                                        MEASURES[measure]['statistic_name'])
            if measure == 'dprime':
                test_text += ' (%s)' % LIST_TYPES[list_type]['dprime_rule']
            if run_name == 'paper':                 # 2,000 resamples: the error bars of Fig. S5
                ledger_ids, location = FIGURE_LEDGER_IDS[measure], FIGURE_LOCATIONS[measure]
            else:                                   # 10,000 resamples: the intervals of the text
                ledger_ids, location = TEXT_LEDGER_IDS.get((list_type, measure), 'L6-073'), TEXT_LOCATION
            tests.add(test_id='T28', ledger_ids=ledger_ids, location=location,
                      test=test_text,
                      statistic_name=MEASURES[measure]['statistic_name'], statistic=float(entry['change']),
                      n=LIST_TYPES[list_type]['n'], unit=LIST_TYPES[list_type]['unit'], df='', p='', p_type='none',
                      effect_size_name=MEASURES[measure]['effect_size_name'], effect_size=float(entry['change']),
                      ci_low=float(entry['low']), ci_high=float(entry['high']), ci_method=ci_method,
                      resampling_unit=LIST_TYPES[list_type]['resampling_unit'], n_resamples=number_of_draws,
                      seed=seed, script=script)
tests.write(paths['outputs'])
