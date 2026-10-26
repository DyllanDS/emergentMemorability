"""
list_bootstrap.py: the changes from the uncrowded to the crowded list and their 95% bootstrap
confidence intervals, the error bars of SI Appendix, Fig. S5 (2,000 resamples, seed 20261010), from
the per-item outputs of the two list simulations.

WHAT IT COMPUTES
  REM at the parameter values of Fig. 1 (g 0.65, 49 features, storage 0.30, copy fidelity 0.60,
  criterion 1.0; "CELL 2" of list_designed.m and list_binned.m). For the change from the uncrowded
  to the crowded list: the change in hit rate, in false alarm rate, in hit rate minus false alarm
  rate, and in d′, each with a 95% percentile bootstrap interval.
    Designed lists: 16 vs 0 similar items on the critical item's list, for each of 300 critical
      items studied on five lists (0, 2, 4, 8 or 16 similar items). Resampled unit: the critical
      item (300 drawn with replacement; the five lists of an item stay together), so the interval
      shows how much the result depends on which items were simulated.
    Random lists: test items with 3 or more close neighbors on their list vs test items with none
      (close = among the item's 50 nearest neighbors in the 1,600-item database). Resampled unit:
      the list (600 drawn with replacement; all test items of a list stay together).
  d′ of a condition comes from the mean hit and false alarm rates, as list_designed.m and
  list_binned.m compute it: z(hit rate) minus z(false alarm rate), with the rates kept within
  1/(2 x 200) and 1 - 1/(2 x 200) on the designed lists (200 simulated participants per list) and
  within 0.0001 and 0.9999 on the random lists.
  Before bootstrapping it checks that the per-item outputs reproduce the CELL 2 tables of the saved
  run (data/paper_run/) at their 3 printed decimals, and stops if they do not.
  The functions below are also used by stats_figS5.py (intervals from 10,000 resamples, seed
  20261013), make_figureS5.py and write_numbers.py.

READS   outputs/list_designed_cell2.mat, outputs/list_binned_cell2.mat (written by list_designed.m and
        list_binned.m)
        data/paper_run/list_designed.txt, data/paper_run/list_binned.txt (the printed tables of the
        saved run)
WRITES  outputs/list_bootstrap.txt (readable report) and outputs/list_bootstrap.json (the numbers)

Run (from the repository root):
    python A2d_list_context_figS5/code/list_bootstrap.py
Seed: one generator, numpy.random.default_rng(20261010): 2,000 resamples for the designed lists
first, then 2,000 for the random lists. Runtime: a few seconds.
"""
import json
import os
import re
import sys

import numpy as np
from scipy.io import loadmat
from scipy.stats import norm

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402

PAPER_NUMBER_OF_BOOTSTRAPS = 2000               # the error bars of Fig. S5
PAPER_RANDOM_SEED = 20261010
SIMILAR_ITEM_LEVELS = [0, 2, 4, 8, 16]          # nSimilar in list_designed.m
NEIGHBOUR_COUNT_BINS = [('0', 0, 1), ('1', 1, 2), ('2', 2, 3), ('3+', 3, np.inf)]   # bins of list_binned.m
CHANGE_NAMES = ['hit', 'false_alarm', 'hits_minus_false_alarms', 'dprime']
INTERVAL_PERCENTILES = [2.5, 97.5]              # a 95% percentile interval


# =============================================================================== reading the tables and the per-item outputs
def read_cell_block(table_file, cell_number):
    """The text of one parameter cell ('CELL <k>: ...' up to the next CELL or READING THIS) of a
    printed table (list_designed.txt or list_binned.txt)."""
    with open(table_file, encoding='utf-8') as handle:
        text = handle.read()
    start = text.index('CELL %d:' % cell_number)
    rest = text[start:]
    ends = [rest.find('CELL %d:' % (cell_number + 1)), rest.find('READING THIS')]
    ends = [position for position in ends if position > 0]
    return rest[:min(ends)] if ends else rest


def designed_table_rows(cell_text):
    """Rows of a designed-list table, by their label (number of similar items):
    {level: dict(crowding, hit, false_alarm, dprime, criterion)}, plus the '0 -> 16' line."""
    rows = {}
    for level in SIMILAR_ITEM_LEVELS:
        match = re.search(r'^\s+%d\s+(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$' % level,
                          cell_text, re.M)
        if match is None:
            raise ValueError('no table row for %d similar items' % level)
        rows[level] = dict(crowding=float(match.group(1)), hit=float(match.group(2)),
                           false_alarm=float(match.group(3)), dprime=float(match.group(4)),
                           criterion=float(match.group(5)))
    change = re.search(r'0 -> (\d+) similar items:\s+hit ([+-][\d.]+)\s+FA ([+-][\d.]+)\s+'
                       r'd-prime ([+-][\d.]+)\s+criterion ([+-][\d.]+)', cell_text)
    if change is None:
        raise ValueError('no "0 -> 16 similar items" line')
    rows['change'] = dict(to_level=int(change.group(1)), hit=float(change.group(2)),
                          false_alarm=float(change.group(3)), dprime=float(change.group(4)),
                          criterion=float(change.group(5)))
    header = re.search(r'(\d+) critical items x (\d+) conditions x (\d+) participants', cell_text)
    rows['design'] = dict(critical_items=int(header.group(1)), conditions=int(header.group(2)),
                          participants_per_list=int(header.group(3)))
    return rows


def random_table_rows(cell_text):
    """Rows of the random-list table 'BY COUNT OF CLOSE NEIGHBORS ON THE LIST', by their label:
    {label: dict(n_old, n_new, hit, false_alarm, hits_minus_false_alarms, dprime)}, plus the
    '0 -> 3+' line and the design line."""
    by_count = cell_text.split('WITHIN ITEM')[0]
    rows = {}
    for label, low, high in NEIGHBOUR_COUNT_BINS:
        match = re.search(r'^\s+%s\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$'
                          % re.escape(label), by_count, re.M)
        if match is None:
            raise ValueError('no table row for the bin %s' % label)
        rows[label] = dict(n_old=int(match.group(1)), n_new=int(match.group(2)), hit=float(match.group(3)),
                           false_alarm=float(match.group(4)), hits_minus_false_alarms=float(match.group(5)),
                           dprime=float(match.group(6)))
    change = re.search(r'0 -> 3\+:\s+([+-][\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)', by_count)
    if change is None:
        raise ValueError('no "0 -> 3+" line')
    rows['change'] = dict(hit=float(change.group(1)), false_alarm=float(change.group(2)),
                          hits_minus_false_alarms=float(change.group(3)), dprime=float(change.group(4)))
    header = re.search(r'(\d+) lists x (\d+) participants; "similar" = inside an item\'s (\d+) nearest neighbors',
                       cell_text)
    rows['design'] = dict(lists=int(header.group(1)), participants_per_list=int(header.group(2)),
                          nearest_neighbors=int(header.group(3)))
    return rows


def load_designed_items(mat_file):
    """The per-critical-item rates of one cell of list_designed.m and the cell's settings."""
    saved = loadmat(mat_file, simplify_cells=True)
    parameters = saved['parameters']
    similar_item_levels = [int(level) for level in np.atleast_1d(parameters['nSimilar'])]
    assert similar_item_levels == SIMILAR_ITEM_LEVELS, similar_item_levels
    return dict(hit_rate_by_item=np.asarray(saved['hitRate'], dtype=float),     # critical items x 5 conditions
                fa_rate_by_item=np.asarray(saved['faRate'], dtype=float),
                participants_per_list=int(parameters['nSubj']),
                similar_item_levels=similar_item_levels,
                parameters=parameters)


def load_random_list_items(mat_file):
    """The per-test-item rates of one cell of list_binned.m and the cell's settings."""
    saved = loadmat(mat_file, simplify_cells=True)
    assert int(np.max(saved['oList'])) == int(saved['parameters']['nLists'])
    return dict(studied_list=np.asarray(saved['oList'], dtype=np.int64),     # which list each studied item was on
                studied_hit=np.asarray(saved['oHit'], dtype=float),          # its hit rate over the list's participants
                studied_count=np.asarray(saved['oCnt'], dtype=np.int64),     # close neighbors among its 39 companions
                lure_list=np.asarray(saved['lList'], dtype=np.int64),
                lure_fa=np.asarray(saved['lFA'], dtype=float),
                lure_count=np.asarray(saved['lCnt'], dtype=np.int64),        # close neighbors among the 40 studied items
                number_of_lists=int(saved['parameters']['nLists']),
                parameters=saved['parameters'])


# =============================================================================== d′ as list_designed.m and list_binned.m compute it
def designed_z(rate, participants_per_list):
    """z transform of list_designed.m: rates kept within 1/(2 x participants) and 1 - 1/(2 x participants)."""
    clip = 1.0 / (2 * participants_per_list)
    return norm.ppf(np.clip(rate, clip, 1 - clip))


def binned_z(rate):
    """z transform of list_binned.m: rates kept within 0.0001 and 0.9999."""
    return norm.ppf(np.clip(rate, 1e-4, 1 - 1e-4))


def summary_changes(hit_start, hit_end, fa_start, fa_end, z_function):
    """Changes from the uncrowded to the crowded condition in hits, false alarms, hits minus false
    alarms, and d′ (d′ from the mean rates, as in list_designed.m and list_binned.m)."""
    dprime_start = z_function(hit_start) - z_function(fa_start)
    dprime_end = z_function(hit_end) - z_function(fa_end)
    return {
        'hit': hit_end - hit_start,
        'false_alarm': fa_end - fa_start,
        'hits_minus_false_alarms': (hit_end - fa_end) - (hit_start - fa_start),
        'dprime': dprime_end - dprime_start,
        'dprime_start': dprime_start,
        'dprime_end': dprime_end,
        'hit_start': hit_start, 'hit_end': hit_end, 'fa_start': fa_start, 'fa_end': fa_end,
    }


def designed_list_changes(designed, chosen_items):
    """Changes from 0 to 16 similar items over the chosen critical items (an item drawn twice
    counts twice)."""
    hit_rate_by_item = designed['hit_rate_by_item']
    fa_rate_by_item = designed['fa_rate_by_item']

    def z_function(rate):
        return designed_z(rate, designed['participants_per_list'])
    return summary_changes(hit_rate_by_item[chosen_items, 0].mean(), hit_rate_by_item[chosen_items, -1].mean(),
                           fa_rate_by_item[chosen_items, 0].mean(), fa_rate_by_item[chosen_items, -1].mean(),
                           z_function)


def random_list_changes(random_lists, chosen_lists):
    """Changes from 0 to 3+ close neighbors, pooling the test items of the chosen lists
    (list numbers 1 to 600; a list drawn twice counts twice)."""
    copies_of_list = np.bincount(chosen_lists, minlength=random_lists['number_of_lists'] + 1)
    studied_weight = copies_of_list[random_lists['studied_list']]
    lure_weight = copies_of_list[random_lists['lure_list']]
    studied_hit = random_lists['studied_hit']
    studied_count = random_lists['studied_count']
    lure_fa = random_lists['lure_fa']
    lure_count = random_lists['lure_count']

    def weighted_mean(values, weights, mask):
        return np.sum(values[mask] * weights[mask]) / np.sum(weights[mask])
    return summary_changes(weighted_mean(studied_hit, studied_weight, studied_count == 0),
                           weighted_mean(studied_hit, studied_weight, studied_count >= 3),
                           weighted_mean(lure_fa, lure_weight, lure_count == 0),
                           weighted_mean(lure_fa, lure_weight, lure_count >= 3), binned_z)


# =============================================================================== the bootstraps
def bootstrap_designed_lists(designed, number_of_draws, random_generator):
    """number_of_draws resamples of the critical items (with replacement), each giving the four
    changes. Draws random_generator.integers(0, n, n) once per resample, in order."""
    number_of_critical_items = designed['hit_rate_by_item'].shape[0]
    draws = {key: [] for key in CHANGE_NAMES}
    for draw in range(number_of_draws):
        chosen_items = random_generator.integers(0, number_of_critical_items, number_of_critical_items)
        changes = designed_list_changes(designed, chosen_items)
        for key in CHANGE_NAMES:
            draws[key].append(changes[key])
    return draws


def bootstrap_random_lists(random_lists, number_of_draws, random_generator):
    """number_of_draws resamples of the lists (with replacement), each giving the four changes.
    Draws random_generator.integers(1, number_of_lists + 1, number_of_lists) once per resample."""
    number_of_lists = random_lists['number_of_lists']
    draws = {key: [] for key in CHANGE_NAMES}
    for draw in range(number_of_draws):
        chosen_lists = random_generator.integers(1, number_of_lists + 1, number_of_lists)
        changes = random_list_changes(random_lists, chosen_lists)
        for key in CHANGE_NAMES:
            draws[key].append(changes[key])
    return draws


def percentile_interval(values):
    """2.5th and 97.5th percentiles of the resampled values (numpy, linear interpolation)."""
    low, high = np.percentile(values, INTERVAL_PERCENTILES)
    return float(low), float(high)


# =============================================================================== checks against the saved tables
def check_designed_against_table(designed, cell_text):
    """The mean rates and d′ per condition equal the saved CELL 2 table (list_designed.txt) at 3 decimals."""
    rows = designed_table_rows(cell_text)
    for column, level in enumerate(SIMILAR_ITEM_LEVELS):
        mean_hit = designed['hit_rate_by_item'][:, column].mean()
        mean_fa = designed['fa_rate_by_item'][:, column].mean()
        rerun_dprime = (designed_z(mean_hit, designed['participants_per_list'])
                        - designed_z(mean_fa, designed['participants_per_list']))
        assert round(mean_hit, 3) == rows[level]['hit'], ('hit', level)
        assert round(mean_fa, 3) == rows[level]['false_alarm'], ('false alarm', level)
        assert round(rerun_dprime, 3) == rows[level]['dprime'], ('d-prime', level)
    assert designed['hit_rate_by_item'].shape[0] == rows['design']['critical_items']
    assert designed['participants_per_list'] == rows['design']['participants_per_list']


def check_random_against_table(random_lists, cell_text):
    """The counts, mean rates and d′ per bin equal the saved CELL 2 table (list_binned.txt) at 3 decimals."""
    rows = random_table_rows(cell_text)
    for label, low, high in NEIGHBOUR_COUNT_BINS:
        studied_in_bin = (random_lists['studied_count'] >= low) & (random_lists['studied_count'] < high)
        lures_in_bin = (random_lists['lure_count'] >= low) & (random_lists['lure_count'] < high)
        mean_hit = random_lists['studied_hit'][studied_in_bin].mean()
        mean_fa = random_lists['lure_fa'][lures_in_bin].mean()
        assert studied_in_bin.sum() == rows[label]['n_old'] and lures_in_bin.sum() == rows[label]['n_new'], label
        assert round(mean_hit, 3) == rows[label]['hit'] and round(mean_fa, 3) == rows[label]['false_alarm'], label
        assert round(binned_z(mean_hit) - binned_z(mean_fa), 3) == rows[label]['dprime'], label
    assert random_lists['number_of_lists'] == rows['design']['lists']


# =============================================================================== the intervals of Fig. S5 (2,000 resamples, seed 20261010)
def main():
    paths = release_paths.module_paths(__file__)
    report_lines = []

    def report(line=''):
        print(line)
        report_lines.append(line)

    designed_file = os.path.join(paths['outputs'], 'list_designed_cell2.mat')
    random_file = os.path.join(paths['outputs'], 'list_binned_cell2.mat')
    for needed in (designed_file, random_file):
        if not os.path.exists(needed):
            sys.exit('list_bootstrap.py: %s is missing; run code/list_designed.m and code/list_binned.m first '
                     '(or python A2d_list_context_figS5/code/run_module.py)' % os.path.relpath(needed, paths['module']))

    random_generator = np.random.default_rng(PAPER_RANDOM_SEED)

    # ----------------------------------------------------------------------- designed lists
    designed = load_designed_items(designed_file)
    check_designed_against_table(designed, read_cell_block(os.path.join(paths['paper_run'], 'list_designed.txt'), 2))
    report('Check against the saved run: designed-list CELL 2 rates and d-prime equal '
           'data/paper_run/list_designed.txt at 3 decimals.')
    number_of_critical_items = designed['hit_rate_by_item'].shape[0]
    designed_point = designed_list_changes(designed, np.arange(number_of_critical_items))
    designed_draws = bootstrap_designed_lists(designed, PAPER_NUMBER_OF_BOOTSTRAPS, random_generator)

    # ----------------------------------------------------------------------- random lists
    random_lists = load_random_list_items(random_file)
    check_random_against_table(random_lists, read_cell_block(os.path.join(paths['paper_run'], 'list_binned.txt'), 2))
    report('Check against the saved run: random-list CELL 2 counts, rates and d-prime equal '
           'data/paper_run/list_binned.txt at 3 decimals.')
    number_of_lists = random_lists['number_of_lists']
    random_point = random_list_changes(random_lists, np.arange(1, number_of_lists + 1))
    random_draws = bootstrap_random_lists(random_lists, PAPER_NUMBER_OF_BOOTSTRAPS, random_generator)

    # ----------------------------------------------------------------------- report
    results = {'seed': PAPER_RANDOM_SEED, 'bootstraps': PAPER_NUMBER_OF_BOOTSTRAPS,
               'interval_percentiles': INTERVAL_PERCENTILES, 'interval_method': 'percentile bootstrap'}
    report('')
    report('95%% bootstrap intervals (%d resamples, seed %d); percentile method' % (PAPER_NUMBER_OF_BOOTSTRAPS,
                                                                                   PAPER_RANDOM_SEED))
    analyses = [('designed', 'designed lists, 0 -> 16 similar items', designed_point, designed_draws,
                 'critical items (%d)' % number_of_critical_items, number_of_critical_items),
                ('random', 'random lists, 0 -> 3+ close neighbors', random_point, random_draws,
                 'lists (%d)' % number_of_lists, number_of_lists)]
    for list_type, name, point, draws, unit, number_of_units in analyses:
        report('')
        report('%s   [resampled: %s]' % (name, unit))
        report("  start: hit %.4f  FA %.4f  d' %.4f   end: hit %.4f  FA %.4f  d' %.4f" % (
            point['hit_start'], point['fa_start'], point['dprime_start'],
            point['hit_end'], point['fa_end'], point['dprime_end']))
        entry = {'start_hit': point['hit_start'], 'start_fa': point['fa_start'], 'start_dprime': point['dprime_start'],
                 'end_hit': point['hit_end'], 'end_fa': point['fa_end'], 'end_dprime': point['dprime_end'],
                 'resampled_unit': unit, 'number_of_units': number_of_units}
        for key in CHANGE_NAMES:
            low, high = percentile_interval(draws[key])
            report('  change in %-24s %+.4f  [%+.4f, %+.4f]' % (key, point[key], low, high))
            entry[key] = {'change': point[key], 'low': low, 'high': high}
        results[list_type] = entry

    with open(os.path.join(paths['outputs'], 'list_bootstrap.txt'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(report_lines) + '\n')
    with open(os.path.join(paths['outputs'], 'list_bootstrap.json'), 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2)
    print('wrote outputs/list_bootstrap.txt and outputs/list_bootstrap.json')


if __name__ == '__main__':
    main()
