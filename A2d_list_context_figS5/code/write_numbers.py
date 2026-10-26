"""
write_numbers.py: writes outputs/numbers.csv, one row for each of the 14 ledger items of this module
(SI Appendix, Section 2, List Context, and SI Appendix, Fig. S5), with the value this run computes
and whether it agrees with the paper's value at the printed precision.

READS (all written by this module's other scripts in the same run)
  outputs/list_designed_cell2.mat, outputs/list_binned_cell2.mat   the settings of the two
                                    simulations as run (list length, numbers of lists, items and
                                    simulated participants, the neighborhood size, the conditions)
  outputs/list_bootstrap.json       the changes from the uncrowded to the crowded list and their
                                    intervals (list_bootstrap.py, 2,000 resamples, the error bars of
                                    Fig. S5)
  outputs/tests.csv                 the same changes with intervals from 10,000 resamples
                                    (stats_figS5.py), the intervals printed in the text
  outputs/equality_with_paper_run.csv   the line-by-line comparison with the saved run's tables
  outputs/figure_data/FigS5a.csv, FigS5b.csv   the plotted bars (make_figureS5.py)
The paper's value, location and quote of every item come from shared/ledger/paper_numbers.csv; the
match is judged by shared/python/numbers_table.py.

WRITES  outputs/numbers.csv

Run (from the repository root):
    python A2d_list_context_figS5/code/write_numbers.py
"""
import csv
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import release_paths  # noqa: E402
import numbers_table  # noqa: E402
import list_bootstrap  # noqa: E402  (this module's readers)

paths = release_paths.module_paths(__file__)
MODE = 'rerun'
SIMULATION_SCRIPTS = 'code/list_designed.m; code/list_binned.m'
BOOTSTRAP_SCRIPTS = 'code/list_designed.m; code/list_binned.m; code/list_bootstrap.py'
TEXT_INTERVAL_SCRIPTS = 'code/list_designed.m; code/list_binned.m; code/stats_figS5.py'

needed_files = {
    'designed': os.path.join(paths['outputs'], 'list_designed_cell2.mat'),
    'random': os.path.join(paths['outputs'], 'list_binned_cell2.mat'),
    'bootstrap': os.path.join(paths['outputs'], 'list_bootstrap.json'),
    'tests': os.path.join(paths['outputs'], 'tests.csv'),
    'equality': os.path.join(paths['outputs'], 'equality_with_paper_run.csv'),
    'panel_a': os.path.join(paths['figure_data'], 'FigS5a.csv'),
    'panel_b': os.path.join(paths['figure_data'], 'FigS5b.csv'),
}
for name, needed in needed_files.items():
    if not os.path.exists(needed):
        sys.exit('write_numbers.py: %s is missing; run python A2d_list_context_figS5/code/run_module.py'
                 % os.path.relpath(needed, paths['module']))

designed = list_bootstrap.load_designed_items(needed_files['designed'])
random_lists = list_bootstrap.load_random_list_items(needed_files['random'])
designed_settings = designed['parameters']
random_settings = random_lists['parameters']
with open(needed_files['bootstrap'], encoding='utf-8') as handle:
    bootstrap = json.load(handle)


def read_csv_rows(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


# CELL 2 lines of this run against the saved run (check_against_paper_run.py)
equality_rows = read_csv_rows(needed_files['equality'])
cell2_lines = 0
cell2_identical = 0
for row in equality_rows:
    if row['cell'] == 'CELL 2':
        cell2_lines += 1
        if row['identical'] == 'yes':
            cell2_identical += 1
equality_note = ('CELL 2 lines of both tables identical to the saved run in data/paper_run/: %d of %d '
                 '(outputs/equality_with_paper_run.csv)' % (cell2_identical, cell2_lines))


# the intervals printed in the text: 10,000 resamples, seed 20261013 (outputs/tests.csv, stats_figS5.py)
TEXT_RESAMPLES = 10000
TEXT_SEED = 20261013
STATISTIC_NAMES = {'hit': 'change in hit rate (crowded minus uncrowded)',
                   'false_alarm': 'change in false alarm rate (crowded minus uncrowded)',
                   'dprime': 'change in d\u2032 (crowded minus uncrowded)'}
text_intervals = {}
for row in read_csv_rows(needed_files['tests']):
    if int(row['n_resamples']) != TEXT_RESAMPLES:
        continue
    assert int(row['seed']) == TEXT_SEED
    list_type = 'designed' if '; designed lists:' in row['test'] else 'random'
    for key, statistic_name in STATISTIC_NAMES.items():
        if row['statistic_name'] == statistic_name:
            # tests.csv holds 6 significant digits; the point estimate is the one of list_bootstrap.json
            assert abs(float(row['statistic']) - bootstrap[list_type][key]['change']) < 1e-5, (list_type, key)
            text_intervals[(list_type, key)] = (float(row['ci_low']), float(row['ci_high']))
assert len(text_intervals) == 6, sorted(text_intervals)


def change_text(list_type, key):
    """'+0.0787 [+0.0698, +0.0876]' for one change and its 95% interval from 10,000 resamples."""
    low, high = text_intervals[(list_type, key)]
    return '%+.4f [%+.4f, %+.4f]' % (bootstrap[list_type][key]['change'], low, high)


table = numbers_table.NumbersTable(script=SIMULATION_SCRIPTS)

# both simulations use the same study list length and parameter values
assert designed_settings['nStudy'] == random_settings['nStudy']
for name in ('g', 'w', 'u', 'c', 'crit'):
    assert designed_settings[name] == random_settings[name], name
similar_item_levels = designed['similar_item_levels']
similar_item_levels_text = ', '.join(str(level) for level in similar_item_levels)
number_of_critical_items = designed['hit_rate_by_item'].shape[0]
random_dprime = bootstrap['random']['dprime']['change']
designed_dprime = bootstrap['designed']['dprime']['change']

# ---------------------------------------------------------------------------- SI Appendix, Section 2, List Context
table.add('L6-069', int(designed_settings['nStudy']), MODE,
          note=('Study list length (nStudy) of both simulations; the items come from a database of %s simulated '
                'items (nDatabase)' % '{:,}'.format(int(designed_settings['nDatabase']))))

parameter_text = 'g %.2f; w %d; u %.2f; c %.2f; crit %.1f' % (
    designed_settings['g'], designed_settings['w'], designed_settings['u'], designed_settings['c'],
    designed_settings['crit'])
table.add('L6-070', parameter_text, MODE,
          note=('REM settings of both simulations (CELL 2 of list_designed.m and list_binned.m), the parameter values '
                'of Fig. 1: g of the geometric distribution, number of features w, storage probability u, copy '
                'fidelity c and decision criterion'))

table.add('L6-071', similar_item_levels[-1], MODE,
          note=('Designed lists (conditions of %s similar items): number of the critical item\'s next most similar '
                'items (neighbor ranks 2 to %d) on its crowded list; its nearest neighbor is its lure and is never '
                'studied' % (similar_item_levels_text, similar_item_levels[-1] + 1)))

text_dprime_random = text_intervals[('random', 'dprime')]
text_dprime_designed = text_intervals[('designed', 'dprime')]
table.add('L6-072', [-random_dprime, -text_dprime_random[1], -text_dprime_random[0],
                     -designed_dprime, -text_dprime_designed[1], -text_dprime_designed[0]],
          MODE, script=TEXT_INTERVAL_SCRIPTS,
          note=('Fall in d\u2032 from the uncrowded to the crowded list (minus the change in d\u2032) and its 95%% '
                'percentile interval from %s resamples (seed %d): random lists, test items with 3 or more close '
                'neighbors on their list against items with none, change %s (d\u2032 %.4f to %.4f; d-prime %+.3f in '
                'the "0 -> 3+" row of CELL 2 in list_binned.txt; hit rate minus false alarm rate %+.4f); designed '
                'lists, 16 against 0 similar items, change %s (d\u2032 %.4f to %.4f; d-prime %+.3f in the "0 -> 16" '
                'line of CELL 2 in list_designed.txt); %s'
                % ('{:,}'.format(TEXT_RESAMPLES), TEXT_SEED, change_text('random', 'dprime'),
                   bootstrap['random']['start_dprime'], bootstrap['random']['end_dprime'], random_dprime,
                   bootstrap['random']['hits_minus_false_alarms']['change'], change_text('designed', 'dprime'),
                   bootstrap['designed']['start_dprime'], bootstrap['designed']['end_dprime'], designed_dprime,
                   equality_note)))

text_hit = text_intervals[('designed', 'hit')]
text_false_alarm = text_intervals[('designed', 'false_alarm')]
table.add('L6-080', [bootstrap['designed']['hit']['change'], text_hit[0], text_hit[1],
                     bootstrap['designed']['false_alarm']['change'], text_false_alarm[0], text_false_alarm[1]],
          MODE, script=TEXT_INTERVAL_SCRIPTS,
          note=('Designed lists, 0 to 16 similar items: change in the hit rate of the critical items (%.4f to %.4f) '
                'and in the false alarm rate of their lures (%.4f to %.4f), means over the %d critical items, with '
                '95%% percentile intervals from %s resamples (seed %d): %s and %s; %s'
                % (bootstrap['designed']['start_hit'], bootstrap['designed']['end_hit'],
                   bootstrap['designed']['start_fa'], bootstrap['designed']['end_fa'], number_of_critical_items,
                   '{:,}'.format(TEXT_RESAMPLES), TEXT_SEED, change_text('designed', 'hit'),
                   change_text('designed', 'false_alarm'), equality_note)))

table.add('L6-073',
          'random lists: hits %+.4f, false alarms %+.4f, hits minus false alarms %+.4f; designed lists: hits %+.4f, '
          'false alarms %+.4f, hits minus false alarms %+.4f'
          % (bootstrap['random']['hit']['change'], bootstrap['random']['false_alarm']['change'],
             bootstrap['random']['hits_minus_false_alarms']['change'], bootstrap['designed']['hit']['change'],
             bootstrap['designed']['false_alarm']['change'], bootstrap['designed']['hits_minus_false_alarms']['change']),
          MODE, script=BOOTSTRAP_SCRIPTS,
          note=('Change from the uncrowded to the crowded list in hit rate, false alarm rate and hit rate minus false '
                'alarm rate, on random lists (3 or more against 0 close neighbors) and designed lists (16 against 0 '
                'similar items); the designed-list hits and false alarms are those of L6-080; %s' % equality_note))

# ---------------------------------------------------------------------------- SI Appendix, Fig. S5
panel_a_parts = []
for row in read_csv_rows(needed_files['panel_a']):
    panel_a_parts.append('%s %s %+.4f [%+.4f, %+.4f]' % (row['list_type'], row['bar_label'].lower(),
                                                         float(row['change']), float(row['ci_low']),
                                                         float(row['ci_high'])))
table.add('L6-074', '; '.join(panel_a_parts), MODE, script='code/make_figureS5.py',
          note=('Fig. S5a: change from the uncrowded to the crowded list in hit rate, false alarm rate and hit rate '
                'minus false alarm rate, with 95% percentile bootstrap intervals (outputs/figures/FigS5_list_context.png; '
                'plotted numbers in outputs/figure_data/FigS5a.csv)'))

panel_b_parts = []
for row in read_csv_rows(needed_files['panel_b']):
    panel_b_parts.append('%s d\u2032 %+.4f [%+.4f, %+.4f]' % (row['list_type'], float(row['change']),
                                                            float(row['ci_low']), float(row['ci_high'])))
table.add('L6-075', '; '.join(panel_b_parts), MODE, script='code/make_figureS5.py',
          note=('Fig. S5b: change in d\u2032 from the uncrowded to the crowded list, with 95% percentile bootstrap '
                'intervals (outputs/figure_data/FigS5b.csv)'))

# ---------------------------------------------------------------------------- SI Appendix, Fig. S5 caption
table.add('L6-281', int(designed_settings['nStudy']), MODE,
          note=('Study list length (nStudy) of both simulations, as in L6-069; both run at the parameter values of '
                'Fig. 1 (%s)' % parameter_text))

table.add('L6-076', [int(random_settings['nLists']), int(random_settings['nSubjPerList'])], MODE,
          note=('Random lists: number of lists (nLists) and simulated participants per list (nSubjPerList); each list '
                'holds %d studied items and %d lures drawn at random' % (int(random_settings['nStudy']),
                                                                         int(random_settings['nLure']))))

top_bin_label, top_bin_lowest_count, top_bin_highest_count = list_bootstrap.NEIGHBOUR_COUNT_BINS[-1]
table.add('L6-077', [int(top_bin_lowest_count), int(random_settings['nNeighbour'])], MODE,
          note=('Random lists: smallest count of close neighbors in the crowded bin (bins %s) and the neighborhood '
                'size (an item\'s %d nearest items in the database, nNeighbour); the count is taken over a studied '
                'item\'s %d list companions and over the %d studied items for a lure; Fig. S5 compares the %s bin '
                'with the 0 bin'
                % (', '.join(label for label, low, high in list_bootstrap.NEIGHBOUR_COUNT_BINS),
                   int(random_settings['nNeighbour']), int(random_settings['nStudy']) - 1,
                   int(random_settings['nStudy']), top_bin_label)))

table.add('L6-078', [number_of_critical_items, designed['participants_per_list']], MODE,
          note=('Designed lists: number of critical items (nCritical) and simulated participants per list (nSubj); '
                'each critical item is studied on %d lists' % len(similar_item_levels)))

table.add('L6-079', [similar_item_levels[0], similar_item_levels[-1]], MODE,
          note='Designed lists: numbers of similar items on the two lists Fig. S5 compares (the simulation runs %s)'
               % similar_item_levels_text)

interval_percentiles = bootstrap['interval_percentiles']
table.add('L6-082', interval_percentiles[1] - interval_percentiles[0], MODE, script=BOOTSTRAP_SCRIPTS,
          note=('Coverage of the %s intervals of Fig. S5 (the %.1fth to %.1fth percentiles of %s resamples, seed %d), '
                'resampling the %d lists (random lists) or the %d critical items (designed lists)'
                % (bootstrap['interval_method'], interval_percentiles[0], interval_percentiles[1],
                   '{:,}'.format(bootstrap['bootstraps']), bootstrap['seed'], bootstrap['random']['number_of_units'],
                   bootstrap['designed']['number_of_units'])))

table.write(paths['outputs'])
for row in table.rows:
    print('  %-7s %-10s paper %-40s computed %s' % (row['ledger_id'], row['match'], row['paper_value'][:40],
                                                    row['computed_value'][:70]))
not_matching = [row['ledger_id'] for row in table.rows if row['match'] == 'no']
if not_matching:
    sys.exit('write_numbers.py: computed values that do not match the paper: %s' % ', '.join(not_matching))
