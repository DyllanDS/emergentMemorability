"""
make_figureS5.py: SI Appendix, Fig. S5, the same item on crowded and uncrowded lists in the REM
model.

WHAT THE FIGURE SHOWS
  The REM model at the parameter values of Fig. 1 (g 0.65, 49 features, storage 0.30, copy
  fidelity 0.60, criterion 1.0), 40-item study lists drawn from a 1,600-item database. Two ways of
  crowding the same item:
    Random lists:   lists drawn at random; items that happened to have 3 or more close neighbors
                    on their list compared with items that had none (close = among the item's 50
                    most similar items in the database).
    Designed lists: each of 300 critical items studied on five lists that carry 0, 2, 4, 8 or 16 of
                    its next most similar items; 16 compared with 0. The lure is the item's most
                    similar item, never studied.
  (a) Change in hit rate (blue), false alarm rate (red) and hit rate minus false alarm rate (dark
      gray) from the uncrowded to the crowded list. Both rates rise, so their difference barely
      moves.
  (b) Change in d′ over the same comparison. d′ falls on both kinds of list, more on designed lists.
  Error bars: 95% percentile bootstrap confidence intervals (designed: resampling the 300 critical
  items; random: resampling the 600 lists; 2,000 resamples, seed 20261010), from list_bootstrap.py.

DATA
  outputs/list_bootstrap.json     the changes and intervals (list_bootstrap.py)
  data/paper_run/list_designed.txt, list_binned.txt   the printed tables of the saved run; the
                                  plotted changes are checked against their CELL 2 "0 -> 16" and
                                  "0 -> 3+" lines

OUTPUTS
  outputs/figures/FigS5_list_context.png (300 dpi)
  outputs/figure_data/FigS5a.csv, FigS5b.csv (the plotted numbers, one row per bar)

Run (from the repository root):
    python A2d_list_context_figS5/code/make_figureS5.py
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import release_paths  # noqa: E402
import figure_style as ls  # noqa: E402
import list_bootstrap  # noqa: E402  (this module's readers of the printed tables)

paths = release_paths.module_paths(__file__)
bootstrap_file = os.path.join(paths['outputs'], 'list_bootstrap.json')
if not os.path.exists(bootstrap_file):
    sys.exit('make_figureS5.py: outputs/list_bootstrap.json is missing; run code/list_bootstrap.py first')
with open(bootstrap_file, encoding='utf-8') as handle:
    bootstrap = json.load(handle)

# ---- check the json against the "0 -> 16" and "0 -> 3+" lines of CELL 2 in the saved tables
designed_change = list_bootstrap.designed_table_rows(
    list_bootstrap.read_cell_block(os.path.join(paths['paper_run'], 'list_designed.txt'), 2))['change']
assert abs(bootstrap['designed']['hit']['change'] - designed_change['hit']) < 0.0006
assert abs(bootstrap['designed']['false_alarm']['change'] - designed_change['false_alarm']) < 0.0006
assert abs(bootstrap['designed']['dprime']['change'] - designed_change['dprime']) < 0.0006
random_change = list_bootstrap.random_table_rows(
    list_bootstrap.read_cell_block(os.path.join(paths['paper_run'], 'list_binned.txt'), 2))['change']
for key in ('hit', 'false_alarm', 'hits_minus_false_alarms', 'dprime'):
    assert abs(bootstrap['random'][key]['change'] - random_change[key]) < 0.0006, key
print('list_bootstrap.json agrees with list_designed.txt and list_binned.txt (CELL 2)')

list_types = ['random', 'designed']
group_labels = ['Random Lists\n(0 vs 3+ Similar)', 'Designed Lists\n(0 vs 16 Similar)']

ls.apply_style('bars')
fig, (axis_rates, axis_dprime) = ls.new_figure(n_panels=2, width='double', height_cm=7.0)

# ---- panel a: change in hits, false alarms, hits minus false alarms
measures = [('hit', ls.HIT_BLUE, 'Hits', 'blue'),
            ('false_alarm', ls.FA_RED, 'False\nAlarms', 'red'),
            ('hits_minus_false_alarms', ls.MEMORY_DARK, 'Hits Minus\nFalse Alarms', 'dark gray')]
bar_width = 0.26
group_positions = ls.category_axis(axis_rates, group_labels)
panel_a_rows = []
for measure_number, (key, color, label, color_name) in enumerate(measures):
    offset = (measure_number - 1) * bar_width
    positions = group_positions + offset
    changes = [bootstrap[list_type][key]['change'] for list_type in list_types]
    lows = [bootstrap[list_type][key]['low'] for list_type in list_types]
    highs = [bootstrap[list_type][key]['high'] for list_type in list_types]
    axis_rates.bar(positions, changes, width=bar_width, color=color, edgecolor='none', zorder=2)
    ls.error_bars(axis_rates, positions, changes, lows, highs)
    # name each bar once, on the designed-list group
    if changes[1] > 0:
        ls.direct_label(axis_rates, positions[1], highs[1], label, color, ha='center', va='bottom',
                        dx_points=0, dy_points=2)
    else:
        ls.direct_label(axis_rates, positions[1], lows[1], label, color, ha='center', va='top',
                        dx_points=0, dy_points=-2)
    for list_index, list_type in enumerate(list_types):
        panel_a_rows.append({'panel': 'a', 'list_type': list_type,
                             'group_label': group_labels[list_index].replace('\n', ' '),
                             'measure': key, 'bar_label': label.replace('\n', ' '), 'bar_color': color_name,
                             'x_position': float(positions[list_index]), 'change': changes[list_index],
                             'ci_low': lows[list_index], 'ci_high': highs[list_index]})
ls.zero_line(axis_rates)
axis_rates.set_ylim(-0.08, 0.16)
axis_rates.set_yticks([-0.08, -0.04, 0, 0.04, 0.08, 0.12, 0.16])
axis_rates.set_ylabel('Change in Rate\n(Crowded Minus Uncrowded List)')
ls.plain_ticks(axis_rates)

# ---- panel b: change in d′
dprime_changes = [bootstrap[list_type]['dprime']['change'] for list_type in list_types]
dprime_lows = [bootstrap[list_type]['dprime']['low'] for list_type in list_types]
dprime_highs = [bootstrap[list_type]['dprime']['high'] for list_type in list_types]
ls.simple_bars(axis_dprime, group_labels, dprime_changes, colors=[ls.MEMORY_DARK] * 2,
               low=dprime_lows, high=dprime_highs, bar_width=0.45)
ls.zero_line(axis_dprime)
axis_dprime.set_ylim(-0.2, 0.05)
axis_dprime.set_yticks([-0.2, -0.15, -0.1, -0.05, 0, 0.05])
axis_dprime.set_ylabel('Change in d′\n(Crowded Minus Uncrowded List)')
ls.plain_ticks(axis_dprime)
panel_b_rows = []
for list_index, list_type in enumerate(list_types):
    panel_b_rows.append({'panel': 'b', 'list_type': list_type,
                         'group_label': group_labels[list_index].replace('\n', ' '),
                         'measure': 'dprime', 'x_position': float(list_index + 1),
                         'change': dprime_changes[list_index], 'ci_low': dprime_lows[list_index],
                         'ci_high': dprime_highs[list_index],
                         'dprime_uncrowded': bootstrap[list_type]['start_dprime'],
                         'dprime_crowded': bootstrap[list_type]['end_dprime']})

ls.panel_label(axis_rates, 'a')
ls.panel_label(axis_dprime, 'b')
ls.save_figure(fig, 'FigS5_list_context', paths['figures'])


def write_rows(file_name, rows):
    """One CSV row per plotted bar; numbers written with full precision (repr)."""
    path = os.path.join(paths['figure_data'], file_name)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (repr(value) if isinstance(value, float) else value) for key, value in row.items()})
    print('wrote', os.path.relpath(path, paths['module']))


write_rows('FigS5a.csv', panel_a_rows)
write_rows('FigS5b.csv', panel_b_rows)

print('\nNUMBERS (change from uncrowded to crowded list, 95% bootstrap interval)')
for list_type in list_types:
    for key in ('hit', 'false_alarm', 'hits_minus_false_alarms', 'dprime'):
        entry = bootstrap[list_type][key]
        print('  %-8s %-24s %+.3f [%+.3f, %+.3f]' % (list_type, key, entry['change'], entry['low'], entry['high']))
        # the claims the figure makes: both rates rise, their difference and d′ fall
        if key in ('hit', 'false_alarm'):
            assert entry['low'] > 0
        else:
            assert entry['high'] < 0
    print("  %-8s d' %.3f -> %.3f" % (list_type, bootstrap[list_type]['start_dprime'], bootstrap[list_type]['end_dprime']))
