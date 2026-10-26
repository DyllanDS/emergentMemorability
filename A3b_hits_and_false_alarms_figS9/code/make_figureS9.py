"""
make_figureS9.py: SI Appendix, Fig. S9, "Similar items on the study list change memorability mainly
through false alarms", and the numbers of its caption and of the text (SI Appendix, Sections 1 and 3).

WHAT THE FIGURE SHOWS
  (a) The memorability inversion experiment (main-text Fig. 3), its 40 critical items, over the 50 kept
      participants. Each critical item was studied by some participants and was an unstudied lure for
      the others, so it has a hit rate and a false alarm rate. x = the number of similar items on the
      study list (0, 1, 2, 4 or 8); y = hit rate (blue) and false alarm rate (red), each point the mean
      over the 8 critical items at that level. False alarms rise much more than hits.
  (b) The same items in d′ (from log-linear corrected rates): it falls as similar items are added.
  (c) The original THINGS data (Kramer et al. 2023), 26,107 photographs: the correlation across
      photographs of the memorability score (hit rate minus false alarm rate, computed from the trial
      records) with the hit rate (blue bar) and with the false alarm rate (red bar). Because false
      alarms sit near 2%, the score is almost a hit-rate score.
  Error bars in (a) and (b): 95% percentile bootstrap intervals from 10,000 resamples of the 8 critical
  items at each level.

  Also printed (and written to outputs/): the Spearman correlations over the 40 items between the
  number of similar items and each measure, the mean published score of the items at each level, and
  the halves comparison (the 20 more and the 20 less memorable items in THINGS, their hit and false
  alarm rates in THINGS and in this experiment).

INPUTS (relative to the repository root)
  03_inversion_experiment_fig3/data/   the inversion experiment, read through code/a3b_common.py and
                                       shared/python/inversion_loader.py (the exclusions described in
                                       Methods leave 50 of the 54 participants)
  A3b_hits_and_false_alarms_figS9/data/provided/per_image_rates.txt
                                       saved results of panel c, computed from the Kramer et al. trial
                                       records (used when input M1 of EXTERNAL_INPUTS.md is not
                                       available); outputs/regenerated/per_image_rates.txt instead
                                       when the trial records are here and code/per_image_rates.py
                                       has run
  A3b_hits_and_false_alarms_figS9/data/provided/halves_check.txt
                                       saved results of the THINGS side of the halves comparison
                                       (used when input M2 of EXTERNAL_INPUTS.md is not available);
                                       computed from external/kramer_trials/things_per_image_counts.csv
                                       when present
  A3b_hits_and_false_alarms_figS9/data/paper_run/figS9_numbers.txt
                                       the saved run that the paper reports: every number here is
                                       compared with it, digit for digit
  shared/python/figure_style.py        the style of the SI Appendix figures

OUTPUTS (A3b_hits_and_false_alarms_figS9/outputs/)
  figures/FigS9_hits_false_alarms.png             the figure, 300 dpi
  figure_data/FigS9a.csv, FigS9b.csv, FigS9c.csv  the plotted numbers of each panel
  level_table.csv          per level: the 8 items' mean hit rate, false alarm rate, memorability (hit
                           rate minus false alarm rate) and d′, the item-bootstrap intervals of the
                           error bars, the mean published score
  level_correlations.csv   the Spearman correlations over the 40 items between the number of similar
                           items and the false alarm rate, the hit rate and d′
  halves.csv               the halves comparison

SEED: numpy default_rng(0), drawn level by level in the order 0, 1, 2, 4, 8, one 10,000 x 8 matrix of
item positions per level.

RUN (from the repository root; a few seconds):  python A3b_hits_and_false_alarms_figS9/code/make_figureS9.py
"""
import os
import sys

sys.dont_write_bytecode = True
for thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(thread_setting, '2')     # at most two threads per numerical library (same results)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3b_common  # noqa: E402  (also puts shared/python on the path)
import figure_style as ls  # noqa: E402

PATHS = a3b_common.PATHS
LEVELS = a3b_common.LEVELS            # number of similar items studied (the design)
NUMBER_OF_RESAMPLES = 10000
RANDOM_SEED = 0
INTERVAL_PERCENTILES = [2.5, 97.5]    # a 95% percentile interval
OFFSET = 0.1                          # (a): false alarms drawn 0.1 left of each level, hits 0.1 right
FIGURE_NAME = 'FigS9_hits_false_alarms'
HALF_SIZE = 20                        # the halves comparison: 20 more and 20 less memorable items

# The measures of the level table: hit rate, false alarm rate and d′ of each item
MEASURES = ['hit_rate', 'fa_rate', 'dprime']


# =============================================================================================
# 1. The 40 critical items on the 50-participant sample: per-level means and item bootstrap
# =============================================================================================
def level_table_with_item_bootstrap(items):
    """The per-level table. For each level (0, 1, 2, 4, 8 similar items): the mean hit rate, false
    alarm rate and d′ over its 8 critical items, and a 95% percentile bootstrap interval from 10,000
    resamples of those 8 items (numpy default_rng(0); one 10,000 x 8 matrix of item positions is drawn
    per level, in the order of LEVELS). Returns the table."""
    random_generator = np.random.default_rng(RANDOM_SEED)
    rows = []
    for level in LEVELS:
        items_at_level = items[items['n_similar_items'] == level]
        number_of_items = len(items_at_level)                     # 8
        # Each row lists the positions of 8 items drawn with replacement from the 8 items of this level.
        resampled_rows = random_generator.integers(0, number_of_items, size=(NUMBER_OF_RESAMPLES, number_of_items))
        row = {'level': level, 'n_items': number_of_items}
        for measure in MEASURES:
            values = items_at_level[measure].to_numpy()
            row[measure] = values.mean()
            # Vectorized: the mean of each of the 10,000 resamples (one per row).
            means_of_resamples = values[resampled_rows].mean(axis=1)
            row[measure + '_ci_low'], row[measure + '_ci_high'] = np.percentile(means_of_resamples, INTERVAL_PERCENTILES)
        row['memorability'] = row['hit_rate'] - row['fa_rate']
        # The design check: the items given more similar items were the more memorable ones in THINGS
        # (each level holds one item from one fifth of the published score range per block).
        row['mean_published_score'] = items_at_level['kramer_cr'].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def level_correlations(items):
    """Spearman rank correlations over the 40 critical items between the designed number of similar
    items and each item's false alarm rate, hit rate and d′. The number of similar items takes 5
    values, each shared by 8 items (tied values get their average rank). p is scipy's two-sided p
    (t approximation with 38 degrees of freedom). Returns {printed name: (rho, p)}."""
    correlations = {}
    for printed_name, measure in [('false alarms', 'fa_rate'), ('hits', 'hit_rate'), ('d-prime', 'dprime')]:
        result = spearmanr(items['n_similar_items'], items[measure])
        correlations[printed_name] = (float(result.statistic), float(result.pvalue))
    return correlations


# =============================================================================================
# 2. The halves comparison
# =============================================================================================
def halves_table(items):
    """The 40 critical items sorted by their published score: the 20 more and the 20 less memorable
    ones. For each half, the mean false alarm rate and hit rate in this experiment (always computed)
    and in THINGS: computed from the per-image count table when it is here (input M2 of
    EXTERNAL_INPUTS.md), otherwise read from the saved results in data/provided/halves_check.txt.
    Returns (table, mode)."""
    scores_from_high_to_low = items.sort_values('kramer_cr', ascending=False, kind='mergesort')
    score_20th = scores_from_high_to_low['kramer_cr'].iloc[HALF_SIZE - 1]
    score_21st = scores_from_high_to_low['kramer_cr'].iloc[HALF_SIZE]
    if score_20th == score_21st:
        raise SystemExit('the 20th and 21st published scores are tied: the halves are not defined')
    more_memorable = scores_from_high_to_low.iloc[:HALF_SIZE]
    less_memorable = scores_from_high_to_low.iloc[HALF_SIZE:]

    mode = a3b_common.halves_mode()
    provided_halves = a3b_common.read_halves_check()
    rows = []
    for name, half in [('more memorable half', more_memorable), ('less memorable half', less_memorable)]:
        row = {'half': name, 'n_items': len(half),
               'published_score_min': half['kramer_cr'].min(), 'published_score_max': half['kramer_cr'].max(),
               'fa_rate_here': half['fa_rate'].mean(), 'hit_rate_here': half['hit_rate'].mean()}
        if mode == 'regenerate':
            things_hit_rates, things_false_alarm_rates = a3b_common.things_rates_from_count_table(list(half['image_name']))
            row['fa_rate_things'] = float(np.mean(things_false_alarm_rates))
            row['hit_rate_things'] = float(np.mean(things_hit_rates))
            row['things_source'] = 'computed from ' + a3b_common.COUNT_TABLE.relative_path
        else:
            row['fa_rate_things'] = provided_halves[name]['fa_things']
            row['hit_rate_things'] = provided_halves[name]['hit_things']
            row['things_source'] = 'from the saved results in data/provided/halves_check.txt'
        # the saved results, for the comparison
        row['paper_run_fa_here'] = provided_halves[name]['fa_here']
        row['paper_run_hit_here'] = provided_halves[name]['hit_here']
        row['paper_run_fa_things'] = provided_halves[name]['fa_things']
        row['paper_run_hit_things'] = provided_halves[name]['hit_things']
        rows.append(row)
    return pd.DataFrame(rows), mode


# =============================================================================================
# 3. Panel c: the original THINGS data
# =============================================================================================
def panel_c_numbers():
    """The numbers of Fig. S9c (read by row label from per_image_rates.txt) and where they come from."""
    mode = a3b_common.panel_c_mode(quiet=True)
    path = a3b_common.panel_c_report_path(mode)
    values = a3b_common.read_per_image_rates(path)
    values['mode'] = mode
    values['source'] = a3b_common.relative(path)
    paper_run = a3b_common.read_per_image_rates(a3b_common.PROVIDED_RATES_FILE)
    if values['images'] != paper_run['images']:
        raise SystemExit('%s reports %d images, the saved results %d' % (values['source'], values['images'],
                                                                      paper_run['images']))
    return values


# =============================================================================================
# 4. The comparison with the saved run
# =============================================================================================
def check_against_paper_run(items, level_table, correlations, panel_b_text):
    """Every number above compared with the saved run (data/paper_run/figS9_numbers.txt), each
    formatted as that run printed it. Stops when one differs."""
    paper_run = a3b_common.read_paper_run_numbers()
    problems = []
    for _, row in level_table.iterrows():
        computed = ['%.3f' % row['hit_rate'], '%.3f' % row['hit_rate_ci_low'], '%.3f' % row['hit_rate_ci_high'],
                    '%.3f' % row['fa_rate'], '%.3f' % row['fa_rate_ci_low'], '%.3f' % row['fa_rate_ci_high'],
                    '%.3f' % row['memorability'],
                    '%.3f' % row['dprime'], '%.3f' % row['dprime_ci_low'], '%.3f' % row['dprime_ci_high']]
        if computed != paper_run['levels'][int(row['level'])]:
            problems.append('level %d: %s, saved run %s' % (row['level'], computed, paper_run['levels'][int(row['level'])]))
    for name, (rho, p) in correlations.items():
        computed = ('%.4f' % rho, '%.3g' % p)
        if computed != paper_run['spearman'][name]:
            problems.append('Spearman %s: %s, saved run %s' % (name, computed, paper_run['spearman'][name]))
    trials_per_item = items['n_old'] + items['n_new']
    computed = ('%d' % trials_per_item.min(), '%d' % trials_per_item.max(), '%d' % items['n_old'].min(),
                '%d' % items['n_new'].min())
    if computed != paper_run['trials_line']:
        problems.append('trials per item: %s, saved run %s' % (computed, paper_run['trials_line']))
    if panel_b_text.replace('\n', ' / ') != paper_run['panel_b_text']:
        problems.append('panel b text: %r, saved run %r' % (panel_b_text, paper_run['panel_b_text']))
    if problems:
        print('DIFFERENT from the saved run (data/paper_run/figS9_numbers.txt):')
        for problem in problems:
            print('  ' + problem)
        raise SystemExit('make_figureS9.py: the numbers differ from the saved run in data/paper_run/')
    print('check: every number above equals the saved run (data/paper_run/figS9_numbers.txt) in every '
          'printed digit: the level table with its intervals (50 values), the three Spearman correlations and '
          'their p, the trial counts and the statistic on panel b')


def check_halves_against_paper_run(halves):
    """The halves comparison against the saved results (data/provided/halves_check.txt), to the
    printed digits. A difference is printed."""
    problems = []
    for _, row in halves.iterrows():
        pairs = [('%.3f' % row['fa_rate_here'], '%.3f' % row['paper_run_fa_here'], 'false alarms here'),
                 ('%.3f' % row['hit_rate_here'], '%.3f' % row['paper_run_hit_here'], 'hits here'),
                 ('%.4f' % row['fa_rate_things'], '%.4f' % row['paper_run_fa_things'], 'false alarms in THINGS'),
                 ('%.3f' % row['hit_rate_things'], '%.3f' % row['paper_run_hit_things'], 'hits in THINGS')]
        for computed, printed, what in pairs:
            if computed != printed:
                problems.append('%s, %s: %s, saved results %s' % (row['half'], what, computed, printed))
    if problems:
        print('DIFFERENT from the saved results (data/provided/halves_check.txt):')
        for problem in problems:
            print('  ' + problem)
        print('  (the saved THINGS values come from per-image counts whose old trials include the short-lag '
              'vigilance repeats)')
    else:
        print('check: the halves comparison equals the saved results (data/provided/halves_check.txt) in every '
              'printed digit')
    return problems


# =============================================================================================
# 5. Draw and save
# =============================================================================================
def draw_figure(level_table, rho_dprime, p_dprime, number_of_items, panel_c):
    """Draws Fig. S9. Returns the texts printed on panels b and c and the positions of the two bars of
    panel c."""
    ls.apply_style('data')
    fig, axes = ls.new_figure(n_panels=3, width='double', height_cm=6.2)
    axis_a, axis_b, axis_c = axes
    level_names = [str(level) for level in LEVELS]

    # ---- (a) hit and false alarm rate by number of similar items
    positions = ls.category_axis(axis_a, level_names)
    ls.line_series(axis_a, positions - OFFSET, level_table['fa_rate'].to_numpy(), color=ls.FA_RED,
                   low=level_table['fa_rate_ci_low'], high=level_table['fa_rate_ci_high'])
    ls.line_series(axis_a, positions + OFFSET, level_table['hit_rate'].to_numpy(), color=ls.HIT_BLUE,
                   low=level_table['hit_rate_ci_low'], high=level_table['hit_rate_ci_high'])
    # 'False Alarms' sits below the red line between 2 and 4 similar items, below the ends of the error bars.
    ls.direct_label(axis_a, (positions[2] + positions[3]) / 2 + 0.05, 0.1, 'False Alarms', ls.FA_RED,
                    ha='center', va='center', dx_points=0, dy_points=0)
    ls.direct_label(axis_a, positions[0] + OFFSET, level_table['hit_rate_ci_high'].iloc[0], 'Hits', ls.HIT_BLUE,
                    ha='left', va='bottom', dx_points=-3, dy_points=4)
    axis_a.set_ylim(0, 1)
    axis_a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axis_a.set_xlabel('Number of Similar Items')
    axis_a.set_ylabel('Hit or False Alarm Rate')
    ls.plain_ticks(axis_a)
    ls.panel_label(axis_a, 'a')

    # ---- (b) d′ by number of similar items
    positions = ls.category_axis(axis_b, level_names)
    ls.line_series(axis_b, positions, level_table['dprime'].to_numpy(), color=ls.MEMORY_DARK,
                   low=level_table['dprime_ci_low'], high=level_table['dprime_ci_high'])
    axis_b.set_ylim(0, 3)
    axis_b.set_yticks([0, 1, 2, 3])
    axis_b.set_xlabel('Number of Similar Items')
    axis_b.set_ylabel('d′')
    panel_b_text = ls.format_r_p(rho_dprime, p_dprime, n=number_of_items, unit='items', symbol='ρ')
    ls.stat_text(axis_b, panel_b_text, corner='upper right')
    ls.plain_ticks(axis_b)
    ls.panel_label(axis_b, 'b')

    # ---- (c) THINGS: correlation of the memorability score with each rate
    bar_positions = ls.category_axis(axis_c, ['Hit\nRate', 'False Alarm\nRate'])
    axis_c.bar(bar_positions, [panel_c['r_cr_hit'], panel_c['r_cr_fa']], width=0.5,
               color=[ls.HIT_BLUE, ls.FA_RED], edgecolor='none', zorder=2)
    ls.zero_line(axis_c)
    axis_c.set_ylim(-0.2, 1)
    axis_c.set_yticks([0, 0.5, 1])
    axis_c.set_ylabel('Correlation With THINGS\nMemorability Score')
    # The two printed numbers sit above the false alarm bar, where the plot is empty.
    panel_c_text = 'Mean false alarm\nrate = %.2f\nn = {:,}\nimages'.format(panel_c['images']) % panel_c['fa_mean']
    axis_c.text(bar_positions[1] - 0.06, 0.55, panel_c_text, ha='center', va='center', fontsize=ls.FONT_SIZE,
                color='black', linespacing=1.3)
    ls.plain_ticks(axis_c)
    ls.panel_label(axis_c, 'c')

    ls.save_figure(fig, FIGURE_NAME, PATHS['figures'])
    return panel_b_text, panel_c_text, bar_positions


# =============================================================================================
# 6. Run
# =============================================================================================
def main():
    print('make_figureS9.py: Fig. S9, hits and false alarms (module A3b)')
    items = a3b_common.critical_items()
    level_table = level_table_with_item_bootstrap(items)
    correlations = level_correlations(items)
    halves, halves_mode = halves_table(items)
    panel_c = panel_c_numbers()
    rho_dprime, p_dprime = correlations['d-prime']
    panel_b_text, panel_c_text, bar_positions = draw_figure(level_table, rho_dprime, p_dprime, len(items), panel_c)

    # ---- every number the caption and the text quote
    trials_per_item = items['n_old'] + items['n_new']
    print('\nNUMBERS')
    print('  (a, b) 50-participant sample; test trials per item %d to %d; at least %d old and %d new'
          % (trials_per_item.min(), trials_per_item.max(), items['n_old'].min(), items['n_new'].min()))
    print('  level  hit [95% CI]             false alarm [95% CI]       memorability   d′ [95% CI]          '
          'mean published score')
    for _, row in level_table.iterrows():
        print('  %5d  %.3f [%.3f, %.3f]   %.3f [%.3f, %.3f]   %.3f          %.3f [%.3f, %.3f]   %.4f'
              % (row['level'], row['hit_rate'], row['hit_rate_ci_low'], row['hit_rate_ci_high'], row['fa_rate'],
                 row['fa_rate_ci_low'], row['fa_rate_ci_high'], row['memorability'], row['dprime'],
                 row['dprime_ci_low'], row['dprime_ci_high'], row['mean_published_score']))
    print('  Spearman over 40 items, number of similar items vs: false alarms %.4f (p = %.3g), hits %.4f (p = %.3g), '
          'd′ %.4f (p = %.3g)'
          % (correlations['false alarms'] + correlations['hits'] + correlations['d-prime']))
    print('  (b) printed on panel: %s' % panel_b_text.replace('\n', ' / '))
    print('  (c) THINGS trial-level rates, %d images (%s): r(memorability, hit) = %+.4f, r(memorability, FA) = %+.4f; '
          'hit mean %.4f SD %.4f; FA mean %.4f SD %.4f'
          % (panel_c['images'], panel_c['source'], panel_c['r_cr_hit'], panel_c['r_cr_fa'], panel_c['hit_mean'],
             panel_c['hit_sd'], panel_c['fa_mean'], panel_c['fa_sd']))
    print('  (c) printed on panel: %s' % panel_c_text.replace('\n', ' '))
    for _, row in halves.iterrows():
        print('  %s (published score %.4f to %.4f): false alarms %.4f in THINGS -> %.3f here; hits %.3f in THINGS '
              '-> %.3f here  [THINGS side %s]'
              % (row['half'], row['published_score_min'], row['published_score_max'], row['fa_rate_things'],
                 row['fa_rate_here'], row['hit_rate_things'], row['hit_rate_here'], row['things_source']))
    print('')
    check_against_paper_run(items, level_table, correlations, panel_b_text)
    check_halves_against_paper_run(halves)

    # ---- the tables
    level_table.to_csv(os.path.join(PATHS['outputs'], 'level_table.csv'), index=False)
    correlation_rows = []
    for printed_name, (rho, p) in correlations.items():
        correlation_rows.append({'x': 'number of similar items', 'y': printed_name.replace('d-prime', 'd′'),
                                 'spearman_rho': rho, 'p_two_sided': p, 'n_items': len(items)})
    pd.DataFrame(correlation_rows).to_csv(os.path.join(PATHS['outputs'], 'level_correlations.csv'), index=False)
    halves.drop(columns=['paper_run_fa_here', 'paper_run_hit_here', 'paper_run_fa_things',
                         'paper_run_hit_things']).to_csv(os.path.join(PATHS['outputs'], 'halves.csv'), index=False)

    # ---- the plotted numbers
    positions = np.arange(1, len(LEVELS) + 1)
    panel_a = pd.DataFrame({'level': level_table['level'], 'n_items': level_table['n_items'],
                            'x_false_alarms': positions - OFFSET, 'fa_rate': level_table['fa_rate'],
                            'fa_ci_low': level_table['fa_rate_ci_low'], 'fa_ci_high': level_table['fa_rate_ci_high'],
                            'x_hits': positions + OFFSET, 'hit_rate': level_table['hit_rate'],
                            'hit_ci_low': level_table['hit_rate_ci_low'], 'hit_ci_high': level_table['hit_rate_ci_high']})
    panel_a.to_csv(os.path.join(PATHS['figure_data'], 'FigS9a.csv'), index=False)
    panel_b = pd.DataFrame({'level': level_table['level'], 'n_items': level_table['n_items'], 'x': positions,
                            'dprime': level_table['dprime'], 'ci_low': level_table['dprime_ci_low'],
                            'ci_high': level_table['dprime_ci_high'], 'spearman_rho_40_items': rho_dprime,
                            'p_two_sided': p_dprime, 'printed_on_panel': panel_b_text.replace('\n', ' / ')})
    panel_b.to_csv(os.path.join(PATHS['figure_data'], 'FigS9b.csv'), index=False)
    panel_c_table = pd.DataFrame({'bar': ['hit rate', 'false alarm rate'], 'x': bar_positions,
                                  'pearson_r_with_memorability_score': [panel_c['r_cr_hit'], panel_c['r_cr_fa']],
                                  'n_images': panel_c['images'], 'mean_false_alarm_rate': panel_c['fa_mean'],
                                  'mode': panel_c['mode'], 'source': panel_c['source']})
    panel_c_table.to_csv(os.path.join(PATHS['figure_data'], 'FigS9c.csv'), index=False)
    print('wrote outputs/level_table.csv, level_correlations.csv, halves.csv and figure_data/FigS9a.csv, '
          'FigS9b.csv, FigS9c.csv (panel c: %s; THINGS side of the halves comparison: %s)'
          % (panel_c['mode'], halves_mode))


if __name__ == '__main__':
    main()
