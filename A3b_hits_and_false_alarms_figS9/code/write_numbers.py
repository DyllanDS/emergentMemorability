"""
write_numbers.py: outputs/numbers.csv of module A3b, one row per ledger item of the module in
shared/ledger/paper_numbers.csv (24 items: SI Appendix, Sections 1 and 3, Fig. S9, and three sentences
outside the SI Appendix that restate them).

WHAT IT DOES
  Reads what the module's other scripts wrote and compares each value with the number printed in the paper
  (shared/ledger/paper_numbers.csv, through shared/python/numbers_table.py):
    outputs/level_table.csv          per-level means and the item-bootstrap intervals of the error bars
    outputs/level_correlations.csv   the Spearman correlations over the 40 critical items
    outputs/halves.csv               the halves comparison
    outputs/tests.csv                the item-bootstrap intervals of the three Spearman correlations (T33)
    the Fig. S9c report              data/provided/per_image_rates.txt (saved results) or
                                     outputs/regenerated/per_image_rates.txt when the Kramer et al.
                                     trial records are here (input M1 of EXTERNAL_INPUTS.md)
  The numbers of items, blocks and participants and the list of levels come from the 40-item table
  (code/a3b_common.py); every other value is read from the files above, and no value is typed in. Every
  row is judged by the rule of its ledger item; values read from the saved results in data/provided/
  carry the mode "provided" ("restatement" for the three sentences outside the SI Appendix: L4-017, L7-125
  and L7-126).

INPUTS   the files above (run code/make_figureS9.py and code/stats_figS9.py first; code/run_module.py runs
         everything in order)
OUTPUTS  outputs/numbers.csv (ledger_id, location, quote, paper_value, computed_value, match, match_type,
         script, mode, note)
SEEDS    none.   RUNTIME: seconds.
RUN (from the repository root):  python A3b_hits_and_false_alarms_figS9/code/write_numbers.py
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3b_common  # noqa: E402  (also puts shared/python on the path)
import make_figureS9  # noqa: E402  (its settings only: resamples, seed, percentiles)

import pandas as pd  # noqa: E402

import bootstrap  # noqa: E402
import numbers_table  # noqa: E402

PATHS = a3b_common.PATHS
FIGURE_SCRIPT = 'code/make_figureS9.py'
RATES_SCRIPT = 'code/per_image_rates.py'
STATS_SCRIPT = 'code/stats_figS9.py'


def read_output_table(file_name):
    path = os.path.join(PATHS['outputs'], file_name)
    if not os.path.exists(path):
        raise SystemExit('outputs/%s is missing: run python A3b_hits_and_false_alarms_figS9/code/make_figureS9.py '
                         'first' % file_name)
    return pd.read_csv(path, float_precision='round_trip')


def read_figure_data(file_name):
    path = os.path.join(PATHS['figure_data'], file_name)
    if not os.path.exists(path):
        raise SystemExit('outputs/figure_data/%s is missing: run python A3b_hits_and_false_alarms_figS9/code/'
                         'make_figureS9.py first' % file_name)
    return pd.read_csv(path)


def read_item_intervals(correlations):
    """{measure: (ci_low, ci_high)}: the item-bootstrap 95% interval of each Spearman correlation, from the
    T33 rows of outputs/tests.csv (code/stats_figS9.py), matched by rho."""
    path = os.path.join(PATHS['outputs'], 'tests.csv')
    if not os.path.exists(path):
        raise SystemExit('outputs/tests.csv is missing: run python A3b_hits_and_false_alarms_figS9/code/stats_figS9.py '
                         'first')
    tests = pd.read_csv(path, float_precision='round_trip')
    rho_rows = tests[(tests['test_id'] == 'T33') & (tests['statistic_name'] == 'Spearman rho')]
    intervals = {}
    for measure, (rho, _p) in correlations.items():
        matching = rho_rows[(rho_rows['statistic'] - rho).abs() < 1e-5]
        if len(matching) != 1:
            raise SystemExit('outputs/tests.csv has %d T33 rows for the correlation with %s' % (len(matching), measure))
        intervals[measure] = (float(matching['ci_low'].iloc[0]), float(matching['ci_high'].iloc[0]))
    return intervals


def interval_text(row, measure, decimals=3):
    """'0.724 [0.676, 0.770]' for one measure of one level."""
    return '%.*f [%.*f, %.*f]' % (decimals, row[measure], decimals, row[measure + '_ci_low'], decimals,
                                  row[measure + '_ci_high'])


def main():
    print('write_numbers.py: outputs/numbers.csv of module A3b')
    level_table = read_output_table('level_table.csv')
    correlations = read_output_table('level_correlations.csv')
    halves = read_output_table('halves.csv')
    items = a3b_common.critical_items(check_against_module_03=False)

    panel_c_mode = a3b_common.panel_c_mode(quiet=True)
    panel_c_path = a3b_common.panel_c_report_path(panel_c_mode)
    panel_c = a3b_common.read_per_image_rates(panel_c_path)
    panel_c_source = a3b_common.relative(panel_c_path)
    halves_from_count_table = halves['things_source'].str.startswith('computed').all()

    level_rows = {}
    for _, row in level_table.iterrows():
        level_rows[int(row['level'])] = row
    first_level = a3b_common.LEVELS[0]
    last_level = a3b_common.LEVELS[-1]
    number_of_levels = len(a3b_common.LEVELS)
    number_of_blocks = items['block'].nunique()
    number_of_participants = a3b_common.number_of_kept_participants()

    correlation_of = {}
    for _, row in correlations.iterrows():
        if row['x'] == 'number of similar items':
            correlation_of[row['y']] = (float(row['spearman_rho']), float(row['p_two_sided']))
    interval_of = read_item_intervals(correlation_of)

    table = numbers_table.NumbersTable(script=FIGURE_SCRIPT)

    # ---------------------------------------------------------------- panels a and b (rerun)
    false_alarms_by_level = ', '.join('%.4f' % level_rows[level]['fa_rate'] for level in a3b_common.LEVELS)
    hits_by_level = ', '.join('%.4f' % level_rows[level]['hit_rate'] for level in a3b_common.LEVELS)
    dprime_by_level = ', '.join('%.3f' % level_rows[level]['dprime'] for level in a3b_common.LEVELS)
    table.add('L7-023', [first_level, last_level, level_rows[first_level]['fa_rate'], level_rows[last_level]['fa_rate']],
              mode='rerun',
              note='numbers of similar items at the two ends of the design (%d and %d) and the mean false alarm rate '
                   'of the 8 critical items at each end (all levels: %s at %s similar items; mean d′ %s, falling '
                   'from %d to %d similar items), over the %d participants of the memorability inversion experiment'
                   % (first_level, last_level, false_alarms_by_level, ', '.join(str(level) for level in a3b_common.LEVELS),
                      dprime_by_level, first_level, last_level, number_of_participants))
    table.add('L7-024', [level_rows[first_level]['hit_rate'], level_rows[last_level]['hit_rate']], mode='rerun',
              note='mean hit rate of the 8 critical items at the two ends of the design (all levels: %s at %s '
                   'similar items)' % (hits_by_level, ', '.join(str(level) for level in a3b_common.LEVELS)))
    panel_a_parts = []
    for level in a3b_common.LEVELS:
        panel_a_parts.append('%d similar %s: hits %s, false alarms %s'
                             % (level, 'item' if level == 1 else 'items', interval_text(level_rows[level], 'hit_rate'),
                                interval_text(level_rows[level], 'fa_rate')))
    panel_a_text = '; '.join(panel_a_parts)
    table.add('L7-030', panel_a_text, mode='rerun',
              note='Fig. S9a as plotted (outputs/figure_data/FigS9a.csv): mean hit and false alarm rates of the 8 '
                   'critical items at each number of similar items, with 95%% percentile bootstrap intervals over the '
                   '8 items (%s resamples, numpy default_rng(%d))'
                   % (format(make_figureS9.NUMBER_OF_RESAMPLES, ','), make_figureS9.RANDOM_SEED))
    rho_dprime, p_dprime = correlation_of['d′']
    panel_b_text = '; '.join('%d: %s' % (level, interval_text(level_rows[level], 'dprime'))
                             for level in a3b_common.LEVELS)
    printed_on_panel_b = read_figure_data('FigS9b.csv')['printed_on_panel'].iloc[0]
    table.add('L7-031', 'd′ by number of similar items %s; Spearman rho %.4f (p = %.3g), printed on the panel as '
                        '"%s"' % (panel_b_text, rho_dprime, p_dprime, printed_on_panel_b), mode='rerun',
              note='Fig. S9b as plotted (outputs/figure_data/FigS9b.csv): mean d′ of the 8 critical items at each '
                   'number of similar items with 95% percentile bootstrap intervals over the items, and the Spearman '
                   'correlation printed on the panel; per-item d′ from the log-linear corrected rates (hits + 0.5) / '
                   '(old trials + 1) and (false alarms + 0.5) / (new trials + 1)')
    levels_text = ', '.join(str(level) for level in sorted(items['n_similar_items'].unique()))
    table.add('L7-033', '%d critical items; %s similar items' % (len(items), levels_text), mode='rerun',
              note='number of critical items in the memorability inversion experiment and the numbers of similar '
                   'items they were studied with (03_inversion_experiment_fig3/data/inversion_design.csv)')
    items_per_level = [int(level_rows[level]['n_items']) for level in a3b_common.LEVELS]
    if min(items_per_level) == max(items_per_level):
        items_per_level_value = items_per_level[0]
    else:
        items_per_level_value = items_per_level
    table.add('L7-034', items_per_level_value, mode='rerun',
              note='number of critical items at each of the %d numbers of similar items (one per block in each of '
                   'the %d blocks)' % (number_of_levels, number_of_blocks))
    rho_false_alarms, p_false_alarms = correlation_of['false alarms']
    low_false_alarms, high_false_alarms = interval_of['false alarms']
    table.add('L7-035', [level_rows[first_level]['fa_rate'], level_rows[last_level]['fa_rate'], rho_false_alarms,
                         low_false_alarms, high_false_alarms, p_false_alarms], mode='rerun',
              script='%s; %s' % (FIGURE_SCRIPT, STATS_SCRIPT),
              note='mean false alarm rate at 0 and at 8 similar items, and the Spearman correlation across the %d '
                   'critical items between the number of similar items (tied: %d levels x %d items) and the false '
                   'alarm rate, with its 95%% percentile bootstrap interval over the items (outputs/tests.csv, T33) '
                   'and two-sided p (t approximation, %d df)'
                   % (len(items), number_of_levels, items_per_level[0], len(items) - 2))
    rho_hits, _p_hits = correlation_of['hits']
    low_hits, high_hits = interval_of['hits']
    table.add('L7-036', [level_rows[first_level]['hit_rate'], level_rows[last_level]['hit_rate'], rho_hits, low_hits,
                         high_hits], mode='rerun',
              script='%s; %s' % (FIGURE_SCRIPT, STATS_SCRIPT),
              note='mean hit rate at 0 and at 8 similar items, and the Spearman correlation across the %d critical '
                   'items between the number of similar items and the hit rate, with its 95%% percentile bootstrap '
                   'interval over the items (outputs/tests.csv, T33)' % len(items))
    low_dprime, high_dprime = interval_of['d′']
    table.add('L7-038', [rho_dprime, low_dprime, high_dprime, p_dprime], mode='rerun',
              script='%s; %s' % (FIGURE_SCRIPT, STATS_SCRIPT),
              note='Spearman correlation across the %d critical items between the number of similar items and d′ '
                   '(log-linear corrected rates), with its 95%% percentile bootstrap interval over the items '
                   '(outputs/tests.csv, T33) and two-sided p (t approximation)' % len(items))
    trials_per_item = items['n_old'] + items['n_new']
    table.add('L7-124', 'every critical item has old and new test trials: %d to %d test trials per item, at least %d '
                        'old and %d new' % (trials_per_item.min(), trials_per_item.max(), items['n_old'].min(),
                                            items['n_new'].min()), mode='rerun',
              note='each of the %d kept participants was tested once on each critical item, as old (studied) or as '
                   'new (an unstudied lure)' % number_of_participants)
    percentiles = [float(value) for value in make_figureS9.INTERVAL_PERCENTILES]
    interval_level = percentiles[1] - percentiles[0]           # the share of resamples between the two percentiles
    table.add('L7-039', interval_level, mode='rerun',
              note='coverage of the Fig. S9a and b error bars: %gth to %gth percentiles of %s resamples of the 8 '
                   'critical items at each number of similar items (numpy default_rng(%d))'
                   % (percentiles[0], percentiles[1], format(make_figureS9.NUMBER_OF_RESAMPLES, ','),
                      make_figureS9.RANDOM_SEED))

    # ---------------------------------------------------------------- the halves comparison
    scores_text = ', '.join('%.3f' % level_rows[level]['mean_published_score'] for level in a3b_common.LEVELS)
    halves_texts = []
    for _, row in halves.iterrows():
        halves_texts.append('%s: hits %.3f in THINGS -> %.3f here, false alarms %.4f -> %.3f'
                            % (row['half'], row['hit_rate_things'], row['hit_rate_here'], row['fa_rate_things'],
                               row['fa_rate_here']))
    halves_value = 'mean published score of the items at %s similar items: %s; %s' % (
        ', '.join(str(level) for level in a3b_common.LEVELS), scores_text, '; '.join(halves_texts))
    if halves_from_count_table:
        halves_mode = 'rerun'
        # the THINGS side compared with the saved results, to their printed digits
        paper_run_halves = a3b_common.read_halves_check()
        differences = []
        for _, row in halves.iterrows():
            printed = paper_run_halves[row['half']]
            if '%.4f' % row['fa_rate_things'] != printed['fa_things_text']:
                differences.append('%s false alarms %.4f (saved %s)' % (row['half'], row['fa_rate_things'],
                                                                       printed['fa_things_text']))
            if '%.3f' % row['hit_rate_things'] != printed['hit_things_text']:
                differences.append('%s hits %.3f (saved %s)' % (row['half'], row['hit_rate_things'],
                                                               printed['hit_things_text']))
        if differences:
            comparison = 'DIFFERS from the saved results: ' + '; '.join(differences)
        else:
            comparison = 'equal to the saved results (data/provided/halves_check.txt) in every printed digit'
        halves_note = ('mean published score (Kramer et al., 2023) of the critical items at each number of similar '
                       'items, and the mean hit and false alarm rates of the 20 more and the 20 less memorable '
                       'critical items in the original THINGS data and in this experiment; the THINGS side computed '
                       'from %s (M2), %s; this experiment\'s side and the mean published scores computed from the '
                       'data of module 03' % (a3b_common.COUNT_TABLE.relative_path, comparison))
    else:
        halves_mode = 'provided'
        halves_note = ('mean published score (Kramer et al., 2023) of the critical items at each number of similar '
                       'items, and the mean hit and false alarm rates of the 20 more and the 20 less memorable '
                       'critical items in the original THINGS data and in this experiment; the THINGS side is read '
                       'from the saved results in data/provided/halves_check.txt (regenerating it needs input M2 of '
                       'EXTERNAL_INPUTS.md); this experiment\'s side and the mean published scores are computed from '
                       'the data of module 03')
    table.add('L7-025', halves_value, mode=halves_mode, note=halves_note)

    # ---------------------------------------------------------------- panel c (provided unless M1 is here)
    if panel_c_mode == 'provided':
        panel_c_mode_text = 'provided'
        source_note = ('from %s (saved results; regenerating them needs the Kramer et al. trial file, input M1 of '
                       'EXTERNAL_INPUTS.md)' % panel_c_source)
    else:
        panel_c_mode_text = 'rerun'
        source_note = 'regenerated from the Kramer et al. trial records (%s)' % panel_c_source
    things_values = [panel_c['fa_mean'], panel_c['images']]
    table.add('L7-027', things_values, mode=panel_c_mode_text, script=RATES_SCRIPT,
              note='mean false alarm rate (SD %.4f) across the THINGS photographs with at least %d old and %d new '
                   'trials in the Kramer et al. (2023) trial records, and the number of those photographs; %s'
                   % (panel_c['fa_sd'], panel_c['minimum_observations'], panel_c['minimum_observations'], source_note))
    low_hit, high_hit = bootstrap.fisher_z_interval(panel_c['r_cr_hit'], panel_c['images'])
    table.add('L7-028', [panel_c['r_cr_hit'], low_hit, high_hit], mode=panel_c_mode_text,
              script='%s; %s' % (RATES_SCRIPT, STATS_SCRIPT),
              note='Pearson r across the %d THINGS photographs between the memorability score (hit rate minus false '
                   'alarm rate, from the trial records) and the hit rate, with its Fisher z 95%% interval '
                   '(outputs/tests.csv, T34), p %s; %s'
                   % (panel_c['images'], panel_c['p_cr_hit_text'], source_note))
    low_fa, high_fa = bootstrap.fisher_z_interval(panel_c['r_cr_fa'], panel_c['images'])
    table.add('L7-029', [panel_c['r_cr_fa'], low_fa, high_fa], mode=panel_c_mode_text,
              script='%s; %s' % (RATES_SCRIPT, STATS_SCRIPT),
              note='Pearson r across the %d THINGS photographs between the memorability score (hit rate minus false '
                   'alarm rate, from the trial records) and the false alarm rate, with its Fisher z 95%% interval '
                   '(outputs/tests.csv, T34), p = %s; %s'
                   % (panel_c['images'], panel_c['p_cr_fa_text'], source_note))
    table.add('L7-032', 'Pearson r of the memorability score with the hit rate %+.4f and with the false alarm rate '
                        '%+.4f across %s photographs; printed on the panel: "Mean false alarm rate = %.2f, n = %s '
                        'images"' % (panel_c['r_cr_hit'], panel_c['r_cr_fa'], format(panel_c['images'], ','),
                                     panel_c['fa_mean'], format(panel_c['images'], ',')),
              mode=panel_c_mode_text, script=FIGURE_SCRIPT,
              note='Fig. S9c as plotted (outputs/figure_data/FigS9c.csv): one bar per rate, a dashed line at zero; %s'
                   % source_note)
    rates_text = ('memorability score against the hit rate r = %+.4f and against the false alarm rate r = %+.4f '
                  'across %s photographs; false alarm rate mean %.4f (SD %.4f), hit rate mean %.4f (SD %.4f)'
                  % (panel_c['r_cr_hit'], panel_c['r_cr_fa'], format(panel_c['images'], ','), panel_c['fa_mean'],
                     panel_c['fa_sd'], panel_c['hit_mean'], panel_c['hit_sd']))
    table.add('L7-121', rates_text, mode=panel_c_mode_text, script=RATES_SCRIPT,
              note='the memorability score of the original THINGS data follows the hit rate and barely the false '
                   'alarm rate (Fig. S9c); %s' % source_note)
    table.add('L7-122', rates_text, mode=panel_c_mode_text, script=RATES_SCRIPT,
              note='per-image false alarm rates of the original THINGS data near floor (mean and SD across the '
                   'photographs of Fig. S9c); %s' % source_note)
    table.add('L7-123', rates_text, mode=panel_c_mode_text, script=RATES_SCRIPT,
              note='Fig. S9c: the memorability score of the original THINGS data barely tracks the false alarm '
                   'rate; %s' % source_note)
    table.add('L7-125', rates_text, mode='restatement', script=RATES_SCRIPT,
              note='Fig. S9c as the main text cites it: the memorability score of the original THINGS data follows '
                   'the hit rate and barely the false alarm rate; %s' % source_note)
    table.add('L7-126', rates_text, mode='restatement', script=RATES_SCRIPT,
              note='SI Appendix, Section 1 as the Methods cite it: per-image false alarm rates of the original THINGS '
                   'continuous recognition data near floor (mean and SD across the photographs of Fig. S9c); %s'
                   % source_note)
    table.add('L7-040', panel_c['images'], mode=panel_c_mode_text,
              script=RATES_SCRIPT,
              note='number of THINGS photographs with at least %d old and %d new trials in the Kramer et al. (2023) '
                   'trial records; %s'
                   % (panel_c['minimum_observations'], panel_c['minimum_observations'], source_note))
    table.add('L7-041', 'per-image rates from the trial records: %s photographs with at least %d old and %d new trials '
                        '(median %d old, %d new); old = long-lag target repeats, new = first showings'
                        % (format(panel_c['images'], ','), panel_c['minimum_observations'],
                           panel_c['minimum_observations'], panel_c['median_old'], panel_c['median_new']),
              mode=panel_c_mode_text, script=RATES_SCRIPT,
              note=source_note)
    table.add('L4-017', 'continuous recognition in the original THINGS data runs at a strict criterion: mean false alarm '
                        'rate %.4f (SD %.4f) against a mean hit rate of %.4f over %s photographs (Fig. S9c)'
                        % (panel_c['fa_mean'], panel_c['fa_sd'], panel_c['hit_mean'], format(panel_c['images'], ',')),
              mode='restatement', script=RATES_SCRIPT,
              note='mean false alarm rate (with its SD) and mean hit rate across the THINGS photographs of Fig. S9c '
                   'in the Kramer et al. (2023) continuous recognition data, the rates behind the General Discussion '
                   'sentence on the biased criterion of continuous recognition; %s'
                   % source_note)

    table.rows.sort(key=lambda row: (row['ledger_id'][:2] != 'L4', row['ledger_id']))
    table.write(PATHS['outputs'])


if __name__ == '__main__':
    main()
