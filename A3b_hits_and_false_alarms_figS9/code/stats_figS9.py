"""
stats_figS9.py: the statistics of SI Appendix, Fig. S9, tests T33 and T34.

WHAT IT COMPUTES (rows of outputs/tests.csv; the convention is in statistics/README.md)
  T33  Fig. S9a, b (SI Appendix, Section 3). Over the 40 critical items of the memorability inversion
       experiment (50 participants), the Spearman rank correlation between the designed number of
       similar items on the study list and each item's false alarm rate, hit rate and d′ (log-linear
       corrected rates). The number of similar items is 0, 1, 2, 4 or 8, each value shared by 8 items,
       so the predictor is tied (tied values get their average rank). rho and p come from scipy's
       spearmanr (two-sided p from its t approximation with 38 df). The 95% interval is a percentile
       bootstrap over the 40 items (10,000 resamples, each item drawn with replacement keeping its
       level).
       Per level (0, 1, 2, 4, 8 similar items): the mean hit rate, false alarm rate and d′ of the 8
       items with the error bar of Fig. S9a and b (10,000 resamples of the 8 items, seed 0, computed
       by make_figureS9.py).
  T34  Fig. S9c (SI Appendix, Section 1). The Pearson correlation, across the 26,107 THINGS
       photographs, of the memorability score (hit rate minus false alarm rate, from the Kramer et al.
       trial records) with the hit rate and with the false alarm rate: r, n, df, the two-sided p and
       the Fisher z 95% interval. Without the trial records (input M1 of EXTERNAL_INPUTS.md) these are
       read from the saved report data/provided/per_image_rates.txt (r to 4 decimals, p as printed
       there); when the trial records are here they are read from the report that
       code/per_image_rates.py regenerated.
  No correction for multiple comparisons is applied: every test is reported on its own.

INPUTS (relative to the repository root)
  03_inversion_experiment_fig3/data/   through code/a3b_common.py (the 50 kept participants, 40 items)
  code/make_figureS9.py                the per-level table with its error bars and the correlations
  A3b_hits_and_false_alarms_figS9/data/provided/per_image_rates.txt (or outputs/regenerated/)

OUTPUTS: outputs/tests.csv (run_module.py keeps the printout as outputs/logs/stats_figS9.log)

SEED: 20261013 (numpy default_rng), one generator, drawn in this order:
  1. 10,000 draws of 50 participant indices (integers(0, 50, 50) each)
  2. the 10,000 item resamples (integers(0, 40, 40) each) behind the T33 intervals
  The error bars per level come from make_figureS9.py, which uses its own generator (seed 0).

RUN (from the repository root; about 15 seconds):
  python A3b_hits_and_false_alarms_figS9/code/stats_figS9.py
  (--resamples N sets a smaller number for a quick check of the code; the default is 10,000)
"""
import argparse
import os
import sys
import time

sys.dont_write_bytecode = True
for thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(thread_setting, '2')     # at most two threads per numerical library (same results)

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a3b_common  # noqa: E402  (also puts shared/python on the path)
import make_figureS9  # noqa: E402
import bootstrap  # noqa: E402
import tests_table  # noqa: E402

PATHS = a3b_common.PATHS
LEVELS = a3b_common.LEVELS
SCRIPT_NAME = 'code/stats_figS9.py'
FIGURE_SCRIPT_NAME = 'code/make_figureS9.py'
SEED = bootstrap.RELEASE_SEED                       # 20261013
T33_LEDGER_IDS = 'L7-023;L7-024;L7-030;L7-031;L7-035;L7-036;L7-038;L7-039'
T34_LEDGER_IDS = 'L7-027;L7-028;L7-029;L7-032;L7-040;L7-041;L7-121;L7-122;L7-123;L7-125;L7-126;L4-017'
T33_LOCATION = ('SI Appendix, Section 3, Similar Items Change Memory Mainly Through False Alarms; SI Appendix, '
                'Fig. S9 caption (Fig. S9a, b)')
T34_LOCATION = 'SI Appendix, Section 1, Measuring Memory (Fig. S9c)'

# The three measures of Fig. S9a, b: the name used in the tables and the column of the item table
CORRELATIONS = [
    ('false alarm rate', 'fa_rate'),
    ('hit rate', 'hit_rate'),
    ('d′', 'dprime'),
]
LEVEL_MEASURES = [('hit rate', 'hit_rate'), ('false alarm rate', 'fa_rate'), ('d′', 'dprime')]


def report(text=''):
    """Print a line at once (run_module.py keeps the printout as outputs/logs/stats_figS9.log)."""
    print(text, flush=True)


def spearman_rho(first_values, second_values):
    """Spearman rank correlation (tied values get their average rank); NaN when a value is missing or
    when one variable does not vary."""
    first_values = np.asarray(first_values, dtype=float)
    second_values = np.asarray(second_values, dtype=float)
    if not (np.all(np.isfinite(first_values)) and np.all(np.isfinite(second_values))):
        return float('nan')
    if np.all(first_values == first_values[0]) or np.all(second_values == second_values[0]):
        return float('nan')
    return float(spearmanr(first_values, second_values).statistic)


def main():
    parser = argparse.ArgumentParser(description='Statistics of Fig. S9 (T33, T34)')
    parser.add_argument('--resamples', type=int, default=10000, help='bootstrap resamples (default 10,000)')
    arguments = parser.parse_args()
    number_of_resamples = arguments.resamples
    started = time.time()
    report('stats_figS9.py: statistics of Fig. S9 (T33, T34; module A3b)')
    report('seed %d; %d bootstrap resamples' % (SEED, number_of_resamples))
    if number_of_resamples != 10000:
        report('a check of the code only: the reported intervals use 10,000 resamples')

    # ---------------------------------------------------------------- 1. the data and the observed values
    items = a3b_common.critical_items()
    number_of_items = len(items)
    number_of_participants = a3b_common.number_of_kept_participants()
    levels_of_items = items['n_similar_items'].to_numpy(dtype=float)
    observed = {'hit_rate': items['hit_rate'].to_numpy(dtype=float), 'fa_rate': items['fa_rate'].to_numpy(dtype=float),
                'dprime': items['dprime'].to_numpy(dtype=float)}
    report('%d critical items, %d kept participants' % (number_of_items, number_of_participants))
    paper_correlations = make_figureS9.level_correlations(items)
    observed_rho = {}
    observed_p = {}
    for name, column in CORRELATIONS:
        result = spearmanr(levels_of_items, observed[column])
        observed_rho[column] = float(result.statistic)
        observed_p[column] = float(result.pvalue)
    # the same values as make_figureS9.py
    for printed_name, column in [('false alarms', 'fa_rate'), ('hits', 'hit_rate'), ('d-prime', 'dprime')]:
        if paper_correlations[printed_name] != (observed_rho[column], observed_p[column]):
            raise SystemExit('the correlation with %s differs from make_figureS9.py' % printed_name)
    for name, column in CORRELATIONS:
        report('Spearman rho over %d items, number of similar items vs %s: %.4f (p = %.3g, t approximation, %d df)'
               % (number_of_items, name, observed_rho[column], observed_p[column], number_of_items - 2))
    level_table = make_figureS9.level_table_with_item_bootstrap(items)

    generator = np.random.default_rng(SEED)

    # ---------------------------------------------------------------- 2. participant indices (draws 1)
    # One draw of 50 participant indices per resample comes first from this generator (not used further);
    # the item resamples below follow them.
    for _resample in range(number_of_resamples):
        generator.integers(0, number_of_participants, number_of_participants)

    # ---------------------------------------------------------------- 3. item bootstrap (draws 2)
    resampled_rho_items = {}
    for _name, column in CORRELATIONS:
        resampled_rho_items[column] = np.empty(number_of_resamples)
    for resample in range(number_of_resamples):
        drawn_items = generator.integers(0, number_of_items, number_of_items)
        for _name, column in CORRELATIONS:
            resampled_rho_items[column][resample] = spearman_rho(levels_of_items[drawn_items],
                                                                 observed[column][drawn_items])

    # ---------------------------------------------------------------- 4. tests.csv, T33
    tests = tests_table.TestsTable(script=SCRIPT_NAME)
    levels_text = ', '.join(str(level) for level in LEVELS[:-1]) + ' or ' + str(LEVELS[-1])
    items_per_level = number_of_items // len(LEVELS)
    unit = 'items (%d participants; the predictor is tied: %d levels x %d items)' % (number_of_participants, len(LEVELS),
                                                                                   items_per_level)
    for name, column in CORRELATIONS:
        item_low, item_high = bootstrap.percentile_interval(resampled_rho_items[column])
        report('  %s: rho %.4f, item bootstrap [%.4f, %.4f]' % (name, observed_rho[column], item_low, item_high))
        test_text = ('Fig. S9: Spearman rank correlation across the %d critical items of the memorability inversion '
                     'experiment between the number of similar items on the study list (%s; %d items per value, so '
                     'the predictor is tied) and the item\'s %s over the %d kept participants; interval: the %d items '
                     'resampled, each keeping its level and its %s'
                     % (number_of_items, levels_text, items_per_level, name, number_of_participants, number_of_items,
                        name))
        tests.add(test_id='T33', ledger_ids=T33_LEDGER_IDS, location=T33_LOCATION, test=test_text,
                  statistic_name='Spearman rho', statistic=observed_rho[column], n=number_of_items, unit=unit,
                  df=number_of_items - 2, p=observed_p[column], p_type='parametric two-sided',
                  effect_size_name='rho', effect_size=observed_rho[column], ci_low=item_low,
                  ci_high=item_high, ci_method='percentile bootstrap', resampling_unit='items',
                  n_resamples=number_of_resamples, seed=SEED)

    # per-level means with the figure's error bars (items, seed 0)
    for name, column in LEVEL_MEASURES:
        for level in LEVELS:
            table_row = level_table[level_table['level'] == level].iloc[0]
            # '1 similar item', '2 similar items', ...
            if level == 1:
                similar_items_text = '1 similar item'
            else:
                similar_items_text = '%d similar items' % level
            tests.add(test_id='T33', ledger_ids=T33_LEDGER_IDS, location=T33_LOCATION,
                      test='Fig. S9%s as plotted: mean %s of the %d critical items studied with %s; '
                           'the figure\'s error bar (%s resamples of the %d items, numpy '
                           'default_rng(%d))' % ('b' if column == 'dprime' else 'a', name, items_per_level,
                                                 similar_items_text, format(make_figureS9.NUMBER_OF_RESAMPLES, ','),
                                                 items_per_level, make_figureS9.RANDOM_SEED),
                      statistic_name='mean %s' % name, statistic=float(table_row[column]), n=items_per_level,
                      unit='items at this level (%d participants)' % number_of_participants, p_type='none',
                      ci_low=float(table_row[column + '_ci_low']), ci_high=float(table_row[column + '_ci_high']),
                      ci_method='percentile bootstrap', resampling_unit='items (as plotted)',
                      n_resamples=make_figureS9.NUMBER_OF_RESAMPLES, seed=make_figureS9.RANDOM_SEED,
                      script=FIGURE_SCRIPT_NAME)

    # ---------------------------------------------------------------- 5. tests.csv, T34
    panel_c = make_figureS9.panel_c_numbers()
    number_of_images = panel_c['images']
    if panel_c['mode'] == 'provided':
        source_text = ('read from the saved report data/provided/per_image_rates.txt (r to 4 decimals; regenerating '
                       'it needs the Kramer et al. trial file, input M1 of EXTERNAL_INPUTS.md)')
    else:
        source_text = 'regenerated from the Kramer et al. trial records (%s)' % panel_c['source']
    for name, r_key, p_key in [('hit rate', 'r_cr_hit', 'p_cr_hit'), ('false alarm rate', 'r_cr_fa', 'p_cr_fa')]:
        r_value = panel_c[r_key]
        fisher_low, fisher_high = bootstrap.fisher_z_interval(r_value, number_of_images)
        p_value = panel_c[p_key]
        p_note = ('; p is below double precision (printed as < 1e-300 in the report, written as 0)'
                  if p_value == 0 else '')
        report('  Fig. S9c: r(memorability score, %s) = %+.4f, n = %d images, p = %s, Fisher-z 95%% CI [%.4f, %.4f]'
               % (name, r_value, number_of_images, panel_c[p_key + '_text'], fisher_low, fisher_high))
        tests.add(test_id='T34', ledger_ids=T34_LEDGER_IDS, location=T34_LOCATION,
                  test=('Fig. S9c: Pearson correlation across the %s THINGS photographs (at least %d old and %d new '
                        'trials each) between the memorability score (hit rate minus false alarm rate, from the Kramer '
                        'et al. 2023 trial records; old = long-lag target repeats, new = first showings) and the %s; '
                        '%s%s') % (format(number_of_images, ','), panel_c['minimum_observations'],
                                   panel_c['minimum_observations'], name, source_text, p_note),
                  statistic_name='Pearson r', statistic=r_value, n=number_of_images,
                  unit='images (THINGS photographs)', df=number_of_images - 2, p=p_value,
                  p_type='parametric two-sided', effect_size_name='r', effect_size=r_value, ci_low=fisher_low,
                  ci_high=fisher_high, ci_method='Fisher z', resampling_unit='')

    # ---------------------------------------------------------------- 6. write
    tests.write(PATHS['outputs'])
    report('No correction for multiple comparisons is applied (every test is reported on its own).')
    report('finished in %.1f minutes' % ((time.time() - started) / 60))


if __name__ == '__main__':
    main()
