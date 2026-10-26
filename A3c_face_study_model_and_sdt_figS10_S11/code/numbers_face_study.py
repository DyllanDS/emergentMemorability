"""
numbers_face_study.py: outputs/numbers.csv, one row for every number and statement of the paper this
module reproduces (44 ledger items: SI Appendix, Section 3, Fig. 4c, and the main-text and Methods
sentences that refer to them).

WHAT IT DOES
Reads the outputs of the module's other scripts and writes, for each ledger item, the value the code
computed beside the paper's printed value (from shared/ledger/paper_numbers.csv). The match column is
judged by shared/python/numbers_table.py with the ledger's rule: counts and design values must be
equal, other numbers must round to the printed value at the printed precision. Statements without a
number (match 'documented') carry the computed values behind them. The note of each row says what the
computed value is and where it comes from.

INPUTS   outputs/ of this module: rem_final_spec.txt, rem_729_rates.csv (rem_final_spec.py,
         rem_729_rates.py); rating_counts_by_face_kind.csv, roc_operating_points.csv (roc_points.py);
         observed_contrasts.csv (observed_contrasts.py); registered_fit_summary.csv, rem_243_summary.csv,
         provided_checks.csv (check_provided.py); rem_like_data_model_side_check.txt (rem_like_data.py);
         figure_data/figS10a_model_settings.csv (make_figureS10.py)
         data/paper_run/rem_final_spec.txt; data/provided/point_fit.json;
         04_rare_feature_faces_fig4/data/ (through face_loader.py and face_study_common.py)
OUTPUTS  outputs/numbers.csv
SEEDS    none (reads the outputs)
RUNTIME  about 2 seconds
RUN      python A3c_face_study_model_and_sdt_figS10_S11/code/numbers_face_study.py
"""
import csv
import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_FOLDER)
sys.path.insert(0, os.path.join(CODE_FOLDER, '..', '..', 'shared', 'python'))
import release_paths         # noqa: E402
import face_loader         # noqa: E402
import numbers_table       # noqa: E402
import face_study_common     # noqa: E402
import rem_final_spec      # noqa: E402  (the grid; importing it runs no sweep)

PATHS = release_paths.module_paths(__file__)
OUTPUTS = PATHS['outputs']
table = numbers_table.NumbersTable(script='code/numbers_face_study.py')
MEMTOOLBOX_NOTE = 'saved fits in data/provided/ (refitting needs MemToolbox2, EXTERNAL_INPUTS.md, item M5)'


def read_rows(file_name, folder=OUTPUTS):
    with open(os.path.join(folder, file_name), newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def interval_text(low, high):
    return '[%+.3f, %+.3f]' % (low, high)


# ============================================================================ read everything once
rates_rows = read_rows('rem_729_rates.csv')
number_of_settings = len(rates_rows)
hit_differences = np.array([float(row['hit_difference_rare_minus_ordinary']) for row in rates_rows])
false_alarm_differences = np.array([float(row['false_alarm_difference_feature_lure_minus_new']) for row in rates_rows])
lure_set_from_evidence = np.array([float(row['lure_set_effect_d_a']) for row in rates_rows])
interaction_four_term = np.array([float(row['interaction_four_term_d_a']) for row in rates_rows])
hits_above = int(np.sum(hit_differences > 0))
false_alarms_above = int(np.sum(false_alarm_differences > 0))
lure_set_above = int(np.sum(lure_set_from_evidence > 0))
four_term_above = int(np.sum(interaction_four_term > 0))

with open(os.path.join(OUTPUTS, 'rem_final_spec.txt'), encoding='utf-8') as handle:
    rerun_report = handle.read().replace('\r\n', '\n')
with open(os.path.join(PATHS['paper_run'], 'rem_final_spec.txt'), encoding='utf-8') as handle:
    paper_run_report = handle.read().replace('\r\n', '\n')
report_equals_paper_run = 'yes' if rerun_report == paper_run_report else 'NO'
EQUALITY_NOTE = ('the registered sweep run again (rem_final_spec.py, seed = setting number); outputs/rem_final_spec.txt '
                 'equals the saved run data/paper_run/rem_final_spec.txt line for line: %s' % report_equals_paper_run)

failing_settings = [row for row in rates_rows if float(row['rate_rare_old']) <= float(row['rate_ord_old'])]
failing_hit_rates = []
for row in failing_settings:
    failing_hit_rates.append(float(row['rate_rare_old']))
    failing_hit_rates.append(float(row['rate_ord_old']))
failures_one_feature = sum(1 for row in failing_settings if row['nfeat'] == '1')
failures_both_above = sum(1 for row in failing_settings
                          if float(row['rate_rare_old']) > 0.89 and float(row['rate_ord_old']) > 0.89)
largest_shortfall = max(float(row['rate_ord_old']) - float(row['rate_rare_old']) for row in failing_settings)

rem_243 = {row['scoring_and_quantity']: row for row in read_rows('rem_243_summary.csv')}
fitted_lure = rem_243['fitted lure-set effect']
fitted_feature = rem_243['fitted feature effect']
with open(os.path.join(OUTPUTS, 'rem_like_data_model_side_check.txt'), encoding='utf-8') as handle:
    model_side_lines = handle.read().splitlines()
model_side_reproduced = face_study_common.find_row(model_side_lines, "the rerun reproduces the saved run's evidence:").strip()
largest_model_side_difference = face_study_common.find_row(
    model_side_lines, "largest difference from the saved run's direct scores (data/provided/rem_like_data.jsonl):").strip()
rating_share_gap = face_study_common.find_row(
    model_side_lines, 'simulated ratings under the shared criteria: largest gap between a simulated and an observed '
                      'rating share, over all settings and ratings:').strip()

fit = {row['quantity']: row for row in read_rows('registered_fit_summary.csv')}
provided_checks = read_rows('provided_checks.csv')
all_provided_checks_agree = all(row['agree'] == 'yes' for row in provided_checks)
CHECKS_NOTE = ('check_provided.py: %d of %d checks of the saved files agree'
               % (sum(1 for row in provided_checks if row['agree'] == 'yes'), len(provided_checks)))
with open(os.path.join(PATHS['provided'], 'point_fit.json'), encoding='utf-8') as handle:
    point_fit = json.load(handle)

observed = {row['contrast']: row for row in read_rows('observed_contrasts.csv')}
rating_counts = {row['face_kind']: row for row in read_rows('rating_counts_by_face_kind.csv')}
operating_points = {row['face_kind']: row for row in read_rows('roc_operating_points.csv')}

participants_table = face_loader.load_participants()
sessions_run = len(participants_table)
registered = int(participants_table['registered'].sum())
recomputed_table = face_loader.recompute_participants()     # stops if the shipped table differs from the rule
records = face_study_common.analysable(face_study_common.load())
features_of_pairs = {}
confidence_levels = set()
ratings_used = set()
for record in records:
    for trial in record['rec']:
        confidence_levels.add(trial['confidence'])
        ratings_used.add(face_study_common.rating(trial))
        if trial['condition'] == 'rare':
            features_of_pairs[trial['cell']] = trial['feature']
pairs_text = ', '.join('%s %s' % (cell, features_of_pairs[cell]) for cell in sorted(features_of_pairs))

# the reasons, counted as module 04 counts them: a session with both faults counts under both reasons
exclusion_reasons = {'too many presses': 0, 'too few repeats': 0, 'both': 0}
for _, row in recomputed_table.iterrows():
    if row['passes_rule'] == 1:
        continue
    too_many = row['oneback_false_presses'] > face_study_common.MAX_ONEBACK_FA
    too_few = row['oneback_hits'] < face_study_common.MIN_ONEBACK_HITS
    if too_many:
        exclusion_reasons['too many presses'] += 1
    if too_few:
        exclusion_reasons['too few repeats'] += 1
    if too_many and too_few:
        exclusion_reasons['both'] += 1

grid = rem_final_spec.GRID
standard_values_in_grid = (0.4 in grid['g'] and 20 in grid['w'] and 0.3 in grid['u'] and 0.7 in grid['c'])
GRID_TEXT = ('g {%s}, w {%s}, u {%s}, c {%s}, features carrying the rare value {%s}, criterion fa {%s}'
             % tuple(', '.join('%g' % value for value in grid[name]) for name in ('g', 'w', 'u', 'c', 'nfeat', 'fa')))

lure = fit['interaction']
feature = fit['equivalence']
rare_unrel = fit['rare_unrel']
rare_matched = fit['rare_matched']
ordinary = fit['ordinary']


def p_words(p_value, resamples):
    if p_value == 0:
        return 'no resample at or below zero (p < %g at %s resamples)' % (2.0 / resamples, '{:,}'.format(resamples))
    return 'two-sided bootstrap p = %.3f' % p_value


def with_participant_interval(row):
    """[estimate, low, high]: a d_a quantity of the registered fit with its 95% interval over participants."""
    return [float(row['estimate']), float(row['participants_ci_low']), float(row['participants_ci_high'])]


def registered_rate_p(row):
    """The registered two-sided bootstrap p of a rate difference of observed_contrasts.csv: twice the smaller
    share of its 10,000 resampled means at or below zero and at or above zero."""
    return 2 * min(int(row['resamples_at_or_below_zero']), int(row['resamples_at_or_above_zero'])) / 10000.0


# ============================================================================ main text and Methods
table.add('L2-028',
          'd_a %.3f (rare faces against same-feature lures) vs %.3f (ordinary faces against never-studied faces): '
          'feature effect %+.3f %s, %s; rare faces against never-studied faces %.3f %s; lure-set effect %+.3f %s, %s'
          % (float(rare_matched['estimate']), float(ordinary['estimate']), float(feature['estimate']),
             interval_text(float(feature['participants_ci_low']), float(feature['participants_ci_high'])),
             p_words(float(feature['participants_p_two_sided']), 10000), float(rare_unrel['estimate']),
             interval_text(float(rare_unrel['participants_ci_low']), float(rare_unrel['participants_ci_high'])),
             float(lure['estimate']), interval_text(float(lure['participants_ci_low']), float(lure['participants_ci_high'])),
             p_words(float(lure['participants_p_two_sided']), 10000)),
          'provided',
          'registered unequal-variance fit: d_a of rare faces against same-feature lures and of ordinary faces against '
          'never-studied faces, their difference (the feature effect), d_a of rare faces against never-studied faces '
          'and the lure-set effect, with 95%% intervals over participants (10,000 refits) and two-sided bootstrap p; '
          '%s; %s' % (MEMTOOLBOX_NOTE, CHECKS_NOTE))
table.add('L9-101',
          with_participant_interval(rare_unrel) + with_participant_interval(rare_matched)
          + with_participant_interval(ordinary), 'provided',
          'd_a of rare faces against never-studied faces, of rare faces against their same-feature lures and of '
          'ordinary faces against never-studied faces, from the registered fit, each with its 95%% interval over '
          'participants (10,000 refits); %s' % MEMTOOLBOX_NOTE)
table.add('L2-029',
          'REM through the face design at %d settings before data collection: hits rare > ordinary at %d, false alarms '
          'lure with the feature > new ordinary face at %d, lure-set effect read off the evidence > 0 at %d; '
          'after data collection, scored like the data: lure-set effect > 0 at %d of %d (saved fits)'
          % (number_of_settings, hits_above, false_alarms_above, lure_set_above, int(fitted_lure['above_zero']),
             int(fitted_lure['settings'])),
          'rerun_and_provided',
          'REM through the face design: settings at which each preregistered prediction holds (729-setting sweep '
          'before data collection; 243-setting run after data collection, scored as the data were, from the saved '
          'fits); %s' % EQUALITY_NOTE)
table.add('L5-044',
          'registered d_a analysis: d_a %.3f / %.3f / %.3f; lure-set effect %+.3f %s; feature effect %+.3f %s'
          % (float(rare_unrel['estimate']), float(rare_matched['estimate']), float(ordinary['estimate']),
             float(lure['estimate']), interval_text(float(lure['participants_ci_low']), float(lure['participants_ci_high'])),
             float(feature['estimate']),
             interval_text(float(feature['participants_ci_low']), float(feature['participants_ci_high']))),
          'provided', 'cross-reference to the registered d_a analysis of SI Appendix, Section 3 (rows L9-062 to '
                      'L9-081): the three d_a values and the lure-set and feature effects with their intervals over '
                      'participants; %s' % MEMTOOLBOX_NOTE)

# ============================================================================ SI Appendix, Section 3: the sweep before data collection
table.add('L9-040', number_of_settings, 'rerun',
          '3^6 settings: %s; 500 simulated participants each; seed = setting number' % GRID_TEXT)
table.add('L9-041',
          'the grid contains REM\'s standard values g 0.4, w 20, u 0.3, c 0.7 (Shiffrin and Steyvers 1997): %s; grid %s'
          % ('yes' if standard_values_in_grid else 'NO', GRID_TEXT), 'rerun', 'from code/rem_final_spec.py (GRID)')
table.add('L9-043', [hits_above, number_of_settings], 'rerun',
          'settings at which the hit rate of studied rare faces exceeds that of studied ordinary faces; %s' % EQUALITY_NOTE)
table.add('L9-044',
          '%d settings where the hit prediction fails; at every one both hit rates are between %.3f and %.3f; %d of %d '
          'with the rare value on 1 feature; %d of %d with both rates above 0.89; largest shortfall %.4f'
          % (len(failing_settings), min(failing_hit_rates), max(failing_hit_rates), failures_one_feature,
             len(failing_settings), failures_both_above, len(failing_settings), largest_shortfall),
          'rerun', 'outputs/rem_729_rates.csv, outputs/rem_729_rates_check.txt')
table.add('L9-045', false_alarms_above, 'rerun',
          'settings at which a new face carrying a studied face\'s rare feature is called old more often than a new '
          'ordinary face (%d of %d); %s' % (false_alarms_above, number_of_settings, EQUALITY_NOTE))
table.add('L9-046', lure_set_above, 'rerun',
          'settings at which d_a(rare vs never-studied) - d_a(rare vs same-feature lure), read off the evidence (mean and '
          'SD of the log odds), is positive: %d of %d (the registered form; mean %+.3f, range %+.3f to %+.3f); the '
          'four-term interaction that rem_final_spec.py prints: %d of %d'
          % (lure_set_above, number_of_settings, lure_set_from_evidence.mean(), lure_set_from_evidence.min(),
             lure_set_from_evidence.max(), four_term_above, number_of_settings))

# ============================================================================ SI Appendix, Section 3: the run after data collection
table.add('L9-047', int(fitted_lure['settings']), 'provided',
          'settings of the REM run after data collection (the sweep grid without the criterion axis), 100 simulated '
          'participants each; rem_like_data.py computes the model side of every setting again (largest difference '
          'from the saved direct scores %s; reproduced: %s); the fitted scores are the %s'
          % (largest_model_side_difference, model_side_reproduced, MEMTOOLBOX_NOTE))
overall_use = []
for rating in range(1, 9):
    overall_use.append(sum(int(rating_counts[kind]['rating_%d' % rating]) for kind in rating_counts))
cumulative = np.cumsum(overall_use) / float(sum(overall_use))
table.add('L9-048',
          'one set of seven criteria at the cumulative proportions %s of the participants\' rating use (%s of %s '
          'trials, ratings 1 to 8); the simulated ratings use the eight ratings in the same shares at every setting '
          '(largest gap %s)'
          % (', '.join('%.4f' % value for value in cumulative[:-1]), ', '.join('{:,}'.format(count) for count in overall_use),
             '{:,}'.format(sum(overall_use)), rating_share_gap),
          'rerun', 'one set of seven criteria shared by every kind of face, placed at the cumulative proportions of the '
                   'participants\' rating use; computed from the data and from the model side (rem_like_data.py)')
table.add('L9-049', [int(fitted_lure['above_zero']), int(fitted_lure['settings'])], 'provided',
          'settings of the 243-setting run at which the fitted lure-set effect (d_a) is positive, and the number of '
          'settings; median %+.3f, range %+.3f to %+.3f; %s'
          % (float(fitted_lure['median']), float(fitted_lure['minimum']), float(fitted_lure['maximum']), MEMTOOLBOX_NOTE))

# ============================================================================ SI Appendix, Fig. S10 and the data it shows
model_rows = read_rows('figS10a_model_settings.csv', PATHS['figure_data'])
plotted_hits = np.array([float(row['hit_difference_rare_minus_ordinary']) for row in model_rows])
plotted_false_alarms = np.array([float(row['false_alarm_difference_feature_lure_minus_new']) for row in model_rows])
table.add('L9-050',
          'Fig. S10a as drawn: %d model settings, hit differences median %.3f (range %.3f to %.3f), false alarm '
          'differences median %.3f (range %.3f to %.3f); data hits %+.3f %s, false alarms %+.3f %s; labels "%d of %d '
          'above zero", "%d of %d above zero"'
          % (len(model_rows), np.median(plotted_hits), plotted_hits.min(), plotted_hits.max(),
             np.median(plotted_false_alarms), plotted_false_alarms.min(), plotted_false_alarms.max(),
             float(observed['hits']['mean_difference']),
             interval_text(float(observed['hits']['ci_low']), float(observed['hits']['ci_high'])),
             float(observed['false_alarms']['mean_difference']),
             interval_text(float(observed['false_alarms']['ci_low']), float(observed['false_alarms']['ci_high'])),
             hits_above, number_of_settings, false_alarms_above, number_of_settings),
          'rerun', 'outputs/figures/FigS10_rem_face_design.png; outputs/figure_data/figS10a_*.csv')
table.add('L9-051',
          'Fig. S10b as drawn from the saved fits: %d fitted settings, lure-set effect median %.3f (range %.3f to %.3f), '
          'feature effect median %.3f (range %.3f to %.3f); data lure set %+.3f %s, feature %+.3f %s; labels "%d of %d '
          'above zero", "%d of %d above zero"'
          % (int(fitted_lure['settings']), float(fitted_lure['median']), float(fitted_lure['minimum']),
             float(fitted_lure['maximum']), float(fitted_feature['median']), float(fitted_feature['minimum']),
             float(fitted_feature['maximum']), float(lure['estimate']),
             interval_text(float(lure['participants_ci_low']), float(lure['participants_ci_high'])),
             float(feature['estimate']),
             interval_text(float(feature['participants_ci_low']), float(feature['participants_ci_high'])),
             int(fitted_lure['above_zero']), int(fitted_lure['settings']), int(fitted_feature['above_zero']),
             int(fitted_feature['settings'])),
          'provided', 'outputs/figures/FigS10_rem_face_design.png; outputs/figure_data/figS10b_*.csv; the model points '
                      'and the observed effects with their intervals come from the %s' % MEMTOOLBOX_NOTE)
table.add('L9-052', registered, 'rerun', 'the registered participants (shared/python/face_loader.py)')
table.add('L9-053', number_of_settings, 'rerun', 'the settings of the sweep run before data collection (as L9-040)')
table.add('L9-054', int(fitted_lure['settings']), 'provided',
          'the settings of the run after data collection (as L9-047); %s' % MEMTOOLBOX_NOTE)
largest_p = max(registered_rate_p(observed['hits']), registered_rate_p(observed['false_alarms']),
                float(lure['participants_p_two_sided']))
table.add('L9-056', largest_p, 'rerun_and_provided',
          'the largest of the three registered two-sided bootstrap p values over participants: hits %+.3f, %s; false '
          'alarms %+.3f, %s (both computed from the data); lure-set effect %+.3f, %s (saved refits)'
          % (float(observed['hits']['mean_difference']), p_words(registered_rate_p(observed['hits']), 10000),
             float(observed['false_alarms']['mean_difference']),
             p_words(registered_rate_p(observed['false_alarms']), 10000), float(lure['estimate']),
             p_words(float(lure['participants_p_two_sided']), 10000)))
table.add('L9-102',
          [float(observed['hits']['mean_difference']), float(observed['hits']['ci_low']), float(observed['hits']['ci_high']),
           float(observed['false_alarms']['mean_difference']), float(observed['false_alarms']['ci_low']),
           float(observed['false_alarms']['ci_high'])], 'rerun',
          'hit rate of studied rare minus studied ordinary faces, and false alarm rate of new faces carrying a studied '
          'face\'s rare feature minus never-studied ordinary faces: mean paired difference over the 100 participants with '
          'the 95% interval of the registered participant bootstrap (10,000 resamples, resample i drawn with '
          'default_rng(i), participants in load_order); computed from the data (observed_contrasts.py)')
table.add('L9-057', int(fitted_feature['above_zero']), 'provided',
          'settings of the 243-setting run at which the fitted effect of the feature itself is positive (%d of %d); '
          'median %+.3f, range %+.3f to %+.3f; %s'
          % (int(fitted_feature['above_zero']), int(fitted_feature['settings']), float(fitted_feature['median']),
             float(fitted_feature['minimum']), float(fitted_feature['maximum']), MEMTOOLBOX_NOTE))
table.add('L9-058',
          'feature effect in the data %+.6f, 95%% interval over participants %s, %s'
          % (float(feature['estimate']),
             interval_text(float(feature['participants_ci_low']), float(feature['participants_ci_high'])),
             p_words(float(feature['participants_p_two_sided']), 10000)),
          'provided', 'feature effect in the data, d_a(rare vs same-feature lure) minus d_a(ordinary vs never-studied), '
                      'with its 95%% interval over participants and two-sided bootstrap p; %s' % MEMTOOLBOX_NOTE)
table.add('L9-059', 95, 'rerun_and_provided',
          '95%% percentile intervals over participants: hits %s and false alarms %s (10,000 resamples, resample i drawn '
          'with default_rng(i), participants in load_order; computed from the data); lure set %s and feature %s '
          '(10,000 saved participant refits)'
          % (interval_text(float(observed['hits']['ci_low']), float(observed['hits']['ci_high'])),
             interval_text(float(observed['false_alarms']['ci_low']), float(observed['false_alarms']['ci_high'])),
             interval_text(float(lure['participants_ci_low']), float(lure['participants_ci_high'])),
             interval_text(float(feature['participants_ci_low']), float(feature['participants_ci_high']))))

# ============================================================================ SI Appendix, Section 3: the preregistered d_a analysis
table.add('L9-060', [registered, sessions_run], 'rerun',
          '%d sessions run; %d excluded by the registered rule (%d with more than 5 presses on faces '
          'that were not repeats, %d with fewer than 4 of the 6 repeats detected, %d of them both); the first %d '
          'passing sessions in order of completion kept'
          % (sessions_run, sessions_run - registered, exclusion_reasons['too many presses'],
             exclusion_reasons['too few repeats'], exclusion_reasons['both'], registered))
table.add('L9-061',
          'the registered sample, the first 100 sessions passing the rule in order of completion, recomputed from the '
          'data: it equals the registered column of face_participants.csv (face_loader.recompute_participants: yes)',
          'rerun', 'the registered sample recomputed from the trials with the preregistered rule '
                   '(shared/python/face_loader.py, recompute_participants)')
criteria = []
running_total = 0.0
for name in ('c1', 'dc2', 'dc3', 'dc4', 'dc5', 'dc6', 'dc7'):
    running_total += point_fit['params'][name]
    criteria.append(running_total)
likelihood_check = [row for row in provided_checks if row['check'].startswith('log likelihood of the shipped data')][0]
table.add('L9-062',
          'never-studied ordinary faces N(0, 1); studied ordinary faces mean %.4f SD %.4f, studied rare faces %.4f, %.4f, '
          'same-feature lures %.4f, %.4f; one set of seven criteria for all four kinds of face (%s); 13 parameters; log '
          'likelihood %.2f, which the data give at these parameters (evaluated, not fitted: %s)'
          % (point_fit['cells']['ordinary_old'][0], point_fit['cells']['ordinary_old'][1], point_fit['cells']['rare_old'][0],
             point_fit['cells']['rare_old'][1], point_fit['cells']['rare_foil'][0], point_fit['cells']['rare_foil'][1],
             ', '.join('%.3f' % value for value in criteria), point_fit['loglike'], likelihood_check['agree']),
          'provided', 'registered fit: mean and SD of each kind of face, one set of seven criteria, 13 parameters, log '
                      'likelihood evaluated on the data; %s' % MEMTOOLBOX_NOTE)
for ledger_id, quantity in (('L9-063', 'rare_unrel'), ('L9-064', 'rare_matched'), ('L9-065', 'ordinary')):
    row = fit[quantity]
    table.add(ledger_id, with_participant_interval(row), 'provided',
              '%s, from the registered fit, with its 95%% interval over participants (10,000 refits); %s'
              % (row['description'], MEMTOOLBOX_NOTE))
table.add('L9-066', 'lure-set effect %+.3f; feature effect %+.3f' % (float(lure['estimate']), float(feature['estimate'])),
          'provided', 'lure-set effect (95%% interval over participants %s) and feature effect (%s) from the registered '
                      'fit; %s'
          % (interval_text(float(lure['participants_ci_low']), float(lure['participants_ci_high'])),
             interval_text(float(feature['participants_ci_low']), float(feature['participants_ci_high'])), MEMTOOLBOX_NOTE))

# ============================================================================ SI Appendix, Section 3: resampling the features
table.add('L9-067', len(features_of_pairs), 'rerun',
          'rare-feature pairs in the test trials of the 100 registered participants: %s' % pairs_text)
table.add('L9-068',
          [float(lure['estimate']), float(lure['items_ci_low']), float(lure['items_ci_high']),
           float(lure['items_p_two_sided']), int(lure['items_resamples']),
           float(feature['estimate']), float(feature['items_ci_low']), float(feature['items_ci_high']),
           float(feature['items_p_two_sided'])],
          'provided', 'the lure-set and feature effects from the registered fit, with the 95%% intervals and two-sided '
                      'bootstrap p over %s resamples of the faces (the six rare-feature pairs, 59 studied and 59 '
                      'never-studied ordinary faces): lure set %s; feature %s; over participants: %s and %s; %s'
          % ('{:,}'.format(int(lure['items_resamples'])),
             p_words(float(lure['items_p_two_sided']), int(lure['items_resamples'])),
             p_words(float(feature['items_p_two_sided']), int(feature['items_resamples'])),
             interval_text(float(lure['participants_ci_low']), float(lure['participants_ci_high'])),
             interval_text(float(feature['participants_ci_low']), float(feature['participants_ci_high'])),
             MEMTOOLBOX_NOTE))
lopo_lure = [float(lure['leave_one_pair_out_min']), float(lure['leave_one_pair_out_max'])]
lopo_feature = [float(feature['leave_one_pair_out_min']), float(feature['leave_one_pair_out_max'])]
table.add('L9-069', lopo_lure, 'provided',
          'lure-set effect over the six leave-one-pair-out refits: %.5f (without %s) to %.5f (without %s); %s'
          % (lopo_lure[0], lure['leave_one_pair_out_min_without'], lopo_lure[1], lure['leave_one_pair_out_max_without'],
             MEMTOOLBOX_NOTE))
table.add('L9-070', lopo_feature, 'provided',
          'feature effect over the six leave-one-pair-out refits: %.5f (without %s) to %.5f (without %s); %s'
          % (lopo_feature[0], feature['leave_one_pair_out_min_without'], lopo_feature[1],
             feature['leave_one_pair_out_max_without'], MEMTOOLBOX_NOTE))

# ============================================================================ Fig. 4c and SI Appendix, Fig. S11b
points_text = '; '.join('%s [%s]' % (kind, ' '.join('%.3f' % float(operating_points[kind]['rated_%d_or_higher' % rating])
                                                    for rating in range(2, 9)))
                        for kind in ('studied_rare', 'same_feature_lure', 'studied_ordinary', 'never_studied'))
table.add('L9-072', 'operating points: %s' % points_text, 'rerun_and_provided',
          'Fig. 4c as drawn: the seven operating points of each kind of face, computed from the data (roc_points.py), and '
          'the curves of the registered unequal-variance fit (data/provided/point_fit.json; refitting needs '
          'MemToolbox2, EXTERNAL_INPUTS.md, item M5); outputs/figures/Fig4c_confidence_roc.png, curve labels "Rare vs '
          'Unrelated New Faces", "Rare vs Same-Feature Lure", "Ordinary vs Unrelated New Faces"')
table.add('L9-073',
          'lure set %+.3f: over participants %s, over features %s; feature %+.3f: over participants %s, over features %s'
          % (float(lure['estimate']), interval_text(float(lure['participants_ci_low']), float(lure['participants_ci_high'])),
             interval_text(float(lure['items_ci_low']), float(lure['items_ci_high'])), float(feature['estimate']),
             interval_text(float(feature['participants_ci_low']), float(feature['participants_ci_high'])),
             interval_text(float(feature['items_ci_low']), float(feature['items_ci_high']))),
          'provided', 'outputs/figures/FigS11b_lure_set_and_feature_effects.png: the two effects with 95%% intervals over '
                      'participants (10,000 refits) and over the six rare features (1,000 refits); %s' % MEMTOOLBOX_NOTE)
table.add('L9-074', registered, 'rerun', 'the registered participants (as L9-052)')
trial_text = ', '.join('%s %s' % (kind, '{:,}'.format(int(rating_counts[kind]['total_trials']))) for kind in rating_counts)
table.add('L9-075', [len(ratings_used), len(confidence_levels)], 'rerun',
          'rating steps used in the data (old or new answer crossed with the confidence rating) and confidence levels; '
          'seven criteria; trials pooled over the 100 participants: %s' % trial_text)
table.add('L9-076', with_participant_interval(rare_unrel), 'provided',
          'd_a, rare faces against never-studied faces (red curve of Fig. 4c), from the registered fit, with its 95%% '
          'interval over participants (10,000 refits); %s' % MEMTOOLBOX_NOTE)
table.add('L9-077', with_participant_interval(rare_matched) + with_participant_interval(ordinary), 'provided',
          'd_a, rare faces against same-feature lures (blue, dashed) and ordinary faces against never-studied faces '
          '(dark gray), from the registered fit, each with its 95%% interval over participants (10,000 refits); %s'
          % MEMTOOLBOX_NOTE)
table.add('L9-078', [float(rare_unrel['estimate']), float(rare_matched['estimate'])], 'provided',
          'the two d_a values whose difference is the lure-set effect: %.5f - %.5f = %.5f; %s'
          % (float(rare_unrel['estimate']), float(rare_matched['estimate']), float(lure['estimate']), MEMTOOLBOX_NOTE))
table.add('L9-079', [95, len(features_of_pairs), int(lure['items_resamples'])], 'provided',
          '95%% percentile intervals: over participants (10,000 refits, registered) and over the %d rare features (%s '
          'refits; the 59 studied ordinary and 59 never-studied faces resampled too); %s'
          % (len(features_of_pairs), '{:,}'.format(int(lure['items_resamples'])), MEMTOOLBOX_NOTE))
table.add('L9-080', with_participant_interval(lure) + [float(lure['participants_p_two_sided'])], 'provided',
          'lure-set effect with its 95%% interval over participants (10,000 refits) and the registered two-sided '
          'bootstrap p: %s; over the features: %s; %s'
          % (p_words(float(lure['participants_p_two_sided']), 10000),
             p_words(float(lure['items_p_two_sided']), int(lure['items_resamples'])), MEMTOOLBOX_NOTE))
table.add('L9-081', with_participant_interval(feature) + [float(feature['participants_p_two_sided'])], 'provided',
          'feature effect with its 95%% interval over participants (10,000 refits) and the registered two-sided '
          'bootstrap p over participants; over the features p = %.3f; %s'
          % (float(feature['items_p_two_sided']), MEMTOOLBOX_NOTE))

# The script that computes each row's value (numbers_face_study.py only gathers them; it counts the
# participants and the rare-feature pairs itself).
SCRIPT_OF = {
    'L2-028': 'code/check_provided.py', 'L5-044': 'code/check_provided.py',
    'L2-029': 'code/rem_729_rates.py; code/check_provided.py',
    'L9-067': 'code/numbers_face_study.py',
    'L9-040': 'code/rem_729_rates.py', 'L9-041': 'code/rem_final_spec.py', 'L9-053': 'code/rem_729_rates.py',
    'L9-043': 'code/rem_final_spec.py; code/rem_729_rates.py', 'L9-044': 'code/rem_729_rates.py',
    'L9-045': 'code/rem_final_spec.py; code/rem_729_rates.py', 'L9-046': 'code/rem_729_rates.py',
    'L9-047': 'code/check_provided.py; code/rem_like_data.py', 'L9-048': 'code/rem_like_data.py',
    'L9-049': 'code/check_provided.py', 'L9-054': 'code/check_provided.py', 'L9-057': 'code/check_provided.py',
    'L9-050': 'code/make_figureS10.py', 'L9-051': 'code/make_figureS10.py',
    'L9-052': 'code/numbers_face_study.py', 'L9-074': 'code/numbers_face_study.py',
    'L9-060': 'code/numbers_face_study.py', 'L9-061': 'code/numbers_face_study.py',
    'L9-056': 'code/observed_contrasts.py; code/check_provided.py',
    'L9-059': 'code/observed_contrasts.py; code/check_provided.py',
    'L9-102': 'code/observed_contrasts.py',
    'L9-072': 'code/roc_points.py; code/make_figureS11.py',
    'L9-073': 'code/make_figureS11.py', 'L9-075': 'code/roc_points.py',
}
for row in table.rows:
    row['script'] = SCRIPT_OF.get(row['ledger_id'], 'code/check_provided.py')

if not all_provided_checks_agree:
    print('WARNING: check_provided.py found a saved number that does not agree; see outputs/provided_checks.csv')
table.write(OUTPUTS)
for row in table.rows:
    print('%-7s %-9s %-11s printed %-32s computed %s' % (row['ledger_id'], row['match'], row['mode'], row['paper_value'][:32],
                                                         row['computed_value'][:90]))
