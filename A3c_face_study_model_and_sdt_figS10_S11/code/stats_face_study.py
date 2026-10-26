"""
stats_face_study.py: the uncertainty statistics of the rare-feature face study's model predictions and
its preregistered d_a analysis (SI Appendix, Section 3, and Fig. 4c; tests T48, T49, T50 and T51 of
statistics/README.md), written to outputs/tests.csv.

WHAT IT REPORTS
  T48  REM's predictions as counts of parameter settings with the predicted sign (SI Appendix, Fig.
       S10). Descriptive, no inference: the 729-setting sweep run before data collection
       (rem_729_rates.py, seed = setting number) and the 243-setting run scored like the data (saved
       fits; refitting needs MemToolbox2).
  T49  The data points of Fig. S10, each with the registered participant bootstrap (the plotted error
       bars) and its two-sided p: the hit and false alarm differences (predictions 1 and 2) and the d_a
       lure-set effect (prediction 3; 10,000 saved refits).
  T50  The three d_a values of the registered unequal-variance fit (Fig. 4c) with their 95% intervals
       from the 10,000 saved participant refits and a description of the fit.
  T51  The effect of the lure set and the effect of the feature itself (SI Appendix, Fig. S11b):
       intervals over participants (registered, 10,000 refits) and over the six rare features (1,000
       refits), the bootstrap p, and the smallest and largest value over the six leave-one-pair-out
       refits.
The bootstrap p of the registered analysis (its preregistration, question 5) is two-sided: twice the
smaller of the share of resamples at or below zero and the share at or above zero. With 10,000
resamples the smallest p it can give above zero is 0.0002 (0.002 with 1,000); a p of 0 means that no
resample reached zero. Every d_a interval comes from the saved refits in data/provided/, because new
refits need MemToolbox2 (EXTERNAL_INPUTS.md, item M5). No correction for multiple comparisons is
applied (statistics/README.md).

INPUTS   04_rare_feature_faces_fig4/data/ (through shared/python/face_loader.py and face_study_common.py);
         outputs/rem_729_rates.csv (rem_729_rates.py); outputs/observed_contrasts.csv
         (observed_contrasts.py); outputs/registered_fit_summary.csv and outputs/rem_243_summary.csv
         (check_provided.py); data/provided/point_fit.json (the log likelihood of the registered fit,
         quoted in the T50 description)
OUTPUTS  outputs/tests.csv
SEEDS    none drawn here (the registered bootstrap of the rate differences comes from
         observed_contrasts.py: resample i drawn with numpy default_rng(i))
RUNTIME  about 5 seconds
RUN      python A3c_face_study_model_and_sdt_figS10_S11/code/stats_face_study.py
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
import release_paths        # noqa: E402
import face_loader        # noqa: E402
import tests_table        # noqa: E402
import face_study_common    # noqa: E402

PATHS = release_paths.module_paths(__file__)
SCRIPT = 'code/stats_face_study.py'
NUMBER_OF_RESAMPLES = 10000
UNIT = 'participants (the 100 registered)'

tests = tests_table.TestsTable(script=SCRIPT)


def read_csv_rows(file_name):
    with open(os.path.join(PATHS['outputs'], file_name), newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def p_text(p_value, number_of_resamples):
    """The bootstrap p in words: 'p = 0.991', or for 0 'no resample at or below zero (p < 0.0002)'."""
    if p_value == 0:
        return 'no resample reached zero (p < %.4f at %s resamples)' % (2.0 / number_of_resamples,
                                                                       '{:,}'.format(number_of_resamples))
    return 'p = %.4f' % p_value


# ============================================================================ T48: counts of settings
GRID_TEXT = ('REM (Shiffrin & Steyvers, 1997) through the face design: g {0.3, 0.4, 0.5} x w {15, 20, 25} x '
             'u {0.2, 0.3, 0.4} x c {0.6, 0.7, 0.8} x number of features carrying the rare value {1, 3, 5}')
registered_rows = read_csv_rows('rem_729_rates.csv')
number_of_settings = len(registered_rows)

for column, description, ledger_ids, location in (
        ('hit_difference_rare_minus_ordinary',
         'prediction 1, hits: studied rare faces minus studied ordinary faces (proportion called old)',
         'L9-043;L2-029', 'SI Appendix, Section 3, The Model Predicted Each Preregistered Face Result (Fig. S10a)'),
        ('false_alarm_difference_feature_lure_minus_new',
         'prediction 2, false alarms: a new face carrying a studied face\'s rare feature minus a new ordinary face',
         'L9-045;L2-029', 'SI Appendix, Section 3, The Model Predicted Each Preregistered Face Result (Fig. S10a)'),
        ('lure_set_effect_d_a',
         'prediction 3, read off the model\'s evidence: d_a(rare vs never-studied) - d_a(rare vs same-feature lure), '
         'the registered form (rem_final_spec.py prints a four-term interaction, also positive at every setting)',
         'L9-046;L2-029', 'SI Appendix, Section 3, The Model Predicted Each Preregistered Face Result')):
    registered_values = np.array([float(row[column]) for row in registered_rows])
    registered_count = int(np.sum(registered_values > 0))
    tests.add(test_id='T48', ledger_ids=ledger_ids, location=location,
              test='%s; %s x criterion placed so that a share fa {0.2, 0.42, 0.6} of never-studied faces is called '
                   'old: 729 settings, 500 simulated participants each, seed = setting number (0 to 728); counts of '
                   'settings above zero, descriptive (no inference)' % (description, GRID_TEXT),
              statistic_name='settings above zero', statistic=registered_count, n=number_of_settings,
              unit='parameter settings', df='', p='', p_type='none', effect_size_name='proportion of settings',
              effect_size=registered_count / number_of_settings, ci_low='', ci_high='',
              ci_method='none (counts over a grid of settings)', resampling_unit='', n_resamples='',
              seed='setting number (0 to 728)')

rem_243 = {row['scoring_and_quantity']: row for row in read_csv_rows('rem_243_summary.csv')}
for name, description, ledger_ids, location in (
        ('fitted lure-set effect', 'effect of the lure set, d_a(rare vs never-studied) - d_a(rare vs same-feature lure)',
         'L9-047;L9-049;L2-029',
         'SI Appendix, Section 3, The Model Predicted Each Preregistered Face Result (Fig. S10b)'),
        ('fitted feature effect', 'effect of the feature itself, d_a(rare vs same-feature lure) - d_a(ordinary vs '
         'never-studied)', 'L9-047;L9-057',
         'SI Appendix, Section 3, The Model Predicted Each Preregistered Face Result (Fig. S10b)')):
    row = rem_243[name]
    count = int(row['above_zero'])
    settings = int(row['settings'])
    tests.add(test_id='T48', ledger_ids=ledger_ids, location=location,
              test='REM run after data collection, scored as the data were: %s; the grid above without the criterion '
                   'axis (243 settings), 100 simulated participants each, seed = setting number (0 to 242); the '
                   'evidence turned into eight-step ratings with one shared set of seven criteria matched to the '
                   'participants\' rating use and fitted with the registered unequal-variance model; median %.3f, '
                   'range %.3f to %.3f; counts of settings above zero, descriptive. Saved fits (data/provided/'
                   'rem_like_data.jsonl); rem_like_data.py reproduces the model side exactly; refitting needs '
                   'MemToolbox2' % (description, float(row['median']), float(row['minimum']), float(row['maximum'])),
              statistic_name='settings above zero', statistic=count, n=settings, unit='parameter settings', df='', p='',
              p_type='none', effect_size_name='proportion of settings', effect_size=count / settings, ci_low='',
              ci_high='', ci_method='none (counts over a grid of settings)', resampling_unit='', n_resamples='',
              seed='setting number (0 to 242)')


# ============================================================================ T49: the data points of Fig. S10
rates_by_completion = face_loader.participant_rates(order='completion')
observed = {row['contrast']: row for row in read_csv_rows('observed_contrasts.csv')}
number_of_participants = len(rates_by_completion)
DATA_LOCATION = 'SI Appendix, Section 3, The Model Predicted Each Preregistered Face Result'

for contrast_name, description, ledger_ids in (
        ('hits',
         'Hit rate, studied rare-feature faces minus studied ordinary faces (preregistered prediction 1; Fig. S10a, '
         'blue point)', 'L9-102;L9-056;L9-059;L9-050'),
        ('false_alarms',
         'False alarm rate, new faces carrying a studied face\'s rare feature minus never-studied ordinary faces '
         '(preregistered prediction 2; Fig. S10a, red point)', 'L9-102;L9-056;L9-059;L9-050')):
    # the registered participant bootstrap (the plotted error bar), from observed_contrasts.py
    row = observed[contrast_name]
    registered_p = 2 * min(int(row['resamples_at_or_below_zero']), int(row['resamples_at_or_above_zero'])) / NUMBER_OF_RESAMPLES
    tests.add(test_id='T49', ledger_ids=ledger_ids, location=DATA_LOCATION + ' (Fig. S10a)',
              test='%s: mean paired difference with the registered participant bootstrap, the plotted error bar '
                   '(resample i drawn with numpy default_rng(i), participants in load_order); '
                   '%s of %s resampled means at or below zero; two-sided bootstrap p as registered: %s'
                   % (description, '{:,}'.format(int(row['resamples_at_or_below_zero'])), '{:,}'.format(NUMBER_OF_RESAMPLES),
                      p_text(registered_p, NUMBER_OF_RESAMPLES)),
              statistic_name='mean paired difference', statistic=float(row['mean_difference']), n=number_of_participants,
              unit=UNIT, df='', p=registered_p, p_type='bootstrap two-sided',
              effect_size_name='difference in proportion', effect_size=float(row['mean_difference']),
              ci_low=float(row['ci_low']), ci_high=float(row['ci_high']),
              ci_method='registered percentile bootstrap (resample i drawn with numpy default_rng(i), participants in '
                        'load_order)',
              resampling_unit='participants', n_resamples=NUMBER_OF_RESAMPLES, seed='i = 0 to 9999 (one per resample)')

registered_fit = {row['quantity']: row for row in read_csv_rows('registered_fit_summary.csv')}
lure_set = registered_fit['interaction']
lure_set_p = float(lure_set['participants_p_two_sided'])
tests.add(test_id='T49', ledger_ids='L9-080;L9-056;L9-059;L9-051;L2-028', location=DATA_LOCATION + ' (Fig. S10b)',
          test='Effect of the lure set in the data (preregistered prediction 3): d_a(rare vs never-studied) - d_a(rare vs '
               'same-feature lure) from the registered unequal-variance fit (Fig. S10b, dark point); 95%% interval and '
               'two-sided bootstrap p from the registered participant bootstrap, each of the 10,000 resamples refitted '
               '(%d at or below zero: %s); saved refits (data/provided/bootstrap_raw.jsonl); new refits need '
               'MemToolbox2' % (int(lure_set['participants_at_or_below_zero']),
                                p_text(lure_set_p, int(lure_set['participants_resamples']))),
          statistic_name='d_a difference', statistic=float(lure_set['estimate']), n=100, unit=UNIT, df='', p=lure_set_p,
          p_type='bootstrap two-sided', effect_size_name='d_a difference', effect_size=float(lure_set['estimate']),
          ci_low=float(lure_set['participants_ci_low']), ci_high=float(lure_set['participants_ci_high']),
          ci_method='registered percentile bootstrap, the model refitted in each resample (saved refits)',
          resampling_unit='participants', n_resamples=int(lure_set['participants_resamples']),
          seed='participants drawn from one numpy default_rng(0) stream; refit seed i')


# ============================================================================ T50: the three d_a values
participants = face_study_common.analysable(face_study_common.load())
trial_counts = {'rare_old': 0, 'rare_foil': 0, 'ordinary_old': 0, 'unrelated': 0}
for participant in participants:
    for trial in participant['rec']:
        if trial['condition'] == 'rare':
            if trial['test_type'] == 'old':
                trial_counts['rare_old'] += 1
            else:
                trial_counts['rare_foil'] += 1
        elif trial['test_type'] == 'old':
            trial_counts['ordinary_old'] += 1
        else:
            trial_counts['unrelated'] += 1

with open(os.path.join(PATHS['provided'], 'point_fit.json'), encoding='utf-8') as handle:
    saved_log_likelihood = json.load(handle)['loglike']     # the registered fit (saved fit)
FIT_TEXT = ('registered unequal-variance signal detection model fitted by maximum likelihood to the 3,600 test trials '
            'pooled over the 100 participants: never-studied ordinary faces N(0, 1); studied ordinary faces, studied '
            'rare faces and same-feature lures each with a free mean and SD; one set of seven criteria shared by all '
            'four kinds of face (eight rating steps); 13 parameters; 12 starting points, then restarts until the log '
            'likelihood improves by less than 0.01 (up to six rounds); log likelihood %.2f; 0 of 10,000 '
            'participant refits failed' % saved_log_likelihood)
for quantity, description, trials_text, ledger_ids in (
        ('rare_unrel', 'd_a, studied rare faces against never-studied ordinary faces (Fig. 4c, red)',
         '%d rare-face trials against %d never-studied trials' % (trial_counts['rare_old'], trial_counts['unrelated']),
         'L9-101;L9-063;L9-076;L9-078;L2-028;L5-044'),
        ('rare_matched', 'd_a, studied rare faces against their same-feature lures (Fig. 4c, blue, dashed)',
         '%d rare-face trials against %d same-feature lure trials' % (trial_counts['rare_old'], trial_counts['rare_foil']),
         'L9-101;L9-064;L9-077;L9-078;L2-028;L5-044'),
        ('ordinary', 'd_a, studied ordinary faces against never-studied ordinary faces (Fig. 4c, dark gray)',
         '%d ordinary-face trials against %d never-studied trials' % (trial_counts['ordinary_old'], trial_counts['unrelated']),
         'L9-101;L9-065;L9-077;L2-028;L5-044')):
    row = registered_fit[quantity]
    estimate = float(row['estimate'])
    tests.add(test_id='T50', ledger_ids=ledger_ids,
              location='SI Appendix, Section 3, The Preregistered Discriminability Analysis of the Face Study (Fig. 4c)',
              test='%s: %s; %s; 95%% interval: 2.5th and 97.5th percentiles of the 10,000 saved participant refits '
                   '(data/provided/bootstrap_raw.jsonl; new refits need MemToolbox2)' % (description, trials_text, FIT_TEXT),
              statistic_name='d_a', statistic=estimate, n=100, unit=UNIT, df='', p='', p_type='none',
              effect_size_name='d_a', effect_size=estimate, ci_low=float(row['participants_ci_low']),
              ci_high=float(row['participants_ci_high']),
              ci_method='registered percentile bootstrap, the model refitted in each resample (saved refits)',
              resampling_unit='participants', n_resamples=int(row['participants_resamples']),
              seed='participants drawn from one numpy default_rng(0) stream; refit seed i')


# ============================================================================ T51: the two contrasts, every interval
for quantity, description, ledger_ids in (
        ('interaction', 'Effect of the lure set, d_a(rare vs never-studied) - d_a(rare vs same-feature lure)',
         'L9-066;L9-068;L9-073;L9-078;L9-079;L9-080;L2-028'),
        ('equivalence', 'Effect of the feature itself, d_a(rare vs same-feature lure) - d_a(ordinary vs never-studied)',
         'L9-058;L9-066;L9-068;L9-073;L9-079;L9-081;L2-028')):
    row = registered_fit[quantity]
    estimate = float(row['estimate'])
    for scheme, scheme_text, method, unit_text, location in (
            ('participants', 'resampling the 100 participants (the registered bootstrap; Fig. S11b, left of each pair)',
             'registered percentile bootstrap, the model refitted in each resample (saved refits)', 'participants',
             DATA_LOCATION + ' (Fig. S11b)'),
            ('items', 'resampling the faces: the 6 rare-feature pairs, the 59 studied ordinary faces and the 59 '
                      'never-studied faces, participants kept (Fig. S11b, right of each pair)',
             'percentile bootstrap over the faces, the model refitted in each resample (saved refits)',
             'the faces (the six rare-feature pairs, the 59 studied and the 59 never-studied ordinary faces)',
             'SI Appendix, Section 3, The Preregistered Discriminability Analysis of the Face Study (Fig. S11b)')):
        p_value = float(row['%s_p_two_sided' % scheme])
        resamples = int(row['%s_resamples' % scheme])
        at_or_below = int(row['%s_at_or_below_zero' % scheme])
        tests.add(test_id='T51', ledger_ids=ledger_ids, location=location,
                  test='%s, %s; estimate from the registered fit; %s of %s resamples at or below zero; two-sided '
                       'bootstrap p as registered: %s; saved refits (data/provided/); new refits need MemToolbox2'
                       % (description, scheme_text, '{:,}'.format(at_or_below), '{:,}'.format(resamples),
                          p_text(p_value, resamples)),
                  statistic_name='d_a difference', statistic=estimate, n=100 if scheme != 'items' else 6,
                  unit=UNIT if scheme != 'items' else 'rare-feature pairs (100 participants kept)', df='', p=p_value,
                  p_type='bootstrap two-sided', effect_size_name='d_a difference', effect_size=estimate,
                  ci_low=float(row['%s_ci_low' % scheme]), ci_high=float(row['%s_ci_high' % scheme]),
                  ci_method=method, resampling_unit=unit_text, n_resamples=resamples,
                  seed='participants: one numpy default_rng(0) stream, refit seed i' if scheme == 'participants'
                  else 'resample i: numpy default_rng(10000 + i), refit seed i')
    # leave one pair out: the smallest and the largest of the six refits, one row each (a range, not an interval)
    for end, column in (('smallest', 'leave_one_pair_out_min'), ('largest', 'leave_one_pair_out_max')):
        value = float(row[column])
        left_out = row[column + '_without']
        tests.add(test_id='T51', ledger_ids='L9-069;L9-070' if quantity == 'equivalence' else 'L9-069',
                  location='SI Appendix, Section 3, The Preregistered Discriminability Analysis of the Face Study',
                  test='%s, leave one pair out: the registered model refitted six times, each time without one '
                       'rare-feature pair (seed 0); the %s of the six refits (without %s); the registered fit with all '
                       'pairs gives %+.5f; saved fits (data/provided/leave_one_pair_out.json); new fits need '
                       'MemToolbox2' % (description, end, left_out, estimate),
                  statistic_name='%s d_a difference over the six leave-one-pair-out refits' % end, statistic=value,
                  n=6, unit='refits (one rare-feature pair left out in each)', df='', p='', p_type='none',
                  effect_size_name='d_a difference', effect_size=value, ci_low='', ci_high='',
                  ci_method='none (a range over refits, not an interval)', resampling_unit='', n_resamples='',
                  seed='0')

tests.write(PATHS['outputs'])
