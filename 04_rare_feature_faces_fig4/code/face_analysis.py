"""
face_analysis.py: the registered descriptive analysis of the rare-feature face study (Fig. 4b;
SI Appendix, Fig. S10a; Methods, Experimental Procedures and Analysis).

THE IDEA
A feature that is rare among the studied faces raises the match signal for ANY face that carries
it, the studied face and a stranger alike. So the analysis looks at hits, false alarms and the
criterion together, for rare faces against ordinary ones, each scored against its own kind of lure:
a rare face against the new face that shares its feature, an ordinary face against a face nobody
studied.

WHAT IT DOES
1. The sample. Applies the preregistered exclusion rule (AsPredicted #311640, question 6) to the
   111 sessions of the study, in the order they were completed: a session passes with 36 test
   trials, at least 4 of the 6 one-back repeats detected and at most 5 presses on faces that were
   not repeats. The first 100 passing sessions are the registered sample; the rest are excluded.
   Writes the participant flow (111 run, 11 excluded by reason, 100 kept).
2. Rates. For each of the 100 participants, the proportion of test faces called old in the four
   kinds of test item: studied faces carrying a rare feature (3 trials), new faces carrying a
   studied face's rare feature (3), studied ordinary faces (15) and never-studied ordinary faces
   (15); and the equal-variance d' and criterion c of each kind of face against its own kind of
   lure, with the log-linear correction (0.5 added to each count, 1 to each number of trials).
   Criterion c is a preregistered secondary measure and the equal-variance d' the preregistered
   baseline (preregistration, question 3); the paper reports discriminability as d_a, from the
   unequal-variance fit of module A3c_face_study_model_and_sdt_figS10_S11.
3. Rare against ordinary on every measure (hit rate, false alarm rate, d', c): a paired t test over
   participants, the Wilcoxon signed-rank test (the preregistered test of predictions 1 and 2),
   Cohen's d_z and a 95% percentile bootstrap interval of the mean paired difference. The
   registered participant bootstrap: 10,000 resamples of the 100 participants; resample number i
   draws its participants with its own generator, numpy.random.default_rng(i). The draw depends on
   the order of the participants, so they are taken in load_order (the session files sorted by
   file name).
4. Writes the results in words and as tables.

INPUTS   data/face_trials.csv and data/face_participants.csv, through shared/python/face_loader.py
OUTPUTS  outputs/results_registered_100.txt   the results in words
         outputs/rates_by_participant.csv     one row per registered participant
         outputs/rate_contrasts.csv           rare against ordinary on the four measures
         outputs/participant_flow.csv         sessions run, excluded by reason, kept (test T54)
SEEDS    bootstrap resample i uses numpy.random.default_rng(i), i = 0 to 9,999
RUN      python 04_rare_feature_faces_fig4/code/face_analysis.py   (from the repository root; about 2 s)
"""
import sys
sys.dont_write_bytecode = True
import csv
import os

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths
import face_loader
import bootstrap

PATHS = release_paths.module_paths(__file__)

MIN_ONEBACK_HITS = face_loader.MIN_ONEBACK_HITS          # 4 of the 6 repeats
MAX_ONEBACK_FALSE_PRESSES = face_loader.MAX_ONEBACK_FA   # presses on faces that were not repeats
EXPECTED_TEST_TRIALS = 36
REGISTERED_SAMPLE_SIZE = 100
NUMBER_OF_RESAMPLES = 10000


# ============================================================================= 1. the sample
def exclusion_reasons(session):
    """Why this session is excluded under the registered rule, or an empty list."""
    reasons = []
    if session['test_trials'] != EXPECTED_TEST_TRIALS:
        reasons.append('%d test trials, expected %d' % (session['test_trials'], EXPECTED_TEST_TRIALS))
    if session['oneback_hits'] < MIN_ONEBACK_HITS:
        reasons.append('1-back hits %d of %d' % (session['oneback_hits'], session['oneback_repeats']))
    if session['oneback_false_presses'] > MAX_ONEBACK_FALSE_PRESSES:
        reasons.append('1-back false alarms %d' % session['oneback_false_presses'])
    return reasons


all_trials = face_loader.load_trials('all')
participant_table = face_loader.recompute_participants()     # checks every column of face_participants.csv

sessions = []                                                # in completion order (S001 = first completed)
for participant in participant_table.sort_values('completion_rank')['participant']:
    own_trials = all_trials[all_trials['participant'] == participant]
    study_rows = own_trials[own_trials['phase'] == 'study']
    repeat_rows = study_rows[study_rows['is_repeat'] == 1]
    first_showing_rows = study_rows[study_rows['is_repeat'] == 0]
    session = dict(participant=participant,
                   completion_rank=int(own_trials['completion_rank'].iloc[0]),
                   load_order=int(own_trials['load_order'].iloc[0]),
                   test_trials=int((own_trials['phase'] == 'recognition').sum()),
                   oneback_repeats=len(repeat_rows),
                   oneback_hits=int((repeat_rows['oneback_correct'] == 1).sum()),
                   oneback_false_presses=int((first_showing_rows['pressed'] == 1).sum()))
    session['reasons'] = exclusion_reasons(session)
    sessions.append(session)

passing_sessions = []
excluded_sessions = []
for session in sessions:
    if session['reasons']:
        excluded_sessions.append(session)
    else:
        passing_sessions.append(session)
registered_sessions = passing_sessions[:REGISTERED_SAMPLE_SIZE]
passing_but_not_needed = passing_sessions[REGISTERED_SAMPLE_SIZE:]

# the registered sample must be the one the shared loader derives (and the one the paper analyzed)
registered_ids_by_completion = [session['participant'] for session in registered_sessions]
if registered_ids_by_completion != face_loader.registered_participants(order='completion'):
    raise SystemExit('the registered sample differs from face_loader.registered_participants()')

# reasons, counted the way the paper's rule is written (each part of the rule on its own)
too_many_false_presses = 0
too_few_repeats_detected = 0
both_parts_failed = 0
wrong_number_of_test_trials = 0
for session in excluded_sessions:
    fails_false_presses = session['oneback_false_presses'] > MAX_ONEBACK_FALSE_PRESSES
    fails_repeats = session['oneback_hits'] < MIN_ONEBACK_HITS
    if fails_false_presses:
        too_many_false_presses += 1
    if fails_repeats:
        too_few_repeats_detected += 1
    if fails_false_presses and fails_repeats:
        both_parts_failed += 1
    if session['test_trials'] != EXPECTED_TEST_TRIALS:
        wrong_number_of_test_trials += 1

# ============================================================================= 2. rates
# The analysis order: load_order (the session files sorted by file name). The seeded bootstrap
# below depends on it.
rates = face_loader.participant_rates(order='load_order')
if sorted(rates['participant']) != sorted(registered_ids_by_completion):
    raise SystemExit('face_loader.participant_rates() returned a different set of participants')
for column, expected_trials in (('n_hr', 3), ('n_fr', 3), ('n_hf', 15), ('n_fn', 15)):
    if not (rates[column] == expected_trials).all():
        raise SystemExit('a participant has an unexpected number of trials in %s' % column)
number_of_participants = len(rates)

# names used below: hit_rare (studied rare-feature faces called old), fa_rare (new faces carrying a
# studied face's rare feature called old), hit_fill (studied ordinary faces called old), fa_new
# (never-studied ordinary faces called old); d_rare, c_rare (rare against its own lure), d_fill,
# c_fill (ordinary against never-studied faces)


# ============================================================================= 3. rare against ordinary
def contrast(rare_values, ordinary_values):
    """Paired comparison of rare against ordinary over participants."""
    differences = rare_values - ordinary_values
    t_value, p_value = ttest_rel(rare_values, ordinary_values)
    try:
        wilcoxon_p = wilcoxon(rare_values, ordinary_values).pvalue
    except ValueError:                                # every difference zero
        wilcoxon_p = np.nan
    resampled_means = bootstrap.seeded_per_resample_mean_bootstrap(differences, NUMBER_OF_RESAMPLES)
    interval = np.percentile(resampled_means, [2.5, 97.5])
    return dict(rare=float(np.mean(rare_values)), ordinary=float(np.mean(ordinary_values)),
                difference=float(np.mean(differences)), ci_low=float(interval[0]), ci_high=float(interval[1]),
                t=float(t_value), df=len(differences) - 1, p=float(p_value), wilcoxon_p=float(wilcoxon_p),
                cohen_dz=float(np.mean(differences) / np.std(differences, ddof=1)),
                positive=int(np.sum(differences > 0)), tied=int(np.sum(differences == 0)),
                negative=int(np.sum(differences < 0)))


measures = [('hit', 'hit rate', 'hit_rare', 'hit_fill'),
            ('fa', 'false alarm rate', 'fa_rare', 'fa_new'),
            ('d', "equal-variance d' (log-linear)", 'd_rare', 'd_fill'),
            ('c', 'criterion c (log-linear)', 'c_rare', 'c_fill')]
contrasts = {}
for key, label, rare_column, ordinary_column in measures:
    contrasts[key] = contrast(rates[rare_column].to_numpy(dtype=float), rates[ordinary_column].to_numpy(dtype=float))
    contrasts[key]['label'] = label

# ============================================================================= 4. write
# ---- participant flow (test T54) ----
flow_file = os.path.join(PATHS['outputs'], 'participant_flow.csv')
with open(flow_file, 'w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['experiment', 'sessions_before_exclusion', 'excluded_total', 'excluded_more_than_5_false_presses',
                     'excluded_fewer_than_4_of_6_repeats_detected', 'excluded_both_parts_of_the_rule',
                     'excluded_not_36_test_trials', 'passed_rule', 'kept', 'passing_sessions_not_needed', 'rules'])
    writer.writerow(['rare-feature face study', len(sessions), len(excluded_sessions), too_many_false_presses,
                     too_few_repeats_detected, both_parts_failed, wrong_number_of_test_trials, len(passing_sessions),
                     len(registered_sessions), len(passing_but_not_needed),
                     'preregistered rule (AsPredicted #311640): at least 4 of the 6 one-back repeats detected and '
                     'at most 5 presses on faces that were not repeats; the first 100 passing sessions in order of '
                     'completion form the sample'])

# ---- one row per registered participant (completion order; load_order kept as a column) ----
rates_file = os.path.join(PATHS['outputs'], 'rates_by_participant.csv')
rates_by_completion = rates.copy()
rank_of = {session['participant']: session['completion_rank'] for session in sessions}
load_of = {session['participant']: session['load_order'] for session in sessions}
rates_by_completion['completion_rank'] = [rank_of[participant] for participant in rates_by_completion['participant']]
rates_by_completion = rates_by_completion.sort_values('completion_rank').reset_index(drop=True)
with open(rates_file, 'w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['participant', 'completion_rank', 'load_order',
                     'studied_rare_called_old', 'same_feature_lure_called_old',
                     'studied_ordinary_called_old', 'never_studied_called_old',
                     'n_studied_rare', 'n_same_feature_lure', 'n_studied_ordinary', 'n_never_studied',
                     'dprime_rare_vs_same_feature_lure', 'criterion_rare_vs_same_feature_lure',
                     'dprime_ordinary_vs_never_studied', 'criterion_ordinary_vs_never_studied'])
    for _, row in rates_by_completion.iterrows():
        writer.writerow([row['participant'], int(row['completion_rank']), load_of[row['participant']],
                         repr(float(row['hit_rare'])), repr(float(row['fa_rare'])),
                         repr(float(row['hit_fill'])), repr(float(row['fa_new'])),
                         int(row['n_hr']), int(row['n_fr']), int(row['n_hf']), int(row['n_fn']),
                         repr(float(row['d_rare'])), repr(float(row['c_rare'])),
                         repr(float(row['d_fill'])), repr(float(row['c_fill']))])

# ---- the four contrasts ----
contrasts_file = os.path.join(PATHS['outputs'], 'rate_contrasts.csv')
with open(contrasts_file, 'w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['measure', 'rare', 'ordinary', 'difference', 'ci_low', 'ci_high', 't', 'df', 'p',
                     'wilcoxon_p', 'cohen_dz', 'participants_positive', 'participants_tied', 'participants_negative',
                     'interval'])
    for key, _, _, _ in measures:
        result = contrasts[key]
        writer.writerow([result['label'], repr(result['rare']), repr(result['ordinary']), repr(result['difference']),
                         repr(result['ci_low']), repr(result['ci_high']), repr(result['t']), result['df'],
                         repr(result['p']), repr(result['wilcoxon_p']), repr(result['cohen_dz']),
                         result['positive'], result['tied'], result['negative'],
                         '95% percentile bootstrap over participants, 10,000 resamples, resample i drawn with '
                         'numpy default_rng(i), participants in load_order'])

# ---- the results in words ----
lines = []
lines.append('Rare-feature face study: the registered descriptive analysis')
lines.append('=' * 86)
lines.append('')
lines.append('%d sessions run; %d pass the registered exclusion rule; the first %d'
             % (len(sessions), len(passing_sessions), len(registered_sessions)))
lines.append('in order of completion are analyzed (%d passing sessions left over).'
             % len(passing_but_not_needed))
lines.append('')
lines.append('EXCLUSIONS (rule: %d test trials, at least %d of the 6 one-back repeats detected, at most %d presses'
             % (EXPECTED_TEST_TRIALS, MIN_ONEBACK_HITS, MAX_ONEBACK_FALSE_PRESSES))
lines.append('on faces that were not repeats)')
for session in excluded_sessions:
    lines.append('  %s  %s' % (session['participant'], '; '.join(session['reasons'])))
lines.append('  %d excluded: %d with more than %d false presses, %d with fewer than %d repeats detected, %d of them both'
             % (len(excluded_sessions), too_many_false_presses, MAX_ONEBACK_FALSE_PRESSES, too_few_repeats_detected,
                MIN_ONEBACK_HITS, both_parts_failed))
lines.append('')
lines.append('RARE FACES AGAINST ORDINARY FACES (%d participants)' % number_of_participants)
lines.append('  Each kind of face is scored against its own kind of lure: a rare face against the new face')
lines.append('  that shares its feature, an ordinary face against a face nobody studied.')
lines.append('')
lines.append('  %-17s %7s %8s %8s %-18s %9s %10s %11s %6s'
             % ('', 'rare', 'ordinary', 'diff', '95% CI', 't(%d)' % (number_of_participants - 1), 'p',
                'Wilcoxon p', 'd_z'))
for key in ('hit', 'fa', 'c'):
    result = contrasts[key]
    short_label = {'hit': 'hit rate', 'fa': 'false alarms', 'c': 'criterion c'}[key]
    lines.append('  %-17s %7.3f %8.3f %+8.3f [%+.3f, %+.3f] %9.2f %10.3g %11.3g %6.2f'
                 % (short_label, result['rare'], result['ordinary'], result['difference'], result['ci_low'],
                    result['ci_high'], result['t'], result['p'], result['wilcoxon_p'], result['cohen_dz']))
lines.append('')
for key in ('hit', 'fa'):
    result = contrasts[key]
    lines.append('  %s: %d participants positive, %d tied, %d reversed'
                 % (result['label'], result['positive'], result['tied'], result['negative']))
lines.append('')
lines.append('  The rare feature raises the hit rate by %+.3f and the false alarm rate by %+.3f; the criterion c'
             % (contrasts['hit']['difference'], contrasts['fa']['difference']))
lines.append('  changes by %+.3f (negative = more liberal for rare faces).' % contrasts['c']['difference'])
lines.append('  95%% CI: percentile bootstrap over participants, %s resamples, resample i drawn with'
             % '{:,}'.format(NUMBER_OF_RESAMPLES))
lines.append('  numpy default_rng(i), participants in load_order.')
lines.append('')
lines.append("  Discriminability is not in this table. Three studied and three lure trials per participant cannot")
lines.append("  identify an unequal-variance model within a participant; the registered d_a analysis fits one")
lines.append("  unequal-variance model with shared criteria to the whole test (module")
lines.append("  A3c_face_study_model_and_sdt_figS10_S11). The equal-variance d' contrast of this analysis is in")
lines.append("  outputs/rate_contrasts.csv: rare %.3f, ordinary %.3f, difference %+.3f [%+.3f, %+.3f]."
             % (contrasts['d']['rare'], contrasts['d']['ordinary'], contrasts['d']['difference'],
                contrasts['d']['ci_low'], contrasts['d']['ci_high']))
lines.append('')
lines.append('RATES BY ITEM CLASS (proportion called old, mean over participants)')
for label, column, trials_per_participant in (('rare, studied', 'hit_rare', 3), ('rare-feature lure (new)', 'fa_rare', 3),
                                              ('ordinary, studied', 'hit_fill', 15), ('never studied', 'fa_new', 15)):
    lines.append('  %-26s %8.4f   (%d trials per participant, %s in all)'
                 % (label, rates[column].mean(), trials_per_participant,
                    '{:,}'.format(trials_per_participant * number_of_participants)))
lines.append('')
lines.append('  A rare-feature lure was never seen; the only thing it shares with anything studied is the feature.')
report_text = '\n'.join(lines) + '\n'

results_file = os.path.join(PATHS['outputs'], 'results_registered_100.txt')
with open(results_file, 'w', encoding='utf-8') as handle:
    handle.write(report_text)
print(report_text)
print('wrote outputs/results_registered_100.txt, outputs/rates_by_participant.csv, outputs/rate_contrasts.csv,')
print('      outputs/participant_flow.csv')
