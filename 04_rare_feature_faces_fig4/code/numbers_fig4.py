"""
numbers_fig4.py: writes outputs/numbers.csv, one row for each of the 47 numbers and statements of
the paper this module computes (ledger ids in shared/ledger/paper_numbers.csv), with the value the
module's code computed and whether it agrees with the paper's value.

WHAT IT DOES
Reads what the other scripts of this module wrote and puts each value next to the paper's value
(shared/python/numbers_table.py judges the agreement):
  design_check.py      outputs/design_check.csv           the design counted from the data
  face_analysis.py     outputs/participant_flow.csv       111 sessions, 11 excluded, 100 kept
                       outputs/rate_contrasts.csv         rare against ordinary (hit and false alarm rates)
  memorability_check.py outputs/figure_data/fig4b_proportion_called_old.csv, figS11a_memorability.csv,
                       figS11a_comparisons.csv            Fig. 4b, SI Appendix, Fig. S11a and the two t tests
  make_figure4.py      outputs/figure_data/fig4a_design_faces.csv   the faces of Fig. 4a
The drop and the shortfall get the t-based 95% CI of the mean paired difference, computed from their
paired t tests (mean +/- t(0.975, df) x mean / t; no random draws). The preregistration number is
read from data/PREREG_SUBMITTED.md.
One statement (ledger L2-041, the closing paragraph of the direct manipulations) is about both
direct-manipulation experiments; for it this script also reads the per-item hit and false alarm
rates of the inversion experiment (03_inversion_experiment_fig3/data, through
shared/python/inversion_loader.py). Run without that folder, it says so and reports the inversion
half of L2-041 as not computed.

Match rules (shared/python/numbers_table.py): 'round' = the computed value rounds to the paper's
value at its precision; 'exact' and 'count' = equal (words such as 'Six' or '1-back' compared as
words); 'documented' = a statement with no number, shown with the computed values behind it.

INPUTS   the outputs listed above; data/face_design.json; data/PREREG_SUBMITTED.md;
         03_inversion_experiment_fig3/data/
OUTPUTS  outputs/numbers.csv
SEEDS    none
RUN      python 04_rare_feature_faces_fig4/code/numbers_fig4.py   (from the repository root, after the
         other scripts; code/run_module.py runs them in order)
"""
import sys
sys.dont_write_bytecode = True
import csv
import json
import os
import re

from scipy.stats import spearmanr, t as t_distribution

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths
import face_loader
import inversion_loader
import numbers_table

PATHS = release_paths.module_paths(__file__)
T_INTERVAL_LEVEL = 0.95     # the interval of a paired t test: the t-based 95% CI of the mean paired difference


def read_rows(folder, file_name):
    with open(os.path.join(PATHS[folder], file_name), newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


# ----------------------------------------------------------------------------- what the scripts wrote
design = {}
for row in read_rows('outputs', 'design_check.csv'):
    design[row['fact']] = row['value']
flow = read_rows('outputs', 'participant_flow.csv')[0]
contrasts = {row['measure']: row for row in read_rows('outputs', 'rate_contrasts.csv')}
rates = {row['condition']: row for row in read_rows('figure_data', 'fig4b_proportion_called_old.csv')}
memorability = {row['condition']: row for row in read_rows('figure_data', 'figS11a_memorability.csv')}
comparisons = {row['comparison']: row for row in read_rows('figure_data', 'figS11a_comparisons.csv')}
panel_a_faces = read_rows('figure_data', 'fig4a_design_faces.csv')
with open(os.path.join(PATHS['data'], 'face_design.json'), encoding='utf-8') as handle:
    stimulus_manifest = json.load(handle)['stimulus_manifest']
with open(os.path.join(PATHS['data'], 'PREREG_SUBMITTED.md'), encoding='utf-8') as handle:
    preregistration_title = handle.readline()        # '# AsPredicted preregistration #311640: ...'
preregistration_number = int(re.search(r'#(\d+)', preregistration_title).group(1))

hit_contrast = contrasts['hit rate']
false_alarm_contrast = contrasts['false alarm rate']
drop = comparisons['drop']
shortfall = comparisons['shortfall']


def number(text):
    return float(text)


def whole(text):
    return int(text)


def seconds_text(milliseconds_text):
    """'3000' -> '3 s'."""
    return '%g s' % (int(milliseconds_text) / 1000.0)


def interval_text(row):
    return '%.3f [%.3f, %.3f]' % (number(row['mean']), number(row['ci_low']), number(row['ci_high']))


def size_with_t_interval(comparison):
    """(size, low, high) of a paired difference with its t-based CI: mean +/- t(0.975, df) x SE, with
    SE = mean / t from the paired t test. A negative difference (the shortfall) is reported as its
    size, (ordinary features) minus (rare feature, carried by a new face), with the interval mirrored."""
    mean_difference = number(comparison['mean_difference'])
    critical_t = t_distribution.ppf(1 - (1 - T_INTERVAL_LEVEL) / 2, whole(comparison['df']))
    half_width = critical_t * mean_difference / number(comparison['t'])
    low, high = mean_difference - half_width, mean_difference + half_width
    if mean_difference < 0:
        return -mean_difference, -high, -low
    return mean_difference, low, high


faces_studied = whole(design['different faces studied per session'])
ordinary_studied = whole(design['ordinary faces studied per session'])
rare_studied = whole(design['rare-feature faces studied per session'])
test_trials = whole(design['test trials per session'])
repeats = whole(design['one-back repeats per session'])
pairs = whole(design['rare-feature pairs in the stimulus set'])
rare_faces = whole(design['rare-feature faces in the stimulus set'])
ordinary_pool = whole(design['ordinary faces in the stimulus set'])
kept = whole(flow['kept'])
excluded = whole(flow['excluded_total'])
sessions = whole(flow['sessions_before_exclusion'])

rates_text = ('studied rare-feature faces %s, studied ordinary faces %s, new faces with the feature %s, '
              'never-studied faces %s' % (interval_text(rates['studied_rare_feature']),
                                          interval_text(rates['studied_ordinary']),
                                          interval_text(rates['new_with_feature']),
                                          interval_text(rates['new_without'])))
memorability_text = ('rare feature, not carried %s; carried by a new face %s; ordinary features %s'
                     % (interval_text(memorability['rare_feature_not_carried']),
                        interval_text(memorability['rare_feature_carried_by_new_face']),
                        interval_text(memorability['ordinary_features'])))
drop_text = ('drop %+.4f, t(%s) = %.3f, p = %.3g' % (number(drop['mean_difference']), drop['df'], number(drop['t']),
                                                     number(drop['p'])))
shortfall_text = ('shortfall %+.4f, t(%s) = %.3f, p = %.3g'
                  % (number(shortfall['mean_difference']), shortfall['df'], number(shortfall['t']),
                     number(shortfall['p'])))
drop_size, drop_low, drop_high = size_with_t_interval(drop)
shortfall_size, shortfall_low, shortfall_high = size_with_t_interval(shortfall)
hit_text = ('hit rate %.4f (studied rare-feature faces) against %.4f (studied ordinary faces): %+.4f [%.3f, %.3f], '
            't(%s) = %.2f, p = %.3g' % (number(hit_contrast['rare']), number(hit_contrast['ordinary']),
                                        number(hit_contrast['difference']), number(hit_contrast['ci_low']),
                                        number(hit_contrast['ci_high']), hit_contrast['df'], number(hit_contrast['t']),
                                        number(hit_contrast['p'])))
false_alarm_text = ('false alarm rate %.4f (new faces with the feature) against %.4f (never-studied faces): %+.4f '
                    '[%.3f, %.3f], t(%s) = %.2f, p = %.3g'
                    % (number(false_alarm_contrast['rare']), number(false_alarm_contrast['ordinary']),
                       number(false_alarm_contrast['difference']), number(false_alarm_contrast['ci_low']),
                       number(false_alarm_contrast['ci_high']), false_alarm_contrast['df'],
                       number(false_alarm_contrast['t']), number(false_alarm_contrast['p'])))
interval_level = int(rates['studied_rare_feature']['interval'].split('%')[0])     # '95% percentile bootstrap ...'

# ----------------------------------------------------------------------------- L2-041: both experiments
# L2-041: in both direct manipulations the cost of a similar item fell mainly on false alarms.
# Inversion experiment: the 40 critical items over the 50 kept participants; mean of the item rates at
# 0 and at 8 similar items, and the Spearman correlation of each rate with the number of similar items.
inversion_trials_file = os.path.join(inversion_loader.DATA_FOLDER, 'inversion_trials.csv')
if os.path.exists(inversion_trials_file):
    inversion_items = inversion_loader.items_table()
    fewest = inversion_items[inversion_items['n_similar_items'] == inversion_items['n_similar_items'].min()]
    most = inversion_items[inversion_items['n_similar_items'] == inversion_items['n_similar_items'].max()]
    inversion_fa_correlation = spearmanr(inversion_items['n_similar_items'], inversion_items['fa_rate'])
    inversion_hit_correlation = spearmanr(inversion_items['n_similar_items'], inversion_items['hit_rate'])
    inversion_text = ('inversion experiment (40 critical items, 50 participants): from %d to %d similar items the '
                      'false alarm rate rises %.3f to %.3f (Spearman rho with the number of similar items %.3f, '
                      'p = %.2g) and the hit rate %.3f to %.3f (rho %.3f, p = %.2g)'
                      % (fewest['n_similar_items'].iloc[0], most['n_similar_items'].iloc[0], fewest['fa_rate'].mean(),
                         most['fa_rate'].mean(), inversion_fa_correlation.statistic, inversion_fa_correlation.pvalue,
                         fewest['hit_rate'].mean(), most['hit_rate'].mean(), inversion_hit_correlation.statistic,
                         inversion_hit_correlation.pvalue))
    fa_rise = most['fa_rate'].mean() - fewest['fa_rate'].mean()
    hit_rise = most['hit_rate'].mean() - fewest['hit_rate'].mean()
    inversion_holds = 'yes' if fa_rise > hit_rise else 'no'
else:
    # this module is being run without the inversion experiment's data folder next to it
    print('NOTE: 03_inversion_experiment_fig3/data/inversion_trials.csv not found; the inversion half of L2-041 '
          'is not computed (run the module inside the complete repository to compute it)')
    inversion_text = 'inversion experiment: not computed (needs 03_inversion_experiment_fig3/data)'
    inversion_holds = 'not computed'
faces_hold = 'yes' if number(false_alarm_contrast['difference']) > number(hit_contrast['difference']) else 'no'

# ----------------------------------------------------------------------------- the rows
table = numbers_table.NumbersTable(script='code/numbers_fig4.py')
ledger = numbers_table.load_ledger()


def mode_of(ledger_id):
    return ledger[ledger_id]['release_mode']


ALL_SESSIONS = 'the same in all %d sessions, outputs/design_check.csv' % sessions
STUDY_MS = '{:,}'.format(whole(design['study time per face (ms)']))
FILLER_MS = '{:,}'.format(whole(design['arithmetic filler task (ms)']))

table.add('L2-017', [faces_studied, seconds_text(design['study time per face (ms)'])], mode_of('L2-017'),
          match_type='exact', script='code/design_check.py',
          note='%d different faces studied (%d ordinary + %d rare-feature; %s); %s ms per face (task settings in '
               'data/face_design.json)' % (faces_studied, ordinary_studied, rare_studied, ALL_SESSIONS, STUDY_MS))
table.add('L2-018', [design['repeat task (each repeat directly follows its own first showing)'], test_trials],
          mode_of('L2-018'), match_type='exact', script='code/design_check.py',
          note='%d one-back repeats per session, each directly after its own first showing; arithmetic filler of %s ms; '
               '%d test trials (%s)' % (repeats, FILLER_MS, test_trials, ALL_SESSIONS))
table.add('L2-019', kept, mode_of('L2-019'), script='code/face_analysis.py',
          note='the first %d sessions passing the preregistered rule, in order of completion (%d of the %d sessions '
               'pass; outputs/participant_flow.csv)' % (kept, whole(flow['passed_rule']), sessions))
table.add('L2-020', [pairs, whole(design['rare features other than the mohawk and the beard'])], mode_of('L2-020'),
          script='code/design_check.py',
          note='rare features of the %d pairs: %s (the paper names the mohawk and the beard and calls the other four '
               'combinations); one member of each pair studied in every session: %s'
               % (pairs, design['rare features'], design['one member of every rare-feature pair studied']))
table.add('L2-021', 'the rare-feature pairs and their identity check (%s) are listed in data/face_design.json; the '
                    'matching attributes are not coded in the data' % stimulus_manifest['distinctness'],
          mode_of('L2-021'), script='code/design_check.py',
          note='the rare-feature pairs of the stimulus manifest and the identity check recorded for them; the matching '
               'attributes are a property of the stimuli; of the twelve rare-feature faces, the two of pair c1 are '
               'included (data/fig4a_faces/)')
table.add('L2-022', 'one member of every pair studied, in each of the %d sessions: %s; for the 100 registered '
                    'participants the member named _foil was the studied one in %s pair assignments'
                    % (sessions, design['one member of every rare-feature pair studied'],
                       design['registered participants: pair members studied that were the file named _foil']),
          mode_of('L2-022'), script='code/design_check.py',
          note='the studied member is drawn at random for every pair of every participant (role_swapped in '
               'data/face_trials.csv)')
table.add('L2-023', [design['share of the ordinary test faces that were studied'],
                     design['share of the ordinary test faces that were studied']], mode_of('L2-023'),
          match_type='exact', script='code/design_check.py',
          note='%s studied and %s never-studied ordinary faces at test in every session'
               % (design['test: studied ordinary faces'], design['test: never-studied ordinary faces']))
table.add('L2-024', '%s; every tested studied face shown once at study, for %s (the %d repeated faces are never '
                    'tested: %s)' % (hit_text, seconds_text(design['study time per face (ms)']), repeats,
                                     design['repeated faces are ordinary faces never tested']),
          mode_of('L2-024'), script='code/face_analysis.py',
          note='hit rates of studied rare-feature and studied ordinary faces (100 participants; 3 and 15 trials each), '
               'their mean paired difference with the registered participant bootstrap interval (resample i drawn '
               'with default_rng(i), participants in load_order), and the paired t test; study time and repeats '
               'from outputs/design_check.csv')
table.add('L2-025', '%s; the hit rate increase is %+.4f' % (false_alarm_text, number(hit_contrast['difference'])),
          mode_of('L2-025'), script='code/face_analysis.py',
          note='false alarm rates of new faces carrying a studied face\'s rare feature and of never-studied ordinary '
               'faces, their mean paired difference with the registered participant bootstrap interval and paired t '
               'test, and the hit rate increase for comparison')
table.add('L2-026', [drop_size, drop_low, drop_high, whole(drop['df']), abs(number(drop['t'])), number(drop['p']),
                     abs(number(drop['cohen_dz']))],
          mode_of('L2-026'), script='code/memorability_check.py',
          note='%s, d_z %.3f (paired t test over the 100 participants); t-based 95%% CI %.4f to %.4f (mean +/- '
               't(0.975, %s) x mean / t)' % (drop_text, number(drop['cohen_dz']), drop_low, drop_high, drop['df']))
table.add('L2-027', [shortfall_size, shortfall_low, shortfall_high, whole(shortfall['df']), abs(number(shortfall['t'])),
                     number(shortfall['p']), abs(number(shortfall['cohen_dz']))],
          mode_of('L2-027'), script='code/memorability_check.py',
          note='%s, d_z %.3f; computed value: the size of the mean paired difference, (ordinary features) minus '
               '(rare feature, carried by a new face), its t-based 95%% CI (mean +/- t(0.975, %s) x mean / t), df, '
               'absolute t, two-sided p and absolute d_z of the paired t test'
               % (shortfall_text, number(shortfall['cohen_dz']), shortfall['df']))
not_carried_minus_ordinary = (number(memorability['rare_feature_not_carried']['mean'])
                              - number(memorability['ordinary_features']['mean']))
table.add('L2-121', '%s; not carried minus ordinary features %+.4f; %s'
                    % (memorability_text, not_carried_minus_ordinary, shortfall_text),
          mode_of('L2-121'), script='code/memorability_check.py',
          note='memorability (hit rate minus false alarm rate, SI Appendix, Fig. S11a) of the rare faces is above that '
               'of ordinary faces while no new face carries their feature, and below it once a new face does (the '
               'shortfall, L2-027)')
table.add('L2-127', int(round(100 * T_INTERVAL_LEVEL)), mode_of('L2-127'), script='code/numbers_fig4.py',
          note="level of the t-based CI of the two paired t tests (the drop, L2-026, and the shortfall, L2-027): mean "
               "+/- t(0.975, df) x mean / t, reported with Cohen's d_z (mean paired difference / SD of the paired "
               "differences)")
faces_in_panel_a = '; '.join('%s: %s (%s)' % (row['stage'], row['file'].split('/')[-1], row['label'])
                             for row in panel_a_faces)
table.add('L2-030', 'Fig. 4a drawn from the five synthetic faces in data/fig4a_faces/: %s; arithmetic filler of %s ms, '
                    'drawn with the example problem 47 + 68' % (faces_in_panel_a, FILLER_MS),
          mode_of('L2-030'), script='code/make_figure4.py',
          note='outputs/figures/Fig4ab_rare_feature_faces.png, panel a: the faces and labels listed in '
               'outputs/figure_data/fig4a_design_faces.csv and the example problem of the 60 s arithmetic task')
role_of_panel_a_face = {}
for row in panel_a_faces:
    role_of_panel_a_face[row['file'].split('/')[-1]] = row['role_in_stimulus_manifest'] or 'not in the stimulus manifest'
table.add('L2-126', 'the %d faces of Fig. 4a are faces of the stimulus set (%s): %s'
                    % (len(role_of_panel_a_face), stimulus_manifest['note'].split(':')[0],
                       '; '.join('%s, %s' % (file_name, role) for file_name, role in role_of_panel_a_face.items())),
          mode_of('L2-126'), script='code/make_figure4.py',
          note='each face drawn in Fig. 4a with its role in the stimulus manifest of data/face_design.json '
               '(outputs/figure_data/fig4a_design_faces.csv)')
table.add('L2-031', rates_text, mode_of('L2-031'), script='code/memorability_check.py',
          note='Fig. 4b (outputs/figure_data/fig4b_proportion_called_old.csv): mean over participants, 95% percentile '
               'bootstrap over participants (numpy default_rng(0), one generator for the panel, bars drawn in '
               'plotted order)')
table.add('L2-032', '%s; braces %s (p = %.2g) and %s (p = %.2g)' % (memorability_text, drop['stars'], number(drop['p']),
                                                                    shortfall['stars'], number(shortfall['p'])),
          mode_of('L2-032'), script='code/memorability_check.py',
          note='SI Appendix, Fig. S11a (outputs/figure_data/figS11a_memorability.csv): mean over participants of hit '
               'rate minus false alarm rate, 95% percentile bootstrap over participants (numpy default_rng(0), one '
               'generator for the panel, points drawn in plotted order), and the p values of the two braces')
table.add('L2-033', [faces_studied, rare_studied, test_trials, kept], mode_of('L2-033'), script='code/design_check.py',
          note='%d faces studied, %d of them with a rare feature, %d test trials (%s); %d participants '
               '(face_analysis.py)' % (faces_studied, rare_studied, test_trials, ALL_SESSIONS, kept))
table.add('L2-034', [number(rates['studied_ordinary']['mean']), number(rates['studied_rare_feature']['mean'])],
          mode_of('L2-034'), script='code/memorability_check.py',
          note='hit rates: studied ordinary faces, studied rare-feature faces (mean over the 100 participants; the '
               'same as pooled over the 1,500 and 300 trials, since every participant has 15 and 3)')
table.add('L2-035', [number(rates['new_without']['mean']), number(rates['new_with_feature']['mean'])],
          mode_of('L2-035'), script='code/memorability_check.py',
          note='false alarm rates: never-studied faces, new faces carrying a studied face\'s rare feature (mean over '
               'the 100 participants; the same as pooled over the 1,500 and 300 trials)')
for ledger_id, condition in (('L2-036', 'rare_feature_not_carried'), ('L2-037', 'rare_feature_carried_by_new_face'),
                             ('L2-038', 'ordinary_features')):
    row = memorability[condition]
    note = ('%s: mean over the 100 participants and the 95%% interval printed in SI Appendix, Section 3 and drawn in '
            'Fig. S11a (numpy default_rng(0), one generator for the panel, points drawn in plotted order)' % row['label'])
    table.add(ledger_id, [number(row['mean']), number(row['ci_low']), number(row['ci_high'])], mode_of(ledger_id),
              script='code/memorability_check.py', note=note)
table.add('L2-039', interval_level, mode_of('L2-039'), script='code/memorability_check.py',
          note='level of the error bars of Fig. 4b and SI Appendix, Fig. S11a: percentile bootstrap over participants, '
               '2.5th and 97.5th percentiles of 10,000 resampled means, numpy default_rng(0)')
table.add('L2-040', [number(drop['p']), number(shortfall['p'])], mode_of('L2-040'), script='code/memorability_check.py',
          note='the two braces: %s (%s); %s (%s)' % (drop_text, drop['stars'], shortfall_text, shortfall['stars']))
table.add('L2-041', '%s; face study: the rare feature raises false alarms by %+.3f and hits by %+.3f'
                    % (inversion_text, number(false_alarm_contrast['difference']), number(hit_contrast['difference'])),
          mode_of('L2-041'), script='code/numbers_fig4.py',
          note='false alarms rise more than hits with a similar item: inversion experiment %s (item rates at the '
               'smallest and the largest number of similar items, from 03_inversion_experiment_fig3/data through '
               'shared/python/inversion_loader.py); face study %s (the rare-feature contrasts)'
               % (inversion_holds, faces_hold))
for ledger_id in ('L2-122', 'L2-123', 'L4-003'):         # abstract, introduction, General Discussion
    table.add(ledger_id, '%s: more memorable than ordinary faces only while no new face carries the feature'
                         % memorability_text,
              mode_of(ledger_id), script='code/memorability_check.py',
              note='memorability in the three conditions of SI Appendix, Fig. S11a (L2-032) with the paired t tests of '
                   'the drop and the shortfall (L2-026, L2-027): %s; %s' % (drop_text, shortfall_text))
table.add('L4-011', 'face study: the rare feature raises hits by %+.3f and false alarms by %+.3f; memorability %.3f '
                    'without and %.3f with a new face carrying the feature (%s)'
                    % (number(hit_contrast['difference']), number(false_alarm_contrast['difference']),
                       number(memorability['rare_feature_not_carried']['mean']),
                       number(memorability['rare_feature_carried_by_new_face']['mean']), drop_text),
          mode_of('L4-011'), script='code/memorability_check.py',
          note='face study: the rare feature\'s effect on hits and on false alarms, and memorability without and with '
               'a new face carrying the feature')
table.add('L4-019', [number(rates['studied_ordinary']['mean']), number(rates['studied_rare_feature']['mean'])],
          mode_of('L4-019'), script='code/memorability_check.py', note='hit rates, as L2-034')
table.add('L4-020', [number(rates['new_without']['mean']), number(rates['new_with_feature']['mean'])],
          mode_of('L4-020'), script='code/memorability_check.py', note='false alarm rates, as L2-035')
table.add('L2-124', 'proportion called old: studied rare-feature faces %.4f against studied ordinary faces %.4f; new faces '
                    'carrying a studied face\'s rare feature %.4f against never-studied faces %.4f'
                    % (number(rates['studied_rare_feature']['mean']), number(rates['studied_ordinary']['mean']),
                       number(rates['new_with_feature']['mean']), number(rates['new_without']['mean'])),
          mode_of('L2-124'), script='code/memorability_check.py',
          note='Fig. 4b (outputs/figure_data/fig4b_proportion_called_old.csv): a face with a rare feature is called old '
               'more often than an ordinary face, and a new face carrying a studied face\'s rare feature more often '
               'than a never-studied face (L2-024, L2-025)')
table.add('L5-035', [rare_faces, pairs], mode_of('L5-035'), script='code/design_check.py',
          note='%d rare-feature faces in %d pairs (stimulus manifest in data/face_design.json); %s different '
               'rare-feature faces appear in the data' % (rare_faces, pairs,
                                                         design['different rare-feature faces used over all sessions']))
table.add('L5-036', ordinary_pool, mode_of('L5-036'), script='code/design_check.py',
          note='%d ordinary faces in the stimulus manifest; the same %s faces appear in the data (the 6 practice '
               'faces are separate)' % (ordinary_pool, design['different ordinary faces used over all sessions']))
table.add('L2-125', preregistration_number, mode_of('L2-125'), script='code/numbers_fig4.py',
          note='the AsPredicted number in the title of the preregistration as submitted (data/PREREG_SUBMITTED.md)')
table.add('L5-037', [faces_studied, ordinary_studied], mode_of('L5-037'), script='code/design_check.py',
          note='%d faces studied: one member of each of the %d pairs and %d ordinary faces (%s)'
               % (faces_studied, pairs, ordinary_studied, ALL_SESSIONS))
table.add('L5-038', [seconds_text(design['study time per face (ms)']), '%s ms' % design['gap between study faces (ms)']],
          mode_of('L5-038'), match_type='exact', script='code/design_check.py',
          note='task settings (data/face_design.json): study_ms %s, study_isi_ms %s'
               % (STUDY_MS, design['gap between study faces (ms)']))
table.add('L5-039', repeats, mode_of('L5-039'), script='code/design_check.py',
          note='%d one-back repeats per session; the repeated faces are ordinary faces that are never tested: %s (%s)'
               % (repeats, design['repeated faces are ordinary faces never tested'], ALL_SESSIONS))
table.add('L5-040', seconds_text(design['arithmetic filler task (ms)']), mode_of('L5-040'), match_type='exact',
          script='code/design_check.py',
          note='task settings (data/face_design.json): filler_task_ms %s' % FILLER_MS)
table.add('L5-041', [test_trials, whole(design['confidence scale points (one rating after every test trial)'])],
          mode_of('L5-041'), script='code/design_check.py',
          note='%d test trials, each followed by a confidence rating from 1 to 4 (%s)' % (test_trials, ALL_SESSIONS))
table.add('L5-042', [whole(design['test: studied rare-feature faces']),
                     whole(design['test: new faces carrying a studied rare feature']),
                     whole(design['test: studied ordinary faces']), whole(design['test: never-studied ordinary faces'])],
          mode_of('L5-042'), script='code/design_check.py',
          note='per session: studied rare-feature faces, new faces with a studied face\'s feature, studied ordinary, '
               'never-studied ordinary (%s); pairs tested with the studied face: c1, c3, c5 in %s registered '
               'sessions, otherwise c2, c4, c6 (drawn at random for each participant)'
               % (ALL_SESSIONS, design['registered participants tested old on pairs c1, c3, c5 (others: c2, c4, c6)']))
table.add('L5-043', 'per-participant hit rate minus false alarm rate: means %.4f / %.4f / %.4f; %s; %s (scipy ttest_rel, '
                    'paired)' % (number(memorability['rare_feature_not_carried']['mean']),
                                 number(memorability['rare_feature_carried_by_new_face']['mean']),
                                 number(memorability['ordinary_features']['mean']), drop_text, shortfall_text),
          mode_of('L5-043'), script='code/memorability_check.py',
          note='per-participant hit rate minus false alarm rate in the three conditions of SI Appendix, Fig. S11a and '
               'the two paired t tests; the false alarm rate is taken from never-studied ordinary faces or from the '
               'new members of the rare-feature pairs')
table.add('L5-068', kept, mode_of('L5-068'), script='code/face_analysis.py',
          note='the first %d passing sessions in order of completion' % kept)
table.add('L5-069', [face_loader.MIN_ONEBACK_HITS, repeats, face_loader.MAX_ONEBACK_FA], mode_of('L5-069'),
          script='code/face_analysis.py',
          note='the rule the code applies: at least %d of the %d repeats detected and at most %d presses on faces '
               'that were not repeats (shared/python/face_loader.py); on the %d sessions it keeps %d and excludes %d'
               % (face_loader.MIN_ONEBACK_HITS, repeats, face_loader.MAX_ONEBACK_FA, sessions, kept, excluded))
table.add('L5-070', excluded, mode_of('L5-070'), script='code/face_analysis.py',
          note='%d of %d sessions excluded: %s with more than 5 false presses, %s with fewer than 4 of 6 repeats '
               'detected, %s of them both' % (excluded, sessions, flow['excluded_more_than_5_false_presses'],
                                             flow['excluded_fewer_than_4_of_6_repeats_detected'],
                                             flow['excluded_both_parts_of_the_rule']))
table.add('ADD-011b', [kept, excluded], mode_of('ADD-011b'), script='code/face_analysis.py',
          note='participant flow (outputs/participant_flow.csv): %d sessions before exclusion; %d excluded (%s with '
               'more than 5 false presses, %s with fewer than 4 of 6 repeats detected, %s both); %d kept'
               % (sessions, excluded, flow['excluded_more_than_5_false_presses'],
                  flow['excluded_fewer_than_4_of_6_repeats_detected'], flow['excluded_both_parts_of_the_rule'], kept))

table.write(PATHS['outputs'])
problems = [row for row in table.rows if row['match'] == 'no']
for row in table.rows:
    print('  %-9s %-11s %-10s %s' % (row['ledger_id'], row['match'], row['match_type'], row['computed_value'][:90]))
if problems:
    print('NOT MATCHING: %s' % ', '.join(row['ledger_id'] for row in problems))
