"""
stats_fig4.py: the statistics of the rare-feature face study (tests T07, T08, T09 and T54 of
statistics/README.md).

WHAT IT DOES
For the 100 registered participants (the first 100 sessions passing the preregistered rule):
  T07  the drop in memorability (hit rate minus false alarm rate) of the rare-feature faces when a
       new face carries their feature: (rare feature, not carried) minus (rare feature, carried by a
       new face). Numerically the false alarm contrast (new faces with the feature minus
       never-studied faces), because the hit rate of the rare faces is shared.
  T08  the shortfall: (rare feature, carried by a new face) minus (ordinary features). The
       preregistration registers the matching contrast on d_a as an estimate with a bootstrap
       interval (module A3c).
  T09  the two rate contrasts of Fig. 4b: hit rate, studied rare-feature faces minus studied
       ordinary faces (preregistered prediction 1); false alarm rate, new faces carrying a studied
       face's rare feature minus never-studied faces (preregistered prediction 2).
Every contrast gets the paired t test: t(99), exact two-sided p, Cohen's d_z (mean paired difference
/ SD of the paired differences) and the t-based 95% CI of the mean difference. The two T09 contrasts
also get the registered participant bootstrap of predictions 1 and 2 (10,000 resamples; resample i
drawn with default_rng(i), participants in load_order).
The condition means of Fig. 4b and SI Appendix, Fig. S11a get the interval drawn in the figure
(seed 0, one generator per panel, conditions in plotted order; see memorability_check.py).
T54: the participant flow (111 sessions, 11 excluded by the registered rule, 100 kept).
No correction for multiple comparisons is applied (statistics/README.md).

INPUTS   data/face_trials.csv, data/face_participants.csv (through shared/python/face_loader.py);
         outputs/figure_data/fig4b_proportion_called_old.csv and figS11a_memorability.csv (the figure
         intervals, written by memorability_check.py); outputs/participant_flow.csv (written by
         face_analysis.py)
OUTPUTS  outputs/tests.csv
SEEDS    i = 0 to 9,999 (one generator per resample) for the registered bootstrap; the figure
         intervals (seed 0) come from memorability_check.py
RUN      python 04_rare_feature_faces_fig4/code/stats_fig4.py   (from the repository root; a few seconds)
"""
import sys
sys.dont_write_bytecode = True
import csv
import os

import numpy as np
from scipy.stats import t as t_distribution, ttest_rel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths
import face_loader
import bootstrap
import tests_table

PATHS = release_paths.module_paths(__file__)
PAPER_SEED = 0
NUMBER_OF_RESAMPLES = 10000
SCRIPT = 'code/stats_fig4.py'
print('stats_fig4.py: registered bootstrap, %s resamples' % '{:,}'.format(NUMBER_OF_RESAMPLES))


def read_rows(folder, file_name):
    with open(os.path.join(PATHS[folder], file_name), newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def t_based_interval(differences, level=0.95):
    """Mean of the paired differences plus or minus t(df) x standard error."""
    number_of_pairs = len(differences)
    standard_error = np.std(differences, ddof=1) / np.sqrt(number_of_pairs)
    critical_t = t_distribution.ppf(1 - (1 - level) / 2, number_of_pairs - 1)
    return float(np.mean(differences) - critical_t * standard_error), float(np.mean(differences) + critical_t * standard_error)


def resamples_at_or_below_zero_text(resampled_means):
    below = int(np.sum(resampled_means <= 0))
    above = int(np.sum(resampled_means >= 0))
    return '%d of %d resampled mean differences at or below zero, %d at or above zero' % (
        below, len(resampled_means), above)


# ----------------------------------------------------------------------------- data
by_completion = face_loader.participant_rates(order='completion')     # S-id order
by_load_order = face_loader.participant_rates(order='load_order')     # the order of the registered bootstrap
number_of_participants = len(by_completion)

studied_rare = by_completion['hit_rare'].to_numpy(dtype=float)
same_feature_lure = by_completion['fa_rare'].to_numpy(dtype=float)
studied_ordinary = by_completion['hit_fill'].to_numpy(dtype=float)
never_studied = by_completion['fa_new'].to_numpy(dtype=float)
rare_not_carried = studied_rare - never_studied
rare_carried_by_new_face = studied_rare - same_feature_lure
ordinary_features = studied_ordinary - never_studied

figure_4b_rows = {row['condition']: row for row in read_rows('figure_data', 'fig4b_proportion_called_old.csv')}
figure_s11a_rows = {row['condition']: row for row in read_rows('figure_data', 'figS11a_memorability.csv')}

tests = tests_table.TestsTable(script=SCRIPT)
UNIT = 'participants (the 100 registered)'


def add_paired_contrast(test_id, ledger_ids, location, description, first, second, registered_bootstrap=False,
                        t_note=''):
    """The rows of one paired contrast: the paired t test and, when registered_bootstrap is True, the
    registered bootstrap of predictions 1 and 2 (resample i drawn with default_rng(i), participants in
    load_order). first and second: per-participant values in completion order.
    t_note: what the preregistration says about this contrast, added to the row text."""
    differences = first - second
    mean_difference = float(np.mean(differences))
    t_value, p_value = ttest_rel(first, second)
    t_value = float(t_value)
    p_value = float(p_value)
    cohen_dz = mean_difference / float(np.std(differences, ddof=1))
    t_low, t_high = t_based_interval(differences)

    tests.add(test_id=test_id, ledger_ids=ledger_ids, location=location,
              test='%s: paired t test over participants (scipy ttest_rel), two-sided; mean paired difference %+.4f; '
                   'CI = t-based 95%% CI of the mean paired difference%s' % (description, mean_difference, t_note),
              statistic_name='t', statistic=t_value, n=number_of_participants, unit=UNIT,
              df=number_of_participants - 1, p=p_value, p_type='parametric two-sided',
              effect_size_name="Cohen's d_z", effect_size=cohen_dz, ci_low=t_low, ci_high=t_high,
              ci_method='t-based 95% CI of the mean paired difference', resampling_unit='', n_resamples='', seed='')

    if registered_bootstrap:
        # the registered bootstrap: participants in load_order, resample i drawn with default_rng(i)
        position_in_load_order = {participant: index for index, participant in enumerate(by_load_order['participant'])}
        ordered_differences = np.empty(number_of_participants)
        for index, participant in enumerate(by_completion['participant']):
            ordered_differences[position_in_load_order[participant]] = differences[index]
        resampled = bootstrap.seeded_per_resample_mean_bootstrap(ordered_differences, NUMBER_OF_RESAMPLES)
        low, high = np.percentile(resampled, [2.5, 97.5])
        tests.add(test_id=test_id, ledger_ids=ledger_ids, location=location,
                  test='%s: mean paired difference with the participant bootstrap given in ci_method; %s'
                       % (description, resamples_at_or_below_zero_text(resampled)),
                  statistic_name='mean paired difference', statistic=mean_difference, n=number_of_participants,
                  unit=UNIT, df='', p='', p_type='none', effect_size_name='difference in proportion',
                  effect_size=mean_difference, ci_low=float(low), ci_high=float(high),
                  ci_method=('registered percentile bootstrap (resample i drawn with numpy default_rng(i), '
                             'participants in load_order)'),
                  resampling_unit='participants', n_resamples=NUMBER_OF_RESAMPLES,
                  seed='i = 0 to 9999 (one per resample)')
    return dict(mean_difference=mean_difference, t=t_value, p=p_value, dz=cohen_dz)


def add_condition_mean(test_id, ledger_ids, location, description, values, figure_row):
    """The row of a plotted condition mean, with the interval drawn in the figure."""
    mean_value = float(np.mean(values))
    figure_low = float(figure_row['ci_low'])
    figure_high = float(figure_row['ci_high'])
    tests.add(test_id=test_id, ledger_ids=ledger_ids, location=location,
              test='%s: mean over participants with the interval drawn in the figure (one numpy default_rng(0) '
                   'generator per panel, conditions drawn in plotted order, participants in order of completion; '
                   'see memorability_check.py)' % description,
              statistic_name='mean', statistic=mean_value, n=number_of_participants, unit=UNIT, df='', p='',
              p_type='none', effect_size_name='', effect_size='', ci_low=figure_low, ci_high=figure_high,
              ci_method='percentile bootstrap (numpy default_rng(0), one generator per panel, conditions in plotted '
                        'order)', resampling_unit='participants',
              n_resamples=NUMBER_OF_RESAMPLES, seed=PAPER_SEED)


MEMORABILITY_LOCATION = ('SI Appendix, Section 3, The Preregistered Discriminability Analysis of the Face Study; '
                         'SI Appendix, Fig. S11a')
RATE_CONTRAST_LOCATION = 'SI Appendix, Section 3, The Model Predicted Each Preregistered Face Result; SI Appendix, Fig. S10a'

# ----------------------------------------------------------------------------- T07
drop = add_paired_contrast(
    'T07', 'L2-026;L2-040;L2-121;L2-122;L2-123;L2-127;L4-003;L5-043',
    "A Rare Feature's Advantage Depends on What the Face Is Tested Against; SI Appendix, Fig. S11a (first brace)",
    'Drop in memorability (hit rate minus false alarm rate) of the studied rare-feature faces when a new face '
    'carries their feature: (rare feature, not carried) minus (rare feature, carried by a new face); numerically '
    'the false alarm contrast of T09 (new faces with the feature minus never-studied faces), because the hit rate '
    'is shared',
    rare_not_carried, rare_carried_by_new_face,
    t_note='; the preregistration tests the same difference as the false alarm contrast (prediction 2, T09) and the '
           'matching contrast on d_a as prediction 3 (module A3c)')
for condition, ledger_ids, description, values in (
        ('rare_feature_not_carried', 'L2-032;L2-036', 'SI Appendix, Fig. S11a, rare feature, not carried (rare faces '
         'against never-studied faces)', rare_not_carried),
        ('rare_feature_carried_by_new_face', 'L2-032;L2-037', 'SI Appendix, Fig. S11a, rare feature, carried by a new '
         'face (rare faces against the new faces with their feature)', rare_carried_by_new_face)):
    add_condition_mean('T07', ledger_ids, MEMORABILITY_LOCATION, description, values, figure_s11a_rows[condition])

# ----------------------------------------------------------------------------- T08
shortfall = add_paired_contrast(
    'T08', 'L2-027;L2-040;L2-121;L2-122;L2-123;L2-127;L4-003;L5-043',
    "A Rare Feature's Advantage Depends on What the Face Is Tested Against; SI Appendix, Fig. S11a (second brace)",
    "Shortfall (the feature's own effect): (rare feature, carried by a new face) minus (ordinary features); the "
    'paper reports the absolute values of the difference, its interval, t and d_z',
    rare_carried_by_new_face, ordinary_features,
    t_note='; the preregistration registers the matching contrast on d_a as an estimate with a bootstrap interval '
           '(module A3c)')
add_condition_mean('T08', 'L2-032;L2-038', MEMORABILITY_LOCATION,
                   'SI Appendix, Fig. S11a, ordinary features (ordinary faces against never-studied faces)',
                   ordinary_features, figure_s11a_rows['ordinary_features'])

# ----------------------------------------------------------------------------- T09
hits = add_paired_contrast(
    'T09', 'L2-024;L2-031;L2-034;L2-124;L4-019', RATE_CONTRAST_LOCATION,
    'Hit rate, studied rare-feature faces minus studied ordinary faces (preregistered prediction 1)',
    studied_rare, studied_ordinary, registered_bootstrap=True,
    t_note='; preregistered prediction 1 (the registered participant bootstrap: next row)')
false_alarms = add_paired_contrast(
    'T09', 'L2-025;L2-031;L2-035;L2-124;L4-020', RATE_CONTRAST_LOCATION,
    'False alarm rate, new faces carrying a studied face\'s rare feature minus never-studied ordinary faces '
    '(preregistered prediction 2)',
    same_feature_lure, never_studied, registered_bootstrap=True,
    t_note='; numerically the same test as the drop (T07); preregistered prediction 2 (the registered participant '
           'bootstrap: next row)')
for condition, ledger_ids, description, values in (
        ('studied_rare_feature', 'L2-031;L2-034;L4-019', 'Fig. 4b, studied faces carrying a rare feature, proportion '
         'called old (hit rate)', studied_rare),
        ('studied_ordinary', 'L2-031;L2-034;L4-019', 'Fig. 4b, studied ordinary faces, proportion called old (hit '
         'rate)', studied_ordinary),
        ('new_with_feature', 'L2-031;L2-035;L4-020', 'Fig. 4b, new faces carrying a studied face\'s rare feature, '
         'proportion called old (false alarm rate)', same_feature_lure),
        ('new_without', 'L2-031;L2-035;L4-020', 'Fig. 4b, never-studied ordinary faces, proportion called old '
         '(false alarm rate)', never_studied)):
    figure_row = figure_4b_rows[condition]
    mean_value = float(np.mean(values))
    tests.add(test_id='T09', ledger_ids=ledger_ids, location='Fig. 4 caption, panel b',
              test='%s: mean over participants with the interval drawn in the figure (one numpy default_rng(0) '
                   'generator for the panel, bars drawn in plotted order, participants in order of completion; the '
                   'paper reports the rate, the interval is the error bar)' % description,
              statistic_name='mean', statistic=mean_value, n=number_of_participants, unit=UNIT, df='', p='',
              p_type='none', effect_size_name='', effect_size='', ci_low=float(figure_row['ci_low']),
              ci_high=float(figure_row['ci_high']),
              ci_method='percentile bootstrap (numpy default_rng(0), one generator per panel, conditions in plotted '
                        'order)',
              resampling_unit='participants', n_resamples=NUMBER_OF_RESAMPLES, seed=PAPER_SEED)

# ----------------------------------------------------------------------------- T54
flow = read_rows('outputs', 'participant_flow.csv')[0]
tests.add(test_id='T54', ledger_ids='ADD-011b;L2-019;L5-068;L5-069;L5-070',
          location='Methods, Experimental Procedures and Analysis',
          test='Participant flow, rare-feature face study: %s sessions run; '
               'excluded %s by the preregistered rule (more than 5 presses on faces that were not repeats %s, fewer '
               'than 4 of the 6 one-back repeats detected %s, both %s); kept %s (the first 100 passing sessions in '
               'order of completion; %s passing sessions left over)'
               % (flow['sessions_before_exclusion'], flow['excluded_total'], flow['excluded_more_than_5_false_presses'],
                  flow['excluded_fewer_than_4_of_6_repeats_detected'], flow['excluded_both_parts_of_the_rule'],
                  flow['kept'], flow['passing_sessions_not_needed']),
          statistic_name='participants kept', statistic=int(flow['kept']), n=int(flow['sessions_before_exclusion']),
          unit='sessions before exclusion', df='', p='', p_type='none', effect_size_name='', effect_size='',
          ci_low='', ci_high='', ci_method='none', resampling_unit='', n_resamples='', seed='')

tests.write(PATHS['outputs'])

# ----------------------------------------------------------------------------- summary
print('')
for label, result in (('T07 drop', drop), ('T08 shortfall', shortfall), ('T09 hits', hits),
                      ('T09 false alarms', false_alarms)):
    print('  %-17s %+.4f  t(%d) = %.3f, p = %.3g, d_z = %.3f'
          % (label, result['mean_difference'], number_of_participants - 1, result['t'], result['p'], result['dz']))
