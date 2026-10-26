"""
memorability_check.py: the numbers of Fig. 4b and SI Appendix, Fig. S11a, and the two comparisons
reported in the section "A Rare Feature's Advantage Depends on What the Face Is Tested Against"
(rare-feature face study, 100 registered participants).

WHAT IT DOES
1. For each registered participant (in order of completion), the proportion called old in the four
   kinds of test item: studied rare-feature faces (3 trials), new faces carrying a studied face's
   rare feature (3), studied ordinary faces (15) and never-studied ordinary faces (15).
2. Memorability (corrected recognition, hit rate minus false alarm rate) in the three conditions of
   SI Appendix, Fig. S11a:
     rare feature, not carried        = hit rate, rare faces  minus  false alarm rate, never-studied faces
     rare feature, carried by a new   = hit rate, rare faces  minus  false alarm rate, new faces with the feature
       face
     ordinary features                = hit rate, ordinary faces  minus  false alarm rate, never-studied faces
3. The two comparisons, paired t tests over participants (scipy ttest_rel), with Cohen's d_z:
     drop (lure-set effect)  = (rare, not carried) minus (rare, carried by a new face)
                               (numerically the false alarm contrast, because the hit rate is shared)
     shortfall (the feature's own effect) = (rare, carried by a new face) minus (ordinary features)
4. 95% percentile bootstrap intervals over participants, all from ONE routine
   (participant_bootstrap_intervals below): one numpy generator, default_rng(seed), draws 10,000
   resamples of the 100 participants (generator.integers(0, 100, 100) per resample) for the first
   condition, then 10,000 for the next condition, and so on; the interval is the 2.5th and 97.5th
   percentile of the resampled means (numpy percentile).
     Figure intervals (the error bars of both panels and the intervals printed in SI Appendix,
       Section 3): the routine is called once per panel, seed 0, with the panel's conditions in the
       order they are plotted (Fig. 4b: studied rare, studied ordinary, new with feature, new
       without; SI Appendix, Fig. S11a: not carried, carried by a new face, ordinary).

INPUTS   data/face_trials.csv, data/face_participants.csv (through shared/python/face_loader.py)
OUTPUTS  outputs/memorability_hit_minus_fa.txt                      the results in words
         outputs/figure_data/fig4b_proportion_called_old.csv        Fig. 4b bars and error bars
         outputs/figure_data/figS11a_memorability.csv               SI Appendix, Fig. S11a points and error bars
         outputs/figure_data/figS11a_comparisons.csv                the two braces of SI Appendix, Fig. S11a
SEEDS    numpy default_rng(0), as described in step 4
RUN      python 04_rare_feature_faces_fig4/code/memorability_check.py   (from the repository root; about 2 s)
"""
import sys
sys.dont_write_bytecode = True
import csv
import os

import numpy as np
from scipy.stats import ttest_rel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths
import face_loader

PATHS = release_paths.module_paths(__file__)
PAPER_SEED = 0
NUMBER_OF_RESAMPLES = 10000


def participant_bootstrap_intervals(values_per_condition, seed=PAPER_SEED, number_of_resamples=NUMBER_OF_RESAMPLES):
    """95% percentile bootstrap interval of the mean over participants, for each condition in turn.

    values_per_condition: a list with one array per condition (one value per participant, the same
    participants in the same order in every array). One generator, default_rng(seed), draws all the
    resamples: first the resamples of the first condition, then those of the second, and so on.
    Returns a list of (low, high), one per condition."""
    generator = np.random.default_rng(seed)
    intervals = []
    for values in values_per_condition:
        number_of_participants = len(values)
        resampled_means = np.empty(number_of_resamples)
        for resample in range(number_of_resamples):
            chosen_participants = generator.integers(0, number_of_participants, number_of_participants)
            resampled_means[resample] = np.mean(values[chosen_participants])
        low, high = np.percentile(resampled_means, [2.5, 97.5])
        intervals.append((float(low), float(high)))
    return intervals


def stars_for(p_value):
    """The brace label of the figure: *** p < 0.001, ** p < 0.01, * p < 0.05, otherwise n.s."""
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return 'n.s.'


# ----------------------------------------------------------------------------- 1. rates per participant
participant_ids = face_loader.registered_participants(order='completion')
test_trials = face_loader.load_trials('recognition')

studied_rare = []          # hit rate, rare-feature faces
same_feature_lure = []     # false alarm rate, new faces carrying a studied face's rare feature
studied_ordinary = []      # hit rate, ordinary faces
never_studied = []         # false alarm rate, never-studied ordinary faces
for participant in participant_ids:
    own_trials = test_trials[test_trials['participant'] == participant]
    said_old = {'studied_rare': [], 'same_feature_lure': [], 'studied_ordinary': [], 'never_studied': []}
    for _, trial in own_trials.iterrows():
        if trial['condition'] == 'rare':
            if trial['test_type'] == 'old':
                kind = 'studied_rare'
            else:
                kind = 'same_feature_lure'
        else:
            if trial['test_type'] == 'old':
                kind = 'studied_ordinary'
            else:
                kind = 'never_studied'
        said_old[kind].append(float(trial['said_old']))
    studied_rare.append(np.mean(said_old['studied_rare']))
    same_feature_lure.append(np.mean(said_old['same_feature_lure']))
    studied_ordinary.append(np.mean(said_old['studied_ordinary']))
    never_studied.append(np.mean(said_old['never_studied']))
studied_rare = np.array(studied_rare)
same_feature_lure = np.array(same_feature_lure)
studied_ordinary = np.array(studied_ordinary)
never_studied = np.array(never_studied)
number_of_participants = len(participant_ids)

# the same rates as the shared loader (and so as face_analysis.py)
loader_rates = face_loader.participant_rates(order='completion')
for mine, column in ((studied_rare, 'hit_rare'), (same_feature_lure, 'fa_rare'),
                     (studied_ordinary, 'hit_fill'), (never_studied, 'fa_new')):
    if not np.array_equal(mine, loader_rates[column].to_numpy(dtype=float)):
        raise SystemExit('rates differ from face_loader.participant_rates(): %s' % column)

# ----------------------------------------------------------------------------- 2. memorability
rare_not_carried = studied_rare - never_studied
rare_carried_by_new_face = studied_rare - same_feature_lure
ordinary_features = studied_ordinary - never_studied

# ----------------------------------------------------------------------------- 3. the two comparisons
comparisons = []
for name, label, first, second, first_position, second_position in (
        ('drop', 'lure-set effect: (rare feature, not carried) minus (rare feature, carried by a new face)',
         rare_not_carried, rare_carried_by_new_face, 1, 2),
        ('shortfall', "feature's own effect: (rare feature, carried by a new face) minus (ordinary features)",
         rare_carried_by_new_face, ordinary_features, 2, 3)):
    differences = first - second
    t_value, p_value = ttest_rel(first, second)
    comparisons.append(dict(name=name, label=label, differences=differences,
                            mean_difference=float(np.mean(differences)), t=float(t_value),
                            df=number_of_participants - 1, p=float(p_value),
                            cohen_dz=float(np.mean(differences) / np.std(differences, ddof=1)),
                            first_position=first_position, second_position=second_position,
                            stars=stars_for(p_value)))

# ----------------------------------------------------------------------------- 4. bootstrap intervals
# Figure intervals: one call per panel, conditions in the order they are plotted (left to right).
figure_4b_values = [studied_rare, studied_ordinary, same_feature_lure, never_studied]
figure_4b_intervals = participant_bootstrap_intervals(figure_4b_values)
figure_s11a_values = [rare_not_carried, rare_carried_by_new_face, ordinary_features]
figure_s11a_intervals = participant_bootstrap_intervals(figure_s11a_values)

# ----------------------------------------------------------------------------- 5. write
figure_4b_rows = [
    ('studied_rare_feature', 'Studied, rare feature', 'studied', 3, studied_rare),
    ('studied_ordinary', 'Studied, ordinary', 'studied', 15, studied_ordinary),
    ('new_with_feature', 'New, with feature', 'new', 3, same_feature_lure),
    ('new_without', 'New, without', 'new', 15, never_studied),
]
with open(os.path.join(PATHS['figure_data'], 'fig4b_proportion_called_old.csv'), 'w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['position', 'condition', 'label', 'bar_group', 'mean', 'ci_low', 'ci_high', 'participants',
                     'trials_per_participant', 'interval'])
    for position, (condition, label, bar_group, trials_per_participant, values) in enumerate(figure_4b_rows, start=1):
        low, high = figure_4b_intervals[position - 1]
        writer.writerow([position, condition, label, bar_group, repr(float(np.mean(values))), repr(low), repr(high),
                         number_of_participants, trials_per_participant,
                         '95% percentile bootstrap over participants, 10,000 resamples per bar, one numpy '
                         'default_rng(0) generator for the panel, bars drawn in plotted order'])

figure_s11a_rows = [
    ('rare_feature_not_carried', 'Rare feature, not carried', rare_not_carried),
    ('rare_feature_carried_by_new_face', 'Rare feature, carried by a new face', rare_carried_by_new_face),
    ('ordinary_features', 'Ordinary features', ordinary_features),
]
with open(os.path.join(PATHS['figure_data'], 'figS11a_memorability.csv'), 'w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['position', 'condition', 'label', 'mean', 'ci_low', 'ci_high', 'participants', 'interval'])
    for position, (condition, label, values) in enumerate(figure_s11a_rows, start=1):
        low, high = figure_s11a_intervals[position - 1]
        writer.writerow([position, condition, label, repr(float(np.mean(values))), repr(low), repr(high),
                         number_of_participants,
                         '95% percentile bootstrap over participants, 10,000 resamples per point, one numpy '
                         'default_rng(0) generator for the panel, points drawn in plotted order'])

with open(os.path.join(PATHS['figure_data'], 'figS11a_comparisons.csv'), 'w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['comparison', 'label', 'from_position', 'to_position', 'mean_difference', 't', 'df', 'p',
                     'cohen_dz', 'stars'])
    for comparison in comparisons:
        writer.writerow([comparison['name'], comparison['label'], comparison['first_position'],
                         comparison['second_position'], repr(comparison['mean_difference']), repr(comparison['t']),
                         comparison['df'], repr(comparison['p']), repr(comparison['cohen_dz']), comparison['stars']])

lines = ['Memorability (hit rate minus false alarm rate), the 100 registered participants',
         '==============================================================================', '',
         'participants: %d (in order of completion)' % number_of_participants, '',
         'SI APPENDIX, FIG. S11a, the error bars and the intervals of SI Appendix, Section 3: one numpy generator,',
         'default_rng(0), draws 10,000 participant resamples for each condition in turn, in plotted order', '']
for (condition, label, values), (low, high) in zip(figure_s11a_rows, figure_s11a_intervals):
    lines.append('  %-38s mean %.4f   95%% bootstrap interval %.3f to %.3f' % (label, np.mean(values), low, high))
lines += ['', 'FIG. 4b, the same routine (proportion called old)', '']
for (condition, label, bar_group, trials_per_participant, values), (low, high) in zip(figure_4b_rows, figure_4b_intervals):
    lines.append('  %-38s mean %.4f   95%% bootstrap interval %.3f to %.3f' % (label, np.mean(values), low, high))
lines += ['', 'THE TWO COMPARISONS (paired t tests over participants)', '']
for comparison in comparisons:
    lines.append('  %s' % comparison['label'])
    lines.append('     mean %+.4f   t(%d) = %.3f, p = %.3g, d_z = %.3f  (%s)'
                 % (comparison['mean_difference'], comparison['df'], comparison['t'], comparison['p'],
                    comparison['cohen_dz'], comparison['stars']))
lines.append('')
lines.append('The drop equals the false alarm contrast (new faces with the feature minus never-studied faces,')
lines.append('%+.4f), because the hit rate of the rare faces is the same in both conditions.'
             % float(np.mean(same_feature_lure - never_studied)))
report_text = '\n'.join(lines) + '\n'
with open(os.path.join(PATHS['outputs'], 'memorability_hit_minus_fa.txt'), 'w', encoding='utf-8') as handle:
    handle.write(report_text)
print(report_text)
print('wrote outputs/memorability_hit_minus_fa.txt and outputs/figure_data/fig4b_proportion_called_old.csv,')
print('      figS11a_memorability.csv, figS11a_comparisons.csv')
