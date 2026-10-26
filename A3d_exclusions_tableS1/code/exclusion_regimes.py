"""
exclusion_regimes.py: the image-set results with and without the participant exclusions
(SI Appendix, Section 3, "The Image Set Results Do Not Depend on the Exclusions", Table S1).

WHAT THIS DOES
For the three image-set experiments and every session that was run, it asks whether the link
between summed similarity and memorability (hit rate minus false alarm rate) depends on which
participants are excluded. For each image set and each regime below it computes
    - Table S1, concept level: across concepts, Spearman rho between corrected recognition and the
      mean summed similarity of the tested concept to the 40 items studied in that block,
and for the maintained and changed sets
    - Fig. 6c and 6f: THINGS feature-memorability correlations against this set's,
    - Fig. 6d and 6g (Table S1, feature level): feature summed-similarity correlations against
      feature-memorability correlations (human space), and 6e, 6h (monkey space),
and, without and with the exclusions, the Fig. 7d statistic (Table S1, variance explained and
reduction): the drop in summed squared feature-memorability correlation from the maintained to the
reduced set (participant bootstrap). With the exclusions of the Methods, applied as module 06
applies them (maintained_vs_changed.m Part 3, maintained_vs_reduced_variance.m Part 4), 150
participants are kept per set, and the six Fig. 6 correlations and the kept counts are compared
with the saved run of module 06 (data/paper_run/).

Exclusion rules (Methods; the same values as shared/python/imageset_loader.py):
    a participant is excluded if
      (1) one response key is used on more than 80% of test trials,
      (2) more than 20% of test trials are faster than 200 ms, or more than 20% are slower
          than 7,500 ms, or
      (3) corrected recognition (hit rate minus false alarm rate, responses 1 to 3 = "old")
          is below 0.05.
    then trials faster than 200 ms or slower than 10,000 ms are dropped, and
    the first 150 remaining participants per set (sorted by participant ID) are kept.

Regimes compared
    all_run       every participant run, no participant or trial exclusion, no cap
    reported      the rules above (the exclusions of the paper)
    stricter_010  the rules above with a corrected-recognition floor of 0.10
Exactly 150 participants of each set pass the rules, so the cap of 150 removes nobody (the
participant table of exclusion_regimes.txt gives the passing and the kept counts).

Two measures of summed similarity appear here (each passes a similarity, the correlation of two
concepts' 49 SPoSE feature values, through the kernel normpdf(similarity, 1, 2)):
    - per test trial: the mean kernel similarity of the tested concept to the 40 concepts studied
      in the same block, with one occurrence of the tested concept removed (computed by
      shared/python/imageset_loader.py); a concept's value is the mean over its test trials.
      This is the measure of the concept level of Table S1.
    - per concept within an image set: the kernel similarity of the concept to every concept,
      weighted by that concept's share of the set's images, after removing one image of the
      concept itself (as module 06 computes it, maintained_vs_changed.m Part 6). This is the
      measure of Fig. 6d, 6e, 6g and 6h and of the feature level of Table S1, computed below from
      the set's image counts.
The full-THINGS measure of Fig. 5 is not used here.

READS (all inside the release; paths are found from this file's location)
    06_image_set_experiments_fig6_fig7/data/<set>_trials.csv (the 602 sessions, de-identified;
        read through shared/python/imageset_loader.py, which also computes each test trial's
        summed similarity to the 40 study items of its block)
    06_image_set_experiments_fig6_fig7/data/sampled_image_sets.mat (how many images of each
        concept each set contains)
    shared/things/concept_table.csv (the 49 SPoSE feature values and the Kramer et al. concept
        memorability of the 1,854 THINGS concepts), shared/things/monkey_features.csv
    06_image_set_experiments_fig6_fig7/data/paper_run/ (the saved Fig. 6 and Fig. 7d statistics
        of module 06, for the comparison)
WRITES (A3d_exclusions_tableS1/outputs/)
    exclusion_regimes.txt            all numbers, human-readable
    participant_flags.csv            one row per participant run (anonymous ids M001.., C001.., R001..)
    concept_level_by_regime.csv      the concept-level rho of Table S1 and its p, per set and regime
    feature_level_by_regime.csv      the feature-level r of Fig. 6 (6c to 6h), per regime, for the
                                     maintained and changed sets
    fig7d_reduction_by_regime.csv    the Fig. 7d statistic without and with the exclusions
    reproduction_check.csv           the 'reported' regime beside the saved run of module 06

SEED (numpy.random.default_rng)
    7         participant resamples for the Fig. 7d statistic, 1,000 per set (a new generator for
              each regime; the maintained set's resamples, then the reduced set's)

RUN (from the repository root; about 2 minutes):
    python A3d_exclusions_tableS1/code/exclusion_regimes.py
"""
import os
import sys
# use at most two threads for numpy (set before numpy is imported)
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.io as sio  # noqa: E402
from scipy import stats  # noqa: E402

sys.dont_write_bytecode = True   # no __pycache__ folder in shared/ when this step is run on its own
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (first: it stops Python from writing __pycache__ folders)
import concept_tables  # noqa: E402
import imageset_loader  # noqa: E402

PATHS = release_paths.module_paths(__file__, make_output_folders=False)
OUTPUTS = PATHS['outputs']
os.makedirs(OUTPUTS, exist_ok=True)
IMAGE_SET_DATA = os.path.join(release_paths.find_module('06'), 'data')
PAPER_RUN = os.path.join(IMAGE_SET_DATA, 'paper_run')

IMAGE_SETS = ['maintained', 'changed', 'reduced']
FIG6_SETS = ['maintained', 'changed']      # the sets of Fig. 6 and of the feature level of Table S1
SAMPLED_SET_STRUCT = {'maintained': 'representativeDataset', 'changed': 'flipMemorabilityDataset'}
PARTICIPANTS_KEPT_PER_SET = 150
NUMBER_OF_BOOTSTRAPS = 1000
REDUCTION_BOOTSTRAP_SEED = 7

# the similarity kernel normpdf(similarity, 1, 2) (as in module 06 and shared/python/imageset_loader.py)
KERNEL_MEAN = 1.0
KERNEL_SD = 2.0

# exclusion rule constants (Methods; the same values as shared/python/imageset_loader.py)
SAME_KEY_LIMIT = 0.8
FAST_MS = 200
SLOW_MS = 7500
TOO_MANY_FAST_OR_SLOW = 0.2
TRIAL_MIN_MS = 200
TRIAL_MAX_MS = 10000

output_lines = []


def say(text=''):
    print(text)
    output_lines.append(text)


def normal_density(x, mean, sd):
    """The same as MATLAB normpdf(x, mean, sd)."""
    return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


# ----------------------------------------------------------------------------- load
# Every test trial of the 602 sessions, in a fixed order (maintained, changed, reduced; participants
# in id order; trials in the order shown), with the summed similarity of the tested concept to the
# 40 study items of the same block.
trials = imageset_loader.load_trials(IMAGE_SETS)
trials['said_old'] = (trials['response_1to6'] <= 3).astype(float)

concept_table = concept_tables.load_concept_table()
human_feature_matrix = concept_tables.spose_features(concept_table)          # 1,854 x 49
monkey_feature_matrix = concept_tables.load_monkey_features()               # 1,854 x 49 (IT components)
number_of_concepts = len(concept_table)

# Weighted summed similarity of each concept within the maintained and the changed set (the image-set
# level measure of Fig. 6d, 6e, 6g and 6h): the kernel of similarity to every concept, weighted by how many
# images of that concept the set contains (one image of the concept itself removed), as module 06 computes
# it (maintained_vs_changed.m Parts 6 and 6b).
human_kernel = normal_density(concept_tables.concept_similarity(human_feature_matrix), KERNEL_MEAN, KERNEL_SD)
monkey_kernel = normal_density(concept_tables.concept_similarity(monkey_feature_matrix), KERNEL_MEAN, KERNEL_SD)
sampled_sets = sio.loadmat(os.path.join(IMAGE_SET_DATA, 'sampled_image_sets.mat'), squeeze_me=True,
                           struct_as_record=False)
weighted_similarity_table = {}
for set_name in FIG6_SETS:
    images_per_concept = np.asarray(sampled_sets[SAMPLED_SET_STRUCT[set_name]].numImagesForConcept, dtype=float)
    for space_name, kernel in (('human', human_kernel), ('monkey', monkey_kernel)):
        values = np.full(number_of_concepts, np.nan)
        for concept in range(number_of_concepts):
            if images_per_concept[concept] > 0:
                counts = images_per_concept.copy()
                counts[concept] = counts[concept] - 1
                weights = counts / counts.sum()
                values[concept] = float(np.sum(kernel[concept, :] * weights))
        weighted_similarity_table[(set_name, space_name)] = values

# ----------------------------------------------------------------------------- participant flags
say('TABLE S1 (SI APPENDIX, SECTION 3). DOES THE IMAGE-SET RESULT DEPEND ON THE PARTICIPANT EXCLUSIONS?')
say('Input: the 602 sessions run (205 maintained, 195 changed, 202 reduced; '
    '06_image_set_experiments_fig6_fig7/data).')
say('')
flag_rows = []
for (set_name, participant), one in trials.groupby(['image_set', 'participant'], sort=True):
    number_of_trials = len(one)
    largest_share_one_key = one['response_1to6'].value_counts().max() / number_of_trials
    share_fast = np.mean(one['rt_ms'] < FAST_MS)
    share_slow = np.mean(one['rt_ms'] > SLOW_MS)
    hit_rate = one.loc[one['is_new'] == 0, 'said_old'].mean()
    false_alarm_rate = one.loc[one['is_new'] == 1, 'said_old'].mean()
    corrected_recognition = hit_rate - false_alarm_rate
    flag_rows.append(dict(
        image_set=set_name, participant=participant, test_trials=number_of_trials,
        largest_share_one_key=largest_share_one_key, share_fast=share_fast, share_slow=share_slow,
        hit_rate=hit_rate, false_alarm_rate=false_alarm_rate,
        corrected_recognition=corrected_recognition,
        same_key=largest_share_one_key > SAME_KEY_LIMIT,
        too_fast=share_fast > TOO_MANY_FAST_OR_SLOW,
        too_slow=share_slow > TOO_MANY_FAST_OR_SLOW,
        below_005=corrected_recognition < 0.05,
        below_010=corrected_recognition < 0.10))
flags = pd.DataFrame(flag_rows)
flags['excluded_005'] = flags['same_key'] | flags['too_fast'] | flags['too_slow'] | flags['below_005']
flags['excluded_010'] = flags['same_key'] | flags['too_fast'] | flags['too_slow'] | flags['below_010']


def kept_participants(set_name, floor_column, cap):
    """Participant IDs that pass the rules, sorted, first `cap` of them (cap=None: all)."""
    passing = flags[(flags['image_set'] == set_name) & (~flags[floor_column])]
    ids = sorted(passing['participant'])
    if cap is not None:
        ids = ids[:cap]
    return ids


say('PARTICIPANTS')
say('%-11s %5s %9s %9s %9s %9s %10s %12s %10s %12s' % (
    'image set', 'run', 'same key', 'too fast', 'too slow', 'CR<0.05', 'excl 0.05',
    'kept (cap)', 'excl 0.10', 'kept 0.10'))
for set_name in IMAGE_SETS:
    one = flags[flags['image_set'] == set_name]
    passing_005 = int(np.sum(~one['excluded_005']))
    say('%-11s %5d %9d %9d %9d %9d %10d %12s %10d %12d' % (
        set_name, len(one), one['same_key'].sum(), one['too_fast'].sum(), one['too_slow'].sum(),
        one['below_005'].sum(), one['excluded_005'].sum(),
        '%d -> %d' % (passing_005, min(passing_005, PARTICIPANTS_KEPT_PER_SET)),
        one['excluded_010'].sum(), min(int(np.sum(~one['excluded_010'])), PARTICIPANTS_KEPT_PER_SET)))
total_run = len(flags)
total_excluded = int(flags['excluded_005'].sum())
total_passing = total_run - total_excluded
total_kept = sum(min(int(np.sum(~flags.loc[flags['image_set'] == s, 'excluded_005'])),
                     PARTICIPANTS_KEPT_PER_SET) for s in IMAGE_SETS)
say('total: %d run, %d excluded by the rules, %d pass, %d kept after the 150-per-set cap, '
    '%d passing participants not used because of the cap'
    % (total_run, total_excluded, total_passing, total_kept, total_passing - total_kept))
say('participants meeting more than one rule: %d' % int(np.sum(
    flags[['same_key', 'too_fast', 'too_slow', 'below_005']].sum(axis=1) > 1)))
say('maintained + changed excluded by the rules: %d (nExcluded of the saved Fig. 6 run of module 06)'
    % int(flags.loc[flags['image_set'].isin(['maintained', 'changed']), 'excluded_005'].sum()))
say('maintained + reduced excluded by the rules: %d (nExcluded of the saved Fig. 7d run of module 06)'
    % int(flags.loc[flags['image_set'].isin(['maintained', 'reduced']), 'excluded_005'].sum()))

# Check: the rules above exclude the same participants as shared/python/imageset_loader.py (the
# Python version of the exclusions that module 06 applies in MATLAB).
shared_kept_trials, shared_flags = imageset_loader.apply_exclusions(trials)
excluded_here = set(flags.loc[flags['excluded_005'], 'participant'])
excluded_by_shared_code = set(shared_flags.loc[shared_flags['excluded'], 'participant'])
if excluded_here != excluded_by_shared_code:
    raise RuntimeError('the exclusions of this script differ from those of shared/python/imageset_loader.py')
say('the same %d participants are excluded by shared/python/imageset_loader.py (the paper\'s rules)'
    % len(excluded_by_shared_code))
say('')
flags.to_csv(os.path.join(OUTPUTS, 'participant_flags.csv'), index=False, float_format='%.6g')


# ----------------------------------------------------------------------------- regimes
def regime_trials(set_name, regime):
    """The trials of one image set that a regime keeps."""
    one_set = trials[trials['image_set'] == set_name]
    if regime == 'all_run':
        return one_set
    if regime == 'reported':
        ids = kept_participants(set_name, 'excluded_005', PARTICIPANTS_KEPT_PER_SET)
    elif regime == 'stricter_010':
        ids = kept_participants(set_name, 'excluded_010', PARTICIPANTS_KEPT_PER_SET)
    else:
        raise ValueError(regime)
    kept = one_set[one_set['participant'].isin(ids)]
    kept = kept[(kept['rt_ms'] >= TRIAL_MIN_MS) & (kept['rt_ms'] <= TRIAL_MAX_MS)]
    return kept


# Check: the 'reported' regime keeps the same trials as shared/python/imageset_loader.py.
for set_name in IMAGE_SETS:
    kept_here = regime_trials(set_name, 'reported')
    kept_by_shared_code = shared_kept_trials[shared_kept_trials['image_set'] == set_name]
    if len(kept_here) != len(kept_by_shared_code) or not np.array_equal(
            kept_here['participant'].to_numpy(), kept_by_shared_code['participant'].to_numpy()):
        raise RuntimeError('the kept trials of the %s set differ from shared/python/imageset_loader.py' % set_name)


def participant_by_concept_counts(set_trials):
    """For each participant (rows) and concept (columns): number of old trials, old trials
    called old, new trials, new trials called old, and the sum and count of the same-block
    summed similarity. Pooling rows with weights gives any resample of participants."""
    participants = sorted(set_trials['participant'].unique())
    row_of = {p: i for i, p in enumerate(participants)}
    shape = (len(participants), number_of_concepts)
    counts = {name: np.zeros(shape) for name in
              ('old_n', 'old_yes', 'new_n', 'new_yes', 'sim_sum', 'sim_n')}
    rows = set_trials['participant'].map(row_of).to_numpy()
    columns = set_trials['concept_number'].to_numpy() - 1
    is_old = set_trials['is_new'].to_numpy() == 0
    said_old = set_trials['said_old'].to_numpy()
    similarity = set_trials['summed_similarity_same_block'].to_numpy()
    has_similarity = np.isfinite(similarity)
    np.add.at(counts['old_n'], (rows[is_old], columns[is_old]), 1)
    np.add.at(counts['old_yes'], (rows[is_old], columns[is_old]), said_old[is_old])
    np.add.at(counts['new_n'], (rows[~is_old], columns[~is_old]), 1)
    np.add.at(counts['new_yes'], (rows[~is_old], columns[~is_old]), said_old[~is_old])
    np.add.at(counts['sim_sum'], (rows[has_similarity], columns[has_similarity]),
              similarity[has_similarity])
    np.add.at(counts['sim_n'], (rows[has_similarity], columns[has_similarity]), 1)
    return participants, counts


def concept_scores(counts, weights):
    """Trial-pooled per-concept hit rate, false alarm rate, corrected recognition and mean
    summed similarity, for participants weighted by `weights` (all ones = everyone once)."""
    pooled = {name: weights @ matrix for name, matrix in counts.items()}
    with np.errstate(invalid='ignore', divide='ignore'):
        hit_rate = pooled['old_yes'] / pooled['old_n']
        false_alarm_rate = pooled['new_yes'] / pooled['new_n']
        mean_similarity = pooled['sim_sum'] / pooled['sim_n']
    corrected = hit_rate - false_alarm_rate
    return hit_rate, false_alarm_rate, corrected, mean_similarity


def spearman(x, y):
    keep = np.isfinite(x) & np.isfinite(y)
    rho, p = stats.spearmanr(x[keep], y[keep])
    return rho, p, int(keep.sum())


def feature_correlations(feature_matrix, values):
    """Spearman correlation of each of the 49 features with `values` across concepts
    (MATLAB corr(..., 'Type', 'Spearman', 'rows', 'complete'))."""
    result = np.full(feature_matrix.shape[1], np.nan)
    for k in range(feature_matrix.shape[1]):
        result[k] = spearman(feature_matrix[:, k], values)[0]
    return result


def pearson_across_features(x, y):
    """Pearson r across the features where both values are numbers."""
    keep = np.isfinite(x) & np.isfinite(y)
    r, _ = stats.pearsonr(x[keep], y[keep])
    return r


def weighted_similarity(set_name, space):
    return weighted_similarity_table[(set_name, space)]


things_memorability = concept_table['kramer_concept_cr'].to_numpy(dtype=float)
original_feature_correlations = feature_correlations(human_feature_matrix, things_memorability)
feature_similarity_correlations = {}
for set_name in FIG6_SETS:
    feature_similarity_correlations[(set_name, 'human')] = feature_correlations(
        human_feature_matrix, weighted_similarity(set_name, 'human'))
    feature_similarity_correlations[(set_name, 'monkey')] = feature_correlations(
        monkey_feature_matrix, weighted_similarity(set_name, 'monkey'))

REGIMES = ['all_run', 'reported', 'stricter_010']
REGIME_LABEL = {'all_run': 'every participant run, nothing excluded',
                'reported': 'rules as reported (CR floor 0.05) + 150 cap',
                'stricter_010': 'stricter CR floor 0.10 + 150 cap'}

concept_rows = []
feature_rows = []
per_participant_concept_rates = {}      # for the reduced-set bootstrap below
for set_name in IMAGE_SETS:
    say('-' * 100)
    say('IMAGE SET: %s' % set_name)
    for regime in REGIMES:
        kept = regime_trials(set_name, regime)
        participants, counts = participant_by_concept_counts(kept)
        number_of_participants = len(participants)
        everyone = np.ones(number_of_participants)
        _, _, corrected, mean_similarity = concept_scores(counts, everyone)
        rho, p, n_concepts = spearman(corrected, mean_similarity)
        say('  %-46s participants %3d  trials %6d  concepts %4d' % (
            REGIME_LABEL[regime], number_of_participants, len(kept), n_concepts))
        say('      concept level: rho(CR, summed similarity to study list) = %+.4f  p = %.3g' % (rho, p))
        concept_rows.append(dict(image_set=set_name, regime=regime, participants=number_of_participants,
                                 trials=len(kept), concepts=n_concepts, rho=rho, p=p))

        if set_name in FIG6_SETS:
            # the feature-level statistics of Fig. 6 (6c to 6h)
            feature_memorability = feature_correlations(human_feature_matrix, corrected)
            feature_memorability_monkey = feature_correlations(monkey_feature_matrix, corrected)
            r_original = pearson_across_features(original_feature_correlations, feature_memorability)
            r_human = pearson_across_features(
                feature_similarity_correlations[(set_name, 'human')], feature_memorability)
            r_monkey = pearson_across_features(
                feature_similarity_correlations[(set_name, 'monkey')], feature_memorability_monkey)
            say('      feature level: r(original THINGS, this set) = %+.4f; r(summed sim, memorability) '
                'human = %+.4f, monkey = %+.4f' % (r_original, r_human, r_monkey))
            feature_rows.append(dict(image_set=set_name, regime=regime, participants=number_of_participants,
                                     r_original_vs_set=r_original, r_summed_similarity_human=r_human,
                                     r_summed_similarity_monkey=r_monkey))

        # per-participant per-concept hit and false alarm rates (NaN where none), as module 06
        # builds them (maintained_vs_reduced_variance.m Part 5), for the Fig. 7d bootstrap
        with np.errstate(invalid='ignore', divide='ignore'):
            per_participant_concept_rates[(set_name, regime)] = (
                counts['old_yes'] / np.where(counts['old_n'] > 0, counts['old_n'], np.nan),
                counts['new_yes'] / np.where(counts['new_n'] > 0, counts['new_n'], np.nan))

pd.DataFrame(concept_rows).to_csv(os.path.join(OUTPUTS, 'concept_level_by_regime.csv'),
                                  index=False, float_format='%.6g')
pd.DataFrame(feature_rows).to_csv(os.path.join(OUTPUTS, 'feature_level_by_regime.csv'),
                                  index=False, float_format='%.6g')


# ----------------------------------------------------------------------------- Fig. 7d bootstrap
def reduction_bootstrap(regime, seed):
    """Percent drop in the summed squared feature-memorability correlation from the maintained
    to the reduced set, resampling participants (as module 06 computes Fig. 7d,
    maintained_vs_reduced_variance.m Parts 6 and 7: per-concept rates are means over resampled
    participants of each participant's own rate). Returns the summary values."""
    generator = np.random.default_rng(seed)
    results = {'maintained': [], 'reduced': []}
    for set_name in ('maintained', 'reduced'):
        hit_matrix, fa_matrix = per_participant_concept_rates[(set_name, regime)]
        n = hit_matrix.shape[0]
        for b in range(NUMBER_OF_BOOTSTRAPS):
            drawn = generator.integers(0, n, n)
            weights = np.bincount(drawn, minlength=n).astype(float)
            hit_sum = weights @ np.nan_to_num(hit_matrix)
            hit_count = weights @ np.isfinite(hit_matrix)
            fa_sum = weights @ np.nan_to_num(fa_matrix)
            fa_count = weights @ np.isfinite(fa_matrix)
            with np.errstate(invalid='ignore', divide='ignore'):
                corrected = hit_sum / hit_count - fa_sum / fa_count
            rho = feature_correlations(human_feature_matrix, corrected)
            results[set_name].append(np.nansum(rho ** 2))
    maintained = np.array(results['maintained'])
    reduced = np.array(results['reduced'])
    percent = (maintained - reduced) / maintained * 100
    one_sided_p = np.mean(reduced >= maintained)
    return (maintained.mean(), reduced.mean(), percent.mean(), one_sided_p,
            np.percentile(maintained, [2.5, 97.5]), np.percentile(reduced, [2.5, 97.5]))


say('-' * 100)
say('FIG. 7d STATISTIC: drop in summed squared feature-memorability correlation, maintained -> reduced')
say('(%d participant resamples per set, numpy seed 7)' % NUMBER_OF_BOOTSTRAPS)
reduction_rows = []
for regime in ('reported', 'all_run'):
    (maintained_mean, reduced_mean, percent_mean, p_one,
     maintained_ci, reduced_ci) = reduction_bootstrap(regime, REDUCTION_BOOTSTRAP_SEED)
    say('  %-46s maintained %.4f, reduced %.4f, drop %.1f%%, one-sided p = %.3f' % (
        REGIME_LABEL[regime], maintained_mean, reduced_mean, percent_mean, p_one))
    say('      95%% bootstrap intervals: maintained [%.4f, %.4f], reduced [%.4f, %.4f]' % (
        maintained_ci[0], maintained_ci[1], reduced_ci[0], reduced_ci[1]))
    reduction_rows.append(dict(regime=regime, maintained=maintained_mean, reduced=reduced_mean,
                               maintained_low=maintained_ci[0], maintained_high=maintained_ci[1],
                               reduced_low=reduced_ci[0], reduced_high=reduced_ci[1],
                               percent_drop=percent_mean, one_sided_p=p_one,
                               participants_maintained=per_participant_concept_rates[('maintained', regime)][0].shape[0],
                               participants_reduced=per_participant_concept_rates[('reduced', regime)][0].shape[0]))
pd.DataFrame(reduction_rows).to_csv(os.path.join(OUTPUTS, 'fig7d_reduction_by_regime.csv'),
                                    index=False, float_format='%.6g')

# ----------------------------------------------------------------------------- comparison with module 06
say('-' * 100)
say('COMPARISON WITH THE SAVED RUN OF MODULE 06 (06_image_set_experiments_fig6_fig7/data/paper_run/; '
    'regime "reported")')
paper = sio.loadmat(os.path.join(PAPER_RUN, 'paperStats_maintained_vs_changed.mat'), squeeze_me=True,
                    struct_as_record=False)['paperStats']
variance = sio.loadmat(os.path.join(PAPER_RUN, 'fig7_variance_paper_run.mat'), squeeze_me=True,
                       struct_as_record=False)['varianceData']
feature_table = pd.DataFrame(feature_rows).set_index(['image_set', 'regime'])
concept_level_table = pd.DataFrame(concept_rows).set_index(['image_set', 'regime'])
checks = [
    ('Fig. 6c r(original, maintained)', paper.r_origVsMaintain,
     feature_table.loc[('maintained', 'reported'), 'r_original_vs_set'], '0.84'),
    ('Fig. 6d r(summed sim, maintained) human', paper.r_sumSimVsMaintain,
     feature_table.loc[('maintained', 'reported'), 'r_summed_similarity_human'], '-0.89'),
    ('Fig. 6e r(summed sim, maintained) monkey', paper.r_monkeyVsMaintain,
     feature_table.loc[('maintained', 'reported'), 'r_summed_similarity_monkey'], '-0.63'),
    ('Fig. 6f r(original, changed)', paper.r_origVsChanged,
     feature_table.loc[('changed', 'reported'), 'r_original_vs_set'], '-0.33'),
    ('Fig. 6g r(summed sim, changed) human', paper.r_sumSimVsChanged,
     feature_table.loc[('changed', 'reported'), 'r_summed_similarity_human'], '-0.75'),
    ('Fig. 6h r(summed sim, changed) monkey', paper.r_monkeyVsChanged,
     feature_table.loc[('changed', 'reported'), 'r_summed_similarity_monkey'], '-0.45'),
    ('Fig. 7d percent drop (bootstrap mean)', variance.mean_percentReduction_summed,
     reduction_rows[0]['percent_drop'], '27%'),
    ('Fig. 7d one-sided p', variance.p_value_summed, reduction_rows[0]['one_sided_p'], '0.036'),
    ('maintained participants kept', paper.nParticipantsMaintain,
     concept_level_table.loc[('maintained', 'reported'), 'participants'], '150'),
    ('changed participants kept', paper.nParticipantsChanged,
     concept_level_table.loc[('changed', 'reported'), 'participants'], '150'),
    ('reduced participants kept', variance.n_equated,
     concept_level_table.loc[('reduced', 'reported'), 'participants'], '150'),
]
say('%-42s %12s %12s %10s' % ('statistic', 'saved', 'recomputed', 'paper'))
check_rows = []
for name, saved, recomputed, printed in checks:
    say('%-42s %12.4f %12.4f %10s' % (name, float(saved), float(recomputed), printed))
    check_rows.append(dict(statistic=name, saved_paper_run=float(saved), recomputed=float(recomputed),
                           printed=printed))
# The six Fig. 6 correlations and the three participant counts must equal the saved run; the two
# Fig. 7d rows come from different random draws (MATLAB rng(42) there, numpy seed 7 here).
for name, saved, recomputed, printed in checks:
    if 'Fig. 7d' not in name and abs(float(saved) - float(recomputed)) > 1e-9:
        raise RuntimeError('%s does not equal the value in the saved run of module 06' % name)
pd.DataFrame(check_rows).to_csv(os.path.join(OUTPUTS, 'reproduction_check.csv'), index=False,
                                float_format='%.10g')

with open(os.path.join(OUTPUTS, 'exclusion_regimes.txt'), 'w') as handle:
    handle.write('\n'.join(output_lines) + '\n')
print('wrote', os.path.join('A3d_exclusions_tableS1', 'outputs', 'exclusion_regimes.txt'))
