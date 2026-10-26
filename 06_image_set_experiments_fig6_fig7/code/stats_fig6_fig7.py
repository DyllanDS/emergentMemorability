"""
stats_fig6_fig7.py: the uncertainty statistics of Figs. 6 and 7 and the Fig. 6 rows of outputs/numbers.csv.

WHAT IT COMPUTES (one or more rows per test in outputs/tests.csv; the columns are described in
statistics/README.md)
  T12 to T17  The six Fig. 6 correlations (panels c to h). Each is a Pearson r across the 49 SPoSE
              features between two "profiles" (one value per feature):
                - a set's feature-memorability profile: for each feature, the Spearman correlation
                  across the set's concepts between the feature and the concepts' corrected
                  recognition (CR = hit rate minus false alarm rate, pooled over all kept trials of
                  the concept; "old" = responses 1 to 3);
                - a fixed profile: the full-THINGS memorability profile (Kramer et al. concept
                  scores; panels c, f), or the set's summed-similarity profile in the human SPoSE
                  space (d, g) or in the monkey IT space (e, h; there the 49 "features" are the
                  first 49 principal components of the IT responses, and the memorability profile
                  is computed over them).
              r is computed first and checked against the saved run in data/paper_run/. Its interval
              and its p come from one bootstrap over the 150 participants of the set (10,000
              resamples): participants are resampled with replacement, every concept's CR is
              recomputed from the resampled participants' trials (a participant drawn twice counts
              twice), the 49 Spearman correlations are recomputed, then r. The fixed profile stays
              fixed, and the 49 features are not resampled.
                - 95% interval: the 2.5 and 97.5 percentiles of the resampled r.
                - two-sided p: twice the share of the resamples on the far side of zero (r >= 0 when
                  the observed r is negative, r <= 0 when it is positive), at most 1; the test text
                  gives the number of these resamples.
  T18         The spread of summed similarity over the concepts of the maintained and the reduced
              set (Fig. 7b, 7c; SD 0.00322 and 0.00156), one descriptive row per set.
  T19         Fig. 7d: the sum over the 49 features of the squared Spearman correlation between the
              feature and concept CR, in the maintained and in the reduced set, and its percent
              reduction (27 percent). CR here is each participant's hit rate and false alarm rate for
              each concept, averaged over the participants (ignoring those who had no such trial for
              the concept). The statistics come from the 1,000 participant resamples of the saved run
              in data/paper_run/ (MATLAB, rng(42)), which maintained_vs_reduced_variance.m draws again;
              when outputs/varianceData.mat exists, the script checks that it holds the same
              resamples. The rows give the percent reduction with the one-sided bootstrap p for the
              predicted direction (the share of resamples in which the reduced-set sum is at least the
              maintained-set sum), and each set's sum (the bars) with its percentile interval (MATLAB
              prctile) and the summed rho^2 of the full sample (no resampling).
  T52         The largest share of one higher-level THINGS category among the reduced set's 1,600
              images, with the category labels of shared/things/concept_table.csv.
  T54         Participant flow: sessions, exclusions by rule, kept, per set and in total.
  No correction for multiple comparisons is applied (every test is reported on its own).

  outputs/numbers.csv: the rows of Fig. 6c to 6h (r, 95% interval and p, as the caption and the text
  print them) and the Methods row (Statistical Reporting) stating that no resample crosses zero.
  run_module.m writes the other rows of the module; this script replaces its own rows in that file and
  keeps the others, so it runs after run_module.m.

INPUTS
  data/maintained_trials.csv, changed_trials.csv, reduced_trials.csv, through
      shared/python/imageset_loader.py (which applies the exclusion criteria of the Methods)
  data/sampled_image_sets.mat (the slots of each concept in each set)
  shared/things/concept_table.csv (SPoSE features, Kramer et al. concept memorability, category
      labels), monkey_features.csv
  data/paper_run/paperStats_maintained_vs_changed.mat and fig7_variance_paper_run.mat (the saved runs)
  shared/ledger/paper_numbers.csv (the printed values of the Fig. 6 rows)
  outputs/ files written by run_module.m, if present (cross-checks, and the other rows of numbers.csv)

OUTPUTS: outputs/tests.csv, outputs/numbers.csv (the Fig. 6 rows), outputs/logs/stats_fig6_fig7_log.txt

SEED: 20261013 (numpy.random.default_rng). One generator for the whole script, drawn in this order:
  1. participant resamples of the maintained set (T12 to T14)
  2. participant resamples of the changed set (T15 to T17)
  Other --resamples values change every later draw.

RUN (from the repository root; under 1 minute):
  python 06_image_set_experiments_fig6_fig7/code/stats_fig6_fig7.py
  (--resamples N sets a smaller number for a quick check of the code; the default is 10,000)
"""
import argparse
import csv
import os
import sys
import time

# Do not write __pycache__ folders into the release (set before the shared helpers are imported)
sys.dont_write_bytecode = True

# Use at most two processor threads for the matrix arithmetic (with these small matrices more threads
# only add overhead). Must be set before numpy is imported.
for thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(thread_setting, '2')

import numpy as np  # noqa: E402
import scipy.io  # noqa: E402
from scipy.stats import pearsonr, rankdata  # noqa: E402

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CODE_FOLDER, '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (the release's folders)
import bootstrap  # noqa: E402
import concept_tables  # noqa: E402
import imageset_loader  # noqa: E402
import numbers_table  # noqa: E402
import tests_table  # noqa: E402

PATHS = release_paths.module_paths(__file__)
SCRIPT_NAME = 'code/stats_fig6_fig7.py'
SEED = bootstrap.RELEASE_SEED
SET_NAMES = ('maintained', 'changed', 'reduced')
STRUCT_OF_SET = {'maintained': 'representativeDataset', 'changed': 'flipMemorabilityDataset',
                 'reduced': 'equatedDataset'}
KERNEL_MEAN = 1.0    # the similarity kernel normpdf(similarity, 1, 2) of every summed-similarity measure
KERNEL_SD = 2.0
# Ledger row of the Methods (Statistical Reporting): no resample of Fig. 6c to 6h crosses zero
NO_CROSSING_LEDGER_ID = 'L3-121'
FIG6_SECTION = 'The Similarity Structure of an Image Set Determines What Will Be Memorable'
FIG7_SECTION = 'Reducing Variability In Distinctiveness Reduces Variability in Memorability'

LOG_LINES = []


def report(text=''):
    """Print a line and keep it for the log file."""
    print(text)
    LOG_LINES.append(text)


# ------------------------------------------------------------------------------------------------
# Small building blocks
# ------------------------------------------------------------------------------------------------

def normal_density(x, mean, sd):
    """The same as MATLAB normpdf(x, mean, sd)."""
    return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


class RankedColumns:
    """The average ranks of every column of a concepts x features table over a chosen set of
    concepts, centered, ready to be correlated with the ranks of one memory score per concept."""

    def __init__(self, columns, usable):
        self.usable = usable.copy()
        ranks = rankdata(columns[usable], axis=0)
        self.centred = ranks - ranks.mean(axis=0)
        self.norms = np.sqrt(np.sum(self.centred ** 2, axis=0))

    def spearman_with(self, values_on_usable_concepts):
        """Spearman correlation (Pearson correlation of average ranks) of every column with the
        given values (one per usable concept)."""
        value_ranks = rankdata(values_on_usable_concepts)
        centred_values = value_ranks - value_ranks.mean()
        return (self.centred.T @ centred_values) / (self.norms * np.sqrt(np.sum(centred_values ** 2)))


def spearman_with_columns(columns, values):
    """Spearman correlation of every column of `columns` (concepts x features) with `values` (one
    per concept), over the concepts where `values` is a number, average ranks for ties: the same as
    MATLAB corr(columns, values, 'Type', 'Spearman', 'rows', 'complete')."""
    usable = np.isfinite(values)
    ranked = RankedColumns(columns, usable)
    return ranked.spearman_with(values[usable])


def pearson_r(x, y):
    """Pearson correlation of two equally long lists of numbers."""
    return float(np.corrcoef(np.asarray(x, dtype=float), np.asarray(y, dtype=float))[0, 1])


def summed_similarity_in_set(kernel, slot_counts):
    """Summed similarity of every concept within one image set (maintained_vs_changed.m Part 6):
    the kernel similarity of the concept to every concept, weighted by that concept's share of the
    set's slots, after removing one slot of the concept itself (weights renormalized). NaN for
    concepts that are not in the set."""
    number_of_concepts = len(slot_counts)
    summed = np.full(number_of_concepts, np.nan)
    for concept in range(number_of_concepts):
        if slot_counts[concept] > 0:
            counts = slot_counts.astype(float).copy()
            counts[concept] = counts[concept] - 1
            weights = counts / counts.sum()
            summed[concept] = float(kernel[concept] @ weights)
    return summed


def percentile_interval(values):
    """The 2.5 and 97.5 percentiles (numpy's linear interpolation; resamples that gave NaN ignored)."""
    low, high = bootstrap.percentile_interval(values)
    return float(low), float(high)


def bootstrap_p_two_sided(resampled, observed):
    """Two-sided bootstrap p of a correlation: twice the share of the resamples on the far side of zero
    (r >= 0 when the observed r is negative, r <= 0 when it is positive), at most 1; resamples that gave
    NaN are ignored. Returns p, the number of resamples on the far side and the number of resamples."""
    resampled = np.asarray(resampled, dtype=float)
    resampled = resampled[np.isfinite(resampled)]
    if observed < 0:
        far_side = int(np.sum(resampled >= 0))
    else:
        far_side = int(np.sum(resampled <= 0))
    return min(1.0, 2.0 * far_side / len(resampled)), far_side, len(resampled)


# ------------------------------------------------------------------------------------------------
# Per-participant counts and the two concept-memorability estimators
# ------------------------------------------------------------------------------------------------

def participant_concept_counts(set_trials, participants, set_concepts):
    """Four participants x concepts tables of trial counts for one set: old trials, old trials
    called old (hits), new trials, new trials called old (false alarms)."""
    row_of_participant = {participant: row for row, participant in enumerate(participants)}
    column_of_concept = {concept: column for column, concept in enumerate(set_concepts)}
    shape = (len(participants), len(set_concepts))
    counts = {'old_trials': np.zeros(shape), 'hits': np.zeros(shape),
              'new_trials': np.zeros(shape), 'false_alarms': np.zeros(shape)}
    for participant, concept, is_new, said_old in zip(set_trials['participant'], set_trials['concept_number'],
                                                      set_trials['is_new'], set_trials['said_old']):
        row = row_of_participant[participant]
        column = column_of_concept[concept]
        if is_new == 1:
            counts['new_trials'][row, column] += 1
            counts['false_alarms'][row, column] += int(said_old)
        else:
            counts['old_trials'][row, column] += 1
            counts['hits'][row, column] += int(said_old)
    return counts


def pooled_concept_rates(participant_weights, counts):
    """Fig. 6 estimator (maintained_vs_changed.m Part 4): each concept's hit rate and false alarm
    rate pooled over the trials of the (weighted) participants; NaN when a concept has no old or no
    new trial. With weights 1 for every participant this is the full-sample value; in a bootstrap
    resample the weight of a participant is the number of times they were drawn (one matrix
    product per resample)."""
    old_trials = participant_weights @ counts['old_trials']
    hits = participant_weights @ counts['hits']
    new_trials = participant_weights @ counts['new_trials']
    false_alarms = participant_weights @ counts['false_alarms']
    with np.errstate(divide='ignore', invalid='ignore'):
        hit_rate = np.where(old_trials > 0, hits / old_trials, np.nan)
        false_alarm_rate = np.where(new_trials > 0, false_alarms / new_trials, np.nan)
    return hit_rate, false_alarm_rate


def participant_rate_tables(counts):
    """Each participant's hit rate and false alarm rate for each concept (NaN when the participant
    had no old, or no new, trial of the concept): the tables of maintained_vs_reduced_variance.m
    Part 5."""
    with np.errstate(divide='ignore', invalid='ignore'):
        hit_rates = np.where(counts['old_trials'] > 0, counts['hits'] / counts['old_trials'], np.nan)
        false_alarm_rates = np.where(counts['new_trials'] > 0, counts['false_alarms'] / counts['new_trials'], np.nan)
    return hit_rates, false_alarm_rates


def averaged_participant_cr(participant_weights, hit_rates, false_alarm_rates):
    """Fig. 7d estimator: the mean of the participants' per-concept hit rates minus the mean of their
    per-concept false alarm rates, ignoring participants without such trials (MATLAB
    mean(..., 2, 'omitnan') over the resampled participants; a participant drawn twice counts twice)."""
    hit_present = np.isfinite(hit_rates)
    false_alarm_present = np.isfinite(false_alarm_rates)
    hit_sum = participant_weights @ np.where(hit_present, hit_rates, 0.0)
    hit_count = participant_weights @ hit_present
    false_alarm_sum = participant_weights @ np.where(false_alarm_present, false_alarm_rates, 0.0)
    false_alarm_count = participant_weights @ false_alarm_present
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_hit = np.where(hit_count > 0, hit_sum / hit_count, np.nan)
        mean_false_alarm = np.where(false_alarm_count > 0, false_alarm_sum / false_alarm_count, np.nan)
    return mean_hit - mean_false_alarm


def draw_participant_weights(generator, number_of_participants):
    """One bootstrap resample of the participants: how often each was drawn (with replacement)."""
    drawn = generator.integers(0, number_of_participants, number_of_participants)
    return np.bincount(drawn, minlength=number_of_participants).astype(float)


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--resamples', type=int, default=10000, help='bootstrap resamples (default 10,000)')
    arguments = parser.parse_args()
    number_of_resamples = arguments.resamples
    started = time.time()
    report('stats_fig6_fig7.py: seed %d, %d bootstrap resamples' % (SEED, number_of_resamples))
    if number_of_resamples != 10000:
        report('NOTE: a reduced number of resamples (a quick check of the code)')
    generator = np.random.default_rng(SEED)
    tests = tests_table.TestsTable(script=SCRIPT_NAME)

    # ---------------- inputs ----------------
    concept_table = concept_tables.load_concept_table()
    concept_names = concept_table['concept'].tolist()
    features = concept_tables.spose_features(concept_table)
    monkey_features = concept_tables.load_monkey_features()
    kramer_memorability = concept_table['kramer_concept_cr'].to_numpy(dtype=float)
    human_kernel = normal_density(concept_tables.concept_similarity(features), KERNEL_MEAN, KERNEL_SD)
    monkey_kernel = normal_density(concept_tables.concept_similarity(monkey_features), KERNEL_MEAN, KERNEL_SD)

    sampled_sets = scipy.io.loadmat(os.path.join(PATHS['data'], 'sampled_image_sets.mat'), squeeze_me=True,
                                    struct_as_record=False)
    slot_counts = {}
    for set_name in SET_NAMES:
        slot_counts[set_name] = np.asarray(sampled_sets[STRUCT_OF_SET[set_name]].numImagesForConcept,
                                           dtype=float).ravel()

    trials = imageset_loader.load_trials(SET_NAMES, with_summed_similarity=False)
    kept_trials, flags = imageset_loader.apply_exclusions(trials)

    set_data = {}
    for set_name in SET_NAMES:
        set_trials = kept_trials[kept_trials['image_set'] == set_name]
        participants = sorted(set_trials['participant'].unique())
        set_concepts = np.flatnonzero(slot_counts[set_name] > 0) + 1          # THINGS concept numbers (from 1)
        if not set(set_trials['concept_number']).issubset(set(set_concepts)):
            raise ValueError('a tested concept is not part of the %s set' % set_name)
        counts = participant_concept_counts(set_trials, participants, set_concepts)
        set_data[set_name] = dict(trials=set_trials, participants=participants, concepts=set_concepts,
                                  counts=counts)
        report('%-10s set: %d participants kept, %d concepts, %d test trials kept' % (
            set_name, len(participants), len(set_concepts), len(set_trials)))

    # ---------------- Fig. 6: r, compared with the saved run ----------------
    report('')
    report('Fig. 6 (T12 to T17): r, compared with the saved run in data/paper_run/')
    things_profile = spearman_with_columns(features, kramer_memorability)   # fixed for c and f
    paper_run = scipy.io.loadmat(os.path.join(PATHS['paper_run'], 'paperStats_maintained_vs_changed.mat'),
                                 squeeze_me=True, struct_as_record=False)['paperStats']
    fig6 = {}
    for set_name in ('maintained', 'changed'):
        data = set_data[set_name]
        concepts_index = data['concepts'] - 1
        all_weights = np.ones(len(data['participants']))
        hit_rate, false_alarm_rate = pooled_concept_rates(all_weights, data['counts'])
        concept_cr = hit_rate - false_alarm_rate
        usable = np.isfinite(concept_cr)
        human_ranked = RankedColumns(features[concepts_index], usable)
        monkey_ranked = RankedColumns(monkey_features[concepts_index], usable)
        memorability_profile = human_ranked.spearman_with(concept_cr[usable])
        monkey_memorability_profile = monkey_ranked.spearman_with(concept_cr[usable])
        human_summed = summed_similarity_in_set(human_kernel, slot_counts[set_name])
        monkey_summed = summed_similarity_in_set(monkey_kernel, slot_counts[set_name])
        human_similarity_profile = spearman_with_columns(features, human_summed)
        monkey_similarity_profile = spearman_with_columns(monkey_features, monkey_summed)
        fig6[set_name] = dict(concept_cr=concept_cr, hit_rate=hit_rate, false_alarm_rate=false_alarm_rate,
                              usable=usable, human_ranked=human_ranked, monkey_ranked=monkey_ranked,
                              memorability_profile=memorability_profile,
                              monkey_memorability_profile=monkey_memorability_profile,
                              human_similarity_profile=human_similarity_profile,
                              monkey_similarity_profile=monkey_similarity_profile, human_summed=human_summed)

    panels = [  # test id, panel, set, fixed profile, which memorability profile, saved-run name, ledger ids
        ('T12', 'c', 'maintained', 'things', 'human', 'origVsMaintain', ('L3-022', 'L3-039')),
        ('T13', 'd', 'maintained', 'human_similarity', 'human', 'sumSimVsMaintain', ('L3-023', 'L3-040')),
        ('T14', 'e', 'maintained', 'monkey_similarity', 'monkey', 'monkeyVsMaintain', ('L3-024', 'L3-041')),
        ('T15', 'f', 'changed', 'things', 'human', 'origVsChanged', ('L3-025', 'L3-042')),
        ('T16', 'g', 'changed', 'human_similarity', 'human', 'sumSimVsChanged', ('L3-026', 'L3-043')),
        ('T17', 'h', 'changed', 'monkey_similarity', 'monkey', 'monkeyVsChanged', ('L3-027', 'L3-044')),
    ]
    fixed_profile_text = {'things': 'the full-THINGS feature-memorability profile (Kramer et al. concept scores)',
                          'human_similarity': 'the set\'s summed-similarity profile in the human SPoSE space',
                          'monkey_similarity': 'the set\'s summed-similarity profile in the monkey IT space'}

    def fixed_profile(set_name, which):
        if which == 'things':
            return things_profile
        if which == 'human_similarity':
            return fig6[set_name]['human_similarity_profile']
        return fig6[set_name]['monkey_similarity_profile']

    def memorability_profile_of(set_name, which):
        if which == 'human':
            return fig6[set_name]['memorability_profile']
        return fig6[set_name]['monkey_memorability_profile']

    observed = {}
    for test_id, panel, set_name, fixed, which, paper_name, ledger_ids in panels:
        x_values = fixed_profile(set_name, fixed)
        y_values = memorability_profile_of(set_name, which)
        r_value, _ = pearsonr(x_values, y_values)
        observed[test_id] = float(r_value)
        paper_r = float(getattr(paper_run, 'r_' + paper_name))
        report('  Fig. 6%s %s: r = %.10f (saved run %.10f, difference %.1e)' % (
            panel, test_id, r_value, paper_r, r_value - paper_r))
        if abs(r_value - paper_r) > 1e-9:
            raise RuntimeError('Fig. 6%s differs from the saved run in data/paper_run/' % panel)

    # ---------------- Fig. 6: participant bootstrap ----------------
    report('')
    report('Fig. 6: participant bootstrap (%d resamples per set)' % number_of_resamples)
    bootstrap_r = {}
    for set_name in ('maintained', 'changed'):
        data = set_data[set_name]
        concepts_index = data['concepts'] - 1
        full_usable = fig6[set_name]['usable']
        number_of_participants = len(data['participants'])
        # the resamples are drawn first, in order, from the one generator
        weights_of_resample = np.zeros((number_of_resamples, number_of_participants))
        for resample in range(number_of_resamples):
            weights_of_resample[resample] = draw_participant_weights(generator, number_of_participants)
        set_tests = [panel_row for panel_row in panels if panel_row[2] == set_name]
        results = {panel_row[0]: np.full(number_of_resamples, np.nan) for panel_row in set_tests}
        resamples_with_fewer_concepts = 0
        for resample in range(number_of_resamples):
            hit_rate, false_alarm_rate = pooled_concept_rates(weights_of_resample[resample], data['counts'])
            concept_cr = hit_rate - false_alarm_rate
            usable = np.isfinite(concept_cr)
            if np.array_equal(usable, full_usable):
                human_ranked = fig6[set_name]['human_ranked']
                monkey_ranked = fig6[set_name]['monkey_ranked']
            else:
                # a concept lost all its old (or new) trials in this resample: rank on the remaining concepts
                resamples_with_fewer_concepts += 1
                human_ranked = RankedColumns(features[concepts_index], usable)
                monkey_ranked = RankedColumns(monkey_features[concepts_index], usable)
            human_profile = human_ranked.spearman_with(concept_cr[usable])
            monkey_profile = monkey_ranked.spearman_with(concept_cr[usable])
            for test_id, panel, _, fixed, which, _, _ in set_tests:
                profile = human_profile if which == 'human' else monkey_profile
                results[test_id][resample] = pearson_r(fixed_profile(set_name, fixed), profile)
        bootstrap_r.update(results)
        report('  %s set: %d of %d resamples had fewer usable concepts than the full sample' % (
            set_name, resamples_with_fewer_concepts, number_of_resamples))

    # interval and two-sided p of each correlation, from the same resamples
    fig6_results = {}
    for test_id, panel, set_name, fixed, which, paper_name, ledger_ids in panels:
        r_value = observed[test_id]
        low, high = percentile_interval(bootstrap_r[test_id])
        p_value, far_side, valid_resamples = bootstrap_p_two_sided(bootstrap_r[test_id], r_value)
        fig6_results[test_id] = dict(r=r_value, low=low, high=high, p=p_value, far_side=far_side)
        number_of_concepts = int(np.sum(fig6[set_name]['usable']))
        unit = '%s (%s set: %d participants, %d concepts)' % (
            'features' if which == 'human' else 'IT components', set_name, len(set_data[set_name]['participants']),
            number_of_concepts)
        profile_kind = 'the 49 SPoSE features' if which == 'human' else 'the 49 monkey IT components'
        tests.add(test_id=test_id, ledger_ids=';'.join(ledger_ids + (NO_CROSSING_LEDGER_ID,)),
                  location='Fig. 6%s caption; %s' % (panel, FIG6_SECTION),
                  test=('Fig. 6%s: Pearson r across %s between %s and the %s set\'s feature-memorability profile '
                        '(each value a Spearman correlation across concepts with concept CR); interval and p: '
                        'participants resampled, concept CR and the profile recomputed, fixed profile kept; p is '
                        'twice the share of the resamples on the far side of zero (%d of %d)'
                        % (panel, profile_kind, fixed_profile_text[fixed], set_name, far_side, valid_resamples)),
                  statistic_name='Pearson r', statistic=r_value, n=49, unit=unit, p=p_value,
                  p_type='bootstrap two-sided', effect_size_name='r', effect_size=r_value, ci_low=low, ci_high=high,
                  ci_method='percentile bootstrap', resampling_unit='participants', n_resamples=number_of_resamples,
                  seed=SEED)
        report('%s Fig. 6%s: r = %.3f, 95%% CI [%.3f, %.3f] (participants), bootstrap p = %.4g (%d of %d resamples on '
               'the far side of zero)' % (test_id, panel, r_value, low, high, p_value, far_side, valid_resamples))

    # the Fig. 6 rows of outputs/numbers.csv (written at the end, after the cross-checks)
    numbers = numbers_table.NumbersTable(script=SCRIPT_NAME)
    for test_id, panel, set_name, fixed, which, paper_name, ledger_ids in panels:
        result = fig6_results[test_id]
        note = ('Fig. 6%s: Pearson r across the 49 %s; 2.5 and 97.5 percentiles of the %d participant resamples; '
                'two-sided bootstrap p (%d resamples on the far side of zero); test %s'
                % (panel, 'SPoSE features' if which == 'human' else 'monkey IT components', number_of_resamples,
                   result['far_side'], test_id))
        for ledger_id in ledger_ids:
            numbers.add(ledger_id, computed_value=[result['r'], result['low'], result['high'], result['p']],
                        mode='rerun', note=note)
    far_side_counts = [fig6_results[panel_row[0]]['far_side'] for panel_row in panels]
    numbers.add(NO_CROSSING_LEDGER_ID,
                computed_value=['none' if sum(far_side_counts) == 0 else sum(far_side_counts), number_of_resamples],
                mode='rerun',
                note=('resamples on the far side of zero in Fig. 6c to 6h: %s, of %d participant resamples each '
                      '(tests T12 to T17)' % (', '.join(str(count) for count in far_side_counts), number_of_resamples)))

    # ---------------- T18: spread of summed similarity (Fig. 7b, 7c) ----------------
    report('')
    summed_maintained = summed_similarity_in_set(human_kernel, slot_counts['maintained'])
    summed_reduced = summed_similarity_in_set(human_kernel, slot_counts['reduced'])
    values_maintained = summed_maintained[np.isfinite(summed_maintained)]
    values_reduced = summed_reduced[np.isfinite(summed_reduced)]
    sd_maintained = float(np.std(values_maintained, ddof=1))
    sd_reduced = float(np.std(values_reduced, ddof=1))
    sd_ledgers = 'L3-047;L3-048;L3-049;L3-051;L3-052;L5-061;L5-062'
    for set_name, sd, values in (('maintained', sd_maintained, values_maintained),
                                 ('reduced', sd_reduced, values_reduced)):
        tests.add(test_id='T18', ledger_ids=sd_ledgers,
                  location='Fig. 7b, 7c caption; %s; Methods, Image Set Structure Manipulations' % FIG7_SECTION,
                  test=('SD (n - 1) of the within-set summed similarity over the concepts of the %s set; descriptive'
                        % set_name),
                  statistic_name='SD of summed similarity', statistic=sd, n=len(values), unit='concepts',
                  p_type='none', ci_method='none')
    report('T18: SD of summed similarity, maintained %.10f (%d concepts), reduced %.10f (%d concepts)' % (
        sd_maintained, len(values_maintained), sd_reduced, len(values_reduced)))

    # ---------------- T19: Fig. 7d ----------------
    report('')
    report('Fig. 7d (T19): the saved run in data/paper_run/ (MATLAB, rng(42), 1,000 participant resamples)')
    paper_variance = scipy.io.loadmat(os.path.join(PATHS['paper_run'], 'fig7_variance_paper_run.mat'),
                                      squeeze_me=True, struct_as_record=False)['varianceData']
    paper_summed_maintained = np.sum(paper_variance.bootstrapFeatureMemCorrs_rep ** 2, axis=1)
    paper_summed_reduced = np.sum(paper_variance.bootstrapFeatureMemCorrs_equated ** 2, axis=1)
    paper_drop = paper_summed_maintained - paper_summed_reduced
    paper_percent = paper_drop / paper_summed_maintained * 100
    paper_values = dict(
        percent=float(np.mean(paper_percent)),
        p_one_sided=float(np.mean(paper_summed_reduced >= paper_summed_maintained)),
        maintained=float(np.mean(paper_summed_maintained)),
        maintained_ci=bootstrap.matlab_prctile(paper_summed_maintained, [2.5, 97.5]),
        reduced=float(np.mean(paper_summed_reduced)),
        reduced_ci=bootstrap.matlab_prctile(paper_summed_reduced, [2.5, 97.5]))
    stored_pairs = [('percent', paper_values['percent'], paper_variance.mean_percentReduction_summed),
                    ('one-sided p', paper_values['p_one_sided'], paper_variance.p_value_summed),
                    ('summed rho^2 maintained', paper_values['maintained'], paper_variance.mean_summed_rep),
                    ('summed rho^2 reduced', paper_values['reduced'], paper_variance.mean_summed_equated),
                    ('maintained CI low', paper_values['maintained_ci'][0], paper_variance.ci_summed_rep[0]),
                    ('maintained CI high', paper_values['maintained_ci'][1], paper_variance.ci_summed_rep[1]),
                    ('reduced CI low', paper_values['reduced_ci'][0], paper_variance.ci_summed_equated[0]),
                    ('reduced CI high', paper_values['reduced_ci'][1], paper_variance.ci_summed_equated[1])]
    largest_difference = 0.0
    for name, recomputed, stored in stored_pairs:
        largest_difference = max(largest_difference, abs(recomputed - float(stored)))
        report('  %-25s recomputed %.10f, stored %.10f' % (name, recomputed, float(stored)))
    report('  largest difference to the stored values: %.2g' % largest_difference)
    if largest_difference > 1e-9:
        raise RuntimeError('the Fig. 7d statistics of the saved run are not reproduced from its resamples')
    rerun_file = os.path.join(PATHS['outputs'], 'varianceData.mat')
    if os.path.exists(rerun_file):
        rerun = scipy.io.loadmat(rerun_file, squeeze_me=True, struct_as_record=False)['varianceData']
        # MATLAB draws the same resamples on every computer; the math library it uses for the correlations
        # differs between computers and can change their last digit (about 1e-16), so the check allows 1e-12
        largest_rerun_difference = np.inf
        if (rerun.bootstrapFeatureMemCorrs_rep.shape == paper_variance.bootstrapFeatureMemCorrs_rep.shape
                and rerun.bootstrapFeatureMemCorrs_equated.shape == paper_variance.bootstrapFeatureMemCorrs_equated.shape):
            largest_rerun_difference = float(np.max(np.abs(np.concatenate([
                rerun.bootstrapFeatureMemCorrs_rep - paper_variance.bootstrapFeatureMemCorrs_rep,
                rerun.bootstrapFeatureMemCorrs_equated - paper_variance.bootstrapFeatureMemCorrs_equated]))))
        report('  the MATLAB run of run_module.m (outputs/varianceData.mat), largest difference to these 1,000 '
               'resamples: %.2g' % largest_rerun_difference)
        if np.isnan(largest_rerun_difference) or largest_rerun_difference > 1e-12:
            raise RuntimeError('the MATLAB run of Fig. 7d differs from the saved run in data/paper_run/')
    else:
        report('  (outputs/varianceData.mat not found: run run_module.m to draw the MATLAB bootstrap itself)')

    # The summed rho^2 of the full sample (no resampling): the effect size of the two bar rows below, and
    # checked against the MATLAB value in cross_check_matlab_outputs()
    report('Fig. 7d (T19): summed rho^2 on the full sample (no resampling)')
    rate_tables = {}
    for set_name in ('maintained', 'reduced'):
        hit_rates, false_alarm_rates = participant_rate_tables(set_data[set_name]['counts'])
        rate_tables[set_name] = (hit_rates, false_alarm_rates)
    concepts_index = {set_name: set_data[set_name]['concepts'] - 1 for set_name in ('maintained', 'reduced')}
    full_sample_sum = {}
    full_usable = {}
    full_ranked = {}
    for set_name in ('maintained', 'reduced'):
        hit_rates, false_alarm_rates = rate_tables[set_name]
        concept_cr = averaged_participant_cr(np.ones(hit_rates.shape[0]), hit_rates, false_alarm_rates)
        full_usable[set_name] = np.isfinite(concept_cr)
        full_ranked[set_name] = RankedColumns(features[concepts_index[set_name]], full_usable[set_name])
        full_sample_sum[set_name] = float(np.sum(full_ranked[set_name].spearman_with(concept_cr[full_usable[set_name]]) ** 2))
        report('  %s set: summed rho^2 on the full sample (no resampling) %.10f' % (set_name, full_sample_sum[set_name]))

    reduction_ledgers = 'L3-053;L3-056;L3-057;L5-084;L3-125'
    reduction_unit = 'participants per set (150 maintained; 150 reduced), 49 features'
    paper_seed = '42 (MATLAB rng, twister)'
    run_specs = [('1,000 participant resamples, MATLAB rng(42)', paper_values, int(paper_variance.nBootstraps),
                  paper_seed, 'percentile bootstrap (MATLAB prctile)')]
    for run_name, values, resamples, seed, method in run_specs:
        tests.add(test_id='T19', ledger_ids=reduction_ledgers,
                  location='Fig. 7d caption; %s; Methods, Experimental Procedures and Analysis' % FIG7_SECTION,
                  test=('Fig. 7d (%s): percent reduction of the summed squared feature-memorability correlation '
                        '(sum over 49 features of Spearman rho^2 with concept CR), maintained to reduced set; mean over '
                        'resamples; participants resampled within each set; one-sided p for the predicted direction '
                        '(share of resamples with the reduced-set sum at least the maintained-set sum)' % run_name),
                  statistic_name='percent reduction', statistic=values['percent'], n='150; 150', unit=reduction_unit,
                  p=values['p_one_sided'], p_type='bootstrap one-sided', ci_method='none',
                  resampling_unit='participants, within each set', n_resamples=resamples, seed=seed)
        for set_label, key in (('maintained', 'maintained'), ('reduced', 'reduced')):
            # The bar of Fig. 7d is the mean over the resamples. Resampling participants adds sampling
            # noise to every concept's CR, which weakens the correlations with the features, so the mean
            # over the resamples (and its interval) lies below the value of the full sample, given beside it.
            tests.add(test_id='T19', ledger_ids=reduction_ledgers, location='Fig. 7d (bars)',
                      test='Fig. 7d (%s): summed squared feature-memorability correlation in the %s set (bar height and '
                           'error bar: the mean and the 2.5 and 97.5 percentiles over the resamples); the effect size '
                           'column gives the value of the full sample (no resampling), which the resampled values lie '
                           'below' % (run_name, set_label),
                      statistic_name='summed rho^2 (mean over the bootstrap resamples)', statistic=values[key], n=150,
                      unit='participants, 49 features', p_type='none',
                      effect_size_name='summed rho^2 on the full sample (no resampling)',
                      effect_size=full_sample_sum[key],
                      ci_low=values[key + '_ci'][0], ci_high=values[key + '_ci'][1], ci_method=method,
                      resampling_unit='participants', n_resamples=resamples, seed=seed)

    # ---------------- T52: category shares of the reduced set ----------------
    # The category of a concept is its label in shared/things/concept_table.csv; 'NA' marks concepts
    # without a higher-level category and is not counted as a category.
    report('')
    labels = concept_table['category_label'].tolist()
    reduced_slots = slot_counts['reduced']
    slots_per_category = {}
    for concept_index, label in enumerate(labels):
        slots_per_category[label] = slots_per_category.get(label, 0) + reduced_slots[concept_index]
    total_slots = float(np.sum(reduced_slots))
    named = sorted([label for label in slots_per_category if label != 'NA'],
                   key=lambda label: -slots_per_category[label])
    largest, second = named[0], named[1]
    largest_share = 100 * slots_per_category[largest] / total_slots
    second_share = 100 * slots_per_category[second] / total_slots
    no_category_share = 100 * slots_per_category.get('NA', 0) / total_slots
    tests.add(test_id='T52', ledger_ids='L3-058',
              location='Reducing Variability In Distinctiveness Reduces Variability in Memorability',
              test=('Largest share of one higher-level category among the reduced set\'s {:,} images: {} {:.1f} '
                    'percent, then {} {:.1f} percent ({} named categories; category labels of concept_table.csv); '
                    'concepts without a category {:.1f} percent; descriptive, no test'.format(
                        int(total_slots), largest, largest_share, second, second_share, len(named),
                        no_category_share)),
              statistic_name='largest category share (percent of images)', statistic=largest_share, n=int(total_slots),
              unit='images (slots) of the reduced set', p_type='none', effect_size_name='share (percent)',
              effect_size=largest_share, ci_method='none')
    report('T52: %s %.2f%%, %s %.2f%%, no category %.2f%% of %d slots' % (
        largest, largest_share, second, second_share, no_category_share, total_slots))

    # ---------------- T54: participant flow (Methods, Participants) ----------------
    report('')
    flow_rows = []
    for set_name in list(SET_NAMES) + ['all three sets']:
        if set_name == 'all three sets':
            own_flags = flags
            kept = sum(len(set_data[name]['participants']) for name in SET_NAMES)
        else:
            own_flags = flags[flags['image_set'] == set_name]
            kept = len(set_data[set_name]['participants'])
        rules_met = (own_flags['same_key'].astype(int) + own_flags['too_fast'].astype(int)
                     + own_flags['too_slow'].astype(int) + own_flags['low_cr'].astype(int))
        flow = dict(processed=len(own_flags), same_key=int(own_flags['same_key'].sum()),
                    too_fast=int(own_flags['too_fast'].sum()), too_slow=int(own_flags['too_slow'].sum()),
                    low_cr=int(own_flags['low_cr'].sum()), more_than_one=int(np.sum(rules_met > 1)),
                    excluded=int(own_flags['excluded'].sum()), kept=kept)
        flow_rows.append((set_name, flow))
        tests.add(test_id='T54', ledger_ids='ADD-011c;L3-038;L5-071;L5-072',
                  location='Methods, Experimental Procedures and Analysis',
                  test=('Participant flow, image-set experiments, %s: %d sessions; excluded %d (one key on more than 80 '
                        'percent of trials %d, more than 20 percent of trials under 200 ms %d, more than 20 percent over '
                        '7,500 ms %d, hit rate minus false alarm rate below 0.05 %d; %d meet more than one rule); '
                        'kept %d (the cap of 150 per set removes nobody)'
                        % (set_name, flow['processed'], flow['excluded'], flow['same_key'], flow['too_fast'],
                           flow['too_slow'], flow['low_cr'], flow['more_than_one'], flow['kept'])),
                  statistic_name='participants kept', statistic=flow['kept'], n=flow['processed'],
                  unit='sessions before exclusion', p_type='none', ci_method='none')
        report('T54 %s: %d sessions, excluded %d (same key %d, too fast %d, too slow %d, CR < 0.05 %d; %d more than one rule), kept %d' % (
            set_name, flow['processed'], flow['excluded'], flow['same_key'], flow['too_fast'], flow['too_slow'],
            flow['low_cr'], flow['more_than_one'], flow['kept']))

    # ---------------- cross-checks against the MATLAB outputs of run_module.m ----------------
    report('')
    cross_check_matlab_outputs(fig6, observed, set_data, flow_rows, sd_maintained, sd_reduced, full_sample_sum,
                               concept_names)

    tests.write(PATHS['outputs'])
    report('wrote outputs/tests.csv (%d rows)' % len(tests.rows))
    rows_of_run_module = add_rows_to_numbers_csv(numbers)
    matches = [row['match'] for row in numbers.rows]
    report('wrote outputs/numbers.csv: %d rows of run_module.m and the %d Fig. 6 rows of this script (yes %d, no %d)'
           % (rows_of_run_module, len(numbers.rows), matches.count('yes'), matches.count('no')))
    for row in numbers.rows:
        if row['match'] == 'no':
            report('  NOT MATCHED: %s printed %s, computed %s' % (row['ledger_id'], row['paper_value'],
                                                                 row['computed_value']))
    if rows_of_run_module == 0:
        report('  (outputs/numbers.csv held no rows of run_module.m: run run_module.m, then this script)')
    report('finished in %.1f minutes' % ((time.time() - started) / 60))
    with open(os.path.join(PATHS['logs'], 'stats_fig6_fig7_log.txt'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(LOG_LINES) + '\n')


def read_csv_rows(file_name):
    with open(os.path.join(PATHS['outputs'], file_name), newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def add_rows_to_numbers_csv(numbers):
    """Write this script's rows into outputs/numbers.csv: rows with the same ledger ids (from an earlier run)
    are replaced, and the rows that run_module.m writes are kept. Returns the number of rows kept."""
    numbers_file = os.path.join(PATHS['outputs'], 'numbers.csv')
    own_ids = set(row['ledger_id'] for row in numbers.rows)
    other_rows = []
    if os.path.exists(numbers_file):
        for row in read_csv_rows('numbers.csv'):
            if row['ledger_id'] not in own_ids:
                other_rows.append({column: row.get(column, '') for column in numbers_table.COLUMNS})
    with open(numbers_file, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=numbers_table.COLUMNS)
        writer.writeheader()
        for row in other_rows + numbers.rows:
            writer.writerow(row)
    return len(other_rows)


def cross_check_matlab_outputs(fig6, observed, set_data, flow_rows, sd_maintained, sd_reduced, full_sample_sum,
                               concept_names):
    """The Python numbers must equal the MATLAB numbers of run_module.m (when its outputs exist)."""
    problems = []
    checked = []
    if os.path.exists(os.path.join(PATHS['outputs'], 'fig6_correlations.csv')):
        panel_of_test = {'T12': 'c', 'T13': 'd', 'T14': 'e', 'T15': 'f', 'T16': 'g', 'T17': 'h'}
        matlab_r = {row['panel']: float(row['pearson_r']) for row in read_csv_rows('fig6_correlations.csv')}
        for test_id, panel in panel_of_test.items():
            if abs(matlab_r[panel] - observed[test_id]) > 1e-9:
                problems.append('Fig. 6%s r: MATLAB %.10f, Python %.10f' % (panel, matlab_r[panel], observed[test_id]))
        checked.append('Fig. 6 correlations')
    if os.path.exists(os.path.join(PATHS['outputs'], 'concept_scores_maintained_changed.csv')):
        largest = 0.0
        for row in read_csv_rows('concept_scores_maintained_changed.csv'):
            data = set_data[row['image_set']]
            position = int(np.flatnonzero(data['concepts'] == int(row['concept_number']))[0])
            python_cr = fig6[row['image_set']]['concept_cr'][position]
            matlab_cr = float(row['cr'])
            if np.isnan(python_cr) != np.isnan(matlab_cr):
                problems.append('concept CR missing in one language: %s' % row['concept'])
            elif not np.isnan(python_cr):
                largest = max(largest, abs(python_cr - matlab_cr))
        if largest > 1e-12:
            problems.append('concept CR: largest difference %.2g' % largest)
        checked.append('per-concept CR of the maintained and changed sets (largest difference %.1g)' % largest)
    if os.path.exists(os.path.join(PATHS['outputs'], 'participant_flow.csv')):
        matlab_flow = {row['image_set']: row for row in read_csv_rows('participant_flow.csv')}
        for set_name, flow in flow_rows:
            row = matlab_flow[set_name]
            pairs = [('processed', 'processed'), ('excluded_same_key', 'same_key'), ('excluded_fast', 'too_fast'),
                     ('excluded_slow', 'too_slow'), ('excluded_low_cr', 'low_cr'),
                     ('excluded_more_than_one_rule', 'more_than_one'), ('excluded_total', 'excluded'), ('kept', 'kept')]
            for matlab_column, python_key in pairs:
                if int(float(row[matlab_column])) != flow[python_key]:
                    problems.append('participant flow %s %s: MATLAB %s, Python %d' % (set_name, matlab_column,
                                                                                      row[matlab_column], flow[python_key]))
        checked.append('participant flow')
    if os.path.exists(os.path.join(PATHS['outputs'], 'fig7_variance_summary.csv')):
        summary = {row['quantity']: float(row['value']) for row in read_csv_rows('fig7_variance_summary.csv')}
        pairs = [('summed similarity, maintained set, SD over concepts', sd_maintained),
                 ('summed similarity, reduced set, SD over concepts', sd_reduced),
                 ('summed rho^2, maintained set, full sample (no resampling)', full_sample_sum['maintained']),
                 ('summed rho^2, reduced set, full sample (no resampling)', full_sample_sum['reduced'])]
        for name, python_value in pairs:
            if abs(summary[name] - python_value) > 1e-9:
                problems.append('%s: MATLAB %.10f, Python %.10f' % (name, summary[name], python_value))
        checked.append('summed-similarity SDs and the full-sample summed rho^2')
    if checked:
        report('cross-checks against the MATLAB outputs: %s' % '; '.join(checked))
    else:
        report('cross-checks skipped: run run_module.m first to compare with the MATLAB outputs')
    if problems:
        for problem in problems:
            report('  DIFFERS: ' + problem)
        raise RuntimeError('the Python statistics do not match the MATLAB outputs')
    if checked:
        report('  all equal')


if __name__ == '__main__':
    main()
