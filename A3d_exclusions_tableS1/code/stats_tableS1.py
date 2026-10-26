"""
stats_tableS1.py: the uncertainty statistics of SI Appendix, Table S1 (T35, T36, T37): the image-set
results with and without the participant exclusions (SI Appendix, Section 3).

WHAT IT COMPUTES (rows of outputs/tests.csv; the convention is in statistics/README.md)
  Each image set is analyzed twice, as in exclusion_regimes.py (the two columns of Table S1):
    without exclusions   (all_run)    every session run: 205 maintained, 195 changed, 202 reduced
    with exclusions      (reported)   the rules of the Methods: 150 per set
  Corrected recognition (CR) of a concept = its hit rate minus its false alarm rate, pooled over
  all kept trials of the concept ("old" = responses 1 to 3).

  T35  Table S1, concept level. Spearman rho, across the concepts of one set, between concept CR and
       the concept's mean summed similarity to the 40 items studied in its block (per test trial:
       the mean of normpdf(similarity, 1, 2) over the 40 same-block study concepts, the tested
       concept removed once; a concept's value is the mean over its test trials). This is a
       different summed-similarity measure from the set-level weighted summed similarity of Fig. 6
       (T36). For each set, without and with exclusions: rho, n concepts, the participants, the
       two-sided parametric p and the 95% interval over 10,000 concept resamples (paired: both
       columns are computed on the same resampled concepts).
  T36  Table S1, feature level (Fig. 6d and 6g without and with exclusions). Pearson r across the 49
       SPoSE features between the set's summed-similarity profile (for each feature, the Spearman
       correlation across the set's concepts between the feature and the set-level weighted summed
       similarity) and the set's feature-memorability profile (for each feature, the Spearman
       correlation with concept CR). For the maintained and changed sets, without and with
       exclusions: r, n = 49 features and the 95% interval over 10,000 participant resamples.
       Concept CR and the 49 feature correlations are recomputed in every resample; the
       summed-similarity profile is a fixed property of the set, and the features are not
       resampled. The with-exclusion rows are the statistics of Fig. 6d and 6g (module 06, T13 and
       T16); their intervals here come from this script's own random draws.
  T37  Table S1, variance explained and the reduction (the Fig. 7d statistic). Sum over the 49
       features of the squared Spearman correlation between the feature and concept CR (CR here:
       each participant's own hit and false alarm rate for the concept, averaged over the
       participants, as in Fig. 7d), in the maintained and the reduced set, and its percent
       reduction from maintained to reduced. Without and with exclusions, from exclusion_regimes.py
       (1,000 participant resamples per set, seed 7): each set's sum (the mean over the resamples
       with its 95% percentile interval) and the mean percent reduction with its one-sided p (share
       of resamples with the reduced-set sum at least the maintained-set sum), with no interval:
       Table S1 reports this test by its p alone.
       Module 06 computes the same statistic with the exclusions for Fig. 7d (its test T19).
  No correction for multiple comparisons is applied (every test is reported on its own).

INPUTS (all inside the release)
  06_image_set_experiments_fig6_fig7/data/<set>_trials.csv through shared/python/imageset_loader.py
  06_image_set_experiments_fig6_fig7/data/sampled_image_sets.mat (images of each concept per set)
  shared/things/concept_table.csv (the 49 SPoSE features)
  outputs/participant_flags.csv, concept_level_by_regime.csv, feature_level_by_regime.csv,
    fig7d_reduction_by_regime.csv (written by exclusion_regimes.py; run it first)
OUTPUTS: outputs/tests.csv, outputs/logs/stats_tableS1_log.txt
  (the T37 rows, computed by exclusion_regimes.py with seed 7, name that script in the script column)

SEED: 20261013 (numpy.random.default_rng), one generator for the whole script, drawn in this order:
  1. T35 concept resamples: maintained, changed, reduced (one draw per resample, shared by the two
     columns)
  2. one draw of the participants per resample for each set, without then with exclusions
  3. T36 participant resamples: maintained then changed; in each, without exclusions, with
     exclusions, then one draw per resample of the participants kept by the stricter
     corrected-recognition floor of 0.10 (147 maintained, 143 changed)
  The draws of 2 and the stricter-floor draws of 3 give no result and hold fixed places in the
  order, so that the seed reproduces the reported intervals.

RUN (from the repository root; about 30 seconds):
  python A3d_exclusions_tableS1/code/stats_tableS1.py
  (--resamples N sets a smaller number for a quick check of the code; the default, 10,000, gives
  the values in outputs/)
"""
import argparse
import os
import sys
import time

# Use at most two processor threads for the matrix arithmetic. Must be set before numpy is
# imported.
for thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(thread_setting, '2')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.io  # noqa: E402
from scipy.stats import pearsonr, rankdata, spearmanr  # noqa: E402

sys.dont_write_bytecode = True   # no __pycache__ folder in shared/ when this step is run on its own
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (first: it stops Python from writing __pycache__ folders)
import bootstrap  # noqa: E402
import concept_tables  # noqa: E402
import imageset_loader  # noqa: E402
import tests_table  # noqa: E402

PATHS = release_paths.module_paths(__file__, make_output_folders=False)
SCRIPT_NAME = 'code/stats_tableS1.py'
SEED = bootstrap.RELEASE_SEED
IMAGE_SET_DATA = os.path.join(release_paths.find_module('06'), 'data')
SET_NAMES = ('maintained', 'changed', 'reduced')
SAMPLED_SET_STRUCT = {'maintained': 'representativeDataset', 'changed': 'flipMemorabilityDataset',
                      'reduced': 'equatedDataset'}
REGIMES = ('all_run', 'reported')      # the two columns of Table S1
# The test text of every row starts with 'Table S1, <result>, <column>'; code/make_tableS1.py finds the
# cells of the table by it.
REGIME_TEXT = {'all_run': 'without exclusions (every session run)',
               'reported': 'with exclusions (the rules of the Methods)'}
LOCATION_TABLE = 'SI Appendix, Table S1'
LOCATION_TABLE_AND_TEXT = ('SI Appendix, Table S1; SI Appendix, Section 3, The Image Set Results Do Not Depend '
                           'on the Exclusions')
# the ledger items (shared/ledger/paper_numbers.csv) of the cells of Table S1
CONCEPT_ROW_LEDGER_ID = {'maintained': 'L7-056', 'changed': 'L7-141', 'reduced': 'L7-142'}
FEATURE_ROW_LEDGER_ID = {'maintained': 'L7-058', 'changed': 'L7-143'}
VARIANCE_ROW_LEDGER_ID = {'maintained': 'L7-053', 'reduced': 'L7-144'}
KERNEL_MEAN = 1.0      # the similarity kernel normpdf(similarity, 1, 2)
KERNEL_SD = 2.0
PARTICIPANTS_KEPT_PER_SET = 150
TRIAL_MIN_MS = 200     # trial filter applied after the participant rules (Methods)
TRIAL_MAX_MS = 10000
REDUCTION_RESAMPLES = 1000
# the T37 rows are computed by exclusion_regimes.py (seed 7) and name that script
REDUCTION_SCRIPT = 'code/exclusion_regimes.py'

LOG_LINES = []


def report(text=''):
    """Print a line and keep it for the log file."""
    print(text, flush=True)
    LOG_LINES.append(text)


# ------------------------------------------------------------------------------------------------
# Small building blocks
# ------------------------------------------------------------------------------------------------

def normal_density(x, mean, sd):
    """The same as MATLAB normpdf(x, mean, sd)."""
    return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def pearson_r(x, y):
    """Pearson correlation of two equally long lists of numbers."""
    return float(np.corrcoef(np.asarray(x, dtype=float), np.asarray(y, dtype=float))[0, 1])


def spearman_rho(x, y):
    """Spearman correlation: the Pearson correlation of the average ranks (ties share a rank)."""
    return pearson_r(rankdata(x), rankdata(y))


def percentile_interval(values):
    """The 2.5 and 97.5 percentiles (numpy's linear interpolation; resamples that gave NaN ignored)."""
    low, high = bootstrap.percentile_interval(values)
    return float(low), float(high)


def draw_participant_weights(generator, number_of_participants):
    """One bootstrap resample of the participants: how often each was drawn (with replacement)."""
    drawn = generator.integers(0, number_of_participants, number_of_participants)
    return np.bincount(drawn, minlength=number_of_participants).astype(float)


class RankedColumns:
    """The average ranks of every column of a concepts x features table over a chosen set of
    concepts, centered, ready to be correlated with the ranks of one memory score per concept."""

    def __init__(self, columns, usable):
        ranks = rankdata(columns[usable], axis=0)
        self.centred = ranks - ranks.mean(axis=0)
        self.norms = np.sqrt(np.sum(self.centred ** 2, axis=0))

    def spearman_with(self, values_on_usable_concepts):
        """Spearman correlation of every column with the given values (one per usable concept)."""
        value_ranks = rankdata(values_on_usable_concepts)
        centred_values = value_ranks - value_ranks.mean()
        return (self.centred.T @ centred_values) / (self.norms * np.sqrt(np.sum(centred_values ** 2)))


def spearman_with_columns(columns, values):
    """Spearman correlation of every column of `columns` (concepts x features) with `values`, over
    the concepts where `values` is a number (MATLAB corr(..., 'Type', 'Spearman', 'rows', 'complete'))."""
    usable = np.isfinite(values)
    return RankedColumns(columns, usable).spearman_with(values[usable])


def summed_similarity_in_set(kernel, images_per_concept):
    """Set-level summed similarity of every concept (Fig. 6): the kernel similarity of the concept to
    every concept, weighted by that concept's share of the set's images, after removing one image
    of the concept itself (weights renormalized). NaN for concepts that are not in the set."""
    number_of_concepts = len(images_per_concept)
    summed = np.full(number_of_concepts, np.nan)
    for concept in range(number_of_concepts):
        if images_per_concept[concept] > 0:
            counts = images_per_concept.astype(float).copy()
            counts[concept] = counts[concept] - 1
            weights = counts / counts.sum()
            summed[concept] = float(np.sum(kernel[concept, :] * weights))
    return summed


# ------------------------------------------------------------------------------------------------
# Participants of each regime, and their per-concept trial counts
# ------------------------------------------------------------------------------------------------

def regime_participants(flags, set_name, regime):
    """The participants a regime keeps in one set, in id order (exclusion_regimes.py, kept_participants)."""
    one_set = flags[flags['image_set'] == set_name]
    if regime == 'all_run':
        return sorted(one_set['participant'])
    if regime == 'reported':
        passing = one_set[~one_set['excluded_005']]
    elif regime == 'stricter_010':
        passing = one_set[~one_set['excluded_010']]
    else:
        raise ValueError(regime)
    return sorted(passing['participant'])[:PARTICIPANTS_KEPT_PER_SET]


def regime_trials(set_trials, participants, regime):
    """The kept trials: the regime's participants; with exclusions, only trials of 200 to 10,000 ms."""
    kept = set_trials[set_trials['participant'].isin(set(participants))]
    if regime != 'all_run':
        kept = kept[(kept['rt_ms'] >= TRIAL_MIN_MS) & (kept['rt_ms'] <= TRIAL_MAX_MS)]
    return kept


def participant_concept_tables(kept_trials, participants, set_concepts):
    """Participants x concepts tables of one set and regime: old trials, hits (old trials called
    old), new trials, false alarms (new trials called old), and the sum and number of the trials'
    summed similarity to the 40 items studied in the same block."""
    row_of_participant = {participant: row for row, participant in enumerate(participants)}
    column_of_concept = {concept: column for column, concept in enumerate(set_concepts)}
    shape = (len(participants), len(set_concepts))
    tables = {'old_trials': np.zeros(shape), 'hits': np.zeros(shape), 'new_trials': np.zeros(shape),
              'false_alarms': np.zeros(shape), 'similarity_sum': np.zeros(shape), 'similarity_count': np.zeros(shape)}
    for participant, concept, is_new, response, similarity in zip(
            kept_trials['participant'], kept_trials['concept_number'], kept_trials['is_new'],
            kept_trials['response_1to6'], kept_trials['summed_similarity_same_block']):
        row = row_of_participant[participant]
        column = column_of_concept[concept]
        said_old = 1.0 if response <= 3 else 0.0
        if is_new == 1:
            tables['new_trials'][row, column] += 1
            tables['false_alarms'][row, column] += said_old
        else:
            tables['old_trials'][row, column] += 1
            tables['hits'][row, column] += said_old
        if np.isfinite(similarity):
            tables['similarity_sum'][row, column] += similarity
            tables['similarity_count'][row, column] += 1
    return tables


def pooled_concept_scores(weights, tables):
    """Each concept's CR (hit rate minus false alarm rate) and mean summed similarity, pooled over
    the trials of the participants, each participant counted `weights` times (all ones = the
    sample itself; in a bootstrap resample, the number of times a participant was drawn). NaN when
    a concept has no old, no new or no similarity trial."""
    old_trials = weights @ tables['old_trials']
    hits = weights @ tables['hits']
    new_trials = weights @ tables['new_trials']
    false_alarms = weights @ tables['false_alarms']
    similarity_sum = weights @ tables['similarity_sum']
    similarity_count = weights @ tables['similarity_count']
    with np.errstate(divide='ignore', invalid='ignore'):
        hit_rate = np.where(old_trials > 0, hits / old_trials, np.nan)
        false_alarm_rate = np.where(new_trials > 0, false_alarms / new_trials, np.nan)
        mean_similarity = np.where(similarity_count > 0, similarity_sum / similarity_count, np.nan)
    return hit_rate - false_alarm_rate, mean_similarity


def averaged_participant_cr(weights, tables):
    """Fig. 7d estimator: the mean over participants of each participant's own hit rate for the
    concept minus the mean of their own false alarm rates (participants without such a trial are
    left out; a participant drawn twice counts twice)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        hit_rates = np.where(tables['old_trials'] > 0, tables['hits'] / tables['old_trials'], np.nan)
        false_alarm_rates = np.where(tables['new_trials'] > 0, tables['false_alarms'] / tables['new_trials'], np.nan)
    hit_present = np.isfinite(hit_rates)
    false_alarm_present = np.isfinite(false_alarm_rates)
    hit_sum = weights @ np.where(hit_present, hit_rates, 0.0)
    hit_count = weights @ hit_present
    false_alarm_sum = weights @ np.where(false_alarm_present, false_alarm_rates, 0.0)
    false_alarm_count = weights @ false_alarm_present
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_hit = np.where(hit_count > 0, hit_sum / hit_count, np.nan)
        mean_false_alarm = np.where(false_alarm_count > 0, false_alarm_sum / false_alarm_count, np.nan)
    return mean_hit - mean_false_alarm


def full_sample_summed_rho_squared(entry, features):
    """The summed squared feature-memorability correlation of the full sample (no resampling): every
    participant counted once, concept CR averaged over participants as in Fig. 7d, the sum over the 49
    features of each feature's squared Spearman correlation with concept CR. No random numbers."""
    tables = entry['tables']
    concept_cr = averaged_participant_cr(np.ones(tables['hits'].shape[0]), tables)
    usable = np.isfinite(concept_cr)
    ranked = RankedColumns(features[entry['concepts'] - 1], usable)
    return float(np.sum(ranked.spearman_with(concept_cr[usable]) ** 2))


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Uncertainty statistics T35 to T37 (SI Appendix, Table S1).')
    parser.add_argument('--resamples', type=int, default=10000, help='bootstrap resamples (default 10,000)')
    arguments = parser.parse_args()
    number_of_resamples = arguments.resamples
    started = time.time()
    report('stats_tableS1.py: seed %d, %d bootstrap resamples' % (SEED, number_of_resamples))
    if number_of_resamples != 10000:
        report('NOTE: fewer resamples than the default (a quick check of the code)')
    generator = np.random.default_rng(SEED)
    tests = tests_table.TestsTable(script=SCRIPT_NAME)

    # ---------------- inputs ----------------
    needed_outputs = ['participant_flags.csv', 'concept_level_by_regime.csv', 'feature_level_by_regime.csv',
                      'fig7d_reduction_by_regime.csv']
    for file_name in needed_outputs:
        if not os.path.exists(os.path.join(PATHS['outputs'], file_name)):
            sys.exit('stats_tableS1.py: outputs/%s is missing; run code/exclusion_regimes.py first' % file_name)
    flags = pd.read_csv(os.path.join(PATHS['outputs'], 'participant_flags.csv'))
    concept_level_by_regime = pd.read_csv(os.path.join(PATHS['outputs'], 'concept_level_by_regime.csv'))
    feature_level_by_regime = pd.read_csv(os.path.join(PATHS['outputs'], 'feature_level_by_regime.csv'))
    reduction_by_regime = pd.read_csv(os.path.join(PATHS['outputs'], 'fig7d_reduction_by_regime.csv'))

    concept_table = concept_tables.load_concept_table()
    features = concept_tables.spose_features(concept_table)
    human_kernel = normal_density(concept_tables.concept_similarity(features), KERNEL_MEAN, KERNEL_SD)
    sampled_sets = scipy.io.loadmat(os.path.join(IMAGE_SET_DATA, 'sampled_image_sets.mat'), squeeze_me=True,
                                    struct_as_record=False)
    trials = imageset_loader.load_trials(SET_NAMES)

    # per set and regime: participants, kept trials, the participants x concepts tables
    data = {}
    for set_name in SET_NAMES:
        set_trials = trials[trials['image_set'] == set_name]
        set_concepts = np.array(sorted(set_trials['concept_number'].unique()))      # tested concepts (THINGS numbers)
        images_per_concept = np.asarray(sampled_sets[SAMPLED_SET_STRUCT[set_name]].numImagesForConcept, dtype=float)
        if np.any(images_per_concept[set_concepts - 1] == 0):
            raise ValueError('a tested concept is not part of the %s set' % set_name)
        set_level_similarity = summed_similarity_in_set(human_kernel, images_per_concept)
        for regime in REGIMES:
            participants = regime_participants(flags, set_name, regime)
            kept = regime_trials(set_trials, participants, regime)
            tables = participant_concept_tables(kept, participants, set_concepts)
            data[(set_name, regime)] = dict(participants=participants, trials=kept, concepts=set_concepts,
                                            tables=tables, set_level_similarity=set_level_similarity)
            report('%-10s %-13s %3d participants, %6d test trials, %4d tested concepts' % (
                set_name, regime, len(participants), len(kept), len(set_concepts)))

    # =============================================================================================
    # T35: concept-level rho (Table S1, concept level)
    # =============================================================================================
    report('')
    report('T35 (Table S1, concept level): Spearman rho across concepts, concept CR vs mean summed similarity to '
           'the 40 items studied in the block')
    observed_rho = {}
    for set_name in SET_NAMES:
        for regime in REGIMES:
            entry = data[(set_name, regime)]
            concept_cr, mean_similarity = pooled_concept_scores(np.ones(len(entry['participants'])), entry['tables'])
            usable = np.isfinite(concept_cr) & np.isfinite(mean_similarity)
            rho, p_value = spearmanr(concept_cr[usable], mean_similarity[usable])
            entry.update(concept_cr=concept_cr, mean_similarity=mean_similarity, usable=usable)
            observed_rho[(set_name, regime)] = (float(rho), float(p_value), int(np.sum(usable)))
            # the same value as exclusion_regimes.py; the csv keeps 6 significant digits
            saved_row = concept_level_by_regime[(concept_level_by_regime['image_set'] == set_name)
                                                & (concept_level_by_regime['regime'] == regime)].iloc[0]
            if abs(rho - saved_row['rho']) > 5e-6 or int(np.sum(usable)) != int(saved_row['concepts']):
                raise RuntimeError('T35 %s %s differs from outputs/concept_level_by_regime.csv' % (set_name, regime))
            report('  %-10s %-13s rho = %+.6f, p = %.3g, %d concepts, %d participants' % (
                set_name, regime, rho, p_value, np.sum(usable), len(entry['participants'])))

    # 1. concept resamples (paired: both columns of a set use the same resampled concepts)
    concept_bootstrap = {}
    for set_name in SET_NAMES:
        usable_by_regime = [data[(set_name, regime)]['usable'] for regime in REGIMES]
        for usable in usable_by_regime[1:]:
            if not np.array_equal(usable, usable_by_regime[0]):
                raise RuntimeError('the usable concepts of the %s set differ between regimes; the paired concept '
                                   'bootstrap assumes they do not' % set_name)
        usable_positions = np.flatnonzero(usable_by_regime[0])
        number_of_usable = len(usable_positions)
        resampled = {regime: np.zeros(number_of_resamples) for regime in REGIMES}
        for resample in range(number_of_resamples):
            drawn = usable_positions[generator.integers(0, number_of_usable, number_of_usable)]
            for regime in REGIMES:
                entry = data[(set_name, regime)]
                resampled[regime][resample] = spearman_rho(entry['concept_cr'][drawn], entry['mean_similarity'][drawn])
        concept_bootstrap[set_name] = resampled
    report('  concept resamples done (%.0f s)' % (time.time() - started))

    # 2. one draw of the participants per resample for each set, without then with exclusions: these draws
    #    give no result and hold their place in the order of the draws (SEED above)
    for set_name in SET_NAMES:
        for regime in REGIMES:
            number_of_participants = len(data[(set_name, regime)]['participants'])
            for _resample in range(number_of_resamples):
                generator.integers(0, number_of_participants, number_of_participants)

    for set_name in SET_NAMES:
        for regime in REGIMES:
            rho, p_value, number_of_concepts = observed_rho[(set_name, regime)]
            number_of_participants = len(data[(set_name, regime)]['participants'])
            unit = 'concepts (%s set; %d participants, %s)' % (set_name, number_of_participants, REGIME_TEXT[regime])
            low, high = percentile_interval(concept_bootstrap[set_name][regime])
            ledger_ids = ([CONCEPT_ROW_LEDGER_ID[set_name]] + (['L7-045'] if regime == 'all_run' else [])
                          + ['L7-046', 'L7-051', 'L7-061', 'L7-148', 'L3-046'])
            tests.add(test_id='T35', ledger_ids=';'.join(ledger_ids),
                      location=LOCATION_TABLE_AND_TEXT if regime == 'all_run' else LOCATION_TABLE,
                      test=('Table S1, concept level, %s set, %s: Spearman rho across concepts between concept CR '
                            '(trial-pooled) and the concept\'s mean summed similarity to the 40 items studied in its '
                            'block (mean over its test trials; not the set-level summed similarity of Fig. 6); '
                            'interval: concepts resampled' % (set_name, REGIME_TEXT[regime])),
                      statistic_name='Spearman rho', statistic=rho, n=number_of_concepts, unit=unit, df='',
                      p=p_value, p_type='parametric two-sided', effect_size_name='rho', effect_size=rho,
                      ci_low=low, ci_high=high, ci_method='percentile bootstrap', resampling_unit='concepts',
                      n_resamples=number_of_resamples, seed=SEED)
            report('  %-10s %-13s rho %.3f, 95%% CI [%.3f, %.3f] (concepts)' % (set_name, regime, rho, low, high))

    # =============================================================================================
    # T36: feature-level r (Table S1, feature level: the statistics of Fig. 6d and 6g without and with exclusions)
    # =============================================================================================
    report('')
    report('T36 (Table S1, feature level): Pearson r across the 49 features, summed-similarity profile vs '
           'memorability profile')
    feature_sets = ('maintained', 'changed')
    observed_r = {}
    for set_name in feature_sets:
        similarity_profile = spearman_with_columns(features, data[(set_name, 'all_run')]['set_level_similarity'])
        for regime in REGIMES:
            entry = data[(set_name, regime)]
            concepts_index = entry['concepts'] - 1
            concept_cr, _ = pooled_concept_scores(np.ones(len(entry['participants'])), entry['tables'])
            usable = np.isfinite(concept_cr)
            ranked = RankedColumns(features[concepts_index], usable)
            memorability_profile = ranked.spearman_with(concept_cr[usable])
            r_value, _ = pearsonr(similarity_profile, memorability_profile)
            entry.update(similarity_profile=similarity_profile, feature_cr=concept_cr, feature_usable=usable,
                         ranked=ranked)
            observed_r[(set_name, regime)] = float(r_value)
            saved_row = feature_level_by_regime[(feature_level_by_regime['image_set'] == set_name)
                                                & (feature_level_by_regime['regime'] == regime)].iloc[0]
            if abs(r_value - saved_row['r_summed_similarity_human']) > 5e-6:
                raise RuntimeError('T36 %s %s differs from outputs/feature_level_by_regime.csv' % (set_name, regime))
            report('  %-10s %-13s r = %+.6f' % (set_name, regime, r_value))

    # 3. participant resamples of each set, then one draw per resample of the participants kept by the
    #    stricter floor (these draws give no result and hold their place in the order of the draws, SEED above)
    participant_bootstrap_r = {}
    for set_name in feature_sets:
        for regime in REGIMES:
            entry = data[(set_name, regime)]
            concepts_index = entry['concepts'] - 1
            number_of_participants = len(entry['participants'])
            values = np.zeros(number_of_resamples)
            resamples_with_fewer_concepts = 0
            for resample in range(number_of_resamples):
                weights = draw_participant_weights(generator, number_of_participants)
                concept_cr, _ = pooled_concept_scores(weights, entry['tables'])
                usable = np.isfinite(concept_cr)
                if np.array_equal(usable, entry['feature_usable']):
                    ranked = entry['ranked']
                else:
                    # a concept lost all its old (or new) trials in this resample: rank on the remaining concepts
                    resamples_with_fewer_concepts += 1
                    ranked = RankedColumns(features[concepts_index], usable)
                profile = ranked.spearman_with(concept_cr[usable])
                values[resample] = pearson_r(entry['similarity_profile'], profile)
            participant_bootstrap_r[(set_name, regime)] = values
            report('  %-10s %-13s %d of %d resamples had fewer usable concepts' % (
                set_name, regime, resamples_with_fewer_concepts, number_of_resamples))
        number_kept_by_stricter_floor = len(regime_participants(flags, set_name, 'stricter_010'))
        for _resample in range(number_of_resamples):
            generator.integers(0, number_kept_by_stricter_floor, number_kept_by_stricter_floor)
    report('  participant resamples done (%.0f s)' % (time.time() - started))

    for set_name in feature_sets:
        for regime in REGIMES:
            r_value = observed_r[(set_name, regime)]
            entry = data[(set_name, regime)]
            unit = 'features (%s set; %d participants, %d concepts, %s)' % (
                set_name, len(entry['participants']), int(np.sum(entry['feature_usable'])), REGIME_TEXT[regime])
            low, high = percentile_interval(participant_bootstrap_r[(set_name, regime)])
            same_as_fig6 = (' (the statistic of Fig. %s, module 06 %s; separate random draws)'
                            % ('6d' if set_name == 'maintained' else '6g', 'T13' if set_name == 'maintained' else 'T16')
                            if regime == 'reported' else '')
            tests.add(test_id='T36', ledger_ids='%s;L7-047;L7-052;L3-046' % FEATURE_ROW_LEDGER_ID[set_name],
                      location=LOCATION_TABLE,
                      test=('Table S1, feature level, %s set, %s: Pearson r across the 49 SPoSE features between the '
                            'set\'s summed-similarity profile (set-level weighted summed similarity) and its '
                            'feature-memorability profile (each value a Spearman correlation across concepts with '
                            'concept CR)%s; interval: participants resampled, concept CR and the profile recomputed in '
                            'each resample (the 49 features are not resampled)'
                            % (set_name, REGIME_TEXT[regime], same_as_fig6)),
                      statistic_name='Pearson r', statistic=r_value, n=49, unit=unit, df='', p_type='none',
                      effect_size_name='r', effect_size=r_value, ci_low=low, ci_high=high,
                      ci_method='percentile bootstrap', resampling_unit='participants', n_resamples=number_of_resamples,
                      seed=SEED)
            report('  %-10s %-13s r %.3f, 95%% CI [%.3f, %.3f] (participants)' % (set_name, regime, r_value, low, high))

    # =============================================================================================
    # T37: Table S1, variance explained and the reduction (the Fig. 7d statistic)
    # =============================================================================================
    report('')
    report('T37 (Table S1, variance explained and reduction): the Fig. 7d statistic without and with the exclusions')
    reduction_by_regime = reduction_by_regime.set_index('regime')
    for regime in REGIMES:
        row = reduction_by_regime.loc[regime]
        participants_text = '%d; %d' % (int(row['participants_maintained']), int(row['participants_reduced']))
        unit = 'participants per set (maintained; reduced), 49 features, %s' % REGIME_TEXT[regime]
        cite_t19 = ('; module 06 computes the same statistic for Fig. 7d (its test T19; 1,000 participant resamples, '
                    'MATLAB rng(42))'
                    if regime == 'reported' else '')
        common = dict(test_id='T37', n=participants_text, unit=unit, resampling_unit='participants, within each set',
                      n_resamples=REDUCTION_RESAMPLES, seed=7, script=REDUCTION_SCRIPT)
        tests.add(ledger_ids='L7-050;L7-060;L7-145;L7-147', location=LOCATION_TABLE_AND_TEXT,
                  test=('Table S1, reduction, %s: percent reduction from the maintained to the reduced set of the summed '
                        'squared feature-memorability correlation (sum over 49 features of Spearman rho^2 with concept '
                        'CR, CR averaged over participants as in Fig. 7d); mean over resamples; participants resampled '
                        'within each set (1,000 resamples, seed 7, exclusion_regimes.py)%s; one-sided p for the '
                        'predicted direction (share of resamples with the reduced-set sum at least the maintained-set '
                        'sum)' % (REGIME_TEXT[regime], cite_t19)),
                  statistic_name='percent reduction', statistic=float(row['percent_drop']), df='',
                  p=float(row['one_sided_p']), p_type='bootstrap one-sided', ci_method='none', **common)
        for set_name in ('maintained', 'reduced'):
            set_common = dict(common, n=int(row['participants_%s' % set_name]),
                              unit='participants, 49 features, %s' % REGIME_TEXT[regime], resampling_unit='participants',
                              ci_method='percentile bootstrap')
            # Table S1 gives the mean over the resamples; the effect size column gives the full-sample value
            # (every participant counted once, no resampling).
            tests.add(ledger_ids='%s;L7-049;L7-059;L7-061' % VARIANCE_ROW_LEDGER_ID[set_name], location=LOCATION_TABLE,
                      test=('Table S1, variance explained, %s set, %s: summed squared feature-memorability correlation '
                            '(the mean and the 2.5 and 97.5 percentiles over the resamples, exclusion_regimes.py); the '
                            'effect size column gives the value of the full sample (no resampling)'
                            % (set_name, REGIME_TEXT[regime])),
                      statistic_name='summed rho^2 (mean over the bootstrap resamples)', statistic=float(row[set_name]),
                      df='', p_type='none', effect_size_name='summed rho^2 on the full sample (no resampling)',
                      effect_size=full_sample_summed_rho_squared(data[(set_name, regime)], features),
                      ci_low=float(row['%s_low' % set_name]), ci_high=float(row['%s_high' % set_name]), **set_common)
        report('  %-13s exclusion_regimes.py (seed 7): %.2f%%, one-sided p %.3f' % (
            regime, row['percent_drop'], row['one_sided_p']))

    # ---------------- write ----------------
    tests.write(PATHS['outputs'])
    report('')
    report('No correction for multiple comparisons is applied (every test is reported on its own).')
    report('stats_tableS1.py finished in %.0f s' % (time.time() - started))
    os.makedirs(PATHS['logs'], exist_ok=True)
    with open(os.path.join(PATHS['logs'], 'stats_tableS1_log.txt'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(LOG_LINES) + '\n')


if __name__ == '__main__':
    main()
