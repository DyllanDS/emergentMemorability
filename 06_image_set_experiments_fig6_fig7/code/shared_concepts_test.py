"""
shared_concepts_test.py: test T19b, the Fig. 7d reduction over the concepts in both the maintained and the reduced set
(Representational Analyses, Reducing Variability In Distinctiveness Reduces Variability in Memorability).

WHAT IT COMPUTES (test T19b; the columns are described in statistics/README.md)
  Fig. 7d (test T19, stats_fig6_fig7.py) compares the two image sets on the sum over the 49 SPoSE features of
  the squared Spearman correlation between the feature and concept CR, each set over its own concepts (970
  maintained, 1,057 reduced). Only 559 concepts are in both sets, so part of that difference could come from
  which concepts each set holds. This script computes the same statistic over the shared concepts only: the
  same concepts with the same 49 features, tested in two image sets.
    - Concept CR is the Fig. 7d estimator over all of a set's kept test trials: the mean over participants of
      each participant's hit rate for the concept minus the mean of each participant's false alarm rate
      (participants without such a trial skipped; "old" = responses 1 to 3).
    - Shared concepts: the concepts with a CR in both sets (559).
    - For each set, the Spearman correlation across the shared concepts between each feature and CR (average
      ranks for ties); summed rho^2 = the sum of the 49 squared correlations.
    - Participant bootstrap (10,000 resamples, the concepts fixed): in each resample the 150
      participants of the maintained set and, separately, the 150 of the reduced set are drawn with
      replacement; CR and the 49 correlations of each set are recomputed (a participant drawn twice counts
      twice; a concept that lost all its old or all its new trials in a resample is left out of that set's
      correlations in that resample).
    - Reported: the percent reduction, 100 x (maintained sum minus reduced sum) / maintained
      sum, as the mean over the resamples with its 2.5 and 97.5 percentiles (the 95% CI), and the two-sided p
      (twice the smaller share of resamples on either side of no drop). The full-sample values (no
      resampling) are given too. Resampling participants adds noise to every concept's CR, which weakens the
      correlations with the features, so the means over the resamples lie below the full-sample sums (as the
      bars of Fig. 7d do).
  The estimator (the per-participant rate tables, the CR average, the Spearman correlations and the
  participant draws) is imported from stats_fig6_fig7.py, the code of T19.
  Self-check: over all concepts of each set, the full-sample summed rho^2 must equal that of T19 (0.781950
  maintained, 0.575111 reduced; the effect size column of the T19 bar rows in outputs/tests.csv), which shows
  that this script computes the Fig. 7d estimator on the same participants and trials.

INPUTS (read only)
  data/maintained_trials.csv, data/reduced_trials.csv of this module, through shared/python/imageset_loader.py
      (the exclusion criteria of the Methods: 150 kept participants per set)
  shared/things/concept_table.csv (the 49 SPoSE features)
  shared/ledger/paper_numbers.csv (the printed values of ledger item L3-065)

OUTPUTS (this module's outputs/)
  tests_shared_concepts.csv            the T19b row (columns of tests.csv)
  shared_concepts_numbers.csv          the numbers of test T19b and the full-sample values (quantity, value)
  numbers_shared_concepts.csv          the ledger row L3-065 judged against the printed values (columns of
                                       numbers.csv; read by collect_numbers.py)
  logs/shared_concepts_test_log.txt    the log of the run

SEED: 20261013 (numpy.random.default_rng). One generator, drawn in this order: for each of the 10,000
  resamples, the maintained set's 150 participant draws, then the reduced set's 150.

RUN (from the repository root; about 10 seconds):
  python 06_image_set_experiments_fig6_fig7/code/shared_concepts_test.py
  (--resamples N sets a smaller number for a quick check of the code; the default is 10,000)
"""
import argparse
import csv
import os
import sys

# Do not write __pycache__ folders into the release (set before the shared helpers are imported)
sys.dont_write_bytecode = True

# Use at most two processor threads for the matrix arithmetic. Must be set before numpy is imported.
for thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(thread_setting, '2')

import numpy as np  # noqa: E402

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_FOLDER)
sys.path.insert(1, os.path.join(CODE_FOLDER, '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (the release's folders)
import bootstrap  # noqa: E402
import concept_tables  # noqa: E402
import imageset_loader  # noqa: E402
import numbers_table  # noqa: E402
import tests_table  # noqa: E402
# The Fig. 7d estimator of test T19
from stats_fig6_fig7 import (RankedColumns, participant_concept_counts, participant_rate_tables,  # noqa: E402
                             averaged_participant_cr, draw_participant_weights)

PATHS = release_paths.module_paths(__file__)
SCRIPT_NAME = 'code/shared_concepts_test.py'
SEED = bootstrap.RELEASE_SEED                          # 20261013
SET_NAMES = ('maintained', 'reduced')
NUMBER_OF_FEATURES = 49
LEDGER_ID = 'L3-065'                                   # the sentence of the paper that reports this test

# T19's full-sample summed rho^2 over all concepts of each set (the effect size column of the T19 bar rows in
# outputs/tests.csv)
T19_FULL_SAMPLE_SUM = {'maintained': 0.7819496170, 'reduced': 0.5751106721}

LOG_LINES = []


def report(text=''):
    """Print a line and keep it for the log file."""
    print(text)
    LOG_LINES.append(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--resamples', type=int, default=10000, help='bootstrap resamples (default 10,000)')
    arguments = parser.parse_args()
    number_of_resamples = arguments.resamples
    report('shared_concepts_test.py: seed %d, %d participant resamples' % (SEED, number_of_resamples))
    if number_of_resamples != 10000:
        report('NOTE: a reduced number of resamples (a quick check of the code)')

    # ---------------- inputs ----------------
    features = concept_tables.spose_features(concept_tables.load_concept_table())   # concepts x 49
    trials = imageset_loader.load_trials(SET_NAMES, data_folder=PATHS['data'], with_summed_similarity=False)
    kept_trials, _ = imageset_loader.apply_exclusions(trials)

    # For each set: the participants, the concepts tested, each participant's hit and false alarm rate per
    # concept, and each concept's CR on the full sample (every participant once)
    set_data = {}
    for set_name in SET_NAMES:
        set_trials = kept_trials[kept_trials['image_set'] == set_name]
        participants = sorted(set_trials['participant'].unique())                # id order, as in T19
        set_concepts = np.array(sorted(set_trials['concept_number'].unique()))   # THINGS numbers (from 1)
        counts = participant_concept_counts(set_trials, participants, set_concepts)
        hit_rates, false_alarm_rates = participant_rate_tables(counts)
        full_sample_cr = averaged_participant_cr(np.ones(len(participants)), hit_rates, false_alarm_rates)
        set_data[set_name] = dict(participants=participants, concepts=set_concepts, hit_rates=hit_rates,
                                  false_alarm_rates=false_alarm_rates, full_sample_cr=full_sample_cr)
        report('%-10s set: %d participants kept, %d concepts tested, %d test trials kept' % (
            set_name, len(participants), len(set_concepts), len(set_trials)))

    # ---------------- self-check: T19's full-sample sums over all concepts of each set ----------------
    for set_name in SET_NAMES:
        data = set_data[set_name]
        concept_cr = data['full_sample_cr']
        usable = np.isfinite(concept_cr)
        all_concepts_rho = RankedColumns(features[data['concepts'] - 1], usable).spearman_with(concept_cr[usable])
        all_concepts_sum = float(np.sum(all_concepts_rho ** 2))
        difference = all_concepts_sum - T19_FULL_SAMPLE_SUM[set_name]
        report('self-check, %s set, all %d concepts: summed rho^2 on the full sample %.10f (T19: %.6f); %s' % (
            set_name, int(np.sum(usable)), all_concepts_sum, T19_FULL_SAMPLE_SUM[set_name],
            'equal' if abs(difference) < 1e-9 else 'DIFFERENT'))
        if abs(difference) >= 1e-9:
            raise RuntimeError('the full-sample summed rho^2 of the %s set differs from T19' % set_name)

    # ---------------- the shared concepts: a CR in both sets ----------------
    concepts_with_cr = {}
    for set_name in SET_NAMES:
        data = set_data[set_name]
        concepts_with_cr[set_name] = data['concepts'][np.isfinite(data['full_sample_cr'])]
    shared_concepts = np.intersect1d(concepts_with_cr['maintained'], concepts_with_cr['reduced'])
    number_shared = len(shared_concepts)
    shared_features = features[shared_concepts - 1]
    for set_name in SET_NAMES:
        data = set_data[set_name]
        columns = np.searchsorted(data['concepts'], shared_concepts)          # the shared concepts' columns
        if not np.array_equal(data['concepts'][columns], shared_concepts):
            raise RuntimeError('shared concepts not found in the %s set' % set_name)
        data['shared_hit_rates'] = data['hit_rates'][:, columns]
        data['shared_false_alarm_rates'] = data['false_alarm_rates'][:, columns]
        data['shared_full_sample_cr'] = data['full_sample_cr'][columns]
    report('shared concepts (a CR in both sets): %d; only in the maintained set %d, only in the reduced set %d' % (
        number_shared, len(concepts_with_cr['maintained']) - number_shared,
        len(concepts_with_cr['reduced']) - number_shared))

    # ---------------- full sample (no resampling), shared concepts ----------------
    ranked_shared_features = RankedColumns(shared_features, np.ones(number_shared, dtype=bool))
    full_sample_sum = {}
    for set_name in SET_NAMES:
        correlations = ranked_shared_features.spearman_with(set_data[set_name]['shared_full_sample_cr'])
        full_sample_sum[set_name] = float(np.sum(correlations ** 2))
    full_sample_drop = full_sample_sum['maintained'] - full_sample_sum['reduced']
    full_sample_percent = 100.0 * full_sample_drop / full_sample_sum['maintained']
    report('full sample, %d shared concepts: summed rho^2 maintained %.6f, reduced %.6f; drop %.6f; percent '
           'reduction %.2f' % (number_shared, full_sample_sum['maintained'], full_sample_sum['reduced'],
                               full_sample_drop, full_sample_percent))

    # ---------------- participant bootstrap (concepts fixed) ----------------
    # First every draw, in the order of the seed note above (resample by resample: maintained, then reduced)
    generator = np.random.default_rng(SEED)
    weights_of_resample = {}
    for set_name in SET_NAMES:
        weights_of_resample[set_name] = np.zeros((number_of_resamples, len(set_data[set_name]['participants'])))
    for resample in range(number_of_resamples):
        for set_name in SET_NAMES:
            weights_of_resample[set_name][resample] = draw_participant_weights(
                generator, len(set_data[set_name]['participants']))
    # Then each resample's summed rho^2 in each set
    resampled_sum = {}
    for set_name in SET_NAMES:
        data = set_data[set_name]
        resampled_sum[set_name] = np.zeros(number_of_resamples)
        resamples_with_fewer_concepts = 0
        for resample in range(number_of_resamples):
            concept_cr = averaged_participant_cr(weights_of_resample[set_name][resample], data['shared_hit_rates'],
                                                 data['shared_false_alarm_rates'])
            usable = np.isfinite(concept_cr)
            if usable.all():
                ranked = ranked_shared_features
            else:
                # a concept lost all its old (or new) trials in this resample: rank on the remaining concepts
                resamples_with_fewer_concepts += 1
                ranked = RankedColumns(shared_features, usable)
            correlations = ranked.spearman_with(concept_cr[usable])
            resampled_sum[set_name][resample] = np.sum(correlations ** 2)
        report('bootstrap, %s set: %d of %d resamples had fewer usable shared concepts than the full sample' % (
            set_name, resamples_with_fewer_concepts, number_of_resamples))

    resampled_drop = resampled_sum['maintained'] - resampled_sum['reduced']
    resampled_percent = 100.0 * resampled_drop / resampled_sum['maintained']
    percent_mean = float(np.mean(resampled_percent))
    percent_low, percent_high = [float(value) for value in bootstrap.percentile_interval(resampled_percent)]
    drop_mean = float(np.mean(resampled_drop))
    share_at_or_above_zero = float(np.mean(resampled_drop >= 0))
    share_at_or_below_zero = float(np.mean(resampled_drop <= 0))
    p_two_sided = min(2 * min(share_at_or_above_zero, share_at_or_below_zero), 1.0)
    report('percent reduction: mean over resamples %.2f, 95%% percentile interval [%.2f, %.2f]; full sample %.2f' % (
        percent_mean, percent_low, percent_high, full_sample_percent))
    report('two-sided bootstrap p %.4f (twice the smaller tail)' % p_two_sided)
    report('rounded as in the paper: %d concepts; dropped by %.0f%% across %s participant resamples (95%% CI [%.0f%%, '
           '%.0f%%], two-sided bootstrap p = %.3f)' % (number_shared, percent_mean, format(number_of_resamples, ','),
                                                      percent_low, percent_high, p_two_sided))

    # ---------------- shared_concepts_numbers.csv: printed numbers and full-sample values ----------------
    number_rows = [
        ('concepts in both image sets (shared concepts)', number_shared),
        ('participants kept, maintained set', len(set_data['maintained']['participants'])),
        ('participants kept, reduced set', len(set_data['reduced']['participants'])),
        ('features', NUMBER_OF_FEATURES),
        ('percent reduction (mean over resamples)', percent_mean),
        ('percent reduction, 2.5th percentile (95% CI low)', percent_low),
        ('percent reduction, 97.5th percentile (95% CI high)', percent_high),
        ('two-sided bootstrap p (twice the smaller tail)', p_two_sided),
        ('summed rho^2, maintained set, shared concepts, full sample', full_sample_sum['maintained']),
        ('summed rho^2, reduced set, shared concepts, full sample', full_sample_sum['reduced']),
        ('drop in summed rho^2 (maintained minus reduced), full sample', full_sample_drop),
        ('percent reduction, full sample', full_sample_percent),
        ('bootstrap resamples', number_of_resamples),
        ('seed (numpy default_rng)', SEED),
    ]
    numbers_file = os.path.join(PATHS['outputs'], 'shared_concepts_numbers.csv')
    with open(numbers_file, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['quantity', 'value'])
        for quantity, value in number_rows:
            writer.writerow([quantity, repr(value) if isinstance(value, float) else value])

    # ---------------- numbers_shared_concepts.csv: the ledger row, judged ----------------
    numbers = numbers_table.NumbersTable(script=SCRIPT_NAME)
    if LEDGER_ID not in numbers.ledger:
        raise RuntimeError('ledger row %s is not in shared/ledger/paper_numbers.csv' % LEDGER_ID)
    # the printed parts, in order: shared concepts; percent reduction; resamples; 95% CI low; 95% CI high;
    # two-sided p
    printed_computed = [number_shared, percent_mean, number_of_resamples, percent_low, percent_high, p_two_sided]
    match = numbers.add(LEDGER_ID, computed_value=printed_computed, mode='rerun',
                        note=('concepts with a CR in both the maintained and the reduced set; percent reduction of the '
                              'summed squared feature-memorability correlation over these concepts (mean over the '
                              'resamples, as in Fig. 7d); participant resamples; 2.5 and 97.5 percentiles of the '
                              'percent reduction; two-sided bootstrap p; full-sample percent reduction %.2f; test T19b'
                              % full_sample_percent))
    numbers.write(PATHS['outputs'], 'numbers_shared_concepts.csv')
    report('paper values (%s): %s; computed %s; match %s' % (
        LEDGER_ID, numbers.ledger[LEDGER_ID]['paper_value'], numbers_table.format_computed(printed_computed), match))

    # ---------------- tests_shared_concepts.csv: the T19b row ----------------
    statistic_text = ('the summed squared feature-memorability correlation (sum over 49 features of Spearman rho^2 '
                      'with concept CR, the Fig. 7d estimator) over the %d concepts in both image sets' % number_shared)
    common = dict(test_id='T19b', ledger_ids=LEDGER_ID,
                  location='Reducing Variability In Distinctiveness Reduces Variability in Memorability',
                  statistic_name='percent reduction', statistic=percent_mean, n='150; 150',
                  unit='participants per set (150 maintained; 150 reduced), 49 features, %d concepts in both sets'
                       % number_shared,
                  effect_size_name='drop in summed rho^2', effect_size=drop_mean, ci_low=percent_low,
                  ci_high=percent_high, ci_method='percentile bootstrap',
                  resampling_unit='participants, within each set (concepts fixed)',
                  n_resamples=number_of_resamples, seed=SEED)
    tests = tests_table.TestsTable(script=SCRIPT_NAME)
    tests.add(test=('Percent reduction of %s, maintained to reduced set; mean over resamples (full sample %.4g '
                    'percent); participants resampled within each set; two-sided p (twice the smaller tail)'
                    % (statistic_text, full_sample_percent)),
              p=p_two_sided, p_type='bootstrap two-sided', **common)
    tests.write(PATHS['outputs'], 'tests_shared_concepts.csv')

    report('wrote outputs/tests_shared_concepts.csv, outputs/shared_concepts_numbers.csv, '
           'outputs/numbers_shared_concepts.csv, outputs/logs/shared_concepts_test_log.txt')
    with open(os.path.join(PATHS['logs'], 'shared_concepts_test_log.txt'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(LOG_LINES) + '\n')


if __name__ == '__main__':
    main()
