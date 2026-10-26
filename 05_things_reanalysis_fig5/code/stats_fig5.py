"""
stats_fig5.py: the 95% intervals and p values of the two Fig. 5 correlations (tests T10 and T11)
and the variance kept by the 49 monkey IT components (T53). Writes outputs/tests.csv and the rows
of these results in outputs/numbers.csv.

WHAT IS TESTED
  Figs. 5c and 5d correlate two lists of 49 numbers, one number per feature:
    x = the feature's Spearman correlation with the concept's summed similarity (all 1,854 concepts)
    y = the feature's Spearman correlation with concept memorability (the 1,853 concepts with a score)
  The statistic is the Pearson r between x and y: r = -0.612 for the 49 human SPoSE features
  (Fig. 5c, T10) and r = -0.296 for the 49 monkey IT components (Fig. 5d, T11). Its interval and its
  p come from a bootstrap over CONCEPTS (the 49 features are never resampled): 10,000 resamples,
  each a draw of 1,854 concepts with replacement; inside each resample all 49 x values and all 49 y
  values are recomputed on the drawn concepts, and then r. Each concept keeps its stored full-THINGS
  summed similarity (the measure is not rebuilt inside a resample). The same draws serve both
  spaces. The 95% interval runs from the 2.5th to the 97.5th percentile of the resampled r, and the
  two-sided p is twice the share of resamples on the far side of zero (r >= 0 for a negative r), at
  most 1.
  T53: the percent of the variance of the concept-averaged IT responses that the first 49 principal
  components keep (computed from the eigenvalues of that PCA).
  No correction for multiple comparisons (statistics/README.md).

INPUTS (all in the repository)
  shared/things/concept_table.csv          SPoSE features, concept memorability and the human and
                                            monkey summed similarity of each concept, as stored
  shared/things/monkey_features.csv        the 49 monkey IT component scores of each concept, as stored
  shared/things/monkey_pca_explained.csv   the 576 eigenvalues of the monkey PCA
  shared/ledger/paper_numbers.csv          the values printed in the paper (the numbers.csv rows are
                                            judged against them)
  05_things_reanalysis_fig5/data/paper_run/paperStats_kramer.mat   r of the saved run the paper reports
                                            (the script checks its own r against it first)
  05_things_reanalysis_fig5/outputs/monkey_variance_explained.csv  (if present: written by the MATLAB
                                            part, run_module.m; checked against the value here)
  05_things_reanalysis_fig5/outputs/numbers.csv  (if present: the rows of the MATLAB part, kept as they are)
OUTPUTS
  05_things_reanalysis_fig5/outputs/tests.csv            one row per statistic (columns: statistics/README.md)
  05_things_reanalysis_fig5/outputs/numbers.csv          the rows L3-009, L3-010, L3-012, L3-013, L3-101 and
                                                          L4-004, added to the rows of the MATLAB part
  05_things_reanalysis_fig5/outputs/logs/stats_fig5.log  everything a full run prints

SEEDS
  20261013: the concept bootstrap (numpy.random.default_rng).

HOW TO RUN (from the repository root; before or after the MATLAB part, since each part keeps the rows
of outputs/numbers.csv that the other wrote)
  python 05_things_reanalysis_fig5/code/stats_fig5.py
  Runtime: about 6 minutes on an Apple M2 Pro laptop (one process).
  Quick check of the code: python 05_things_reanalysis_fig5/code/stats_fig5.py --resamples 100
  (every step in seconds; a run with numbers other than the defaults prints its results but writes no file)
"""
import argparse
import contextlib
import csv
import io
import os
import sys
import time

sys.dont_write_bytecode = True           # no __pycache__ folders inside the repository
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths                     # noqa: E402  (first: it keeps the shared folder free of __pycache__)
import bootstrap                         # noqa: E402
import concept_tables                    # noqa: E402
import numbers_table                     # noqa: E402
import tests_table                       # noqa: E402

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import scipy.io                          # noqa: E402
from scipy.stats import pearsonr, rankdata   # noqa: E402

SEED = bootstrap.RELEASE_SEED            # 20261013
NUMBER_OF_RESAMPLES = 10000              # concept bootstrap resamples
COMPONENTS_KEPT = 49                     # monkey IT components that form the monkey space
SCRIPT = 'code/stats_fig5.py'


class PrintToScreenAndLog:
    """Everything the script prints goes to the screen and to the log file."""

    def __init__(self, log_path):
        self.screen = sys.stdout
        self.log = open(log_path, 'w', encoding='utf-8')

    def write(self, text):
        self.screen.write(text)
        self.log.write(text)

    def flush(self):
        self.screen.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ----------------------------------------------------------------------------------------------
# Correlation helpers (Spearman rho = Pearson correlation of the average ranks, which is what
# MATLAB's corr(..., 'Type', 'Spearman') computes; tied values share their average rank)
# ----------------------------------------------------------------------------------------------
def pearson_correlation(values_a, values_b):
    """Pearson correlation of two equally long lists of numbers."""
    return float(np.corrcoef(values_a, values_b)[0, 1])


def rank_each_feature(feature_matrix):
    """Average ranks of every feature column (concepts in rows, features in columns)."""
    feature_ranks = np.empty(feature_matrix.shape)
    for feature_index in range(feature_matrix.shape[1]):
        feature_ranks[:, feature_index] = rankdata(feature_matrix[:, feature_index])
    return feature_ranks


def correlations_with_each_feature(values, feature_ranks):
    """The Spearman correlation of one concept-level measure with each feature: the Pearson
    correlation of the measure's ranks with each column of feature ranks (same concepts, same order)."""
    value_ranks = rankdata(values)
    correlations = np.empty(feature_ranks.shape[1])
    for feature_index in range(feature_ranks.shape[1]):
        correlations[feature_index] = pearson_correlation(value_ranks, feature_ranks[:, feature_index])
    return correlations


def feature_level_correlation(feature_matrix, memorability, summed_similarity):
    """The Fig. 5c/5d statistic on a set of concepts (rows):
    y = each feature's Spearman correlation with memorability (concepts with a score only),
    x = each feature's Spearman correlation with summed similarity (all concepts),
    returns the Pearson r between x and y across the features, and x and y."""
    has_score = np.isfinite(memorability)
    memorability_correlations = correlations_with_each_feature(
        memorability[has_score], rank_each_feature(feature_matrix[has_score]))
    similarity_correlations = correlations_with_each_feature(
        summed_similarity, rank_each_feature(feature_matrix))
    r = pearson_correlation(similarity_correlations, memorability_correlations)
    return r, similarity_correlations, memorability_correlations


# ----------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------
def load_spaces():
    """The two representational spaces, with the concepts in THINGS concept order."""
    concept_table = concept_tables.load_concept_table()
    memorability = concept_table['kramer_concept_cr'].to_numpy(dtype=float)
    spaces = {
        'human': dict(features=concept_tables.spose_features(concept_table),
                      summed_similarity=concept_table['weighted_similarity_full_things_human'].to_numpy(dtype=float),
                      figure='Fig. 5c', test_id='T10'),
        'monkey': dict(features=concept_tables.load_monkey_features(),
                       summed_similarity=concept_table['weighted_similarity_full_things_monkey'].to_numpy(dtype=float),
                       figure='Fig. 5d', test_id='T11'),
    }
    return memorability, spaces


# ----------------------------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------------------------
def correlation_text(r, interval, p):
    """A correlation with its interval and p as text: 'r = -0.612, 95% CI [-0.677, -0.505], p < 0.001'."""
    return 'r = %s, 95%% CI [%s, %s], %s' % (tests_table.format_value(r), tests_table.format_value(interval[0]),
                                            tests_table.format_value(interval[1]), tests_table.format_p(p))


def add_rows_to_numbers_csv(outputs_folder, rows):
    """Writes the rows of this script into outputs/numbers.csv, which also holds the rows of the MATLAB
    part (run_module.m). Those rows are kept as they are, the rows of this script replace any rows with
    the same ledger ids, and all rows are sorted by ledger id. Returns the number of rows kept."""
    numbers_file = os.path.join(outputs_folder, 'numbers.csv')
    ids_of_this_script = set(row['ledger_id'] for row in rows)
    rows_of_matlab_part = []
    if os.path.exists(numbers_file):
        with open(numbers_file, newline='', encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle):
                if row['ledger_id'] not in ids_of_this_script:
                    rows_of_matlab_part.append(row)
    all_rows = sorted(rows_of_matlab_part + list(rows), key=lambda row: row['ledger_id'])
    with open(numbers_file, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=numbers_table.COLUMNS, lineterminator='\n')   # as run_module.m
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    return len(rows_of_matlab_part)


def main():
    parser = argparse.ArgumentParser(description='Statistics T10, T11 and T53 of module 05 (Fig. 5)')
    parser.add_argument('--resamples', type=int, default=NUMBER_OF_RESAMPLES,
                        help='concept-bootstrap resamples (default 10000)')
    arguments = parser.parse_args()
    number_of_resamples = arguments.resamples
    full_run = number_of_resamples == NUMBER_OF_RESAMPLES

    start_time = time.time()
    paths = release_paths.module_paths(__file__)
    log_writer = None
    if full_run:
        log_writer = PrintToScreenAndLog(os.path.join(paths['logs'], 'stats_fig5.log'))
        sys.stdout = log_writer
    else:
        print('QUICK CHECK: %d resamples (full run: %d); the results are printed only, no file is written'
              % (number_of_resamples, NUMBER_OF_RESAMPLES))
    print('stats_fig5.py: seed %d (bootstrap)' % SEED)

    memorability, spaces = load_spaces()
    number_of_concepts = len(memorability)
    has_score = np.isfinite(memorability)
    number_with_score = int(np.sum(has_score))
    number_of_features = spaces['human']['features'].shape[1]
    print('%d concepts (%d with a memorability score), %d features per space'
          % (number_of_concepts, number_with_score, number_of_features))

    # ------------------------------------------------------------------------------------------
    # 1. The two correlations of Figs. 5c and 5d, checked against the saved run in data/paper_run/
    # ------------------------------------------------------------------------------------------
    paper_run = scipy.io.loadmat(os.path.join(paths['paper_run'], 'paperStats_kramer.mat'),
                                 squeeze_me=True, struct_as_record=False)['paperStats']
    paper_r_values = {'human': float(paper_run.r_human), 'monkey': float(paper_run.r_monkey)}
    observed = {}
    for space_name, space in spaces.items():
        r, similarity_correlations, memorability_correlations = feature_level_correlation(
            space['features'], memorability, space['summed_similarity'])
        r_check = float(pearsonr(similarity_correlations, memorability_correlations)[0])
        paper_r = paper_r_values[space_name]
        print('%s space (%s): r = %.10f (saved run: r = %.10f)' % (space_name, space['figure'], r, paper_r))
        if abs(r - paper_r) > 1e-12 or abs(r_check - r) > 1e-12:
            raise SystemExit('The %s correlation is not equal to the saved run in data/paper_run/.' % space_name)
        observed[space_name] = dict(r=r, similarity_correlations=similarity_correlations,
                                    memorability_correlations=memorability_correlations)
    print('  both equal to the saved run (data/paper_run/paperStats_kramer.mat)')

    # ------------------------------------------------------------------------------------------
    # 2. Concept bootstrap (the same concept draws for both spaces)
    # ------------------------------------------------------------------------------------------
    def both_spaces_on_drawn_concepts(drawn_concepts):
        """r of both spaces recomputed on one bootstrap draw of concepts (rows may repeat)."""
        results = []
        for space_name in ('human', 'monkey'):
            space = spaces[space_name]
            r, _, _ = feature_level_correlation(space['features'][drawn_concepts],
                                                memorability[drawn_concepts],
                                                space['summed_similarity'][drawn_concepts])
            results.append(r)
        return results

    print('\nConcept bootstrap: %d resamples of %d concepts, seed %d ...' % (number_of_resamples, number_of_concepts, SEED))
    bootstrap_start = time.time()
    resampled = bootstrap.percentile_bootstrap(both_spaces_on_drawn_concepts, number_of_concepts,
                                               n_resamples=number_of_resamples, seed=SEED)
    interval_low, interval_high = bootstrap.percentile_interval(resampled, level=0.95)
    bootstrap_interval = {}
    resamples_with_value = {}
    far_side_count = {}
    bootstrap_p = {}
    for column, space_name in enumerate(('human', 'monkey')):
        values = resampled[:, column]
        finite_values = values[np.isfinite(values)]
        # The far side of zero: r >= 0 when the observed r is negative, r <= 0 when it is positive
        if observed[space_name]['r'] < 0:
            on_far_side = finite_values >= 0
        else:
            on_far_side = finite_values <= 0
        resamples_with_value[space_name] = len(finite_values)
        far_side_count[space_name] = int(np.sum(on_far_side))
        bootstrap_p[space_name] = min(1.0, 2.0 * far_side_count[space_name] / resamples_with_value[space_name])
        bootstrap_interval[space_name] = (float(interval_low[column]), float(interval_high[column]))
        print('  %s: 95%% interval [%.4f, %.4f]; mean of the resamples %.4f, SD %.4f; resamples on the far side '
              'of zero: %d of %d, bootstrap p = %.4g; resamples without a value: %d'
              % (space_name, interval_low[column], interval_high[column], np.nanmean(values),
                 np.nanstd(values, ddof=1), far_side_count[space_name], resamples_with_value[space_name],
                 bootstrap_p[space_name], len(values) - resamples_with_value[space_name]))
    print('  (%.0f s)' % (time.time() - bootstrap_start))

    # ------------------------------------------------------------------------------------------
    # 3. T53: variance kept by the first 49 monkey IT components
    # ------------------------------------------------------------------------------------------
    pca_table = pd.read_csv(concept_tables.things_file('monkey_pca_explained.csv'), float_precision='round_trip')
    eigenvalues = pca_table['eigenvalue'].to_numpy(dtype=float)
    total_variance = 0.0
    for eigenvalue in eigenvalues:
        total_variance += eigenvalue
    kept_variance = 0.0
    for eigenvalue in eigenvalues[:COMPONENTS_KEPT]:
        kept_variance += eigenvalue
    percent_kept = 100.0 * kept_variance / total_variance
    print('\nMonkey IT PCA: the first %d of %d components keep %.2f percent of the variance'
          % (COMPONENTS_KEPT, len(eigenvalues), percent_kept))
    matlab_file = os.path.join(paths['outputs'], 'monkey_variance_explained.csv')
    if os.path.exists(matlab_file):
        matlab_percent = float(pd.read_csv(matlab_file, float_precision='round_trip')['percent_of_variance_kept'].iloc[0])
        if abs(matlab_percent - percent_kept) > 1e-9:
            raise SystemExit('The percent of variance differs from the MATLAB output (outputs/monkey_variance_explained.csv).')
        print('  equal to outputs/monkey_variance_explained.csv (MATLAB part)')

    # ------------------------------------------------------------------------------------------
    # 4. outputs/tests.csv
    # ------------------------------------------------------------------------------------------
    tests = tests_table.TestsTable(script=SCRIPT)
    unit = 'features (%s concepts; %s with a memorability score)' % (
        '{:,}'.format(number_of_concepts), '{:,}'.format(number_with_score))
    descriptions = {
        'human': dict(ledger_ids='L3-009;L3-012;L3-006;L3-105;L4-004;L3-101', location='Fig. 5 caption (Fig. 5c)',
                      features='the 49 SPoSE features (human similarity space)'),
        'monkey': dict(ledger_ids='L3-010;L3-013;L3-007;L3-106;L4-004;L3-101', location='Fig. 5 caption (Fig. 5d)',
                       features='the 49 monkey IT components (monkey space)'),
    }
    for space_name in ('human', 'monkey'):
        space = spaces[space_name]
        about = descriptions[space_name]
        r = observed[space_name]['r']
        test_text = ("Pearson r across %s between each feature's Spearman correlation with full-THINGS summed "
                     "similarity and its Spearman correlation with concept memorability; concept-bootstrap "
                     "interval and p (twice the share of resamples on the far side of zero: %d of %s)"
                     % (about['features'], far_side_count[space_name],
                        '{:,}'.format(resamples_with_value[space_name])))
        tests.add(test_id=space['test_id'], ledger_ids=about['ledger_ids'], location=about['location'],
                  test=test_text, statistic_name='Pearson r', statistic=r, n=number_of_features, unit=unit,
                  p=bootstrap_p[space_name], p_type='bootstrap two-sided', effect_size_name='r', effect_size=r,
                  ci_low=bootstrap_interval[space_name][0], ci_high=bootstrap_interval[space_name][1],
                  ci_method='percentile bootstrap (the 49 correlations with memorability and the 49 with summed '
                            'similarity recomputed in each resample; each concept keeps its stored summed similarity)',
                  resampling_unit='concepts', n_resamples=number_of_resamples, seed=SEED)

    tests.add(test_id='T53', ledger_ids='L5-048', location='Methods, Representational Similarity Analysis',
              test='Percent of the variance of the concept-averaged multi-unit responses of the 576 IT channels '
                   'kept by the first 49 principal components (the monkey space of Fig. 5d)',
              statistic_name='percent of variance', statistic=percent_kept, n=number_of_concepts,
              unit='concepts (576 IT channels)', p_type='none', effect_size_name='percent of variance',
              effect_size=percent_kept, ci_method='none', resampling_unit='')

    # ------------------------------------------------------------------------------------------
    # 5. The rows of outputs/numbers.csv computed here (judged against shared/ledger/paper_numbers.csv)
    # ------------------------------------------------------------------------------------------
    numbers = numbers_table.NumbersTable(script=SCRIPT)
    printed_parts = {}
    for space_name in ('human', 'monkey'):
        printed_parts[space_name] = [observed[space_name]['r'], bootstrap_interval[space_name][0],
                                     bootstrap_interval[space_name][1], bootstrap_p[space_name]]
    numbers.add('L3-009', printed_parts['human'], 'rerun',
                "Pearson r across the 49 SPoSE features between each feature's Spearman correlation with summed "
                "similarity and its Spearman correlation with memorability (human space); 95% concept-bootstrap "
                "interval and bootstrap p (T10).")
    numbers.add('L3-010', printed_parts['monkey'], 'rerun',
                "Pearson r across the 49 monkey IT components between each component's Spearman correlation with "
                "summed similarity and its Spearman correlation with memorability (monkey space); 95% "
                "concept-bootstrap interval and bootstrap p (T11).")
    numbers.add('L3-012', printed_parts['human'], 'rerun', 'The Fig. 5c correlation (same values as L3-009).')
    numbers.add('L3-013', printed_parts['monkey'], 'rerun', 'The Fig. 5d correlation (same values as L3-010).')
    resamples_on_far_side = far_side_count['human'] + far_side_count['monkey']
    numbers.add('L3-101', ['none' if resamples_on_far_side == 0 else resamples_on_far_side, number_of_resamples],
                'rerun',
                "Concept resamples on the far side of zero: %d of %d for Fig. 5c and %d of %d for Fig. 5d ('none' "
                "when both counts are 0); the bootstrap p is twice that share, %.4g and %.4g."
                % (far_side_count['human'], resamples_with_value['human'], far_side_count['monkey'],
                   resamples_with_value['monkey'], bootstrap_p['human'], bootstrap_p['monkey']))
    numbers.add('L4-004', 'restates Fig. 5c and 5d: %s (human); %s (monkey)'
                % (correlation_text(observed['human']['r'], bootstrap_interval['human'], bootstrap_p['human']),
                   correlation_text(observed['monkey']['r'], bootstrap_interval['monkey'], bootstrap_p['monkey'])),
                'rerun', 'Restates the Fig. 5c and 5d correlations (L3-009 and L3-010).')
    print('\nRows of outputs/numbers.csv computed here:')
    for row in numbers.rows:
        print('  %s  %-10s printed %-34s computed %s' % (row['ledger_id'], row['match'], row['paper_value'],
                                                         row['computed_value']))
    not_matching = [row['ledger_id'] for row in numbers.rows if row['match'] == 'no']

    print('')
    if full_run:
        # tests_table.write() prints the full path of the file; print the path inside the module instead
        with contextlib.redirect_stdout(io.StringIO()):
            tests.write(paths['outputs'])
        print('wrote outputs/tests.csv: %d rows' % len(tests.rows))
        rows_of_matlab_part = add_rows_to_numbers_csv(paths['outputs'], numbers.rows)
        print('wrote outputs/numbers.csv: %d rows of this script and %d rows of the MATLAB part'
              % (len(numbers.rows), rows_of_matlab_part))
        if rows_of_matlab_part == 0:
            print('  (no rows of the MATLAB part yet: run_module.m adds them and keeps the rows of this script)')
        if not_matching:
            print('NOT MATCHING the printed values: %s' % ', '.join(not_matching))
    print('\nDone in %.0f s' % (time.time() - start_time))
    if log_writer is not None:
        sys.stdout = log_writer.screen
        log_writer.close()
    if full_run and not_matching:
        raise SystemExit('stats_fig5.py: computed values that do not match the printed values: %s'
                         % ', '.join(not_matching))


if __name__ == '__main__':
    main()
