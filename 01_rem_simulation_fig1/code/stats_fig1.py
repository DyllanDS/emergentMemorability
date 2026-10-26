"""
stats_fig1.py: the uncertainty statistics of Fig. 1, tests T01 and T02 (statistics/README.md lists
every test and the columns of tests.csv), and the numbers.csv rows of the two correlations of Fig. 1.

What it does
  T01  Item memorability against item similarity across the 1,600 simulated items (Fig. 1b).
       The correlation of the Fig. 1 caption is the Pearson r with similarity = 1 / mean Euclidean
       distance to the other items. For it the script gives the value of the saved run, the
       two-sided p of the t test of r (1,598 df) and a 95% percentile bootstrap interval over items
       (10,000 resamples of the 1,600 items). It gives the same bootstrap interval for the negated
       mean distance (Spearman, the rho = -0.55 of Methods, Basic REM Simulation of Memorability,
       and the measure of SI Appendix, Fig. S2).
  T02  The feature-level correlation (Fig. 1d): the Spearman correlation across the 49 features
       between each feature's Spearman correlation with similarity and its Spearman correlation with
       memorability. The script gives the value of the saved run with the p value of MATLAB corr and
       a 95% percentile bootstrap interval over items (all 49 feature correlations recomputed
       inside each of the 10,000 resamples; the features are never resampled).
  numbers.csv  The rows of the two correlations as the paper prints them (r or rho, 95% interval,
       p): L1-019 and L1-022 (Fig. 1b), L1-021 and L1-024 (Fig. 1d). The p of Fig. 1b is the
       two-sided p of the t test of r (incomplete beta function); the p of Fig. 1d is the one
       MATLAB corr stored in the saved run. These rows are merged into the outputs/numbers.csv that
       run_module.m or fig1_numbers.m writes, so this script runs after them.

Input (inside 01_rem_simulation_fig1/)
  data/paper_run/fig1_paper_run.mat
      the saved run: per-item memorability, the 1,600 x 49 item features, the distances

Outputs (inside 01_rem_simulation_fig1/)
  outputs/tests.csv               one row per statistic (T01: 2 rows, T02: 2 rows), columns as in
                                  statistics/README.md
  outputs/numbers.csv             the four rows above, merged into the file run_module.m writes
  outputs/logs/stats_fig1.log     the messages of the run

Seed: 20261013 (numpy.random.default_rng) for the bootstrap.
Run from the repository root:  python 01_rem_simulation_fig1/code/stats_fig1.py   (about 1 minute)
"""
import os
import sys
import time

import csv

import numpy as np
import scipy.io
from scipy import special, stats

sys.dont_write_bytecode = True      # no __pycache__ folders inside the release
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths      # noqa: E402  (the release's folders; imported after the path line above)
import bootstrap        # noqa: E402
import numbers_table    # noqa: E402
import tests_table      # noqa: E402

RELEASE_SEED = 20261013
NUMBER_OF_RESAMPLES = 10000
PAPER_RUN_FILE = 'fig1_paper_run.mat'
SCRIPT = 'code/stats_fig1.py'

log_lines = []


def report(line=''):
    """Print a line and keep it for the log file."""
    print(line)
    log_lines.append(line)


# ---------------------------------------------------------------------------------------------
# Correlations, written out so the computation is visible
# ---------------------------------------------------------------------------------------------

def pearson_correlation(x_values, y_values):
    """Pearson r of two equally long vectors."""
    x_centered = x_values - np.mean(x_values)
    y_centered = y_values - np.mean(y_values)
    return np.sum(x_centered * y_centered) / np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))


def spearman_correlation(x_values, y_values):
    """Spearman rho = the Pearson r of the ranks (tied values get their average rank), which is how
    MATLAB's corr(..., 'Type', 'Spearman') and scipy's spearmanr compute it."""
    return pearson_correlation(stats.rankdata(x_values), stats.rankdata(y_values))


def correlation_of_each_feature(feature_ranks, item_value_ranks):
    """Spearman rho of every feature with one item variable (memorability or similarity), from
    ranks computed beforehand: feature_ranks has one column of ranks per feature."""
    number_of_features = feature_ranks.shape[1]
    correlations = np.empty(number_of_features)
    for feature in range(number_of_features):
        correlations[feature] = pearson_correlation(feature_ranks[:, feature], item_value_ranks)
    return correlations


def t_test_p_value(r, number_of_items):
    """Two-sided p of the t test of a Pearson r, p = I(df / (df + t^2); df / 2, 1 / 2) with
    t = r sqrt(df / (1 - r^2)) and df = n - 2."""
    degrees_of_freedom = number_of_items - 2
    t_statistic = r * np.sqrt(degrees_of_freedom / (1.0 - r ** 2))
    return float(special.betainc(degrees_of_freedom / 2.0, 0.5,
                                 degrees_of_freedom / (degrees_of_freedom + t_statistic ** 2)))


def merge_into_numbers_csv(numbers, outputs_folder):
    """Write this script's numbers.csv rows into outputs/numbers.csv: the rows run_module.m wrote
    are kept, and any earlier copy of this script's rows is replaced."""
    path = os.path.join(outputs_folder, 'numbers.csv')
    own_ids = set(row['ledger_id'] for row in numbers.rows)
    kept_rows = []
    if os.path.exists(path):
        with open(path, newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                if row['ledger_id'] not in own_ids:
                    kept_rows.append({column: row.get(column, '') for column in numbers_table.COLUMNS})
    else:
        report('outputs/numbers.csv not found: run run_module.m first for the other rows of this module')
    numbers.rows = kept_rows + numbers.rows
    numbers.write(outputs_folder)


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------

def load_paper_run(paths):
    """The per-item results of the saved run in data/paper_run/ (and the values the simulation stored)."""
    contents = scipy.io.loadmat(os.path.join(paths['paper_run'], PAPER_RUN_FILE),
                                squeeze_me=True, struct_as_record=False)
    saved = contents['figureData']
    paper_run = {
        'seed': int(saved.seedValue),
        'memorability': np.asarray(saved.hitFARate, dtype=float),
        'item_features': np.asarray(saved.imagesOfInterest, dtype=float),
        'mean_distance': np.asarray(saved.avgDistances, dtype=float),
        'similarity': np.asarray(saved.similarity, dtype=float),
        'stored_pearson_inverse_mean_distance': float(saved.r_sim),
        'stored_spearman_across_features': float(saved.r_meta),
        'stored_p_spearman_across_features': float(saved.p_meta),
        'stored_feature_correlations_with_memorability': np.asarray(saved.featureCorrsMemory, dtype=float),
        'stored_feature_correlations_with_similarity': np.asarray(saved.featureCorrsSimil, dtype=float),
    }
    return paper_run


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

def main():
    start = time.time()
    paths = release_paths.module_paths(__file__)
    report('stats_fig1.py: uncertainty statistics T01 and T02 of Fig. 1 (module 01)')
    report('seed %d; %d bootstrap resamples of the items' % (RELEASE_SEED, NUMBER_OF_RESAMPLES))

    paper_run = load_paper_run(paths)
    memorability = paper_run['memorability']
    item_features = paper_run['item_features']
    similarity = paper_run['similarity']
    mean_distance = paper_run['mean_distance']
    number_of_items, number_of_features = item_features.shape
    report('saved run (data/paper_run/): seed %d, %d items, %d features' % (paper_run['seed'], number_of_items, number_of_features))

    # -----------------------------------------------------------------------------------------
    # Step 1. The two correlations of Fig. 1, recomputed from the per-item results of the saved run
    # -----------------------------------------------------------------------------------------
    pearson_inverse_mean_distance = pearson_correlation(similarity, memorability)
    spearman_negated_mean_distance = spearman_correlation(-mean_distance, memorability)

    all_feature_ranks = stats.rankdata(item_features, axis=0)       # ranks of each feature column
    feature_with_memorability = correlation_of_each_feature(all_feature_ranks, stats.rankdata(memorability))
    feature_with_similarity = correlation_of_each_feature(all_feature_ranks, stats.rankdata(similarity))
    spearman_across_features = spearman_correlation(feature_with_similarity, feature_with_memorability)

    differences = [
        abs(pearson_inverse_mean_distance - paper_run['stored_pearson_inverse_mean_distance']),
        abs(spearman_across_features - paper_run['stored_spearman_across_features']),
        np.max(np.abs(feature_with_memorability - paper_run['stored_feature_correlations_with_memorability'])),
        np.max(np.abs(feature_with_similarity - paper_run['stored_feature_correlations_with_similarity'])),
    ]
    report('')
    report('Step 1. The two correlations of Fig. 1 recomputed from the per-item results of the saved run')
    report('  Pearson r, memorability vs 1 / mean distance (Fig. 1b, reported)  %.6f  (MATLAB stored %.6f)'
           % (pearson_inverse_mean_distance, paper_run['stored_pearson_inverse_mean_distance']))
    report('  Spearman rho across the 49 features (Fig. 1d, reported)           %.6f  (MATLAB stored %.6f, p %.6f)'
           % (spearman_across_features, paper_run['stored_spearman_across_features'],
              paper_run['stored_p_spearman_across_features']))
    report('  largest difference from the MATLAB values: %.2g' % max(differences))
    if max(differences) > 1e-12:
        raise RuntimeError('the Python recomputation does not equal the MATLAB values of the saved run')

    # -----------------------------------------------------------------------------------------
    # Step 2. Bootstrap over items: every statistic recomputed in each resample of the 1,600 items
    # -----------------------------------------------------------------------------------------
    statistic_names = ['pearson_inverse_mean_distance', 'spearman_negated_mean_distance',
                       'spearman_across_features']

    def statistics_of_one_resample(item_indices):
        """All three statistics on one resample of the items (an item drawn twice counts twice).
        Each item keeps its own memorability, similarity and features: the resampling unit is the
        item, and the 49 features are never resampled."""
        resampled_memorability = memorability[item_indices]
        resampled_similarity = similarity[item_indices]
        resampled_mean_distance = mean_distance[item_indices]
        resampled_features = item_features[item_indices, :]

        # T01: item memorability against the two similarity measures
        resample_pearson_inverse = pearson_correlation(resampled_similarity, resampled_memorability)
        resample_spearman_negated = spearman_correlation(-resampled_mean_distance, resampled_memorability)

        # T02: the 49 feature correlations recomputed in this resample. The one vectorized step of
        # each resample: the 49 feature columns are ranked in one call (rankdata with axis=0 ranks
        # each column on its own, ties averaged); the correlations are then computed feature by
        # feature.
        resampled_feature_ranks = stats.rankdata(resampled_features, axis=0)
        resample_with_memorability = correlation_of_each_feature(resampled_feature_ranks,
                                                                 stats.rankdata(resampled_memorability))
        resample_with_similarity = correlation_of_each_feature(resampled_feature_ranks,
                                                               stats.rankdata(resampled_similarity))
        resample_spearman_across = spearman_correlation(resample_with_similarity, resample_with_memorability)

        return [resample_pearson_inverse, resample_spearman_negated, resample_spearman_across]

    bootstrap_start = time.time()
    resampled_statistics = bootstrap.percentile_bootstrap(statistics_of_one_resample, number_of_items,
                                                          n_resamples=NUMBER_OF_RESAMPLES, seed=RELEASE_SEED)
    interval_low, interval_high = bootstrap.percentile_interval(resampled_statistics)
    bootstrap_interval = {}
    for column, name in enumerate(statistic_names):
        bootstrap_interval[name] = (float(interval_low[column]), float(interval_high[column]))
    report('')
    report('Step 2. Bootstrap over the %d items (%d resamples, seed %d): %.0f s'
           % (number_of_items, NUMBER_OF_RESAMPLES, RELEASE_SEED, time.time() - bootstrap_start))

    # -----------------------------------------------------------------------------------------
    # Step 3. tests.csv
    # -----------------------------------------------------------------------------------------
    t_test_p = t_test_p_value(pearson_inverse_mean_distance, number_of_items)
    tests = tests_table.TestsTable(script=SCRIPT)
    unit_items = 'simulated items (each with its hit rate and false alarm rate from the 10,000 simulated experiments)'
    unit_features = 'features (each correlation over the 1,600 simulated items)'
    ledger_t01 = 'L1-019;L1-022;L5-007'
    ledger_t02 = 'L1-021;L1-024'
    location_t01 = ('Fig. 1 caption (panel b); Simulations of Memorability as an Emergent Property; '
                    'Methods, Basic REM Simulation of Memorability')
    location_t02 = 'Fig. 1 caption (panel d); Simulations of Memorability as an Emergent Property'

    # T01, the reported correlation
    tests.add(test_id='T01', ledger_ids=ledger_t01, location=location_t01,
              test='Pearson correlation across the simulated items between item memorability (hit rate minus '
                   'false alarm rate) and item similarity = 1 / mean Euclidean distance to the other items '
                   '(Fig. 1b)',
              statistic_name='Pearson r', statistic=pearson_inverse_mean_distance, n=number_of_items, unit=unit_items,
              df=number_of_items - 2, p=t_test_p, p_type='parametric two-sided', effect_size_name='r squared',
              effect_size=pearson_inverse_mean_distance ** 2,
              ci_low=bootstrap_interval['pearson_inverse_mean_distance'][0],
              ci_high=bootstrap_interval['pearson_inverse_mean_distance'][1],
              ci_method='percentile bootstrap', resampling_unit='items', n_resamples=NUMBER_OF_RESAMPLES,
              seed=RELEASE_SEED)
    tests.add(test_id='T01', ledger_ids=ledger_t01, location=location_t01,
              test='Spearman correlation of item memorability with the negated mean Euclidean distance (the '
                   'rho = -0.55 of Methods, Basic REM Simulation of Memorability; the measure of SI Appendix, Fig. S2)',
              statistic_name='Spearman rho', statistic=spearman_negated_mean_distance, n=number_of_items,
              unit=unit_items, df='', p='', p_type='none', effect_size_name='rho',
              effect_size=spearman_negated_mean_distance,
              ci_low=bootstrap_interval['spearman_negated_mean_distance'][0],
              ci_high=bootstrap_interval['spearman_negated_mean_distance'][1],
              ci_method='percentile bootstrap', resampling_unit='items', n_resamples=NUMBER_OF_RESAMPLES,
              seed=RELEASE_SEED)

    # T02, the reported feature-level correlation
    tests.add(test_id='T02', ledger_ids=ledger_t02, location=location_t02,
              test='Spearman correlation across the 49 features between each feature\'s Spearman correlation with '
                   'item similarity (1 / mean distance) and its Spearman correlation with item memorability '
                   '(Fig. 1d)',
              statistic_name='Spearman rho', statistic=spearman_across_features, n=number_of_features,
              unit=unit_features, df='', p='', p_type='none',
              effect_size_name='rho', effect_size=spearman_across_features,
              ci_low=bootstrap_interval['spearman_across_features'][0],
              ci_high=bootstrap_interval['spearman_across_features'][1],
              ci_method='percentile bootstrap (all 49 feature correlations recomputed in each resample)',
              resampling_unit='items', n_resamples=NUMBER_OF_RESAMPLES, seed=RELEASE_SEED)
    tests.add(test_id='T02', ledger_ids=ledger_t02, location=location_t02,
              test='the same correlation with the p value of MATLAB corr (Spearman, Edgeworth series '
                   'approximation), as in fig1_remSim.m',
              statistic_name='Spearman rho', statistic=spearman_across_features, n=number_of_features,
              unit=unit_features, df='', p=paper_run['stored_p_spearman_across_features'],
              p_type='parametric two-sided', effect_size_name='rho', effect_size=spearman_across_features,
              ci_low='', ci_high='', ci_method='', resampling_unit='', n_resamples='', seed='')
    tests.write(paths['outputs'])

    report('')
    report('outputs/tests.csv: %d rows' % len(tests.rows))

    # -----------------------------------------------------------------------------------------
    # Step 4. numbers.csv: the two correlations of Fig. 1 with their intervals and p values
    # -----------------------------------------------------------------------------------------
    p_across_features = paper_run['stored_p_spearman_across_features']
    interval_b = bootstrap_interval['pearson_inverse_mean_distance']
    interval_d = bootstrap_interval['spearman_across_features']
    values_b = [pearson_inverse_mean_distance, interval_b[0], interval_b[1], t_test_p]
    values_d = [spearman_across_features, interval_d[0], interval_d[1], p_across_features]
    note_b = ('Pearson r across the %d simulated items between memorability (hit rate minus false alarm rate) and '
              'similarity (1 / mean Euclidean distance to the other items); 95%% percentile bootstrap interval over '
              'items (test T01); two-sided p of the t test of r (%d df), computed with the incomplete beta function'
              % (number_of_items, number_of_items - 2))
    note_d = ('Spearman rho across the %d features between each feature\'s Spearman correlation with similarity '
              '(1 / mean Euclidean distance) and its Spearman correlation with memorability, each over the %d '
              'items; 95%% percentile bootstrap interval over items (test T02); two-sided p from MATLAB corr'
              % (number_of_features, number_of_items))
    numbers = numbers_table.NumbersTable(script=SCRIPT)
    numbers.add('L1-019', values_b, 'saved_run', note_b)
    numbers.add('L1-022', values_b, 'saved_run', 'Same value as L1-019: ' + note_b)
    numbers.add('L1-021', values_d, 'saved_run', note_d)
    numbers.add('L1-024', values_d, 'saved_run', 'Same value as L1-021: ' + note_d)
    report('')
    for row in numbers.rows:
        report('numbers.csv %s: printed %s, computed %s, match %s'
               % (row['ledger_id'], row['paper_value'], row['computed_value'], row['match']))
    merge_into_numbers_csv(numbers, paths['outputs'])
    report('')
    report('Finished in %.0f s' % (time.time() - start))

    log_path = os.path.join(paths['logs'], 'stats_fig1.log')
    with open(log_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(log_lines) + '\n')
    print('wrote outputs/logs/stats_fig1.log')


if __name__ == '__main__':
    main()
