"""
stats_figS16.py: step 4 of module A6b. The uncertainty of the Fig. S16 statistic in each refit and
in the published embedding (the uncertainty statistics, test T47 of statistics/README.md).

WHAT IT DOES
For the published 49-feature embedding (the black point of Fig. S16) and each of the 14 refits (the
rows of outputs/refit_sweep.csv), the Fig. 5c statistic r (a Pearson correlation across the k
features of the embedding, between each feature's Spearman correlation with summed similarity and
its Spearman correlation with concept memorability) gets:
  - n = k features (each feature's two correlations are computed over the 1,854 THINGS concepts;
    1,853 have a memorability score).
  - the two-sided parametric p (t test on k - 2 degrees of freedom), the p of SI Appendix, Section 6
    for the refits.
  - a 95% percentile bootstrap interval over concepts: the 1,854 concepts are resampled with
    replacement, both per-feature correlations are recomputed over the resampled concepts (each
    concept keeps the summed similarity it has in the full embedding; summed similarity is not
    rebuilt inside a resample), and r is recomputed. 2,000 resamples for each refit and 10,000 for
    the published embedding (the number module 05 uses for the same statistic, test T10, so that
    both tables give one interval for it).
  - the effect size: r itself.
For the published embedding, tests.csv carries the bootstrap p of Fig. 5c instead of the parametric p:
twice the share of resamples on the far side of zero (r >= 0 for a negative r), as module 05 gives it
for test T10.

INPUTS
  outputs/refit_sweep.csv (step 2), data/refits/spose_*.txt, the THINGS inputs of analysis_core.py
  (shared/things/)

OUTPUTS
  outputs/figS16_statistics.csv  one row per embedding: r, n, parametric p and df, bootstrap
                                 interval, number of resamples, seed
  outputs/tests.csv              the T47 rows of the uncertainty-statistics table: per embedding
                                 one row with the p and the bootstrap interval

SEEDS
  20261013 for the resamples. Every embedding starts a new generator from it, so each row
  reproduces on its own (and the published embedding's row reproduces module 05's T10 row draw for
  draw).

RUN (from the repository root; about 7 minutes on an Apple M2 Pro, one processor core)
  python A6b_embedding_size_figS16/code/stats_figS16.py
  options: --resamples <n> (refits, default 2000), --published-resamples <n> (default 10000);
           smaller numbers only to test the code quickly

MULTIPLE COMPARISONS
  No correction for multiple comparisons is applied (statistics/README.md).
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True     # no __pycache__ folders inside the release
sys.path.insert(0, os.path.join(CODE_FOLDER, '..', '..', 'shared', 'python'))
import release_paths     # noqa: E402  (shared helper: release folders)
import bootstrap       # noqa: E402  (shared helper: seeded bootstrap functions)
import tests_table     # noqa: E402  (shared helper: the tests.csv writer)
import analysis_core   # noqa: E402  (this module: the Fig. 5c analysis)

RELEASE_SEED = bootstrap.RELEASE_SEED                 # 20261013
DEFAULT_REFIT_RESAMPLES = 2000                        # concept bootstrap, each refit
DEFAULT_PUBLISHED_RESAMPLES = 10000                   # concept bootstrap, published embedding (as T10)
PUBLISHED_NAME = 'published 49-d'
SCRIPT = 'code/stats_figS16.py'
LOCATION = 'SI Appendix, Section 6, Embedding Size'
PUBLISHED_ITEM = 'L9-037'                             # ledger id of the published embedding's r


# ------------------------------------------------------------------------------- the two statistics
def make_resample_statistic(embedding, memorability, summed_similarity):
    """statistic(concept indices of one resample) -> the Fig. 5c r over the resampled concepts, each
    keeping its full-embedding summed similarity."""
    def statistic(concept_indices):
        resampled_embedding = embedding[concept_indices]
        resampled_memorability = memorability[concept_indices]
        resampled_summed_similarity = summed_similarity[concept_indices]
        has_score = np.isfinite(resampled_memorability)
        memorability_correlations = analysis_core.spearman_with_each_column(
            resampled_memorability[has_score], resampled_embedding[has_score])
        similarity_correlations = analysis_core.spearman_with_each_column(
            resampled_summed_similarity, resampled_embedding)
        return analysis_core.pearson_r_and_p(similarity_correlations, memorability_correlations)[0]
    return statistic


# ------------------------------------------------------------------------------------------- main
def main():
    paths = release_paths.module_paths(__file__)
    parser = argparse.ArgumentParser(description='T47: concept-bootstrap interval of the Fig. S16 statistic '
                                                 'in every embedding')
    parser.add_argument('--resamples', type=int, default=DEFAULT_REFIT_RESAMPLES,
                        help='bootstrap resamples per refit (default 2000)')
    parser.add_argument('--published-resamples', type=int, default=DEFAULT_PUBLISHED_RESAMPLES,
                        help='bootstrap resamples for the published embedding (default 10000)')
    arguments = parser.parse_args()
    started = time.time()
    print('step 4: T47, concept-bootstrap interval of the Fig. S16 statistic')
    print('  seed %d; %d concept resamples per refit, %d for the published embedding'
          % (RELEASE_SEED, arguments.resamples, arguments.published_resamples))

    sweep_file = os.path.join(paths['outputs'], 'refit_sweep.csv')
    if not os.path.exists(sweep_file):
        raise SystemExit('outputs/refit_sweep.csv is missing: run run_refit_sweep.py first')
    sweep = pd.read_csv(sweep_file, keep_default_na=False, na_values=['NaN'], float_precision='round_trip')
    inputs = analysis_core.load_things_inputs()
    memorability = inputs['memorability']
    has_score = np.isfinite(memorability)
    number_of_concepts = len(memorability)
    number_with_score = int(np.sum(has_score))
    unit = 'features (%s concepts; %s with a memorability score)' % ('{:,}'.format(number_of_concepts),
                                                                    '{:,}'.format(number_with_score))

    # the embeddings, in the order of outputs/refit_sweep.csv (the published embedding first)
    embeddings = []
    for _, sweep_row in sweep.iterrows():
        if sweep_row['embedding'] == PUBLISHED_NAME:
            embedding = inputs['published_embedding']
        else:
            embedding = analysis_core.load_embedding(os.path.join(paths['data'], 'refits', sweep_row['file']))
        embeddings.append(dict(name=sweep_row['embedding'], embedding=embedding,
                               number_of_features=int(sweep_row['number_of_features']),
                               starting_features=sweep_row['starting_features'],
                               saved_r=float(sweep_row['fig5c_r'])))
    if embeddings[0]['name'] != PUBLISHED_NAME:
        raise SystemExit('outputs/refit_sweep.csv must start with the published embedding')

    tests = tests_table.TestsTable(script=SCRIPT)
    statistics_rows = []
    for entry in embeddings:
        embedding_started = time.time()
        name = entry['name']
        embedding = entry['embedding']
        number_of_features = entry['number_of_features']
        is_published = name == PUBLISHED_NAME

        # the observed statistic, the same value as step 2
        fig5c = analysis_core.fig5c_statistic(embedding, memorability, inputs['image_counts'])
        r = fig5c['r']
        parametric_p = fig5c['p']
        if r != entry['saved_r']:
            raise SystemExit('%s: r = %r differs from outputs/refit_sweep.csv (%r)' % (name, r, entry['saved_r']))

        # concept bootstrap
        if is_published:
            number_of_resamples = arguments.published_resamples
        else:
            number_of_resamples = arguments.resamples
        resample_statistic = make_resample_statistic(embedding, memorability, fig5c['summed_similarity'])
        resampled = bootstrap.percentile_bootstrap(resample_statistic, n_units=number_of_concepts,
                                                   n_resamples=number_of_resamples, seed=RELEASE_SEED)
        interval_low, interval_high = bootstrap.percentile_interval(resampled)
        interval_low = float(interval_low)
        interval_high = float(interval_high)
        resamples_without_value = int(np.sum(~np.isfinite(resampled)))
        # bootstrap p from the same resamples: twice the share on the far side of zero
        finite_resampled = resampled[np.isfinite(resampled)]
        if r < 0:
            far_side_count = int(np.sum(finite_resampled >= 0))
        else:
            far_side_count = int(np.sum(finite_resampled <= 0))
        bootstrap_p = min(1.0, 2.0 * far_side_count / len(finite_resampled))

        statistics_rows.append(dict(
            embedding=name, number_of_features=number_of_features, r=r,
            parametric_p=parametric_p, degrees_of_freedom=number_of_features - 2,
            bootstrap_low=interval_low, bootstrap_high=interval_high,
            bootstrap_resamples_without_value=resamples_without_value,
            number_of_resamples=number_of_resamples, seed=RELEASE_SEED))
        print('  %-22s k = %2d  r = %+.3f  parametric p = %.3g  95%% CI [%+.3f, %+.3f] (%d resamples)  (%.0f s)'
              % (name, number_of_features, r, parametric_p, interval_low, interval_high, number_of_resamples,
                 time.time() - embedding_started))

        # ---- the tests.csv rows of this embedding ----
        if is_published:
            ledger_ids = PUBLISHED_ITEM
            what = 'the published 49-feature SPoSE embedding (the black point of Fig. S16)'
            p_text = ('concept-bootstrap interval and p (twice the share of resamples on the far side of zero: '
                      '%d of %s)' % (far_side_count, '{:,}'.format(len(finite_resampled))))
            p_value, p_type, degrees_of_freedom = bootstrap_p, 'bootstrap two-sided', None
        else:
            ledger_ids = 'L9-030;L9-034;L9-035;L9-036'
            what = ('the SPoSE refit %s (%d features kept of %d at the start)'
                    % (name, number_of_features, int(entry['starting_features'])))
            p_text = 'parametric p, concept-bootstrap interval'
            p_value, p_type, degrees_of_freedom = parametric_p, 'parametric two-sided', number_of_features - 2
        test_text = ("Pearson r across the features of %s between each feature's Spearman correlation with "
                     "summed similarity in the same features and its Spearman correlation with concept "
                     "memorability" % what)
        tests.add(test_id='T47', ledger_ids=ledger_ids, location=LOCATION,
                  test=test_text + '; ' + p_text,
                  statistic_name='Pearson r', statistic=r, n=number_of_features, unit=unit,
                  df=degrees_of_freedom, p=p_value, p_type=p_type,
                  effect_size_name='r', effect_size=r, ci_low=interval_low, ci_high=interval_high,
                  ci_method='percentile bootstrap (all %d feature correlations recomputed in each resample; '
                            'each concept keeps its full-embedding summed similarity)' % number_of_features,
                  resampling_unit='concepts', n_resamples=number_of_resamples, seed=RELEASE_SEED)

    # ---- outputs ----
    statistics_file = os.path.join(paths['outputs'], 'figS16_statistics.csv')
    columns = ['embedding', 'number_of_features', 'r', 'parametric_p', 'degrees_of_freedom', 'bootstrap_low',
               'bootstrap_high', 'bootstrap_resamples_without_value', 'number_of_resamples', 'seed']
    with open(statistics_file, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in statistics_rows:
            cells = []
            for column in columns:
                value = row[column]
                if isinstance(value, float):
                    cells.append(repr(value))
                else:
                    cells.append(str(value))
            writer.writerow(cells)
    print('  wrote', os.path.relpath(statistics_file, release_paths.release_root()))
    tests.write(paths['outputs'])

    # ---- summary ----
    refit_rows = [row for row in statistics_rows if row['embedding'] != PUBLISHED_NAME]
    intervals_below_zero = sum(1 for row in refit_rows if row['bootstrap_high'] < 0)
    print('  refits: bootstrap interval entirely below zero in %d of %d' % (intervals_below_zero, len(refit_rows)))
    print('  step 4 took %.1f minutes' % ((time.time() - started) / 60.0))


if __name__ == '__main__':
    main()
