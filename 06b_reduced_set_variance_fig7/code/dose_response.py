"""
dose_response.py: which features lost the most predictive power in the reduced image set (Reducing Variability In
Distinctiveness Reduces Variability in Memorability; Methods, Experimental Procedures and Analysis).

WHAT IT DOES
  On the 559 concepts tested in both the maintained and the reduced set (concept CR pooled over each set's kept
  participants, as in reliability_variance.py), for each of the 49 SPoSE features: its Spearman correlation with
  concept CR in each set, and its Spearman correlation with summed similarity in the maintained set (module 06,
  outputs/summed_similarity_by_concept.csv). Tied values take their average rank.
    drop = rho^2(feature, CR) in the maintained set minus rho^2(feature, CR) in the reduced set
    tie  = rho^2(feature, summed similarity) in the maintained set
  1. The Spearman correlation, across the 49 features, between tie and drop.
  2. The same, controlling for rho^2(feature, CR) in the maintained set (features with a larger rho^2 have more
     room to drop): the ranks of tie and of drop are each regressed by least squares (with an intercept) on the
     ranks of the control, and the Spearman correlation of the two residuals is reported.
  Both p values are those of scipy.stats.spearmanr (t distribution, 47 degrees of freedom).

OUTPUTS  outputs/feature_dose_response.csv (one row per feature), outputs/dose_response.csv (the two correlations)
SEEDS    none
"""
import os
import sys

# Use at most two processor threads for the matrix arithmetic. Must be set before numpy is imported.
for thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(thread_setting, '2')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', '..', 'shared', 'python'))
import concept_tables  # noqa: E402
import release_paths  # noqa: E402
import reliability_variance as rv  # noqa: E402

PATHS = release_paths.module_paths(__file__)
SUMMED_SIMILARITY_FILE = os.path.join(release_paths.find_module('06'), 'outputs', 'summed_similarity_by_concept.csv')


def profile(F, values):
    """Spearman rho between each feature column and values."""
    f = np.apply_along_axis(rankdata, 0, F)
    f = f - f.mean(0)
    r = rankdata(values)
    r = r - r.mean()
    return (f * r[:, None]).sum(0) / np.sqrt((f ** 2).sum(0) * (r ** 2).sum())


def partial_spearman(x, y, control):
    """Spearman correlation of the residuals of rank(x) and rank(y) after least-squares regression on rank(control)."""
    rx, ry, rc = (rankdata(v) for v in (x, y, control))
    design = np.column_stack([np.ones(len(rc)), rc])
    ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    return spearmanr(ex, ey)


def main(loaded=None):
    data, shared, cols = rv.load() if loaded is None else loaded
    features = concept_tables.spose_features(concept_tables.load_concept_table())
    ss = pd.read_csv(SUMMED_SIMILARITY_FILE).set_index('concept_number')
    F = features[shared - 1]
    rho = {}
    for s in rv.SETS:
        rho[s] = profile(F, data[s]['cr'][cols[s]])
    sim = ss.loc[shared, 'summed_similarity_maintained'].to_numpy()
    good = np.isfinite(sim)
    rho_sumsim = profile(F[good], sim[good])
    start = rho['maintained'] ** 2
    drop = start - rho['reduced'] ** 2
    tie = rho_sumsim ** 2
    n_features = len(drop)
    r0 = spearmanr(tie, drop)
    r1 = partial_spearman(tie, drop, start)
    pd.DataFrame(dict(feature=np.arange(1, n_features + 1), rho_cr_maintained=rho['maintained'],
                      rho_cr_reduced=rho['reduced'], rho_sumsim_maintained=rho_sumsim)).to_csv(
        os.path.join(PATHS['outputs'], 'feature_dose_response.csv'), index=False)
    out = pd.DataFrame([dict(test='drop vs tie', rho=r0[0], p=r0[1], n=n_features),
                        dict(test='drop vs tie, controlling for rho^2 with CR in the maintained set', rho=r1[0],
                             p=r1[1], n=n_features)])
    out.to_csv(os.path.join(PATHS['outputs'], 'dose_response.csv'), index=False)
    print('  dose-response over the %d features (%d shared concepts): rho %.2f (p = %.3g); partial rho %.2f '
          '(p = %.3g)' % (n_features, len(shared), r0[0], r0[1], r1[0], r1[1]))
    return out


if __name__ == '__main__':
    main()
