"""
locality_common.py: the crowding measure of SI Appendix, Section 4 (One Measure of Distinctiveness Across
Analyses) and the datasets it is applied to.

Crowding of item i against a reference set with weights w_j (image counts, set slots, or 1):

    crowding_tau(i) = sum_j w_j exp(-d_ij / tau) / sum_j w_j          (j != i; higher = more crowded)

d is 1 - Pearson r of two concepts' SPoSE vectors, or the Euclidean distance between two simulated items divided
by its median over item pairs. Small tau: only the nearest neighbors count. tau = 1e6 is the all-items limit,
where every item of the reference set counts equally.

Datasets (each gives per-item hit and false alarm rates):
  REM-gen     Fig. 1 simulation (module 01, data/paper_run/fig1_paper_run.mat), 1,600 generated items
  REM-SPoSE   SI Appendix, Fig. S6 model at retention noise 0.05 (module A2e,
              outputs/continuous_rem_items_sigma0.05.csv), 1,600 concepts
  HGCM-list   SI Appendix, Fig. S7 hybrid model with rarity computed from the study list (module A2f,
              outputs/core_items_basic.csv, variant "Hybrid GCM, list"); the items are rebuilt from the seed
              of A2f's 01_core.py, so their crowding is computed from the same feature vectors
  THINGS      Kramer et al. (2023) trial table (EXTERNAL_INPUTS.md M1; module A5 writes
              external/kramer_trials/derived/trials.npz from the trial file): concept hit rate on target
              repeats, false alarm rate on first showings; reference = all 1,854 concepts weighted by image
              count. Only when the trial table is in place; otherwise the THINGS results come from
              data/provided/ (correlations only).
  maintained / changed / reduced   the image-set experiments of module 06 (its loader and exclusions;
              pooled concept rates; reference = the set's slots)
THINGS and the image sets also carry each concept's higher-level category (0 = none), for the within-category
correlations.
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.spatial.distance import cdist
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402

ROOT = release_paths.release_root()
sys.path.insert(1, os.path.join(ROOT, '06_image_set_experiments_fig6_fig7', 'code'))
sys.path.insert(2, os.path.join(ROOT, 'A2f_hybrid_exemplar_model_figS7', 'code'))

TAUS = [0.05, 0.1, 0.2, 0.4, 1.0, 1e6]
TRIAL_TABLE = os.path.join(ROOT, 'external', 'kramer_trials', 'derived', 'trials.npz')
# The THINGS trial table (EXTERNAL_INPUTS.md M1), for release_paths.regeneration_mode
TRIAL_INPUT = release_paths.Input('M1', 'the Kramer et al. trial table that module A5 writes from the trial file',
                                  'external/kramer_trials/derived/trials.npz')
SETS = ('maintained', 'changed', 'reduced')


def tau_label(tau):
    return 'all' if tau > 1e5 else '%g' % tau


def crowding(dissim, w, tau):
    k = np.exp(-dissim / tau)
    np.fill_diagonal(k, 0.0)
    return (k @ w) / (w.sum() - w)


def dprime(h, f):
    """d' with rates clipped to [0.005, 0.995]."""
    h = np.clip(h, 0.005, 0.995)
    f = np.clip(f, 0.005, 0.995)
    return norm.ppf(h) - norm.ppf(f)


def within(values, groups):
    v = pd.Series(values)
    return (v - v.groupby(groups).transform('mean')).to_numpy()


def concept_space():
    """The concept table, SPoSE dissimilarity (1 - r), higher-level category (0 = none) and image counts."""
    ct = pd.read_csv(os.path.join(ROOT, 'shared', 'things', 'concept_table.csv'))
    F = ct[[c for c in ct.columns if c.startswith('feature_')]].to_numpy()
    return dict(table=ct, features=F, dissim=1 - np.corrcoef(F),
                category=ct['category_number'].fillna(0).to_numpy().astype(int),
                counts=ct['concept_image_count_as_stored'].to_numpy(float))


def scaled_euclidean(X):
    D = cdist(X, X)
    return D / np.median(D[np.triu_indices_from(D, 1)])


def rem_generated():
    fd = sio.loadmat(os.path.join(ROOT, '01_rem_simulation_fig1', 'data', 'paper_run', 'fig1_paper_run.mat'),
                     squeeze_me=True, struct_as_record=False)['figureData']
    X = np.asarray(fd.imagesOfInterest, float)
    return dict(dissim=scaled_euclidean(X), weights=np.ones(len(X)), items=np.arange(len(X)),
                hit=np.asarray(fd.hitRate, float).ravel(), fa=np.asarray(fd.faRate, float).ravel(), category=None)


def rem_spose(cs):
    s6 = pd.read_csv(os.path.join(ROOT, 'A2e_measured_representations_figS6', 'outputs',
                                  'continuous_rem_items_sigma0.05.csv'))
    rows = s6['concept_row_in_embedding'].to_numpy() - 1      # concept table and SPoSE file share row order
    w = np.zeros(len(cs['table']))
    w[rows] = 1
    return dict(dissim=cs['dissim'], weights=w, items=rows,
                hit=(s6.hit_count / s6.old_tests).to_numpy(), fa=(s6.false_alarm_count / s6.new_tests).to_numpy(),
                category=None)


def hybrid_list():
    """Per-item rates of the SI Appendix, Fig. S7 run (A2f) for the list-rarity hybrid model, with the A2f items."""
    import hgcm  # A2f's model code (on the path above)
    core = pd.read_csv(os.path.join(ROOT, 'A2f_hybrid_exemplar_model_figS7', 'outputs', 'core_items_basic.csv'))
    core = core[core.variant == 'Hybrid GCM, list'].sort_values('item')
    # the items of A2f's 01_core.py: the same seed and the same two draws
    rng = np.random.default_rng(1)
    full = hgcm.make_items(rng, 26107, 49, 0.65)
    X = full[rng.choice(26107, 1600, replace=False)].astype(float)
    neg_mean_dist, _ = hgcm.crowding(X)
    r = np.corrcoef(neg_mean_dist[core.item.to_numpy()], core.crowding.to_numpy())[0, 1]
    if r < 0.999999:
        raise RuntimeError('rebuilt A2f items do not match outputs/core_items_basic.csv (r = %.6f)' % r)
    return dict(dissim=scaled_euclidean(X), weights=np.ones(len(X)), items=core.item.to_numpy(),
                hit=(core.hits / core.old_tests).to_numpy(), fa=(core.false_alarms / core.new_tests).to_numpy(),
                category=None)


def things(cs):
    """Concept hit and false alarm rates from the Kramer et al. trial table (M1). Never written to disk."""
    tr = np.load(TRIAL_TABLE)
    concept = tr['concept'].astype(int)
    ttype = tr['ttype']
    perf = tr['perf']
    n = len(cs['table'])
    old = ttype == 2
    new = (ttype == 1) | (ttype == 3)
    hits = np.bincount(concept[old] - 1, weights=(perf[old] == 11), minlength=n)
    n_old = np.bincount(concept[old] - 1, minlength=n)
    fas = np.bincount(concept[new] - 1, weights=(perf[new] == 13), minlength=n)
    n_new = np.bincount(concept[new] - 1, minlength=n)
    items = np.where((n_old > 0) & (n_new > 0))[0]
    return dict(dissim=cs['dissim'], weights=cs['counts'], items=items, hit=hits[items] / n_old[items],
                fa=fas[items] / n_new[items], category=cs['category'][items])


def image_sets(cs):
    """Pooled concept rates in the three image sets (module 06 loader and exclusions)."""
    import imageset_loader
    from stats_fig6_fig7 import participant_concept_counts
    ss = pd.read_csv(os.path.join(ROOT, '06_image_set_experiments_fig6_fig7', 'outputs',
                                  'summed_similarity_by_concept.csv'))
    ss = ss.set_index('concept_number').reindex(cs['table']['concept_number'])
    trials = imageset_loader.load_trials(SETS, data_folder=os.path.join(ROOT, '06_image_set_experiments_fig6_fig7', 'data'),
                                         with_summed_similarity=False)
    kept, _ = imageset_loader.apply_exclusions(trials)
    out = {}
    for s in SETS:
        t = kept[kept['image_set'] == s]
        parts = sorted(t['participant'].unique())
        concepts = np.array(sorted(t['concept_number'].unique()))
        c = participant_concept_counts(t, parts, concepts)
        o, h, nw, fa = (c[k].sum(0) for k in ('old_trials', 'hits', 'new_trials', 'false_alarms'))
        good = (o > 0) & (nw > 0)
        items = concepts[good] - 1
        out[s] = dict(dissim=cs['dissim'], weights=ss['slots_%s' % s].fillna(0).to_numpy(), items=items,
                      hit=h[good] / o[good], fa=fa[good] / nw[good], category=cs['category'][items])
    return out


def all_datasets(include_things):
    """Ordered dict name -> dataset, and the concept space. THINGS is included when include_things is True
    (the trial table of M1 is in place)."""
    cs = concept_space()
    d = {'REM-gen': rem_generated(), 'REM-SPoSE': rem_spose(cs), 'HGCM-list': hybrid_list()}
    if include_things:
        d['THINGS'] = things(cs)
    sets = image_sets(cs)
    for s in SETS:
        d[s] = sets[s]
    return d, cs
