"""
reliability_variance.py: the reliability-corrected variance of concept memorability in the maintained and the
reduced image set, over the concepts tested in both, and the permutation test that compares the two sets (Fig. 7
caption; Reducing Variability In Distinctiveness Reduces Variability in Memorability; Methods, Experimental
Procedures and Analysis).

WHAT IT DOES
  Concept memorability is corrected recognition (CR): the hit rate minus the false alarm rate, each pooled over the
  trials of a group of participants ("old" = responses 1 to 3), the estimator of Fig. 6. The trials and exclusions
  are those of module 06 (shared/python/imageset_loader.py; 150 participants per set). The shared concepts are the
  559 concepts with a CR in both sets.
  1. The reliability-corrected variance of each set: the participants are split at random into two halves of 75,
     CR is computed from the pooled trials of each half, and the covariance of the two halves' CR across the shared
     concepts (those with a CR in both halves) is averaged over 200 splits. Trial-to-trial noise is independent
     between the halves and does not contribute to the covariance.
  2. The permutation test: the 300 participants of both sets are assigned at random to two groups of 150 (2,000
     permutations); in each permutation the covariance of each group is averaged over 10 splits, and the difference
     between the groups enters the null distribution. The observed difference (maintained minus reduced, each set
     averaged over 200 splits drawn for the test) is compared with it: p = (1 + the number of null differences at
     least as large in absolute value) / 2,001.

RANDOM DRAWS
  One generator, numpy default_rng(SEED), makes every draw, in the order below. A split draws a random order of a
  group's participants; the first half of the order forms one half of the group. Unused splits give no result and
  hold fixed places in the order, so that the seed reproduces the reported values.
    a. 6 x 200 unused splits, of the maintained and the reduced set in turn
    b. 200 splits of the maintained set, then 200 of the reduced set (step 1)
    c. 4 x 200 unused splits, of the maintained and the reduced set in turn
    d. 200 splits of the maintained set, then 200 of the reduced set (the observed difference of step 2), then
       4 x 200 unused splits, of the two sets in turn
    e. for each of the 2,000 permutations: a random order of the 300 participants (the first 150 form the first
       group), 10 splits of each group, then 4 x 10 unused splits, of the two groups in turn

OUTPUTS  outputs/reliability_variance.csv, outputs/permutation_test.csv
SEEDS    20261024
"""
import os
import sys

# Use at most two processor threads for the matrix arithmetic (with these small matrices more threads only add
# overhead). Must be set before numpy is imported.
for thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(thread_setting, '2')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402

sys.path.insert(1, os.path.join(release_paths.find_module('06'), 'code'))
import imageset_loader  # noqa: E402
from stats_fig6_fig7 import participant_concept_counts  # noqa: E402

PATHS = release_paths.module_paths(__file__)
SETS = ('maintained', 'reduced')
N_SPLITS = 200
N_PERM = 2000
SPLITS_PER_PERM = 10
SEED = 20261024


def rates(weights, counts, cols=None):
    """Pooled hit and false alarm rate per concept over the weighted participants."""
    c = counts if cols is None else {k: v[:, cols] for k, v in counts.items()}
    with np.errstate(invalid='ignore', divide='ignore'):
        hit = (weights @ c['hits']) / (weights @ c['old_trials'])
        fa = (weights @ c['false_alarms']) / (weights @ c['new_trials'])
    return hit, fa


def split_half_covariance(rng, counts, n_splits, cols=None):
    """Mean over n_splits random splits of the participants into two halves of the covariance, across the concepts
    with a CR in both halves, of the two halves' concept CR."""
    n = counts['hits'].shape[0]
    covariances = []
    for _ in range(n_splits):
        order = rng.permutation(n)
        w1 = np.zeros(n)
        w1[order[: n // 2]] = 1
        w2 = 1 - w1
        h1, f1 = rates(w1, counts, cols)
        h2, f2 = rates(w2, counts, cols)
        cr1, cr2 = h1 - f1, h2 - f2
        ok = np.isfinite(cr1) & np.isfinite(cr2)
        covariances.append(np.cov(cr1[ok], cr2[ok])[0, 1])
    return float(np.mean(covariances))


def draw_unused_splits(rng, n_participants, n_splits):
    """The unused splits of the draw order (RANDOM DRAWS above): n_splits random orders of the participants."""
    for _ in range(n_splits):
        rng.permutation(n_participants)


def load():
    """Per-participant trial counts of each set and the shared concepts (with their columns in each set)."""
    trials = imageset_loader.load_trials(SETS, with_summed_similarity=False)
    kept, _ = imageset_loader.apply_exclusions(trials)
    data = {}
    for s in SETS:
        t = kept[kept['image_set'] == s]
        parts = sorted(t['participant'].unique())
        concepts = np.array(sorted(t['concept_number'].unique()))
        counts = participant_concept_counts(t, parts, concepts)
        hit, fa = rates(np.ones(len(parts)), counts)
        data[s] = dict(parts=parts, concepts=concepts, counts=counts, hit=hit, fa=fa, cr=hit - fa)
    ok_m = data['maintained']['concepts'][np.isfinite(data['maintained']['cr'])]
    ok_r = data['reduced']['concepts'][np.isfinite(data['reduced']['cr'])]
    shared = np.intersect1d(ok_m, ok_r)
    cols = {s: np.searchsorted(data[s]['concepts'], shared) for s in SETS}
    return data, shared, cols


def main(loaded=None):
    data, shared, cols = load() if loaded is None else loaded
    n_participants = {s: len(data[s]['parts']) for s in SETS}
    rng = np.random.default_rng(SEED)

    # Step 1: the reliability-corrected variance of each set (draws a to c)
    for _ in range(3):
        for s in SETS:
            draw_unused_splits(rng, n_participants[s], N_SPLITS)
    variance = {}
    for s in SETS:
        variance[s] = split_half_covariance(rng, data[s]['counts'], N_SPLITS, cols[s])
    for _ in range(2):
        for s in SETS:
            draw_unused_splits(rng, n_participants[s], N_SPLITS)

    # Step 2: the permutation test (draws d and e), on the participants of both sets, maintained first
    pool = {k: np.vstack([data['maintained']['counts'][k][:, cols['maintained']],
                          data['reduced']['counts'][k][:, cols['reduced']]]) for k in data['maintained']['counts']}
    n_m = n_participants['maintained']
    n_tot = pool['hits'].shape[0]

    def group_difference(group_1, group_2, n_splits):
        """Covariance of group 1 minus covariance of group 2, each averaged over n_splits splits; then the unused
        splits that follow them in the draw order."""
        cov_1 = split_half_covariance(rng, {k: v[group_1] for k, v in pool.items()}, n_splits)
        cov_2 = split_half_covariance(rng, {k: v[group_2] for k, v in pool.items()}, n_splits)
        for _ in range(2):
            draw_unused_splits(rng, len(group_1), n_splits)
            draw_unused_splits(rng, len(group_2), n_splits)
        return cov_1 - cov_2

    observed = group_difference(np.arange(n_m), np.arange(n_m, n_tot), N_SPLITS)
    null = []
    for i in range(N_PERM):
        order = rng.permutation(n_tot)
        null.append(group_difference(order[:n_m], order[n_m:], SPLITS_PER_PERM))
        if (i + 1) % 250 == 0:
            print('  permutation %d of %d' % (i + 1, N_PERM), flush=True)
    null = np.array(null)
    p_two_sided = (1 + np.sum(np.abs(null) >= abs(observed))) / (N_PERM + 1)

    pd.DataFrame([dict(image_set=s, reliability_corrected_variance=variance[s], n_concepts=len(shared),
                       n_participants=n_participants[s]) for s in SETS]).to_csv(
        os.path.join(PATHS['outputs'], 'reliability_variance.csv'), index=False)
    pd.DataFrame([dict(observed_difference=observed, null_low=np.percentile(null, 2.5),
                       null_high=np.percentile(null, 97.5), p_two_sided=p_two_sided)]).to_csv(
        os.path.join(PATHS['outputs'], 'permutation_test.csv'), index=False)
    print('  reliability-corrected variance over the %d shared concepts: maintained %.5f, reduced %.5f (%.1f%% lower)'
          % (len(shared), variance['maintained'], variance['reduced'],
             100 * (1 - variance['reduced'] / variance['maintained'])))
    print('  permutation test: difference %.5f, p two-sided %.4f' % (observed, p_two_sided))


if __name__ == '__main__':
    main()
