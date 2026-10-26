"""
run_module.py: entry point of module 06b, the reliability-corrected variance of concept memorability on the
concepts tested in both the maintained and the reduced image set, and the feature dose-response (Fig. 7 caption;
Reducing Variability In Distinctiveness Reduces Variability in Memorability; Methods, Experimental Procedures and
Analysis). It reads the trials of module 06 through shared/python/imageset_loader.py, which applies the exclusions
of the Methods.

Run from the repository root:
    python 06b_reduced_set_variance_fig7/code/run_module.py

WHAT IT DOES
  1. reliability_variance.py: the reliability-corrected variance of concept memorability in the maintained and the
     reduced set over the 559 shared concepts, and the permutation test that compares the two sets.
  2. dose_response.py: the feature dose-response across the 49 SPoSE features.
  3. outputs/numbers.csv and outputs/tests.csv (tests T55 to T57).
  4. outputs/logs/run_module.log: everything printed.

INPUTS   module 06's data/ (trials) and outputs/summed_similarity_by_concept.csv; shared/things/concept_table.csv;
         shared/ledger/paper_numbers.csv
OUTPUTS  outputs/reliability_variance.csv, permutation_test.csv, feature_dose_response.csv, dose_response.csv,
         numbers.csv, tests.csv, logs/run_module.log
SEEDS    20261024 (reliability_variance.py)
RUNTIME  about 10 seconds on an Apple M2 Pro; under 1 GB of memory
"""
import io
import os
import sys

# Use at most two processor threads for the matrix arithmetic (with these small matrices more threads only add
# overhead). Must be set before numpy is imported.
for thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(thread_setting, '2')

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', '..', 'shared', 'python'))

import pandas as pd  # noqa: E402

import numbers_table  # noqa: E402
import release_paths  # noqa: E402
import tests_table  # noqa: E402

import dose_response  # noqa: E402
import reliability_variance  # noqa: E402

PATHS = release_paths.module_paths(__file__)
LOCATION_RESULTS = 'Reducing Variability In Distinctiveness Reduces Variability in Memorability'
LOCATION_METHODS = 'Methods, Experimental Procedures and Analysis'


class Tee(io.TextIOBase):
    def __init__(self, screen):
        self.screen, self.text = screen, []

    def write(self, s):
        self.screen.write(s)
        self.text.append(s)
        return len(s)


def tables(n_shared):
    rel = pd.read_csv(os.path.join(PATHS['outputs'], 'reliability_variance.csv')).set_index('image_set')
    perm = pd.read_csv(os.path.join(PATHS['outputs'], 'permutation_test.csv')).iloc[0]
    dr = pd.read_csv(os.path.join(PATHS['outputs'], 'dose_response.csv'))
    v = rel.reliability_corrected_variance
    drop = 100 * (1 - v['reduced'] / v['maintained'])
    n_parts = int(rel.n_participants.sum())
    n_group = int(rel.n_participants['maintained'])
    n_features = int(dr.n[0])
    n_splits, n_perm, splits_per_perm = (reliability_variance.N_SPLITS, reliability_variance.N_PERM,
                                         reliability_variance.SPLITS_PER_PERM)

    T = numbers_table.NumbersTable('code/reliability_variance.py')
    note = ('percent by which the reliability-corrected variance of concept CR (shared concepts, mean over %d splits) '
            'is lower in the reduced set (%.5f against %.5f); two-sided permutation p'
            % (n_splits, v['reduced'], v['maintained']))
    T.add('N3-001', [drop, perm.p_two_sided], 'rerun', note=note)
    T.add('N3-002', [dr.rho[0], n_features, dr.p[0], dr.rho[1], dr.p[1]], 'rerun', script='code/dose_response.py',
          note='Spearman rho across the features between the drop in rho^2(feature, CR) from the maintained to the '
               'reduced set and rho^2(feature, summed similarity) in the maintained set; number of features; its p; '
               'the partial rho controlling for rho^2(feature, CR) in the maintained set; its p')
    T.add('N3-003', [n_shared, n_splits], 'rerun', note='shared concepts; random splits per set')
    T.add('N3-004', [n_parts, n_group, n_perm, splits_per_perm], 'rerun',
          note='participants of both sets; participants per group; permutations; splits per group in each permutation')
    T.add('N3-005', [n_shared, drop, perm.p_two_sided], 'rerun', note='shared concepts; ' + note)
    T.add('N3-006', "concept CR pooled over the trials of each half; covariance of the two halves' CR across the "
          'concepts with a CR in both halves; mean over %d splits: maintained %.5f, reduced %.5f'
          % (n_splits, v['maintained'], v['reduced']), 'rerun',
          note='the reliability-corrected variance (split_half_covariance in code/reliability_variance.py)')
    T.add('N3-007', [n_features], 'rerun', script='code/dose_response.py',
          note='features across which the drop in rho^2(feature, CR) is correlated with rho^2(feature, summed '
               'similarity) in the maintained set, directly and as a partial Spearman correlation')
    T.write(PATHS['outputs'])

    tt = tests_table.TestsTable('code/reliability_variance.py')
    tt.add(test_id='T55', ledger_ids='N3-001;N3-005',
           location='%s; Fig. 7 caption; %s' % (LOCATION_RESULTS, LOCATION_METHODS),
           test='Permutation test on the %d concepts tested in both sets: the reliability-corrected variance of '
                'concept corrected recognition (between-half covariance, mean over splits), maintained minus reduced, '
                'each set averaged over %d splits drawn for the test, against %d random assignments of the %d '
                'participants to two groups of %d (%d splits per group); effect size: percent reduction of the '
                'variances in reliability_variance.csv (%d splits per set)'
                % (n_shared, n_splits, n_perm, n_parts, n_group, splits_per_perm, n_splits),
           statistic_name='difference in reliability-corrected variance', statistic=perm.observed_difference,
           n=n_parts, unit='participants', p=perm.p_two_sided, p_type='permutation two-sided',
           effect_size_name='percent lower in the reduced set', effect_size=drop,
           ci_method='none (the 95%% range of the permutation null is %.5f to %.5f)' % (perm.null_low, perm.null_high),
           resampling_unit='participants', n_resamples=n_perm, seed=reliability_variance.SEED)
    dose_test = ('across the %d SPoSE features between the drop in squared Spearman correlation with concept '
                 'corrected recognition (maintained minus reduced, %d shared concepts) and the squared correlation '
                 'with summed similarity in the maintained set' % (n_features, n_shared))
    tt.add(test_id='T56', ledger_ids='N3-002', location='%s; %s' % (LOCATION_RESULTS, LOCATION_METHODS),
           test='Spearman rho ' + dose_test,
           statistic_name='Spearman rho', statistic=dr.rho[0], n=n_features, unit='features', df=n_features - 2,
           p=dr.p[0], p_type='parametric two-sided', effect_size_name='rho', effect_size=dr.rho[0],
           script='code/dose_response.py')
    tt.add(test_id='T57', ledger_ids='N3-002', location='%s; %s' % (LOCATION_RESULTS, LOCATION_METHODS),
           test='Partial Spearman rho ' + dose_test + ', controlling for the squared correlation with corrected '
                'recognition in the maintained set: the Spearman correlation of the residuals of the ranks of both '
                'variables after least-squares regression on the ranks of the control',
           statistic_name='partial Spearman rho', statistic=dr.rho[1], n=n_features, unit='features',
           df=n_features - 2, p=dr.p[1], p_type='parametric two-sided', effect_size_name='rho',
           effect_size=dr.rho[1], script='code/dose_response.py')
    tt.write(PATHS['outputs'])


def main():
    tee = Tee(sys.stdout)
    sys.stdout = tee
    try:
        loaded = reliability_variance.load()
        reliability_variance.main(loaded)
        dose_response.main(loaded)
        tables(n_shared=len(loaded[1]))
    finally:
        sys.stdout = tee.screen
        with open(os.path.join(PATHS['logs'], 'run_module.log'), 'w', encoding='utf-8') as f:
            f.write(''.join(tee.text))


if __name__ == '__main__':
    main()
