"""
run_module.py: entry point of module A4a (SI Appendix, Section 4, One Measure of Distinctiveness Across Analyses:
SI Appendix, Fig. S12 and Table S2).

Run from the repository root:
    python A4a_crowding_across_scales_figS12_tableS2/code/run_module.py
    python A4a_crowding_across_scales_figS12_tableS2/code/run_module.py --provided-only
        (the THINGS results from data/provided/things_results.json even when the trial table is present)

WHAT IT DOES
  1. locality_analyses.py: the locality profiles, SI Appendix, Table S2, the same concept in two image sets,
     THINGS between and within categories, and the spread of crowding in the reduced set (see its docstring).
     The THINGS parts are recomputed when the Kramer et al. trial table is in place (EXTERNAL_INPUTS.md M1:
     module A5 writes external/kramer_trials/derived/trials.npz from the trial file) and otherwise read from
     data/provided/things_results.json, which holds only correlations.
  2. make_figureS12.py: outputs/figures/FigS12_crowding_across_scales.png.
  3. outputs/numbers.csv (ledger items N4-001 to N4-026) and outputs/tests.csv (T58 to T63).
  4. outputs/logs/run_module.log: everything printed.

INPUTS   module 01 data/paper_run/fig1_paper_run.mat; module A2e outputs/continuous_rem_items_sigma0.05.csv;
         module A2f outputs/core_items_basic.csv (and its hgcm.py, to rebuild the items); module 06 data and
         outputs/summed_similarity_by_concept.csv; shared/things/concept_table.csv; optionally M1
OUTPUTS  outputs/locality_profiles.csv, table_s2.csv, concept_changes.csv, things_categories.csv,
         reduced_set_spread.csv, locality_results.txt, numbers.csv, tests.csv, figures/, figure_data/, logs/
         (and regenerated/things_results.json with M1)
SEEDS    none (the items of module A2f are rebuilt from its seed 1)
RUNTIME  about 10 seconds; under 1 GB of memory
"""
import argparse
import io
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', '..', 'shared', 'python'))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import numbers_table  # noqa: E402
import release_paths  # noqa: E402
import tests_table  # noqa: E402

import locality_analyses  # noqa: E402
import locality_common  # noqa: E402
import make_figureS12  # noqa: E402

PATHS = release_paths.module_paths(__file__)
SCRIPT = 'code/locality_analyses.py'
PEOPLE = ['THINGS', 'maintained', 'changed', 'reduced']
MODELS = ['REM-gen', 'REM-SPoSE', 'HGCM-list']
LOCATION = 'SI Appendix, Section 4, One Measure of Distinctiveness Across Analyses'
TEST_LOCATION = LOCATION + ' (Hits reflect two influences that crowding pulls in opposite directions)'
CI_METHOD = 'Fisher z with the Bonett and Wright (2000) standard error'


class Tee(io.TextIOBase):
    def __init__(self, screen):
        self.screen, self.text = screen, []

    def write(self, s):
        self.screen.write(s)
        self.text.append(s)
        return len(s)


def fisher_ci(r, n):
    """95% interval for Spearman rho: Fisher z with the Bonett and Wright (2000) standard error."""
    se = np.sqrt((1 + r ** 2 / 2) / (n - 3))
    z = np.arctanh(r)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def numbers(mode):
    prof = pd.read_csv(os.path.join(PATHS['outputs'], 'locality_profiles.csv'), dtype={'tau': str})
    s2 = pd.read_csv(os.path.join(PATHS['outputs'], 'table_s2.csv'), dtype={'tau': str})
    ch = pd.read_csv(os.path.join(PATHS['outputs'], 'concept_changes.csv'))
    cats = pd.read_csv(os.path.join(PATHS['outputs'], 'things_categories.csv'))
    spread = pd.read_csv(os.path.join(PATHS['outputs'], 'reduced_set_spread.csv'), dtype={'tau': str})
    plotted = pd.read_csv(os.path.join(PATHS['figure_data'], make_figureS12.FIGURE_NAME + '.csv'), dtype={'tau': str})
    concept_table = pd.read_csv(release_paths.shared_path('things', 'concept_table.csv'))
    if mode == 'regenerate':
        things_file = os.path.join(PATHS['outputs'], 'regenerated', 'things_results.json')
        things_mode = mixed_mode = 'rerun'
    else:
        things_file = os.path.join(PATHS['provided'], 'things_results.json')
        things_mode, mixed_mode = 'provided', 'rerun_and_provided'
    with open(things_file, encoding='utf-8') as handle:
        published = json.load(handle)['published_check']
    allp = prof[prof.scope == 'all']
    sel = lambda ds, m, scope='all': prof[(prof.dataset.isin(ds)) & (prof.measure == m) & (prof.scope == scope)].rho
    T = numbers_table.NumbersTable(SCRIPT)

    taus = [float(t) for t in allp.tau.unique() if t != 'all']
    T.add('N4-001', taus + [allp.dataset.nunique()], mixed_mode,
          note='kernel falloffs below the all-items limit; number of datasets in outputs/locality_profiles.csv')
    T.add('N4-002', published, things_mode,
          note='Spearman rho, THINGS concept hit rate minus false alarm rate against the published concept scores '
               '(shared/things/concept_table.csv)')
    fa = allp[allp.measure == 'fa']
    n_pos = int((fa.groupby('dataset').rho.min() > 0).sum())
    T.add('N4-003', [n_pos, fa.rho.min(), fa.rho.max()], mixed_mode,
          note='datasets whose false alarm correlation is above 0 at every tau; smallest and largest false alarm '
               'rho over datasets and taus')
    w = sel(['maintained', 'changed'], 'fa', 'within category')
    T.add('N4-004', [w.min(), w.max()], 'rerun',
          note='within-category false alarm rho, maintained and changed sets, all taus')
    rem_hits = sel(['REM-gen', 'REM-SPoSE'], 'hit')
    hyb_hits = sel(['HGCM-list'], 'hit')
    T.add('N4-005', 'REM hit rho %.2f to %.2f; hybrid hit rho %.2f to %.2f' % (rem_hits.min(), rem_hits.max(),
          hyb_hits.min(), hyb_hits.max()), 'rerun', note='hit correlations of the models over taus')
    ph = sel(PEOPLE, 'hit')
    T.add('N4-006', 'people hit rho %.2f to %.2f' % (ph.min(), ph.max()), mixed_mode,
          note='hit correlations of the four human datasets over taus')
    g = ch.set_index(['sets', 'measure']).rho
    T.add('N4-007', [g['maintained to reduced', 'fa'], g['maintained to changed', 'fa'],
                     g['maintained to reduced', 'hit'], g['maintained to changed', 'hit']], 'rerun',
          note='Spearman rho, change in local crowding (tau 0.05) against change in false alarm rate and in hit rate')
    c = cats.set_index('component').rho
    T.add('N4-008', [c['between categories'], c['within categories']], things_mode,
          note='THINGS local crowding (tau 0.05) against concept hit rate, category means and deviations from them')
    rng = []
    for ds in PEOPLE:
        v = sel([ds], 'dprime')
        rng += [v.max(), v.min()]
    v = sel(MODELS, 'dprime')
    rng += [v.max(), v.min()]
    T.add('N4-009', rng, mixed_mode, note="largest and smallest d' rho over taus: THINGS, maintained, changed, "
          'reduced, then over the three models')
    wi = sel(['maintained', 'changed', 'reduced'], 'dprime', 'within category')
    wt = sel(['THINGS'], 'dprime', 'within category')
    T.add('N4-010', [wi.max(), wi.min(), wt.max(), wt.min()], mixed_mode,
          note="within-category d' rho: image sets, then THINGS")
    s = s2[s2.measure.isin(['cr', 'dprime'])].set_index(['dataset', 'measure', 'tau']).r.sort_index()
    T.add('N4-011', "THINGS d' below 0 at every tau: %s; THINGS corrected recognition above 0 at tau 0.05 and 0.2 "
          "and below 0 at tau 1 and all: %s; changed set below 0 at every tau with both measures: %s; maintained set "
          "below -0.5 at tau 0.2, 1 and all and within 0.1 of 0 at tau 0.05 with both measures: %s" % (
              bool((s['THINGS', 'dprime'] < 0).all()),
              bool(s['THINGS', 'cr', '0.05'] > 0 and s['THINGS', 'cr', '0.2'] > 0
                   and s['THINGS', 'cr', '1'] < 0 and s['THINGS', 'cr', 'all'] < 0),
              bool((s['changed'] < 0).all()),
              bool((s['maintained'].drop('0.05', level='tau') < -0.5).all()
                   and (s['maintained'].xs('0.05', level='tau').abs() < 0.1).all())),
          mixed_mode, note="signs and sizes in outputs/table_s2.csv (corrected recognition and d' rows)")
    for ledger_id, ds, m in [('N4-012', 'THINGS', 'cr'), ('N4-013', 'THINGS', 'dprime'),
                             ('N4-014', 'maintained', 'cr'), ('N4-015', 'maintained', 'dprime'),
                             ('N4-016', 'changed', 'cr'), ('N4-017', 'changed', 'dprime')]:
        T.add(ledger_id, [s[ds, m, t] for t in ('0.05', '0.2', '1', 'all')],
              things_mode if ds == 'THINGS' else 'rerun', note='Fig. 5c statistic at tau 0.05, 0.2, 1 and all')
    sp = spread.set_index('tau').ratio_reduced_to_maintained
    T.add('N4-018', [sp['all'], sp['0.2'], 0.2, sp['0.05'], 0.05], 'rerun',
          note='SD of crowding across concepts, reduced / maintained (all concepts with slots in each set)')
    T.add('N4-019', make_figureS12.describe(), mixed_mode,
          note='panels and line styles of code/make_figureS12.py; data from outputs/locality_profiles.csv (scope all)',
          script='code/make_figureS12.py')
    things_fa = sel(['THINGS'], 'fa')
    T.add('N4-020', "THINGS, Table S2 d' below 0 at every tau: %s; THINGS false alarm rho above 0 at every tau: %s"
          % (bool((s['THINGS', 'dprime'] < 0).all()),
             bool(len(things_fa) == len(locality_common.TAUS) and (things_fa > 0).all())),
          things_mode, note="outputs/table_s2.csv (THINGS, d') and outputs/locality_profiles.csv (THINGS, false alarms)")
    T.add('N4-021', 'locality_common.crowding: weighted mean of exp(-d/tau) over the other items of the reference '
          'set (weights: THINGS image counts, set slots, or 1); d = 1 - Pearson r of SPoSE vectors (concepts) or '
          'Euclidean distance / its median (simulated items); tau = 1e6 for the all-items limit', 'rerun',
          note='the crowding measure, code/locality_common.py')
    T.add('N4-022', 'datasets: %s' % ', '.join(allp.dataset.unique()), mixed_mode,
          note='REM-gen: module 01 (Fig. 1); REM-SPoSE: module A2e, retention noise 0.05 (SI Appendix, Fig. S6); '
               'HGCM-list: module A2f, variant Hybrid GCM, list (SI Appendix, Fig. S7); THINGS: hits on target '
               'repeats, false alarms on first showings; maintained, changed, reduced: module 06')
    t05 = allp[(allp.dataset == 'THINGS') & (allp.tau == '0.05')].set_index('measure').rho
    T.add('N4-023', "THINGS at tau 0.05: hit rho %.2f, corrected recognition rho %.2f, d' rho %.2f"
          % (t05['hit'], t05['cr'], t05['dprime']), things_mode,
          note='nearest-neighbor crowding against the THINGS concept rates (outputs/locality_profiles.csv)')
    T.add('N4-024', sum(1 for column in concept_table.columns if column.startswith('feature_')), 'rerun',
          note='SPoSE features in shared/things/concept_table.csv, the features of Table S2')
    T.add('N4-025', min(float(t) for t in plotted.tau.unique() if t != 'all'), 'rerun',
          note='smallest tau on the x axis of SI Appendix, Fig. S12 (outputs/figure_data/%s.csv)'
               % make_figureS12.FIGURE_NAME, script='code/make_figureS12.py')
    T.add('N4-026', "THINGS at tau 0.05: hit rho %.2f, corrected recognition rho %.2f, d' rho %.2f"
          % (t05['hit'], t05['cr'], t05['dprime']), 'restatement',
          note='nearest-neighbor crowding against the THINGS concept rates (outputs/locality_profiles.csv), the '
               'SI Appendix, Section 4 result of N4-023')
    T.write(PATHS['outputs'])

    tt = tests_table.TestsTable(SCRIPT)
    for test_id, r in zip(('T58', 'T59', 'T60', 'T61'), ch.itertuples()):
        lo, hi = fisher_ci(r.rho, r.n)
        tt.add(test_id=test_id, ledger_ids='N4-007', location=TEST_LOCATION,
               test='Spearman rho across the concepts tested in both sets (%s) between the change in local crowding '
                    '(tau = 0.05) and the change in the concept %s rate' % (r.sets, 'false alarm' if r.measure == 'fa' else 'hit'),
               statistic_name='Spearman rho', statistic=r.rho, n=r.n, unit='concepts', df=r.n - 2, p=r.p,
               p_type='parametric two-sided', effect_size_name='rho', effect_size=r.rho, ci_low=lo, ci_high=hi,
               ci_method=CI_METHOD)
    category = concept_table.category_number.fillna(0)
    n_categories = int(category[category > 0].nunique())
    between = cats.set_index('component').loc['between categories']
    tt.add(test_id='T62', ledger_ids='N4-008', location=TEST_LOCATION,
           test='Spearman rho over the %d categorized THINGS concepts between the category mean of local crowding '
                '(tau = 0.05) and the category mean of concept hit rate (%d higher-level categories); every concept '
                'carries the means of its category, so the correlation is descriptive (no p value or interval)'
                % (between.n, n_categories),
           statistic_name='Spearman rho', statistic=float(between.rho), n=int(between.n), unit='concepts',
           p_type='none', effect_size_name='rho', effect_size=float(between.rho))
    within = cats.set_index('component').loc['within categories']
    lo, hi = fisher_ci(within.rho, within.n)
    tt.add(test_id='T63', ledger_ids='N4-008', location=TEST_LOCATION,
           test='Spearman rho over the %d categorized THINGS concepts between local crowding (tau = 0.05) and concept '
                'hit rate, each as the deviation from its category mean' % within.n,
           statistic_name='Spearman rho', statistic=float(within.rho), n=int(within.n), unit='concepts',
           df=int(within.n) - 2, p=float(within.p), p_type='parametric two-sided', effect_size_name='rho',
           effect_size=float(within.rho), ci_low=lo, ci_high=hi, ci_method=CI_METHOD)
    tt.write(PATHS['outputs'])


def main():
    parser = argparse.ArgumentParser(description='Module A4a: SI Appendix, Section 4, Fig. S12 and Table S2')
    parser.add_argument('--provided-only', action='store_true',
                        help='use data/provided/things_results.json even when the Kramer et al. trial table is present')
    args = parser.parse_args()
    tee = Tee(sys.stdout)
    sys.stdout = tee
    try:
        if args.provided_only:
            mode = 'provided'
            print('--provided-only: the THINGS results come from data/provided/things_results.json')
        else:
            mode = release_paths.regeneration_mode([locality_common.TRIAL_INPUT])
        locality_analyses.main(mode)
        make_figureS12.main()
        numbers(mode)
    finally:
        sys.stdout = tee.screen
        with open(os.path.join(PATHS['logs'], 'run_module.log'), 'w', encoding='utf-8') as f:
            f.write(''.join(tee.text))


if __name__ == '__main__':
    main()
