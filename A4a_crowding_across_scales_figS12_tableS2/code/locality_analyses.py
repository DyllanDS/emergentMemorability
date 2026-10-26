"""
locality_analyses.py: the analyses of SI Appendix, Section 4 (One Measure of Distinctiveness Across Analyses).

WHAT IT DOES
  1. Locality profiles (SI Appendix, Fig. S12): in every dataset of locality_common.py, the Spearman correlation
     across items between crowding at each tau and the hit rate, false alarm rate, hit rate minus false alarm rate
     and d'. For THINGS and the three image sets the same correlations within higher-level categories: crowding
     and the memory measure are each centered on their category means first (concepts without a category are
     left out).
  2. SI Appendix, Table S2: the Fig. 5c statistic (Pearson r across the 49 SPoSE features between each feature's
     Spearman correlation with crowding and its Spearman correlation with the memory measure) at tau = 0.05, 0.2,
     1 and all, with corrected recognition and with d', for THINGS and the maintained and changed sets.
  3. The same concept in two image sets: for the concepts tested in both the maintained and the reduced set,
     and in both the maintained and the changed set, the Spearman correlation between the change in local
     crowding (tau = 0.05) and the change in the concept's false alarm rate and hit rate.
  4. THINGS, between and within higher-level categories: local crowding (tau = 0.05) against concept hit rate,
     using each category's means (between) and the deviations from them (within), over the categorized concepts.
     Every concept carries the means of its category, so the between-category correlation has no p value.
  5. What the reduced set equalized: the SD of crowding across the concepts of the reduced set divided by that of
     the maintained set, at tau = 0.05, 0.2 and all (all concepts with slots in the set).
  The THINGS parts (1, 2 and 4) need the Kramer et al. trial table (EXTERNAL_INPUTS.md M1). Without it they are
  read from data/provided/things_results.json, which holds only correlations; with it they are recomputed,
  written to outputs/regenerated/things_results.json and compared with that file. No per-trial, per-image or
  per-concept THINGS value is written anywhere.

OUTPUTS  outputs/locality_profiles.csv, outputs/table_s2.csv, outputs/concept_changes.csv,
         outputs/things_categories.csv, outputs/reduced_set_spread.csv, outputs/locality_results.txt
         (and outputs/regenerated/things_results.json with M1)
SEEDS    none (the items of module A2f are rebuilt from its seed 1; see locality_common.py)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

sys.dont_write_bytecode = True
import locality_common as lc  # noqa: E402
import release_paths  # noqa: E402

PATHS = release_paths.module_paths(__file__)
OUT = PATHS['outputs']
PROVIDED = os.path.join(PATHS['provided'], 'things_results.json')
REGENERATED = os.path.join(OUT, 'regenerated', 'things_results.json')
LOCAL = 0.05
TABLE_TAUS = [0.05, 0.2, 1.0, 1e6]
TABLE_MEASURES = ('cr', 'dprime')
SPREAD_TAUS = [0.05, 0.2, 1e6]


def memory_measures(hit, fa):
    return {'hit': hit, 'fa': fa, 'cr': hit - fa, 'dprime': lc.dprime(hit, fa)}


def feature_profile(F, values):
    f = np.apply_along_axis(rankdata, 0, F)
    f = f - f.mean(0)
    r = rankdata(values)
    r = r - r.mean()
    return (f * r[:, None]).sum(0) / np.sqrt((f ** 2).sum(0) * (r ** 2).sum())


def profiles(name, d):
    rows = []
    mem = memory_measures(d['hit'], d['fa'])
    for tau in lc.TAUS:
        c = lc.crowding(d['dissim'], d['weights'], tau)[d['items']]
        for m, v in mem.items():
            rows.append(dict(dataset=name, tau=lc.tau_label(tau), scope='all', measure=m,
                             rho=spearmanr(c, v)[0], n=len(v)))
        if d['category'] is not None:
            ok = d['category'] > 0
            cw = lc.within(c[ok], d['category'][ok])
            for m, v in mem.items():
                rows.append(dict(dataset=name, tau=lc.tau_label(tau), scope='within category', measure=m,
                                 rho=spearmanr(cw, lc.within(v[ok], d['category'][ok]))[0], n=int(ok.sum())))
    return rows


def table_s2(name, d, F):
    rows = []
    mem = memory_measures(d['hit'], d['fa'])
    Fi = F[d['items']]
    mem_prof = {m: feature_profile(Fi, mem[m]) for m in TABLE_MEASURES}
    for tau in TABLE_TAUS:
        pc = feature_profile(Fi, lc.crowding(d['dissim'], d['weights'], tau)[d['items']])
        for m in TABLE_MEASURES:
            rows.append(dict(dataset=name, tau=lc.tau_label(tau), measure=m, r=pearsonr(pc, mem_prof[m])[0]))
    return rows


def concept_changes(data, n_concepts):
    rows = []
    full = {}
    for s in lc.SETS:
        d = data[s]
        hit = np.full(n_concepts, np.nan)
        fa = np.full(n_concepts, np.nan)
        hit[d['items']] = d['hit']
        fa[d['items']] = d['fa']
        full[s] = dict(hit=hit, fa=fa, local=lc.crowding(d['dissim'], d['weights'], LOCAL))
    for a, b in (('maintained', 'reduced'), ('maintained', 'changed')):
        A, B = full[a], full[b]
        ok = np.isfinite(A['hit']) & np.isfinite(B['hit']) & np.isfinite(A['fa']) & np.isfinite(B['fa'])
        d_local = B['local'][ok] - A['local'][ok]
        for m in ('fa', 'hit'):
            r, p = spearmanr(d_local, B[m][ok] - A[m][ok])
            rows.append(dict(sets='%s to %s' % (a, b), measure=m, rho=r, p=p, n=int(ok.sum())))
    return rows


def things_categories(d):
    ok = d['category'] > 0
    g = d['category'][ok]
    local = lc.crowding(d['dissim'], d['weights'], LOCAL)[d['items']][ok]
    hit = d['hit'][ok]
    between = spearmanr(pd.Series(local).groupby(g).transform('mean'), pd.Series(hit).groupby(g).transform('mean'))
    within = spearmanr(lc.within(local, g), lc.within(hit, g))
    return [dict(component='between categories', rho=between[0], p=None, n=int(ok.sum())),
            dict(component='within categories', rho=within[0], p=within[1], n=int(ok.sum()))]


def reduced_set_spread(cs):
    ss = pd.read_csv(os.path.join(lc.ROOT, '06_image_set_experiments_fig6_fig7', 'outputs',
                                  'summed_similarity_by_concept.csv'))
    ss = ss.set_index('concept_number').reindex(cs['table']['concept_number'])
    rows = []
    for tau in SPREAD_TAUS:
        sd = {}
        for s in ('maintained', 'reduced'):
            slots = ss['slots_%s' % s].fillna(0).to_numpy()
            sd[s] = np.std(lc.crowding(cs['dissim'], slots, tau)[slots > 0])
        rows.append(dict(tau=lc.tau_label(tau), sd_maintained=sd['maintained'], sd_reduced=sd['reduced'],
                         ratio_reduced_to_maintained=sd['reduced'] / sd['maintained']))
    return rows


def published_check(d, cs):
    """Spearman of the THINGS concept hit rate minus false alarm rate with the published concept scores."""
    pub = cs['table']['kramer_concept_cr'].to_numpy()[d['items']]
    return spearmanr(d['hit'] - d['fa'], pub, nan_policy='omit')[0]


def reported_things(things):
    """The THINGS results this module uses: the locality profiles, the corrected recognition and d' rows of
    Table S2, the two category correlations (no p value between categories) and the correlation with the
    published scores."""
    return dict(profiles=[dict(r) for r in things['profiles']],
                table_s2=[dict(r) for r in things['table_s2'] if r['measure'] in TABLE_MEASURES],
                categories=[dict(r, p=None) if r['component'] == 'between categories' else dict(r)
                            for r in things['categories']],
                published_check=things['published_check'])


def load_provided():
    with open(PROVIDED, encoding='utf-8') as handle:
        return reported_things(json.load(handle))


def largest_difference(saved, things):
    """Largest absolute difference between two sets of THINGS results, matched by their labels, and the number
    of correlations compared."""
    def values(results):
        out = {('published check',): results['published_check']}
        for r in results['profiles']:
            out[('profile', r['dataset'], r['tau'], r['scope'], r['measure'])] = r['rho']
        for r in results['table_s2']:
            out[('Table S2', r['dataset'], r['tau'], r['measure'])] = r['r']
        for r in results['categories']:
            out[('categories', r['component'])] = r['rho']
        return out
    a, b = values(saved), values(things)
    if set(a) != set(b):
        raise SystemExit('the recomputed THINGS results and data/provided/things_results.json hold different '
                         'correlations: %s' % sorted(set(a) ^ set(b)))
    return max(abs(a[key] - b[key]) for key in a), len(a)


def main(mode=None):
    """mode 'regenerate': THINGS from the trial table of M1; 'provided': THINGS from
    data/provided/things_results.json. Without a mode, release_paths.regeneration_mode chooses."""
    if mode is None:
        mode = release_paths.regeneration_mode([lc.TRIAL_INPUT])
    data, cs = lc.all_datasets(include_things=(mode == 'regenerate'))
    prof, s2 = [], []
    for name, d in data.items():
        prof += profiles(name, d)
        if name in ('THINGS', 'maintained', 'changed'):
            s2 += table_s2(name, d, cs['features'])
    changes = concept_changes(data, len(cs['table']))
    spread = reduced_set_spread(cs)
    if mode == 'regenerate':
        things = dict(profiles=[r for r in prof if r['dataset'] == 'THINGS'],
                      table_s2=[r for r in s2 if r['dataset'] == 'THINGS'],
                      categories=things_categories(data['THINGS']),
                      published_check=published_check(data['THINGS'], cs))
        os.makedirs(os.path.dirname(REGENERATED), exist_ok=True)
        with open(REGENERATED, 'w', encoding='utf-8') as f:
            json.dump(things, f, indent=1)
        difference, count = largest_difference(load_provided(), things)
        print('THINGS: recomputed from the trial table; largest difference from data/provided/things_results.json '
              'over %d correlations: %.2g' % (count, difference))
    else:
        things = load_provided()
        print('THINGS: the saved correlations in data/provided/things_results.json')
        prof += things['profiles']
        s2 += things['table_s2']
    pd.DataFrame(prof).to_csv(os.path.join(OUT, 'locality_profiles.csv'), index=False)
    pd.DataFrame(s2).to_csv(os.path.join(OUT, 'table_s2.csv'), index=False)
    pd.DataFrame(changes).to_csv(os.path.join(OUT, 'concept_changes.csv'), index=False)
    pd.DataFrame(spread).to_csv(os.path.join(OUT, 'reduced_set_spread.csv'), index=False)
    pd.DataFrame(things['categories']).to_csv(os.path.join(OUT, 'things_categories.csv'), index=False)
    write_report(pd.DataFrame(prof), pd.DataFrame(s2), changes, spread, things, mode)
    return mode


def write_report(prof, s2, changes, spread, things, mode):
    source = 'recomputed from the trial table' if mode == 'regenerate' else 'data/provided/things_results.json'
    L = ['SI Appendix, Section 4, One Measure of Distinctiveness Across Analyses (THINGS: %s)' % source, '',
         '1. Spearman rho of crowding with each memory measure']
    for (ds, scope), g in prof.groupby(['dataset', 'scope'], sort=False):
        L.append('  %s, %s (n = %d)' % (ds, scope, g.n.iloc[0]))
        L.append('    tau        hit      FA      CR      d\'')
        for tau, t in g.groupby('tau', sort=False):
            v = t.set_index('measure').rho
            L.append('    %-6s %7.2f %7.2f %7.2f %7.2f' % (tau, v['hit'], v['fa'], v['cr'], v['dprime']))
    L += ['', '2. SI Appendix, Table S2: the Fig. 5c statistic by scale and memory measure']
    for (ds, m), g in s2.groupby(['dataset', 'measure'], sort=False):
        L.append('  %-12s %-7s ' % (ds, m) + '  '.join('%s %.2f' % (t, r) for t, r in zip(g.tau, g.r)))
    L += ['', '3. The same concept in two sets: change in local crowding (tau = 0.05) against change in rates']
    for r in changes:
        L.append('  %-24s %-4s rho %+.2f (p = %.2g, n = %d)' % (r['sets'], r['measure'], r['rho'], r['p'], r['n']))
    L += ['', '4. THINGS, local crowding (tau = 0.05) against concept hit rate']
    for r in things['categories']:
        p_text = '' if r['p'] is None else 'p = %.2g, ' % r['p']
        L.append('  %-20s rho %+.2f (%sn = %d)' % (r['component'], r['rho'], p_text, r['n']))
    L.append('  THINGS concept hit rate minus false alarm rate against the published concept scores: rho = %.3f'
             % things['published_check'])
    L += ['', '5. SD of crowding across concepts, reduced set / maintained set']
    for r in spread:
        L.append('  tau %-6s %.2f' % (r['tau'], r['ratio_reduced_to_maintained']))
    text = '\n'.join(L) + '\n'
    print(text)
    with open(os.path.join(OUT, 'locality_results.txt'), 'w', encoding='utf-8') as f:
        f.write(text)


if __name__ == '__main__':
    main()
