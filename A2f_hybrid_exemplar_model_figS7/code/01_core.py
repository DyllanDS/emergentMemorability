"""The basic simulation of Fig. 1 and the designed lists of SI Appendix, Fig. S5, with a GCM and a
hybrid-similarity GCM in place of REM (SI Appendix, Section 2, A Hybrid-Similarity Exemplar Model).
Writes outputs/core.txt, outputs/core.json and the per-item tables outputs/core_items_basic.csv and
outputs/core_items_designed.csv.

A  per-item memorability (hit rate minus false alarm rate) against crowding across 10,000
   simulated experiments, plus the feature-level relationship of Fig. 1d
D  lists built to place 0 to 16 close neighbors beside the same critical item, tested against
   its nearest neighbor as the foil

The paper's hybrid model is the variant "Hybrid GCM, list" (rarity from the participant's own study
list) and its GCM is the variant "GCM". The five variants run on one item set and draw, in the order
of VARIANTS, from one random stream (numpy.random.default_rng with seed 1), so the script always runs
whole: the list variant's numbers depend on the three variants before it. analysis_a() and
analysis_d() fill the optional dictionary 'items' with the per-item counts and rates they compute
(no random draw), and the per-item tables written from them are the input of the item bootstrap in
stats_figS7.py.
Run from the repository root:  python A2f_hybrid_exemplar_model_figS7/code/01_core.py  (about 3 min)
"""
import sys
sys.dont_write_bytecode = True      # keep code/ free of __pycache__
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

import hgcm as h

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"       # the module's outputs/ folder
OUT.mkdir(exist_ok=True)

BASE = dict(g=0.65, w=49, u=0.30, cenc=0.60, n_full=26107, n_db=1600, n_study=40, n_old=20, n_new=20,
            target_fa=0.15, target_d=1.61)
VARIANTS = {"GCM": dict(gamma=0.0, delta=0.0, q=0.10, ref="set"),
            "Hybrid GCM, world": dict(gamma=2.0, delta=0.0, q=0.10, ref="world"),
            "Hybrid GCM, set": dict(gamma=2.0, delta=0.0, q=0.10, ref="set"),
            "Hybrid GCM, list": dict(gamma=2.0, delta=0.0, q=0.10, ref="list"),
            "Hybrid GCM, mix": dict(gamma=2.0, delta=0.0, q=0.10, ref="mix")}


# ---- 1. one batch of the basic design: R experiments at once ----
def basic_batch(rng, X, p, R):
    n = X.shape[0]
    perm = np.argsort(rng.random((R, n)), axis=1)[:, :p["n_study"] + p["n_new"]]
    st, nw = perm[:, :p["n_study"]], perm[:, p["n_study"]:]
    old = st[:, :p["n_old"]]
    tr, msk = h.encode(rng, X[st], p["u"], p["cenc"], p["g"])
    tab = h.ref_table(p, X[st])
    return old, nw, h.familiarity(X[old], tr, msk, p, tab), h.familiarity(X[nw], tr, msk, p, tab)


# ---- 2. width calibration to REM's d' ----
def calibrate(rng, X, p, md):
    """Choose the generalization width whose d' comes closest to REM's (1.61), with the criterion at FA = .15."""
    best = None
    for frac in (0.03, 0.05, 0.08, 0.12, 0.18, 0.28, 0.42, 0.65, 1.0):
        q = dict(p, c=1.0 / (md * frac))
        _, _, fo, fn = basic_batch(rng, X, q, 600)
        crit = np.quantile(fn, 1 - p["target_fa"])
        d = h.zc((fo > crit).mean()) - h.zc((fn > crit).mean())
        if best is None or abs(d - p["target_d"]) < best[0]:
            best = (abs(d - p["target_d"]), q["c"], crit, d)
    return best[1], best[2], best[3]


# ---- 3. analysis A: per-item memorability against crowding ----
def analysis_a(rng, X, p, crowd, n_sim=10000, batch=500, items=None):
    # Names, per item of the 1,600: hc = times the item was called old when it was studied (hits),
    # hn = times it was studied and tested, fc = times it was called old when it was new (false
    # alarms), fnn = times it was tested as new. Each batch of simulated participants (basic_batch)
    # returns which items were old and new and their familiarities (fo, fn); an item is called old
    # when its familiarity exceeds the criterion p["crit"]; np.add.at adds one count per test to that
    # item. Then memorability = hit rate - false alarm rate per item, and rho = its Spearman
    # correlation with the item's crowding. The per-item counts are also handed back (items, below;
    # written to outputs/core_items_basic.csv), and stats_figS7.py recomputes the rates from them.
    n = X.shape[0]
    hc, hn, fc, fnn = (np.zeros(n) for _ in range(4))
    for _ in range(n_sim // batch):
        old, nw, fo, fn = basic_batch(rng, X, p, batch)
        np.add.at(hc, old.ravel(), (fo > p["crit"]).ravel()); np.add.at(hn, old.ravel(), 1)
        np.add.at(fc, nw.ravel(), (fn > p["crit"]).ravel()); np.add.at(fnn, nw.ravel(), 1)
    ok = (hn > 0) & (fnn > 0)
    if items is not None:
        # hand back the per-item counts (hits, old tests, false alarms, new tests) and the crowding
        # of every item, for the item bootstrap. No random draw happens here.
        items.update(hits=hc.copy(), old_tests=hn.copy(), false_alarms=fc.copy(), new_tests=fnn.copy(),
                     crowding=np.asarray(crowd).copy())
    hit, fa = hc[ok] / hn[ok], fc[ok] / fnn[ok]
    mem = hit - fa
    rho = spearmanr(mem, crowd[ok])[0]
    cm = np.array([spearmanr(X[ok, k], mem)[0] for k in range(X.shape[1])])
    cd = np.array([spearmanr(X[ok, k], crowd[ok])[0] for k in range(X.shape[1])])
    return dict(hit=hit.mean(), fa=fa.mean(), dprime=float(h.zc(hit.mean()) - h.zc(fa.mean())),
                rho_item=float(rho), r_feat=float(pearsonr(cm, cd)[0]),
                rho_hit=float(spearmanr(hit, crowd[ok])[0]), rho_fa=float(spearmanr(fa, crowd[ok])[0]))


# ---- 4. analysis D: designed lists with 0 to 16 close neighbors ----
def analysis_d(rng, X, p, D, n_crit=200, n_subj=200, n_sim=(0, 2, 4, 8, 16), items=None):
    Dn = np.where(np.isnan(D), np.inf, D)
    order = np.argsort(Dn, axis=1)
    crit_items = rng.choice(X.shape[0], n_crit, replace=False)
    H = np.zeros((n_crit, len(n_sim))); F = np.zeros_like(H)
    for i, c in enumerate(crit_items):
        o = order[c][order[c] != c]
        lure, pool = o[0], o[1:]
        far = pool[len(pool) // 2:]
        for k, ns in enumerate(n_sim):
            # Each participant's fillers are distinct items, so no list contains a repeat.
            fill = far[np.argsort(rng.random((n_subj, len(far))), axis=1)[:, :p["n_study"] - 1 - ns]]
            st = np.concatenate([np.full((n_subj, 1), c), np.tile(pool[:ns], (n_subj, 1)), fill], axis=1)
            tr, msk = h.encode(rng, X[st], p["u"], p["cenc"], p["g"])
            tab = h.ref_table(p, X[st])
            fcrit = h.familiarity(X[[c]][None].repeat(n_subj, 0), tr, msk, p, tab)
            flure = h.familiarity(X[[lure]][None].repeat(n_subj, 0), tr, msk, p, tab)
            H[i, k] = (fcrit > p["crit"]).mean(); F[i, k] = (flure > p["crit"]).mean()
    mH, mF = H.mean(0), F.mean(0)
    if items is not None:
        # hand back each critical item's hit and false alarm rate (over the n_subj simulated
        # participants) at every neighbor count, for the item bootstrap. No random draw.
        items.update(critical_items=np.asarray(crit_items).copy(), hit_rate=H.copy(), fa_rate=F.copy(),
                     n_similar=tuple(n_sim), n_subj=n_subj)
    za = lambda q: h.zc(q, 1 / (2 * n_subj))
    return [dict(similar=int(ns), hit=float(mH[k]), fa=float(mF[k]), hits_minus_fa=float(mH[k] - mF[k]),
                 dprime=float(za(mH[k]) - za(mF[k])), criterion=float(-0.5 * (za(mH[k]) + za(mF[k]))))
            for k, ns in enumerate(n_sim)]


# ---- 5. run every variant on one shared item set ----
if __name__ == "__main__":
    item_rows_basic, item_rows_designed = [], []     # per-item tables
    rng = np.random.default_rng(1)
    full = h.make_items(rng, BASE["n_full"], BASE["w"], BASE["g"])
    X = full[rng.choice(BASE["n_full"], BASE["n_db"], replace=False)]
    crowd, D = h.crowding(X)
    md = float(np.nanmedian(D))
    results = {}
    lines = ["THE BASIC SIMULATION AND THE DESIGNED LISTS WITH A GCM AND A HYBRID-SIMILARITY GCM", "=" * 92,
             "Design, encoding and distinctiveness measure follow Fig. 1 (g = .65, w = 49, u = .30,",
             "copy fidelity = .60, 1,600 items, 40 studied, 20 old and 20 new tested). Crowding is mean Euclidean",
             "distance to the rest of the set, negated, so a NEGATIVE memorability-crowding correlation",
             "means isolated items are better remembered (REM at the Fig. 1 values: -0.606).", ""]
    for name, v in VARIANTS.items():
        p = dict(BASE, **v)
        p["set_freq"] = h.set_freq(X)
        p["c"], p["crit"], d0 = calibrate(rng, X, p, md)
        items_a, items_d = {}, {}
        a = analysis_a(rng, X, p, crowd, items=items_a)
        dd = analysis_d(rng, X, p, D, items=items_d)
        results[name] = dict(params={k: p[k] for k in ("c", "crit", "gamma", "delta", "q", "ref")}, A=a, D=dd)
        # one row per item of the 1,600-item set (basic simulation, counts over the 10,000
        # simulated experiments) and one row per critical item and neighbor count (designed
        # lists, counts over the simulated participants)
        for item in range(X.shape[0]):
            item_rows_basic.append(dict(variant=name, item=item,
                                        hits=int(items_a["hits"][item]), old_tests=int(items_a["old_tests"][item]),
                                        false_alarms=int(items_a["false_alarms"][item]),
                                        new_tests=int(items_a["new_tests"][item]),
                                        crowding=float(items_a["crowding"][item])))
        n_subj = items_d["n_subj"]
        for row, item in enumerate(items_d["critical_items"]):
            for column, n_similar in enumerate(items_d["n_similar"]):
                item_rows_designed.append(dict(variant=name, critical_item=int(item), n_similar=int(n_similar),
                                               hits=int(round(items_d["hit_rate"][row, column] * n_subj)),
                                               false_alarms=int(round(items_d["fa_rate"][row, column] * n_subj)),
                                               participants=int(n_subj)))
        lines += ["-" * 92, f"{name}  (c = {p['c']:.4f}; gamma = {p['gamma']}, delta = {p['delta']}, distinctive = frequency in the {p['ref']} < {p['q']})", "-" * 92,
                  f"A. basic simulation, 10,000 runs: hit {a['hit']:.3f}, FA {a['fa']:.3f}, d' {a['dprime']:.2f}",
                  f"   memorability vs crowding        rho = {a['rho_item']:+.3f}",
                  f"   hit rate vs crowding            rho = {a['rho_hit']:+.3f}",
                  f"   false alarm rate vs crowding    rho = {a['rho_fa']:+.3f}",
                  f"   feature level (Fig. 1d)         r   = {a['r_feat']:+.3f}",
                  "D. designed lists, same critical items, nearest neighbor as foil",
                  "   similar     hit      FA   hits-FA   d'    criterion"]
        for r in dd:
            lines.append(f"   {r['similar']:7d} {r['hit']:7.3f} {r['fa']:7.3f} {r['hits_minus_fa']:8.3f} {r['dprime']:6.3f} {r['criterion']:8.3f}")
        lines += [f"   0 -> 16: hit {dd[-1]['hit'] - dd[0]['hit']:+.3f}, FA {dd[-1]['fa'] - dd[0]['fa']:+.3f}, "
                  f"hits-FA {dd[-1]['hits_minus_fa'] - dd[0]['hits_minus_fa']:+.3f}, d' {dd[-1]['dprime'] - dd[0]['dprime']:+.3f}", ""]
        print("\n".join(lines[-14:]))
    (OUT / "core.txt").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "core.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    # the per-item tables (input of stats_figS7.py)
    pd.DataFrame(item_rows_basic).to_csv(OUT / "core_items_basic.csv", index=False)
    pd.DataFrame(item_rows_designed).to_csv(OUT / "core_items_designed.csv", index=False)
