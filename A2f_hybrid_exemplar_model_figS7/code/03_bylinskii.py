"""The Bylinskii context simulation with a GCM and a hybrid-similarity GCM, swept like the REM version.

Follows the REM version of SI Appendix, Fig. S3 (module A2b): categories of items sharing a set of
prototype feature values (reference: 21 categories x 300 items, 10 category-defining features,
g = .40, w = 49, u = .30, c = .70), 75 targets per category. In the within-category context,
lists of 30 targets and 90 fillers come from one category; in the across-category context, the
same targets appear in lists mixed over all categories. The quantity of interest is the
Spearman correlation, across the same targets, between memorability (hit rate minus false alarm rate)
in the two contexts, reported with the split-half reliability of each context's memorability.

A feature value counts as distinctive when its frequency on that dimension is below q in the
reference population: the world (the generating distribution), the whole categorized item
set, the participant's own study list, or an even list/world mix. Under the list reference, a
category's prototype values are common on every within-category list and so are not treated as
distinctive there.

Criteria: the REM settings of SI Appendix, Fig. S3 vary the two contexts' likelihood-ratio
criteria; the GCM has no natural criterion scale, so here each context's criterion is set to a
target false alarm rate (reference .37 within, .05 across, close to REM's rates), and
equal-criterion cells use one absolute criterion in both contexts. The similarity width is
calibrated once per model at the reference cell, to the across-category d' closest to REM's
(2.6), and held fixed relative to the median inter-item distance in every other cell. With this
design's encoding noise no candidate width brings the GCM-family models to 2.6: across-category d'
peaks near 1.6 for the hybrids and 1.1 for the plain GCM, so the calibrated width is the one that
comes closest.

Writes outputs/bylinskii.csv and outputs/bylinskii.txt.

The paper reports the [list] rows (rarity from the participant's own study list). The width
calibration runs before the worker pool starts, and every cell has its own seed, zlib.crc32(cell
label), so the number of worker processes does not change any result, and renaming a cell changes
its numbers.
Run from the repository root:  python A2f_hybrid_exemplar_model_figS7/code/03_bylinskii.py  (about 17 min)
"""
import sys
sys.dont_write_bytecode = True      # keep code/ free of __pycache__
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import hgcm as h

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"       # the module's outputs/ folder
REF = dict(g=0.40, w=49, u=0.30, cenc=0.70, n_cat=21, per_cat=300, n_targets=75, cat_feat=10,
           n_study_t=30, n_study_f=90, runs_within=300, runs_across=6000,
           fa_within=0.37, fa_across=0.05, equal=None, gamma=2.0, delta=0.0, q=0.10, ref="set")
REFS = ("world", "set", "list", "mix")


# ---- 1. categorized items ----
def build(rng, p):
    rows = []
    for _ in range(p["n_cat"]):
        proto = rng.geometric(p["g"], p["cat_feat"])
        rest = rng.geometric(p["g"], (p["per_cat"], p["w"] - p["cat_feat"]))
        rows.append(np.hstack([np.tile(proto, (p["per_cat"], 1)), rest]))
    X = np.vstack(rows).astype(np.int16)
    cat = np.repeat(np.arange(p["n_cat"]), p["per_cat"])
    is_t = np.zeros(len(X), bool)
    for c in range(p["n_cat"]):
        is_t[c * p["per_cat"]: c * p["per_cat"] + p["n_targets"]] = True
    return X, cat, is_t


def median_distance(rng, X, n=400):
    a = X[rng.choice(len(X), n)].astype(float)
    b = X[rng.choice(len(X), n)].astype(float)
    return float(np.median(np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1))))


# ---- 2. one context ----
def run_context(rng, X, tpool, fpool, n_runs, p, batch=50):
    out_t, out_f = [], []
    for b in range(0, n_runs, batch):
        R = min(batch, n_runs - b)
        st_t = np.array([rng.choice(tpool, p["n_study_t"], replace=False) for _ in range(R)])
        st_f = np.array([rng.choice(fpool, p["n_study_f"], replace=False) for _ in range(R)])
        # As in the REM version, a small category tests fewer foils than targets.
        n_foil = min(p["n_study_t"], len(tpool) - p["n_study_t"])
        foils = np.array([rng.choice(np.setdiff1d(tpool, s), n_foil, replace=False) for s in st_t])
        studied = X[np.hstack([st_t, st_f])]
        tr, msk = h.encode(rng, studied, p["u"], p["cenc"], p["g"])
        tab = h.ref_table(p, studied)
        out_t.append((st_t, h.familiarity(X[st_t], tr, msk, p, tab)))
        out_f.append((foils, h.familiarity(X[foils], tr, msk, p, tab)))
    return out_t, out_f


def counts(out_t, out_f, crit, n):
    hc, hn, fc, fnn = (np.zeros(n) for _ in range(4))
    for idx, fam in out_t:
        np.add.at(hc, idx.ravel(), (fam > crit).ravel()); np.add.at(hn, idx.ravel(), 1)
    for idx, fam in out_f:
        np.add.at(fc, idx.ravel(), (fam > crit).ravel()); np.add.at(fnn, idx.ravel(), 1)
    return hc, hn, fc, fnn


def mem(cnt, ok):
    hc, hn, fc, fnn = cnt
    return hc[ok] / hn[ok] - fc[ok] / fnn[ok]


def contexts(rng, X, cat, is_t, p):
    W_t, W_f = [], []
    for c in range(p["n_cat"]):
        rows = np.flatnonzero(cat == c)
        a, b = run_context(rng, X, rows[:p["n_targets"]], rows[p["n_targets"]:], p["runs_within"], p)
        W_t += a; W_f += b
    A_t, A_f = run_context(rng, X, np.flatnonzero(is_t), np.flatnonzero(~is_t), p["runs_across"], p)
    return W_t, W_f, A_t, A_f


# ---- 2b. width calibration at the reference cell ----
def calibrate(rng, X, is_t, p, md):
    best = None
    for frac in (0.03, 0.05, 0.08, 0.12, 0.18, 0.28, 0.42, 0.65):
        q = dict(p, c=1.0 / (frac * md))
        ct, cf = run_context(rng, X, np.flatnonzero(is_t), np.flatnonzero(~is_t), 200, q)
        ft = np.concatenate([f.ravel() for _, f in ct]); ff = np.concatenate([f.ravel() for _, f in cf])
        k = np.quantile(ff, .95)
        d = h.zc((ft > k).mean()) - h.zc((ff > k).mean())
        if best is None or abs(d - 2.6) < best[0]:
            best = (abs(d - 2.6), frac, d)
    return best[1], best[2]


# ---- 3. one cell ----
def run_cell(arg):
    label, change, frac = arg
    rng = np.random.default_rng(zlib.crc32(label.encode()))
    p = dict(REF, **change)
    X, cat, is_t = build(rng, p)
    p["set_freq"] = h.set_freq(X)
    p["c"] = 1.0 / (frac[(p["gamma"] > 0, p["ref"])] * median_distance(rng, X))
    W_t, W_f, A_t, A_f = contexts(rng, X, cat, is_t, p)
    famWf = np.concatenate([f.ravel() for _, f in W_f]); famAf = np.concatenate([f.ravel() for _, f in A_f])
    if p["equal"] is None:
        cw, ca = np.quantile(famWf, 1 - p["fa_within"]), np.quantile(famAf, 1 - p["fa_across"])
    else:
        cw = ca = np.quantile(famAf, 1 - p["equal"])
    n = len(X)
    w, a = counts(W_t, W_f, cw, n), counts(A_t, A_f, ca, n)
    ok = is_t & (w[1] > 0) & (w[3] > 0) & (a[1] > 0) & (a[3] > 0)
    rho = float(spearmanr(mem(w, ok), mem(a, ok))[0])

    # Split-half reliability of each context's item memorability (alternate batches),
    # Spearman-Brown corrected, as a ceiling on the cross-context correlation.
    def rel(ot, of, crit):
        hv = [counts(ot[k::2], of[k::2], crit, n) for k in (0, 1)]
        ok2 = ok & (hv[0][1] > 0) & (hv[0][3] > 0) & (hv[1][1] > 0) & (hv[1][3] > 0)
        r = spearmanr(mem(hv[0], ok2), mem(hv[1], ok2))[0]
        return float(2 * r / (1 + r))
    relW, relA = rel(W_t, W_f, cw), rel(A_t, A_f, ca)

    def rates(c):
        return (c[0][ok] / c[1][ok]).mean(), (c[2][ok] / c[3][ok]).mean()
    (hW, fW), (hA, fA) = rates(w), rates(a)
    return dict(cell=label, model="hybrid" if p["gamma"] > 0 else "GCM", ref=p["ref"], gamma=p["gamma"], q=p["q"],
                cat_feat=p["cat_feat"], n_cat=p["n_cat"], per_cat=p["per_cat"],
                criteria="equal FA %.2f" % p["equal"] if p["equal"] is not None else "FA %.2f / %.2f" % (p["fa_within"], p["fa_across"]),
                rho=rho, rel_within=relW, rel_across=relA, ceiling=float(np.sqrt(max(relW, 0) * max(relA, 0))),
                hit_within=hW, fa_within=fW, hit_across=hA, fa_across=fA,
                dprime_within=float(h.zc(hW) - h.zc(fW)), dprime_across=float(h.zc(hA) - h.zc(fA)))


# ---- 4. the cells ----
def cells():
    out = [("reference", {})]
    out += [(f"criteria FA {a} / {b}", dict(fa_within=a, fa_across=b)) for a, b in ((0.25, 0.05), (0.37, 0.10), (0.50, 0.05))]
    out += [(f"equal criterion FA {e}", dict(equal=e)) for e in (0.05, 0.15, 0.30)]
    out += [(f"category-defining features = {k}", dict(cat_feat=k)) for k in (3, 5, 15)]
    out += [(f"categories = {k}", dict(n_cat=k)) for k in (10, 40)]
    out += [(f"items per category = {k}", dict(per_cat=k, n_targets=k // 4)) for k in (150, 600)]
    out += [(f"rare-feature boost gamma = {g}", dict(gamma=g)) for g in (1.0, 3.0)]
    out += [(f"distinctive if set frequency < {q}", dict(q=q)) for q in (0.05, 0.20)]
    out += [("mismatch penalty delta = 0.5", dict(delta=0.5))]
    gcm = [("GCM: " + lab, dict(ch, gamma=0.0)) for lab, ch in out[:7]]
    hyb = [(lab if r == "set" else f"{lab} [{r}]", dict(ch, ref=r)) for r in REFS for lab, ch in out]
    return hyb + gcm


# ---- 5. calibrate each model once, run every cell, summarize ----
if __name__ == "__main__":
    rng = np.random.default_rng(11)
    X, cat, is_t = build(rng, REF)
    md = median_distance(rng, X)
    frac, dprime = {}, {}
    p0 = dict(REF, set_freq=h.set_freq(X))
    frac[(False, "set")], dprime["GCM"] = calibrate(rng, X, is_t, dict(p0, gamma=0.0), md)
    for r in REFS:
        frac[(True, r)], dprime[r] = calibrate(rng, X, is_t, dict(p0, ref=r), md)
    OUT.mkdir(exist_ok=True)
    with Pool(2) as pool:           # 2 worker processes; every cell has its own seed
        rows = pool.map(run_cell, [(lab, ch, frac) for lab, ch in cells()])
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bylinskii.csv", index=False)
    lines = ["BYLINSKII CONTEXT SIMULATION, GCM AND HYBRID-SIMILARITY GCM, SWEPT LIKE THE REM VERSION", "=" * 120,
             "rho: Spearman correlation across the same targets between memorability within category and across",
             "categories (Bylinskii et al. report .60 for hit rates; REM gives .24 to .76 in SI Appendix, Fig. S3).",
             "ceiling: sqrt of the two contexts' split-half reliabilities.",
             "Width calibration at the reference cell (across-category d', target 2.6, which no width reaches): " + ", ".join(f"{k} {v:.2f}" for k, v in dprime.items()), "",
             f"{'cell':52s} {'criteria':>15s} {'rho':>7s} {'ceiling':>8s} {'d W':>6s} {'d A':>6s}", "-" * 120]
    for r in rows:
        lines.append(f"{r['cell']:52s} {r['criteria']:>15s} {r['rho']:+7.3f} {r['ceiling']:8.2f} {r['dprime_within']:6.2f} {r['dprime_across']:6.2f}")
    hy, gc = df[df.model == "hybrid"], df[df.model == "GCM"]
    lines.append("")
    for r in REFS:
        d = hy[hy.ref == r]
        lines.append(f"[{r}] hybrid cells (n = {len(d)}): rho {d.rho.min():+.3f} to {d.rho.max():+.3f}, median {d.rho.median():+.3f}; positive in {int((d.rho > 0).sum())}; median ceiling {d.ceiling.median():.2f}.")
    lines += [
              f"GCM cells (n = {len(gc)}): rho {gc.rho.min():+.3f} to {gc.rho.max():+.3f}, median {gc.rho.median():+.3f}."]
    (OUT / "bylinskii.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
