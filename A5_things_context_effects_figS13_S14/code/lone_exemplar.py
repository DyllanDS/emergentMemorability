"""
lone_exemplar.py: first showings with and without an item of the probe's higher-level category already
shown; step 8 of the SI Appendix, Section 5 chain (the 7.8 percent).

WHAT THE PAPER USES
    "first showings: 3,586,410 trials, 7.8% of them lone" (ALL CONCEPTS block): the share of first
    showings after at least 20 preceding items whose same-category count (category_crowding.npz, from
    crowding.py) is zero.

INPUT   external/kramer_trials/derived/trials_with_context.npz and category_crowding.npz
        shared/things/spose_embedding_49d_sorted.txt
OUTPUT  outputs/regenerated/lone_exemplar.txt
SEEDS   none.  RUNTIME  under a minute.
Run from the repository root:  python A5_things_context_effects_figS13_S14/code/lone_exemplar.py

Concepts are numbered as in the Kramer et al. files and that number, minus one, is the row index used
in the SPoSE file, as in compute_context.py.

WHAT ELSE THE REPORT LISTS
    For every first showing of an image, either an item of the probe's category has already appeared
    in that participant's stream or none has ("lone"). The report compares the two kinds of trial in
    two ways, because a lone trial is more likely early in the stream and more likely for some images
    than others:

      1. Direct standardization. Trials are cross-classified by image and by decile of serial
         position; the lone and not-lone false alarm rates are then both re-weighted to the same
         image-by-position distribution, so the two rates are compared over identical images at
         identical positions.
      2. Per-participant regression, as in crowding.py: within-image-centered lone indicator plus
         serial position (and lag for hits), one coefficient per participant, inference across
         participants.

    Both are given for false alarms on first showings, for hits on target repeats, and for the d'
    that results, in all concepts and in the typical and atypical halves of the concepts (above and
    below the median of each concept's mean similarity to its ten nearest SPoSE neighbors).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_1samp

import a5_common

DERIVED = a5_common.DERIVED
OUT = a5_common.make_regenerated_folder()

MIN_PRIOR = 20          # matches the main trial-level analysis
MIN_TRIALS = 30
N_POS_BINS = 10
TOPK = 10
SPOSE = a5_common.spose_file()


# ---- 1. concept typicality: crowding in the full 1,854-concept space ----
def concept_typicality():
    E = np.loadtxt(SPOSE)
    R = np.corrcoef(E).astype(np.float32)
    np.fill_diagonal(R, -np.inf)
    return np.sort(R, axis=1)[:, -TOPK:].mean(1).astype(np.float32)


# ---- 2. standardized rates ----
def standardised_rates(y, lone, img, posbin):
    """Rates for lone and not-lone trials, both re-weighted to the joint
    image-by-position distribution of the cells that contain both kinds."""
    key = img.astype(np.int64) * N_POS_BINS + posbin
    df = pd.DataFrame({"k": key, "lone": lone, "y": y})
    g = df.groupby(["k", "lone"])["y"].agg(["sum", "count"]).unstack("lone")
    g = g.dropna()                       # cells with both kinds of trial
    n0, n1 = g[("count", 0)], g[("count", 1)]
    r0, r1 = g[("sum", 0)] / n0, g[("sum", 1)] / n1
    w = n0 + n1                          # weight each cell by its total trials
    rate0 = float((r0 * w).sum() / w.sum())
    rate1 = float((r1 * w).sum() / w.sum())
    return rate0, rate1, int(n0.sum()), int(n1.sum()), len(g)


# ---- 3. per-participant regression (closed form, as in crowding.py) ----
def group_center(x, g):
    ok = np.isfinite(x)
    ng = int(g.max()) + 1
    n = np.bincount(g[ok], minlength=ng)
    s = np.bincount(g[ok], weights=x[ok], minlength=ng)
    m = np.divide(s, n, out=np.full(ng, np.nan), where=n > 0)
    out = x - m[g]
    out[n[g] < 2] = np.nan
    return out


def fit(y, preds, sub, names, min_trials=MIN_TRIALS):
    ok = np.isfinite(y)
    for p in preds:
        ok &= np.isfinite(p)
    y = y[ok]
    P = [p[ok] for p in preds]
    s = sub[ok]
    ns = int(s.max()) + 1
    n = np.bincount(s, minlength=ns).astype(float)
    nn = np.maximum(n, 1)

    def cen(a, b):
        sa, sb, sab = (np.bincount(s, weights=v, minlength=ns)
                       for v in (a, b, a * b))
        return sab - sa * sb / nn

    k = len(P)
    A = np.empty((ns, k, k))
    rhs = np.empty((ns, k))
    for i in range(k):
        rhs[:, i] = cen(P[i], y)
        for j in range(k):
            A[:, i, j] = cen(P[i], P[j])
    A += np.eye(k) * 1e-10
    det = np.linalg.det(A)
    good = np.isfinite(det) & (np.abs(det) > 1e-16) & (n >= min_trials)
    B = np.full((ns, k), np.nan)
    if good.any():
        B[good] = np.linalg.solve(A[good], rhs[good][:, :, None])[:, :, 0]
    return {nm: B[:, i] for i, nm in enumerate(names)}


def coef_line(b, label):
    b = b[np.isfinite(b)]
    t, p = ttest_1samp(b, 0)
    ci = 1.96 * b.std(ddof=1) / np.sqrt(len(b))
    return (f"    {label:34s} b={b.mean():+.5f} [{b.mean()-ci:+.5f}, {b.mean()+ci:+.5f}]"
            f"  t({len(b)-1})={t:+.2f}  p={p:.3g}  d={b.mean()/b.std(ddof=1):+.3f}")


def dprime(h, f):
    return norm.ppf(h) - norm.ppf(f)


def main():
    typ_of_concept = concept_typicality()

    base = np.load(DERIVED / "trials_with_context.npz")
    cc = np.load(DERIVED / "category_crowding.npz")
    cat_crowd = cc["cat_crowd"].astype(np.int32)
    sub = base["sub"].astype(np.int64)
    img = base["image_nr"].astype(np.int64)
    conc = base["concept"].astype(np.int32) - 1
    ttype, perf, lag = base["ttype"], base["perf"], base["lag"]
    trial = base["trial"].astype(float)
    n_prior = base["n_prior"].astype(int)
    feat = base["ctx_top10"].astype(float)

    lone_all = (cat_crowd == 0)
    typ = typ_of_concept[conc]
    typical = typ >= np.median(typ_of_concept)   # crowded half of the space

    enough = n_prior >= MIN_PRIOR
    fa_mask = np.isin(ttype, [1, 3]) & enough
    hit_mask = (ttype == 2) & enough & (lag > 0)
    y_fa = (perf == 13).astype(float)
    y_hit = (perf == 11).astype(float)

    # position deciles over the trials that enter the false alarm analysis
    edges = np.quantile(trial[fa_mask], np.linspace(0, 1, N_POS_BINS + 1)[1:-1])
    posbin = np.searchsorted(edges, trial).astype(np.int64)

    fh = open(OUT / "lone_exemplar.txt", "w", encoding="utf-8")

    def w(s=""):
        print(s, flush=True)
        fh.write(s + "\n")

    w("THE LONE-EXEMPLAR CONTRAST, WITHIN IMAGE AND AT MATCHED SERIAL POSITION")
    w("=" * 100)
    w(f"{len(sub):,} trials, {len(np.unique(sub)):,} participants; trials require at "
      f"least {MIN_PRIOR} preceding items.")
    w("A trial is 'lone' when no item of the probe's THINGS higher-level category has yet")
    w("appeared in that participant's stream. Rates are standardized to a common")
    w(f"image-by-position distribution ({N_POS_BINS} position deciles), so the two kinds of")
    w("trial are compared over the same images at the same positions. Regressions are")
    w("per participant, with the lone indicator centered within image and serial position")
    w("(and lag, for hits) entered alongside it.")
    w("")

    def block(title, sel):
        w("-" * 100)
        w(title)
        w("-" * 100)
        m_fa = fa_mask & sel
        m_hit = hit_mask & sel
        share = lone_all[m_fa].mean()
        w(f"  first showings: {m_fa.sum():,} trials, {share:.1%} of them lone")
        r0, r1, n0, n1, ncell = standardised_rates(
            y_fa[m_fa], lone_all[m_fa].astype(int), img[m_fa], posbin[m_fa])
        w(f"  false alarm rate, standardized:   category-mate preceded {r0:.4f}  "
          f"(n={n0:,})   lone {r1:.4f}  (n={n1:,})   over {ncell:,} image-by-position cells")
        raw0 = y_fa[m_fa & ~lone_all].mean()
        raw1 = y_fa[m_fa & lone_all].mean()
        w(f"  false alarm rate, raw:            category-mate preceded {raw0:.4f}   lone {raw1:.4f}")
        h0, h1, hn0, hn1, hcell = standardised_rates(
            y_hit[m_hit], lone_all[m_hit].astype(int), img[m_hit], posbin[m_hit])
        w(f"  hit rate, standardized:           category-mate preceded {h0:.4f}  "
          f"(n={hn0:,})   lone {h1:.4f}  (n={hn1:,})   over {hcell:,} cells")
        w(f"  d' from standardized rates:       category-mate preceded {dprime(h0, r0):.3f}   "
          f"lone {dprime(h1, r1):.3f}   (difference {dprime(h1, r1) - dprime(h0, r0):+.3f})")
        w("")

        # per-participant regressions
        for name, mask, y, extra, exn in [
                ("false alarms on first showings", m_fa, y_fa, [], []),
                ("hits on target repeats", m_hit, y_hit,
                 [np.where(m_hit, lag.astype(float), np.nan)], ["lag"])]:
            yy = np.where(mask, y, np.nan)
            pos = np.where(mask, trial, np.nan)
            L = np.where(mask, lone_all.astype(float), np.nan)
            Lc = group_center(L, img)
            F = np.where(mask, feat, np.nan)
            F = (F - np.nanmean(F)) / np.nanstd(F)
            Fc = group_center(F, img)
            w(f"  {name}: per-participant regression")
            c1 = fit(yy, [Lc, pos] + extra, sub, ["lone", "position"] + exn)
            w(coef_line(c1["lone"], "lone (within image, + position)"))
            c2 = fit(yy, [Lc, Fc, pos] + extra, sub,
                     ["lone", "feature", "position"] + exn)
            w(coef_line(c2["lone"], "lone, with feature crowding entered"))
            w(coef_line(c2["feature"], "feature crowding (z), same model"))
            w("")

    block("ALL CONCEPTS", np.ones(len(sub), bool))
    block("TYPICAL CONCEPTS (crowded half of the SPoSE space: nearest neighbors close)",
          typical)
    block("ATYPICAL CONCEPTS (isolated half of the SPoSE space)", ~typical)

    w("-" * 100)
    w("READING THIS")
    w("-" * 100)
    w("  The standardized rows compare lone and not-lone trials over the same images at")
    w("  the same serial positions (the image-by-position cells that hold both kinds of")
    w("  trial). The regression rows give the same contrast in rate units per")
    w("  participant, with inference across participants; the second model adds feature")
    w("  crowding (the top-10 SPoSE measure, z-scored and centered within image) to the")
    w("  lone indicator. The TYPICAL and ATYPICAL blocks repeat the analysis in the")
    w("  halves of the concepts at or above and below the median of a concept's mean")
    w("  SPoSE correlation with its ten most similar concepts (over all 1,854")
    w("  concepts).")
    fh.close()
    print(f"\nwrote {a5_common.relative(OUT / 'lone_exemplar.txt')}")


if __name__ == "__main__":
    main()
