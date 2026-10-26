"""
crowding.py: feature crowding against same-category count as predictors of false alarms, with the
image held constant; step 7 of the SI Appendix, Section 5 chain (the 4.6 ratio).

WHAT IT REPORTS (crowding.txt)
    Per-participant multiple regressions of the response on two measures of the preceding stream,
    each z-scored and centered within image: feature crowding (the top-10 SPoSE measure of
    context_k.npz) and the number of preceding items of the probe's higher-level category; serial
    position (and lag, for hits) entered beside them; trials with at least 50 preceding items;
    participants with at least 30 such trials. The coefficients are averaged over participants and
    tested against zero (t over participants). M3, both measures together, gives the 4.6 ratio for
    false alarms (+0.00286 / +0.00062). M1 and M2 enter each measure alone, and M4 adds database
    typicality and its product with the same-category count; the report gives all four models for
    false alarms and for hits.

    Same-category count of a trial: the number of earlier trials in the session that showed another
    image whose concept has the probe concept's category code in the category table (code 0 marks
    the concepts without a named higher-level category). Typicality, used for the interaction in M4,
    is a concept's own crowding in the full 1,854-concept space: the mean of its ten highest SPoSE
    correlations with the other concepts, irrespective of category.

    Concepts are numbered as in the Kramer et al. files and that number, minus one, is the row index
    used in the SPoSE file and the category table, as in compute_context.py.

INPUT   external/kramer_trials/derived/trials.npz, trials_with_context.npz, context_k.npz
        shared/things/spose_embedding_49d_sorted.txt
        shared/things/concept_categories.csv (via a5_common.concept_categories_alphabetical(): the
        higher-level category of each concept in the concept numbering of the Kramer et al. files)
OUTPUT  external/kramer_trials/derived/category_crowding.npz   (per trial: same-category count and the
                                                               probe's category code; not distributed)
        outputs/regenerated/crowding.txt
SEEDS   none.  RUNTIME  under a minute.
Run from the repository root:  python A5_things_context_effects_figS13_S14/code/crowding.py
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_1samp

import a5_common

DERIVED = a5_common.DERIVED
OUT = a5_common.make_regenerated_folder()

MIN_PRIOR = 50
MIN_TRIALS = 30
TOPK = 10
SPOSE = a5_common.spose_file()


# ---- 1. concept-level structure: category membership and database typicality ----
def concept_structure():
    E = np.loadtxt(SPOSE)
    R = np.corrcoef(E).astype(np.float32)
    np.fill_diagonal(R, -np.inf)
    typ = np.sort(R, axis=1)[:, -TOPK:].mean(1)      # crowding in the full space
    cc = a5_common.concept_categories_alphabetical()    # the public category file, by concept number
    cat = cc["category"].to_numpy()
    # every concept has a code in the category table (0 = no named higher-level category); a concept
    # without a finite code would get its own singleton id below
    bad = ~np.isfinite(cat)
    cat = np.where(bad, -1, cat).astype(np.int32)
    lone = np.flatnonzero(cat == -1)
    cat[lone] = -(np.arange(len(lone)) + 2)
    return cat, typ.astype(np.float32), int((~bad).sum())


# ---- 2. per-trial category crowding ----
def category_crowding(cat_of_concept):
    d = np.load(DERIVED / "trials.npz")
    sub, image, concept = d["sub"], d["image_nr"], d["concept"]
    cat = cat_of_concept[concept.astype(np.int32) - 1]
    n = len(sub)
    out = np.zeros(n, dtype=np.int16)
    bounds = np.searchsorted(sub, np.arange(sub[-1] + 2))
    t0 = time.time()
    for s in range(len(bounds) - 1):
        a, b = bounds[s], bounds[s + 1]
        if b - a < 2:
            continue
        c, img = cat[a:b], image[a:b]
        same = (c[:, None] == c[None, :]) & (img[:, None] != img[None, :])
        out[a:b] = np.tril(same, -1).sum(1)
        if s % 5000 == 0:
            print(f"  subject {s} ({time.time()-t0:.0f}s)", flush=True)
    return out, cat


# ---- 3. per-participant regression, closed form ----
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
    """Per-participant multiple regression; returns one coefficient set each."""
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


def report(w, label, coefs, keys):
    w(f"  {label}")
    for k in keys:
        b = coefs[k][np.isfinite(coefs[k])]
        if len(b) < 2:
            w(f"    {k:26s} (insufficient)")
            continue
        t, p = ttest_1samp(b, 0)
        ci = 1.96 * b.std(ddof=1) / np.sqrt(len(b))
        w(f"    {k:26s} b={b.mean():+.5f} [{b.mean()-ci:+.5f}, {b.mean()+ci:+.5f}]  "
          f"t({len(b)-1})={t:+.2f}  p={p:.3g}  d={b.mean()/b.std(ddof=1):+.3f}")
    w("")


def main():
    cat_of_concept, typ_of_concept, n_categorized = concept_structure()
    print(f"{n_categorized} of 1854 concepts have a category code", flush=True)

    ccfile = DERIVED / "category_crowding.npz"
    if ccfile.exists():
        z = np.load(ccfile)
        cat_crowd, cat_tr = z["cat_crowd"], z["cat_tr"]
    else:
        cat_crowd, cat_tr = category_crowding(cat_of_concept)
        np.savez_compressed(ccfile, cat_crowd=cat_crowd, cat_tr=cat_tr)

    base = np.load(DERIVED / "trials_with_context.npz")
    ck = np.load(DERIVED / "context_k.npz")
    sub = base["sub"].astype(np.int64)
    img = base["image_nr"].astype(np.int64)
    conc = base["concept"].astype(np.int32) - 1
    ttype, perf, lag = base["ttype"], base["perf"], base["lag"]
    trial = base["trial"].astype(float)
    enough = ck["n_prior"] >= MIN_PRIOR

    feat = ck["top10"].astype(float)
    catc = cat_crowd.astype(float)
    typ = typ_of_concept[conc].astype(float)

    fh = open(OUT / "crowding.txt", "w", encoding="utf-8")

    def w(s=""):
        print(s)
        fh.write(s + "\n")

    w("CATEGORY ISOLATION OR FEATURE CROWDING")
    w("=" * 100)
    w(f"{len(sub):,} trials, {len(np.unique(sub)):,} participants. Trials require at "
      f"least {MIN_PRIOR} preceding items.")
    w(f"{n_categorized} of 1,854 concepts have a code in the THINGS category table "
      f"(0 = no named category); two")
    w("concepts are category-mates when their codes are equal.")
    w("Both predictors are z-scored and centered within image, so every comparison "
      "holds the item fixed.")
    w("")

    for name, mask, outcome in [
            ("FALSE ALARMS on new items", np.isin(ttype, [1, 3]) & enough,
             (perf == 13).astype(float)),
            ("HITS on target repeats", (ttype == 2) & enough & (lag > 0),
             (perf == 11).astype(float))]:
        w("-" * 100)
        w(f"{name}   n = {mask.sum():,} trials")
        w("-" * 100)
        y = np.where(mask, outcome, np.nan)
        pos = np.where(mask, trial, np.nan)
        lg = np.where(mask & (lag > 0), lag.astype(float), np.nan)

        f_ = np.where(mask, feat, np.nan)
        c_ = np.where(mask, catc, np.nan)
        f_ = (f_ - np.nanmean(f_)) / np.nanstd(f_)
        c_ = (c_ - np.nanmean(c_)) / np.nanstd(c_)
        fc, cc_ = group_center(f_, img), group_center(c_, img)
        tz = np.where(mask, typ, np.nan)
        tz = (tz - np.nanmean(tz)) / np.nanstd(tz)

        ok = np.isfinite(fc) & np.isfinite(cc_)
        w(f"  correlation between the two crowding measures (within image): "
          f"{np.corrcoef(fc[ok], cc_[ok])[0,1]:+.3f}")
        w("")

        extra = [lg] if "HITS" in name else []
        exn = ["lag"] if extra else []

        report(w, "M1  feature crowding alone",
               fit(y, [fc, pos] + extra, sub, ["feature", "position"] + exn),
               ["feature"])
        report(w, "M2  category crowding alone",
               fit(y, [cc_, pos] + extra, sub, ["category", "position"] + exn),
               ["category"])
        report(w, "M3  both together",
               fit(y, [fc, cc_, pos] + extra, sub,
                   ["feature", "category", "position"] + exn),
               ["feature", "category"])
        report(w, "M4  both, plus category crowding x database typicality",
               fit(y, [fc, cc_, tz, cc_ * tz, pos] + extra, sub,
                   ["feature", "category", "typicality", "category_x_typicality",
                    "position"] + exn),
               ["feature", "category", "category_x_typicality"])

    w("-" * 100)
    w("READING THIS")
    w("-" * 100)
    w("  M1 and M2 enter each measure alone. M3 enters both together, so each")
    w("  coefficient is the effect of one measure with the other held constant.")
    w("  M4 adds database typicality (the mean of a concept's ten highest SPoSE")
    w("  correlations over all 1,854 concepts) and its product with category")
    w("  crowding, which asks whether category-mates matter less for concepts that")
    w("  are already crowded in the full space.")
    fh.close()
    print(f"\nwrote {a5_common.relative(OUT / 'crowding.txt')}")


if __name__ == "__main__":
    main()
