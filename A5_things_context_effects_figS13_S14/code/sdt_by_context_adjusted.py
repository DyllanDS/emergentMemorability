"""
sdt_by_context_adjusted.py: the within-image context effect with serial position and lag held
constant, step 4 of the SI Appendix, Section 5 chain (Fig. S13, filled circles; the 0.13 and its
interval; the across-image SD of d').

INPUT   external/kramer_trials/derived/trials_with_context.npz   (from compute_context.py)
OUTPUT  outputs/regenerated/sdt_by_context_adjusted.txt   (the printed report)
        outputs/regenerated/sdt_by_context_adjusted.npz   (per bin: H, F, dprime, crit and their 2.5 and
                                                           97.5 percentiles; the bin 1 minus bin 20
                                                           contrast and its z-scale parts with intervals;
                                                           the across-image spread of d'; the descriptive
                                                           bins; the position and lag profile of the bins)
SEED    numpy default_rng(0), one stream, in this order: 400 multinomial participant resamples for the
        descriptive bins (step 1), 400 for the adjusted bins (step 2), 400 participant resamples of the
        regression route (step 4), 400 for the common bins (step 6) and 400 for the vigilance repeats
        (step 7).  RUNTIME  under a minute.
Run from the repository root:  python A5_things_context_effects_figS13_S14/code/sdt_by_context_adjusted.py

The descriptive version of this curve (sdt_by_context.py) ranks each image's
trials by the similarity of the stream that preceded them and reads off the hit
and false alarm rates of each rank bin. That comparison holds image identity
constant but nothing else, and one other thing moves with the bins: similarity
to the ten nearest preceding items can only grow as the stream lengthens,
because a longer stream offers more candidates to be near. In these data the
least crowded twentieth of an image's trials sits at a mean serial position of
32 and the most crowded at 237, and late trials differ in memory for reasons
other than crowding: more intervening material, a drifting criterion, fatigue.

Re-weighting those raw bins to a common position distribution does not remove
this, because bins that far apart barely share support: the earliest bin has no
trials at all in the late position strata. Step 1 of the report gives that
version with its coverage.

Step 2 bins a context measure from which the stream-length trend has been
removed: for each trial, how crowded its preceding stream was relative to what
is typical that far into a stream. Bins of that residual span the whole range of
serial positions, so re-weighting them to a common position distribution (and,
for repeats, a common lag distribution) is an exact adjustment rather than an
extrapolation. What remains is attributable to what preceded the probe rather
than to when it arrived.

Step 4 computes the same contrast a second way. Direct standardization makes no
functional-form assumption but can only adjust for strata it can populate; the
per-participant regression assumes the bin effect is linear but adjusts for
position and lag continuously.

The context measure is identical for old and new trials: ctx_top10, the mean of
the ten highest SPoSE correlations between the probe and the items already
shown in that participant's stream, with prior presentations of the probe
itself excluded from the window in both cases. The position trend removed from
it is likewise one common trend, estimated over old and new trials together, so
a given value means the same thing on either kind of trial.
"""
import time
import numpy as np
from scipy.stats import norm, ttest_1samp
from scipy.sparse import coo_matrix

import a5_common

DERIVED = a5_common.DERIVED
OUT = a5_common.make_regenerated_folder()

N_BINS = 20
N_POS = 10          # serial-position strata used for standardization
N_LAG = 10          # lag strata, crossed with position, for repeats
N_POS_FINE = 50     # strata the common position trend in context is removed over
N_BOOT = 400
MIN_PRIOR = 20      # matches the descriptive analysis
MIN_TRIALS = 30     # participants need enough trials for a usable slope
MIN_OBS = 30        # per-image minimum for the across-image d' spread
KERNEL = "ctx_top10"
EPS = 1e-6          # keeps a predicted rate inside the domain of the z transform
RNG = np.random.default_rng(0)


# ---- 1. binning and stratification ----
def within_image_bin(x, img, n_bins=N_BINS):
    """Rank bin of x within each image; -1 where the image has too few trials.

    Equal-frequency ranks mean every image contributes the same number of
    trials to every bin, so image composition cannot differ across bins.
    """
    out = np.full(len(x), -1, dtype=np.int8)
    ok = np.isfinite(x)
    idx = np.flatnonzero(ok)
    order = idx[np.lexsort((x[idx], img[idx]))]
    im = img[order]
    _, start, counts = np.unique(im, return_index=True, return_counts=True)
    rank = np.arange(len(order)) - np.repeat(start, counts)
    cnt = np.repeat(counts, counts)
    frac = rank / np.maximum(cnt - 1, 1)
    b = np.minimum((frac * n_bins).astype(np.int8), n_bins - 1)
    b[cnt < n_bins] = -1
    out[order] = b
    return out


def quantile_index(x, mask, n_q):
    """Quantile stratum of x, with cut points taken from the analyzed trials."""
    edges = np.quantile(x[mask], np.linspace(0, 1, n_q + 1)[1:-1])
    return np.searchsorted(edges, x).astype(np.int64)


def detrend_by(x, stratum, mask, n_strata):
    """x minus its mean in each stratum, the mean taken over the analyzed trials.

    One trend for every trial type, so the detrended measure keeps a common
    meaning for old and new trials.
    """
    s = stratum[mask]
    n = np.bincount(s, minlength=n_strata).astype(float)
    tot = np.bincount(s, weights=x[mask], minlength=n_strata)
    mu = np.divide(tot, n, out=np.full(n_strata, np.nan), where=n > 0)
    return x - mu[stratum], mu


# ---- 2. direct standardization over strata, with a participant bootstrap ----
def cell_matrices(sub, n_sub, b, stratum, n_strata, y, mask):
    """Per-participant successes and opportunities in each (bin, stratum) cell.

    Stored sparse because a participant touches only a few dozen of the cells,
    which turns each bootstrap replicate into one small matrix-vector product.
    Transposed on the way out because that is the orientation the bootstrap
    multiplies in.
    """
    m = mask & (b >= 0)
    rows = sub[m]
    cols = b[m].astype(np.int64) * n_strata + stratum[m]
    shape = (n_sub, N_BINS * n_strata)
    N = coo_matrix((np.ones(int(m.sum())), (rows, cols)), shape=shape)
    K = coo_matrix((y[m].astype(float), (rows, cols)), shape=shape)
    return K.T.tocsr(), N.T.tocsr()


def std_rate(K, N, w):
    """Rate per bin, re-weighted to the common stratum distribution w.

    Cells a bin never populates are dropped and the weights renormalized, so a
    bin is never credited with the grand mean of a stratum it does not occupy.
    The report gives the coverage of each analysis: at full coverage the
    renormalization does nothing, and at low coverage a bin is compared on the
    part of the standard population it reaches.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(N > 0, K / np.maximum(N, 1.0), 0.0)
    wm = np.where(N > 0, w[None, :], 0.0)
    return (p * wm).sum(1) / wm.sum(1)


def totals(KT, NT, n_strata):
    """Summed counts per (bin, stratum), from the transposed per-participant form."""
    K = np.asarray(KT.sum(1)).ravel().reshape(N_BINS, n_strata)
    N = np.asarray(NT.sum(1)).ravel().reshape(N_BINS, n_strata)
    return K, N


def sdt_from_rates(H, F):
    zH, zF = norm.ppf(H), norm.ppf(F)
    return zH - zF, -0.5 * (zH + zF), zH, zF


def bootstrap_curves(hKT, hNT, hw, fKT, fNT, fw, n_sub, n_boot=N_BOOT):
    """Resample participants; the same resample drives hits and false alarms.

    Multinomial cell counts are the same thing as drawing participants with
    replacement, and let a replicate be taken without materializing the
    resampled data.
    """
    nsh, nsf = hw.size, fw.size
    p = np.full(n_sub, 1.0 / n_sub)
    out = {k: np.empty((n_boot, N_BINS)) for k in "HFdc"}
    for i in range(n_boot):
        m = RNG.multinomial(n_sub, p).astype(float)
        H = std_rate((hKT @ m).reshape(N_BINS, nsh),
                     (hNT @ m).reshape(N_BINS, nsh), hw)
        F = std_rate((fKT @ m).reshape(N_BINS, nsf),
                     (fNT @ m).reshape(N_BINS, nsf), fw)
        d, c, _, _ = sdt_from_rates(H, F)
        out["H"][i], out["F"][i], out["d"][i], out["c"][i] = H, F, d, c
    return {k: np.percentile(v, [2.5, 97.5], axis=0) for k, v in out.items()}, out


# ---- 3. per-participant regression, closed form over 21k participants ----
def group_center(x, g):
    """Subtract the image mean, ignoring trials that carry NaN for this analysis."""
    ok = np.isfinite(x)
    ng = int(g.max()) + 1
    n = np.bincount(g[ok], minlength=ng)
    s = np.bincount(g[ok], weights=x[ok], minlength=ng)
    m = np.divide(s, n, out=np.full(ng, np.nan), where=n > 0)
    out = x - m[g]
    out[n[g] < 2] = np.nan
    return out


def fit(y, preds, sub, names, min_trials=MIN_TRIALS):
    """One multiple regression per participant, solved from per-participant sums."""
    ok = np.isfinite(y)
    for p in preds:
        ok &= np.isfinite(p)
    ns = int(sub.max()) + 1
    y = y[ok]
    P = [p[ok] for p in preds]
    s = sub[ok]
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
    return (f"  {label:40s} b={b.mean():+.5f} [{b.mean()-ci:+.5f}, {b.mean()+ci:+.5f}]"
            f"  t({len(b)-1})={t:+.2f}  p={p:.3g}  n={len(b):,}")


def main():
    t0 = time.time()
    d = np.load(DERIVED / "trials_with_context.npz")
    sub_raw = d["sub"].astype(np.int64)
    img = d["image_nr"].astype(np.int64)
    ttype, n_prior, lag = d["ttype"], d["n_prior"], d["lag"]
    perf = d["perf"]
    trial = d["trial"].astype(float)
    ctx = d[KERNEL].astype(float)

    subs = np.unique(sub_raw)
    remap = np.zeros(subs.max() + 1, dtype=np.int64)
    remap[subs] = np.arange(len(subs))
    sub = remap[sub_raw]
    n_sub = len(subs)

    enough = n_prior >= MIN_PRIOR
    fa_mask = np.isin(ttype, [1, 3]) & enough
    y_fa = (perf == 13).astype(float)
    y_hit = (perf == 11).astype(float)
    hit_masks = {"target": (ttype == 2) & enough & (lag > 0),
                 "vigil": (ttype == 4) & enough & (lag > 0)}
    analysed = fa_mask | hit_masks["target"] | hit_masks["vigil"]

    # ---- context with the stream-length trend removed, one trend for all types ----
    fine = quantile_index(trial, analysed, N_POS_FINE)
    ctx_adj, trend = detrend_by(ctx, fine, analysed, N_POS_FINE)

    # ---- position strata for false alarms, shared by every block below ----
    fa_pos = quantile_index(trial, fa_mask, N_POS)
    fw = np.bincount(fa_pos[fa_mask], minlength=N_POS).astype(float)
    fw /= fw.sum()

    def fa_bins(x):
        return within_image_bin(np.where(fa_mask, x, np.nan), img)

    fb_raw, fb_adj = fa_bins(ctx), fa_bins(ctx_adj)
    fKT_raw, fNT_raw = cell_matrices(sub, n_sub, fb_raw, fa_pos, N_POS, y_fa, fa_mask)
    fKT_adj, fNT_adj = cell_matrices(sub, n_sub, fb_adj, fa_pos, N_POS, y_fa, fa_mask)

    lines = []

    def w(t=""):
        print(t, flush=True)
        lines.append(t)

    def run_block(label, hit_mask, hb, fb_use, fKT_use, fNT_use, verbose):
        """One hit-type by binning-scheme analysis, adjusted and unadjusted."""
        hit_pos = quantile_index(trial, hit_mask, N_POS)
        hit_lag = quantile_index(lag.astype(float), hit_mask, N_LAG)
        cell = hit_pos * N_LAG + hit_lag
        nsh = N_POS * N_LAG
        hw = np.bincount(cell[hit_mask], minlength=nsh).astype(float)
        hw /= hw.sum()
        hKT, hNT = cell_matrices(sub, n_sub, hb, cell, nsh, y_hit, hit_mask)

        hK, hN = totals(hKT, hNT, nsh)
        fK, fN = totals(fKT_use, fNT_use, N_POS)
        H = std_rate(hK, hN, hw)
        F = std_rate(fK, fN, fw)
        dp, cr, zH, zF = sdt_from_rates(H, F)
        ci, boot = bootstrap_curves(hKT, hNT, hw, fKT_use, fNT_use, fw, n_sub)

        # how much of the standard population each bin actually reaches; an
        # adjustment is only an adjustment where this is near 1
        cov_h = np.where(hN > 0, hw[None, :], 0.0).sum(1)
        cov_f = np.where(fN > 0, fw[None, :], 0.0).sum(1)

        # unadjusted counterpart, pooled exactly as the descriptive analysis does
        def pooled(b, y, mask):
            m = mask & (b >= 0)
            bb = b[m].astype(np.int64)
            k = np.bincount(bb, weights=y[m], minlength=N_BINS)
            n = np.bincount(bb, minlength=N_BINS).astype(float)
            return (k + 0.5) / (n + 1.0), n
        Hu, hn_u = pooled(hb, y_hit, hit_mask)
        Fu, fn_u = pooled(fb_use, y_fa, fa_mask)
        dpu, cru, zHu, zFu = sdt_from_rates(Hu, Fu)

        # what each bin actually contains, in context and in position
        def by_bin(v, mask, b):
            return np.array([v[mask & (b == i)].mean() for i in range(N_BINS)])
        prof = dict(pos_h=by_bin(trial, hit_mask, hb),
                    lag_h=by_bin(lag.astype(float), hit_mask, hb),
                    ctx_h=by_bin(ctx, hit_mask, hb),
                    pos_f=by_bin(trial, fa_mask, fb_use),
                    ctx_f=by_bin(ctx, fa_mask, fb_use))

        # contrast: least crowded minus most crowded, and its two z-scale parts
        con = {"d": dp[0] - dp[-1],
               "hit_part": zH[0] - zH[-1],
               "fa_part": zF[-1] - zF[0]}
        bd = boot["d"][:, 0] - boot["d"][:, -1]
        bh = norm.ppf(boot["H"][:, 0]) - norm.ppf(boot["H"][:, -1])
        bf = norm.ppf(boot["F"][:, -1]) - norm.ppf(boot["F"][:, 0])
        con_ci = {"d": np.percentile(bd, [2.5, 97.5]),
                  "hit_part": np.percentile(bh, [2.5, 97.5]),
                  "fa_part": np.percentile(bf, [2.5, 97.5])}
        con_u = {"d": dpu[0] - dpu[-1],
                 "hit_part": zHu[0] - zHu[-1],
                 "fa_part": zFu[-1] - zFu[0]}

        if verbose:
            w("bin  hit rate [95% CI]          FA rate [95% CI]             "
              "d' [95% CI]              c [95% CI]")
            for i in range(N_BINS):
                w(f"{i+1:3d}  {H[i]:.4f} [{ci['H'][0][i]:.4f}, {ci['H'][1][i]:.4f}]   "
                  f"{F[i]:.4f} [{ci['F'][0][i]:.4f}, {ci['F'][1][i]:.4f}]   "
                  f"{dp[i]:.3f} [{ci['d'][0][i]:.3f}, {ci['d'][1][i]:.3f}]   "
                  f"{cr[i]:+.3f} [{ci['c'][0][i]:+.3f}, {ci['c'][1][i]:+.3f}]")
            w()
            w("  bin 1 = least crowded preceding context, bin 20 = most crowded")
            w(f"  hit rate   bin1 {H[0]:.4f} -> bin20 {H[-1]:.4f}  "
              f"(change {H[-1]-H[0]:+.4f})   r with bin index "
              f"{np.corrcoef(np.arange(N_BINS), H)[0,1]:+.3f}")
            w(f"  FA rate    bin1 {F[0]:.4f} -> bin20 {F[-1]:.4f}  "
              f"(change {F[-1]-F[0]:+.4f})   r with bin index "
              f"{np.corrcoef(np.arange(N_BINS), F)[0,1]:+.3f}")
            w(f"  d'         bin1 {dp[0]:.3f} -> bin20 {dp[-1]:.3f}   "
              f"monotone decreasing: {bool(np.all(np.diff(dp) < 0))}   "
              f"r with bin index {np.corrcoef(np.arange(N_BINS), dp)[0,1]:+.3f}")
            w(f"  criterion  bin1 {cr[0]:+.3f} -> bin20 {cr[-1]:+.3f}   "
              f"r with bin index {np.corrcoef(np.arange(N_BINS), cr)[0,1]:+.3f}")
            w()
            w(f"  trials: {int(hn_u.sum()):,} repeats, {int(fn_u.sum()):,} first showings")
            w(f"  coverage (share of the standard population each bin reaches): "
              f"hits {cov_h.min():.3f}-{cov_h.max():.3f}, "
              f"false alarms {cov_f.min():.3f}-{cov_f.max():.3f}")
            w(f"  serial position, repeats:   bin1 {prof['pos_h'][0]:.1f} "
              f"-> bin20 {prof['pos_h'][-1]:.1f}")
            w(f"  serial position, new items: bin1 {prof['pos_f'][0]:.1f} "
              f"-> bin20 {prof['pos_f'][-1]:.1f}")
            w(f"  lag, repeats:               bin1 {prof['lag_h'][0]:.1f} "
              f"-> bin20 {prof['lag_h'][-1]:.1f}")
            w(f"  raw context, repeats:       bin1 {prof['ctx_h'][0]:.4f} "
              f"-> bin20 {prof['ctx_h'][-1]:.4f} "
              f"(span {prof['ctx_h'][-1]-prof['ctx_h'][0]:+.4f})")
            w(f"  raw context, new items:     bin1 {prof['ctx_f'][0]:.4f} "
              f"-> bin20 {prof['ctx_f'][-1]:.4f} "
              f"(span {prof['ctx_f'][-1]-prof['ctx_f'][0]:+.4f})")
            w()

        return dict(H=H, F=F, dprime=dp, crit=cr, ci=ci, con=con, con_ci=con_ci,
                    con_u=con_u, Hu=Hu, Fu=Fu, dpu=dpu, cru=cru, hn=hn_u,
                    fn=fn_u, cov_h=cov_h, cov_f=cov_f, hb=hb, **prof)

    # ---- header ----
    w("D' BY PRECEDING-CONTEXT SIMILARITY, STANDARDIZED FOR POSITION AND LAG")
    w("=" * 104)
    w(f"{len(sub):,} trials, {n_sub:,} participants, {len(np.unique(img)):,} images.")
    w(f"Context = {KERNEL}: the mean of the ten highest SPoSE correlations between the probe")
    w("and the items already shown in that participant's stream, excluding prior showings of")
    w("the probe itself. The measure, and the predecessor window it runs over, are identical")
    w("for old and new trials; only the outcome differs.")
    w(f"Trials require at least {MIN_PRIOR} preceding items. Bins are within-image rank bins")
    w(f"({N_BINS} of them), so each image contributes equally to every bin.")
    w(f"Standardization: {N_POS} serial-position strata for false alarms and "
      f"{N_POS}x{N_LAG} position-by-lag strata")
    w("for hits, each bin re-weighted to the same stratum distribution, so bins are compared")
    w("at the same positions and lags.")
    w(f"Confidence intervals are {N_BOOT} bootstrap resamples of participants.")
    w()

    # ---- 1. the descriptive bins, standardized as far as their support allows ----
    w("-" * 104)
    w("STEP 1. THE DESCRIPTIVE BINS, RE-WEIGHTED")
    w("-" * 104)
    w("Bins here are within-image ranks of raw context similarity, exactly as in")
    w("sdt_by_context.py. The coverage line gives the share of the standard")
    w("population each bin reaches; where it is below 1, a bin has no trials in some")
    w("strata and is compared on the strata it has.")
    w()
    raw = run_block("descriptive bins", hit_masks["target"],
                    within_image_bin(np.where(hit_masks["target"], ctx, np.nan), img),
                    fb_raw, fKT_raw, fNT_raw, verbose=True)

    # ---- 2. bins of context with the stream-length trend removed ----
    w("-" * 104)
    w("STEP 2. ADJUSTED CURVE: bins of context relative to what is typical at that position")
    w("-" * 104)
    w(f"Context is detrended against {N_POS_FINE} serial-position strata, using one common")
    w("trend estimated over old and new trials together, so bin 1 now means 'an unusually")
    w("sparse stream for this point in the session' rather than 'early'. Those bins span the")
    w("whole position range, so re-weighting them to a common position (and lag) distribution")
    w("is an exact adjustment. Fig. S13 plots this curve (filled circles).")
    w()
    adj = run_block("adjusted bins", hit_masks["target"],
                    within_image_bin(np.where(hit_masks["target"], ctx_adj, np.nan), img),
                    fb_adj, fKT_adj, fNT_adj, verbose=True)

    sa = adj["con"]["d"]
    su = raw["con_u"]["d"]

    # ---- 3. unadjusted versus adjusted, side by side ----
    w("-" * 104)
    w("STEP 3. UNADJUSTED VERSUS ADJUSTED")
    w("-" * 104)
    w("The unadjusted rows reproduce sdt_by_context.py exactly: raw context bins, pooled")
    w("without re-weighting. The adjusted rows are step 2. The z-scale column is the")
    w("contribution of that measure to the d' contrast, so the two z entries add to it.")
    w()
    w("                             bin 1     bin 20    change     z-scale contribution")
    for nm, u, a, key in [("hit rate", raw["Hu"], adj["H"], "hit_part"),
                          ("FA rate", raw["Fu"], adj["F"], "fa_part")]:
        w(f"  {nm:10s} unadjusted   {u[0]:.4f}    {u[-1]:.4f}   {u[-1]-u[0]:+.4f}"
          f"    {raw['con_u'][key]:+.4f}")
        w(f"  {nm:10s} adjusted     {a[0]:.4f}    {a[-1]:.4f}   {a[-1]-a[0]:+.4f}"
          f"    {adj['con'][key]:+.4f}")
    w(f"  {'d prime':10s} unadjusted   {raw['dpu'][0]:.3f}     {raw['dpu'][-1]:.3f}"
      f"    {raw['dpu'][-1]-raw['dpu'][0]:+.3f}")
    w(f"  {'d prime':10s} adjusted     {adj['dprime'][0]:.3f}     {adj['dprime'][-1]:.3f}"
      f"    {adj['dprime'][-1]-adj['dprime'][0]:+.3f}")
    w()
    w(f"  least-to-most-crowded d' contrast, unadjusted: {su:+.3f}")
    w(f"  least-to-most-crowded d' contrast, adjusted:   {sa:+.3f} "
      f"[{adj['con_ci']['d'][0]:+.3f}, {adj['con_ci']['d'][1]:+.3f}]")
    w()
    w("  Decomposition on the common z scale; the two parts add to the d' contrast:")
    for key, nm in [("hit_part", "hits  z(H bin1) - z(H bin20)"),
                    ("fa_part", "FAs   z(F bin20) - z(F bin1)")]:
        lo, hi = adj["con_ci"][key]
        w(f"    {nm:30s} unadjusted {raw['con_u'][key]:+.4f} "
          f"({100*raw['con_u'][key]/su:.0f}% of the unadjusted contrast)")
        w(f"    {'':30s} adjusted   {adj['con'][key]:+.4f} [{lo:+.4f}, {hi:+.4f}] "
          f"({100*adj['con'][key]/sa:.0f}% of the adjusted contrast)")
    w()

    # ---- 4. model-based route ----
    w("-" * 104)
    w("STEP 4. MODEL-BASED ROUTE: per-participant regressions with position and lag entered")
    w("-" * 104)
    w("Each participant contributes one slope on the context-bin index (1-20, centered within")
    w("image), with serial position, and lag for hits, entered as continuous covariates.")
    w("Inference is across participants. Slopes are in probability per bin step; predicted")
    w("rates at bin 1 and bin 20 come from moving 19 bin steps around each measure's overall")
    w("rate, and those predicted rates are converted to d'. Run on the adjusted bins, this is")
    w("the same contrast as step 2 by a route that assumes linearity instead of relying on")
    w("stratum overlap.")
    w()

    hit_mask = hit_masks["target"]
    posf = np.where(fa_mask, trial, np.nan)
    posh = np.where(hit_mask, trial, np.nan)
    lagh = np.where(hit_mask, lag.astype(float), np.nan)

    def slopes(fb_use, hb_use):
        bf_ = group_center(np.where(fa_mask, fb_use.astype(float), np.nan), img)
        bh_ = group_center(np.where(hit_mask, hb_use.astype(float), np.nan), img)
        cf_ = fit(np.where(fa_mask, y_fa, np.nan), [bf_, posf], sub,
                  ["bin", "position"])
        ch_ = fit(np.where(hit_mask, y_hit, np.nan), [bh_, posh, lagh], sub,
                  ["bin", "position", "lag"])
        return cf_["bin"], ch_["bin"]

    bf_adj, bh_adj = slopes(fb_adj, adj["hb"])
    bf_raw, bh_raw = slopes(fb_raw, raw["hb"])
    w(coef_line(bf_adj, "false alarms, per adjusted bin step"))
    w(coef_line(bh_adj, "hits, per adjusted bin step"))
    w(coef_line(bf_raw, "false alarms, per descriptive bin step"))
    w(coef_line(bh_raw, "hits, per descriptive bin step"))

    # coefficients per SD of the raw context measure as well
    both_mask = fa_mask | hit_mask
    z = (ctx - np.nanmean(ctx[both_mask])) / np.nanstd(ctx[both_mask])
    czf = fit(np.where(fa_mask, y_fa, np.nan),
              [group_center(np.where(fa_mask, z, np.nan), img), posf], sub,
              ["ctx", "position"])
    czh = fit(np.where(hit_mask, y_hit, np.nan),
              [group_center(np.where(hit_mask, z, np.nan), img), posh, lagh], sub,
              ["ctx", "position", "lag"])
    w(coef_line(czf["ctx"], "false alarms, per SD of raw context"))
    w(coef_line(czh["ctx"], "hits, per SD of raw context"))
    w()

    Fbar = y_fa[fa_mask].mean()
    Hbar = y_hit[hit_mask].mean()
    half = (N_BINS - 1) / 2

    def predicted(bh_, bf_):
        """Bin1-minus-bin20 d' implied by a pair of linear slopes."""
        clip = lambda v: np.clip(v, EPS, 1 - EPS)
        H1, H20 = clip(Hbar - half * bh_), clip(Hbar + half * bh_)
        F1, F20 = clip(Fbar - half * bf_), clip(Fbar + half * bf_)
        d1 = norm.ppf(H1) - norm.ppf(F1)
        d20 = norm.ppf(H20) - norm.ppf(F20)
        return d1 - d20, norm.ppf(H1) - norm.ppf(H20), norm.ppf(F20) - norm.ppf(F1)

    m_d, m_h, m_f = predicted(np.nanmean(bh_adj), np.nanmean(bf_adj))
    # participants are resampled jointly, so the hit and FA slopes stay paired
    bs = np.empty((N_BOOT, 3))
    for i in range(N_BOOT):
        r = RNG.integers(0, n_sub, n_sub)
        bs[i] = predicted(np.nanmean(bh_adj[r]), np.nanmean(bf_adj[r]))
    bs_ci = np.percentile(bs, [2.5, 97.5], axis=0)

    w(f"  overall rates used as the anchor: hit {Hbar:.4f}, FA {Fbar:.4f}")
    w(f"  predicted bin1 - bin20 d' contrast: {m_d:+.3f} "
      f"[{bs_ci[0,0]:+.3f}, {bs_ci[1,0]:+.3f}]")
    w(f"    hits part {m_h:+.4f} [{bs_ci[0,1]:+.4f}, {bs_ci[1,1]:+.4f}]   "
      f"FA part {m_f:+.4f} [{bs_ci[0,2]:+.4f}, {bs_ci[1,2]:+.4f}]")
    w(f"  direct-standardization contrast for comparison: {sa:+.3f} "
      f"[{adj['con_ci']['d'][0]:+.3f}, {adj['con_ci']['d'][1]:+.3f}]")
    w()

    # ---- 5. descriptive scale comparison ----
    w("-" * 104)
    w("STEP 5. DESCRIPTIVE SCALE COMPARISON against the across-image spread")
    w("-" * 104)
    is_old = ttype == 2
    is_new = np.isin(ttype, [1, 3])
    nimg = int(img.max()) + 1
    hk = np.bincount(img[is_old], (perf[is_old] == 11).astype(float), nimg)
    hn = np.bincount(img[is_old], None, nimg).astype(float)
    fk = np.bincount(img[is_new], (perf[is_new] == 13).astype(float), nimg)
    fn = np.bincount(img[is_new], None, nimg).astype(float)
    keep = (hn >= MIN_OBS) & (fn >= MIN_OBS)
    di = (norm.ppf((hk[keep] + 0.5) / (hn[keep] + 1.0)) -
          norm.ppf((fk[keep] + 0.5) / (fn[keep] + 1.0)))
    p5, p95 = np.percentile(di, [5, 95])
    spread = p95 - p5
    w(f"  across-image d' over {int(keep.sum()):,} images: SD {di.std(ddof=1):.3f}, "
      f"5th-95th percentile range {spread:.3f}")
    w(f"  adjusted within-image contrast {sa:.3f} = {100*sa/spread:.0f}% of that "
      f"5th-95th range, or {sa/di.std(ddof=1):.2f} across-image SD")
    w(f"  unadjusted contrast {su:.3f} = {100*su/spread:.0f}% of the same range")
    w()

    # ---- 6. robustness: one common within-image bin definition for old and new ----
    w("-" * 104)
    w("STEP 6. ROBUSTNESS: adjusted bins defined on one common within-image distribution")
    w("-" * 104)
    w("Above, old and new trials are ranked separately within image, so bin 20 means the most")
    w("crowded twentieth of this image's repeats and, separately, of its first showings. Here")
    w("both are ranked against the same pooled within-image distribution, so a bin index")
    w("refers to the same absolute context level for hits and false alarms. Image composition")
    w("is then no longer exactly balanced across bins.")
    w()
    cb = within_image_bin(np.where(fa_mask | hit_mask, ctx_adj, np.nan), img)
    cfb = np.where(fa_mask, cb, -1).astype(np.int8)
    chb = np.where(hit_mask, cb, -1).astype(np.int8)
    cfKT, cfNT = cell_matrices(sub, n_sub, cfb, fa_pos, N_POS, y_fa, fa_mask)
    com = run_block("common bins", hit_mask, chb, cfb, cfKT, cfNT, verbose=False)
    w(f"  adjusted   hit {com['H'][0]:.4f} -> {com['H'][-1]:.4f}   "
      f"FA {com['F'][0]:.4f} -> {com['F'][-1]:.4f}   "
      f"d' {com['dprime'][0]:.3f} -> {com['dprime'][-1]:.3f}")
    w(f"  adjusted contrast {com['con']['d']:+.3f} "
      f"[{com['con_ci']['d'][0]:+.3f}, {com['con_ci']['d'][1]:+.3f}]   "
      f"(hits {com['con']['hit_part']:+.4f}, FAs {com['con']['fa_part']:+.4f})")
    w(f"  coverage: hits {com['cov_h'].min():.3f}-{com['cov_h'].max():.3f}, "
      f"false alarms {com['cov_f'].min():.3f}-{com['cov_f'].max():.3f}")
    w()

    # ---- 7. secondary: vigilance repeats ----
    w("-" * 104)
    w("STEP 7. SECONDARY: vigilance repeats (short lag)")
    w("-" * 104)
    vig = run_block("vigilance repeats", hit_masks["vigil"],
                    within_image_bin(np.where(hit_masks["vigil"], ctx_adj, np.nan), img),
                    fb_adj, fKT_adj, fNT_adj, verbose=False)
    w(f"  adjusted   hit {vig['H'][0]:.4f} -> {vig['H'][-1]:.4f}   "
      f"d' {vig['dprime'][0]:.3f} -> {vig['dprime'][-1]:.3f}")
    w(f"  adjusted contrast {vig['con']['d']:+.3f} "
      f"[{vig['con_ci']['d'][0]:+.3f}, {vig['con_ci']['d'][1]:+.3f}]   "
      f"(hits {vig['con']['hit_part']:+.4f}, FAs {vig['con']['fa_part']:+.4f})")
    w(f"  unadjusted contrast {vig['con_u']['d']:+.3f}")
    w()

    (OUT / "sdt_by_context_adjusted.txt").write_text("\n".join(lines) + "\n",
                                                     encoding="utf-8")

    np.savez(OUT / "sdt_by_context_adjusted.npz",
             H=adj["H"], F=adj["F"], dprime=adj["dprime"], crit=adj["crit"],
             Hlo=adj["ci"]["H"][0], Hhi=adj["ci"]["H"][1],
             Flo=adj["ci"]["F"][0], Fhi=adj["ci"]["F"][1],
             dlo=adj["ci"]["d"][0], dhi=adj["ci"]["d"][1],
             clo=adj["ci"]["c"][0], chi=adj["ci"]["c"][1],
             H_unadj=raw["Hu"], F_unadj=raw["Fu"],
             dprime_unadj=raw["dpu"], crit_unadj=raw["cru"],
             H_rawbin_std=raw["H"], F_rawbin_std=raw["F"],
             dprime_rawbin_std=raw["dprime"],
             cov_h=adj["cov_h"], cov_f=adj["cov_f"],
             cov_h_rawbin=raw["cov_h"], cov_f_rawbin=raw["cov_f"],
             contrast=sa, contrast_ci=adj["con_ci"]["d"], contrast_unadj=su,
             hit_part=adj["con"]["hit_part"], hit_part_ci=adj["con_ci"]["hit_part"],
             fa_part=adj["con"]["fa_part"], fa_part_ci=adj["con_ci"]["fa_part"],
             hit_part_unadj=raw["con_u"]["hit_part"],
             fa_part_unadj=raw["con_u"]["fa_part"],
             model_contrast=m_d, model_contrast_ci=bs_ci[:, 0],
             model_hit_part=m_h, model_fa_part=m_f,
             across_image_range=spread, across_image_sd=di.std(ddof=1),
             pos_by_bin=adj["pos_h"], lag_by_bin=adj["lag_h"],
             fpos_by_bin=adj["pos_f"], ctx_by_bin=adj["ctx_f"],
             pos_by_bin_raw=raw["pos_h"], fpos_by_bin_raw=raw["pos_f"],
             ctx_by_bin_raw=raw["ctx_f"],
             H_common=com["H"], F_common=com["F"], dprime_common=com["dprime"],
             contrast_common=com["con"]["d"],
             H_vigil=vig["H"], dprime_vigil=vig["dprime"],
             contrast_vigil=vig["con"]["d"],
             position_trend=trend)

    print(f"\nwrote {a5_common.relative(OUT / 'sdt_by_context_adjusted.txt')} and .npz "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
