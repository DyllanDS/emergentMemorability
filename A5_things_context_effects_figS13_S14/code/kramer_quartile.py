"""
kramer_quartile.py: Kramer et al.'s (2023) category-count quartile test, run on the trial data, beside
the within-image context effect in the same participants and images; step 5 of the SI Appendix,
Section 5 chain (Fig. S14).

INPUT   external/kramer_trials/derived/trials_with_context.npz   (from compute_context.py)
        shared/things/concept_categories.csv (via a5_common.concept_categories_alphabetical(): the
        higher-level category of each concept in the concept numbering of the Kramer et al. files)
OUTPUT  outputs/regenerated/kramer_quartile.txt           (the printed report)
        outputs/regenerated/kramer_quartile_summary.csv   (one row per category)
        external/kramer_trials/derived/kramer_quartile.npz
                    (per category: each scored image's corrected recognition in the top and bottom
                    quartile, the 200 shuffled-split correlations, the per-bin rates and d', and the
                    quartile of every participant (<category>_quartile_of_participant), which
                    stats_from_trials.py uses for the intervals of T41; image- and participant-level,
                    so not distributed)
SEED    numpy default_rng(0), one stream through the six categories in order (animal, food, clothing,
        container, tool, vehicle); per category: the random tie-break of the quartile ranking, the 200
        shuffled splits, then 50 random halves of the top quartile and 50 of the bottom quartile.
RUNTIME about a minute.
Run from the repository root:  python A5_things_context_effects_figS13_S14/code/kramer_quartile.py

Kramer et al. asked whether memorability is stable across changes in how much of
a category a participant saw: split participants into quartiles by the number of
exemplars of a category in their session, score that category's images separately
inside the top and bottom quartile, correlate the two score vectors, and test
whether the score distributions differ.

That test asks whether the ordering of a fixed population of images survives a
coarse change in category count. The within-image test asks whether one image,
held identical, is discriminated worse when the specific stream that preceded it
crowded it in feature space. Both are computed here on the same trials.

Corrected recognition (hit rate minus false alarm rate) is the score of Kramer et
al. and is kept as they defined it. The report also lists how far the quartile
split moves crowding, what d' change that predicts, the split-half reliability of
the per-image scores, and, per category, the correlation of category count with
crowding.
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, pearsonr, norm, linregress

import a5_common
import sdt as shared_sdt

within_image_bin = shared_sdt.within_image_bin
sdt = shared_sdt.sdt

DERIVED = a5_common.DERIVED
OUT = a5_common.make_regenerated_folder()

CATEGORIES = ["animal", "food", "clothing", "container", "tool", "vehicle"]
CTX = "ctx_top10"        # nearest-competitor crowding; the measure the d' effect is defined on
MIN_PRIOR = 20           # matches the main trial-level analysis
N_BINS = 20              # within-image context bins, full-sample
N_BINS_Q = 5             # within-image context bins inside one quartile, where trials per image are 4x fewer
MIN_IMG_TRIALS = 5       # per image per group, separately for repeats and for first showings
N_NULL = 200             # shuffled quartile assignments
N_SPLIT = 50             # split-half reliability resamples
RNG = np.random.default_rng(0)


# ---- 1. per-image scoring inside an arbitrary set of participants ----
def image_scores(img_idx, is_hit_trial, hit_y, is_fa_trial, fa_y, sel, n_img):
    """Hit rate, false alarm rate and corrected recognition per image, over the
    trials in `sel`. Returns rates plus the trial counts they rest on, since the
    counts decide which images are scoreable in both halves of a comparison."""
    # h, f: True for the trials of `sel` that can be a hit (a repeat of an image) or a false alarm (a
    # first showing). hn, fn: those trials counted per image (np.bincount adds one per trial to its
    # image); hk, fk: the trials among them answered "old" (hits, false alarms). H = hk / hn is each
    # image's hit rate, F = fk / fn its false alarm rate, and H - F its corrected recognition.
    h = sel & is_hit_trial
    f = sel & is_fa_trial
    hn = np.bincount(img_idx[h], minlength=n_img).astype(float)
    hk = np.bincount(img_idx[h], weights=hit_y[h], minlength=n_img)
    fn = np.bincount(img_idx[f], minlength=n_img).astype(float)
    fk = np.bincount(img_idx[f], weights=fa_y[f], minlength=n_img)
    with np.errstate(invalid="ignore", divide="ignore"):
        H = hk / hn
        F = fk / fn
    return H, F, H - F, hn, fn


def pooled_sdt(is_hit_trial, hit_y, is_fa_trial, fa_y, sel):
    """Aggregate hit rate, false alarm rate and d' over every trial in `sel`."""
    h = sel & is_hit_trial
    f = sel & is_fa_trial
    dp, cr, da, H, F = sdt(hit_y[h].sum(), h.sum(), fa_y[f].sum(), f.sum())
    return float(H), float(F), float(dp)


# ---- 2. d' across within-image context bins ----
def context_dprime(ctx, img, hit_mask, hit_y, fa_mask, fa_y, n_bins):
    """Pooled hit rate, false alarm rate and d' per within-image context bin.

    Hits and false alarms come from different trials, so each is binned within
    image on its own and the two rate vectors are combined bin by bin. Binning
    within image is what makes this a same-image comparison: every bin contains
    the same images, differing only in what preceded them."""
    # hb, fb: the context bin of every hit trial and every false alarm trial (binned within its own
    # image; -1 for a trial outside the analysis); okh, okf: the trials that have a bin. hn, fn: trials
    # per bin; hk, fk: "old" answers per bin; dp, cr: d' and the criterion of each bin from those
    # pooled counts (shared/python/sdt.py); hs, fs: the summed context similarity per bin, so mctx is
    # each bin's mean context similarity.
    hb = within_image_bin(np.where(hit_mask, ctx, np.nan), img, n_bins)
    fb = within_image_bin(np.where(fa_mask, ctx, np.nan), img, n_bins)
    okh, okf = hb >= 0, fb >= 0
    hn = np.bincount(hb[okh], minlength=n_bins).astype(float)
    hk = np.bincount(hb[okh], weights=hit_y[okh], minlength=n_bins)
    fn = np.bincount(fb[okf], minlength=n_bins).astype(float)
    fk = np.bincount(fb[okf], weights=fa_y[okf], minlength=n_bins)
    dp, cr, da, H, F = sdt(hk, hn, fk, fn)
    # raw crowding per bin, so the bin axis can be put back on its own scale and
    # a between-participant crowding difference translated into predicted d'
    hs = np.bincount(hb[okh], weights=ctx[okh], minlength=n_bins)
    fs = np.bincount(fb[okf], weights=ctx[okf], minlength=n_bins)
    mctx = (hs + fs) / np.maximum(hn + fn, 1)
    return dict(H=H, F=F, dprime=dp, crit=cr, hn=hn, fn=fn, mctx=mctx)


def partial_r(x, y, z):
    """Correlation of x and y with z partialled out of both."""
    rxy = pearsonr(x, y)[0]
    rxz = pearsonr(x, z)[0]
    ryz = pearsonr(y, z)[0]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def main():
    t_start = time.time()

    # ---- 3. load trials and attach THINGS higher-level category ----
    d = np.load(DERIVED / "trials_with_context.npz")
    sub = d["sub"].astype(np.int64)
    img = d["image_nr"].astype(np.int64)
    concept = d["concept"].astype(np.int32)
    ttype, perf, lag, n_prior = d["ttype"], d["perf"], d["lag"], d["n_prior"]
    ctx = d[CTX].astype(float)

    link = a5_common.concept_categories_alphabetical()    # the public category file, by concept number
    # category 0 ("NA") marks concepts without a named higher-level category; the six categories
    # analyzed here are named ones, so those concepts do not enter the quartile test
    cat_of_concept = link["category"].to_numpy().astype(np.int32)
    label_of_cat = dict(zip(link["category"], link["category_label"]))
    cat = cat_of_concept[concept - 1]

    subs = np.unique(sub)
    remap = np.zeros(subs.max() + 1, dtype=np.int64)
    remap[subs] = np.arange(len(subs))
    s_idx = remap[sub]
    n_sub = len(subs)

    stream_len = np.bincount(s_idx, minlength=n_sub).astype(float)

    is_fa_trial = np.isin(ttype, [1, 3])          # first showings: a "old" response is a false alarm
    is_hit_trial = (ttype == 2) & (lag > 0)       # target repeats, the trials memorability is scored on
    fa_y = (perf == 13).astype(float)
    hit_y = (perf == 11).astype(float)
    enough = n_prior >= MIN_PRIOR

    fh = open(OUT / "kramer_quartile.txt", "w", encoding="utf-8")

    def w(t=""):
        print(t, flush=True)
        fh.write(t + "\n")

    w("KRAMER ET AL.'S QUARTILE TEST AND THE WITHIN-IMAGE CONTEXT EFFECT, SAME TRIALS")
    w("=" * 100)
    w(f"{len(sub):,} trials, {n_sub:,} participants, {len(np.unique(img)):,} images.")
    w("")
    w("Kramer et al. (2023) split participants into quartiles by how many exemplars of a")
    w("category they saw, scored that category's images separately within the top and the")
    w("bottom quartile, correlated the two score vectors, and compared the score")
    w("distributions. That test is reproduced here on the same trial-level data.")
    w("Alongside it, in the same participants and the same images, is our test: bin each")
    w("image's trials by how crowded the preceding stream was in feature space (mean")
    w(f"similarity to the {CTX.split('_')[1]} nearest preceding items, binned WITHIN image so")
    w("image identity is held constant) and read off d' from the least to the most crowded bin.")
    w("")
    w("Corrected recognition (hit rate minus false alarm rate) is kept as Kramer et al.")
    w("defined it for their test.")
    w("")

    summary_rows = []
    per_image_store = {}

    # ---- 4. the overall split of the context effect into hits and false alarms ----
    all_ok = enough
    res_all = context_dprime(ctx, img, is_hit_trial & all_ok, hit_y,
                             is_fa_trial & all_ok, fa_y, N_BINS)
    zH = norm.ppf(res_all["H"])
    zF = norm.ppf(res_all["F"])
    d_hit_all = zH[0] - zH[-1]
    d_fa_all = zF[-1] - zF[0]
    d_tot_all = res_all["dprime"][0] - res_all["dprime"][-1]

    w("-" * 100)
    w("WHOLE DATASET, ALL CATEGORIES: where the within-image effect lives")
    w("-" * 100)
    w(f"  d' least-crowded bin {res_all['dprime'][0]:.3f} -> most-crowded bin "
      f"{res_all['dprime'][-1]:.3f}   (drop {d_tot_all:.3f})")
    w(f"  hit rate {res_all['H'][0]:.4f} -> {res_all['H'][-1]:.4f}     "
      f"false alarm rate {res_all['F'][0]:.4f} -> {res_all['F'][-1]:.4f}")
    w(f"  of the {d_tot_all:.3f} drop in d', {d_hit_all:.3f} ({100*d_hit_all/d_tot_all:.0f}%) comes from hits "
      f"and {d_fa_all:.3f} ({100*d_fa_all/d_tot_all:.0f}%) from false alarms")
    w("")

    # per-image corrected recognition against its two components, over all images
    Hi, Fi, CRi, hni, fni = image_scores(
        np.searchsorted(np.unique(img), img), is_hit_trial, hit_y,
        is_fa_trial, fa_y, np.ones(len(sub), bool), len(np.unique(img)))
    ok = (hni >= 20) & (fni >= 20)
    w(f"  across the {ok.sum():,} images with at least 20 trials of each kind, corrected")
    w(f"  recognition correlates r = {pearsonr(CRi[ok], Hi[ok])[0]:+.3f} with hit rate and "
      f"r = {pearsonr(CRi[ok], Fi[ok])[0]:+.3f} with false alarm rate;")
    w(f"  the mean false alarm rate is {np.average(Fi[ok], weights=fni[ok]):.4f}.")
    w("  Across images, corrected recognition therefore follows the hit rate far more closely")
    w("  than the false alarm rate.")
    w("")

    # ---- 5. one block per category ----
    for name in CATEGORIES:
        cid = int(link.loc[link["category_label"] == name, "category"].iloc[0])
        m_cat = cat == cid
        cat_img = np.unique(img[m_cat])
        n_img = len(cat_img)
        img_idx = np.zeros(len(sub), dtype=np.int64)
        img_idx[m_cat] = np.searchsorted(cat_img, img[m_cat])

        # exemplars of the category seen per participant: each image occurs once
        # as a first showing, so first showings count distinct exemplars
        seen = np.bincount(s_idx[m_cat & is_fa_trial], minlength=n_sub)
        # total presentations, repeats included, is the other reading of "how many
        # exemplars did they see" and is reported because it sets the scale of the split
        seen_pres = np.bincount(s_idx[m_cat], minlength=n_sub)

        # equal-sized quartiles by that count, ties broken at random so the four
        # groups are the same size despite the counts being small integers
        order = np.lexsort((RNG.random(n_sub), seen))
        qlab = np.empty(n_sub, dtype=np.int8)
        qlab[order] = np.minimum((np.arange(n_sub) * 4) // n_sub, 3)

        q_of_trial = qlab[s_idx]
        sel_top = m_cat & (q_of_trial == 3)
        sel_bot = m_cat & (q_of_trial == 0)

        w("=" * 100)
        w(f"CATEGORY: {name.upper()}   "
          f"({(link['category'] == cid).sum()} concepts, {n_img:,} images, {m_cat.sum():,} trials)")
        w("=" * 100)
        w("")
        w("  Kramer's split: participants ranked by exemplars of this category seen, in quartiles")
        w("  quartile   n participants   mean exemplars   range     presentations   hit rate   FA rate      d'")
        qmeans = []
        qd = []
        for q in range(4):
            qm = m_cat & (q_of_trial == q)
            H, F, dp = pooled_sdt(is_hit_trial, hit_y, is_fa_trial, fa_y, qm)
            s = seen[qlab == q]
            qmeans.append(s.mean())
            qd.append(dp)
            w(f"  Q{q+1}         {(qlab == q).sum():>8,}        {s.mean():>8.1f}       "
              f"{s.min():>3d}-{s.max():<4d}      {seen_pres[qlab == q].mean():>8.1f}      "
              f"{H:.4f}     {F:.4f}    {dp:.3f}")
        w(f"  Q4 minus Q1: d' {qd[3]-qd[0]:+.3f}")
        w("")

        # per-image corrected recognition in the top and the bottom quartile
        Ht, Ft, CRt, hnt, fnt = image_scores(img_idx, is_hit_trial, hit_y,
                                             is_fa_trial, fa_y, sel_top, n_img)
        Hb, Fb, CRb, hnb, fnb = image_scores(img_idx, is_hit_trial, hit_y,
                                             is_fa_trial, fa_y, sel_bot, n_img)
        keep = ((hnt >= MIN_IMG_TRIALS) & (fnt >= MIN_IMG_TRIALS) &
                (hnb >= MIN_IMG_TRIALS) & (fnb >= MIN_IMG_TRIALS))
        rho = spearmanr(CRt[keep], CRb[keep]).statistic
        r_pear = pearsonr(CRt[keep], CRb[keep])[0]
        wstat = wilcoxon(CRt[keep], CRb[keep])

        # shuffled null: the same procedure with quartile membership permuted
        # across participants, which preserves group sizes but destroys any
        # relation to how much of the category was seen
        null = np.empty(N_NULL)
        for i in range(N_NULL):
            perm = RNG.permutation(qlab)
            qt = perm[s_idx]
            st = m_cat & (qt == 3)
            sb = m_cat & (qt == 0)
            _, _, ct, hnA, fnA = image_scores(img_idx, is_hit_trial, hit_y,
                                              is_fa_trial, fa_y, st, n_img)
            _, _, cb, hnB, fnB = image_scores(img_idx, is_hit_trial, hit_y,
                                              is_fa_trial, fa_y, sb, n_img)
            k = ((hnA >= MIN_IMG_TRIALS) & (fnA >= MIN_IMG_TRIALS) &
                 (hnB >= MIN_IMG_TRIALS) & (fnB >= MIN_IMG_TRIALS))
            null[i] = spearmanr(ct[k], cb[k]).statistic
        nlo, nhi = np.percentile(null, [2.5, 97.5])

        # split-half reliability inside each quartile, so the ceiling on any
        # cross-quartile correlation is on the table next to the correlation
        def split_half(group):
            members = np.flatnonzero(qlab == group)
            out = []
            for _ in range(N_SPLIT):
                p = RNG.permutation(members)
                a = np.zeros(n_sub, bool); a[p[:len(p) // 2]] = True
                b = np.zeros(n_sub, bool); b[p[len(p) // 2:]] = True
                _, _, ca, hA, fA = image_scores(img_idx, is_hit_trial, hit_y,
                                                is_fa_trial, fa_y, m_cat & a[s_idx], n_img)
                _, _, cb, hB, fB = image_scores(img_idx, is_hit_trial, hit_y,
                                                is_fa_trial, fa_y, m_cat & b[s_idx], n_img)
                k = ((hA >= 3) & (fA >= 3) & (hB >= 3) & (fB >= 3))
                out.append(spearmanr(ca[k], cb[k]).statistic)
            r = float(np.mean(out))
            return r, 2 * r / (1 + r)      # Spearman-Brown to full quartile size

        rel_t_half, rel_t = split_half(3)
        rel_b_half, rel_b = split_half(0)
        corrected = rho / np.sqrt(max(rel_t * rel_b, 1e-9))

        w("  Kramer's test, reproduced")
        w(f"    images scoreable in both quartiles (>= {MIN_IMG_TRIALS} trials of each kind): {keep.sum():,} of {n_img:,}")
        w(f"    trials per image behind each score: top quartile {hnt[keep].mean():.1f} repeats and "
          f"{fnt[keep].mean():.1f} first showings; bottom quartile {hnb[keep].mean():.1f} and {fnb[keep].mean():.1f}")
        w(f"    Spearman rho between top- and bottom-quartile image scores: {rho:+.3f}   (Pearson {r_pear:+.3f})")
        w(f"    shuffled quartile assignment, {N_NULL} permutations: mean rho {null.mean():+.3f}  "
          f"95% interval [{nlo:+.3f}, {nhi:+.3f}]")
        w(f"    split-half reliability of the scores within a quartile (Spearman-Brown corrected): "
          f"top {rel_t:.3f}, bottom {rel_b:.3f}")
        w(f"    cross-quartile rho corrected for that unreliability: {corrected:+.3f}"
          + ("   (at the ceiling; the estimate exceeds 1 because the reliabilities are themselves noisy)"
             if corrected > 1 else ""))
        w(f"    corrected recognition, mean over images: top quartile {CRt[keep].mean():.4f}, "
          f"bottom quartile {CRb[keep].mean():.4f}  (difference {CRt[keep].mean()-CRb[keep].mean():+.4f})")
        w(f"    Wilcoxon signed-rank on the paired per-image scores: W = {wstat.statistic:.0f}, "
          f"p = {wstat.pvalue:.3g}")
        w("")

        # ---- our test, same category, same participants ----
        m_ok = m_cat & enough
        res = context_dprime(ctx, img, is_hit_trial & m_ok, hit_y,
                             is_fa_trial & m_ok, fa_y, N_BINS)
        zH = norm.ppf(res["H"]); zF = norm.ppf(res["F"])
        d_hit = zH[0] - zH[-1]
        d_fa = zF[-1] - zF[0]
        d_tot = res["dprime"][0] - res["dprime"][-1]
        r_bin = np.corrcoef(np.arange(N_BINS), res["dprime"])[0, 1]

        w(f"  Our test, same images: d' by within-image bin of preceding-stream crowding ({N_BINS} bins)")
        w("    bin                 least crowded ->  most crowded      drop     r with bin index")
        w(f"    d'                       {res['dprime'][0]:.3f}          {res['dprime'][-1]:.3f}"
          f"        {-d_tot:+.3f}        {r_bin:+.3f}")
        w(f"    hit rate                 {res['H'][0]:.4f}         {res['H'][-1]:.4f}       "
          f"{res['H'][-1]-res['H'][0]:+.4f}")
        w(f"    false alarm rate         {res['F'][0]:.4f}         {res['F'][-1]:.4f}       "
          f"{res['F'][-1]-res['F'][0]:+.4f}")
        w(f"    share of the d' drop: hits {100*d_hit/d_tot:.0f}%, false alarms {100*d_fa/d_tot:.0f}%")
        w("")

        # the same effect inside each of the two quartiles Kramer's test compares
        qres = {}
        for tag, sel in [("top", sel_top), ("bottom", sel_bot)]:
            mq = sel & enough
            r = context_dprime(ctx, img, is_hit_trial & mq, hit_y,
                               is_fa_trial & mq, fa_y, N_BINS_Q)
            qres[tag] = r
        r_full_q = context_dprime(ctx, img, is_hit_trial & m_ok, hit_y,
                                  is_fa_trial & m_ok, fa_y, N_BINS_Q)
        w(f"    the same effect inside each quartile ({N_BINS_Q} bins, since a quartile has a quarter of the trials)")
        w("      group              d' least crowded   d' most crowded    drop")
        for tag, r in [("all participants", r_full_q), ("top quartile", qres["top"]),
                       ("bottom quartile", qres["bottom"])]:
            w(f"      {tag:18s}      {r['dprime'][0]:.3f}             {r['dprime'][-1]:.3f}        "
              f"{r['dprime'][0]-r['dprime'][-1]:+.3f}")
        w("")

        # ---- is category count even a proxy for feature crowding? ----
        # per participant, the mean crowding of this category's trials in their
        # own stream, against how many exemplars of the category they saw
        cm = m_cat & enough & np.isfinite(ctx)
        cnt = np.bincount(s_idx[cm], minlength=n_sub).astype(float)
        tot = np.bincount(s_idx[cm], weights=ctx[cm], minlength=n_sub)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_ctx = tot / cnt
        good = cnt >= 5
        r_cc = pearsonr(seen[good].astype(float), mean_ctx[good])
        rho_cc = spearmanr(seen[good].astype(float), mean_ctx[good])
        # a longer session raises both the exemplar count and the crowding of every
        # trial in it, for reasons that have nothing to do with the category, so the
        # partial correlation is the one that speaks to the proxy question
        r_cc_partial = partial_r(seen[good].astype(float), mean_ctx[good], stream_len[good])
        w(f"  Is category count a proxy for feature crowding? Across the {good.sum():,} participants with at")
        w(f"  least 5 scoreable trials of this category, the number of exemplars they saw correlates")
        w(f"  r = {r_cc[0]:+.3f} (rho = {rho_cc.statistic:+.3f}, p = {r_cc[1]:.3g}) with the mean crowding of")
        w(f"  those trials' preceding streams; with session length partialled out, r = {r_cc_partial:+.3f}.")
        w("")

        # how far Kramer's split actually moves crowding, on the scale that the
        # within-image binning uses, and what d' change that predicts
        ctx_top = float(np.average(ctx[sel_top & enough & np.isfinite(ctx)]))
        ctx_bot = float(np.average(ctx[sel_bot & enough & np.isfinite(ctx)]))
        bin_range = res["mctx"][-1] - res["mctx"][0]
        slope = linregress(res["mctx"], res["dprime"]).slope
        pred = slope * (ctx_top - ctx_bot)
        obs_q = qd[3] - qd[0]
        w("  How far does the quartile split move the thing the effect depends on?")
        w(f"    mean crowding of this category's trials: top quartile {ctx_top:.4f}, "
          f"bottom quartile {ctx_bot:.4f}  (difference {ctx_top-ctx_bot:+.4f})")
        w(f"    the within-image bins span {res['mctx'][0]:.4f} to {res['mctx'][-1]:.4f}, a range of "
          f"{bin_range:.4f}, so the split moves crowding {100*(ctx_top-ctx_bot)/bin_range:.0f}% of that range")
        w(f"    d' falls {slope:.2f} per unit crowding across the bins, so the split predicts a d' difference")
        w(f"    of {pred:+.3f}; the observed top-minus-bottom quartile difference is {obs_q:+.3f}")
        w("")

        summary_rows.append(dict(
            category=name, n_concepts=int((link["category"] == cid).sum()),
            n_images=n_img, n_trials=int(m_cat.sum()),
            mean_seen_top=qmeans[3], mean_seen_bottom=qmeans[0],
            n_images_scored=int(keep.sum()),
            rho_cross=rho, null_mean=float(null.mean()), null_lo=nlo, null_hi=nhi,
            rel_top=rel_t, rel_bottom=rel_b, rho_corrected=corrected,
            CR_top=float(CRt[keep].mean()), CR_bottom=float(CRb[keep].mean()),
            wilcoxon_W=float(wstat.statistic), wilcoxon_p=float(wstat.pvalue),
            dprime_q_top=pooled_sdt(is_hit_trial, hit_y, is_fa_trial, fa_y, sel_top)[2],
            dprime_q_bottom=pooled_sdt(is_hit_trial, hit_y, is_fa_trial, fa_y, sel_bot)[2],
            HR_q_top=pooled_sdt(is_hit_trial, hit_y, is_fa_trial, fa_y, sel_top)[0],
            HR_q_bottom=pooled_sdt(is_hit_trial, hit_y, is_fa_trial, fa_y, sel_bot)[0],
            FA_q_top=pooled_sdt(is_hit_trial, hit_y, is_fa_trial, fa_y, sel_top)[1],
            FA_q_bottom=pooled_sdt(is_hit_trial, hit_y, is_fa_trial, fa_y, sel_bot)[1],
            dprime_ctx_lo=res["dprime"][0], dprime_ctx_hi=res["dprime"][-1],
            dprime_ctx_drop=d_tot, r_with_bin=r_bin,
            hit_share=d_hit / d_tot, fa_share=d_fa / d_tot,
            drop_top_quartile=qres["top"]["dprime"][0] - qres["top"]["dprime"][-1],
            drop_bottom_quartile=qres["bottom"]["dprime"][0] - qres["bottom"]["dprime"][-1],
            drop_all_5bin=r_full_q["dprime"][0] - r_full_q["dprime"][-1],
            r_count_crowding=r_cc[0], r_count_crowding_partial=r_cc_partial,
            ctx_top=ctx_top, ctx_bottom=ctx_bot, ctx_bin_range=float(bin_range),
            ctx_split_pct_of_range=float(100 * (ctx_top - ctx_bot) / bin_range),
            dprime_q_diff=float(obs_q), dprime_q_diff_predicted=float(pred),
            mean_seen_pres_top=float(seen_pres[qlab == 3].mean()),
            mean_seen_pres_bottom=float(seen_pres[qlab == 0].mean()),
        ))
        per_image_store[f"{name}_CR_top"] = CRt[keep]
        per_image_store[f"{name}_CR_bottom"] = CRb[keep]
        per_image_store[f"{name}_image_nr"] = cat_img[keep]
        per_image_store[f"{name}_null_rho"] = null
        per_image_store[f"{name}_dprime_bins"] = res["dprime"]
        per_image_store[f"{name}_H_bins"] = res["H"]
        per_image_store[f"{name}_F_bins"] = res["F"]
        per_image_store[f"{name}_quartile_of_participant"] = qlab    # used by stats_from_trials.py

    # ---- 6. the summary tables, one row per category ----
    S = pd.DataFrame(summary_rows)
    w("=" * 100)
    w("THE TABLE")
    w("=" * 100)
    w("")
    w("  Kramer's test on the left, ours on the right, same trials.")
    w("")
    w("  category      exemplars seen    cross-quartile   shuffled null      Wilcoxon    within-image d'")
    w("                 top / bottom       rho            mean [95% CI]         p        least->most crowded")
    w("  " + "-" * 96)
    for r in summary_rows:
        w(f"  {r['category']:<12s}  {r['mean_seen_top']:>5.1f} / {r['mean_seen_bottom']:<5.1f}      "
          f"{r['rho_cross']:+.3f}          {r['null_mean']:+.3f} [{r['null_lo']:+.3f},{r['null_hi']:+.3f}]   "
          f"{r['wilcoxon_p']:>8.2g}    {r['dprime_ctx_lo']:.2f} -> {r['dprime_ctx_hi']:.2f} "
          f"({-r['dprime_ctx_drop']:+.2f})")
    w("")
    w("  category      quartile hit rate    quartile FA rate    quartile d'      d' drop within image")
    w("                 top / bottom         top / bottom       top / bottom     all / top Q / bottom Q")
    w("  " + "-" * 96)
    for r in summary_rows:
        w(f"  {r['category']:<12s}  {r['HR_q_top']:.4f} / {r['HR_q_bottom']:.4f}     "
          f"{r['FA_q_top']:.4f} / {r['FA_q_bottom']:.4f}    {r['dprime_q_top']:.3f} / {r['dprime_q_bottom']:.3f}    "
          f"{r['drop_all_5bin']:.2f} / {r['drop_top_quartile']:.2f} / {r['drop_bottom_quartile']:.2f}")
    w("")
    w("  category      share of the within-image d' effect     r(exemplars, crowding)     quartile split moves")
    w("                  from hits      from false alarms        raw / session length      crowding, % of the")
    w("                                                              partialled out        within-image range")
    w("  " + "-" * 96)
    for r in summary_rows:
        w(f"  {r['category']:<12s}      {100*r['hit_share']:>5.0f}%              {100*r['fa_share']:>5.0f}%"
          f"              {r['r_count_crowding']:+.3f} / {r['r_count_crowding_partial']:+.3f}"
          f"                {r['ctx_split_pct_of_range']:>5.0f}%")
    w("")
    w("  category      between-participant d' difference the quartile split gives")
    w("                 predicted from the crowding it moves    observed (Q4 minus Q1)    within-image effect")
    w("  " + "-" * 96)
    for r in summary_rows:
        w(f"  {r['category']:<12s}          {r['dprime_q_diff_predicted']:+.3f}                          "
          f"{r['dprime_q_diff']:+.3f}                  {-r['dprime_ctx_drop']:+.3f}")
    w("")

    # ---- 7. notes ----
    w("-" * 100)
    w("NOTES")
    w("-" * 100)
    w("")
    w("Hit and false alarm trials are different trials, binned separately within image and")
    w("combined per bin, so d' pairs rates from matched but not identical trials. Similarity")
    w("is at the concept level, since SPoSE embeddings are per concept.")
    w("")
    w(f"Runtime {time.time()-t_start:.0f}s.")
    fh.close()

    S.to_csv(OUT / "kramer_quartile_summary.csv", index=False)
    np.savez_compressed(DERIVED / "kramer_quartile.npz", **per_image_store)
    print(f"\nwrote {a5_common.relative(OUT / 'kramer_quartile.txt')}")
    print(f"wrote {a5_common.relative(OUT / 'kramer_quartile_summary.csv')}")
    print(f"wrote {a5_common.relative(DERIVED / 'kramer_quartile.npz')}")
    return S


if __name__ == "__main__":
    main()
