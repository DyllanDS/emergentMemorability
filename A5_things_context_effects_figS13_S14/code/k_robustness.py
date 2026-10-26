"""
k_robustness.py: the top-k and kernel context measures of every trial, and the context effect for each
of them over trials with at least 50 preceding items; step 6 of the SI Appendix, Section 5 chain.
crowding.py and stats_from_trials.py read the top-10 measure and n_prior from its per-trial table
context_k.npz (the feature crowding behind the 4.6 ratio).

INPUT   external/kramer_trials/derived/trials.npz and trials_with_context.npz
        shared/things/spose_embedding_49d_sorted.txt
OUTPUT  external/kramer_trials/derived/context_k.npz   (one row per trial; not distributed)
        outputs/regenerated/k_robustness.txt and k_robustness.npz   (per measure: d', c, H, F per bin)
SEEDS   none.  RUNTIME  about two minutes.
Run from the repository root:  python A5_things_context_effects_figS13_S14/code/k_robustness.py

WHAT IT DOES
    For every trial it computes, from the SPoSE similarities (Pearson correlation of the SPoSE rows)
    between the probe and the items already shown in the session (earlier showings of the probe left
    out): top-k, the mean of the k highest similarities, for k = 1, 2, 5, 10, 20 and 50; the mean of
    exp((similarity - 1) / tau) for tau = 0.1 and 0.25; the means of the two normal kernels of
    compute_context.py; and the unnormalized sums of the steep normal kernel and of the tau = 0.1
    exponential. The analysis uses the trials with at least 50 preceding items, so all values of k
    are computed over the same trials. Concepts are numbered as in the Kramer et al. files and that
    number, minus one, is the row index used in the SPoSE file, as in compute_context.py. The binning
    and SDT functions come from shared/python/sdt.py.

    The report k_robustness.txt gives, for each measure, d', the hit and false alarm rates and the
    criterion in bin 1 and bin 20 of the 20 within-image bins, and the Spearman correlation of d' and
    of the criterion with the bin index.
"""
import time
import numpy as np
from scipy.stats import norm, spearmanr

import a5_common
import sdt as S                 # shared/python/sdt.py: within_image_bin, cell_counts, sdt, N_BINS

DERIVED = a5_common.DERIVED
OUT = a5_common.make_regenerated_folder()

MIN_PRIOR = 50
N_BINS = 20
KS = [1, 2, 5, 10, 20, 50]
TAUS = [0.10, 0.25]
SPOSE = a5_common.spose_file()


def compute_measures():
    """One pass over participants, every context measure from the same sort."""
    d = np.load(DERIVED / "trials.npz")
    sub, image, concept = d["sub"], d["image_nr"], d["concept"]
    n = len(sub)

    E = np.loadtxt(SPOSE)
    R = np.corrcoef(E).astype(np.float32)
    np.fill_diagonal(R, 1.0)
    K_paper = norm.pdf(R, 1, 2).astype(np.float32)
    K_sharp = norm.pdf(R, 1, 0.25).astype(np.float32)

    names = ([f"top{k}" for k in KS] + [f"exp{t:g}" for t in TAUS]
             + ["mean_paper", "mean_sharp", "sum_sharp", "sum_exp0.1"])
    ctx = {nm: np.full(n, np.nan, dtype=np.float32) for nm in names}
    n_prior = np.zeros(n, dtype=np.int16)

    conc0 = concept.astype(np.int32) - 1
    bounds = np.searchsorted(sub, np.arange(sub[-1] + 2))
    t0 = time.time()
    for s in range(len(bounds) - 1):
        a, b = bounds[s], bounds[s + 1]
        if b - a < 2:
            continue
        c = conc0[a:b]
        img = image[a:b]
        T = len(c)

        keep = np.tril(np.ones((T, T), bool), -1) & (img[:, None] != img[None, :])
        cnt = keep.sum(1)
        n_prior[a:b] = cnt

        sim = R[np.ix_(c, c)]
        masked = np.where(keep, sim, -np.inf)
        srt = -np.sort(-masked, axis=1)          # descending, invalid at the tail
        for k in KS:
            ctx[f"top{k}"][a:b] = srt[:, :k].mean(1)

        with np.errstate(over="ignore"):
            for tau in TAUS:
                w = np.where(keep, np.exp((sim - 1.0) / tau), 0.0)
                ctx[f"exp{tau:g}"][a:b] = w.sum(1) / np.maximum(cnt, 1)
        for nm, KK in [("mean_paper", K_paper), ("mean_sharp", K_sharp)]:
            ctx[nm][a:b] = (KK[np.ix_(c, c)] * keep).sum(1) / np.maximum(cnt, 1)
        # unnormalized sums: same kernels, but not divided by how much context there is
        ctx["sum_sharp"][a:b] = (K_sharp[np.ix_(c, c)] * keep).sum(1)
        with np.errstate(over="ignore"):
            ctx["sum_exp0.1"][a:b] = np.where(keep, np.exp((sim - 1.0) / 0.10), 0.0).sum(1)

        if s % 4000 == 0:
            print(f"  subject {s} ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(DERIVED / "context_k.npz", n_prior=n_prior, **ctx)
    print(f"wrote context_k.npz ({time.time()-t0:.0f}s)")
    return names


def main():
    if not (DERIVED / "context_k.npz").exists():
        names = compute_measures()
    else:
        names = [k for k in np.load(DERIVED / "context_k.npz").files if k != "n_prior"]

    base = np.load(DERIVED / "trials_with_context.npz")
    ck = np.load(DERIVED / "context_k.npz")
    sub = base["sub"].astype(np.int64)
    img = base["image_nr"].astype(np.int64)
    ttype, perf, lag = base["ttype"], base["perf"], base["lag"]
    enough = ck["n_prior"] >= MIN_PRIOR

    subs = np.unique(sub)
    remap = np.zeros(subs.max() + 1, dtype=np.int64)
    remap[subs] = np.arange(len(subs))
    s_idx = remap[sub]
    n_sub = len(subs)
    S.N_BINS = N_BINS

    fa_mask = np.isin(ttype, [1, 3]) & enough
    hit_mask = (ttype == 2) & enough & (lag > 0)

    fh = open(OUT / "k_robustness.txt", "w", encoding="utf-8")

    def w(t=""):
        print(t)
        fh.write(t + "\n")

    w("Robustness of the context effect to how the nearest competitors are counted")
    w("=" * 104)
    w(f"{n_sub:,} participants; {N_BINS} within-image bins; trials require at least "
      f"{MIN_PRIOR} preceding items")
    w(f"hits from target repeats ({hit_mask.sum():,} trials), false alarms from all "
      f"new items ({fa_mask.sum():,} trials)")
    w("d' and c are pooled within bin; rho is the Spearman correlation of the measure "
      "with bin index")
    w()
    w(f"{'measure':12s} {'d1':>7s} {'d20':>7s} {'delta':>7s} {'mono':>5s} "
      f"{'rho(d)':>7s} | {'hit1':>6s} {'hit20':>6s} {'fa1':>7s} {'fa20':>7s} | "
      f"{'c1':>6s} {'c20':>6s} {'rho(c)':>7s}")
    w("-" * 104)

    rows = {}
    for nm in ([f"top{k}" for k in KS] + [f"exp{t:g}" for t in TAUS]
               + ["sum_sharp", "sum_exp0.1", "mean_sharp", "mean_paper"]):
        x = ck[nm].astype(float)
        fb = S.within_image_bin(np.where(fa_mask, x, np.nan), img)
        hb = S.within_image_bin(np.where(hit_mask, x, np.nan), img)
        fk, fn = S.cell_counts(s_idx, fb, (perf == 13).astype(float), n_sub)
        hk, hn = S.cell_counts(s_idx, hb, (perf == 11).astype(float), n_sub)
        dp, cr, da, H, F = S.sdt(hk.sum(0), hn.sum(0), fk.sum(0), fn.sum(0))
        mono = bool(np.all(np.diff(dp) < 0))
        rho_d = spearmanr(np.arange(N_BINS), dp).statistic
        rho_c = spearmanr(np.arange(N_BINS), cr).statistic
        rows[nm] = dict(dprime=dp, crit=cr, da=da, H=H, F=F,
                        mono=mono, rho_d=rho_d, rho_c=rho_c)
        w(f"{nm:12s} {dp[0]:7.3f} {dp[-1]:7.3f} {dp[-1]-dp[0]:7.3f} "
          f"{str(mono):>5s} {rho_d:7.3f} | {H[0]:6.3f} {H[-1]:6.3f} "
          f"{F[0]:7.4f} {F[-1]:7.4f} | {cr[0]:6.3f} {cr[-1]:6.3f} {rho_c:7.3f}")

    w()
    w("Reading the table")
    w("-" * 104)
    tops = [rows[f"top{k}"] for k in KS]
    dd = [r["dprime"][-1] - r["dprime"][0] for r in tops]
    w(f"  Across k = {KS}, the change in d' from bin 1 to bin 20 is "
      f"{', '.join(f'{v:.3f}' for v in dd)}; rho(d') is at most "
      f"{max(r['rho_d'] for r in tops):.3f} at every k.")
    w("  k = 1 is the similarity to the single nearest preceding item; a larger k averages")
    w("  the k highest similarities, and the exponential measures weight every preceding")
    w("  item by exp((similarity - 1) / tau).")
    w()
    w("  The steep normal kernel (sd 0.25), summed over the preceding items and averaged")
    w("  over them:")
    for nm, lab in [("sum_sharp", "summed"), ("mean_sharp", "averaged")]:
        r = rows[nm]
        w(f"    {lab:9s} {nm:11s} d' drop {r['dprime'][-1]-r['dprime'][0]:+.3f}  "
          f"rho(d') {r['rho_d']:+.3f}")
    w("  The exponential kernel (tau 0.1), summed and averaged:")
    for nm, lab in [("sum_exp0.1", "summed"), ("exp0.1", "averaged")]:
        r = rows[nm]
        w(f"    {lab:9s} {nm:11s} d' drop {r['dprime'][-1]-r['dprime'][0]:+.3f}  "
          f"rho(d') {r['rho_d']:+.3f}")
    w("  A summed measure adds the kernel values of all preceding items; an averaged")
    w("  measure divides that sum by the number of preceding items, which differs from")
    w("  trial to trial. top-k averages only the k highest similarities. The steep")
    w("  kernels give most of their weight to the closest preceding items (normal kernel")
    w("  with sd 0.25; exponential with tau 0.1).")
    w()
    w("  Spearman correlation of d' and of the criterion with the bin index, by measure:")
    w(f"{'':4s}{'measure':12s} {'rho(d-prime)':>13s} {'rho(criterion)':>15s}")
    for nm in [f"top{k}" for k in KS] + ["sum_sharp", "exp0.1", "mean_sharp", "mean_paper"]:
        r = rows[nm]
        w(f"{'':4s}{nm:12s} {r['rho_d']:13.3f} {r['rho_c']:15.3f}")
    w("  rho(d-prime) and rho(criterion) are the Spearman correlations of the 20 bin")
    w("  values of d' and of the criterion c with the bin index; bin 1 holds each image's")
    w("  trials with the least similar preceding items and bin 20 those with the most")
    w("  similar.")
    w()
    w("  All measures here are computed over the trials with at least 50 preceding items,")
    w("  so that every k uses the same trials; sdt_by_context.py and Fig. S13 use the")
    w("  trials with at least 20 preceding items.")

    fh.close()
    np.savez(OUT / "k_robustness.npz",
             ks=np.array(KS),
             **{f"{nm}_{f}": rows[nm][f] for nm in rows
                for f in ("dprime", "crit", "H", "F")})
    print(f"\nwrote {a5_common.relative(OUT / 'k_robustness.txt')}")


if __name__ == "__main__":
    main()
