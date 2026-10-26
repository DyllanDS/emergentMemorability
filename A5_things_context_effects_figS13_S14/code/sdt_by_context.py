"""
sdt_by_context.py: d', criterion and hit and false alarm rates in 20 within-image bins of preceding
context, step 3 of the SI Appendix, Section 5 chain (Fig. S13, open circles; the SPoSE five-bin drop of the
deep-network comparison).

WHAT IT DOES
    Each image's trials are ranked by the similarity of the stream that preceded them and cut into 20
    equal bins within that image (bin 1 = least similar preceding context), so every bin holds the
    same images. Only trials with at least 20 preceding items count. Hit and false alarm trials are
    different trials, so the two rates are binned separately within image and combined per bin:
      hits          target repeats (trial type 2) with the image's earlier showing in the stream;
                    a second set of curves uses the vigilance repeats (type 4) instead
      false alarms  first showings (types 1 and 3) called "old"
    Per bin, counts are pooled over participants and turned into log-linear corrected rates
    ((k + 0.5) / (n + 1)), d' = z(H) - z(F), criterion c = -(z(H) + z(F)) / 2 and d_a at slope 0.8.
    The 95% intervals come from 400 resamples of participants (the unit of replication), drawn with
    numpy default_rng(0) in the order ctx_top10 target, ctx_top10 vigilance, ctx_sharp target,
    ctx_sharp vigilance, ctx_paper target, ctx_paper vigilance. The paper uses ctx_top10 with target
    repeats; the script also computes the two kernel means and the vigilance repeats, whose resamples
    follow in the order above.

    The binning (within_image_bin), the per-participant cell counts (cell_counts) and the SDT step
    (sdt) are in shared/python/sdt.py, which module A2c also uses.

INPUT   external/kramer_trials/derived/trials_with_context.npz   (from compute_context.py)
OUTPUT  outputs/regenerated/sdt_results.txt                       (the printed report)
        outputs/regenerated/sdt_<measure>_<target or vigil>.npz   (per bin: dprime, crit, da, H, F and
                                                                   the 2.5 and 97.5 percentiles dlo, dhi,
                                                                   clo, chi, Hlo, Hhi, Flo, Fhi)
SEED    numpy default_rng(0), 400 participant resamples per curve.  RUNTIME  under a minute.
Run from the repository root:  python A5_things_context_effects_figS13_S14/code/sdt_by_context.py
"""
import numpy as np

import a5_common
import sdt as shared_sdt

DERIVED = a5_common.DERIVED
OUT = a5_common.make_regenerated_folder()

N_BINS = shared_sdt.N_BINS          # 20
N_BOOT = 400
KERNELS = ["ctx_top10", "ctx_sharp", "ctx_paper"]   # primary first
MIN_PRIOR = 20
DA_SLOPE = shared_sdt.DA_SLOPE      # 0.80, the fixed z-ROC slope of d_a (Didi-Barnea et al. 2016)
RNG = np.random.default_rng(0)

# the binning, cell-count and SDT functions of shared/python/sdt.py
within_image_bin = shared_sdt.within_image_bin
cell_counts = shared_sdt.cell_counts
sdt = shared_sdt.sdt


def main():
    d = np.load(DERIVED / "trials_with_context.npz")
    sub = d["sub"].astype(np.int64)
    img = d["image_nr"].astype(np.int64)
    ttype, n_prior, lag = d["ttype"], d["n_prior"], d["lag"]
    ctx = d["ctx_sharp"].astype(float)
    enough = n_prior >= MIN_PRIOR

    subs = np.unique(sub)
    remap = np.zeros(subs.max() + 1, dtype=np.int64)
    remap[subs] = np.arange(len(subs))
    s_idx = remap[sub]
    n_sub = len(subs)

    fh = open(OUT / "sdt_results.txt", "w", encoding="utf-8")

    def w(t=""):
        print(t)
        fh.write(t + "\n")

    w("d' and criterion as a function of preceding-context similarity")
    w("=" * 96)
    w(f"{n_sub:,} participants; context binned into {N_BINS} within-image bins")
    w(f"bin 1 = least similar preceding context; {N_BOOT} participant bootstraps")
    w()

    for kern in KERNELS:
        ctx = d[kern].astype(float)
        fa_mask = np.isin(ttype, [1, 3]) & enough
        fb = within_image_bin(np.where(fa_mask, ctx, np.nan), img)
        fk, fn = cell_counts(s_idx, fb, (d["perf"] == 13).astype(float), n_sub)

        for tag, label, tmask in [("target", "target repeats (long lag)", ttype == 2),
                                  ("vigil", "vigilance repeats (short lag)", ttype == 4)]:
            hit_mask = tmask & enough & (lag > 0)
            hb = within_image_bin(np.where(hit_mask, ctx, np.nan), img)
            hk, hn = cell_counts(s_idx, hb, (d["perf"] == 11).astype(float), n_sub)

            dp, cr, da, H, F = sdt(hk.sum(0), hn.sum(0), fk.sum(0), fn.sum(0))

            boot = {k: np.empty((N_BOOT, N_BINS)) for k in "dcHF"}
            for i in range(N_BOOT):
                r = RNG.integers(0, n_sub, n_sub)
                bd, bc, _, bH, bF = sdt(hk[r].sum(0), hn[r].sum(0),
                                        fk[r].sum(0), fn[r].sum(0))
                boot["d"][i], boot["c"][i] = bd, bc
                boot["H"][i], boot["F"][i] = bH, bF
            ci = {k: np.percentile(v, [2.5, 97.5], axis=0) for k, v in boot.items()}

            w("-" * 96)
            w(f"{kern}   HITS FROM: {label}")
            w("-" * 96)
            w("bin   hit rate   FA rate      d'  [95% CI]           "
              "criterion c  [95% CI]        d_a")
            for i in range(N_BINS):
                w(f"{i+1:3d}   {H[i]:.4f}    {F[i]:.4f}   {dp[i]:.3f} "
                  f"[{ci['d'][0][i]:.3f}, {ci['d'][1][i]:.3f}]   {cr[i]:+.3f} "
                  f"[{ci['c'][0][i]:+.3f}, {ci['c'][1][i]:+.3f}]   {da[i]:.3f}")

            def describe(v, name):
                mono = bool(np.all(np.diff(v) < 0))
                r = np.corrcoef(np.arange(N_BINS), v)[0, 1]
                w(f"  {name}: bin1 {v[0]:+.4f} -> bin{N_BINS} {v[-1]:+.4f} "
                  f"(change {v[-1]-v[0]:+.4f}); monotone decreasing: {mono}; "
                  f"r with bin index = {r:+.3f}")

            w()
            for v, nm in [(H, "hit rate"), (F, "FA rate"), (dp, "d'"),
                          (da, "d_a"), (cr, "criterion c")]:
                describe(v, nm)
            w()
            np.savez(OUT / f"sdt_{kern}_{tag}.npz",
                     dprime=dp, crit=cr, da=da, H=H, F=F,
                     dlo=ci["d"][0], dhi=ci["d"][1],
                     clo=ci["c"][0], chi=ci["c"][1],
                     Hlo=ci["H"][0], Hhi=ci["H"][1],
                     Flo=ci["F"][0], Fhi=ci["F"][1])

    fh.close()
    print(f"\nwrote {a5_common.relative(OUT / 'sdt_results.txt')}")


if __name__ == "__main__":
    main()
