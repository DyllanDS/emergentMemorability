"""
compute_context.py: how similar each trial's preceding stream was to the image shown, step 2 of the
SI Appendix, Section 5 chain (Fig. S13).

WHAT IT DOES
    The original memorability study is continuous recognition, so the interference set for a given
    trial is whatever has already been presented in that participant's stream, not the whole image
    set. For each trial it computes the similarity of the probe to the items shown before it, in the
    49-dimensional SPoSE space (similarity of two concepts = the Pearson correlation of their SPoSE
    rows). A prior presentation of the same image is the memory trace being probed, not context, so
    it is left out of the context and recorded separately as the lag.

    Three context measures:
      ctx_top10  the mean of the 10 highest similarities to the items already shown  (used in the paper)
      ctx_paper  the mean of normpdf(similarity, 1, 2) over the items already shown (the kernel of the
                 summed similarity of Figs. 5 to 7)
      ctx_sharp  the mean of normpdf(similarity, 1, 0.25) over the items already shown
    Context is a mean rather than a sum, so it reflects how similar the prior context is, separately
    from how much of it there is; list length is kept as n_prior.

    Outcome coding: a first showing (trial type 1 or 3) answered "old" is a false alarm; a repeat
    (type 2 or 4) answered "old" is a hit.

    Concepts are numbered as in the Kramer et al. files (things_image_mapping.csv, alphabetical
    order); that number, minus one, is the row index used in the SPoSE file.

INPUT   external/kramer_trials/derived/trials.npz   (from extract_trials.py)
        shared/things/spose_embedding_49d_sorted.txt
OUTPUT  external/kramer_trials/derived/trials_with_context.npz   (one row per trial; not distributed)
SEEDS   none.  RUNTIME  about a minute.
Run from the repository root:  python A5_things_context_effects_figS13_S14/code/compute_context.py
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import norm

import a5_common

OUT = a5_common.make_derived_folder()

SPOSE = a5_common.spose_file()
TOPK = 10


def build_kernels():
    """Concept-by-concept similarity, transformed three ways."""
    E = np.loadtxt(SPOSE)
    R = np.corrcoef(E).astype(np.float32)
    np.fill_diagonal(R, 1.0)
    ker = {
        "paper": norm.pdf(R, 1, 2).astype(np.float32),
        "sharp": norm.pdf(R, 1, 0.25).astype(np.float32),
    }
    return R, ker


def main():
    d = np.load(OUT / "trials.npz")
    sub, trial, image, concept = d["sub"], d["trial"], d["image_nr"], d["concept"]
    ttype, perf = d["ttype"], d["perf"]
    n = len(sub)
    print(f"{n:,} trials, {len(np.unique(sub)):,} subjects")

    R, ker = build_kernels()
    conc0 = concept.astype(np.int32) - 1  # SPoSE rows are 0-based

    ctx = {k: np.full(n, np.nan, dtype=np.float32) for k in ker}
    ctx["top10"] = np.full(n, np.nan, dtype=np.float32)
    n_prior = np.zeros(n, dtype=np.int16)
    lag = np.full(n, -1, dtype=np.int16)

    # ---- walk subjects; each stream is short enough to do densely ----
    bounds = np.searchsorted(sub, np.arange(sub[-1] + 2))
    t0 = time.time()
    for s in range(len(bounds) - 1):
        a, b = bounds[s], bounds[s + 1]
        if b - a < 2:
            continue
        c = conc0[a:b]
        img = image[a:b]
        T = len(c)

        same_img = img[:, None] == img[None, :]
        earlier = np.tril(np.ones((T, T), dtype=bool), -1)
        ctx_mask = earlier & ~same_img          # prior items that are not this item

        n_prior[a:b] = ctx_mask.sum(1)

        # lag to this image's own previous presentation, if any
        rep = earlier & same_img
        has = rep.any(1)
        if has.any():
            idx = np.arange(T)
            last = np.where(has, T - 1 - np.argmax(rep[:, ::-1], axis=1), -1)
            lag[a:b] = np.where(has, idx - last, -1)

        for name, K in ker.items():
            sm = K[np.ix_(c, c)]
            tot = (sm * ctx_mask).sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                ctx[name][a:b] = tot / np.maximum(ctx_mask.sum(1), 1)

        # top-k uses the raw correlation, where "most similar" is meaningful
        sm = R[np.ix_(c, c)]
        sm = np.where(ctx_mask, sm, -np.inf)
        part = np.sort(sm, axis=1)[:, -TOPK:]
        part = np.where(np.isfinite(part), part, np.nan)
        with np.errstate(invalid="ignore"):
            ctx["top10"][a:b] = np.nanmean(part, axis=1)

        if s % 2000 == 0:
            print(f"  subject {s}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- outcome coding ----
    is_first = np.isin(ttype, [1, 3])     # never seen before in this stream
    is_rep = np.isin(ttype, [2, 4])       # second presentation
    fa = np.where(is_first, (perf == 13).astype(np.int8), -1)
    hit = np.where(is_rep, (perf == 11).astype(np.int8), -1)

    out = dict(sub=sub, trial=trial, image_nr=image, concept=concept,
               ttype=ttype, perf=perf, n_prior=n_prior, lag=lag,
               is_first=is_first, is_rep=is_rep, fa=fa, hit=hit,
               ctx_paper=ctx["paper"], ctx_sharp=ctx["sharp"],
               ctx_top10=ctx["top10"])
    np.savez_compressed(OUT / "trials_with_context.npz", **out)

    print(f"\nwrote {a5_common.relative(OUT / 'trials_with_context.npz')}  ({time.time()-t0:.0f}s)")
    for k in ["paper", "sharp", "top10"]:
        v = ctx[k][np.isfinite(ctx[k])]
        print(f"  ctx_{k:6s} mean={v.mean():.4f} sd={v.std():.4f}")
    print(f"  first presentations: {is_first.sum():,}  FA rate {fa[is_first].mean():.4f}")
    print(f"  repeats:             {is_rep.sum():,}  hit rate {hit[is_rep].mean():.4f}")
    print(f"  median lag on repeats: {np.median(lag[is_rep & (lag > 0)]):.0f}")


if __name__ == "__main__":
    main()
