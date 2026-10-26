"""
fit_on_sequence.py: the within-image context effect with crowding measured in 14 deep-network spaces,
the weighting over spaces fitted on half of the participant sessions and scored on the other half;
step 9 of the SI Appendix, Section 5 chain (A Similarity Space Without Human Judgments, the 0.51 drop).

WHAT IT DOES
    For each of the 14 spaces (AlexNet, VGG16-BN, ResNet18, InceptionV3, CORnet-S, DINOv3 ViT-B/16 and
    CLIP ViT-B/32, an early and a late layer each; 512 dimensions per layer, principal components where
    the layer has more, the raw 512 units for ResNet18's late layer; each column standardized) every
    image's distance to every other image is the correlation distance, z-scored with the mean and SD of
    a 2,000-image sample (offset = the 1st percentile of the sample's z-scored distances). Crowding of a
    trial = log of the summed exp(-(d - offset) / tau) over the items already shown in that session
    (prior showings of the same image left out), at five falloffs tau = 0.5, 1, 2, 4, 8: 70 measures,
    each centered within image. Trials with at least 20 preceding
    items count; a repeat answered "old" and a first showing not answered "old" are correct. Half of
    the sessions (numpy default_rng(0) permutation) fit a logistic regression of correct on the 70
    measures (C = 1.0); the other half are scored: each image's held-out trials are ranked by the
    fitted crowding into five equal bins, and d' = z(hit rate) - z(false alarm rate) per bin (rates
    clipped to [0.0001, 0.9999]). The best single space and falloff, chosen on the fitting half, is
    scored the same way for comparison, and the report lists the share of the fitted weighting
    carried by each space and each falloff.

    The image -> embedding row lookup reads the embedding_row column of shared/things/things_images.csv
    and is checked against the file names stored in every embedding file. The 14 spaces are the
    names in a5_common.NETWORK_SPACES. If PyTorch is installed, it computes the distance matrix product
    (on the CPU when there is no GPU); otherwise numpy does.

INPUT   external/kramer_trials/derived/trials.npz   (from extract_trials.py)
        shared/embeddings_things/<network>_<layer>.npz, all 14 (this repository includes the four late
        layers AlexNet, VGG16-BN, ResNet18 and CLIP; the other ten are input M3, EXTERNAL_INPUTS.md)
        shared/things/things_images.csv   (image number -> row of the embedding files)
OUTPUT  outputs/regenerated/fit_on_sequence.txt and fit_on_sequence.json   (the report)
        external/kramer_trials/derived/fit_on_sequence_heldout_counts.npz
                    (per held-out session and bin of the fitted combination: hits, repeats, false
                    alarms, first showings; for stats_from_trials.py; not distributed)
SEED    0 (numpy default_rng(0) for the 2,000-image distance sample of every space and, separately,
        for the half split of the sessions).
RUNTIME about 15 minutes and 5 GB of memory on an Apple M2 Pro without a GPU (one 26,107 x 26,107
        distance matrix at a time plus the 6.2 million x 70 crowding table).
Run from the repository root:  python A5_things_context_effects_figS13_S14/code/fit_on_sequence.py
    --first-participants N   use only the first N participant sessions (a quick check that the code
                             runs; the numbers are then not those of the full data)

The weighting is fitted to single trials, not to per-image scores. Each of the
fourteen network/layer spaces contributes its own crowding of the probe against
the preceding stream at each falloff, and the combination is fitted to whether
that trial was answered correctly, after centering within image so item identity
contributes nothing. An uncentered fit could learn which images are memorable;
once each image's own mean is removed, the variance left is that of the same
image after different streams. Participants are split in half: the weighting
(and the best single space and falloff) is chosen on one half, and every number
in the report comes from the other half.
"""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

import a5_common

EMB = a5_common.embeddings_folder()
OUT = a5_common.make_regenerated_folder()
DERIVED = a5_common.DERIVED
TRIALS = DERIVED / "trials.npz"

TAUS = [0.5, 1.0, 2.0, 4.0, 8.0]
N_BINS = 5
MIN_PRIOR = 20
SEED = 0


def pairwise_distance(X):
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        torch, dev = None, None
    Xn = X - X.mean(1, keepdims=True)
    Xn /= np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-8)
    N = len(Xn)
    D = np.empty((N, N), np.float32)
    if torch is not None:
        T = torch.from_numpy(Xn).to(dev)
        for a in range(0, N, 2048):
            b = min(a + 2048, N)
            D[a:b] = (1.0 - (T[a:b] @ T.T)).cpu().numpy()
        del T
        if dev == "cuda":
            torch.cuda.empty_cache()
    else:
        for a in range(0, N, 2048):
            b = min(a + 2048, N)
            D[a:b] = 1.0 - (Xn[a:b] @ Xn.T)
    rng = np.random.default_rng(SEED)
    s = rng.choice(N, 2000, replace=False)
    sub = D[np.ix_(s, s)][np.triu_indices(2000, 1)]
    mu, sd = float(sub.mean()), float(sub.std() + 1e-8)
    D -= mu
    D /= sd
    return D, float(np.percentile((sub - mu) / sd, 1))


def stream_context(D, off, sub, img0, taus, want_nprior=False):
    n = len(sub)
    out = np.full((n, len(taus)), np.nan, np.float32)
    n_prior = np.zeros(n, np.int32) if want_nprior else None
    bounds = np.flatnonzero(np.r_[True, sub[1:] != sub[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        ids = img0[a:b]
        m = len(ids)
        if m < 2:
            continue
        M = D[np.ix_(ids, ids)].astype(np.float32)
        same = ids[:, None] == ids[None, :]
        keep = np.tril(np.ones((m, m), bool), -1) & ~same
        if want_nprior:
            n_prior[a:b] = keep.sum(1)
        Mk = np.where(keep, M, np.inf)
        for j, t in enumerate(taus):
            z = -(Mk - off) / t
            z[~keep] = -np.inf
            with np.errstate(invalid="ignore"):
                out[a:b, j] = logsumexp(z, axis=1)
    return (out, n_prior) if want_nprior else out


def centre_within(v, img, valid):
    """Subtract each image's own mean, over the trials that count.

    Removing the image mean is what strips the between-item variance out; what
    is left can only reflect the company the image kept on that trial.
    """
    out = np.full(len(v), np.nan, np.float32)
    idx = np.flatnonzero(valid & np.isfinite(v))
    if len(idx) == 0:
        return out
    im = img[idx]
    s = np.zeros(img.max() + 1)
    c = np.zeros(img.max() + 1)
    np.add.at(s, im, v[idx].astype(float))
    np.add.at(c, im, 1.0)
    out[idx] = v[idx] - (s[im] / np.maximum(c[im], 1))
    return out


def within_image_bin(ctx, img, n_bins=N_BINS):
    out = np.full(len(ctx), -1, np.int8)
    ok = np.isfinite(ctx)
    if not ok.any():
        return out
    order = np.lexsort((ctx[ok], img[ok]))
    idx = np.flatnonzero(ok)[order]
    im = img[idx]
    starts = np.flatnonzero(np.r_[True, im[1:] != im[:-1], True])
    for a, b in zip(starts[:-1], starts[1:]):
        m = b - a
        if m < n_bins:
            continue
        out[idx[a:b]] = np.minimum((np.arange(m) * n_bins) // m, n_bins - 1)
    return out


def sdt(hit, n_old, fa, n_new):
    hr = np.clip(hit / max(n_old, 1), 1e-4, 1 - 1e-4)
    far = np.clip(fa / max(n_new, 1), 1e-4, 1 - 1e-4)
    return norm.ppf(hr) - norm.ppf(far), hr, far


def image_to_embedding_row():
    """img2row[image_nr] = the row of that image in every embedding file (-1 for no image), from the
    embedding_row column of shared/things/things_images.csv; also the file names in row order."""
    images = pd.read_csv(a5_common.things_images_file(), keep_default_na=False, na_values=['NaN'])
    img2row = np.full(int(images.image_nr.max()) + 1, -1, np.int32)
    img2row[images.image_nr.to_numpy()] = images.embedding_row.to_numpy().astype(np.int32)
    filenames_in_row_order = images.sort_values("embedding_row")["filename"].tolist()
    return img2row, filenames_in_row_order


def main():
    parser = argparse.ArgumentParser(description="Crowding in 14 deep-network spaces, fitted on half the sessions")
    parser.add_argument("--first-participants", type=int, default=None,
                        help="use only the first N participant sessions (a quick check that the code runs)")
    args = parser.parse_args()

    img2row, filenames_in_row_order = image_to_embedding_row()

    d = np.load(TRIALS)
    sub = d["sub"].astype(np.int64)
    img0 = img2row[d["image_nr"].astype(np.int64)]
    ttype, perf = d["ttype"], d["perf"]
    keep = img0 >= 0
    if args.first_participants is not None:
        first_sessions = np.unique(sub)[:args.first_participants]
        keep = keep & np.isin(sub, first_sessions)
        print(f"QUICK CHECK: only the first {args.first_participants} participant sessions are used")
    sub, img0, ttype, perf = sub[keep], img0[keep], ttype[keep], perf[keep]

    names = sorted(a5_common.NETWORK_SPACES)
    spaces = [EMB / (name + ".npz") for name in names]
    missing = [a5_common.relative(f) for f in spaces if not f.exists()]
    if missing:
        raise SystemExit("missing deep-network spaces (EXTERNAL_INPUTS.md M3): " + ", ".join(missing))

    fh = open(OUT / "fit_on_sequence.txt", "w", encoding="utf-8")

    def w(s=""):
        print(s, flush=True)
        fh.write(s + "\n")

    w("WEIGHTS AND FALLOFF FITTED TO THE TRIAL DATA")
    w("=" * 96)
    w(f"{len(sub):,} trials, {len(np.unique(sub)):,} participants, "
      f"{len(spaces)} network/layer spaces x {len(TAUS)} falloffs.")
    w("Participants are split in half. The weighting over spaces and the falloff")
    w("are chosen on the fitting half; every number below is from the held-out half.")
    w("Crowding is centered within image before fitting, so the fit cannot learn")
    w("which images are memorable, only what the surrounding stream does.")
    w()

    # ---- per-space crowding features, one space at a time to bound memory ----
    feats = np.empty((len(sub), len(spaces) * len(TAUS)), np.float32)
    n_prior = None
    for i, f in enumerate(spaces):
        space = np.load(f, allow_pickle=True)
        if [str(name) for name in space["filenames"]] != filenames_in_row_order:
            raise SystemExit(f"{a5_common.relative(f)}: rows are not in the order of shared/things/things_images.csv")
        X = space["features"].astype(np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        D, off = pairwise_distance(X)
        if n_prior is None:
            c, n_prior = stream_context(D, off, sub, img0, TAUS, want_nprior=True)
        else:
            c = stream_context(D, off, sub, img0, TAUS)
        feats[:, i * len(TAUS):(i + 1) * len(TAUS)] = c
        del D, c
        print(f"  [{i+1}/{len(spaces)}] {f.stem}", flush=True)

    enough = n_prior >= MIN_PRIOR
    is_hit_trial = (ttype == 2) & enough
    is_fa_trial = np.isin(ttype, [1, 3]) & enough
    used = is_hit_trial | is_fa_trial
    # correct on a repeat means detecting it; correct on a first showing means
    # not calling it old, so one outcome covers both trial kinds
    correct = np.where(is_hit_trial, perf == 11, perf != 13).astype(np.int8)

    rng = np.random.default_rng(SEED)
    subs = np.unique(sub)
    fit_subs = set(rng.permutation(subs)[: len(subs) // 2].tolist())
    in_fit = np.fromiter((s in fit_subs for s in sub), bool, len(sub))

    # ---- center features within image, then fit on the fitting half ----
    C = np.empty_like(feats)
    for j in range(feats.shape[1]):
        C[:, j] = centre_within(feats[:, j], img0, used)
    ok = used & np.isfinite(C).all(1)
    del feats

    fit_m = ok & in_fit
    test_m = ok & ~in_fit
    w(f"  {int(fit_m.sum()):,} fitting trials, {int(test_m.sum()):,} held-out trials")
    w()

    mdl = LogisticRegression(C=1.0, max_iter=1000)
    mdl.fit(C[fit_m], correct[fit_m])
    score = C @ mdl.coef_.ravel()
    # larger score means more likely correct; flip so larger means more crowded
    score = -score

    # ---- held-out d' per bin; the best single space and falloff, chosen on the fitting half ----
    w("-" * 96)
    w("Held-out d-prime across bins of context crowding")
    w("-" * 96)
    w(f"{'measure':>28} " + " ".join(f"bin{b+1:>7}" for b in range(N_BINS)) +
      f"{'  bin1-bin5':>12}")

    results = {}
    variants = {"fitted combination": score}
    # the best single space and tau on the fitting half, for comparison
    best_j, best_gap = None, -np.inf
    for j in range(C.shape[1]):
        b_ = within_image_bin(np.where(fit_m, C[:, j], np.nan), img0)
        g = []
        for k in (0, N_BINS - 1):
            m = b_ == k
            g.append(sdt(int((perf[m & is_hit_trial] == 11).sum()),
                         int((m & is_hit_trial).sum()),
                         int((perf[m & is_fa_trial] == 13).sum()),
                         int((m & is_fa_trial).sum()))[0])
        if g[0] - g[1] > best_gap:
            best_gap, best_j = g[0] - g[1], j
    sp_name = names[best_j // len(TAUS)]
    sp_tau = TAUS[best_j % len(TAUS)]
    variants[f"best single space ({sp_name}, tau {sp_tau:g})"] = C[:, best_j]

    for lab, v in variants.items():
        b_ = within_image_bin(np.where(test_m, v, np.nan), img0)
        dp, hr, fa = [], [], []
        for k in range(N_BINS):
            m = b_ == k
            a, h, f_ = sdt(int((perf[m & is_hit_trial] == 11).sum()),
                           int((m & is_hit_trial).sum()),
                           int((perf[m & is_fa_trial] == 13).sum()),
                           int((m & is_fa_trial).sum()))
            dp.append(a); hr.append(h); fa.append(f_)
        results[lab] = dict(dprime=dp, hit=hr, fa=fa)
        if lab == "fitted combination":
            save_heldout_counts(b_, sub, perf, is_hit_trial, is_fa_trial)
        for nm, vals in (("d-prime", dp), ("hit rate", hr), ("FA rate", fa)):
            w(f"{lab[:22]:>22} {nm:>5} " +
              " ".join(f"{x:>10.3f}" for x in vals) +
              f"{vals[0]-vals[-1]:>12.3f}")
        w()

    # ---- what the fit weighted ----
    coef = mdl.coef_.ravel()
    agg = {}
    for j, c_ in enumerate(coef):
        agg.setdefault(names[j // len(TAUS)], 0.0)
        agg[names[j // len(TAUS)]] += abs(c_)
    tot = sum(agg.values()) or 1.0
    w("-" * 96)
    w("Share of the fitted weighting carried by each space")
    w("-" * 96)
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        w(f"  {k:22s} {100*v/tot:5.1f}%")
    w()
    tau_agg = {t: 0.0 for t in TAUS}
    for j, c_ in enumerate(coef):
        tau_agg[TAUS[j % len(TAUS)]] += abs(c_)
    w("Share carried by each falloff")
    for t, v in tau_agg.items():
        w(f"  tau {t:<6g} {100*v/tot:5.1f}%")
    w()
    w("-" * 96)
    w("READING THIS")
    w("-" * 96)
    w("  bin 1 is the least crowded preceding stream, bin 5 the most crowded, and")
    w("  bins are formed within image, so the same photograph appears in all five.")
    w("  A d-prime drop from bin 1 to bin 5 is caused by the stream, not the item.")
    w("  The best single space and falloff is chosen on the fitting half (the largest")
    w("  bin 1 minus bin 5 d' drop) and scored on the held-out half in the same way as")
    w("  the fitted combination.")
    fh.close()
    json.dump({"results": results,
               "space_share": agg, "tau_share": {str(k): v for k, v in tau_agg.items()},
               "best_single": [sp_name, sp_tau]},
              open(OUT / "fit_on_sequence.json", "w"), indent=2)
    print(f"\nwrote {a5_common.relative(OUT / 'fit_on_sequence.txt')}")


def save_heldout_counts(bins, sub, perf, is_hit_trial, is_fa_trial):
    """Per held-out session and bin of the fitted combination, the counts behind the held-out d' (hits
    and repeats, false alarms and first showings), for the session resampling of
    stats_from_trials.py. Written next to the trial table."""
    binned = bins >= 0
    sessions, session_index = np.unique(sub[binned], return_inverse=True)
    cell = session_index * N_BINS + bins[binned].astype(np.int64)
    size = len(sessions) * N_BINS
    hit_trial = is_hit_trial[binned]
    fa_trial = is_fa_trial[binned]
    hits = np.bincount(cell, weights=(hit_trial & (perf[binned] == 11)).astype(float), minlength=size)
    repeats = np.bincount(cell, weights=hit_trial.astype(float), minlength=size)
    false_alarms = np.bincount(cell, weights=(fa_trial & (perf[binned] == 13)).astype(float), minlength=size)
    first_showings = np.bincount(cell, weights=fa_trial.astype(float), minlength=size)
    np.savez_compressed(DERIVED / "fit_on_sequence_heldout_counts.npz", sessions=sessions,
                        hits=hits.reshape(-1, N_BINS), repeats=repeats.reshape(-1, N_BINS),
                        false_alarms=false_alarms.reshape(-1, N_BINS),
                        first_showings=first_showings.reshape(-1, N_BINS))
    print(f"  wrote {a5_common.relative(DERIVED / 'fit_on_sequence_heldout_counts.npz')} "
          f"({len(sessions):,} held-out sessions)")


if __name__ == "__main__":
    main()
