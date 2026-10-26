"""
fit_inversion.py: the continuous-feature REM run through the memorability inversion experiment
(SI Appendix, Section 3; Fig. S8a).

WHAT IT DOES
  Each THINGS image is a point in a 49-dimensional space: the late-layer activations of AlexNet,
  VGG16 (torchvision vgg16_bn), ResNet18 and CLIP ViT-B/32 for all 26,107 THINGS photographs (512
  values per layer, principal components except for ResNet18, whose 512 values are used as they are),
  every column standardized, the four put side by side (2,048 columns) and reduced to 49 principal
  components (scikit-learn PCA, random_state 0). For every block of every participant, the model
  stores the 40 images of the study list (each feature stored with probability 0.30, copied correctly
  with probability 0.60 and then given Gaussian noise, otherwise replaced by the value of a random
  THINGS image without noise), then judges each test image of that block by its familiarity: the
  mean, over the 40 traces, of the product of the likelihood ratios over the trace's stored features,
  with the rarity of a feature value judged against that feature's distribution over THINGS. 120
  simulated participants per block.

  Two settings are calibrated to two numbers, the experiment's overall hit rate and overall false
  alarm rate: sigma (the noise on a stored feature, a fraction of each feature's SD, from a grid) and
  the criterion (set so that the model's false alarm rate equals the observed one). The sigma whose
  hit rate comes closest to the observed hit rate is kept. Every item-level number the model produces
  after that is a prediction (which item was remembered, which lure was falsely recognized, and how
  memorability relates to the published THINGS scores). Familiarity is computed once per sigma, so
  the criterion is chosen afterwards and the sweep is one pass over the data.

INPUTS (inside the repository)
  shared/embeddings_things/alexnet_late.npz, vgg16_late.npz, resnet18_late.npz, clip_late.npz
      (26,107 x 512 each)
  shared/things/things_images.csv (the file name of each embedding row: sorted by embedding_row it
      gives the row order of the embedding files)
  outputs/model_inputs/inversion_study.csv, inversion_test.csv, inversion_observed_items.csv
      (written by code/extract_inversion.py: the 50 participants of Fig. 3e)

OUTPUTS (in outputs/)
  fit_inversion.txt        the sigma sweep and the operating point
  fit_inversion_items.csv  the 40 critical items: observed hitRate, faRate, cr, nOld, nNew, origMem
                           and the model's pred_hit, pred_fa, pred_cr at the chosen sigma
  The script also prints the share of the variance that the 49 components keep.

SEED 0: the PCA (scikit-learn random_state 0) and, with numpy default_rng(0), the 6,000 images of the
  density estimates and the simulated participants (a fresh generator with seed 0 for every sigma,
  drawn block by block in the order of participant and block; within a block, in order: the stored
  mask, the correct-copy mask, the Gaussian noise, then for each feature the random THINGS images for
  the wrongly copied values).

RUN (from the repository root; about 2 minutes):  python A3a_model_through_own_lists_figS8/code/fit_inversion.py
"""
import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, norm, gaussian_kde
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (first: it stops Python from writing __pycache__ folders)

PATHS = release_paths.module_paths(__file__)
EMB = Path(release_paths.shared_path('embeddings_things'))            # the four late-layer embedding files
IMAGES = Path(release_paths.shared_path('things', 'things_images.csv'))
OUT = Path(PATHS['outputs'])
MODEL_INPUTS = OUT / "model_inputs"
OBS = MODEL_INPUTS / "inversion_observed_items.csv"

SELECTION = ["alexnet_late", "vgg16_late", "resnet18_late", "clip_late"]
NDIM = 49
U, C_ENC = 0.30, 0.60            # storage probability per feature; probability a stored feature is copied correctly
NSUBJ = 120                      # simulated participants per subject-block
SIGMAS = [0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.6]     # noise grid, as a fraction of each feature's SD
SEED = 0


def build_space():
    """The 49-d space: standardize each late layer, put the four side by side, PCA to 49."""
    mats = []
    for nm in SELECTION:
        X = np.load(EMB / f"{nm}.npz", allow_pickle=True)["features"].astype(np.float32)
        mats.append((X - X.mean(0)) / (X.std(0) + 1e-8))
    pca = PCA(n_components=NDIM, random_state=SEED)
    Z = pca.fit_transform(np.concatenate(mats, axis=1)).astype(np.float32)
    print(f"PCA of the {sum(m.shape[1] for m in mats):,} standardized columns: {NDIM} components keep "
          f"{pca.explained_variance_ratio_.sum():.4f} of the variance", flush=True)
    return Z


def marginals(Z, ngrid=512):
    """Each feature's distribution over THINGS: a Gaussian kernel density (bandwidth factor 0.2)
    from 6,000 randomly chosen images, tabulated on 512 points."""
    rng = np.random.default_rng(SEED)
    samp = rng.choice(len(Z), 6000, replace=False)
    grids, dens = [], []
    for j in range(Z.shape[1]):
        col = Z[samp, j]
        kde = gaussian_kde(col, bw_method=0.2)
        lo, hi = col.min(), col.max()
        pad = 0.15 * (hi - lo)
        g = np.linspace(lo - pad, hi + pad, ngrid)
        grids.append(g)
        dens.append(np.maximum(kde(g), 1e-9))
    return np.array(grids), np.array(dens)


def block_familiarity(Z, st, probes_idx, grids, dens, sigma, rng):
    """Mean likelihood ratio of each probe against one participant's traces."""
    # The model's variables (the README's "What the code does" section gives the same rule in words):
    #   st          the images on this participant's study list (rows of Z); n of them, w features each
    #   stored      for each of the NSUBJ simulated participants, which features of each studied image
    #               were stored (probability U)
    #   correct     which stored features were copied correctly (probability C_ENC)
    #   T           the traces: the image's value plus Gaussian noise (SD sigma, per feature); a feature
    #               copied wrongly takes that feature's value of a THINGS image drawn at random
    #   P           the test images (probes)
    #   fT          the density of each trace value in the THINGS population (grids, dens: marginals())
    #   kern        the Gaussian density of the trace value around the probe's value (SD sigma)
    #   logL        the log likelihood ratio of each probe against each trace: the sum over the stored
    #               features of log(1 + C_ENC (kern / fT - 1)), each factor at least 1e-12
    #   result      familiarity = the mean over the studied traces of exp(logL), per probe and
    #               simulated participant
    n = len(st)
    w = Z.shape[1]
    stored = rng.random((NSUBJ, n, w)) < U
    correct = rng.random((NSUBJ, n, w)) < C_ENC
    T = np.repeat(Z[st][None], NSUBJ, axis=0) + rng.standard_normal((NSUBJ, n, w)) * sigma
    # a feature that was stored incorrectly carries a value from the population
    for j in range(w):
        bad = ~correct[:, :, j]
        if bad.any():
            T[:, :, j][bad] = Z[rng.integers(0, len(Z), int(bad.sum())), j]
    P = Z[probes_idx]
    logL = np.zeros((len(P), NSUBJ, n))
    for j in range(w):
        t = T[:, :, j]
        fT = np.maximum(np.interp(t, grids[j], dens[j],
                                  left=dens[j][0], right=dens[j][-1]), 1e-9)
        d = t[None] - P[:, j][:, None, None]
        kern = np.exp(-0.5 * (d / sigma[j]) ** 2) / (sigma[j] * np.sqrt(2 * np.pi))
        logL += np.log(np.maximum(1.0 + C_ENC * (kern / fT[None] - 1.0), 1e-12)) \
            * stored[:, :, j][None]
    return np.exp(logL).mean(axis=2)          # nP x NSUBJ


def main():
    # embedding row of every image file name (things_images.csv sorted by embedding_row is the row
    # order of the embedding files)
    order = pd.read_csv(IMAGES).sort_values("embedding_row")
    row = pd.Series(np.arange(len(order)), index=order.filename)
    study = pd.read_csv(MODEL_INPUTS / "inversion_study.csv")
    test = pd.read_csv(MODEL_INPUTS / "inversion_test.csv")
    obs = pd.read_csv(OBS, float_precision="round_trip")

    study["r"] = study.img.map(row).astype(int)
    test["r"] = test.img.map(row).astype(int)

    # the operating point the model has to reproduce
    tgt_hit = test[test.is_old == 1].said_old.mean()
    tgt_fa = test[test.is_old == 0].said_old.mean()
    za = lambda v: norm.ppf(np.clip(v, 1e-3, 1 - 1e-3))
    print(f"observed: hit {tgt_hit:.3f}, FA {tgt_fa:.3f}, d' {za(tgt_hit)-za(tgt_fa):.2f}",
          flush=True)

    Z = build_space()
    grids, dens = marginals(Z)
    sd = Z.std(0)
    print(f"space: {Z.shape[0]:,} images x {Z.shape[1]} dims", flush=True)

    blocks = list(study.groupby(["sub", "block"]))
    lines = ["CONTINUOUS REM AT THE PARTICIPANTS' OPERATING POINT",
             "=" * 96,
             f"{len(blocks)} subject-blocks x {NSUBJ} simulated participants.",
             f"space: {' + '.join(SELECTION)} reduced to {NDIM} dimensions.",
             f"storage {U}, stored-correctly {C_ENC}.",
             f"observed hit {tgt_hit:.3f}, FA {tgt_fa:.3f}, "
             f"d' {za(tgt_hit)-za(tgt_fa):.2f}",
             "sigma and the criterion are set from those two numbers; every",
             "item-level correlation below is a prediction.", "",
             f"{'sigma':>6} {'crit':>9} {'hit':>7} {'FA':>7} {'d-prime':>8}"
             f"{'  vs obs CR':>12}{'  vs obs hit':>13}{'  vs obs FA':>12}"
             f"{'  vs origMem':>13}"]
    print(lines[-1], flush=True)

    best = None
    for frac in SIGMAS:
        sigma = (frac * sd).astype(np.float32)
        rng = np.random.default_rng(SEED)
        fams, imgs, olds = [], [], []
        for (sub, blk), g in blocks:
            tt = test[(test["sub"] == sub) & (test.block == blk)]
            if not len(tt):
                continue
            f = block_familiarity(Z, g.r.to_numpy(), tt.r.to_numpy(),
                                  grids, dens, sigma, rng)
            fams.append(f)
            imgs.append(np.repeat(tt.img.to_numpy()[:, None], NSUBJ, axis=1))
            olds.append(np.repeat(tt.is_old.to_numpy()[:, None], NSUBJ, axis=1))
        # F: the familiarity of every test trial of every simulated participant; I: its image; O: True
        # when the image was on that block's study list (an old item). mh and mf below are the model's
        # hit rate and false alarm rate of each image; d holds the observed and predicted rates.
        F = np.concatenate([x.ravel() for x in fams])
        I = np.concatenate([x.ravel() for x in imgs])
        O = np.concatenate([x.ravel() for x in olds]).astype(bool)

        # criterion that reproduces the observed false alarm rate
        crit = np.quantile(F[~O], 1 - tgt_fa)
        said = F > crit
        hit, fa = said[O].mean(), said[~O].mean()

        dfm = pd.DataFrame({"img": I, "old": O, "said": said})
        mh = dfm[dfm.old].groupby("img").said.mean()
        mf = dfm[~dfm.old].groupby("img").said.mean()
        d = obs.copy()
        d["pred_hit"] = d.imageName.map(mh)
        d["pred_fa"] = d.imageName.map(mf)
        d["pred_cr"] = d.pred_hit - d.pred_fa
        ok = d.pred_cr.notna()
        r_cr = spearmanr(d.loc[ok, "pred_cr"], d.loc[ok, "cr"]).statistic
        r_h = spearmanr(d.loc[ok, "pred_hit"], d.loc[ok, "hitRate"]).statistic
        r_f = spearmanr(d.loc[ok, "pred_fa"], d.loc[ok, "faRate"]).statistic
        r_o = spearmanr(d.loc[ok, "pred_cr"], d.loc[ok, "origMem"]).statistic
        s = (f"{frac:>6.2f} {crit:>9.3g} {hit:>7.3f} {fa:>7.3f} "
             f"{za(hit)-za(fa):>8.2f}{r_cr:>12.3f}{r_h:>13.3f}{r_f:>12.3f}{r_o:>13.3f}")
        lines.append(s)
        print(s, flush=True)
        if best is None or abs(hit - tgt_hit) < best[0]:
            best = (abs(hit - tgt_hit), frac, d.copy(), (hit, fa, crit))

    _, frac, d, (hit, fa, crit) = best
    obs_o = spearmanr(d.cr, d.origMem).statistic
    lines += ["",
              f"closest to the observed operating point: sigma {frac:g}, "
              f"criterion {crit:.3g}, hit {hit:.3f}, FA {fa:.3f}",
              f"observed CR vs original THINGS memorability: {obs_o:+.3f}",
              f"model CR    vs original THINGS memorability: "
              f"{spearmanr(d.pred_cr, d.origMem).statistic:+.3f}",
              "", "-" * 96, "READING THIS", "-" * 96,
              "  In the experiment, the more memorable a critical item was by its published",
              "  score, the more items similar to it were on its block's study list (from 0",
              "  items in the lowest quintile to 8 in the highest), so observed memorability",
              "  correlates negatively with the published scores. The model has no item-level",
              "  intrinsic memorability term, so the same negative sign in its predictions",
              "  comes from the composition of the study lists. Only the two overall rates (hit",
              "  rate and false alarm rate) set sigma and the criterion; every item-level column",
              "  is a prediction. vs obs CR, hit and FA: Spearman rho over the 40 critical items,",
              "  model against observed; vs origMem: the model's CR against the published score."]
    (OUT / "fit_inversion.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    d.to_csv(OUT / "fit_inversion_items.csv", index=False)
    print(f"\nwrote outputs/fit_inversion.txt and outputs/fit_inversion_items.csv")


if __name__ == "__main__":
    main()
