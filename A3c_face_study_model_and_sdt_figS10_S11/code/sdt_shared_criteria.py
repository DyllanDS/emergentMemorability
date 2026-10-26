"""
sdt_shared_criteria.py: the preregistered discriminability analysis of the rare-feature face study
(AsPredicted #311640; SI Appendix, Section 3; Fig. 4c and SI Appendix, Fig. S11b).

WHAT IT DOES
An unequal-variance signal detection model is fitted by maximum likelihood to the confidence ratings
of the 100 registered participants, pooled over participants (3,600 test trials). Never-studied
ordinary faces define the axis, N(0, 1); studied ordinary faces, studied rare-feature faces and the
rare faces' never-studied partners (same-feature lures) each get a free mean and SD; all four kinds of
face share one set of seven criteria (eight rating steps: the old/new answer crossed with the 1 to 4
confidence rating). 13 parameters. Fitting protocol (registered): 12 starting points, then restarts
from the best solution until the log likelihood improves by less than 0.01, up to six rounds. From the
fitted means and SDs:
    d_a(A vs B) = (mean_A - mean_B) x sqrt(2 / (SD_A^2 + SD_B^2))
    rare_matched   d_a(rare vs its same-feature lure)       1.00
    rare_unrel     d_a(rare vs never-studied faces)         1.67
    ordinary       d_a(ordinary vs never-studied faces)     1.00
    interaction    rare_unrel - rare_matched = the effect of the lure set (registered prediction 3), 0.67
    equivalence    rare_matched - ordinary = the effect of the feature itself, 0.00
With --boot N it also refits N participant resamples (serially); bootstrap_parallel.py runs the
registered 10,000 resamples in parallel.

The fit needs the MemToolbox2 Python package (UVSDModel, ByCell, MLE) in shared/memtoolbox2/
(EXTERNAL_INPUTS.md, item M5). Without it the script prints MISSING PACKAGE and PROVIDED OUTPUT and
names the saved files that stand in for it: data/provided/point_fit.json (the fitted parameters),
data/provided/sdt_shared_criteria.txt (the registered report, written by bootstrap_parallel.py) and
data/provided/bootstrap_raw.jsonl (the 10,000 resampled fits).

INPUTS   04_rare_feature_faces_fig4/data/face_trials.csv (through face_study_common.py: the 100
         registered participants in load_order)
OUTPUTS  (with MemToolbox2) outputs/sdt_shared_criteria.txt (the report; with --boot 0 the point fit
         only) and outputs/point_fit.json (the fit in the format of data/provided/point_fit.json,
         which check_provided.py compares with the saved fit)
SEEDS    fit_mle(data, seed): the 12 starting points come from numpy.random.default_rng(seed) and the
         restart jitter from default_rng(seed + 1); the point fit uses seed 0. Bootstrap resample i:
         the participants are drawn from one default_rng(0) stream, resample after resample, and the
         refit uses seed i, starting from the full-sample solution.
RUNTIME  about 1.5 minutes for the point fit; about a minute of computing per bootstrap refit (the
         10,000 take 7.4 hours on 22 parallel workers)
RUN      python A3c_face_study_model_and_sdt_figS10_S11/code/sdt_shared_criteria.py --boot 0

One familiarity axis, one criterion set, four kinds of face on it.

The test is intermixed, so a participant cannot tell a rare foil from a face nobody studied
until they have looked at it. They therefore bring ONE set of criteria to the whole test,
and fitting each kind of item its own criteria describes an observer who does not exist.

Four item types. Never-studied ordinary faces anchor the axis at N(0, 1); studied ordinary
faces, studied rare faces and rare foils each get a free mean and SD on it. Coding every
non-anchor type as "old" is what lets the stock UVSD likelihood do this: the model's new-item
branch is the fixed anchor, and its old-item branch reads the mean and SD of whichever cell
the trial belongs to. Discriminability between any two types is then

    d_a(A vs B) = (mu_A - mu_B) * sqrt(2 / (sigma_A^2 + sigma_B^2))

which reduces to the usual d_a when B is the anchor. Two contrasts are registered: the
interaction, d_a(rare vs anchor) - d_a(rare vs its own foil), which scores the same faces
two ways; and what the feature buys, d_a(rare vs its own foil) - d_a(ordinary vs anchor).

Participant-level estimation is not available here. Three studied and three foil trials per
participant cannot identify this model within a participant, so the model is fitted to pooled
trials and every interval comes from a bootstrap that resamples participants.

    python A3c_face_study_model_and_sdt_figS10_S11/code/sdt_shared_criteria.py
        the point fit and 400 participant resamples, refitted one after another (--boot 400, the default)
    python A3c_face_study_model_and_sdt_figS10_S11/code/sdt_shared_criteria.py --boot 0
        the point fit only
"""
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import face_study_common                                   # noqa: E402
mt = face_study_common.import_memtoolbox(quiet=True)       # None while MemToolbox2 is not in shared/memtoolbox2/

# face_study_common gives each participant's test trials (imported as F).
import face_study_common as F                              # noqa: E402

OUT = face_study_common.release_paths.module_paths(__file__)['outputs']
NRATINGS = 8

# The item types free to sit anywhere on the axis; never-studied ordinary faces are the
# anchor and are not among them. The design's "ordinary" pair is not a separate condition: its
# studied members are ordinary studied faces and its foils are faces nobody studied.
CELL_NAMES = ["ordinary_old", "rare_old", "rare_foil"]
CELLS = list(range(len(CELL_NAMES)))
CODE = {k: i for i, k in enumerate(CELL_NAMES)}
ANCHOR_CELL = CODE["ordinary_old"]   # label only; a new-item trial's cell never enters its likelihood


def item_type(t):
    """Which distribution this test trial is a draw from, or None if unused.

    Four types: never-studied ordinary faces anchor the axis, and studied ordinary faces,
    studied rare faces and rare foils are placed on it. The condition label ("filler" or
    "ord") does not enter, only whether the face was studied and whether it carries a rare
    feature, which is the only distinction the design makes.
    """
    c, tt = t["condition"], t["test_type"]
    if c == "rare":
        return "rare_old" if tt == "old" else "rare_foil"
    if c in ("filler", "ord"):
        return "ordinary_old" if tt == "old" else "unrelated"
    return None


def subject_data(rec):
    """One participant's 36 test trials as {isOld, rating, cell}."""
    isOld, rat, cell = [], [], []
    for t in rec:
        k = item_type(t)
        if k is None:
            continue
        isOld.append(0 if k == "unrelated" else 1)
        rat.append(F.rating(t))
        cell.append(ANCHOR_CELL if k == "unrelated" else CODE[k])
    return {"isOld": np.array(isOld, int), "rating": np.array(rat, int),
            "cell": np.array(cell, int)}


def pooled(datasets):
    return {k: np.concatenate([d[k] for d in datasets]) for k in ("isOld", "rating", "cell")}


N_STARTS = 12


def build_model(seed=0):
    """UVSD with per-cell mean and SD, and ONE criterion set across all cells.

    Thirteen parameters make the likelihood surface bumpy enough that the stock three
    starting points are not enough, and a fit that lands in a local optimum can move a d_a
    by a quarter of a unit. The fitting protocol is therefore fixed in advance rather than
    discovered per dataset.
    """
    base = mt.UVSDModel(nRatings=NRATINGS)
    spec = {p: ("percell" if p in ("dprime", "sigma") else "shared") for p in base.ParamNames}
    model = mt.ByCell(base, spec, Factors={"cell": CELLS})
    lo = np.asarray(model.LowerBound, float)
    hi = np.asarray(model.UpperBound, float)
    rng = np.random.default_rng(seed)
    starts = [np.asarray(r, float) for r in np.atleast_2d(model.Start)]
    while len(starts) < N_STARTS:
        base_row = starts[rng.integers(0, len(np.atleast_2d(model.Start)))]
        jit = np.clip(base_row * np.exp(rng.normal(0, 0.4, base_row.size)), lo + 1e-6, hi - 1e-6)
        starts.append(jit)
    model.Start = np.vstack(starts)
    return model


def fit_mle(data, seed=0, rounds=6, tol=0.01, warm=None):
    """Fit, then restart from the best solution found so far until it stops improving.

    The likelihood surface has local optima far enough apart to matter. Random restarts
    alone are not a protocol either, since how close they get depends on how many you can
    afford. Restarting from the incumbent converges to the same solution from independent
    seeds, which is what makes the estimate a property of the data rather than of the
    optimizer.
    """
    if warm is None:
        best = mt.MLE(data, build_model(seed))
    else:
        # A bootstrap resample differs from the full sample only by resampling, so the
        # full-sample solution is a good neighborhood to start in; the refinement loop
        # below still has to earn its way to the resample's own optimum.
        model = build_model(seed)
        lo = np.asarray(model.LowerBound, float)
        hi = np.asarray(model.UpperBound, float)
        w = np.asarray(warm, float)
        r0 = np.random.default_rng(seed)
        model.Start = np.vstack([w] + [np.clip(w * np.exp(r0.normal(0, 0.2, w.size)),
                                               lo + 1e-6, hi - 1e-6) for _ in range(3)])
        best = mt.MLE(data, model)
    rng = np.random.default_rng(seed + 1)
    for r in range(rounds):
        model = build_model(seed + 100 + r)
        lo = np.asarray(model.LowerBound, float)
        hi = np.asarray(model.UpperBound, float)
        p = np.asarray(best["MaxLikelihoodParams"], float)
        rows = [p] + [np.clip(p * np.exp(rng.normal(0, 0.25, p.size)), lo + 1e-6, hi - 1e-6)
                      for _ in range(3)]
        model.Start = np.vstack(rows)
        f = mt.MLE(data, model)
        if f["LogLike"] > best["LogLike"] + tol:
            best = f
        else:
            break
    return best


def da(muA, sdA, muB, sdB):
    return (muA - muB) * np.sqrt(2.0 / (sdA ** 2 + sdB ** 2))


def contrasts(par):
    """par maps item type -> (mu, sigma). The anchor is N(0, 1)."""
    g = lambda c: par[c]
    out = {}
    out["rare_matched"] = da(*g("rare_old"), *g("rare_foil"))    # rare vs its own foil
    out["rare_unrel"] = da(*g("rare_old"), 0.0, 1.0)             # rare vs never-studied
    out["ordinary"] = da(*g("ordinary_old"), 0.0, 1.0)           # ordinary vs never-studied
    # The registered interaction: the same rare faces scored two ways.
    out["interaction"] = out["rare_unrel"] - out["rare_matched"]
    # The registered equivalence contrast: does the rare feature buy any discriminability
    # once each kind of face is scored against its own kind of lure?
    out["equivalence"] = out["rare_matched"] - out["ordinary"]
    return out


def params_to_cells(model, values):
    names = list(model.ParamNames)
    v = dict(zip(names, np.asarray(values, float)))
    par = {}
    for code, name in enumerate(CELL_NAMES):
        suf = "_cell%d" % code
        mu = next(v[nm] for nm in names if nm.startswith("dprime") and nm.endswith(suf))
        sd = next(v[nm] for nm in names if nm.startswith("sigma") and nm.endswith(suf))
        par[name] = (mu, sd)
    return par


def save_point_fit(model, fit, path):
    """The point fit in the format of data/provided/point_fit.json (cells: mean and SD of each
    kind of face; params: every parameter by name; loglike; contrasts)."""
    values = np.asarray(fit["MaxLikelihoodParams"], float)
    cells = params_to_cells(model, values)
    point = contrasts(cells)
    saved = {"cells": {name: [float(mean), float(sd)] for name, (mean, sd) in cells.items()},
             "params": {name: float(value) for name, value in zip(list(model.ParamNames), values)},
             "loglike": float(fit["LogLike"]),
             "contrasts": {name: float(value) for name, value in point.items()}}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(saved, handle, indent=1)
    print("saved the point fit to outputs/%s" % os.path.basename(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if mt is None:
        # Without MemToolbox2 the fit cannot run: say so and name the saved results.
        face_study_common.import_memtoolbox()          # prints MISSING PACKAGE and PROVIDED OUTPUT
        face_study_common.report_provided_files("sdt_shared_criteria.py",
                                              ["point_fit.json", "sdt_shared_criteria.txt", "bootstrap_raw.jsonl"])
        return

    subs = [subject_data(s["rec"]) for s in F.analysable(F.load())]
    n = len(subs)
    model = build_model()
    print("%d participants, %d trials each, %d parameters"
          % (n, subs[0]["isOld"].size, len(model.ParamNames)))
    print("params:", list(model.ParamNames))

    # ---- 1. frequentist: one fit to the pooled trials, bootstrap over participants ----
    t0 = time.time()
    fit = fit_mle(pooled(subs))
    point = contrasts(params_to_cells(model, fit["MaxLikelihoodParams"]))
    cells = params_to_cells(model, fit["MaxLikelihoodParams"])
    print("point fit in %.1f s" % (time.time() - t0))
    save_point_fit(model, fit, os.path.join(OUT, "point_fit.json"))

    boot = {k: [] for k in point}
    if args.boot:
        rng = np.random.default_rng(0)
        t0 = time.time()
        for i in range(args.boot):
            d = [subs[j] for j in rng.integers(0, n, n)]
            try:
                f = fit_mle(pooled(d), seed=i, warm=fit["MaxLikelihoodParams"])
                c = contrasts(params_to_cells(model, f["MaxLikelihoodParams"]))
            except Exception:
                c = {k: np.nan for k in point}
            for k, val in c.items():
                boot[k].append(val)
        print("%d bootstrap fits in %.1f s" % (args.boot, time.time() - t0))

    def ci(k):
        v = np.array(boot[k], float)
        v = v[np.isfinite(v)]
        return np.percentile(v, [2.5, 97.5]) if v.size else (np.nan, np.nan)

    # ---- 2. report ----
    path = os.path.join(OUT, "sdt_shared_criteria.txt")
    with open(path, "w", encoding="utf-8") as fh:
        W = fh.write
        W("d_a with ONE criterion set across the intermixed test\n")
        W("=" * 74 + "\n\n")
        W("%d participants, %d test trials each. Never-studied unrelated faces anchor the\n"
          "axis at N(0, 1); studied ordinary faces, studied rare faces and rare foils are\n"
          "placed on it, and all four share one set of %d criteria, because the test is\n"
          "intermixed and a participant cannot know which kind of item is in front of them.\n\n"
          % (n, subs[0]["isOld"].size, NRATINGS - 1))

        W("WHERE EACH ITEM TYPE SITS\n")
        W("  %-14s %8s %8s\n" % ("item type", "mean", "SD"))
        W("  %-14s %8.3f %8.3f   (anchor)\n" % ("unrelated", 0.0, 1.0))
        for c in CELL_NAMES:
            W("  %-14s %8.3f %8.3f\n" % (c, cells[c][0], cells[c][1]))
        W("\n")

        # The interval and p need bootstrap resamples; with --boot 0 each row gives the estimate only.
        W("DISCRIMINABILITY (d_a), frequentist, %d participant bootstrap resamples\n" % args.boot)
        for k, lab in [("rare_matched", "rare vs its own foil"),
                       ("rare_unrel", "rare vs never-studied"),
                       ("ordinary", "ordinary studied vs never-studied")]:
            if args.boot > 0:
                lo, hi = ci(k)
                W("  %-26s %7.3f  [%+.3f, %+.3f]\n" % (lab, point[k], lo, hi))
            else:
                W("  %-26s %7.3f\n" % (lab, point[k]))
        W("\n")
        for k, lab in [("interaction", "INTERACTION (rare: never-studied minus own foil)"),
                       ("equivalence", "EQUIVALENCE (rare own-foil minus ordinary)")]:
            if args.boot > 0:
                lo, hi = ci(k)
                v = np.array(boot[k], float); v = v[np.isfinite(v)]
                p2 = 2 * min(np.mean(v <= 0), np.mean(v >= 0)) if v.size else np.nan
                W("  %-34s %+.3f  [%+.3f, %+.3f]  two-sided p = %.4f\n"
                  % (lab, point[k], lo, hi, p2))
            else:
                W("  %-34s %+.3f\n" % (lab, point[k]))
        W("\n")

    print(open(path, encoding="utf-8").read())


if __name__ == "__main__":
    main()
