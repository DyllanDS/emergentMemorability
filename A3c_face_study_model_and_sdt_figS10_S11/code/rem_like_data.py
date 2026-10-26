"""
rem_like_data.py: REM run again after data collection and scored the way the data were scored (SI
Appendix, Section 3, "The Model Predicted Each Preregistered Face Result"; Fig. S10b, gray points).

WHAT IT DOES
The 729-setting grid of rem_final_spec.py without the criterion axis: g x w x u x c x nfeat = 243
settings, 100 simulated participants each (as many as the experiment). At every setting REM's evidence
for the 3,600 simulated test trials is scored two ways:
  direct  d_a read off the evidence (mean and SD of the log odds), as rem_final_spec.py does
  fitted  the evidence turned into eight-step ratings with ONE set of seven criteria for every kind of
          face, placed where they reproduce how often the participants used each rating (313, 495,
          434, 97, 46, 247, 628 and 1,340 of their 3,600 test trials), and the ratings fitted with the
          registered unequal-variance model (sdt_shared_criteria.py), exactly as the data were
For both it saves the effect of the lure set (fitted: "interaction"; positive at 242 of the 243
settings) and the effect of the feature itself (fitted: "equivalence"; positive at 18 of 243).

The fitted scoring needs the MemToolbox2 Python package in shared/memtoolbox2/
(EXTERNAL_INPUTS.md, item M5). In every mode the script first computes everything that does not need
the package (regenerate_model_side()):
  - the participants' rating use, from the data;
  - REM's evidence at all 243 settings (the same seeds) and its direct scores, compared with the saved
    run data/provided/rem_like_data.jsonl;
  - the simulated ratings under the shared criteria, and how closely their use matches the
    participants';
Without the package it then prints MISSING PACKAGE and PROVIDED OUTPUT for the fitted scores, which
come from the saved run; with it, it then refits the 243 settings into outputs/. check_provided.py
counts the 242 and 18 of 243 from the saved run.

INPUTS   04_rare_feature_faces_fig4/data/face_trials.csv (through face_study_common.py); code/rem_final_spec.py
         (evidence() and the grid); data/provided/rem_like_data.jsonl (the saved run, for the comparison)
OUTPUTS  outputs/rem_like_data_model_side.csv and outputs/rem_like_data_model_side_check.txt (always);
         with MemToolbox2 also outputs/rem_like_data.jsonl and outputs/rem_like_data.txt
SEEDS    setting i (0 to 242, in itertools.product order of g, w, u, c, nfeat) draws its simulated
         participants from numpy.random.default_rng(i) (rem_final_spec.evidence); the fit uses seed i
RUNTIME  about 10 s without MemToolbox2; with it, one registered fit per setting (several hours)
RUN      python A3c_face_study_model_and_sdt_figS10_S11/code/rem_like_data.py
         (--model-side-only: the part without MemToolbox2 only, even when the package is there;
         --report-only: the report from outputs/rem_like_data.jsonl, or from the saved run when
         outputs/ has none)

REM scored the way the data are scored: one shared criterion set, then the same fit.

The grid in rem_final_spec.py reads discriminability straight off the log-odds, as moments of
the evidence distribution. The experiment cannot do that. It sees only an eight-point rating,
produced by an observer who brings ONE set of criteria to an intermixed test, and its d_a comes
from fitting an unequal-variance model to those ratings. Two different estimators, and when one
cell's evidence distribution is much wider than another's they need not agree.

So the model's prediction for a contrast the experiment measures has to be put through the
experiment's own measuring instrument. Here REM's evidence is converted to ratings by placing
seven criteria on it (one set, shared across every item type, exactly as the fitted model
assumes), and the resulting ratings are fitted with the identical model and protocol used on
the participants. Only then is the model's "what the feature buys" comparable to the data's.

Criterion placement is fixed by the participants' own response usage: the criteria sit at the
quantiles of the simulated evidence that reproduce the observed marginal frequencies of the
eight ratings. That equates how freely the simulated observer uses each response with how freely
the participants did, so the comparison is not confounded by bias.
"""

import argparse
import csv
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.dont_write_bytecode = True

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rem_final_spec as R                      # noqa: E402  (importing it runs no sweep)
import sdt_shared_criteria as S                 # noqa: E402

OUT = S.OUT                                      # the module's outputs/ folder
RAW = os.path.join(OUT, "rem_like_data.jsonl")
PROVIDED_RAW = os.path.join(R.PATHS['provided'], "rem_like_data.jsonl")

N_SUB_FIT = 100        # match the experiment, so estimator noise is comparable
TRIALS = 36

# How the 36 probes of one simulated participant map onto the four distributions the fitted
# model knows about. The ord pair is not a condition: its studied member is an ordinary studied
# face and its foil is a face nobody studied, which is how the data are coded
# (sdt_shared_criteria.item_type).
CELL_OF = {"rare_old": "rare_old", "rare_foil": "rare_foil",
           "filler_old": "ordinary_old", "ord_old": "ordinary_old",
           "unrelated": None, "ord_foil": None}


def observed_rating_margins():
    """The participants' own use of the eight responses, as cumulative proportions."""
    import face_study_common as F       # each participant's test trials
    subs = F.analysable(F.load())
    r = np.concatenate([[F.rating(t) for t in s["rec"]] for s in subs])
    counts = np.bincount(r, minlength=S.NRATINGS + 1)[1:S.NRATINGS + 1].astype(float)
    return np.cumsum(counts / counts.sum())[:-1]      # 7 interior cut points


def to_datasets(phi, lab, cum):
    """REM evidence -> one participant's {isOld, rating, cell}, under one shared criterion set."""
    ev = np.log(phi)
    # One set of criteria for the whole test, placed on the pooled evidence so that the
    # simulated observer uses the eight responses as often as the participants did.
    crit = np.quantile(ev, cum)
    rating = 1 + np.searchsorted(crit, ev, side="right")

    n = ev.size // TRIALS
    ev_r, lab_r, rat_r = ev.reshape(n, TRIALS), lab.reshape(n, TRIALS), rating.reshape(n, TRIALS)
    out = []
    for i in range(n):
        isOld, cell = [], []
        for k in lab_r[i]:
            name = CELL_OF[k]
            isOld.append(0 if name is None else 1)
            cell.append(S.ANCHOR_CELL if name is None else S.CODE[name])
        out.append({"isOld": np.array(isOld, int), "rating": rat_r[i].astype(int),
                    "cell": np.array(cell, int)})
    return out


def one(job):
    """One parameter combination, scored both ways."""
    i, vals, cum = job
    try:
        pars = dict(zip(["g", "w", "u", "c", "nfeat"], vals))
        phi, lab = R.evidence(seed=i, n_sub=N_SUB_FIT, **pars)
        # (a) the model's own reading: moments of the log-odds
        def da(a, b):
            la, lb = np.log(phi[lab == a]), np.log(phi[lab == b])
            return ((la.mean() - lb.mean())
                    * np.sqrt(2.0 / (la.std(ddof=1) ** 2 + lb.std(ddof=1) ** 2)))
        direct = {"interaction": da("rare_old", "unrelated") - da("rare_old", "rare_foil"),
                  "buys": da("rare_old", "rare_foil") - da("filler_old", "unrelated")}
        # (b) the experiment's reading: ratings under one criterion set, then the same fit
        ds = to_datasets(phi, lab, np.asarray(cum))
        f = S.fit_mle(S.pooled(ds), seed=i)
        fitted = S.contrasts(S.params_to_cells(S.build_model(), f["MaxLikelihoodParams"]))
        return i, pars, {k: float(v) for k, v in direct.items()}, \
            {k: float(v) for k, v in fitted.items()}, None
    except Exception as e:
        return i, None, None, None, "%s: %s" % (type(e).__name__, e)


def regenerate_model_side(combos, cum):
    """Everything of the 243-setting run that does not need MemToolbox2.
    For every setting, REM's evidence (the same seed as the saved run), its direct scores (the same
    computation as the first half of one() above) and the simulated ratings under the shared
    criteria; the direct scores are compared with the saved run, setting by setting."""
    saved_rows = {}
    with open(PROVIDED_RAW, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                saved_rows[row["i"]] = row
    observed_shares = np.diff(np.concatenate([[0.0], cum, [1.0]]))   # the participants' share of each rating
    table = []
    largest_difference = 0.0
    largest_rating_share_gap = 0.0
    for i, vals in enumerate(combos):
        pars = dict(zip(["g", "w", "u", "c", "nfeat"], vals))
        saved = saved_rows[i]
        if saved["pars"] != pars:
            raise SystemExit("data/provided/rem_like_data.jsonl: setting %d has parameters %s, the grid %s"
                             % (i, saved["pars"], pars))
        phi, lab = R.evidence(seed=i, n_sub=N_SUB_FIT, **pars)

        def da(a, b):
            la, lb = np.log(phi[lab == a]), np.log(phi[lab == b])
            return ((la.mean() - lb.mean())
                    * np.sqrt(2.0 / (la.std(ddof=1) ** 2 + lb.std(ddof=1) ** 2)))
        direct = {"interaction": da("rare_old", "unrelated") - da("rare_old", "rare_foil"),
                  "buys": da("rare_old", "rare_foil") - da("filler_old", "unrelated")}

        # the simulated ratings: how often the simulated observer used each of the eight ratings
        datasets = to_datasets(phi, lab, np.asarray(cum))
        all_ratings = np.concatenate([dataset["rating"] for dataset in datasets])
        simulated_shares = np.bincount(all_ratings, minlength=S.NRATINGS + 1)[1:S.NRATINGS + 1] / all_ratings.size
        rating_share_gap = float(np.max(np.abs(simulated_shares - observed_shares)))
        largest_rating_share_gap = max(largest_rating_share_gap, rating_share_gap)

        difference_interaction = abs(float(direct["interaction"]) - saved["direct"]["interaction"])
        difference_buys = abs(float(direct["buys"]) - saved["direct"]["buys"])
        largest_difference = max(largest_difference, difference_interaction, difference_buys)
        table.append({"setting_number": i, "g": pars["g"], "w": pars["w"], "u": pars["u"], "c": pars["c"],
                      "nfeat": pars["nfeat"], "simulated_participants": len(datasets),
                      "direct_lure_set_effect_rerun": float(direct["interaction"]),
                      "direct_lure_set_effect_saved": saved["direct"]["interaction"],
                      "direct_feature_effect_rerun": float(direct["buys"]),
                      "direct_feature_effect_saved": saved["direct"]["buys"],
                      "largest_rating_share_gap": rating_share_gap,
                      "fitted_lure_set_effect_saved": saved["fitted"]["interaction"],
                      "fitted_feature_effect_saved": saved["fitted"]["equivalence"]})

    with open(os.path.join(OUT, "rem_like_data_model_side.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        for row in table:
            writer.writerow({name: (repr(value) if isinstance(value, float) else value) for name, value in row.items()})

    direct_lure = np.array([row["direct_lure_set_effect_rerun"] for row in table])
    direct_feature = np.array([row["direct_feature_effect_rerun"] for row in table])
    lines = ["REM after data collection, the part that does not need MemToolbox2 (computed in every run)",
             "=========================================================================", "",
             "settings: %d (g x w x u x c x nfeat), %d simulated participants each, %d test trials each"
             % (len(table), N_SUB_FIT, TRIALS),
             "participants' rating use, cumulative proportions at the seven criteria: %s"
             % ", ".join("%.4f" % value for value in cum),
             "simulated ratings under the shared criteria: largest gap between a simulated and an observed"
             " rating share, over all settings and ratings: %.4f" % largest_rating_share_gap,
             "",
             "direct scores (read off the log odds), this run:",
             "  lure-set effect:  mean %+.3f  range %+.3f to %+.3f   positive %d of %d"
             % (direct_lure.mean(), direct_lure.min(), direct_lure.max(), int((direct_lure > 0).sum()), len(table)),
             "  feature effect:   mean %+.3f  range %+.3f to %+.3f   positive %d of %d"
             % (direct_feature.mean(), direct_feature.min(), direct_feature.max(),
                int((direct_feature > 0).sum()), len(table)),
             "largest difference from the saved run's direct scores (data/provided/rem_like_data.jsonl): %.2g"
             % largest_difference,
             "the rerun reproduces the saved run's evidence: %s" % ("yes" if largest_difference < 1e-12 else "NO")]
    report = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "rem_like_data_model_side_check.txt"), "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only the first N combinations")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--model-side-only", action="store_true",
                    help="only regenerate_model_side(), even when MemToolbox2 is present")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    cum = observed_rating_margins()
    print("criterion placement from data, cumulative:", np.round(cum, 4).tolist(), flush=True)

    keys = ["g", "w", "u", "c", "nfeat"]          # fa is not used: criteria come from the data
    combos = list(itertools.product(*[R.GRID[k] for k in keys]))
    if args.limit:
        combos = combos[:args.limit]

    # The part of the run that does not need MemToolbox2 always runs and is compared with the saved
    # run (about 10 seconds; numbers_face_study.py reads its check file in every mode).
    regenerate_model_side(combos, cum)
    if S.mt is None or args.model_side_only:
        # Without MemToolbox2 (or with --model-side-only) the fitted scores are the saved run's.
        if S.mt is None:
            S.face_study_common.import_memtoolbox()    # prints MISSING PACKAGE and PROVIDED OUTPUT
        else:
            print("--model-side-only: the 243 fits are not run")
        S.face_study_common.report_provided_files("rem_like_data.py", ["rem_like_data.jsonl", "rem_like_data.txt"])
        return

    done = set()
    if os.path.exists(RAW) and not args.fresh:
        for line in open(RAW, encoding="utf-8"):
            try:
                done.add(json.loads(line)["i"])
            except Exception:
                pass
        print("resuming: %d combinations already done" % len(done), flush=True)
    todo = [(i, v, cum.tolist()) for i, v in enumerate(combos) if i not in done]
    print("%d combinations to run on %d workers" % (len(todo), args.workers), flush=True)

    t0 = time.time()
    with open(RAW, "a", encoding="utf-8") as fh:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for k, (i, pars, direct, fitted, err) in enumerate(ex.map(one, todo, chunksize=1), 1):
                fh.write(json.dumps({"i": i, "pars": pars, "direct": direct,
                                     "fitted": fitted, "err": err}) + "\n")
                fh.flush()
                if k % 10 == 0 or k == len(todo):
                    el = time.time() - t0
                    print("  %d/%d  %.1f min elapsed, ~%.1f min left"
                          % (k, len(todo), el / 60, el / k * (len(todo) - k) / 60), flush=True)
    report()


def report():
    raw_file = RAW
    if not os.path.exists(raw_file):
        # Without a new run in outputs/, the report is made from the saved run in data/provided/.
        raw_file = PROVIDED_RAW
        print("report from the saved run data/provided/rem_like_data.jsonl")
    rows = [json.loads(l) for l in open(raw_file, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("fitted")]
    if not rows:
        print("nothing to report")
        return
    d_buys = np.array([r["direct"]["buys"] for r in rows])
    f_buys = np.array([r["fitted"]["equivalence"] for r in rows])
    d_int = np.array([r["direct"]["interaction"] for r in rows])
    f_int = np.array([r["fitted"]["interaction"] for r in rows])

    lines = ["REM read through the experiment's own instrument", "=" * 60, "",
             "%d parameter combinations, %d simulated participants each. Every combination is" % (len(rows), N_SUB_FIT),
             "scored twice: directly off the log-odds, and by placing ONE shared set of seven",
             "criteria on the evidence and fitting the model the data are fitted with.", "",
             "WHAT THE FEATURE BUYS  d_a(rare vs its own foil) - d_a(ordinary vs never-studied)",
             "  read off the log-odds   mean %+.3f  range %+.3f to %+.3f   positive %d%%"
             % (d_buys.mean(), d_buys.min(), d_buys.max(), round(100 * (d_buys > 0).mean())),
             "  fitted like the data    mean %+.3f  range %+.3f to %+.3f   positive %d%%"
             % (f_buys.mean(), f_buys.min(), f_buys.max(), round(100 * (f_buys > 0).mean())),
             "",
             "INTERACTION",
             "  read off the log-odds   mean %+.3f  range %+.3f to %+.3f   positive %d%%"
             % (d_int.mean(), d_int.min(), d_int.max(), round(100 * (d_int > 0).mean())),
             "  fitted like the data    mean %+.3f  range %+.3f to %+.3f   positive %d%%"
             % (f_int.mean(), f_int.min(), f_int.max(), round(100 * (f_int > 0).mean())),
             ""]
    txt = "\n".join(lines) + "\n"
    open(os.path.join(OUT, "rem_like_data.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report()
    else:
        main()
