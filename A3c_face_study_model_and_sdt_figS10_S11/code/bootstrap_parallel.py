"""
bootstrap_parallel.py: the registered participant bootstrap of the face study's d_a analysis, 10,000
resamples, each refitting the unequal-variance model (SI Appendix, Section 3; Fig. 4c and SI
Appendix, Figs. S10b and S11b, the intervals over participants).

WHAT IT DOES
Fits the registered model to all 100 participants (sdt_shared_criteria.py), then draws 10,000
resamples of the 100 participants with replacement and refits the model to each, starting from the
full-sample solution. Each resample's five d_a quantities are appended to a JSONL file as they finish
(an interrupted run resumes). Finally it writes the registered report: the three d_a values with the
2.5th and 97.5th percentiles of the resampled values, and for the two registered contrasts (the
effect of the lure set, "interaction"; the effect of the feature itself, "equivalence") the two-sided
bootstrap p, twice the smaller of the share of resamples at or below zero and the share at or above
zero (the definition in the preregistration).

run_module.py does not run this script (7.4 hours on 22 parallel workers). The saved resamples are
data/provided/bootstrap_raw.jsonl (one line per resample: its number i and the five fitted d_a
quantities; 10,000 lines, no failed refit) and the saved report is
data/provided/sdt_shared_criteria.txt. check_provided.py recomputes every interval and p of that
report from the saved resamples.

The refits need the MemToolbox2 Python package in shared/memtoolbox2/ (EXTERNAL_INPUTS.md,
item M5). Without it the script prints MISSING PACKAGE and PROVIDED OUTPUT and names the saved files.

INPUTS   04_rare_feature_faces_fig4/data/face_trials.csv (through face_study_common.py)
OUTPUTS  (with MemToolbox2) outputs/bootstrap_raw.jsonl and outputs/sdt_shared_criteria.txt; the
         saved files in data/provided/ are never written to
SEEDS    resample i: participant indices from one numpy.random.default_rng(0) stream, drawn in the order
         of the serial loop (rng.integers(0, 100, 100) for i = 0, 1, ...); refit with seed i
RUNTIME  7.4 hours on 22 workers; with the default of 2 workers, about 3.4 days
RUN      python A3c_face_study_model_and_sdt_figS10_S11/code/bootstrap_parallel.py --workers 2
         (--fresh ignores resamples already in outputs/bootstrap_raw.jsonl)

The registered 10,000-resample bootstrap, spread across cores.

The registered protocol lives in sdt_shared_criteria.py and is imported from there, not
restated: same model, same fit_mle, same warm start from the full-sample solution, same
seed per resample. The only thing that changes is which process does the arithmetic.

Equivalence to the serial loop is what makes that safe. The serial version draws resample i
from one rng stream in order, so the draws are made here in that same order, up front,
and only then handed out. Resample i therefore sees the participants it would have seen
serially and is fitted with the seed it would have had.

Results are appended to a JSONL file as they land, so an interrupted run resumes rather than
restarting: a refit takes about a minute and there are ten thousand of them.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.dont_write_bytecode = True

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sdt_shared_criteria as S          # noqa: E402

OUT = S.OUT                               # the module's outputs/ folder
RAW = os.path.join(OUT, "bootstrap_raw.jsonl")

_SUBS = None      # per-worker state, built once per process rather than per resample
_WARM = None


def _init(subs, warm):
    global _SUBS, _WARM
    _SUBS, _WARM = subs, warm


def _one(job):
    """One resample: refit the registered model to a participant-level resample."""
    i, idx = job
    try:
        d = [_SUBS[j] for j in idx]
        f = S.fit_mle(S.pooled(d), seed=i, warm=_WARM)
        cells = S.params_to_cells(S.build_model(), f["MaxLikelihoodParams"])
        c = S.contrasts(cells)
        return i, {k: float(v) for k, v in c.items()}, float(f["LogLike"])
    except Exception as e:
        return i, None, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fresh", action="store_true", help="ignore any resamples already on disk")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if S.mt is None:
        # Without MemToolbox2 the refits cannot run: name the saved results instead.
        S.face_study_common.import_memtoolbox()        # prints MISSING PACKAGE and PROVIDED OUTPUT
        S.face_study_common.report_provided_files("bootstrap_parallel.py",
                                                ["bootstrap_raw.jsonl", "sdt_shared_criteria.txt"])
        return

    # ---- 1. the sample and the full-sample fit both bootstrap resamples warm-start from ----
    subs = [S.subject_data(s["rec"]) for s in S.F.analysable(S.F.load())]
    n = len(subs)
    print("%d participants" % n, flush=True)
    t0 = time.time()
    fit = S.fit_mle(S.pooled(subs))
    warm = fit["MaxLikelihoodParams"]
    point = S.contrasts(S.params_to_cells(S.build_model(), warm))
    print("point fit in %.1f s: %s" % (time.time() - t0,
                                       {k: round(v, 3) for k, v in point.items()}), flush=True)

    # ---- 2. the resample draws, in the order the serial loop would have drawn them ----
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, n, n) for _ in range(args.boot)]

    done = {}
    if os.path.exists(RAW) and not args.fresh:
        for line in open(RAW, encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("c"):
                    done[r["i"]] = r["c"]
            except Exception:
                continue
        print("resuming: %d resamples already on disk" % len(done), flush=True)
    todo = [(i, draws[i].tolist()) for i in range(args.boot) if i not in done]

    # ---- 3. fan the refits out ----
    t0, failed = time.time(), 0
    with open(RAW, "a", encoding="utf-8") as fh:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init, initargs=(subs, warm)) as ex:
            for k, (i, c, info) in enumerate(ex.map(_one, todo, chunksize=1), 1):
                if c is None:
                    failed += 1
                else:
                    done[i] = c
                fh.write(json.dumps({"i": i, "c": c}) + "\n")
                if k % 100 == 0:
                    fh.flush()
                    el = time.time() - t0
                    print("  %d/%d  %.1f min elapsed, ~%.1f min left, %d failed"
                          % (k, len(todo), el / 60, el / k * (len(todo) - k) / 60, failed),
                          flush=True)
    print("%d resamples in %.1f min, %d failed" % (len(todo), (time.time() - t0) / 60, failed),
          flush=True)

    # ---- 4. the registered report ----
    keys = list(point)
    boot = {k: np.array([done[i][k] for i in sorted(done) if np.isfinite(done[i][k])], float)
            for k in keys}
    path = os.path.join(OUT, "sdt_shared_criteria.txt")
    cells = S.params_to_cells(S.build_model(), warm)
    with open(path, "w", encoding="utf-8") as fh:
        W = fh.write
        W("d_a with ONE criterion set across the intermixed test\n")
        W("=" * 74 + "\n\n")
        W("%d participants, %d test trials each. Never-studied unrelated faces anchor the\n"
          "axis at N(0, 1); studied ordinary faces, studied rare faces and rare foils are\n"
          "placed on it, and all four share one set of %d criteria, because the test is\n"
          "intermixed and a participant cannot know which kind of item is in front of them.\n\n"
          % (n, subs[0]["isOld"].size, S.NRATINGS - 1))
        W("WHERE EACH ITEM TYPE SITS\n")
        W("  %-14s %8s %8s\n" % ("item type", "mean", "SD"))
        W("  %-14s %8.3f %8.3f   (anchor)\n" % ("unrelated", 0.0, 1.0))
        for c in S.CELL_NAMES:
            W("  %-14s %8.3f %8.3f\n" % (c, cells[c][0], cells[c][1]))
        W("\n")
        W("DISCRIMINABILITY (d_a), frequentist, %d participant bootstrap resamples\n" % len(done))
        for k, lab in [("rare_matched", "rare vs its own foil"),
                       ("rare_unrel", "rare vs never-studied"),
                       ("ordinary", "ordinary studied vs never-studied")]:
            lo, hi = np.percentile(boot[k], [2.5, 97.5])
            W("  %-34s %7.3f  [%+.3f, %+.3f]\n" % (lab, point[k], lo, hi))
        W("\n")
        for k, lab in [("interaction", "INTERACTION (rare: never-studied minus own foil)"),
                       ("equivalence", "WHAT THE FEATURE BUYS (rare own-foil minus ordinary)")]:
            v = boot[k]
            lo, hi = np.percentile(v, [2.5, 97.5])
            p2 = 2 * min(np.mean(v <= 0), np.mean(v >= 0))
            W("  %-52s %+.3f  [%+.3f, %+.3f]  two-sided p = %.4f\n"
              % (lab, point[k], lo, hi, p2))
        W("\n")
    print(open(path, encoding="utf-8").read())


if __name__ == "__main__":
    main()
