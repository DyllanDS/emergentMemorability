"""
leave_one_pair_out.py: the face study's d_a contrasts refitted six times, each time without one of
the six rare-feature pairs (SI Appendix, Section 3, "Resampling the features": the lure-set effect
stays between 0.47 and 0.85 and the feature's own effect between -0.12 and 0.15). Not preregistered.

WHAT IT DOES
First the hit and false alarm rate of each rare pair, pooled over the 100 participants (the studied
face called old; the never-studied partner called old). Then seven fits of the registered
unequal-variance model: all pairs (the registered fit) and without each pair c1 to c6 (the pair's
rare-feature trials removed; the ordinary faces stay). The report lists the effect of the lure set
("interaction") and the effect of the feature itself ("feature buys") for each fit and their range
over the six deletions.

The fits need the MemToolbox2 Python package in shared/memtoolbox2/ (EXTERNAL_INPUTS.md, item
M5). Without it the script still prints the per-pair rates (computed from the data), then MISSING
PACKAGE and PROVIDED OUTPUT, and names the saved fits: data/provided/leave_one_pair_out.json (every
quantity of the seven fits) and data/provided/leave_one_pair_out.txt (the report). check_provided.py
reads both.

INPUTS   04_rare_feature_faces_fig4/data/face_trials.csv (through face_study_common.py)
OUTPUTS  (with MemToolbox2) outputs/leave_one_pair_out.txt and outputs/leave_one_pair_out.json
SEEDS    every fit uses fit_mle(..., seed=0) (sdt_shared_criteria.py)
RUNTIME  about 7 minutes with 2 workers; the per-pair rates take a second
RUN      python A3c_face_study_model_and_sdt_figS10_S11/code/leave_one_pair_out.py

Do the registered contrasts depend on rare features in general, or on six particular faces?

The whole rare condition rests on six pairs, one per demographic cell. With six items, a single
pair that happens to be memorable for reasons of its own (a distinctive jaw, a better-lit
photograph) can carry a contrast by itself, and no amount of participants will reveal it,
because every participant sees the same six.

So the model is refitted six times, each time with one pair's trials removed entirely. A
contrast that is a property of rare features survives every deletion. One that is an item effect
collapses when its item goes.

This is not the registered analysis, and the registered contrasts are unchanged by it. It only
asks what the registered estimate rests on.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.dont_write_bytecode = True

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import face_study_common as F             # noqa: E402  (each participant's test trials)
import sdt_shared_criteria as S         # noqa: E402

OUT = S.OUT                              # the module's outputs/ folder


def cells_present(subs):
    """The pair labels actually used by the critical trials."""
    seen = set()
    for s in subs:
        for t in s["rec"]:
            if t.get("condition") == "rare" and t.get("cell"):
                seen.add(t["cell"])
    return sorted(seen)


def fit_without(job):
    """Refit with one pair's trials dropped, or with none dropped when drop is None."""
    drop, subs_rec = job
    try:
        ds = []
        for rec in subs_rec:
            keep = [t for t in rec if not (drop is not None and t.get("cell") == drop
                                           and t.get("condition") == "rare")]
            ds.append(S.subject_data(keep))
        f = S.fit_mle(S.pooled(ds), seed=0)
        c = S.contrasts(S.params_to_cells(S.build_model(), f["MaxLikelihoodParams"]))
        return drop, {k: float(v) for k, v in c.items()}, None
    except Exception as e:
        return drop, None, "%s: %s" % (type(e).__name__, e)


def main():
    subs = F.analysable(F.load())
    recs = [s["rec"] for s in subs]
    pairs = cells_present(subs)
    print("%d participants, rare pairs: %s" % (len(subs), pairs), flush=True)

    # Per-pair descriptive rates first: cheap, and already shows whether one pair is odd.
    print("\nper-pair raw rates (pooled over participants)")
    print("  %-6s %8s %8s %6s %6s" % ("pair", "hit", "FA", "n_old", "n_foil"))
    for p in pairs:
        old = [t for s in subs for t in s["rec"]
               if t.get("condition") == "rare" and t.get("cell") == p and t.get("test_type") == "old"]
        foil = [t for s in subs for t in s["rec"]
                if t.get("condition") == "rare" and t.get("cell") == p and t.get("test_type") == "foil"]
        h = np.mean([t["said_old"] for t in old]) if old else np.nan
        fa = np.mean([t["said_old"] for t in foil]) if foil else np.nan
        print("  %-6s %8.3f %8.3f %6d %6d" % (p, h, fa, len(old), len(foil)), flush=True)

    if S.mt is None:
        # Without MemToolbox2 the seven refits cannot run: name the saved results instead.
        print()
        F.import_memtoolbox()                      # prints MISSING PACKAGE and PROVIDED OUTPUT
        F.report_provided_files("leave_one_pair_out.py", ["leave_one_pair_out.json", "leave_one_pair_out.txt"],
                                show_contents=True)
        return

    jobs = [(None, recs)] + [(p, recs) for p in pairs]
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(max_workers=2) as ex:
        for drop, c, err in ex.map(fit_without, jobs, chunksize=1):
            res[drop] = c if c else {"err": err}
            print("  fitted %-14s %.1f min" % (str(drop), (time.time() - t0) / 60), flush=True)

    full = res[None]
    lines = ["Leave-one-pair-out: what the registered contrasts rest on", "=" * 60, "",
             "Six rare pairs, one per demographic cell. Each row refits the model with that",
             "pair's trials removed. Exploratory; the registered fit is the 'all pairs' row.", "",
             "  %-16s %12s %12s" % ("dropped", "interaction", "feature buys")]
    lines.append("  %-16s %+12.3f %+12.3f" % ("all pairs", full["interaction"], full["equivalence"]))
    ints, buys = [], []
    for p in pairs:
        c = res.get(p) or {}
        if "err" in c:
            lines.append("  %-16s %s" % (p, c["err"]))
            continue
        ints.append(c["interaction"]); buys.append(c["equivalence"])
        lines.append("  %-16s %+12.3f %+12.3f" % ("without " + p, c["interaction"], c["equivalence"]))
    if buys:
        lines += ["",
                  "  interaction  across deletions: %+.3f to %+.3f" % (min(ints), max(ints)),
                  "  feature buys across deletions: %+.3f to %+.3f" % (min(buys), max(buys)),
                  "",
                  ("  The feature-buys estimate stays positive with any single pair removed."
                   if min(buys) > 0 else
                   "  The feature-buys estimate changes sign when at least one pair is removed,"
                   "\n  so it is not independent of which six faces were used.")]
    txt = "\n".join(lines) + "\n"
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "leave_one_pair_out.txt"), "w", encoding="utf-8").write(txt)
    json.dump({str(k): v for k, v in res.items()},
              open(os.path.join(OUT, "leave_one_pair_out.json"), "w"), indent=1)
    print("\n" + txt)


if __name__ == "__main__":
    main()
