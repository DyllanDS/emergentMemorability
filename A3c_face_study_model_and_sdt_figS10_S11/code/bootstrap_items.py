"""
bootstrap_items.py: the face study's d_a contrasts with the faces resampled instead of the
participants (SI Appendix, Section 3, "Resampling the features"; SI Appendix, Fig. S11b, the
intervals over features). Not preregistered: the registered interval resamples participants.

WHAT IT DOES
1,000 resamples, each refitting the registered unequal-variance model:
  items  the 6 rare-feature pairs, the 59 studied ordinary faces and the 59 never-studied faces are
         each resampled with replacement (a face drawn twice counts twice; the participants are kept)
For the effect of the lure set ("interaction") and the effect of the feature itself ("equivalence")
the report gives the 2.5th and 97.5th percentiles and the two-sided bootstrap p (twice the smaller of
the shares at or below zero and at or above zero).

run_module.py does not run the resamples (about 2 hours on 20 workers). The saved resamples are
data/provided/bootstrap_items.jsonl (1,000 lines: resample number, scheme, the five fitted d_a
quantities, no errors) and the saved report is data/provided/bootstrap_items.txt. check_provided.py
recomputes every interval and p of that report from the saved resamples. --report-only rewrites the
report from the resamples (outputs/bootstrap_items.jsonl when a new run made one, otherwise the saved
data/provided/bootstrap_items.jsonl) into outputs/bootstrap_items.txt; it does not need MemToolbox2.

The refits need the MemToolbox2 Python package in shared/memtoolbox2/ (EXTERNAL_INPUTS.md,
item M5). Without it, main() prints MISSING PACKAGE and PROVIDED OUTPUT and names the saved files.

INPUTS   04_rare_feature_faces_fig4/data/face_trials.csv (through face_study_common.py)
OUTPUTS  (with MemToolbox2) outputs/bootstrap_items.jsonl, outputs/bootstrap_items.txt
SEEDS    resample i: numpy.random.default_rng(10000 + i) draws the faces; the refit uses seed i
RUNTIME  about 2 hours on 20 workers
RUN      python A3c_face_study_model_and_sdt_figS10_S11/code/bootstrap_items.py --boot 1000 --workers 2 --modes items

How much of the contrast depends on which faces were used, rather than which people.

The registered bootstrap resamples participants. That answers "would another hundred people
show this?" and says nothing about "would another six rare pairs show this?", and the rare
condition rests on six pairs, a small sample of faces no matter how many people view them.

So the same contrasts are recomputed with the faces resampled:

  items  the six rare pairs, and the ordinary and never-studied images, resampled

Item resampling is done over IMAGES, not trials. Everyone who saw a given face contributes to
the same item, so resampling trials would treat one face viewed by eighty people as eighty
independent observations and hide exactly the variance being measured.

The registered interval is the participant one; this says what it leaves out.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.dont_write_bytecode = True

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import face_study_common as F             # noqa: E402  (each participant's test trials)
import sdt_shared_criteria as S         # noqa: E402

OUT = S.OUT                              # the module's outputs/ folder
RAW = os.path.join(OUT, "bootstrap_items.jsonl")
PROVIDED_RAW = os.path.join(F.release_paths.module_paths(__file__, make_output_folders=False)['provided'],
                            "bootstrap_items.jsonl")

_RECS = None


def _init(recs):
    global _RECS
    _RECS = recs


def item_keys(rec):
    """What item each test trial belongs to: a rare pair by its cell, everything else by image."""
    out = []
    for t in rec:
        if t.get("condition") == "rare":
            out.append(("rare", t.get("cell")))
        elif t.get("is_old"):
            out.append(("ord_old", t.get("image")))
        else:
            out.append(("new", t.get("image")))
    return out


def pools(recs):
    """The distinct items of each kind, across the whole sample."""
    p = defaultdict(set)
    for rec in recs:
        for kind, key in item_keys(rec):
            p[kind].add(key)
    return {k: sorted(v) for k, v in p.items()}


def one(job):
    """One resample of the faces (the items scheme)."""
    i, mode = job
    try:
        rng = np.random.default_rng(10_000 + i)
        recs = _RECS
        pl = pools(recs)
        # Resample each pool to its own size, with replacement. A face drawn twice counts
        # twice, which is what carries item variance into the interval.
        chosen = {k: list(rng.choice(v, size=len(v), replace=True)) for k, v in pl.items()}
        mult = {k: defaultdict(int) for k in chosen}
        for k, v in chosen.items():
            for key in v:
                mult[k][key] += 1
        ds = []
        for rec in recs:
            keep = []
            for t, (kind, key) in zip(rec, item_keys(rec)):
                keep.extend([t] * mult[kind].get(key, 0))
            if keep:
                ds.append(S.subject_data(keep))

        f = S.fit_mle(S.pooled(ds), seed=i)
        c = S.contrasts(S.params_to_cells(S.build_model(), f["MaxLikelihoodParams"]))
        return i, mode, {k: float(v) for k, v in c.items()}, None
    except Exception as e:
        return i, mode, None, "%s: %s" % (type(e).__name__, e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--modes", default="items")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if S.mt is None:
        # Without MemToolbox2 the refits cannot run: name the saved results instead.
        F.import_memtoolbox()                      # prints MISSING PACKAGE and PROVIDED OUTPUT
        F.report_provided_files("bootstrap_items.py", ["bootstrap_items.jsonl", "bootstrap_items.txt"])
        return

    recs = [s["rec"] for s in F.analysable(F.load())]
    pl = pools(recs)
    print("%d participants; items: %s" % (len(recs), {k: len(v) for k, v in pl.items()}), flush=True)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        if mode != "items":
            raise SystemExit("--modes: %r is not a resampling scheme (the scheme is items)" % mode)
    done = set()
    if os.path.exists(RAW) and not args.fresh:
        for line in open(RAW, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["mode"], r["i"]))
            except Exception:
                pass
        print("resuming: %d resamples on disk" % len(done), flush=True)

    todo = [(i, m) for m in modes for i in range(args.boot) if (m, i) not in done]
    print("%d resamples to run on %d workers" % (len(todo), args.workers), flush=True)

    t0 = time.time()
    with open(RAW, "a", encoding="utf-8") as fh:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                 initargs=(recs,)) as ex:
            for k, (i, mode, c, err) in enumerate(ex.map(one, todo, chunksize=1), 1):
                fh.write(json.dumps({"i": i, "mode": mode, "c": c, "err": err}) + "\n")
                if k % 25 == 0:
                    fh.flush()
                    el = time.time() - t0
                    print("  %d/%d  %.1f min elapsed, ~%.1f min left"
                          % (k, len(todo), el / 60, el / k * (len(todo) - k) / 60), flush=True)
    report()


def report():
    raw_file = RAW
    if not os.path.exists(raw_file):
        # Without a new run in outputs/, the report is made from the saved resamples in data/provided/.
        raw_file = PROVIDED_RAW
        print("report from the saved resamples data/provided/bootstrap_items.jsonl")
    rows = [json.loads(l) for l in open(raw_file, encoding="utf-8") if l.strip()]
    by = defaultdict(list)
    for r in rows:
        if r.get("c"):
            by[r["mode"]].append(r["c"])

    lines = ["What the contrasts rest on: participants, or faces?", "=" * 62, "",
             "The registered interval resamples participants. The rare condition rests on six",
             "pairs, so it carries no information about how much the estimate depends on which",
             "six. Resampling the faces adds that.", ""]
    for key, lab in (("interaction", "INTERACTION (rare: never-studied minus own foil)"),
                     ("equivalence", "WHAT THE FEATURE BUYS (rare own-foil minus ordinary)")):
        lines.append(lab)
        v = np.array([c[key] for c in by.get("items", []) if np.isfinite(c[key])], float)
        if v.size >= 20:
            lo, hi = np.percentile(v, [2.5, 97.5])
            p2 = 2 * min(np.mean(v <= 0), np.mean(v >= 0))
            lines.append("  %-14s n=%5d   [%+.3f, %+.3f]   p = %.4f   %s"
                         % ("items", v.size, lo, hi, p2,
                            "excludes zero" if lo > 0 or hi < 0 else "INCLUDES ZERO"))
        lines.append("")
    txt = "\n".join(lines) + "\n"
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "bootstrap_items.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report()
    else:
        main()
