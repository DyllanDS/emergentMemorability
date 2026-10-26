"""Parameter sweep for the hybrid-similarity GCM, modeled on the REM parameter sweep of module A2a
(SI Appendix, Fig. S2).

Each cell changes one thing from the reference (or crosses g with w, as the REM sweep does)
and runs the basic simulation with that setting. A setting equal to the reference is not run
again as its own cell, so every cell is a distinct parameter setting. Each reference
population keeps its own calibrated width from outputs/core.json, held fixed relative to the
median inter-item distance, and the criterion is set in each cell to the target false alarm
rate, so overall accuracy is free to move with the parameters.
Writes outputs/sweep.csv and outputs/sweep.txt.

The paper reports the [list] population (rarity from the participant's own study list). The
script reads the calibrated widths from outputs/core.json, so 01_core.py runs first. Every cell
has its own seed, zlib.crc32 of its label (its key in cells() followed by " [world]", " [list]" or " [mix]",
and no suffix for the set population), so the number of worker processes does not change any result, and
renaming a cell changes its numbers.
Run from the repository root:  python A2f_hybrid_exemplar_model_figS7/code/02_sweep.py  (about 3.5 min)
"""
import sys
sys.dont_write_bytecode = True      # keep code/ free of __pycache__
import zlib
from multiprocessing import Pool
from pathlib import Path
import json

import numpy as np
import pandas as pd

import hgcm as h
core = __import__("01_core")

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"       # the module's outputs/ folder
REF = dict(core.BASE, gamma=2.0, delta=0.0, q=0.10, list=40, ref="set")
REFS = ("world", "set", "list", "mix")


# ---- 1. the cells: one-at-a-time changes from the reference, plus the g x w grid ----
def cells():
    out = [("reference (g .65, w 49, hybrid gamma 2)", {})]
    for g in (0.30, 0.40, 0.50, 0.65):
        for w in (20, 30, 49):
            out.append((f"grid g={g:.2f} w={w}", dict(g=g, w=w)))
    out += [(f"storage u={u}", dict(u=u)) for u in (0.15, 0.30, 0.45)]
    out += [(f"copy c={c}", dict(cenc=c)) for c in (0.50, 0.60, 0.70, 0.85)]
    out += [(f"list length={n}", dict(n_study=n)) for n in (40, 100, 187)]
    out += [(f"false-alarm target={f}", dict(target_fa=f)) for f in (0.05, 0.15, 0.30)]   # seeded by this label; written as "false alarm target=..."
    out += [(f"rare-feature boost gamma={gm}", dict(gamma=gm)) for gm in (0.0, 1.0, 2.0, 3.0)]
    out += [(f"distinctive if set frequency < {q}", dict(q=q)) for q in (0.05, 0.10, 0.20)]
    out += [(f"mismatch penalty delta={d}", dict(delta=d)) for d in (0.0, 0.5)]
    return [out[0]] + [(lab, ch) for lab, ch in out[1:] if any(REF[k] != v for k, v in ch.items())]


# ---- 2. one cell ----
def run(arg):
    label, change, c_rel = arg
    rng = np.random.default_rng(zlib.crc32(label.encode()))
    p = dict(REF, **change)
    full = h.make_items(rng, p["n_full"], p["w"], p["g"])
    X = full[rng.choice(p["n_full"], p["n_db"], replace=False)]
    crowd, D = h.crowding(X)
    p["set_freq"] = h.set_freq(X)
    p["c"] = c_rel / float(np.nanmedian(D))
    _, _, _, fn = core.basic_batch(rng, X, p, 600)
    p["crit"] = float(np.quantile(fn, 1 - p["target_fa"]))
    a = core.analysis_a(rng, X, p, crowd, n_sim=5000, batch=250 if p["n_study"] > 100 else 500)
    return dict(cell=label.replace("false-alarm", "false alarm"), g=p["g"], w=p["w"], u=p["u"], cenc=p["cenc"], list=p["n_study"],
                fa_target=p["target_fa"], gamma=p["gamma"], q=p["q"], delta=p["delta"], ref=p["ref"], **a)


# ---- 3. run every cell for every reference population, then summarize ----
if __name__ == "__main__":
    params = json.loads((OUT / "core.json").read_text())
    # Each population's calibrated width, relative to the reference database's median distance.
    rng = np.random.default_rng(1)
    full = h.make_items(rng, core.BASE["n_full"], core.BASE["w"], core.BASE["g"])
    X = full[rng.choice(core.BASE["n_full"], core.BASE["n_db"], replace=False)]
    md = float(np.nanmedian(h.crowding(X)[1]))
    c_rel = {r: params[f"Hybrid GCM, {r}"]["params"]["c"] * md for r in REFS}
    with Pool(2) as pool:           # 2 worker processes; every cell has its own seed
        jobs = [(lab if r == "set" else f"{lab} [{r}]", dict(ch, ref=r), c_rel[r]) for r in REFS for lab, ch in cells()]
        rows = pool.map(run, jobs)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sweep.csv", index=False)
    lines = ["HYBRID-SIMILARITY GCM PARAMETER SWEEP", "=" * 100,
             "rho_item: Spearman correlation between per-item memorability (hit rate minus false alarm rate) and",
             "crowding (negative = isolated items better remembered). r_feat: the Fig. 1d analog.", "",
             f"{'cell':42s} {'rho_item':>9s} {'r_feat':>8s} {'hit':>7s} {'fa':>7s} {'d':>6s}", "-" * 100]
    for r in rows:
        lines.append(f"{r['cell']:42s} {r['rho_item']:+9.3f} {r['r_feat']:+8.3f} {r['hit']:7.3f} {r['fa']:7.3f} {r['dprime']:6.2f}")
    for r in REFS:
        d = df[(df.gamma > 0) & (df.ref == r)]
        lines.append(f"[{r}] distinct cells with a rare-feature boost (n = {len(d)}): rho_item {d.rho_item.min():+.3f} to {d.rho_item.max():+.3f}, median {d.rho_item.median():+.3f}; negative in {int((d.rho_item < 0).sum())}.")
    (OUT / "sweep.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
