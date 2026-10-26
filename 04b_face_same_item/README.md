# 04b: Each face scored against itself in the rare-feature face study (main text)

This module reproduces one result of the section "A Rare Feature's Advantage Depends on What the Face Is Tested
Against": the preregistered rare-feature face study scored as memorability is usually scored, each face against
itself across participants. From the data of module 04 it computes the hit rate, the false alarm rate, the hit
rate minus the false alarm rate and d′ of each of the 12 rare-feature faces and 59 ordinary faces, and the mean
of each measure over the rare-feature faces and over the ordinary faces (hit rate minus false alarm rate 0.33
against 0.38, d′ 1.12 against 1.10, hit rate 0.90 against 0.79 and false alarm rate 0.57 against 0.41). The
module writes ledger items N2-001 and N2-002 to `outputs/numbers.csv`, has no statistical test and draws no
figure.

## How to run

From the repository root:

    python 04b_face_same_item/code/run_module.py

The module takes a few seconds on the tested laptop (Apple M2 Pro, 16 GB) and needs Python 3 with numpy, scipy
and pandas. It has no options and draws no random numbers.

## Inputs

- `04_rare_feature_faces_fig4/data/face_trials.csv` and `face_participants.csv`: the trials and sessions of the
  face study (columns in the README of module 04), read through `shared/python/face_loader.py`, which applies the
  preregistered exclusion rule and returns the 100 registered participants.
- The shared helpers `face_loader.py`, `sdt.py`, `numbers_table.py`, `tests_table.py` and `release_paths.py` in
  `shared/python/`, and `shared/ledger/paper_numbers.csv` (the values printed in the paper).

## What the code does

For each participant, one member of each rare-feature pair is drawn to be studied, and each ordinary face is
studied by some participants and new for others, so every face is tested both after it has been studied and
when it has not. When a rare-feature face is new, the other member of its pair, which carries the same feature,
has been studied (`test_type` is `foil` on these trials, and the code stops if one is not). For each face, over
the recognition trials of the 100 registered participants, the code counts the hits among the old trials and the
false alarms among the new trials (15 to 35 trials of each kind per face) and computes the hit rate, the false
alarm rate, their difference and the log-linear d′, z((hits + 0.5) / (old trials + 1)) - z((false alarms + 0.5)
/ (new trials + 1)) (`sdt.dprime_loglinear_counts`). Each measure is then averaged over the 12 rare-feature
faces and over the 59 ordinary faces, each face weighted equally.

## Outputs

- `outputs/numbers.csv`: each number of the paper this module computes, with the printed value, the computed
  value and whether they agree at the printed precision (columns in the top-level README).
- `outputs/tests.csv`: the header only (the module has no test).
- `outputs/same_face_items.csv`: one row per face with `image`, `rare` (True for a rare-feature face), `hit`,
  `fa`, `cr` (hit rate minus false alarm rate), `dprime`, `n_old` and `n_new`.
- `outputs/same_face_summary.csv`: the mean of each measure over the rare-feature faces (`rare`) and over the
  ordinary faces (`ordinary`).
- `outputs/logs/run_module.log`: everything the module printed.

## Code

| File | What it does |
|---|---|
| `code/run_module.py` | runs `same_face_scores.py` and writes `outputs/numbers.csv`, `outputs/tests.csv` and the log |
| `code/same_face_scores.py` | the scores of each face and their means over the rare-feature and the ordinary faces |
