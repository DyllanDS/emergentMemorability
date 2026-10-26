# A2a: REM parameter sweeps (SI Appendix, Fig. S2)

This module tests whether the REM result of Fig. 1 (distinctive items are more memorable) depends on the
parameter values used for Fig. 1. It runs the two sweeps of SI Appendix, Section 2 (REM Parameters), which
the main text (Simulations of Memorability as an Emergent Property) and the Methods cite: a sweep of 32
settings of REM's parameters (SI Appendix, Fig. S2a and b) and the REM simulation of Fig. 1 with 5 to 120
features per item. The sweep's cell at the standard values of Shiffrin and Steyvers (1997) gives the
ρ = -0.52, 95% CI [-0.56, -0.49], of the Methods (Basic REM Simulation of Memorability); the -0.55 that the
Methods compares it with is the same measure (Spearman ρ with the negated mean distance) on the saved Fig. 1
run of module 01, computed here. The module's statistics are tests T20, T24 and T25 (`outputs/tests.csv`,
collected in `statistics/all_tests.csv`).

## How to run

From the repository root:

    python A2a_rem_parameter_sweep_figS2/code/run_module.py

On the tested laptop (Apple M2 Pro, 16 GB) this takes about 9 minutes: about 2 for the MATLAB sweep, about
3 for the feature-count sweep (up to about 12 when other jobs share the processor), about 3.5 for the
statistics and a few seconds for each of the other steps. MATLAB runs as one serial session of about 1 GB.

| Option | What it does |
|---|---|
| `--skip-long` | skips the feature-count sweep, so the module takes about 6 minutes; the feature-count numbers then come from `outputs/sweep_rem.csv` if an earlier full run left it there, else from `data/paper_run/sweep_rem.csv`, and the `mode` column of `numbers.csv` says which (`rerun` or `saved_run`) |
| `--skip-matlab` | skips the MATLAB sweep and uses the outputs of an earlier run in `outputs/` (it stops if they are missing) |

The same steps one at a time, in the order `run_module.py` runs them (each prints what it reads and writes):

    matlab -batch "restoredefaultpath; cd('A2a_rem_parameter_sweep_figS2/code'); rem_parameter_sweep(true)"
    python A2a_rem_parameter_sweep_figS2/code/accuracy_link.py
    python A2a_rem_parameter_sweep_figS2/code/feature_count_sweep.py
    python A2a_rem_parameter_sweep_figS2/code/stats_figS2.py
    python A2a_rem_parameter_sweep_figS2/code/make_figureS2.py
    python A2a_rem_parameter_sweep_figS2/code/check_against_paper_run.py

`run_module.py` finds MATLAB from the environment variable `MATLAB`, else on the PATH, else in a standard
installation folder (preferring R2023b). It starts MATLAB with `restoredefaultpath`, for that session only,
so that no folder on the user's MATLAB path can replace a MATLAB function the sweep uses (a `tcdf.m` from
another toolbox, such as FieldTrip's, returns 0 for p values below about 1e-16, and every p value of the
sweep is below 1e-29), and `rem_parameter_sweep.m` prints a warning when a function is replaced. With
`--paper-run`, `accuracy_link.py` and then `make_figureS2.py` read the saved run in `data/paper_run/` instead
of this run and write files whose names end in `_paper_run`. `feature_count_sweep.py --nsim 200 --widths 5 49
--output sweep_rem_quick_test.csv` is a quick test of the code, which does not give the paper's numbers.

Requirements: MATLAB R2023b with the Statistics and Machine Learning Toolbox (no parallel pool), and Python 3
with numpy, scipy and matplotlib (tested with Python 3.12.4, numpy 1.26.4, scipy 1.13.1 and matplotlib 3.8.4).

## Inputs

- None for the simulations, which generate their own items.
- `data/paper_run/rem_parameter_sweep.csv` and `rem_parameter_sweep.txt`: the saved run of
  `rem_parameter_sweep.m` (its table and printed report), the output the paper reports. Module A2f reads the
  row "paper values" of the table (the cell with the Fig. 1 values).
- `data/paper_run/sweep_rem.csv`: the saved run of the feature-count sweep (the first eight columns of
  `outputs/sweep_rem.csv`).
- `01_rem_simulation_fig1/data/paper_run/fig1_paper_run.mat` (module 01): the saved Fig. 1 run, read for the
  two Fig. 1 rows of test T20 and for the comparison that `check_against_paper_run.py` prints. Without it
  those rows are left out, with a message.
- `shared/matlab/release_paths.m`; `shared/python/release_paths.py`, `bootstrap.py`, `figure_style.py`,
  `numbers_table.py` and `tests_table.py`; `shared/ledger/paper_numbers.csv` (the numbers of the paper, read
  by `numbers_table.py`).

## What the code does

**The 32-cell sweep** (`rem_parameter_sweep.m`). The script calls `rng(1, 'twister')` once and then runs the
32 cells serially in file order, so the random numbers of a cell depend on every cell before it and a cell
reproduces only within the whole sweep (run it without a parallel pool). Each cell generates 1,600 items
whose feature values are geometric draws with parameter g plus 1, and runs 10,000 recognition experiments.
In each experiment a study list (40 items; 100, 187 or 400 in the list-length cells) and 20 unstudied items
are drawn from the 1,600, and 20 list items are tested as old along with the 20 unstudied items as new. A
studied feature is stored with probability u (storage) and, once stored, copied correctly with probability c
(copy fidelity); a wrongly copied feature gets a new geometric value counted from 1. Familiarity is the mean
REM likelihood ratio over the list's traces (Shiffrin and Steyvers, 1997), and a test item is called old
when its familiarity is above the criterion (1.0, except in the criterion cells). An item's memorability is
its hit rate minus its false alarm rate over the experiments that tested it (every item is tested at least
80 times as old and 80 times as new in every cell), and its summed similarity is minus its mean Euclidean
distance to the other 1,599 items, so a negative correlation means that distinctive items are remembered
better. Each cell's statistic is the Spearman ρ between the two across the 1,600 items (tied values take
their average rank), with p from the t approximation with n - 2 degrees of freedom; its overall d′ is
z(mean hit rate) minus z(mean false alarm rate), with both rates kept within 0.001 and 0.999.

The grid crosses g = 0.30, 0.35, 0.40, 0.50 and 0.65 with 20, 30 and 49 features (15 cells). Storage (0.15,
0.30, 0.45), copy fidelity (0.50, 0.60, 0.70, 0.85), list length (40, 100, 187, 400) and criterion (0.5,
1.0, 2.0, 5.0) vary one at a time from the standard values (g 0.40, 20 features, storage 0.30, copy fidelity
0.70, list length 40, criterion 1.0). The standard cell and the cell with the Fig. 1 values (g 0.65, 49
features, copy fidelity 0.60, labeled "paper values (g=0.65 w=49 c=0.60)" in the table) complete the 32. Six
cells therefore use the standard values (the standard cell, the grid cell with g 0.40 and 20 features, and the
standard point of each one-at-a-time group); they differ only in their random draws, and their ρ runs from
-0.55 to -0.51. The list length 187 is the number of distinct images in a 295-trial session of Kramer et al.
(2023), 50 targets and 137 fillers, and the code uses it as a fixed number (counting it needs their trial
file, EXTERNAL_INPUTS.md, M1). The "Fig. 1 Values" point of Fig. S2 is this sweep's cell with the Fig. 1
values, not the Fig. 1 run of module 01. The same cell gives the REM ρ of -0.61 and the REM d′ of 1.61 in
SI Appendix, Section 2, A Hybrid-Similarity Exemplar Model. The table also holds rFeat and pFeat for each
cell: the Pearson r across the features between each feature's Spearman correlation with memorability and its
Spearman correlation with summed similarity (the feature-level relationship of Fig. 1d), and its p. With
`saveItems` (on by default) the script also writes each cell's per-item counts and distances.

**Accuracy and ρ** (`accuracy_link.py`). The script computes the Pearson r between each cell's d′ and its ρ
over the 32 cells, lists the six runs at the standard values, and finds where the cell with the Fig. 1
values falls among the 32: its ρ is the 4th most negative, and its d′ of 1.61 is close to the median d′ of
1.58 (row L6-027 of `numbers.csv`).

**The feature-count sweep** (`feature_count_sweep.py`). This is the REM simulation of Fig. 1 in NumPy,
run at 5, 10, 15, 20, 25, 30, 35, 40, 45, 49, 55, 60, 66, 70, 80, 90, 100 and 120 features with seeds 12345,
777 and 20260731 (54 runs). It uses g = 0.65, storage 0.30, copy fidelity 0.60 and criterion 1.0, generates
26,107 items and draws a random 1,600 of them as the set, and each of its 10,000 experiments studies 40 items
and tests 20 old and 20 new ones. Each run has its own `numpy.random.default_rng(seed)`, so a run depends
only on its seed and feature count, and the draws come in a fixed order (the generated items, the
permutation that picks the set, then per experiment the permutation of studied and new items, the stored and
copied-correctly uniforms, and the values of the wrongly copied features). As in module 01
(`fig1_remSim.m`), a wrongly copied feature value is a geometric draw counted from 0 and a 0 counts as not
stored, while `rem_parameter_sweep.m` adds 1 to the draw. Summed similarity here is 1 / mean Euclidean
distance, which has the same ranks as the negated mean distance, and SI Appendix, Section 2 reports the
range of ρ over the 54 single runs.

**Statistics** (`stats_figS2.py`). The intervals for correlations over simulated items are percentile
bootstraps over the 1,600 items (10,000 resamples with `numpy.random.default_rng(20261013)`, the same seed for
each cell, so that each cell's interval reproduces on its own). Every Spearman p uses the t approximation
with n - 2 degrees of freedom (the formula of the `pItem` column). The interval for r(d′, ρ) is a Fisher z
interval. The six runs at the standard values (Monte Carlo spread) and the 54 feature-count runs are reported
as ranges, not confidence intervals, and no correction for multiple comparisons is applied.

**Figure and comparison.** `make_figureS2.py` draws Fig. S2 from this run, with each cell's p computed from
the per-item file, and writes the plotted numbers. `check_against_paper_run.py` compares this run with the
saved runs in `data/paper_run/` column by column (allowed difference 1e-6, and for p values 1e-6 of the value)
and the printed report line by line, and writes `numbers.csv`.

## Outputs

- `outputs/numbers.csv`: each number of the paper this module computes, with the printed value, the computed
  value and whether they agree at the printed precision (columns in the top-level README).
- `outputs/tests.csv`: T20 (ρ at the standard values; the six standard runs; from the saved Fig. 1 run, the
  Pearson r of Fig. 1b with its bootstrap interval and the Spearman ρ, the -0.55 of the Methods),
  T24 (each cell's ρ, n, df, p and bootstrap interval; the number of negative cells with the range of ρ and the
  largest p; the feature-count sweep over its 54 runs) and T25 (r between d′ and ρ over the 32 cells, with its
  Fisher z interval).
- `outputs/figures/FigS2_rem_parameter_sweep.png`: Fig. S2a and b (300 dpi).
- `outputs/figure_data/FigS2a.csv`, `FigS2b.csv`, `FigS2b_fit_line.csv`: the numbers each panel plots,
  including the least-squares line of panel b.
- `outputs/rem_parameter_sweep.csv` and `.txt`: one row per cell (label, g, w for the number of features, u for
  storage, c for copy fidelity, nStudy for the list length, crit, rhoItem, pItem, rFeat, pFeat, meanHit,
  meanFA, dprime), and the printed report of the same table.
- `outputs/rem_parameter_sweep_items.csv`: cell_number, item, old_trials, hits, new_trials, false_alarms and
  neg_mean_distance, one row per cell and item (51,200 rows).
- `outputs/accuracy_link.json`: r(d′, ρ) over the 32 cells, the six runs at the standard
  values and the cell with the Fig. 1 values (the keys that start with `fig1_values_`).
- `outputs/sweep_rem.csv`: the feature-count sweep, one row per run, the eight columns of the saved run first.
- `outputs/equality_with_paper_run.csv`: this run against the saved runs in `data/paper_run/`, by column.
- `outputs/logs/`: one log per step and the sweep's progress log, written when the code runs.

## Code

| File | What it does |
|---|---|
| `code/run_module.py` | runs the six steps below in order |
| `code/rem_parameter_sweep.m` | the 32-cell REM sweep; with `saveItems` (default true) it also writes every cell's per-item counts and distances, and it warns when a folder on the MATLAB path replaces a function it uses |
| `code/accuracy_link.py` | r between d′ and ρ over the 32 cells; the six runs at the standard values; where the cell with the Fig. 1 values falls |
| `code/feature_count_sweep.py` | the REM simulation of Fig. 1 in NumPy at 5 to 120 features, seeds 12345, 777 and 20260731 |
| `code/stats_figS2.py` | tests T20, T24 and T25: item-bootstrap intervals for every cell, p values, the ranges of the standard runs and of the feature-count sweep, the Fisher z interval for r(d′, ρ) |
| `code/make_figureS2.py` | Fig. S2 and its plotted numbers |
| `code/check_against_paper_run.py` | compares this run with the saved runs in `data/paper_run/`; writes `numbers.csv` |
| `code/sweep_tables.py` | readers of the sweep's tables, used by the three scripts above |
