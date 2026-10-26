# 01: REM simulation of memorability (Fig. 1)

This module runs the REM simulation of Fig. 1: 10,000 simulated recognition memory experiments over a set
of 1,600 generated items, each a vector of 49 random feature values. It reproduces the four panels of
Fig. 1 and the numbers of the sections "Simulations of Memorability as an Emergent Property" and "Basic
REM Simulation of Memorability" (Methods), and it computes tests T01 (Fig. 1b) and T02 (Fig. 1d) of
`statistics/all_tests.csv`. The simulation is seeded, and `outputs/rerun_vs_paper_run.csv` compares every
array of the rerun with the saved run in `data/paper_run/`, the output the paper reports.

## How to run

From the repository root:

    matlab -batch "restoredefaultpath; cd('01_rem_simulation_fig1/code'); run_module"
    python 01_rem_simulation_fig1/code/stats_fig1.py

The MATLAB step needs the Statistics and Machine Learning Toolbox (MATLAB R2023b tested, no parallel
pool). On an Apple M2 Pro with 16 GB it takes about 2.5 min, with about 2.5 GB of memory at the peak,
while the rerun and the saved run are compared. The Python step needs numpy and scipy and takes about
1 min (0.5 GB of memory). It computes from the saved run and adds the rows of the two correlations of
Fig. 1 to `numbers.csv`, so it runs after each MATLAB call that writes that file (`run_module` or
`fig1_numbers`).

| MATLAB call (in `code/`) | What it does | Time |
|---|---|---|
| `run_module` | the rerun (seed 12345, 10,000 simulated experiments) and its comparison with the saved run; Fig. 1, `numbers.csv` and `summed_similarity_measures.csv` from the rerun | about 2.5 min |
| `run_module(500)` | a quick test with 500 simulated experiments. The numbers then do not match the paper, but the run checks that its item set and its 500 experiments equal the first 500 of the saved run. It writes over `outputs/`, so run both steps again afterwards (or test in a copy of the folder) | about 1 min |
| `fig1_makeFigure1` | Fig. 1 from the saved run, without a rerun | seconds |
| `fig1_numbers` | `numbers.csv` and `summed_similarity_measures.csv` from the saved run, without a rerun (mode `saved_run`). The saved run does not store the study-list length, so in this mode the three rows that print it (L1-007, L1-018, L5-011) are marked `documented` | seconds |
| `fig1_remSim(12345, 'file.mat')` | the simulation alone, saved to `file.mat` (`run_module` does not save the rerun) | about 2 min |

## Inputs

- `data/paper_run/fig1_paper_run.mat`: the saved run behind Fig. 1 and the numbers of the paper
  (`code/fig1_remSim.m` with seed 12345 and 10,000 simulated experiments). It holds one struct,
  `figureData`, with every per-item and per-feature result and the per-simulation arrays (likelihood
  ratios and test masks, 1,600 items x 10,000 simulated experiments). `stats_fig1.py` computes tests T01
  and T02 from it, and modules A1, A2a, A2e and A4a read it too.
- `shared/things/spose_embedding_49d_sorted.txt`: the 49-dimensional SPoSE embedding of the THINGS
  concepts (Hebart et al., 2020). The module only counts its columns (the 49 features of rows L1-003
  and L1-105).
- Shared helpers: `shared/matlab/release_paths.m` and `write_numbers_csv.m` (which reads the printed
  values from `shared/ledger/paper_numbers.csv`), and `shared/python/release_paths.py`, `bootstrap.py`,
  `tests_table.py` and `numbers_table.py`.

The simulation generates its own items, so the module needs nothing from outside the repository.

## What the code does

`run_module.m` reruns the simulation, compares the rerun with the saved run, draws Fig. 1
(`fig1_makeFigure1.m`) and writes the numbers (`fig1_measures.m`, `fig1_numbers.m`), in that order and in
one MATLAB session. The simulation itself is `fig1_remSim.m`.

**Items.** `rng(12345, 'twister')` is called once. The code then generates 26,107 items (the number of
THINGS images) with 49 features each and draws a random 1,600 of them as the simulated image set. Each
feature value is a geometric draw with g = 0.65 plus 1, so values start at 1 (1 to 12 in this set).

**Experiments.** Each of the 10,000 simulated experiments draws 60 different items of the 1,600 in random
order. The first 40 are studied, the first 20 of those are tested as old items, and the last 20 are the
new foils. Every item is tested as an old item 88 to 167 times and as a foil 92 to 165 times. Experiment k
draws its list and its encoding from its own random stream (Threefry, seed 12345, substream k), so the
draws do not depend on the order in which the experiments run. The code runs them one after another and
restores the session's random stream afterwards.

**Encoding.** Each feature of a studied item is stored with probability u = 0.3. A stored feature is
copied correctly with probability c = 0.6; otherwise it gets a new geometric draw (g = 0.65) without the
plus 1, and a new value of 0 (probability g) counts as not stored. About 22% of the features of a studied
item are therefore stored (0.3 x 0.6 + 0.3 x 0.4 x 0.35 = 0.222).

**Decision.** A test item is compared with each of the 40 memory traces. Every stored feature that does
not match multiplies the likelihood ratio by (1 - c), and every matching feature of value v multiplies it
by (c + (1 - c) P(v)) / P(v), where P(v) = g (1 - g)^(v - 1) is the geometric probability of the value.
The item is called "old" when the mean likelihood ratio over the 40 traces is greater than 1, for every
item and every experiment.

**Memorability.** An item's memorability is its hit rate (over the experiments in which it is a tested
old item) minus its false alarm rate (over the experiments in which it is a foil). The mean hit rate is
0.853 and the mean false alarm rate 0.202, and memorability runs from 0.352 to 0.911 (Fig. 1a).

**Similarity.** Distances are Euclidean, in the 49-dimensional feature space. An item's mean distance is
the mean over all 1,600 distances, its own distance of 0 included (dividing by 1,599 changes no
correlation), and its similarity is 1 / mean distance. The x axis of Fig. 1b, summed similarity, is the
sum over the other 1,599 items of 1 / (distance + eps), computed in `fig1_measures.m`. The r, CI and p of
the Fig. 1 caption are for similarity = 1 / mean distance; panel b plots the summed similarity.
`outputs/summed_similarity_measures.csv` gives all four measures:

| Measure (per item) | Statistic with memorability | Value | Where it is used |
|---|---|---|---|
| similarity = 1 / mean Euclidean distance | Pearson r | -0.557 | Fig. 1 caption (panel b) and Simulations of Memorability as an Emergent Property (r = -0.56) |
| summed similarity = sum over the other items of 1 / distance | Pearson r | -0.565 | the x axis of Fig. 1b |
| mean Euclidean distance | Pearson r | 0.536 | the distance of Methods, Basic REM Simulation of Memorability (larger = more distinctive) |
| negated mean Euclidean distance | Spearman ρ | -0.551 | Methods, Basic REM Simulation of Memorability (ρ = -0.55); the measure of SI Appendix, Fig. S2 |

**Features.** For each feature, the code computes its Spearman correlation across the 1,600 items with
memorability (Fig. 1c: 0.059 to 0.153, all 49 with p < 0.05) and with similarity (1 / mean distance; the
x axis of Fig. 1d, -0.114 to -0.008). The ρ = -0.44 of Fig. 1d is the Spearman correlation of these two
sets of 49 correlations across the features (MATLAB `corr` with 'Type', 'Spearman'; p = 0.0015). Spearman
correlations give tied values their average rank (MATLAB `corr`, scipy `rankdata`), and the per-feature
correlations have many ties because the feature values are small integers.

**Numbers.** The p of the Fig. 1b correlation (row L1-019 of `numbers.csv` and test T01, p = 4.4e-131)
is the two-sided p of the t test of r with 1,598 degrees of freedom, computed with the incomplete beta
function (`scipy.special.betainc` in `stats_fig1.py`). The Methods section "Basic REM Simulation of
Memorability" compares the ρ = -0.55 of this run with the ρ = -0.52 at the standard parameter values of
Shiffrin and Steyvers (1997), which the parameter sweep of module A2a (`rem_parameter_sweep.m`) computes.
Both are Spearman correlations of memorability with the negated mean distance. The sweep adds 1 to the new
draw of a miscopied feature, so every miscopied feature stays stored.

**Comparison with the saved run.** The p values that MATLAB's `corr` computes with `tcdf.m` (the arrays
marked "the simulation and tcdf.m" in `outputs/rerun_vs_paper_run.csv`) can differ from the saved run in
their last digits, and the saved run stores the Fig. 1b p as 0.

**Statistics.** `stats_fig1.py` recomputes the two correlations of Fig. 1 and all 98 per-feature
correlations from the saved run and stops if they differ from the MATLAB values. Its 95% intervals are
percentile bootstrap intervals over the 1,600 simulated items (10,000 resamples, numpy
`default_rng(20261013)`). Each item keeps its memorability, similarity and 49 feature values, the
features are never resampled, and all 49 per-feature correlations are recomputed in every resample. The p of Fig. 1d is the one MATLAB `corr` gives. No correction for
multiple comparisons is applied. The script writes the rows of the two correlations as the paper prints
them (r or ρ, 95% CI, p) to `numbers.csv`: L1-019 and L1-022 (Fig. 1b), L1-021 and L1-024 (Fig. 1d).

## Outputs

- `outputs/numbers.csv`: each number of the paper that this module computes (42 rows), with the printed
  value, the computed value and whether they agree at the printed precision (columns in the top-level
  README). `fig1_numbers.m` writes 38 rows and `stats_fig1.py` the other 4.
- `outputs/tests.csv`: tests T01 and T02, one row per statistic (table below).
- `outputs/figures/fig1_rem_simulation.png`: Fig. 1, panels a to d (300 dpi). The x axis of panel d,
  labeled "Correlation With Summed Similarity", shows each feature's Spearman correlation with similarity
  (1 / mean distance).
- `outputs/figure_data/fig1a_item_memorability_rank_order.csv`,
  `fig1b_memorability_vs_summed_similarity.csv`, `fig1c_feature_correlation_rank_order.csv` and
  `fig1d_feature_memorability_vs_similarity.csv`: the numbers each panel plots (item or feature number in
  the order of the simulation, the plotted values, the least-squares line).
- `outputs/rerun_vs_paper_run.csv`: every array of the saved run against the same array of the rerun
  (what it depends on, identical or not, largest absolute and relative difference).
- `outputs/summed_similarity_measures.csv`: the four similarity measures of the table above, in the rerun
  and in the saved run.

| Test | Statistic | n | 95% CI | p |
|---|---|---|---|---|
| T01 | Pearson r = -0.557, memorability vs similarity = 1 / mean distance (r² = 0.310) | 1,600 items | [-0.590, -0.522] (bootstrap over items) | 4.4e-131 (t test, 1,598 df) |
| T01 | Spearman ρ = -0.551, memorability vs negated mean distance | 1,600 items | [-0.587, -0.513] | |
| T02 | Spearman ρ = -0.444 across the features | 49 features | [-0.702, -0.299] (bootstrap over the 1,600 items) | 0.0015 (MATLAB `corr`) |

## Code

| File | What it does |
|---|---|
| `code/run_module.m` | entry point: the rerun, its comparison with the saved run, Fig. 1 and the numbers |
| `code/fig1_remSim.m` | the REM simulation: items, experiments, per-item rates, distances and feature correlations |
| `code/fig1_measures.m` | every number of Fig. 1 from one run, with each similarity measure and statistic named |
| `code/fig1_makeFigure1.m` | Fig. 1 (panels a to d in one 2 x 2 figure) and the numbers each panel plots |
| `code/fig1_numbers.m` | `outputs/numbers.csv` and `outputs/summed_similarity_measures.csv` |
| `code/stats_fig1.py` | tests T01 and T02 (`outputs/tests.csv`) and the `numbers.csv` rows of the Fig. 1b and 1d correlations |
