# 02: REM simulation of the Bylinskii et al. (2015) context experiment (Fig. 2)

Bylinskii et al. (2015) measured the memorability of the same target scenes in two contexts, among
scenes of their own category (within category) and among scenes of all categories (across
categories). This module simulates both contexts in REM and draws the two panels of Fig. 2. It
computes the 30 numbers the paper reports about this simulation, in the Fig. 2 caption, the section
"Stable Memorability Across Contexts Does Not Require Intrinsic Memorability", the Methods section
"Bylinskii et al. (2015) Context Manipulation Simulation" and the sentences of the Introduction and of
"Direct Manipulations: Changing What an Item Is Studied or Tested With" that restate Fig. 2a
(`outputs/numbers.csv`), and tests T03 to T05 (`outputs/tests.csv`, collected in
`statistics/all_tests.csv`). The simulation generates its own items; `data/paper_run/` holds the saved
run that the paper reports.

## How to run

From the repository root:

    matlab -batch "restoredefaultpath; cd('02_bylinskii_simulation_fig2/code'); run_module('full')"
    python 02_bylinskii_simulation_fig2/code/stats_fig2.py

The MATLAB step needs the Statistics and Machine Learning Toolbox (MATLAB R2023b tested) and runs
serially (no parallel pool). On the tested laptop (Apple M2 Pro, 16 GB) it takes about 15 minutes in
one MATLAB process of about 1 GB. The Python step needs numpy and scipy, takes about 15 seconds and
reads only the saved run in `data/paper_run/`, so it also runs without MATLAB. It adds the rows of
the four Fig. 2 correlations to the `numbers.csv` that the MATLAB step writes, so it runs second.
`run_module` takes one optional argument:

| Call | What it does | Mode in `numbers.csv` | Time |
|---|---|---|---|
| `run_module` or `run_module('full')` | runs the simulation (seed 47826, 1,000 runs per context), draws Fig. 2 and writes `numbers.csv` from the new run, and compares the run field by field with the saved run in `data/paper_run/` | `rerun` | about 15 min |
| `run_module('skip_long')` | runs the same code with 10 runs per context only to check that it runs (its results are not kept), then draws Fig. 2 and writes `numbers.csv` from the saved run | `saved_run` | under 1 min |
| `run_module('report')` | no new simulation: draws Fig. 2, writes `numbers.csv` and repeats the comparison from the run that `run_module('full')` saved in `outputs/simulation_runs/`, and stops with an error if there is none | `rerun` | under 1 min |

`python run_all.py --skip-long` runs this module as `run_module('skip_long')`.

## Inputs

- `data/paper_run/fig2_paper_run.mat`: the saved run that the paper reports (seed 47826, 1,000 runs
  per context), as the MATLAB struct `figureData` with the memorability of the 1,575 targets in each
  context (`amt1_valid`, `amt2_valid`), their summed similarity (`summedSim_amt1`, `summedSim_amt2`),
  the targets tested in both contexts (`validBoth`), the correlations with their p values, and the
  parameters. `stats_fig2.py` computes tests T03 to T05 from it, `run_module('skip_long')` draws
  Fig. 2 from it, and `run_module('full')` and `run_module('report')` compare the full run with it.
- Shared code: `shared/matlab/release_paths.m` and `write_numbers_csv.m`, and
  `shared/python/release_paths.py`, `bootstrap.py`, `tests_table.py` and `numbers_table.py`
  (`write_numbers_csv.m` and `numbers_table.py` read the printed values from
  `shared/ledger/paper_numbers.csv`).

## What the code does

### The simulation (`bylinskiiRemSim.m`)

Each of the 21 categories has 300 items, 75 targets and 225 fillers (1,575 targets and 4,725
fillers in all). An item has 49 features, each a geometric draw with g = 0.65 plus 1 (values 1, 2,
3, ...). Features 1 to 10 are a category prototype, one draw of 10 values per category that all 300
members share; features 11 to 49 are drawn for every item separately.

Within category (Simulation 1, `AMT 1` in the code, criterion 20), each category gets 1,000 runs,
and each run studies 30 of the category's 75 targets and 90 of its 225 fillers. Across categories
(Simulation 2, `AMT 2`, criterion 5), each of 1,000 runs studies 30 of the 1,575 targets and 90 of
the 4,725 fillers, drawn from all categories.

A studied feature is stored with probability 0.3 and copied correctly with probability 0.6; a
miscopied feature gets a new geometric draw without the plus 1, so it can be 0, which counts as not
stored. When a probe is compared with a trace, each stored feature that does not match multiplies
the likelihood ratio by 1 - 0.6, and each matching feature of value v multiplies it by
(0.6 + 0.4 P(v)) / P(v), with P(v) = g (1 - g)^(v - 1). A response is "old" when the mean
likelihood ratio over the traces is strictly greater than the criterion. Only the 30 studied
targets of a run are probed, each against the run's 120 traces; the 90 fillers are never tested.
For false alarms, every run generates and stores 30 new lure items (geometric features, no category
prototype), and each studied target is probed against these 30 lure traces. Memorability is the hit
rate minus the false alarm rate. On average a target is tested 400 times within category and 19
times across categories, and all 1,575 targets are tested in both contexts.

Summed similarity is minus the mean Euclidean distance, over all 49 features, from a target to the
other 299 items of its category (within category) or to the other 6,299 items (across categories),
so higher values mean more similar, that is, less distinctive. Over the targets tested in both
contexts, the code saves four Spearman correlations (MATLAB `corr`; tied values share their average
rank): `rho_spearman` (memorability within against across, Fig. 2a), `rho_withinMem_sim` and
`rho_acrossMem_sim` (memorability against summed similarity in each context, the colors of
Fig. 2a), and `rho_sim` (summed similarity within against across, Fig. 2b). The saved run in
`data/paper_run/` stores p = 0 for these correlations; `outputs/tests.csv` gives the two-sided p values.

All random numbers come from one Mersenne Twister stream, set once with `rng(47826, 'twister')`;
naming the generator keeps the stream independent of the generator a MATLAB session used before.
The code runs serially (no `parfor`). For categories 1 to 21 it draws the 10 prototype values and
then the 300 x 49 item features (features 1 to 10 are then replaced by the prototype); then come
the within-category runs (category 1 to 21, 1,000 runs each) and the 1,000 across-category runs.
Each run draws `randperm` of the targets, `randperm` of the fillers, the storage of the 120 studied
items (`rand`, `rand`, `geornd`), the 30 lures (`geornd`) and their storage (`rand`, `rand`,
`geornd`). The summed similarities draw nothing. The number of runs, the output file and the seed
are inputs of `bylinskiiRemSim.m` (defaults 1,000, `outputs/simulation_runs/` and 47826), and every
run saves the study-list sizes (30 targets and 90 fillers) with its parameters.

### Fig. 2 (`bylinskii_makeFigure2.m`)

This function draws both panels from a saved run and stops with an error unless the run's summed
similarities are negative. In panel a, the color of a target is the mean of its z-scored summed
similarities in the two contexts, multiplied by 1.4 above 0 and divided by 1.4 below 0, on a scale
from -2 (orange, more distinctive) through green to 4 (dark blue, less distinctive). Panel b
subtracts the smallest plotted value (-14.884) from both summed similarities, so its axes run from 0
to 8.57, and adds the dashed line y = x. Both panels have a gray least-squares line, and each is a
separate 300 dpi PNG.

### Numbers and the check against the saved run (`bylinskii_numbers.m`)

This function writes 22 rows of `outputs/numbers.csv`, the design, parameters, figure panels and the
two sentences that restate Fig. 2a, from the run's parameters and per-target values. It recomputes the
four correlations from the per-target values and stops if one differs from the saved one by more than
1e-12. In modes `full` and `report` it then compares every field of the full run with
`data/paper_run/fig2_paper_run.mat`: per-target values, correlations and parameters must be identical,
and two p values count as equal when both are below 1e-15. With seed 47826 the run equals the saved
run in every field.

### Statistics (`stats_fig2.py`)

For each of the four correlations, the script gives Spearman ρ, n = 1,575 targets, the
two-sided p (t approximation with 1,573 degrees of freedom, as in scipy) and a 95% percentile
bootstrap interval, and writes these values to the 8 rows of `numbers.csv` that quote the four
correlations (mode `saved_run`). The bootstrap draws 10,000 resamples of the 1,575 targets with
replacement (seed 20261013, numpy `default_rng`) and uses the same resamples for all four
correlations. It resamples targets rather than simulation runs, because the memorability of each
target is already an average over its tests in the 1,000 runs per context. The p of the Fig. 2b
correlation lies below the smallest double-precision number, where scipy and MATLAB return 0, so the
script computes it through its logarithm and writes it as text (5.81097e-682). Every time it runs, the
script checks this route against scipy's p for the other three correlations (to 6 decimals of log10 p)
and checks the recomputed correlations against the saved ones (within 1e-12).

## Outputs

- `outputs/numbers.csv`: the 30 numbers, each with the printed value, the computed value and whether
  they agree at the printed precision (columns in the top-level README).
- `outputs/tests.csv`: tests T03 to T05, one row per correlation, with Spearman ρ, n = 1,575
  simulated targets, df = 1,573, the two-sided p (all below 0.001) and the 95% percentile
  bootstrap interval over targets (10,000 resamples, seed 20261013):

  | Test | Correlation over the 1,575 targets | ρ | 95% CI |
  |---|---|---|---|
  | T03 | memorability within against across categories (Fig. 2a) | 0.601 | [0.5649, 0.634] |
  | T04 | memorability against summed similarity, within category (Fig. 2a) | -0.246 | [-0.296, -0.1954] |
  | T04 | memorability against summed similarity, across categories (Fig. 2a) | -0.544 | [-0.580, -0.506] |
  | T05 | summed similarity within against across categories (Fig. 2b) | 0.929 | [0.920, 0.936] |

- `outputs/figures/fig2a_memorability_within_vs_across.png` and
  `outputs/figures/fig2b_summed_similarity_within_vs_across.png`: Fig. 2a and 2b.
- `outputs/figure_data/fig2a_memorability_within_vs_across.csv` and
  `outputs/figure_data/fig2b_summed_similarity_within_vs_across.csv`: the numbers each panel plots
  at full precision (17 significant digits), one row per target in the order the targets were
  generated (category 1 first).
- `outputs/simulation_runs/seed_47826_runs_1000_figureData.mat`: the run of `run_module('full')`,
  with the fields of the saved run plus the study-list sizes.
- `outputs/logs/`: the summary of the run (`run_summary_<mode>.txt`) and, in modes `full` and
  `report`, every field of the run against the saved run
  (`seed_47826_runs_1000_figureData_vs_paper_run.csv`), written when the code runs.

## Code

| File | What it does |
|---|---|
| `code/run_module.m` | entry point: runs the simulation or reads a saved run, then draws Fig. 2 and writes `numbers.csv` |
| `code/bylinskiiRemSim.m` | the REM simulation of both contexts |
| `code/bylinskii_makeFigure2.m` | Fig. 2a and 2b and the numbers each panel plots |
| `code/bylinskii_numbers.m` | `outputs/numbers.csv` (design, parameters, figure panels) and the comparison with the saved run |
| `code/stats_fig2.py` | tests T03 to T05 (`outputs/tests.csv`) and the `numbers.csv` rows of the four correlations |
