# A2b: Category Structure and Decision Criteria (SI Appendix, Fig. S3)

This module reproduces SI Appendix, Section 2, Category Structure and Decision Criteria, and SI Appendix, Fig. S3.
It repeats the REM simulation of the Bylinskii et al. (2015) context manipulation (Fig. 2a) 18 times, varying one
factor at a time: the decision criteria of the two contexts, the number of category-defining features, the number
of categories and the number of items per category. For each simulation it computes the Spearman correlation,
across target items, between memorability (hit rate minus false alarm rate) in the within-category context and in
the across-category context, and the Pearson correlation between each target's summed similarity within its
category and across the whole set. It writes the statistics of test T26 (collected in `statistics/all_tests.csv`)
and the paper's numbers with ledger ids L6-036 to L6-044 and L6-046.

## How to run

From the repository root:

    python A2b_category_structure_criteria_figS3/code/run_module.py

On the tested laptop (Apple M2 Pro, 16 GB) the module takes about 6.5 minutes: about 3.5 minutes for the MATLAB
simulations (peak memory about 2.4 GB), about 3 minutes for the statistics and a few seconds for the figure and
the comparison. It needs MATLAB (tested with R2023b) with the Statistics and Machine Learning Toolbox, without a
parallel pool, and Python 3 with numpy, scipy and matplotlib. `run_module.py` runs the four steps below in order
and writes a log of each step to `outputs/logs/` as it runs.

| Command | What it does | Time |
|---|---|---|
| `python A2b_category_structure_criteria_figS3/code/run_module.py` | the whole module | about 6.5 minutes |
| `python A2b_category_structure_criteria_figS3/code/run_module.py --skip-matlab` | steps 2 to 4 on the simulation outputs already in `outputs/` | about 3 minutes |
| `matlab -batch "restoredefaultpath; cd('A2b_category_structure_criteria_figS3/code'); bylinskii_robustness(true)"` | step 1 alone | about 3.5 minutes |
| `python A2b_category_structure_criteria_figS3/code/make_figureS3.py --paper-run` | Fig. S3 from the saved run in `data/paper_run/`, without MATLAB (written as `FigS3_bylinskii_robustness_paper_run.png` and `FigS3_paper_run.csv`) | seconds |
| `python A2b_category_structure_criteria_figS3/code/check_against_paper_run.py --paper-run` | `outputs/numbers.csv` from the saved run, without MATLAB (mode `saved_run`; it replaces the `numbers.csv` of a full run, and the command above must run first, because the row of Fig. S3, L6-041, describes the figure from its figure data) | seconds |

`--skip-long` is accepted (`run_all.py --skip-long` passes it to this module) and changes nothing, because this
module has no long step. `run_module.py` finds MATLAB from the environment variable `MATLAB`, else `matlab` on
the PATH, else a standard installation, and starts it with `restoredefaultpath` for that session, so that folders
added to the MATLAB path cannot replace MATLAB's own functions (a `tcdf` from another toolbox, such as
FieldTrip's, returns 0 for the smallest p values).

## Inputs

- None for the simulations: `code/bylinskii_robustness.m` generates the items.
- `data/paper_run/bylinskii_robustness.csv`: the saved run of the 18 simulations, the output the paper reports
  (one row per simulation, the columns of `outputs/bylinskii_robustness.csv`, CRLF line endings).
  `code/check_against_paper_run.py` compares this run with it, and the `--paper-run` options use it in place of
  a MATLAB run.
- `shared/matlab/release_paths.m`; `shared/python/release_paths.py`, `figure_style.py`, `bootstrap.py`,
  `numbers_table.py` and `tests_table.py`; `shared/ledger/paper_numbers.csv` (the numbers of the paper).

The module needs no input from outside the repository.

## What the code does

**Step 1, the simulations** (`code/bylinskii_robustness.m`). Every simulation uses REM with g = 0.40 and copy
fidelity c = 0.70 (the standard values of Shiffrin and Steyvers, 1997), storage probability u = 0.30 and 49
features.

- Items: each category has a prototype of the category-defining features (each a geometric draw with parameter
  g, plus 1) that every member shares, and the other features are drawn for each item (geometric plus 1). The
  first quarter of each category's items are the targets and the rest are fillers (75 of 300 items;
  `round(nPerCat / 4)` when the number of items per category varies, so 38 of 150).
- The reference cell has 21 categories of 300 items, 10 category-defining features and criteria of 20 within and
  5 across categories. In file order, the 18 simulations are the reference cell; criteria (within / across)
  20 / 5, 10 / 2.5, 40 / 10, 5 / 5, 20 / 20, 1 / 1 and 50 / 50; 3, 5, 10 and 15 category-defining features;
  10, 21 and 40 categories; and 150, 300 and 600 items per category. Each sets one factor and keeps every other
  setting of the reference cell, so five simulations (the reference cell, "crit 20 / 5", 10 features, 21
  categories and 300 items per category) have the settings of the reference cell: they share the category
  structure and criteria of Fig. 2a and differ only in their random draws.
- Encoding: each feature of a studied item is miscopied with probability 1 - c (a new geometric value plus 1)
  and then stored with probability u (otherwise 0).
- Familiarity is the mean, over the list's 120 traces, of the REM likelihood ratio (Shiffrin and Steyvers,
  1997). A matching stored value v multiplies a trace's ratio by (c + (1 - c) P(v)) / P(v), with
  P(v) = g (1 - g)^(v - 1), a mismatching stored value multiplies it by 1 - c, and a trace with nothing stored
  has ratio 1. The response is "old" when the familiarity is above the criterion of the context.
- Within-category context: for each category, 250 study lists of 30 of its targets and 90 of its fillers. Every
  studied target is tested, and so are up to 30 of the category's targets that were not studied in that list
  (the foils). Across-category context: 5,000 study lists of 30 targets and 90 fillers drawn from all categories,
  with foils from all targets not studied in the list. In the reference cell each target is tested on average
  100 times as old and 100 times as new within its category, and about 95 times each across categories.
- Measures: memorability is hit rate minus false alarm rate in each context, and rho is the Spearman correlation
  of the two over targets (MATLAB `corr`, average ranks for ties, p from the t approximation with n - 2 degrees
  of freedom). Summed similarity within is minus the mean Euclidean distance to the other items of the target's
  category, and across is minus the mean Euclidean distance to all items of the set (the target included, at
  distance 0); simCorr is their Pearson correlation. dWithin and dAcross are z(mean hit rate) - z(mean false
  alarm rate), with the rates kept within 0.001 and 0.999. Every target is tested as old and as new in both
  contexts, so every target enters the correlations.
- Random numbers: `rng(11, 'twister')` once, then the 18 simulations serially in file order. The draws of a
  simulation depend on every simulation before it, so one simulation reproduces only as part of the whole grid,
  and the grid must not run in a parallel pool. Before `rng`, the script warns when a function it uses comes
  from outside MATLAB's own folders; this check and the per-target file (`saveItems`, default true) draw no
  random numbers.

**Step 2, the statistics** (`code/stats_figS3.py`, test T26). From the per-target file, the script first checks
that it gives back each simulation's rho, p and simCorr (rho and r within 1e-12, p within 1e-6 of its value).
Each Spearman rho of memorability then gets n, df = n - 2, its two-sided p (t approximation) and a 95% percentile
bootstrap interval over target items, and each Pearson r of summed similarity gets the same bootstrap
interval. Every bootstrap draws 10,000 resamples of the target items with `numpy.random.default_rng(20261013)`,
the same seed in every simulation, so each interval reproduces on its own;
within a simulation the memorability and summed-similarity bootstraps use the same resamples, and the category
structure stays fixed. In 16 simulations the p of the summed-similarity correlation is below the smallest positive
double (about 1e-308), so it is computed through logarithms (the regularized incomplete beta function, as in
module 02) and written as text. Three summary rows give the range and the sign count over all 18 simulations,
over the four with equal criteria and for the summed-similarity correlation. The 18 simulations form a sweep,
and no correction for multiple comparisons is applied.

**Step 3, the figure** (`code/make_figureS3.py`). Fig. S3 has one row per simulation, grouped by the factor
varied and labeled by its value (criteria as within / across), with a point at the Spearman rho of memorability
across contexts, a dashed line at zero, and the number of simulations above zero with the largest p. The five
simulations with the category structure and criteria of Fig. 2a are black, and the first of them carries the
label "Fig. 2a Design".

**Step 4, the comparison and the paper's numbers** (`code/check_against_paper_run.py`). The script compares this
run with the saved run in `data/paper_run/`: the 18 rows with their labels and settings, and every numeric column
(within 1e-6, and within 1e-6 of the value for pRho). On the tested laptop, 250 of the 252 numeric values of
this run are identical to the last written digit (15 significant digits), and the summed-similarity correlations
of "crit 1 / 1" and "categories = 40" differ by 1e-15. The script also checks that the per-target file gives back
each simulation's rho, p, simCorr and number of targets, and then writes `numbers.csv`.

## Outputs

- `outputs/numbers.csv`: each number of the paper this module computes (L6-036 to L6-044 and L6-046), with
  the printed value, the computed value and whether they agree at the printed precision (columns in the
  top-level README).
- `outputs/tests.csv`: test T26, 39 rows. For each simulation, the Spearman rho of memorability across contexts
  (18 rows) and the Pearson r of summed similarity (18 rows), each with its bootstrap interval; then three
  summary rows, whose intervals are ranges over simulations and not confidence intervals.
- `outputs/bylinskii_robustness.csv`: one row per simulation with the columns label, critWithin, critAcross,
  catFeatures, nCat, nPerCat, rho, pRho, simCorr, hitWithin, faWithin, hitAcross, faAcross (mean rates over
  targets), dWithin and dAcross. The first row, labeled "reference (crit 20 / 5)", is the reference cell.
- `outputs/bylinskii_robustness.txt`: a printed report of the same table.
- `outputs/bylinskii_robustness_items.csv`: one row per simulation and target (29,748 rows) with the old trials,
  hits, new trials and false alarms of the target in each context and its summed similarity in each context
  (17 significant digits). Each simulation has 1,575 targets, except those with 10 categories (750), 40
  categories (3,000), 150 items per category (798) and 600 items per category (3,150).
- `outputs/equality_with_paper_run.csv`: this run against the saved run, simulation by simulation and column by
  column, and the checks of the per-target file.
- `outputs/figures/FigS3_bylinskii_robustness.png` (300 dpi): Fig. S3.
- `outputs/figure_data/FigS3.csv`: the plotted values, one row per point from top to bottom (group, row label,
  simulation, y position, rho, pRho, marker).
- `outputs/logs/`: one log per step and the progress log of the simulations (`bylinskii_progress.log`, one line
  per finished simulation), written when the code runs.

## Code

| File | What it does |
|---|---|
| `code/run_module.py` | Runs steps 1 to 4 in order. |
| `code/bylinskii_robustness.m` | Step 1: the 18 simulations; writes the table, the report and the per-target file. |
| `code/stats_figS3.py` | Step 2: the statistics of test T26. |
| `code/make_figureS3.py` | Step 3: Fig. S3 and its figure data. |
| `code/check_against_paper_run.py` | Step 4: the comparison with the saved run, and `outputs/numbers.csv`. |
| `code/robustness_tables.py` | Readers of the tables and the grouping of the 18 simulations (used by steps 2 to 4). |
