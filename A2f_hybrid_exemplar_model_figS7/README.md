# A2f: hybrid-similarity exemplar model (SI Appendix, Fig. S7)

This module tests whether the simulation results of the paper depend on REM's likelihood ratio. It
replaces REM with a hybrid-similarity exemplar model (Nosofsky and Zaki, 2003; Knapp, Nosofsky and Busey,
2006) and with the generalized context model (GCM; Nosofsky, 1986), keeps the design of Fig. 1, and runs
the basic simulation, the designed lists, a parameter sweep and the Bylinskii et al. (2015) context
simulation. It reproduces every number of "A Hybrid-Similarity Exemplar Model" in SI Appendix, Section 2,
the three panels of SI Appendix, Fig. S7, and the statements that point to them in the main text
(Implications of REM Simulations) and the General Discussion (Distinctiveness Relative to What?). Its
statistics are test T30 (`outputs/tests.csv`, collected in `statistics/all_tests.csv`).

## How to run

From the repository root:

    python A2f_hybrid_exemplar_model_figS7/code/run_module.py

On the tested laptop (Apple M2 Pro, 16 GB) this takes about 25 minutes: about 3 for `01_core.py`, 3.5 for
`02_sweep.py`, 17 for `03_bylinskii.py` and 2 for `bylinskii_reference_cells.py`, and about 25 seconds for
the last three steps. `02_sweep.py` and `03_bylinskii.py` use 2 worker processes; the largest process takes
about 1.6 to 2.0 GB (the sweep settings with 187-item lists), and the two sweep workers together about 3 to
4 GB. Requirements: Python 3 with numpy, scipy, pandas and matplotlib (tested with Python 3.12.4, numpy
1.26.4, scipy 1.13.1, pandas 2.2.2 and matplotlib 3.8.4).

| Option | What it does |
|---|---|
| `--skip-long` | skips the 83-setting Bylinskii run (about 8 minutes in all). `bylinskii_reference_cells.py` still runs the three Bylinskii settings whose values the paper reports (the reference, the lowest and the highest) exactly as the full run does. The count of settings above zero comes from `outputs/bylinskii.csv` if an earlier full run left it there (`mode` is then `rerun` in `numbers.csv`), else from `data/paper_run/bylinskii.csv` (`mode` is then `rerun_and_saved_run`: the three settings from this run, the count from the saved run) |
| `--provided-only` | runs no simulation: the comparison, the statistics and the figure use `outputs/`, and `data/paper_run/` for any file not in `outputs/` (under a minute) |

The same steps one at a time, in the order `run_module.py` runs them (each prints what it reads and
writes; `02_sweep.py` reads the calibrated widths from `outputs/core.json`, so `01_core.py` runs first):

    python A2f_hybrid_exemplar_model_figS7/code/01_core.py
    python A2f_hybrid_exemplar_model_figS7/code/02_sweep.py
    python A2f_hybrid_exemplar_model_figS7/code/03_bylinskii.py
    python A2f_hybrid_exemplar_model_figS7/code/bylinskii_reference_cells.py
    python A2f_hybrid_exemplar_model_figS7/code/check_against_paper_run.py
    python A2f_hybrid_exemplar_model_figS7/code/stats_figS7.py
    python A2f_hybrid_exemplar_model_figS7/code/make_figureS7.py

## Inputs

- None for the simulations, which generate their own items.
- `data/paper_run/`: the saved outputs of the run the paper reports, from the same three scripts
  (`core.json`, `core.txt`, `sweep.csv`, `sweep.txt`, `bylinskii.csv`, `bylinskii.txt`). A new run is
  compared with them, and the statistics and the figure read them for any file not in `outputs/`.
- The REM values beside the model, from the saved runs of two other modules (not the Fig. 1 run of
  module 01): the row labeled "paper values" (the Fig. 1 values: ρ = -0.6055, d′ = 1.6064) of
  `../A2a_rem_parameter_sweep_figS2/data/paper_run/rem_parameter_sweep.csv`, and CELL 2 (the designed lists
  at the Fig. 1 values) of `../A2d_list_context_figS5/data/paper_run/list_designed.txt`, whose rates are
  printed to three decimals; the red curves of Fig. S7b and c are differences of those values.
- Shared code: `shared/python/release_paths.py` (the module's folders), `bootstrap.py`, `tests_table.py`,
  `numbers_table.py` (which reads the printed values from `shared/ledger/paper_numbers.csv`) and
  `figure_style.py` (the figure style).

## What the code does

**The model** (`hgcm.py`). Items, encoding and lists follow Fig. 1: each item has 49 features drawn from a
geometric distribution (g = 0.65, values 1, 2, 3, ...), a simulated participant studies 40 items, each
feature is stored with probability 0.30 and copied correctly with probability 0.60 (otherwise replaced by
a new geometric draw), and 20 studied and 20 unstudied items are tested. Familiarity is summed similarity:

    similarity(probe, trace) = exp(-c d) x exp(gamma m - delta n)
    familiarity              = the sum of the similarities over the studied traces
    respond "old" when familiarity exceeds the criterion

Here d is the Euclidean distance between probe and trace over the features the trace stored, rescaled to
all 49 features (multiplied by the square root of 49 / the number stored), m is the number of stored
features on which probe and trace share a distinctive value, and n is the number on which they differ and
at least one value is distinctive (its weight delta is 0 in both models of the paper). gamma is 2 in the
hybrid model and 0 in the GCM. A value is distinctive when its frequency on that feature in a reference
population is below q = 0.10; in the paper's hybrid model (the variant "Hybrid GCM, list") the reference
is the participant's own 40 studied items, counted from the items as studied rather than from the noisy
traces, so a value carried by 3 of the 40 is distinctive and one carried by 4 is not. The scripts also run
three other references (world, the generating distribution; set, the 1,600 items; mix, half list and half
world). In `01_core.py` and in the Bylinskii calibration the variants draw in turn from one random stream,
so the numbers of the list variant depend on the variants run before it. An item's memorability is its hit
rate minus its false alarm rate, and its crowding (the summed similarity of Fig. S7a) is minus its mean
Euclidean distance to the other 1,599 items, so a negative Spearman ρ means isolated items are remembered
better.

**Basic simulation and designed lists** (`01_core.py`). The script generates 26,107 items and draws 1,600
of them (a numpy draw with the design of Fig. 1, not the item set of module 01). For each model it sets
c = 1 / (f x median inter-item distance), trying f = 0.03, 0.05, 0.08, 0.12, 0.18, 0.28, 0.42, 0.65 and
1.0 (median distance 8.83), and keeps the f whose overall d′ in a 600-experiment batch comes closest to
REM's 1.61: c = 1.4153 for the hybrid model and 2.2646 for the GCM. The criterion is the 85th percentile of
new-item familiarity in the same batch (a target false alarm rate of 0.15), 0.0322 for the hybrid model and
0.00106 for the GCM. The basic simulation runs 10,000 experiments (hybrid model and GCM: d′ 0.82 and 0.66,
false alarm rates 0.148 and 0.157). The designed lists take 200 critical items, each studied by 200
simulated participants per list. A list holds the critical item, its 0, 2, 4, 8 or 16 closest items after
its nearest neighbor, and fillers drawn for each participant from the far half of its distance ranking (40
items in all); the nearest neighbor, never studied, is the foil. Items at equal distances from the critical
item are ranked as numpy's default sort orders them (`environment.md`). d′ comes from the mean hit and false
alarm rates over the critical items, with rates kept within 1/400 and 1 - 1/400.

**Parameter sweep** (`02_sweep.py`). Each setting changes one value of the reference (g 0.65, w 49,
storage u 0.30, copy fidelity 0.60, list length 40, target false alarm rate 0.15, gamma 2, q 0.10,
delta 0) to u 0.15 or 0.45, copy fidelity 0.50, 0.70 or 0.85, list length 100 or 187, target false alarm
rate 0.05 or 0.30, gamma 0, 1 or 3, q 0.05 or 0.20, or delta 0.5, and a grid crosses g = 0.30, 0.40, 0.50
and 0.65 with w = 20, 30 and 49. A setting equal to the reference is not run twice, which leaves 27
settings, 26 with gamma above 0. Each runs 5,000 experiments on its own 1,600 items, with c held at the
reference value relative to the median inter-item distance and the criterion set in each setting at the
target false alarm rate (from a batch of 600 experiments), so overall accuracy moves with the parameters.

**Bylinskii context simulation** (`03_bylinskii.py`). The reference setting has the structure of the REM
version in SI Appendix, Fig. S3 (module A2b) at REM's standard values of g and copy fidelity: 21 categories
of 300 items with 10 category-defining features (one prototype draw per category), g 0.40, w 49, u 0.30,
copy fidelity 0.70, 75 targets per category (1,575 in all), lists of 30 targets and 90 fillers, 300
within-category lists per category, 6,000 across-category lists, and criteria at false alarm rates of 0.37
within and 0.05 across categories. The width is calibrated once per model at the reference setting, from
the candidates c = 1 / (f x median inter-item distance) with f = 0.03, 0.05, 0.08, 0.12, 0.18, 0.28, 0.42
and 0.65 (200 across-category lists each, criterion at a false alarm rate of 0.05): the calibration keeps
the candidate whose across-category d′ comes closest to REM's 2.6, which none reaches (1.58 for the hybrid
model, 1.09 for the GCM), and holds it fixed relative to the median inter-item distance in the other
settings. The other 18 settings change the criteria (0.25 / 0.05, 0.37 / 0.10 or 0.50 / 0.05, or one
criterion for both contexts, set at an across-category false alarm rate of 0.05, 0.15 or 0.30), the number
of category-defining features (3, 5, 15), categories (10, 40) or items per category (150 or 600, with 37
or 150 targets), gamma (1, 3), q (0.05, 0.20) or delta (0.5). They run for each of the four references,
plus the GCM at the reference and the six criterion settings (83 in all).

**Seeds.** `01_core.py` draws everything from one `numpy.random.default_rng(1)` stream in the order GCM,
world, set, list, mix (the items first, then for each variant its calibration, the basic simulation and
the designed lists), so it always runs whole. In `02_sweep.py` and `03_bylinskii.py` each setting has its
own generator, `default_rng(zlib.crc32(label))`, with the setting's label as the code builds it (in
`02_sweep.py`, its key in `cells()` followed by ` [world]`, ` [list]` or ` [mix]`, and no suffix for the set
population), so a setting reproduces on its own, the number of worker processes does not change any result,
and renaming a setting changes its numbers.
`03_bylinskii.py` calibrates the widths first with `default_rng(11)` (GCM, world, set, list, mix), and
`02_sweep.py` rebuilds the item set of `01_core.py` with `default_rng(1)` for its median distance.

**Statistics** (`stats_figS7.py`). The script checks `hgcm.familiarity` against the similarity rule written
with explicit loops, counts how many of the 40 studied items can share a value that still counts as
distinctive (3, with `hgcm.ref_table`), and checks that the per-item tables give back `core.json`. The
intervals are 95% percentile bootstrap intervals (10,000 resamples, `numpy.random.default_rng(20261013)`
for each) over the 1,600 simulated items (item-level ρ and overall d′), the 200 critical items (changes
from 0 to 16 neighbors) and the 1,575 targets (Bylinskii cross-context ρ); `tests.csv` holds them all and
`numbers.csv` the ones the paper prints. The two sweeps are counts and ranges of settings. There are no
p values, since a p value from simulated items reflects only how many items were simulated.

**Comparison with the saved run** (`check_against_paper_run.py`). Every value of `core.json`, `sweep.csv`,
`bylinskii.csv` and `bylinskii_reference_cells.csv` is compared with `data/paper_run/` (tolerance 1e-9),
the text reports line by line after removing carriage returns, and the calibrated d′ with the "Width
calibration" line of `data/paper_run/bylinskii.txt`. With the tested versions, on Apple silicon and on x86
processors without AVX-512 (`environment.md`), numpy draws the same random numbers as the saved run, every
value agrees within 1e-15, and the text reports agree line for line.

## Outputs

- `outputs/numbers.csv`: each number of the paper this module computes, with the printed value, the
  computed value and whether they agree at the printed precision (columns in the top-level README).
- `outputs/tests.csv`: T30 (17 rows): the item-level ρ and the overall d′ of the hybrid model and the GCM
  and their changes from 0 to 16 neighbors in hit rate, false alarm rate and d′, and the Bylinskii
  cross-context ρ, each with its bootstrap interval; the counts and ranges of the two sweeps.
- `outputs/figures/FigS7_hybrid_exemplar_model.png`: Fig. S7a to c (300 dpi).
- `outputs/figure_data/FigS7a.csv`, `FigS7b.csv`, `FigS7c.csv`: the numbers each panel plots, with sources.
- `outputs/core.json`, `core.txt`: basic simulation and designed lists of the five variants;
  `core_items_basic.csv` and `core_items_designed.csv`: their per-item counts (the input of the bootstrap).
- `outputs/sweep.csv`, `sweep.txt`: the parameter sweep (27 settings x 4 references, 108 rows).
- `outputs/bylinskii.csv`, `bylinskii.txt`: the Bylinskii simulation (83 rows); `bylinskii_calibration.csv`,
  `bylinskii_reference_cells.csv` and `bylinskii_reference_list_targets.csv`: the calibration, the three
  settings and each target's counts in the reference setting.
- `outputs/check_against_paper_run.csv`: every compared value from this run and the saved run.
- `outputs/logs/`: one log per step and `run_module_steps.log`, written when the code runs.

## Code

| File | What it does |
|---|---|
| `code/run_module.py` | runs the seven steps of "How to run" in order |
| `code/hgcm.py` | the model: items, noisy encoding, the four references for rarity, summed hybrid similarity, crowding |
| `code/01_core.py` | the basic simulation (10,000 experiments) and the designed lists (0 to 16 close neighbors) for the GCM and the four hybrid variants; calibrates c and the criterion |
| `code/02_sweep.py` | the parameter sweep: 27 settings for each of the four references (5,000 experiments each) |
| `code/03_bylinskii.py` | the Bylinskii context simulation: 19 settings for each of the four references plus 7 GCM settings |
| `code/bylinskii_reference_cells.py` | the Bylinskii width calibration and the three settings whose values the paper reports (the reference, the lowest, the highest), run as in `03_bylinskii.py`; each target's counts in the reference setting |
| `code/check_against_paper_run.py` | compares this run with `data/paper_run/` |
| `code/stats_figS7.py` | `numbers.csv` (29 ledger items) and `tests.csv` (T30); the check of `hgcm.familiarity` |
| `code/make_figureS7.py` | Fig. S7 and its plotted numbers |
| `code/figS7_inputs.py` | the readers that `stats_figS7.py` and `make_figureS7.py` share (model results, REM values) |
