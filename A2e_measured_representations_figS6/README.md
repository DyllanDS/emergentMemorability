# A2e: REM in the measured SPoSE space (SI Appendix, Fig. S6)

This module runs the simulation of SI Appendix, Section 2 (Measured Representations) and draws SI Appendix,
Fig. S6. Each simulated item is one of 1,600 THINGS concepts (Hebart et al., 2019), and its features are the
concept's 49 SPoSE values (Hebart et al., 2020). The model keeps REM's likelihood ratio and changes two things,
because SPoSE values are continuous numbers: how the rarity of a value is judged (against the distribution of
that feature across the 1,600 concepts) and how a probe value is matched to a stored one (through Gaussian
retention noise, the one parameter the model adds). The module runs the simulation, compares it with the saved
run in `data/paper_run/`, adds the bootstrap intervals drawn in Fig. S6, and computes the 20 numbers and
statements the paper reports about this simulation (`outputs/numbers.csv`) and test T29 (`outputs/tests.csv`,
collected in `statistics/all_tests.csv`). The white bar of Fig. S6 (REM with its generated features, at the
parameters of Fig. 1) comes from the saved run of module A2d, and the ρ of -0.55 for REM's generated features
(the run of Fig. 1) from the saved run of module 01.

## How to run

From the repository root:

    python A2e_measured_representations_figS6/code/run_module.py

On the tested laptop (Apple M2 Pro, 16 GB) this takes about 1.5 minutes: about 1 minute for the MATLAB
simulation, about 25 seconds for `stats_figS6.py` and a few seconds for each of the other steps. With
`--skip-matlab`, `run_module.py` leaves out the simulation and uses the MATLAB outputs already in `outputs/`
(it stops if they are missing); `python run_all.py --skip-matlab` runs the module this way. The same steps one
at a time, in the order `run_module.py` runs them:

    matlab -batch "restoredefaultpath; cd('A2e_measured_representations_figS6/code'); continuous_rem_spose(1)"
    python A2e_measured_representations_figS6/code/bootstrap_ci.py
    python A2e_measured_representations_figS6/code/stats_figS6.py
    python A2e_measured_representations_figS6/code/make_figureS6.py
    python A2e_measured_representations_figS6/code/check_against_paper_run.py

The simulation needs MATLAB R2023b with the Statistics and Machine Learning Toolbox and runs serially (no
parallel pool). The Python steps need numpy, scipy and matplotlib (tested with Python 3.12.4, numpy 1.26.4,
scipy 1.13.1 and matplotlib 3.8.4). `run_module.py` finds MATLAB from the environment variable `MATLAB`, else
on the PATH, else in a standard installation folder (preferring R2023b). It starts MATLAB with
`restoredefaultpath` for that session only, so that no folder on the user's MATLAB path can replace a MATLAB
function the simulation uses, and `continuous_rem_spose.m` prints a warning when one of its functions comes
from outside MATLAB's own toolboxes.

## Inputs

- `shared/things/spose_embedding_49d_sorted.txt`: the 49-dimensional SPoSE embedding of the 1,854 THINGS
  concepts, one row per concept (Hebart et al., 2020; CC BY 4.0, `shared/things/LICENSE_spose.txt`), and
  `shared/things/unique_id.txt`, the concept names in the same row order (for `outputs/concept_subset.csv`).
- `data/paper_run/continuous_rem_spose.txt`: the saved report of the run the paper reports.
  `check_against_paper_run.py` compares this run with it, and `make_figureS6.py` checks the gray bars against
  it.
- `../A2d_list_context_figS5/data/paper_run/list_designed.txt` (CELL 2) for the white bar of Fig. S6, and
  `../01_rem_simulation_fig1/data/paper_run/fig1_paper_run.mat` (the saved run of Fig. 1) for the ρ of -0.55
  with REM's generated features. A copy of this module needs these two folders beside it.
- Shared code: `shared/matlab/release_paths.m`, and `shared/python/release_paths.py`, `bootstrap.py`,
  `tests_table.py`, `numbers_table.py` (which reads the printed values from `shared/ledger/paper_numbers.csv`)
  and `figure_style.py`.

Every input ships with the repository.

## What the code does

**The simulation** (`continuous_rem_spose.m`). The rarity of a value t on dimension j, f_j(t), is the density
of that dimension over the 1,600 concepts, estimated with MATLAB `ksdensity` (normal kernel, default
bandwidth) on 512 points, read off by linear interpolation (extrapolated beyond the grid) and floored at 1e-9.
A studied feature is stored with probability 0.30 and, once stored, copied correctly with probability
c = 0.60. A correct copy keeps the studied value plus Gaussian retention noise, whose standard deviation
sigma_j is the noise fraction times the standard deviation of that dimension across the 1,600 concepts; an
incorrect copy takes that dimension's value from a concept drawn at random. A probe value x is compared with a
stored value t through lambda_j = 1 + c (N(t - x; sigma_j) / f_j(t) - 1), floored at 1e-12, where N is the
normal density of the retention noise. The likelihood ratio of a trace is the product of lambda_j over its
stored features, familiarity is the mean of that product over the 40 traces of the study list, and the model
responds "old" when familiarity is above 1.0. The study and test design and the storage and copy values are
those of Fig. 1. The script runs both analyses below at four noise fractions, 0.02, 0.03, 0.05 and 0.08, in
that order. Fig. S6 shows the first three; the results at 0.08 are in the report, the item files and
`outputs/continuous_rem_bootstrap_ci.csv`.

Between items, each noise level has 2,500 simulated study-test experiments with one simulated participant
each. An experiment draws 60 different concepts, studies the first 40 and tests 20 of them as old along with
the other 20 as new, so each concept is tested about 31 times as old and 31 times as new (all 1,600 are tested
both ways). A concept's memorability is its hit rate minus its false alarm rate, and its local crowding is
minus its mean correlation distance (1 minus the Pearson correlation of two concepts' SPoSE vectors) to its 20
nearest concepts. The statistic is the Spearman ρ between the two across the 1,600 concepts (MATLAB `corr`;
tied values share their average rank). The report also gives the correlation with global crowding (minus the
mean correlation distance to all other concepts) and the d′ of the mean hit and false alarm rates, with each
rate kept between 0.001 and 0.999.

On the designed lists (Fig. S6), 120 critical concepts are drawn at random. For each of them the other
concepts are ranked by correlation distance, and the nearest one is the lure, which is never studied. The
crowded list holds the critical concept, its 2nd to 17th nearest concepts (the 16 close neighbors) and 23
concepts drawn without replacement from the farther half of the ranking; the other list holds the critical
concept and 39 concepts from the farther half. Each list is studied by 100 simulated participants, and each of
them is tested once on the critical concept and once on its lure. The d′ change is d′ with 16 neighbors minus
d′ with none, each computed from the mean hit and false alarm rates across the 120 critical concepts, with
every rate kept between 1/200 and 199/200. The last column of the report gives the same change in hit rate
minus false alarm rate.

All random numbers come from one Mersenne Twister stream, set once with `rng(1, 'twister')`, and the script
runs serially. The draw of 1,600 of the 1,854 concepts (`randperm`) comes first. Then each noise level in turn
runs its 2,500 between-item experiments and then its designed lists. An experiment draws its 60 concepts and
the 20 studied concepts it tests with `randperm`, then the storage of its study list (`rand` for storage,
`rand` for correct copying, and for each feature `randn` for the noise and `randi` for the concepts that give
incorrect copies). The designed lists draw the 120 critical concepts with `randperm`, then for each critical
concept and list (the list with no neighbors first) a `randperm` of the farther half and the same storage draws
for the 100 participants. The densities, distances and crowding draw no random numbers, and every setting is
written to `outputs/continuous_rem_settings.csv`.

**Bootstrap intervals** (`bootstrap_ci.py`). The error bars of Fig. S6 are 95% percentile bootstrap intervals
from 2,000 resamples of the 120 critical concepts (with replacement), with the d′ change recomputed on each
resample with the same rate limits. One generator, numpy `default_rng(20261010)`, serves the whole script and
draws for each noise level in turn (0.02, 0.03, 0.05, 0.08) first 2,000 resamples of the 1,600 concepts, which
no output uses, and then the 2,000 resamples of the critical concepts. The script also gives each correlation
and its p from `scipy.stats.spearmanr` (average ranks for ties; two-sided p from the t approximation).

**Statistics** (`stats_figS6.py`). For each noise level of Fig. S6, test T29 gives the correlation with an
interval from 10,000 resamples of the concepts, and the d′ change with the interval drawn in the figure and a
second interval from 10,000 resamples of the critical concepts. Each 10,000-resample interval uses a new
generator, `default_rng(20261013)`, so all noise levels are resampled with the same 10,000 sets of concepts (or
of critical concepts). The same resamples give intervals for the change in the false alarm rate to the lure and
in the hit rate of the critical concept. The white bar is listed with its value and without an interval (its
item bootstrap is test T28 of module A2d), and a count gives how many gray-bar intervals lie below it. Spearman
p values are two-sided, and no correction for multiple comparisons is applied.

**Figure and check** (`make_figureS6.py`, `check_against_paper_run.py`). `make_figureS6.py` draws Fig. S6 from
`outputs/continuous_rem_bootstrap_ci.csv` and stops if a gray bar differs from the saved report at its 3
printed decimals. The white bar is CELL 2 of the designed lists of module A2d at the Fig. 1 values (300
critical items, 200 simulated participants per list). `check_against_paper_run.py` compares the run with the
saved report, computes the ρ of the run of Fig. 1 from module 01's saved run (memorability against the negated
mean Euclidean distance to the other items, across its 1,600 simulated items), writes `outputs/numbers.csv`,
and stops with an error if a number of the saved report is not reproduced or a ledger number does not match.

## Outputs

- `outputs/numbers.csv`: the 20 numbers and statements of the paper this module computes or documents (ledger
  items L6-083 to L6-089, L6-091 to L6-097, L6-099, L6-100 and L6-301 to L6-304), each with the printed value,
  the computed value and whether they agree at the printed precision (columns in the top-level README).
- `outputs/tests.csv`: test T29, 17 rows. At every noise level the Spearman ρ has n = 1,600 concepts, df = 1,598
  and a two-sided p below 0.001 (the largest 6.3e-75), and the d′ change has n = 120 critical concepts and no p.

  | Noise | ρ | 95% CI (10,000 resamples) | d′ change | 95% CI, Fig. S6 (2,000 resamples) | 95% CI (10,000 resamples) |
  |---|---|---|---|---|---|
  | 0.02 | -0.4352 | [-0.475, -0.393] | -0.414 | [-0.502, -0.310] | [-0.504, -0.314] |
  | 0.03 | -0.488 | [-0.527, -0.448] | -0.436 | [-0.551, -0.322] | [-0.548, -0.324] |
  | 0.05 | -0.523 | [-0.560, -0.483] | -0.341 | [-0.418, -0.264] | [-0.418, -0.265] |

  From 0 to 16 close neighbors, the false alarm rate to the lure rises by 0.052 [0.035, 0.069], 0.092 [0.068,
  0.119] and 0.115 [0.087, 0.144], and the hit rate changes by 0.000 [-0.005, 0.005], 0.015 [0.008, 0.021] and
  0.020 [0.011, 0.029] (10,000 resamples). With either set of intervals, all three gray-bar intervals lie below
  the white bar (d′ change -0.132).
- `outputs/figures/FigS6_continuous_feature_rem.png`: Fig. S6 (300 dpi).
- `outputs/figure_data/FigS6.csv`: the numbers Fig. S6 plots, one row per bar (value, interval and source).
- `outputs/continuous_rem_spose.txt`: the report of the run, with the settings, the surprisal line (Spearman ρ
  of each concept's surprisal, minus the summed log density of its 49 values, with global and local crowding),
  the results table at each noise level and closing notes.
- `outputs/concept_subset.csv`: the 1,600 concepts (`item_number`, the row of the item files;
  `concept_row_in_embedding`; `concept`).
- `outputs/continuous_rem_settings.csv`: every setting as run.
- `outputs/continuous_rem_items_sigma<noise>.csv` (noise 0.02, 0.03, 0.05 and 0.08): one row per concept, with
  its hit and false alarm counts, its numbers of old and new tests, and its local and global crowding.
- `outputs/continuous_rem_designed_sigma<noise>.csv`: one row per critical concept, with its hit and false
  alarm rates with 0 and with 16 close neighbors.
- `outputs/continuous_rem_bootstrap_ci.csv` and `.txt`: the point values and p values, the 2,000-resample
  intervals of the d′ change, and the mean rates on the designed lists, at each noise level.
- `outputs/equality_with_paper_run.csv`: compares this run with the saved run in `data/paper_run/`, the output
  the paper reports: every number of the results table, the design, storage and surprisal lines, the whole
  report line by line (counted, without stopping the check), and the point values of the Python steps.
- `outputs/logs/`: one log per step, written when `run_module.py` runs.

## Code

| File | What it does |
|---|---|
| `code/run_module.py` | runs the five steps in order and stops at the first that fails |
| `code/continuous_rem_spose.m` | the simulation: the 1,600 concepts, the densities, crowding, the between-item experiments and the designed lists at each noise level |
| `code/bootstrap_ci.py` | the error bars of Fig. S6 (2,000 resamples, seed 20261010) |
| `code/stats_figS6.py` | test T29 (`outputs/tests.csv`) |
| `code/make_figureS6.py` | Fig. S6 and the numbers it plots |
| `code/check_against_paper_run.py` | the comparison with the saved run, and `outputs/numbers.csv` |
