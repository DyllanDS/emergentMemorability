# A2d: The same item on crowded and uncrowded lists (SI Appendix, Fig. S5)

This module reproduces SI Appendix, Fig. S5 (panels a and b) and the numbers of SI Appendix, Section 2 (List
Context). Two REM simulations at the parameter values of Fig. 1 compare test items studied on crowded and uncrowded
40-item lists. On random lists (600 lists, 120 simulated participants each), close neighbors appear on a list by
chance, and test items with three or more close neighbors on their list are compared with test items that have
none. On designed lists (300 critical items, 200 simulated participants per list), the same item is studied on five
lists that carry 0, 2, 4, 8 or 16 of its next most similar items (Fig. S5 compares 16 with 0), and its most similar
item, never studied, is its lure. The module writes the change from the uncrowded to the crowded list in hit rate,
false alarm rate, hit rate minus false alarm rate and d′, with bootstrap intervals (test T28, which
`statistics/all_tests.csv` collects), the figure with its plotted numbers, and the 14 numbers of these passages in
`outputs/numbers.csv`.

## How to run

From the repository root:

    python A2d_list_context_figS5/code/run_module.py

`run_module.py` runs the seven steps listed below in order and stops at the first step that fails. The module takes
about 2 minutes on the tested laptop (Apple M2 Pro, 16 GB), about 50 s for `list_designed.m` and 35 s for
`list_binned.m`, each including the start of MATLAB, and about 15 s for the Python steps. The MATLAB session uses
about 1 GB of memory.

| Option | What it does |
|---|---|
| `--skip-matlab` | leaves out the two MATLAB simulations and uses their outputs in `outputs/`, which ship with the repository. `python run_all.py --skip-matlab` runs the module this way. |

The steps can also be run one at a time, in this order:

    matlab -batch "restoredefaultpath; cd('A2d_list_context_figS5/code'); list_designed"
    matlab -batch "restoredefaultpath; cd('A2d_list_context_figS5/code'); list_binned"
    python A2d_list_context_figS5/code/check_against_paper_run.py
    python A2d_list_context_figS5/code/list_bootstrap.py
    python A2d_list_context_figS5/code/stats_figS5.py
    python A2d_list_context_figS5/code/make_figureS5.py
    python A2d_list_context_figS5/code/write_numbers.py

The simulations need MATLAB with the Statistics and Machine Learning Toolbox (tested with R2023b) and start no
parallel pool; the Python steps need numpy, scipy and matplotlib. `run_module.py` looks for MATLAB in the
environment variable `MATLAB`, then on the PATH, then in the standard install folders (R2023b first). It starts
MATLAB with `restoredefaultpath` (for that session only), so that no folder on the user's MATLAB path can replace a
MATLAB function; FieldTrip's `tcdf.m`, for example, makes the p values of the within-item correlations in the
printed tables print as 0. The two MATLAB scripts warn when a function they use comes from outside MATLAB. The files
in `outputs/` come from MATLAB R2023b (23.2.0, Apple silicon), Python 3.12.4, numpy 1.26.4, scipy 1.13.1 and
matplotlib 3.8.4.

## Inputs

- None for the simulations, which generate their own items.
- `data/paper_run/list_designed.txt`, `data/paper_run/list_binned.txt`: the printed tables of the saved run of the
  two simulations, the output the paper reports. `check_against_paper_run.py` compares the tables of a new run with
  them line by line, and `list_bootstrap.py` and `make_figureS5.py` check the per-item outputs and the plotted
  changes against their CELL 2 rows. Modules A2e and A2f read CELL 2 of `list_designed.txt` for the REM values of
  SI Appendix, Figs. S6, S7b and S7c.
- From `shared/`: `python/release_paths.py` (the module's folders), `python/figure_style.py` (figure style),
  `python/numbers_table.py` with `ledger/paper_numbers.csv` (`numbers.csv`), `python/tests_table.py` (`tests.csv`)
  and `matlab/release_paths.m` (the output folder of the MATLAB scripts).

## What the code does

### The two simulations

`list_designed.m` and `list_binned.m` use the same REM as `rem_parameter_sweep.m` of module A2a. Each calls
`rng(1, 'twister')` once (the Mersenne Twister) and then runs two parameter cells serially in file order, first
CELL 1, with the standard values of Shiffrin and Steyvers (1997; g 0.40, 20 features, copy fidelity 0.70), then
CELL 2, with the parameter values of Fig. 1 (g 0.65, 49 features, storage probability 0.30, copy
fidelity 0.60, criterion 1.0). Fig. S5 uses CELL 2, and because both cells draw from one stream, CELL 2 reproduces
only when CELL 1 runs before it in the same MATLAB session. Each cell first draws a database of 1,600 simulated
items, each feature value a geometric draw with parameter g plus 1, and closeness is the Euclidean distance between
two items' feature lists. At study, each feature of a studied item is stored with probability u, and a stored
feature is copied correctly with probability c or else replaced by a new geometric value counted from 1, which gives
one trace per studied item. At test, the familiarity of an item is its mean likelihood ratio over the list's 40
traces, and a simulated participant says "old" when the familiarity is above the criterion.

### Designed lists

`list_designed.m` draws 300 critical items at random and ranks the other items of the database by their distance
from each one. The nearest item is the critical item's lure (tested, never studied), and the next 16 (neighbor ranks
2 to 17) are its similar items. Each critical item is studied on five 40-item lists that hold 0, 2, 4, 8 or 16 of
its similar items, and the other slots are filled at random, without replacement, from the far half of its distance
ranking (the code's `far`, 800 candidates). The code takes the critical items in turn and each item's lists in the
order 0, 2, 4, 8, 16; for each list it draws the fillers (`randperm`), then storage (`rand`), copy accuracy (`rand`)
and the replacement values (`geornd`). Each list is studied by 200 simulated participants, who are tested on the
critical item and its lure. The d′ of a condition is z(mean hit rate over the 300 critical items) minus z(mean false
alarm rate over their lures), with both rates kept within 1/400 and 399/400. Fig. S5 uses the "0 -> 16 similar
items" line of CELL 2. For each condition the printed table also lists the mean crowding (minus the mean distance
from the critical item to the other items of its list) and the response criterion, -0.5 x (z(hit rate) + z(false
alarm rate)), and it gives within-item Spearman correlations of crowding with the hit and false alarm rates.

### Random lists

`list_binned.m` builds 600 lists. Each list draws 60 different items at random from the 1,600 (`randperm`), 40 to
study and 20 lures, and is studied by 120 simulated participants, who are tested on all 60 items; storage, copy
accuracy and replacement values are then drawn as on the designed lists. An item's close neighbors are its 50
nearest items in the database, a set that depends on the database and not on any list. The script counts the close
neighbors on the list over a studied item's 39 list companions, and over the 40 studied items for a lure. Test
items are binned by this count (0, 1, 2, 3 or more), and Fig. S5 compares the bin of 3 or more (2,812 studied items
and 1,499 lures) with the bin of 0 (6,825 studied items and 3,291 lures), the "0 -> 3+" line of CELL 2. The hit and
false alarm rates of a bin are means over its items, and d′ comes from these means, with the rates kept within
0.0001 and 0.9999. The printed table also lists the mean number of close neighbors per item and a within-item
version, which centers each item's count and rates on its own means over the lists it appeared on.

### Bootstrap intervals

`list_bootstrap.py` first checks that the per-item outputs of CELL 2 (`list_designed_cell2.mat`,
`list_binned_cell2.mat`) reproduce the rates, counts and d′ of the saved CELL 2 tables at their three printed
decimals. It then computes the 95% percentile bootstrap intervals that Fig. S5 plots as error bars (2.5th and 97.5th
percentiles, numpy's default linear interpolation), from 2,000 resamples drawn by one generator,
`numpy.random.default_rng(20261010)`. The 300 critical items are resampled first (`integers(0, 300, 300)` per
resample, an item's five lists kept together), then the 600 lists (`integers(1, 601, 600)` per resample, a list's
40 studied items and 20 lures kept together, and a list drawn twice counted twice). Each resample recomputes d′ from
its mean rates. `stats_figS5.py` repeats the bootstrap with `numpy.random.default_rng(20261013)` and 10,000
resamples per kind of list, in the same order, which gives the intervals printed in the text, and writes both sets
of intervals to `tests.csv`. These are
simulations, so `tests.csv` gives no p values; the effect size is the change itself, and no correction for multiple
comparisons is applied.

### Check against the saved run

`check_against_paper_run.py` compares every line of the two printed tables with the saved run in `data/paper_run/`
and stops with an error when a line of CELL 2 differs. A difference elsewhere, or a difference only in the p values
of the within-item correlations, is printed and marked in `outputs/equality_with_paper_run.csv` but does not stop
the run. A run with MATLAB R2023b reproduces both tables line for line.

## Outputs

- `outputs/numbers.csv`: the 14 numbers of the paper this module computes (ledger items L6-069 to L6-080, L6-082 and
  L6-281), each with the paper's value, the computed value and whether they agree at the printed precision (columns
  in the top-level README). Ten rows are judged against a printed number; the other four (L6-070, L6-073, L6-074,
  L6-075) record the parameter values, the changes that a sentence of the text describes and the two figure panels.
- `outputs/tests.csv`: test T28, 16 rows. For each kind of list and each of the four changes (crowded minus
  uncrowded), a row gives n (300 critical items or 600 lists), the 95% percentile interval, the resampling unit,
  the number of resamples (2,000, the error bars of Fig. S5, or 10,000) and the seed.
- `outputs/figures/FigS5_list_context.png`: Fig. S5, panels a and b (300 dpi).
- `outputs/figure_data/FigS5a.csv`, `FigS5b.csv`: the change and interval of each plotted bar, one row per bar.
- `outputs/list_designed.txt`, `outputs/list_binned.txt`: the printed tables of the two simulations (both cells).
- `outputs/list_designed_cell2.mat` (MATLAB v7): for CELL 2, `hitRate` and `faRate` (300 critical items x 5
  conditions), `crowd` (minus the mean distance from the critical item to the other items of its list), `crit` (the
  database numbers of the critical items, not the criterion) and `parameters` (the cell's settings and the seed).
- `outputs/list_binned_cell2.mat` (MATLAB v7): for CELL 2, `oItem`, `oHit`, `oCnt` and `oList` for the 24,000
  studied items and `lItem`, `lFA`, `lCnt` and `lList` for the 12,000 lures (database number, hit or false alarm
  rate, number of close neighbors on the list, list number), and `parameters`.
- `outputs/equality_with_paper_run.csv`: every line of the two tables of this run beside the same line of the saved
  run in `data/paper_run/`, and whether the two are identical.
- `outputs/list_bootstrap.txt`, `outputs/list_bootstrap.json`: the uncrowded and crowded rates and d′ of each kind
  of list, and the four changes with their intervals from 2,000 resamples.
- `outputs/logs/`: one log per step, written when `run_module.py` runs.

## Code

| File | What it does |
|---|---|
| `code/list_designed.m` | designed-list simulation: 300 critical items, each studied on five lists with 0, 2, 4, 8 or 16 of its close neighbors; prints the table of both cells and saves each critical item's rates and the settings of CELL 2 |
| `code/list_binned.m` | random-list simulation: 600 lists, test items binned by their number of close neighbors on the list; prints the table of both cells and saves every test item's rate, count and list number of CELL 2 |
| `code/check_against_paper_run.py` | compares the two printed tables with the saved run in `data/paper_run/`, line by line |
| `code/list_bootstrap.py` | the changes from the uncrowded to the crowded list and their intervals from 2,000 resamples (seed 20261010); also holds the table readers and bootstrap functions of the next scripts |
| `code/stats_figS5.py` | test T28: the changes with their intervals from 2,000 resamples (from `list_bootstrap.py`) and from 10,000 resamples (seed 20261013, the intervals of the text) |
| `code/make_figureS5.py` | Fig. S5 and its plotted numbers |
| `code/write_numbers.py` | `outputs/numbers.csv` |
| `code/run_module.py` | runs all of the above in order |
