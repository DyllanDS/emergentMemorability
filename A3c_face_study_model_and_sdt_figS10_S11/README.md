# A3c: REM's predictions and the preregistered d_a analysis of the rare-feature face study (Fig. 4c; SI Appendix, Figs. S10 and S11b)

This module reproduces the face study of SI Appendix, Section 3 ("The Model Predicted Each Preregistered
Face Result" and "The Preregistered Discriminability Analysis of the Face Study") and the main-text and
Methods sentences that refer to it. It runs REM through the face design at the 729 parameter settings
behind the preregistered predictions (AsPredicted #311640) and again at the 243 settings of the run after
data collection, scored as the data were, and sets both beside the data (SI Appendix, Fig. S10). The
preregistered analysis of d_a is an unequal-variance fit of the confidence ratings, drawn as the confidence
ROCs of Fig. 4c; the module also gives the effects of the lure set and of the feature itself with their
intervals over participants and over the six rare features (SI Appendix, Fig. S11b), and the refits that
leave out one rare-feature pair. The data are those of module 04 (`04_rare_feature_faces_fig4/data/`),
which also draws SI Appendix, Fig. S11a. The module writes 44 rows of `outputs/numbers.csv` and tests T48
to T51 of `statistics/all_tests.csv`.

## How to run

From the repository root:

    python A3c_face_study_model_and_sdt_figS10_S11/code/run_module.py

Without MemToolbox2 (see Inputs) the module takes about 4 to 5 minutes on the tested laptop (Apple M2 Pro,
16 GB), most of it in two simulations of about 2 minutes each, and uses at most about 170 MB of memory per
step. It needs Python 3 with numpy, scipy, pandas and matplotlib. `run_module.py` runs the twelve scripts
from `rem_final_spec.py` to `numbers_face_study.py` in the order of the Code table, each in its own process
with a log in `outputs/logs/`, and stops at the first failure; each script can also be run alone from the
repository root, in the same order.

| Option | What it does |
|---|---|
| `--provided-only` | uses the saved fits in `data/provided/` even when MemToolbox2 is present |
| `--skip-long` | with MemToolbox2, leaves out the 243 fits of the run after data collection |

With MemToolbox2 in place, `run_module.py` also runs three sets of fits into `outputs/`: the point fit
(about 1.5 minutes), the seven leave-one-pair-out fits (about 7 minutes on 2 workers) and the 243 fits of
the run after data collection (several hours). `check_provided.py` compares each with the saved fit (they
agree when every d_a quantity is within 0.001, the precision of the reports), and the module's numbers,
tables and figures still come from the saved fits. Each can also be run alone:

    python A3c_face_study_model_and_sdt_figS10_S11/code/sdt_shared_criteria.py --boot 0
    python A3c_face_study_model_and_sdt_figS10_S11/code/leave_one_pair_out.py
    python A3c_face_study_model_and_sdt_figS10_S11/code/rem_like_data.py --workers 2

`run_module.py` does not run the two bootstraps, and the module uses their saved resamples in both modes.
The participant bootstrap (10,000 refits) takes 7.4 hours on 22 workers, or about 3.4 days on the default 2
workers; the bootstrap over the faces (1,000 refits) takes about 2 hours on 20 workers. Run by hand, they
write to `outputs/` and never to `data/provided/`:

    python A3c_face_study_model_and_sdt_figS10_S11/code/bootstrap_parallel.py --workers 2
    python A3c_face_study_model_and_sdt_figS10_S11/code/bootstrap_items.py --boot 1000 --workers 2 --modes items

## Inputs

- `04_rare_feature_faces_fig4/data/face_trials.csv` and `face_participants.csv` (described in module 04's
  README), read through `shared/python/face_loader.py`: 111 sessions, of which the first 100 to pass the
  preregistered rule (at least 4 of the 6 repeats detected, at most 5 presses on faces that were not
  repeats) form the sample. `code/face_study_common.py` gives the fitting scripts each participant's 36 test
  trials in the order they were run, with the participants in `load_order`, the order of the registered
  analysis, on which the draws of the registered bootstrap depend.
- `data/paper_run/rem_final_spec.txt`: the saved report of the registered sweep, the output the paper
  reports (its lines end in CR LF; the comparison ignores line ends).
- `data/provided/`: the saved results of the unequal-variance analysis, from which every d_a value comes.

| File | What it holds |
|---|---|
| `point_fit.json` | the registered fit: mean and SD of each kind of face (never-studied faces are N(0, 1)), the 13 parameters by name, the log likelihood (-5836.65) and the five d_a quantities |
| `sdt_shared_criteria.txt` | the registered report: the three d_a values with their intervals over 10,000 participant resamples, and the effect of the lure set ("INTERACTION") and of the feature itself ("WHAT THE FEATURE BUYS") with interval and bootstrap p |
| `bootstrap_raw.jsonl` | the 10,000 participant resamples, one line each: the resample number and its five refitted d_a quantities |
| `bootstrap_items.jsonl`, `bootstrap_items.txt` | the 1,000 resamples of the faces and their report |
| `leave_one_pair_out.json`, `leave_one_pair_out.txt` | the seven fits (all pairs, and without each of the six pairs) and their report |
| `rem_like_data.jsonl`, `rem_like_data.txt` | the 243 settings of the run after data collection: the parameters, the d_a scores read off the evidence ("direct") and fitted like the data ("fitted"), and their report |

- Not included: the MemToolbox2 Python package (its `UVSDModel`, `ByCell` and `MLE`), which the fits need
  and which is not publicly available (EXTERNAL_INPUTS.md, item M5). It goes in
  `shared/memtoolbox2/` (as `memtoolbox/` or `python/memtoolbox/`); a copy elsewhere on the Python
  path is not used. Without it the fitting scripts print `MISSING PACKAGE: MemToolbox2 ...` and
  `PROVIDED OUTPUT: regenerating needs ...`, and the module uses the saved results in `data/provided/`.
  `check_provided.py` recomputes every number of those reports from the saved fits and resamples, and it
  evaluates the registered model at the saved parameters on the data in this repository: the log
  likelihood equals the saved one (-5836.647653), so the saved fit was made on these data.
- Shared code in `shared/python/`: `face_loader.py` (with `sdt.py`), `bootstrap.py`, `tests_table.py`,
  `numbers_table.py` (with the paper's values in `shared/ledger/paper_numbers.csv`), `figure_style.py` and
  `release_paths.py`.
- `04_rare_feature_faces_fig4/outputs/`: module 04's `rate_contrasts.csv` and `numbers.csv`, for the
  equality check of `observed_contrasts.py`. Module 04 runs first (`run_all.py` keeps this order), and the
  check stops the module when one of these files is missing.

## What the code does

The scripts call the effect of the lure set, d_a(rare vs never-studied) minus d_a(rare vs same-feature
lure), "interaction" (preregistered prediction 3), and the effect of the feature itself, d_a(rare vs
same-feature lure) minus d_a(ordinary vs never-studied), "equivalence" or "what the feature buys".

1. `rem_final_spec.py` runs REM (Shiffrin & Steyvers, 1997) through the design at 729 settings: g {0.3,
   0.4, 0.5}, w {15, 20, 25}, u {0.2, 0.3, 0.4}, c {0.6, 0.7, 0.8}, the number of features carrying the
   rare value {1, 3, 5}, and a criterion placed so that a share fa {0.2, 0.42, 0.6} of unrelated new faces
   is called old. At each setting 500 simulated participants study 38 fillers, 6 rare-feature faces and 6
   ordinary faces, and are tested on 36 probes: 3 studied rare faces, 3 never-studied partners carrying
   the rare feature, 3 studied ordinary faces and 3 never-studied ordinary partners (the same 3 of the 6
   pairs of each kind are tested with the studied face), 12 studied fillers and 12 unrelated new faces.
   Feature values follow REM's geometric distribution with parameter g; the rare value is the smallest
   value whose base rate g(1 - g)^(v - 1) is below 0.02, and no other face carries it on the features it
   occupies. Setting i (0 to 728, in `itertools.product` order) draws from numpy `default_rng(i)`. The
   script compares its report with `data/paper_run/rem_final_spec.txt` line for line.
2. `rem_729_rates.py` runs the same simulation with the same seeds and saves, for every setting, the rates,
   the hit and false alarm differences of SI Appendix, Fig. S10a, and the d_a contrasts read off the
   evidence (mean and SD, ddof 1, of the log odds). The hit prediction fails at 17 settings, each with both
   hit rates between 0.873 and 1.000.
3. `roc_points.py` counts the 3,600 test trials of the 100 participants at each of the eight rating steps
   (the old or new answer crossed with the 1 to 4 confidence rating; 1 = surest new, 8 = surest old) for
   the four kinds of face (300, 300, 1,500 and 1,500 trials), and writes the seven operating points of each
   kind for Fig. 4c: the shares of trials rated 2 or higher, 3 or higher, ..., 8.
4. `observed_contrasts.py` computes the hit and false alarm differences of Fig. S10a with the registered
   participant bootstrap (10,000 resamples; resample i draws its participants with numpy `default_rng(i)`,
   participants in `load_order`). It stops unless its values equal those in module 04's outputs.
5. `sdt_shared_criteria.py` fits the registered unequal-variance model by maximum likelihood to the
   ratings pooled over participants: never-studied ordinary faces N(0, 1); a mean and SD for studied
   ordinary faces, studied rare faces and same-feature lures; one set of seven criteria (the first
   criterion and six steps) shared by all four kinds of face; 13 parameters. The fit starts from 12 points
   (the package's own, then copies multiplied by exp(z), z normal with SD 0.4 from `default_rng(seed)`;
   seed 0 for the point fit) and restarts from the best solution and three copies jittered with SD 0.25
   (`default_rng(seed + 1)`) until the log likelihood improves by less than 0.01, for up to six rounds.
   `bootstrap_parallel.py` refits 10,000 participant resamples. All of them are drawn from one
   `default_rng(0)` stream in serial order before the work is shared out, so the number of workers changes
   no resample; refit i uses seed i and starts from the full-sample solution and three copies of it
   jittered with SD 0.2.
6. `leave_one_pair_out.py` prints the hit and false alarm rate of each rare-feature pair and refits the
   model seven times (seed 0): with all pairs, and without each pair's rare-feature trials, keeping every
   ordinary face. `bootstrap_items.py` resamples, with replacement, the 6 rare-feature pairs, the 59
   studied ordinary faces and the 59 never-studied faces (resample i draws from `default_rng(10000 + i)`;
   refit seed i).
7. `rem_like_data.py` runs REM again at the 243 settings of the grid without the criterion axis, 100
   simulated participants each, seed = setting number. It places one set of seven criteria on the pooled
   log odds at the cumulative proportions of the participants' rating use (0.0869, 0.2244, 0.3450, 0.3719,
   0.3847, 0.4533, 0.6278) and, with MemToolbox2, fits the simulated ratings as the data were fitted (fit
   seed i for setting i). The part without the fit runs in both modes, and `check_provided.py` stops the
   module unless it reproduces the saved direct scores.
8. `check_provided.py` recomputes every number of the saved reports from the saved fits and resamples (35
   checks without MemToolbox2) and writes the summary tables that the figures, statistics and numbers read.
9. `make_figureS10.py` and `make_figureS11.py` draw the figures. The gray points of Fig. S10 are spread
   sideways at random, with `default_rng(1)` to `default_rng(4)`, one per strip; their sideways position
   carries no information.
10. `stats_face_study.py` writes tests T48 to T51, and `numbers_face_study.py` writes `outputs/numbers.csv`.

Every bootstrap interval is the 2.5th and 97.5th percentile of the resampled values (numpy `percentile`,
linear interpolation). The bootstrap p of the registered analysis (question 5 of the preregistration,
`04_rare_feature_faces_fig4/data/PREREG_SUBMITTED.md`) is two-sided, twice the smaller of the shares of
resamples at or below zero and at or above zero. With 10,000 resamples the smallest p above zero is
0.0002 (0.002 with 1,000), and a p of 0 means that no resample reached zero.

## Outputs

- `outputs/numbers.csv`: each number of the paper this module computes (44 rows), with the printed value,
  the computed value and whether they agree at the printed precision (columns in the top-level README).
- `outputs/tests.csv`: tests T48 to T51 (19 rows), each with its statistic, n, df, p, effect size, 95%
  interval and how the interval was computed: T48, the counts of settings at which each prediction holds
  (descriptive); T49, the hit and false alarm differences and the lure-set effect with the registered
  bootstrap behind the error bars; T50, the three d_a values; T51, the two effects with their intervals
  over participants and over the six rare features and their leave-one-pair-out ranges. No correction for
  multiple comparisons is applied.
- `outputs/figures/Fig4c_confidence_roc.png` (Fig. 4c), `FigS10_rem_face_design.png` (SI Appendix,
  Fig. S10) and `FigS11b_lure_set_and_feature_effects.png` (SI Appendix, Fig. S11b), at 300 dpi, and the
  plotted numbers in `outputs/figure_data/` (`fig4c_*.csv`, `figS10a_*.csv`, `figS10b_*.csv`,
  `figS11b_contrasts.csv`).
- Intermediate tables and reports: `rem_final_spec.txt` (equal to the saved report in `data/paper_run/`),
  `rem_729_rates.csv` and its check, `rem_like_data_model_side.csv` and its check (the direct scores of the
  243 settings beside the saved ones), `rating_counts_by_face_kind.csv`, `roc_operating_points.csv` and
  `rating_counts_check.txt`, `observed_contrasts.csv`, `provided_checks.csv` and `check_provided.txt`,
  `registered_fit_summary.csv` (the five d_a quantities with every interval, bootstrap p and
  leave-one-pair-out range) and `rem_243_summary.csv` (the fitted scores of the 243 settings).
- With MemToolbox2, the new fits add `point_fit.json`, `sdt_shared_criteria.txt` (the point fit only),
  `leave_one_pair_out.json` and `.txt`, and `rem_like_data.jsonl` and `.txt`.

## Code

| File | What it does |
|---|---|
| `code/run_module.py` | runs the scripts below, from `rem_final_spec.py` to `numbers_face_study.py`, in order |
| `code/rem_final_spec.py` | the preregistered REM sweep at 729 settings |
| `code/rem_729_rates.py` | the same sweep, one row per setting, for Fig. S10a |
| `code/roc_points.py` | the rating counts and the operating points of Fig. 4c |
| `code/observed_contrasts.py` | the observed hit and false alarm differences |
| `code/sdt_shared_criteria.py` | the registered unequal-variance fit (needs MemToolbox2) |
| `code/leave_one_pair_out.py` | the rates of each rare-feature pair and the leave-one-pair-out refits (the refits need MemToolbox2) |
| `code/rem_like_data.py` | REM at 243 settings, scored like the data (the fits need MemToolbox2) |
| `code/check_provided.py` | recomputes every saved number and writes the summary tables |
| `code/make_figureS10.py` | SI Appendix, Fig. S10 |
| `code/make_figureS11.py` | Fig. 4c and SI Appendix, Fig. S11b |
| `code/stats_face_study.py` | tests T48 to T51 |
| `code/numbers_face_study.py` | `outputs/numbers.csv` |
| `code/bootstrap_parallel.py` | the registered participant bootstrap, 10,000 refits (not run by `run_module.py`) |
| `code/bootstrap_items.py` | the bootstrap over the faces, 1,000 refits (not run by `run_module.py`) |
| `code/face_study_common.py` | what the scripts share: each participant's test trials, the MemToolbox2 switch, the readers of the saved reports |
