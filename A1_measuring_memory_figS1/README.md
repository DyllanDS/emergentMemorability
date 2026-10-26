# A1: Measuring memory (SI Appendix, Fig. S1)

This module reproduces the image-set results of SI Appendix, Section 1 (Measuring Memory) and Fig. S1; the THINGS
results of that section come from module A3b. In the three image-set experiments (450 participants), responding
was close to neutral (c = 0.29), and corrected recognition (hit rate minus false alarm rate) orders the concepts
almost exactly as d_a (r = 0.98) and d′ (r = 0.96) do. Methods (Experimental
Procedures and Analysis) reports the same c and r, and the General Discussion (Measurement and Model Predictions)
refers to this result. For the statement in Simulations of Memorability as an Emergent Property that the analyses
hold with d′ or d_a, the module also computes the same agreement across the 1,600 items of the Fig. 1 REM
simulation. It writes one row for each of the 25 numbers and statements of these passages to `outputs/numbers.csv`
and the statistics of tests T21 (the criterion) and T22 (the correlations) to `outputs/tests.csv`, which
`statistics/all_tests.csv` collects. Everything is computed from data in the repository, mostly the trials of the
image-set experiments (module 06) and the saved Fig. 1 run (module 01).

## How to run

From the repository root:

    python A1_measuring_memory_figS1/code/run_module.py

`run_module.py` runs the four steps listed below in order and stops at the first step that fails. The whole module
takes about 1 minute on the tested laptop (Apple M2 Pro, 16 GB), about 25 s for the MATLAB step, 10 s for the
Python checks, 1 s for the figure and 25 s for the statistics. The MATLAB step needs about 2 GB of memory at its
peak, because it holds the four 1,600 x 10,000 arrays of the Fig. 1 run, and the Python steps need less than 1 GB.

| Option | What it does |
|---|---|
| `--skip-matlab` | leaves out `criterion_dprime_da.m`. `criterion_dprime_da_checks.py` then takes every number from its own Python recomputation, which equals the MATLAB values to about 1e-15, and `numbers.csv` names the script that computed each value. `python run_all.py --skip-matlab` runs the module this way. |
| `--resamples N` | the number of bootstrap resamples in `stats_figS1.py` (default 10,000; `tests.csv` records the number used, and the intervals in `numbers.csv` stop the run on a mismatch only at 10,000) |

The four steps can also be run one at a time:

    matlab -batch "restoredefaultpath; cd('A1_measuring_memory_figS1/code'); criterion_dprime_da"
    python A1_measuring_memory_figS1/code/criterion_dprime_da_checks.py
    python A1_measuring_memory_figS1/code/make_figureS1.py
    python A1_measuring_memory_figS1/code/stats_figS1.py

`criterion_dprime_da_checks.py` stops if the MATLAB output `outputs/paperStats_criterionDpDa.mat` is missing, unless
it is given `--without-matlab`. With that option it takes every number from its own recomputation and still
compares them with a MATLAB output already in `outputs/`. The last two steps read the tables that
`criterion_dprime_da_checks.py` writes. The MATLAB step needs the Statistics and Machine Learning Toolbox (tested
with R2023b). `run_module.py` looks for MATLAB in the environment variable `MATLAB`, then on the PATH, then in the
standard install folders (R2023b first), and starts it with `restoredefaultpath` so that no folder on the user's
MATLAB path can replace a MATLAB function.

## Inputs

- `06_image_set_experiments_fig6_fig7/data/maintained_trials.csv`, `changed_trials.csv` and `reduced_trials.csv`
  (module 06): the 602 de-identified sessions of the three image-set experiments, one row per study or test trial
  (participant, block, phase, trial_in_block, image_filename, concept, things_image_index, image_number, is_new,
  response_1to6, rt_ms). The scripts read them through `shared/matlab/load_image_set_trials.m` and
  `shared/python/imageset_loader.py`. These loaders give each set a dataset name, which the MATLAB script prints
  and saves: representativeImages for the maintained set, flipMemorabilityImages for the changed set and
  equatedImages for the reduced set.
- `shared/things/concept_info_human.mat` (MATLAB) and `shared/things/concept_table.csv` (Python): the names of the
  1,854 THINGS concepts.
- `01_rem_simulation_fig1/data/paper_run/fig1_paper_run.mat` (module 01): the Fig. 1 run, with the familiarity
  odds of every item in each of its 10,000 simulations (`simHitOdds`, `simFaOdds`) and the masks of the simulations
  that tested the item as studied or as new (`simHitMask`, `simFaMask`).
- `03_inversion_experiment_fig3/data/inversion_trials.csv` and `04_rare_feature_faces_fig4/data/face_trials.csv`:
  read only to count the old items at test in those two experiments (Methods).
- `data/paper_run/paperStats_criterionDpDa.mat`: the saved results of the image-set analysis that the paper reports
  (MATLAB struct `paperStats`). Both scripts compare `criterion_c`, `dprime_overall`, `overallHR`, `overallFAR`,
  `r_hitFA_dp` and `r_hitFA_da` of each run with it.
- Shared code: `shared/matlab/release_paths.m`, `load_concept_info.m`, `load_image_set_trials.m`,
  `apply_participant_exclusions.m` and `sdt_rates.m`; `shared/python/release_paths.py`, `imageset_loader.py`,
  `inversion_loader.py`, `face_loader.py`, `concept_tables.py`, `sdt.py`, `numbers_table.py` (it reads the printed
  values from `shared/ledger/paper_numbers.csv`), `tests_table.py`, `bootstrap.py` and `figure_style.py`.

## What the code does

### Participants and trials

Part 1 of `criterion_dprime_da.m` and `criterion_dprime_da_checks.py` apply the exclusions in Methods through
`shared/matlab/apply_participant_exclusions.m` and `shared/python/imageset_loader.py`. A participant is excluded
for pressing one key on more than 80% of trials, for more than 20% of trials under 200 ms, for more than 20% of
trials over 7,500 ms, or for a hit rate minus false alarm rate below 0.05. Trials under 200 ms or over 10,000 ms are
then dropped, and the first 150 participants of each set in sorted id order are kept, which removes nobody. Of the
602 sessions, 152 are excluded (maintained 55, changed 45, reduced 52), which leaves 150 participants per set and
282,100 test trials.

### Scores

A response of 1 to 3 on the six-point scale (1 = sure old, 6 = sure new) counts as old. Each concept is scored
separately in each image set, from the kept trials of that set's 150 participants, as corrected recognition, d′ and
d_a. For d′, each rate is kept inside [1/(2n), 1 - 1/(2n)], with n the concept's number of old (or new) test
trials, and a concept with fewer than 2 old or new trials has no d′. The d_a of a concept comes from the counts of
the six responses to its old and new images. At least two of the five cumulative boundaries must give finite z
values, and the rates at the middle boundary (responses 1 to 3) enter d_a = sqrt(2 / (1 + s²)) (z(H) - s z(F)) with
the slope s fixed at 0.80 (Didi-Barnea et al., 2016; Macmillan & Creelman, 2004). Of the 2,554 concept-by-set
scores, the 2,453 with a finite d′ and d_a are analyzed. The other 101 are left out, 93 because a hit rate of 1 or
a false alarm rate of 0 at the middle boundary makes d_a infinite and 8 because fewer than two boundaries give
finite z values. An analyzed score rests on a median of 35 old and 36 new trials.

### Criterion and correlations

The criterion c = -(z(H) + z(F)) / 2 uses the mean hit rate H (0.7174) and the mean false alarm rate F (0.1229) of
the 2,453 scores. Each mean is kept inside [0.5/40, 1 - 0.5/40] (no effect at these rates), which gives c = 0.2928,
and the scripts also compute c for each set. Over the 2,453 scores, corrected recognition correlates with d_a at
r = 0.9762 and with d′ at r = 0.9579 (Pearson r), and the scripts also compute both correlations within each set.
The parametric p values of these correlations are below the smallest double-precision number, so MATLAB and scipy
return 0, and they are reported as p < 0.001 (t(2,451) = 222.7 for d_a). `criterion_dprime_da_checks.py` also
measures the z-ROC slope of the confidence ratings. This is the least-squares slope of z(hit rate) on z(false alarm
rate) over the five cumulative boundaries of the confidence counts of all kept trials, leaving out a boundary with
a rate of 0 or 1. The slope is 0.592; the script rounds it to 0.59 before it computes d_a at that slope, and
corrected recognition correlates with this d_a at r = 0.9836.

### REM simulation

Part 2 of `criterion_dprime_da.m` reads module 01's Fig. 1 run by name and stops with an error if the file is
missing. An item is called old when its familiarity odds exceed 1, the criterion of the simulation. For each of the
1,600 items, the hit rate and the false alarm rate are kept inside [1/(2n), 1 - 1/(2n)], with n the number of
simulations that tested the item as studied (or as new), 125 on average (88 to 167). Each item's z-ROC slope is
fitted over the criteria 0.5, 0.75, 1, 1.5 and 2 (median 0.73), and its d_a uses that slope. An item tested fewer
than twice in either role would be left out; no item of the Fig. 1 run is. Across the 1,600 items, corrected
recognition correlates with d′ at r = 0.989 and with d_a at r = 0.969 (`numbers.csv` row L1-013).

### Comparisons

`criterion_dprime_da_checks.py` recomputes Parts 1 and 2 in Python and stops with an error, after writing its
outputs, when a value differs from the MATLAB run by more than 1e-9 or when a row of `numbers.csv` does not agree
with the paper. Both scripts compare the six image-set values with the saved run in `data/paper_run/` (a difference
above 1e-9 gives a warning in MATLAB and an error in Python).

### Statistics (T21, T22)

The first three steps draw no random numbers. `stats_figS1.py` follows the convention of `statistics/README.md`:
95% percentile intervals from 10,000 resamples, seed 20261013, two-sided, no correction for multiple tests.
Correlations over concept scores resample the concept scores, as the Statistical Reporting paragraph of Methods
describes. The criterion c is a bias measure and its own effect size, so T21 gives it as an estimate without a p
value, with an interval that resamples participants. One generator, `numpy.random.default_rng(20261013)`, draws
in every resample, for each set in the order maintained, changed, reduced, `generator.integers(0, 150, 150)` over
the 150 kept participants of the set in sorted id order. The 2,453 analyzed concept scores are the fixed design,
and their hit and false alarm rates are recomputed from the drawn participants (the script first checks its
recomputed scores against `outputs/concept_scores.csv`). The same draws give the interval of the measured z-ROC
slope. The interval of each correlation in T22 resamples the 2,453 concept-by-set scores with
`percentile_bootstrap` of `shared/python/bootstrap.py` (its own `default_rng(20261013)`, the same draws for the
three correlations). `stats_figS1.py` then adds the four rows that print an interval (L5-080, L6-012, L6-013,
L6-015) to `outputs/numbers.csv`.

## Outputs

- `outputs/numbers.csv`: each number and statement of the paper this module computes (ledger ids L1-013, L4-018,
  L5-077, L5-078, L5-080, L5-081, L6-001 to L6-016 and L6-201 to L6-203), with the printed value, the
  computed value and whether they agree at the printed precision (columns in the top-level README).
- `outputs/tests.csv`: T21, c with its participant-bootstrap interval; T22, the Pearson r of corrected
  recognition with d_a at slope 0.80, with d′ and with d_a at slope 0.59, each with its concept-score bootstrap
  interval, and the measured z-ROC slope with its participant-bootstrap interval.
- `outputs/stats_figS1.txt`: the numbers of participants and trials, the check of the recomputed concept scores,
  and the estimate, bootstrap mean and interval of the criterion and the z-ROC slope.
- `outputs/figures/FigS1_criterion_dprime_da.png`: Fig. S1 (300 dpi).
- `outputs/figure_data/FigS1_points.csv` and `FigS1_fit_and_statistics.csv`: the 2,453 plotted points, and the gray
  least-squares line with the r, p and n printed in the panel.
- `outputs/concept_scores.csv`: the 2,453 analyzed concept-by-set scores with image_set, concept_number (THINGS
  order), concept, old_trials, new_trials, hit_rate, false_alarm_rate, corrected_recognition, d_prime,
  d_a_slope080, d_a_measured_slope (slope 0.59), and the counts of the six responses to old (old_resp1 to
  old_resp6) and new (new_resp1 to new_resp6) images.
- `outputs/criterion_table.csv`: hit rate, false alarm rate, d′ and c from the mean concept rates, overall
  and per set.
- `outputs/rem_item_scores.csv`: the 1,600 REM items, with the number of simulations that tested each item as
  studied and as new, its hit rate, false alarm rate, corrected recognition, d′, z-ROC slope and d_a.
- `outputs/paperStats_criterionDpDa.mat`: the MATLAB results (struct `paperStats`): the image-set statistics
  (`criterion_c`, `dprime_overall`, `overallHR`, `overallFAR`, `r_hitFA_dp`, `r_hitFA_da` and their p values), the
  REM correlations of the Fig. 1 run (`r_hitFA_dp_rem`, `r_hitFA_da_rem` and their p values), the counts, the
  per-set values and further REM values (Spearman rho, mean rates, median z-ROC slope).
- `outputs/criterion_dprime_da.txt`: every number of `criterion_dprime_da_checks.py`, as text.
- `outputs/logs/`: the printout of each step, written when `run_module.py` runs.

## Code

| File | What it does |
|---|---|
| `code/run_module.py` | runs the four steps below in order |
| `code/criterion_dprime_da.m` | Part 1, the image-set experiments: concept scores, c and the correlations, pooled and per set. Part 2, the REM correlations of the Fig. 1 run. Saves `outputs/paperStats_criterionDpDa.mat` and compares it with the saved run |
| `code/criterion_dprime_da_checks.py` | the same numbers in Python, the measured z-ROC slope and d_a at that slope, the criterion table, the old items at test, the comparisons with the MATLAB run and the saved run, and `numbers.csv` |
| `code/make_figureS1.py` | Fig. S1 from `outputs/concept_scores.csv` |
| `code/stats_figS1.py` | the statistics T21 and T22, and the rows of `numbers.csv` with an interval |
