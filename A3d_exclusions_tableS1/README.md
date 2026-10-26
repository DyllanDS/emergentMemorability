# A3d: Image-set results with and without the exclusions (SI Appendix, Table S1)

This module reproduces the part of SI Appendix, Section 3 titled *The Image Set Results Do Not Depend on the
Exclusions*, SI Appendix, Table S1, and the main-text sentence on the exclusion criteria (in *The Similarity Structure
of an Image Set Determines What Will Be Memorable*). It recomputes the results of the three image-set experiments of
module 06 three ways: on all 602 sessions that were run, removing no participant or trial; on the 450 participants
that the exclusions of the Methods keep (150 per set); and with a stricter corrected-recognition floor of 0.10. Every
number is recomputed (mode rerun) from the trials in `06_image_set_experiments_fig6_fig7/data/`, except the main-text
Fig. 7d statistic that Section 3 cites (27%, p = 0.036), which step 1 reads from the saved run of module 06 (mode
restatement). The module writes its 28 ledger items to `outputs/numbers.csv` and the statistics of tests T35 (concept
level), T36 (feature level) and T37 (variance explained and the reduction) to `outputs/tests.csv`, which
`statistics/all_tests.csv` collects.

## How to run

From the repository root:

    python A3d_exclusions_tableS1/code/run_module.py

`run_module.py` runs the four steps below in order and stops at the first step that fails. Each step reads the
outputs of the steps before it. The module needs Python 3 with numpy, scipy and pandas (tested versions in
`environment.md`) and no MATLAB, and it takes about 2.5 minutes on the tested laptop (Apple M2 Pro, 16 GB).

| Step | Script | What it does | Time |
|---|---|---|---|
| 1 | `code/exclusion_regimes.py` | the three participant regimes below, the correlations of Table S1, the Fig. 7d statistic without and with the exclusions (seed 7) and the comparison with the saved run of module 06 | about 2 min |
| 2 | `code/stats_tableS1.py` | tests T35 and T36 (10,000 bootstrap resamples, seed 20261013) and T37 (from step 1) | about 30 s |
| 3 | `code/make_tableS1.py` | Table S1 (`outputs/tableS1.csv`) | seconds |
| 4 | `code/write_numbers.py` | `outputs/numbers.csv` | seconds |

Each step can also be run on its own, in this order, as `python A3d_exclusions_tableS1/code/<script>.py`.
`python A3d_exclusions_tableS1/code/run_module.py --resamples 200` runs step 2 with 200 resamples, a quick check that
the code runs; it overwrites `outputs/tests.csv`, `tableS1.csv` and `numbers.csv`, whose values come from the default
of 10,000, so run the full command afterwards. `run_module.py` keeps the printout of each step in `outputs/logs/`.

## Inputs

The module has no `data/` folder of its own. Every input is in the repository, and the scripts read these files in
place:

- `06_image_set_experiments_fig6_fig7/data/maintained_trials.csv`, `changed_trials.csv` and `reduced_trials.csv`:
  every study and test trial of the 602 de-identified sessions (205 maintained, 195 changed, 202 reduced;
  participants M001 to M205, C001 to C195 and R001 to R202), one row per trial. The scripts read them through
  `shared/python/imageset_loader.py`, which also computes each test trial's summed similarity to the 40 items
  studied in its block.
- `06_image_set_experiments_fig6_fig7/data/sampled_image_sets.mat`: how many of the 1,600 image slots of each set
  belong to each concept (`numImagesForConcept`), for the set-level summed similarity of the feature level of
  Table S1.
- `shared/things/concept_table.csv` and `shared/things/monkey_features.csv`: the 49 SPoSE features (Hebart et al.,
  2020) and the Kramer et al. (2023) concept memorability of the 1,854 THINGS concepts, and the 49 monkey IT
  components (Papale et al., 2025). The memorability and the IT components enter only Fig. 6c, 6e, 6f and 6h, which
  step 1 computes in each regime and compares with module 06.
- `06_image_set_experiments_fig6_fig7/data/paper_run/paperStats_maintained_vs_changed.mat` and
  `fig7_variance_paper_run.mat`: the saved run of module 06 (the Fig. 6 correlations and the Fig. 7d bootstrap),
  the output the paper reports.
- `shared/ledger/paper_numbers.csv`: the printed values, locations and quotes of the module's ledger items.

## What the code does

**Regimes.** Step 1 flags every session against the exclusion rules of the Methods and analyzes each image set in
three ways, of which Table S1 shows the first two.

| Regime | Who is kept | Maintained, changed, reduced |
|---|---|---|
| Without exclusions (`all_run`) | every session run and every test trial, with no response-time filter | 205, 195 and 202 participants (602); 131,200, 124,800 and 129,280 test trials (640 per participant) |
| With exclusions (`reported`) | a participant is excluded who used one response key on more than 80% of test trials, had more than 20% of test trials faster than 200 ms or more than 20% slower than 7,500 ms, or had a corrected recognition below 0.05; trials faster than 200 ms or slower than 10,000 ms are then dropped, and the first 150 passing participants of each set in id order are kept | 150 per set (55, 45 and 52 excluded, 152 in all); 94,003, 94,095 and 94,002 test trials |
| Stricter floor (`stricter_010`) | the same rules with a corrected-recognition floor of 0.10 instead of 0.05 | 147, 143 and 148 |

Exactly 150 participants of each set pass the rules, so the cap of 150 removes nobody. Step 1 checks that its rules
exclude the same participants, and keep the same trials, as `shared/python/imageset_loader.py`.

**Memorability.** A concept's corrected recognition (CR) is its hit rate minus its false alarm rate, with responses 1
to 3 of the 6-point scale counted as "old". The concept and feature levels of Table S1 pool every kept trial of a
concept, whereas its variance explained rows, like Fig. 7d, average each participant's own rates for the concept.

**Concept level.** Table S1 gives Spearman rho, across the concepts of one set, between CR and the concept's mean
summed similarity to the study list. For each test trial the code averages normpdf(similarity, 1, 2) between the
tested concept and the 40 concepts studied in the same block (the tested concept removed once when it was studied
there), and a concept's value is the mean over its test trials. Similarity is the Pearson correlation of two
concepts' 49 SPoSE values, and rho is computed over the concepts with both values (970, 527 and 1,057). The interval
comes from 10,000 resamples of the concepts.

**Feature level.** Table S1 repeats Fig. 6d and 6g without and with the exclusions. For each of the 49 SPoSE features
the code computes two Spearman correlations across the set's concepts, one with the set-level summed similarity (the
kernel similarity to every concept, weighted by that concept's share of the set's 1,600 image slots, with one slot of
the concept itself removed and the weights renormalized) and one with CR. Table S1 gives the Pearson r across the 49
features between these two profiles, with an interval from 10,000 resamples of the participants.

**Variance explained and the reduction (the Fig. 7d statistic).** For the maintained and the reduced set, the code
sums over the 49 features the squared Spearman correlation between the feature and CR, and the statistic is the
percent reduction of this sum from maintained to reduced. In each of 1,000 resamples the participants of the two sets
are drawn separately with replacement. Table S1 gives each set's mean over the resamples with the 2.5th and 97.5th
percentiles, and the reduction, the mean of the per-resample reductions, with its one-sided p, the share of
resamples in which the reduced-set sum is at least the maintained-set sum.

**Comparison with module 06.** With the exclusions, the six correlations of Fig. 6 (6c to 6h) and the three kept
counts must equal the saved run of module 06 to 1e-9, or step 1 stops. `outputs/reproduction_check.csv` lists them
together with the Fig. 7d statistic, a bootstrap whose resamples this module draws with numpy (seed 7: 28.1%,
one-sided p = 0.031, as in Table S1) and module 06 with MATLAB `rng(42)` (27.3%, p = 0.036, as in Fig. 7d).

**Statistics** (`stats_tableS1.py`; the convention is in `statistics/README.md`).

- T35 (concept level): for each set without and with exclusions, rho, the numbers of concepts and participants, the
  parametric two-sided p and the 95% interval over 10,000 concept resamples, the same resamples for both columns.
- T36 (feature level): for the maintained and changed sets without and with exclusions, r, n = 49 features and the
  95% interval over 10,000 participant resamples. Concept CR and the 49 feature correlations are recomputed in every
  resample, and the features are not resampled. The with-exclusion rows are the statistics of Fig. 6d and 6g
  (module 06, tests T13 and T16), here with this script's own resamples.
- T37 (variance explained and the reduction): without and with exclusions, from the 1,000 resamples of step 1, the
  percent reduction with its one-sided p (no interval: Table S1 reports this test by its p alone), and each set's sum
  (the mean over the resamples, with the full-sample sum in the effect size column). Module 06 gives the
  with-exclusion statistic of Fig. 7d as its test T19.

No correction for multiple comparisons is applied.

**Seeds.** Every random draw uses numpy `default_rng` in the fixed order given here, and the numbers repeat when
each step runs whole and the steps run one after another. Step 1 draws 1,000 participant resamples per set with seed
7 for the Fig. 7d statistic (a new generator for each regime, the maintained set's resamples first). Step 2 uses one
generator, seed 20261013, in this order: the T35 concept resamples (maintained, changed, reduced; each resample
shared by the two columns), one draw of the participants per resample for each set without and then with
exclusions, and the T36 participant resamples (maintained, then changed; without exclusions, with exclusions, then
one draw per resample of the participants of the stricter floor). The draws between the two tests and those of the
stricter floor give no result and hold their places in the order, so that the seed reproduces the reported
intervals. Participants, concepts and features enter the resampling in ascending order (participant id, THINGS
concept number, SPoSE dimension). Intervals are the 2.5th and 97.5th percentiles of the resamples (numpy's linear
interpolation; `stats_tableS1.py` ignores resamples that give NaN). Steps 1 and 2 limit numpy to two threads.

## Outputs

- `outputs/numbers.csv`: the 28 ledger items of the module, each with the value printed in the paper, the computed
  value and whether they agree at the printed precision (columns in the top-level README).
- `outputs/tests.csv`: tests T35 to T37 as described above, each row with its statistic, n, df, p, effect size, 95%
  interval and how the interval was computed (resampling unit, resamples, seed and the script that computed it).
- `outputs/tableS1.csv`: Table S1, one row per result. `without_exclusions` and `with_exclusions` hold the cells as
  printed (the estimate and its 95% interval to two decimals; for the reduction, the percent and the one-sided p),
  and the other columns hold the same values to six significant digits (for the reduction, the percent in the
  estimate columns).
- `outputs/exclusion_regimes.txt`: every number of step 1 in readable form, from the participant counts of each rule
  to the comparison with module 06. `write_numbers.py` reads the stricter floor, the number of resamples and the
  interval level of step 1 from it.
- `outputs/participant_flags.csv`: one row per session (602) with set, participant, test trials, the shares of the
  most used key, of fast trials and of slow trials, hit and false alarm rates, CR, each rule, `excluded_005` (the
  exclusions of the Methods) and `excluded_010` (the stricter floor).
- `outputs/concept_level_by_regime.csv`: the concept-level rho of Table S1 for every set and regime, with the
  participants, trials, concepts and p.
- `outputs/feature_level_by_regime.csv`: for the maintained and changed sets and every regime, the feature-level r of
  Table S1 (Fig. 6d, 6g), the r with the THINGS profile (Fig. 6c, 6f) and the monkey-space r (Fig. 6e, 6h).
- `outputs/fig7d_reduction_by_regime.csv`: the Fig. 7d statistic without and with exclusions (each set's sum with its
  interval, the percent reduction with its one-sided p and the participants per set).
- `outputs/reproduction_check.csv`: the with-exclusion values of this module beside the saved run in
  `06_image_set_experiments_fig6_fig7/data/paper_run/`, the output the paper reports, and the printed values.
- `outputs/logs/`: the printout of each step, written by `run_module.py`, and `stats_tableS1_log.txt`, written by
  step 2.

## Code

| File | What it does |
|---|---|
| `code/run_module.py` | runs steps 1 to 4 in order and keeps a log of each |
| `code/exclusion_regimes.py` | step 1: participant flags, the three regimes, the correlations of Table S1, the Fig. 7d statistic and the comparison with module 06 |
| `code/stats_tableS1.py` | step 2: tests T35 to T37 (`outputs/tests.csv`) |
| `code/make_tableS1.py` | step 3: Table S1 (`outputs/tableS1.csv`) |
| `code/write_numbers.py` | step 4: `outputs/numbers.csv` |
