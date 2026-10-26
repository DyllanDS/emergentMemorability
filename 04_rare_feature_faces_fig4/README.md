# 04: Preregistered rare-feature face study (Fig. 4a, 4b and SI Appendix, Fig. S11a)

This module holds the data of the preregistered rare-feature face experiment (AsPredicted #311640). It
reproduces the section "A Rare Feature's Advantage Depends on What the Face Is Tested Against": Fig. 4a and
4b, memorability in the three conditions (SI Appendix, Fig. S11a, with the values printed in SI Appendix,
Section 3) and the paired t tests of the drop and the shortfall. It also computes the design and participant
numbers of the Methods ("Behavioral Validation Experiments" and "Experimental Procedures and Analysis"), the
sentences of the abstract, introduction and General Discussion that restate the result, and tests T07, T08,
T09 and T54 of `statistics/all_tests.csv`. Every number is computed from data in the repository. Fig. 4c
(the confidence ROCs with the preregistered unequal-variance fit) and the rest of the face study in SI
Appendix, Section 3 (the model predictions of Fig. S10 and the d_a analysis of Fig. S11b) come from module
`A3c_face_study_model_and_sdt_figS10_S11`, and the main-text scores of each face against itself from module
`04b_face_same_item`. Both read these data through `shared/python/face_loader.py`.

## How to run

From the repository root:

    python 04_rare_feature_faces_fig4/code/run_module.py

The module takes about 10 s on the tested laptop (Apple M2 Pro, 16 GB) and needs Python 3 with numpy,
scipy, pandas and matplotlib. `run_module.py` runs the six scripts of the Code table in that order, each in
its own process, and writes one log per script to `outputs/logs/`. Each script can also be run alone from
the repository root, in the same order, because later scripts read the outputs of earlier ones.
When `run_all.py` runs with `--skip-long` or `--provided-only`, it passes the option to `run_module.py`,
which accepts it and runs as usual, because the module has no long step and all of its inputs are in the
repository.

## Inputs

- `data/face_trials.csv`: one row per trial of the experiment proper for the 111 sessions run, 56 study
  rows and 36 test rows per session, in the order they were run (no practice, instruction or arithmetic
  rows). Columns: `participant` (S001 to S111, in order of completion), `completion_rank`, `load_order`
  (step 2 below), `phase` (study or recognition), `trial_in_phase`, `image` (stimulus file name), `kind`
  (filler or critical), `condition` (filler = ordinary face, rare = rare-feature face), `feature` and
  `cell` (the rare feature and its pair, c1 to c6), `is_repeat`, `pressed` and `oneback_correct` (study
  rows: the one-back task), `test_type` (old = studied face; new = never-studied ordinary face; foil = the
  unstudied member of a rare-feature pair, the new face carrying a studied face's feature), `is_old`,
  `said_old` (the old/new answer), `confidence_1to4` (1 Guess, 2 Maybe, 3 Probably, 4 Certain), `rt_ms`
  and `rt_confidence_ms` (response times of the two answers), `studied_image` (on critical test rows, the
  member of the pair that was studied) and `role_swapped` (1 when the studied member of a pair was the file
  named `_foil`).
- `data/face_participants.csv`: one row per session with `participant`, `completion_rank`, `load_order`,
  `oneback_hits` (repeats detected, of 6), `oneback_false_presses` (presses on faces that were not
  repeats), `passes_rule` and `registered`. The code recomputes every column from `face_trials.csv` and
  stops if one differs (`face_loader.recompute_participants()`).
- `data/face_design.json`: the task settings of the experiment program (the same in every session, without
  the Prolific completion code and the experimenter's name), the stimulus manifest (6 rare-feature pairs, 59
  ordinary faces, 6 practice faces) and design facts. `task_constants.experiment` and
  `stimulus_manifest.stim_version` are the labels the experiment program gave the task and the stimulus set.
- `data/fig4a_faces/`: the five synthetic faces of Fig. 4a (generated portraits, not photographs of real
  people): practice faces `filler_006.jpg` and `filler_002.jpg`, ordinary face `filler_009.jpg` and the
  mohawk pair c1 (`rare_mohawk_c1_target.jpg`, `rare_mohawk_c1_foil.jpg`). No other stimulus image is
  included.
- `data/PREREG_SUBMITTED.md`: the preregistration as submitted (AsPredicted #311640).
- The tables are de-identified. They hold only the columns above: no Prolific, study or session ids,
  completion times, session durations, browser or screen fields, focus and blur events, demographic
  answers, practice trials or arithmetic responses. The mapping from Prolific ids to S ids is not included.
- Shared code: `shared/python/face_loader.py` (reads the tables, applies the registered rule and computes
  each participant's rates), `sdt.py` (log-linear d′ and c), `bootstrap.py`, `tests_table.py`,
  `numbers_table.py` (with the paper's values in `shared/ledger/paper_numbers.csv`), `figure_style.py`,
  `release_paths.py` and `inversion_loader.py`. The last one reads `03_inversion_experiment_fig3/data/` for
  one statement about both direct-manipulation experiments (L2-041). Without that folder the module still
  runs and reports that half of the statement as not computed.

## What the code does

1. `design_check.py` counts the design of the Methods in every session and stops if a session breaks it.
   The studied member of each rare-feature pair is drawn for every pair of every participant (the member
   named `_foil` is studied in 293 of the 600 pair assignments of the registered participants). The six
   repeated faces are ordinary faces that are never tested, each shown again right after its first showing
   (56 study presentations). Which three pairs are tested with the studied face is one draw per
   participant (pairs c1, c3 and c5 for 44 registered participants, c2, c4 and c6 for 56).
2. `face_analysis.py` runs the registered descriptive analysis. The preregistered rule
   (`PREREG_SUBMITTED.md`, question 6) excludes a participant who does not complete the 36 test trials,
   detects fewer than 4 of the 6 repeats or presses on more than 5 faces that were not repeats. A one-back
   hit is a press on a repeat trial (`oneback_correct`); a false press is a press on any other study trial.
   The sample is the first 100 passing sessions in order of completion. The code stops unless this sample
   equals `face_loader.registered_participants()`.

   | Sessions | Excluded | More than 5 false presses | Fewer than 4 of 6 repeats detected | Both | Kept |
   |---|---|---|---|---|---|
   | 111 | 11 | 8 | 4 | 1 | 100 (no passing session left over) |

   For each participant the script takes the proportion called old for the four kinds of test item. Every
   participant has 3, 3, 15 and 15 trials in these cells, so the means over participants equal the pooled
   rates. It also computes the log-linear d′ and c of each kind of face against its own kind of lure (a
   rare face against the new face with its feature, an ordinary face against a face nobody studied). The
   corrected rates are H = (hits + 0.5) / (old trials + 1) and F = (false alarms + 0.5) / (new trials +
   1), with d′ = z(H) - z(F) and c = -(z(H) + z(F)) / 2. Criterion c is a preregistered secondary measure
   and the equal-variance d′ the preregistered baseline; the paper reports discriminability as d_a (module
   A3c). Rare and ordinary faces are compared on each measure with a paired t test, Cohen's d_z, the
   Wilcoxon signed-rank test (the preregistered test of predictions 1 and 2, with scipy defaults) and the
   registered participant bootstrap. The bootstrap draws 10,000 resamples, and resample i draws its
   participants with its own generator, numpy `default_rng(i)` for i = 0 to 9,999. This draw depends on
   the order of the participants, so the bootstrap takes them in `load_order`, the order of the raw
   session files (not included) sorted by file name. It gives [0.0653, 0.143] for the hit rate contrast and
   [0.117, 0.233] for the false alarm rate contrast. Every other analysis takes the participants in order
   of completion.
3. `memorability_check.py` computes memorability, the hit rate minus the false alarm rate, for each
   participant in the three conditions of SI Appendix, Fig. S11a:

   | Condition | Hit rate of | False alarm rate of | Mean | 95% interval |
   |---|---|---|---|---|
   | Rare feature, not carried | the 3 studied rare-feature faces | the 15 never-studied faces | 0.493 | 0.438 to 0.5453 |
   | Rare feature, carried by a new face | the 3 studied rare-feature faces | the 3 new faces with the feature | 0.317 | 0.247 to 0.387 |
   | Ordinary features | the 15 studied ordinary faces | the 15 never-studied faces | 0.388 | 0.349 to 0.426 |

   Paired t tests over the 100 participants compare the conditions (scipy `ttest_rel`, two-sided; Cohen's
   d_z is the mean paired difference divided by the SD of the differences). The drop, the first condition
   minus the second, is +0.176, t(99) = 5.89, p = 5.3e-08, d_z = 0.59. It equals the false alarm contrast
   (0.587 minus 0.411), because the two conditions share their hit rate. The shortfall, the second
   condition minus the third, is -0.071, t(99) = -2.17, p = 0.033, d_z = -0.22. The intervals of the
   conditions come from one bootstrap routine: one numpy generator, `default_rng(0)`, draws 10,000
   resamples of the 100 participants for the first condition, then 10,000 for the next, and so on, and the
   interval is the 2.5th and 97.5th percentile of the resampled means (numpy `percentile`, linear
   interpolation). Called once per panel with the conditions in plotted order (Fig. 4b: studied rare,
   studied ordinary, new with feature, new without; SI Appendix, Fig. S11a: the order of the table), it
   gives the error bars of both panels and the intervals printed in SI Appendix, Section 3.
4. `make_figure4.py` draws the two figures from `outputs/figure_data/` and the faces in `data/fig4a_faces/`.
5. `stats_fig4.py` writes tests T07 (the drop), T08 (the shortfall), T09 (the two rate contrasts of
   Fig. 4b, preregistered predictions 1 and 2) and T54 (the participant flow). Each contrast gets the
   paired t test with Cohen's d_z and the t-based 95% CI of the mean difference, the mean plus or minus
   t(0.975, 99) times its standard error (0.117 to 0.2353 for the drop, -0.137 to -0.006 for the
   shortfall). The two rate contrasts (SI Appendix, Section 3 and Fig. S10a) also get the registered
   bootstrap interval of step 2. Each plotted condition mean gets the interval drawn in the figure. No
   correction for multiple comparisons is applied.
6. `numbers_fig4.py` writes `outputs/numbers.csv`. It computes the t-based intervals of the drop and the
   shortfall from their t tests. Steps 1, 4 and 6 draw no random numbers.

## Outputs

- `outputs/numbers.csv`: each number of the paper this module computes (47 rows), with the printed value,
  the computed value and whether they agree at the printed precision. Statements without a printed number
  are listed as documented (columns in the top-level README).
- `outputs/tests.csv`: tests T07, T08, T09 and T54 (14 rows), each with its statistic, n, df, p, effect
  size, 95% interval and how the interval was computed.
- `outputs/figures/Fig4ab_rare_feature_faces.png`: Fig. 4a (the design, drawn with the faces in
  `data/fig4a_faces/`) and 4b (the proportion called old, studied faces blue and new faces red, with 95%
  bootstrap intervals over participants), 300 dpi, 17.8 x 12.7 cm. The space to the right of panel b is
  left for Fig. 4c, which module A3c draws.
- `outputs/figures/FigS11a_memorability_by_condition.png`: SI Appendix, Fig. S11a, 300 dpi, 8.7 x 6.7 cm
  (memorability in the three conditions with 95% bootstrap intervals over participants, and braces for
  the two comparisons labeled from their p values, *** p < 0.001 and * p < 0.05).
- `outputs/figure_data/`: the plotted numbers of each panel (`fig4a_design_faces.csv`: each face of
  Fig. 4a with its position, label and role in the stimulus manifest; `fig4b_proportion_called_old.csv`;
  `figS11a_memorability.csv`; `figS11a_comparisons.csv`).
- `outputs/results_registered_100.txt` and `outputs/memorability_hit_minus_fa.txt`: steps 2 and 3 in words.
- `outputs/rate_contrasts.csv` and `outputs/rates_by_participant.csv`: the contrasts of step 2 (hit rate,
  false alarm rate, d′ and c) and each registered participant's rates, d′ and c.
- `outputs/participant_flow.csv` (T54, collected into `statistics/participant_flow.csv`) and
  `outputs/design_check.csv` (each design fact of step 1 with its value and source).

## Code

| File | What it does |
|---|---|
| `code/run_module.py` | runs the six scripts below in order |
| `code/design_check.py` | counts the design in the data and stops if a session breaks it |
| `code/face_analysis.py` | the registered descriptive analysis: participant flow, rates, rare against ordinary contrasts |
| `code/memorability_check.py` | the values of Fig. 4b and SI Appendix, Fig. S11a with their intervals, and the two paired t tests |
| `code/make_figure4.py` | draws Fig. 4a and 4b and SI Appendix, Fig. S11a |
| `code/stats_fig4.py` | tests T07, T08, T09 and T54 (`outputs/tests.csv`) |
| `code/numbers_fig4.py` | writes `outputs/numbers.csv` |
