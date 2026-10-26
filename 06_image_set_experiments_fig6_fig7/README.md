# 06: Image-set experiments (Figs. 6 and 7)

This module reproduces the three image-set experiments of *Representational Analyses: Memorability Follows the
Structure of the Image Set*, in its parts *The Similarity Structure of an Image Set Determines What Will Be
Memorable* (Fig. 6) and *Reducing Variability In Distinctiveness Reduces Variability in Memorability* (Fig. 7),
together with the Methods sections *Image Set Structure Manipulations* and *Experimental Procedures and Analysis*
for these experiments, one sentence of the Abstract, two of the Introduction and four statements of the General
Discussion. Every number is recomputed from the trials of the 602 sessions and the three image sets in `data/` and
from the THINGS tables in `shared/things/` (mode rerun, or restatement for the six rows of the Abstract, the
Introduction and the General Discussion that repeat a result of Fig. 6 or Fig. 7), except the study timing of the
Methods, which the trial files do not store. The module's tests T12 to T19, T19b, T52 and T54 go into
`statistics/all_tests.csv`. Module 06b computes the other results of *Reducing Variability In Distinctiveness
Reduces Variability in Memorability* (the reliability-corrected variance and the feature analysis) from these
trials, and modules A1, A3a, A3d, A4a and A6b also read them.

## How to run

From the repository root:

    matlab -batch "restoredefaultpath; cd('06_image_set_experiments_fig6_fig7/code'); run_module"
    python 06_image_set_experiments_fig6_fig7/code/stats_fig6_fig7.py
    python 06_image_set_experiments_fig6_fig7/code/shared_concepts_test.py

The MATLAB step needs MATLAB R2023b with the Statistics and Machine Learning Toolbox (no parallel pool), and the
Python steps need Python 3 with numpy, scipy and pandas (tested versions in `environment.md`). Run the MATLAB step
first, because `stats_fig6_fig7.py` checks its numbers against the MATLAB outputs, stops if they differ, and adds
the rows of Fig. 6c to 6h to the `outputs/numbers.csv` that the MATLAB step writes. The MATLAB outputs are included
in the repository, so the two Python steps also run without MATLAB (`python run_all.py --module 06 --skip-matlab`).

| Command | What it does | Time (Apple M2 Pro, 16 GB) |
|---|---|---|
| `run_module` | steps 1 to 9 of the Code table below, every figure and `outputs/numbers.csv` except the rows of Fig. 6c to 6h | about 2 min |
| `stats_fig6_fig7.py` | tests T12 to T19, T52 and T54 (`outputs/tests.csv`) and the rows of Fig. 6c to 6h in `outputs/numbers.csv`, with 10,000 bootstrap resamples | under 1 min |
| `stats_fig6_fig7.py --resamples 200` | a quick check that the code runs; it overwrites `outputs/tests.csv` and the Fig. 6 rows of `outputs/numbers.csv`, so run the full command afterwards | under 30 s |
| `shared_concepts_test.py` | test T19b, the Fig. 7d reduction over the 559 concepts in both sets (10,000 resamples) | about 10 s |
| `shared_concepts_test.py --resamples 200` | a quick check that the code runs; it overwrites the T19b outputs, so run the full command afterwards | under 10 s |

## Inputs

- `data/maintained_trials.csv`, `changed_trials.csv`, `reduced_trials.csv`: every study and test trial of the 602
  sessions (205 maintained, 195 changed, 202 reduced), one row per trial, 640 study and 640 test rows per session.
  The columns are participant (M001 to M205, C001 to C195, R001 to R202), block (1 to 16), phase (study or test),
  trial_in_block, image_filename, concept, things_image_index (the public THINGS image number), image_number (the
  experiment's own index, not analyzed), is_new, response_1to6 (1 = sure old to 6 = sure new) and rt_ms, the last
  three on test rows only. Within each set the participant codes follow the order in which the analysis loads the
  sessions, applies the cap of 150 and draws bootstrap resamples. The files hold no recruitment ids, timestamps,
  browser strings, screen sizes, or practice and instruction trials.
- `data/sampled_image_sets.mat`: the three image sets, `representativeDataset` (maintained),
  `flipMemorabilityDataset` (changed) and `equatedDataset` (reduced), each with a datasetName label, conceptIndex,
  conceptName, imageIndex and imageFilename (one entry per slot, 1,600 slots) and numImagesForConcept (the slots of
  each of the 1,854 concepts). The maintained and reduced sets hold 1,600 distinct images; in the changed set, 20
  concepts have more slots than THINGS has images of them, so 110 images fill two slots each.
- `data/paper_run/paperStats_maintained_vs_changed.mat` and `data/paper_run/fig7_variance_paper_run.mat`: the saved runs
  of the six Fig. 6 correlations and of the Fig. 7 analysis (MATLAB, `rng(42)`: the within-set summed similarities,
  the two 1,000 x 49 matrices of bootstrap correlations and the Fig. 7 numbers), the output the paper reports.
  Module A3d (SI Appendix, Table S1) reads both files and module A6b (SI Appendix, Fig. S16) the first.
- `shared/things/concept_info_human.mat` and `concept_info_monkey.mat`, read through
  `shared/matlab/load_concept_info.m`: the 49 SPoSE features of the 1,854 concepts (Hebart et al., 2020), the Kramer
  et al. (2023) concept memorability and the 49 monkey IT components (Papale et al., 2025). The Python code reads
  the same values from `shared/things/concept_table.csv` (with the category labels) and `monkey_features.csv`.
  `category_shares.m` reads the category labels from `concept_table.csv` and the 26,107 THINGS images with their
  concepts from `things_images.csv`.
- The helpers in `shared/matlab/` and `shared/python/` (trial loaders, exclusions, bootstrap, the PiYG color map,
  the PCA camera angles and the writers of `numbers.csv` and `tests.csv`), and `shared/ledger/paper_numbers.csv`,
  the paper's printed values. Every input is in the repository.

## What the code does

**Image sets** (`reproduce_set_selection.m`). Each set has 1,600 slots drawn with replacement from the 1,854 THINGS
concepts, so a concept drawn more than once fills several slots. The maintained set (970 concepts, up to 7 slots
each) scores each concept by the sum over the 49 features of feature value x sign(r) x r^2, where r is the Spearman
correlation of the feature with full-THINGS summed similarity, and draws its concepts with these scores, shifted to
start at 0 and normalized, as probabilities. The changed set (527 concepts, up to 23 slots) scores each concept as
minus its values on the features that correlate positively with full-THINGS memorability plus its values on the
negatively correlated features, each weighted by the squared correlation; from the 500 highest-scoring concepts it
chooses 160 greedily (the top concept first, then each time the candidate with the lowest mean kernel similarity to
those already chosen) and draws 1,440 more with probability proportional to the squared positive part of minus the
score. The reduced set (1,057 concepts, up to 7 slots) rescales minus the changed-set score to 0 to 1 and raises it
to an exponent from 0.01 to 0.50 in steps of 0.01, each exponent repeated 10 times (500 candidate sets of 1,600
draws), and keeps the candidate whose 49 Spearman correlations between feature and summed similarity (over its 1,600
slots) have the lowest mean plus maximum (candidate 420, exponent 0.20). The function draws the three sets again
with `rng(6)`, in the order 1,440 changed-set draws, 1,600 maintained-set draws, 500 candidates, and the drawn
concepts equal those of `data/sampled_image_sets.mat` (`outputs/set_selection_check.csv`). The images of the slots
come from that file. `shared/things/provenance/SampleDatasetsThatFlip.m` shows the whole selection, including how
each drawn concept gets its THINGS images, and needs files that are not included.

**Task and exclusions** (`check_task_design.m`, `participant_flow.m` and the shared
`apply_participant_exclusions.m`). Every session has 16 blocks, each with 40 study images and then 40 test trials
(20 old, 20 new) rated from 1, sure old, to 6, sure new (`outputs/task_design_check.csv` shows that these counts are
the same in all 602 sessions), and responses 1 to 3 count as "old". The study timing that the Methods give (each
image for 1,000 ms, with an 800 ms interstimulus interval) is a setting of the experiment software and is not stored
in the trial files. A participant is excluded when one response key was used on more than 80 percent of test trials,
when more than 20 percent of test trials were faster than 200 ms, when more than 20 percent were slower than
7,500 ms, or when the hit rate minus the false alarm rate was below 0.05. The analyses then drop trials faster than
200 ms or slower than 10,000 ms (5,900 of the 288,000 test trials of the kept participants) and keep the first 150
participants of each set in the order of their codes, which removes nobody. Because 75 participants meet more than
one rule, the rule columns add up to more than the excluded column.

| Set | Sessions | Same key | Too fast | Too slow | CR below 0.05 | Excluded | Kept |
|---|---|---|---|---|---|---|---|
| maintained | 205 | 4 | 44 | 1 | 43 | 55 | 150 |
| changed | 195 | 4 | 32 | 0 | 29 | 45 | 150 |
| reduced | 202 | 8 | 40 | 0 | 26 | 52 | 150 |
| all three | 602 | 16 | 116 | 1 | 98 | 152 | 450 |

**Fig. 6** (`maintained_vs_changed.m`). In the maintained and the changed set (150 participants each), the corrected
recognition (CR) of a concept is its hit rate minus its false alarm rate, each pooled over all kept test trials of
the concept. A set's memorability profile is, for each of the 49 SPoSE features, the Spearman correlation across the
set's concepts between the feature and CR; the full-THINGS profile correlates the features with the Kramer et al.
concept memorability instead (1,853 concepts; goalpost has no score). The summed similarity of a concept within a
set is its kernel similarity normpdf(s, 1, 2) to every concept (s = Pearson correlation of the two concepts' 49
feature values), weighted by each concept's share of the set's slots after one slot of the concept itself is
removed, and its profile is the Spearman correlation across the set's concepts between each feature and summed
similarity. Fig. 6e and 6h repeat this in the monkey space, where the first 49 principal components of the IT
responses averaged within concept (84.6 percent of the variance, `shared/things/monkey_pca_explained.csv`) take the
place of the features for both profiles. Each of Fig. 6c to 6h is a Pearson r across the 49 features between two
profiles, with the 95% interval and the two-sided p of the participant bootstrap of tests T12 to T17 (see
Statistics):

| Panel | x (one value per feature) | y (one value per feature) | r | 95% CI | p |
|---|---|---|---|---|---|
| 6c | full-THINGS memorability profile | maintained memorability profile | 0.840 | [0.7946, 0.864] | < 0.001 |
| 6d | maintained summed-similarity profile, human | maintained memorability profile | -0.893 | [-0.909, -0.848] | < 0.001 |
| 6e | maintained summed-similarity profile, monkey | maintained memorability profile, monkey components | -0.6346 | [-0.684, -0.533] | < 0.001 |
| 6f | full-THINGS memorability profile | changed memorability profile | -0.327 | [-0.428, -0.209] | < 0.001 |
| 6g | changed summed-similarity profile, human | changed memorability profile | -0.7451 | [-0.806, -0.640] | < 0.001 |
| 6h | changed summed-similarity profile, monkey | changed memorability profile, monkey components | -0.446 | [-0.510, -0.363] | < 0.001 |

Fig. 6a, 6b and 7a show all 1,854 concepts in the first three principal components of the 49 SPoSE features, with
the projection and camera angles of Fig. 5b (`shared/matlab/pca3d_feature_space_view.csv`). Marker size is 20 + 200
x (the concept's share of the set's slots / the largest share), so concepts outside the set have the smallest
markers, and color is the concept's summed similarity to the set (`summed_similarity_to_set.m`: the kernel
similarity to every other concept, weighted by that concept's share of the set's slots), with color limits 0.174 to
0.185. In Fig. 6c to 6h each point is a feature (an IT component in 6e and 6h), colored by the rank of its
full-THINGS memorability correlation on the PiYG scale (pink for poor memory, green for good memory; each IT
component by its own correlation, as in Fig. 5d), with the least-squares line and r in the title.

**Fig. 7** (`maintained_vs_reduced_variance.m`, `make_figure7a_pca.m`). Over the concepts of each set, the
within-set summed similarity has mean 0.18231 and SD 0.0032246 in the maintained set (970 concepts) and mean 0.17976
and SD 0.0015579 in the reduced set (1,057 concepts; MATLAB `std`, n - 1), the values printed with Fig. 7b, 7c. The
histograms of Fig. 7b, 7c plot the summed similarity of all 1,854 concepts to the set, as in Fig. 6a, 6b (30 bins,
proportion of concepts, x from 0.17 to 0.19). For Fig. 7d, the CR of a concept is the mean over participants of each
participant's hit rate for the concept minus the mean of each participant's false alarm rate (participants without
an old, or a new, trial of the concept are skipped), and the summed variance explained of a set is the sum over the
49 features of the squared Spearman correlation between feature and CR across the set's concepts. `rng(42)` is set
right before the bootstrap loop, and each of the 1,000 resamples draws the 150 participants of the maintained set
with replacement (`randi`), then those of the reduced set, with participants indexed in the order of their sorted
codes, before CR, the 49 correlations and both sums are recomputed. These resamples equal those of the saved run in
`data/paper_run/`. The bars are the means over resamples (0.549 maintained, 0.393 reduced) and the error bars the
2.5 and 97.5 percentiles (MATLAB `prctile`). The percent reduction is the mean over resamples of (maintained sum -
reduced sum) / maintained sum x 100, 27.3 percent, and the one-sided p is the share of resamples in which the
reduced-set sum is at least the maintained-set sum, 0.036.

**Categories** (`category_shares.m`). The higher-level category of a concept is its label in
`shared/things/concept_table.csv` (23 named categories; the 530 concepts without one carry the label NA). Food fills
13.2 percent and animal 12.0 percent of the reduced set's 1,600 slots, the two largest named categories, and
concepts without a category fill 27.0 percent. In the whole database the tool category has 79 concepts with 1,060
images, and there is no face category.

**Statistics** (`stats_fig6_fig7.py`, `shared_concepts_test.py`). The Python code reads the same trials through
`shared/python/imageset_loader.py`, which applies the same exclusions, and recomputes CR and the profiles. It stops
if a Fig. 6 correlation differs from the saved run by more than 1e-9 or, when the MATLAB outputs exist, if the
correlations, the concept CR, the participant flow, the two summed-similarity SDs or the full-sample sums of Fig. 7d
differ from them. No correction for multiple comparisons is applied.

- T12 to T17 (Fig. 6c to 6h): r (n = 49 features) with a 95 percent percentile bootstrap interval and a two-sided
  bootstrap p over participants. Each of 10,000 resamples draws the 150 participants of the set with replacement,
  recomputes every concept's CR from their trials (a participant drawn twice counts twice) and then the 49
  correlations and r, with the other profile fixed; the 49 features are not resampled. The p is twice the share of
  resamples on the far side of zero (at most 1). No resample crosses zero in any panel, so each p is 0 (printed as
  p < 0.001), and the test column gives the count.
- T18 (Fig. 7b, 7c): the SD of the within-set summed similarity in each set (descriptive).
- T19 (Fig. 7d): the percent reduction with its one-sided bootstrap p (no interval: the paper reports this test by
  its p alone) and each set's summed rho squared with the percentiles of MATLAB `prctile`, computed from the 1,000
  resamples in `data/paper_run/fig7_variance_paper_run.mat`; the script checks that `outputs/varianceData.mat` holds
  the same resamples. The two bar rows also give the full-sample summed rho squared (0.782 maintained, 0.575
  reduced), which lies above the bar heights because resampling participants adds noise to every concept's CR.
- T19b (`shared_concepts_test.py`): the T19 statistic and estimator over the 559 concepts with a CR in both the
  maintained and the reduced set (411 concepts are only in the maintained set and 498 only in the reduced set), with
  10,000 participant resamples and the concepts fixed. The reduction is 34.1 percent (95 percent interval [4.2,
  57.0], two-sided p = 0.032). The script first checks that its full-sample sums over all concepts equal those of
  T19.
- T52: the largest category share of the reduced set (13.2 percent, food). T54: the participant flow above, per set
  and in total.

**Seeds and order of random draws.** The MATLAB steps draw random numbers only after `rng(6)` in
`reproduce_set_selection.m` and `rng(42)` in `maintained_vs_reduced_variance.m`, in the orders given above.
`stats_fig6_fig7.py` uses one generator, `numpy.random.default_rng(20261013)`, drawn in this order: participant
resamples of the maintained set, then of the changed set. Other `--resamples` values change every later draw.
`shared_concepts_test.py` uses its own `default_rng(20261013)`, drawing for each resample the maintained participants
and then the reduced ones.

**Settings.** Spearman correlations use average ranks for ties and only the concepts with a value (MATLAB `corr`
with `'rows', 'complete'`). Fig. 6 pools a concept's trials over participants, whereas Fig. 7d averages each
participant's per-concept rates. The composite score of a reduced-set candidate adds the mean of its 49 signed
correlations to the largest of them. The Python intervals are numpy percentiles (linear interpolation), except for
T19, whose percentiles reproduce MATLAB `prctile`.

## Outputs

- `outputs/numbers.csv`: each number of the paper this module computes (66 ledger rows), with the printed value, the
  computed value and whether they agree at the printed precision (columns in the top-level README). `run_module.m`
  writes 53 rows, and `stats_fig6_fig7.py` adds the 13 rows of Fig. 6c to 6h and of the Methods statement that no
  resample crosses zero; `outputs/numbers_shared_concepts.csv` holds row L3-065 (T19b).
- `outputs/tests.csv`: tests T12 to T19, T52 and T54 (16 rows with statistic, n, df, p, effect size, 95% interval
  and how it was computed; columns in `statistics/README.md`), and `outputs/tests_shared_concepts.csv`: T19b (one
  row).
- `outputs/figures/`: Fig. 6a to 6h and 7a to 7d (PNG, 300 dpi); `outputs/figure_data/`: the numbers each panel
  plots, one CSV per panel.
- `outputs/fig6_correlations.csv` and `paperStats_maintained_vs_changed.mat`: the six Fig. 6 correlations next to the
  saved run in `data/paper_run/`; `outputs/fig7_variance_summary.csv` and `varianceData.mat`: every Fig. 7 number
  next to the saved run, with the full-sample summed rho squared, the mean and SD of the plotted histograms and
  the largest difference of the bootstrap correlations from the saved run.
- `outputs/concept_scores_maintained_changed.csv`: per concept and set, slots, hit rate, false alarm rate, CR,
  summed similarity (human and monkey) and the Kramer et al. concept memorability;
  `outputs/summed_similarity_by_concept.csv`: per concept, slots and within-set summed similarity in each of the
  three sets (NaN when the concept is not in the set), read by modules 06b and A4a.
- `outputs/participant_flow.csv` (the table above, collected into `statistics/participant_flow.csv`) and
  `outputs/participant_exclusion_flags.csv` (one row per session with the shares behind each rule).
- `outputs/maintained_set_images.csv`, `changed_set_images.csv`, `reduced_set_images.csv`: the image sets, one row
  per slot (slot, concept_number, concept, things_image_index, image_filename).
- `outputs/set_selection_check.csv`, `task_design_check.csv`, `category_shares.csv`, `things_category_sizes.csv` and
  `shared_concepts_numbers.csv`: the checks and numbers described above.
- `outputs/logs/`: the log of each script, written when it runs.

## Code

| File | What it does |
|---|---|
| `code/run_module.m` | runs steps 1 to 9 below, in this order, and writes the log |
| `code/export_sampled_sets.m` | step 1: the three image sets as CSV tables |
| `code/reproduce_set_selection.m` | step 2: redraws the concepts of the three image sets with `rng(6)` and compares them with `data/sampled_image_sets.mat` |
| `code/check_task_design.m` | step 3: blocks, study and test items and response values of every session |
| `code/participant_flow.m` | step 4: the exclusions of the 602 sessions and the participant flow |
| `code/maintained_vs_changed.m` | step 5: Fig. 6 (concept CR, the feature profiles, the six correlations and the eight panels) |
| `code/maintained_vs_reduced_variance.m` | step 6: Fig. 7b to 7d (within-set summed similarity, the participant bootstrap and three panels) |
| `code/make_figure7a_pca.m` | step 7: Fig. 7a |
| `code/category_shares.m` | step 8: the higher-level categories of the reduced set and of THINGS |
| `code/write_module_numbers.m` | step 9: `outputs/numbers.csv`, except the rows of Fig. 6c to 6h |
| `code/summed_similarity_to_set.m` | the summed similarity of every concept to a set, as plotted in Figs. 6a, 6b and 7a to 7c |
| `code/stats_fig6_fig7.py` | tests T12 to T19, T52 and T54, the rows of Fig. 6c to 6h in `outputs/numbers.csv`, and the checks against the saved run and the MATLAB outputs; modules 06b and A4a import its `participant_concept_counts` |
| `code/shared_concepts_test.py` | test T19b |
