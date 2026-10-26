# A5: Context Effects in the Original THINGS Data (SI Appendix, Figs. S13 and S14)

This module reproduces SI Appendix, Section 5 (Context Effects in the Original THINGS Data) and SI Appendix, Figs.
S13 and S14 from the trial-level records of the THINGS memorability study (Kramer et al., 2023): the same image is
remembered less well after more similar preceding images (The Same Image After Different Streams, Fig. S13),
Kramer et al.'s quartile test misses a context effect present in the same trials (Kramer et al.'s Quartile Test,
Fig. S14), and the effect appears at a similar size when similarity is measured in deep-network spaces (A
Similarity Space Without Human Judgments). It writes the numbers with ledger ids L3-018, L8-002 to L8-049, L8-101 to
L8-109 and ADD-007, which include the sentences of the main text and of SI Appendix, Section 4 that refer to these
results, and the statistics of tests T38 to T42 (collected in `statistics/all_tests.csv`). The analyses need the
Kramer et al. trial file, which this repository does not include (EXTERNAL_INPUTS.md, M1); without it the module
prints the provided-output message and reads the saved results of these analyses in `data/provided/` (aggregate
numbers only). Two results are computed in every run, the 62% and 4% from the public SPoSE embedding and category
table (`neighbor_share.py`) and the SPoSE five-bin drop of the deep-network comparison from the 20-bin rates
(`spose_quintiles.py`).

## How to run

From the repository root:

    python A5_things_context_effects_figS13_S14/code/run_module.py

Without the trial file this takes about 10 seconds on an Apple M2 Pro (16 GB). With the trial file in
`external/kramer_trials/` (or its trial table in `external/kramer_trials/derived/`), the same command runs the
analyses first. The extraction is long (every trial's image name is an HDF5 reference); from the trial table the
analyses, statistics and figures take about 23 minutes and at most 5 GB of memory, about 15 minutes of it in
`fit_on_sequence.py`, which runs only when all 14 deep-network spaces are present (M3). With the trial file, run
this module before modules A2c, A3b and A4a, which read its trial table.

| Option | What it does |
|---|---|
| `--provided-only` | uses `data/provided/` even when the trial file is present (`run_all.py --provided-only` passes it) |
| `--skip-deep-network` | with the trial file, leaves out `fit_on_sequence.py`; the deep-network drop then comes from `data/provided/` (`run_all.py --skip-long` passes it) |
| `--reuse-regenerated` | skips the analyses and uses the reports already in `outputs/regenerated/` |

Each step runs as its own Python process and writes its log to `outputs/logs/<step>.log`. The code needs Python
3.11 or 3.12 with numpy, scipy, pandas, matplotlib and scikit-learn (versions in `requirements.txt`), and h5py to
read the trial file. If PyTorch is installed, it computes the distance matrix product of `fit_on_sequence.py` (on a
CUDA GPU when there is one); otherwise numpy computes it.

## Inputs

- `data/provided/`: the saved reports of the analyses. `sdt_ctx_top10_target.npz` holds the 20 within-image bins
  of the whole stream (hit and false alarm rates, d′ and c with their 95% intervals, and d_a), and
  `sdt_results.txt` is the report of `sdt_by_context.py` (the paper uses its `ctx_top10` / "target repeats (long
  lag)" block). `sdt_by_context_adjusted.npz` and `.txt` hold the bins with serial position and lag held constant,
  the bin 1 minus bin 20 contrast and its z-scale parts with intervals, and the across-image SD of d′.
  `kramer_quartile_summary.csv` (one row per category) and `kramer_quartile.txt` hold the quartile test,
  `crowding.txt` and `lone_exemplar.txt` the 4.6 and the 7.8%, and `fit_on_sequence.txt` and `.json` the held-out
  deep-network curve. `trial_intervals.csv` holds the 95% intervals that only the trial file gives (the
  whole-stream decline and its hit and false alarm parts, the 4.6 ratio, the animal drops and the quartile
  difference, and the SPoSE five-bin drop), one row per statistic with its test id, value, interval, number of
  resamples and seed; with the trial file, `stats_section5.py` computes these intervals and writes this file.
  Module A2c also reads `sdt_ctx_top10_target.npz`, `sdt_results.txt` and `sdt_by_context_adjusted.txt`.
- `shared/things/` (the 49-dimensional SPoSE embedding of Hebart et al., 2020, the higher-level category of each
  concept in `concept_categories.csv`, and the image-to-concept and image-to-embedding tables),
  `shared/embeddings_things/` (the four late deep-network layers, which SI Appendix, Fig. S8 also uses),
  `shared/python/` and `shared/ledger/paper_numbers.csv`.
- Not included (EXTERNAL_INPUTS.md): M1, the Kramer et al. trial file `imbysub-all-11-16-21.mat` (about 6.3 GB,
  MATLAB v7.3, shared by the original authors on request; their public data page, osf.io/5a7z6, has the per-image
  scores and the concept categories but not the trials), in `external/kramer_trials/`, and M3, the ten other
  deep-network spaces, in `shared/embeddings_things/`. The per-trial tables made from M1 (`trials.npz`,
  `trials_with_context.npz`, `context_k.npz`, `category_crowding.npz`, `kramer_quartile.npz`,
  `fit_on_sequence_heldout_counts.npz`) go to `external/kramer_trials/derived/` and are not distributed.

## What the code does

**Trials and context.** `extract_trials.py` keeps the 21,137 of 21,572 sessions with an image order (6,223,953
trials) and does not read the sessions' name field (an MTurk worker id). Each trial has a type in the Kramer et al.
file: 1 a first showing of a target, 2 a target repeat, 3 a first showing of a filler and 4 a vigilance repeat (a
short-lag repeat). Concepts are numbered as in the Kramer et al. files (`concept_nr` of
`shared/things/things_image_mapping.csv`), and the code reads concept k from row k of the SPoSE file. The
similarity of two concepts is the Pearson correlation of their SPoSE rows, and the context of a trial
(`ctx_top10` in the reports) is the mean of the ten highest similarities between the probe and the items already
shown in the session (earlier showings of the same image left out).

**Fig. S13.** Trials count after at least 20 preceding items. The hit rate comes from target repeats (type 2) with
an earlier showing and the false alarm rate from first showings (types 1 and 3), each as the share answered "old";
each kind of trial is ranked within image by context and cut into 20 equal bins, leaving out an image with fewer
than 20 trials of that kind. Counts are pooled over participants, rates are (k + 0.5) / (n + 1), d′ = z(H) - z(F)
and c = -(z(H) + z(F)) / 2, and the error bars are the 2.5 and 97.5 percentiles of 400 participant resamples. For
the filled circles, the code bins context minus its mean in 50 serial-position quantile strata (one trend for old
and new trials) and re-weights each bin to the same 10 serial-position strata (false alarms) and 10 x 10
position-by-lag strata (hits). The across-image SD of d′ behind the 1.34 (0.379) is over the 26,107 images with at
least 30 target repeats and 30 first showings (each image's d′ from the log-linear rates of all its trials). The
"r with bin index" of `sdt_results.txt` is a Pearson correlation; the ρ = -1.00 of the text is the Spearman
correlation that `check_provided.py` computes from the saved per-bin d′.

**The 4.6, the 7.8% and the 62%.** The same-category count of a trial is the number of earlier trials of the
session that showed another image whose concept has the probe concept's code in `concept_categories.csv` (code 0
marks the 530 concepts without a named category). `crowding.py` regresses a false alarm (first showings after at
least 50 preceding items) per session on feature crowding (the top-10 measure, from `context_k.npz` of
`k_robustness.py`) and the same-category count, both z-scored and centered within image, with serial position, in
the 21,118 sessions with at least 30 such trials; the 4.6 is the ratio of the two mean coefficients, 0.00286
and 0.00062 (model M3 of `crowding.txt`). The 7.8% is the share of first showings after at least 20 preceding
items with a same-category count of zero (`lone_exemplar.txt`, ALL CONCEPTS). The 62% is the share of the ten
nearest SPoSE neighbors of the 1,324 concepts with a named category that have the same code; chance is 1 of 27
categories.

**Fig. S14.** For each of six categories, participants are ranked by the number of distinct exemplars they saw
(first showings), ties broken at random, and cut into four equal groups (5,284 or 5,285 sessions). Each image is
scored by corrected recognition (hit rate minus false alarm rate, from all its trials in the quartile) in the top
and in the bottom quartile; images with at least 5 target repeats and 5 first showings in both enter the Spearman
correlation, which is set against 200 random permutations of the quartile labels. Split-half reliability uses 50
random halves of a quartile's participants (images with at least 3 trials of each kind in both halves),
Spearman-Brown corrected. The animal images' trials after at least 20 preceding items are binned within image, into
20 bins in all participants for the 0.63 of the text and into five bins in all participants and inside each
quartile for the bars of panel b. The dashed line is the pooled d′ of the bottom minus the top quartile, and the
prediction of -0.096 is the slope of d′ on the mean crowding of the 20 bins times the quartiles' difference in mean
crowding.

**Deep-network spaces.** In each of the 14 spaces (AlexNet, VGG16-BN, ResNet18, InceptionV3, CORnet-S, DINOv3
ViT-B/16 and CLIP ViT-B/32, an early and a late layer each; 512 dimensions per layer, principal components where a
layer has more, the raw units of ResNet18's late layer; columns standardized), the correlation distance between
images is z-scored with the mean and SD of a 2,000-image sample, with an offset at its 1st percentile. A trial's
crowding at falloff tau is the log of the sum of exp(-(d - offset) / tau) over the items already shown (the same
image left out), for tau = 0.5, 1, 2, 4 and 8, and the 70 measures are centered within image. On trials after at
least 20 preceding items, half of the sessions fit a logistic regression of a correct response (a target repeat
called old, or a first showing not called old) on the 70 measures (C = 1.0, lbfgs, 1,000 iterations). Each
image's held-out trials are ranked into five bins by the fitted crowding, and d′ comes from rates clipped to
[0.0001, 0.9999]. For the SPoSE drop on five bins (0.39), each fifth's hit and false alarm rate is the mean of four
adjacent rates of the 20-bin curve.

**Seeds and order of draws.** Each script uses numpy `default_rng`, and its draws come in the order below when it
runs from start to end. `sdt_by_context.py`: seed 0, 400 participant resamples per curve in the order ctx_top10
target, ctx_top10 vigilance, ctx_sharp target, ctx_sharp vigilance, ctx_paper target, ctx_paper vigilance.
`sdt_by_context_adjusted.py`: one seed 0 stream, with 400 multinomial participant draws for step 1, then 400 for
step 2, 400 participant resamples for step 4 and 400 multinomial draws each for steps 6 and 7.
`kramer_quartile.py`: one seed 0 stream through animal, food, clothing, container, tool and vehicle, per category
the tie-break, the 200 random splits and 50 random halves of the top and then of the bottom quartile.
`fit_on_sequence.py`: seed 0 for the 2,000-image sample of each space and, separately, for the half split.
`stats_from_trials.py` (trial file only): seed 20261013, a fresh generator per statistic, 2,000 participant
resamples; it also replays the 400 resamples of `sdt_by_context.py` and checks them against the saved per-bin
intervals.

## Outputs

- `outputs/numbers.csv`: each number of the paper this module computes, with the printed value, the computed value
  and whether they agree at the printed precision (columns in the top-level README).
- `outputs/tests.csv`: tests T38 to T42 (statistic, n, df, p, effect size, 95% interval and how it was computed).
  The intervals with serial position and lag held constant come from the 400 participant resamples of the saved
  reports (seed 0). The intervals that need the trial file (the whole-stream contrast and its hit and false alarm
  parts, the 4.6 ratio, the animal drops and the quartile difference, and the SPoSE five-bin drop) come from 2,000
  participant resamples (seed 20261013), read from `data/provided/trial_intervals.csv`; the interval of the
  held-out deep-network drop needs the trial file and all 14 deep-network spaces. For the quartile test (T40) the
  interval columns hold the middle 95% of the random splits. No correction for multiple comparisons.
- `outputs/figures/`: `FigS13_things_trial_level_context.png` (SI Appendix, Fig. S13a to c) and
  `FigS14_kramer_quartile.png` (Fig. S14a and b); `outputs/figure_data/` holds the plotted values of each panel
  (`FigS13a_dprime.csv` to `FigS14b_animal_drops.csv`).
- `outputs/neighbor_share.txt` and `outputs/spose_quintiles.txt`: the two results computed in every run.
- `outputs/regenerated/` (with the trial file): the reports of `data/provided/` written by this run, with
  `k_robustness.txt` and `.npz` and the curves of the other context measures (`sdt_<measure>_<target or vigil>.npz`).

## Code

| File | What it does |
|---|---|
| `code/run_module.py` | runs the steps below and chooses the mode |
| `code/a5_common.py` | folders, inputs, the mode switch, `trial_intervals.csv` and the concept category table |
| `code/extract_trials.py` | the trial table from the Kramer et al. file |
| `code/compute_context.py` | the context of each trial (top-10 measure and two kernel means), the lag and the outcome coding |
| `code/sdt_by_context.py` | the 20 within-image bins over the whole stream (Fig. S13, open circles) |
| `code/sdt_by_context_adjusted.py` | the bins with position and lag held constant, and the across-image SD of d′ |
| `code/kramer_quartile.py` | Kramer et al.'s quartile test and the within-image effect in the same trials (Fig. S14) |
| `code/k_robustness.py` | the top-k and kernel context measures of every trial (`context_k.npz`) and the context effect for each |
| `code/crowding.py` | feature crowding against same-category count (the 4.6) |
| `code/lone_exemplar.py` | first showings with no earlier item of the probe's category (the 7.8%) |
| `code/fit_on_sequence.py` | crowding in the 14 deep-network spaces, fitted on half of the sessions |
| `code/neighbor_share.py` | the 62% and 4%, from public data |
| `code/spose_quintiles.py` | the SPoSE drop on five bins, from the 20-bin rates |
| `code/stats_section5.py`, `code/stats_from_trials.py` | `outputs/tests.csv`; the second holds the resampling that needs the trial tables |
| `code/read_reports.py`, `code/check_provided.py` | read every number of the reports by the label of its row; write `outputs/numbers.csv` |
| `code/make_figureS13.py`, `make_figureS14.py` | the figures and their plotted values |
