# A3a: The model run through the participants' own lists (SI Appendix, Fig. S8)

This module reproduces SI Appendix, Section 3, "The Model Predicts Memory From Participants' Own Lists"
(Fig. S8a to c), the part of the same-image change in d′ that false alarms carry ("Similar Items Change
Memory Mainly Through False Alarms"), and the main-text sentence in "The Similarity Structure of an Image
Set Determines What Will Be Memorable" that refers to these results. The continuous-feature REM of SI
Appendix, Fig. S6 is given the study lists and test trials that each participant received, in the
memorability inversion experiment (the 50 participants of Fig. 3e) and in the image set experiments (the
450 participants of Figs. 6 and 7). Each THINGS photograph is a point in a 49-dimensional space built from
the late layers of four computer vision networks, and only the model's encoding noise and its criterion
are set from the data, from each experiment's overall hit and false alarm rates. Fig. S8a compares the
model's predictions for the 40 critical items with what the participants remembered, and Fig. S8b and c
compare each of the 1,023 images tested in more than one image set with itself. The module also computes
the uncertainty statistics T31 and T32 (collected in `statistics/all_tests.csv`). It reads the data of
modules 03 and 06 and the THINGS embeddings in `shared/`, so it has no `data/` folder of its own, and it
computes every result from these inputs.

## How to run

From the repository root:

    python A3a_model_through_own_lists_figS8/code/run_module.py

On the tested laptop (Apple M2 Pro, 16 GB) the full run takes about 6 minutes: about 2 minutes for the
model run through the inversion experiment, 3 for the image set experiments, under 1 for the uncertainty
statistics and a few seconds for each other step. It needs Python 3 with numpy, scipy, pandas,
scikit-learn and matplotlib (versions in `requirements.txt`), no MATLAB, and about 1.1 GB of memory at
most (the 26,107 x 2,048 feature table and its principal components). It reads module 03's
`outputs/inversion_items_50.csv`, which is in the repository and which module 03 writes when it runs.

| Option | What it does |
|---|---|
| `--skip-fits` (or `--skip-long`, which `run_all.py --skip-long` passes) | uses the two model fits already in `outputs/` (`fit_inversion.*`, `fit_datasets.*`) instead of computing them; about 1 minute |
| `--resamples N` | sets the bootstrap resamples of `stats_figS8.py` (default 10,000, the number used for the paper); a small number such as 200 checks the code |
| `--keep-model-inputs` | keeps the four per-trial tables of `outputs/model_inputs/`, which the run otherwise deletes at the end |

`run_module.py` runs ten scripts of `code/` serially, each in its own Python process with at most two
processor threads (it sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and
`VECLIB_MAXIMUM_THREADS` to 2 unless they are already set), and with these settings repeated runs on one
machine give identical outputs. Each script reads what the earlier ones wrote and can also be run alone,
in this order: `extract_inversion.py`, `fit_inversion.py`, `disattenuate_inversion.py`,
`extract_datasets.py`, `fit_datasets.py`, `bin_by_crowding.py`, `paired_crowding.py`, `stats_figS8.py`,
`make_figureS8.py` and `write_numbers.py` (for example
`python A3a_model_through_own_lists_figS8/code/fit_inversion.py`).
`run_module.py` writes a log of each step to `outputs/logs/`, which is not part of the repository.

## Inputs

- `shared/embeddings_things/alexnet_late.npz`, `vgg16_late.npz`, `resnet18_late.npz` and `clip_late.npz`:
  the late-layer embeddings of the 26,107 THINGS photographs (26,107 x 512 each, float32; see
  `shared/embeddings_things/README.md`). The module stops if one of them is missing.
- `shared/things/things_images.csv` (file name and embedding row of every photograph; sorted by
  embedding_row, it gives the row order of the embedding files) and `shared/things/things_image_mapping.csv`
  (THINGS image number to file name).
- `03_inversion_experiment_fig3/data/inversion_trials.csv` and `inversion_design.csv` (the 54 sessions of
  the memorability inversion experiment), read through `shared/python/inversion_loader.py`, and
  `03_inversion_experiment_fig3/outputs/inversion_items_50.csv` (module 03's table of the 40 critical items).
- `03_inversion_experiment_fig3/outputs/tests.csv`, when it is there: module 03's test T06, whose intervals
  `stats_figS8.py` compares with its own intervals of the published score against observed memorability.
- `06_image_set_experiments_fig6_fig7/data/maintained_trials.csv`, `changed_trials.csv` and
  `reduced_trials.csv` (the 602 sessions of the image set experiments), read through
  `shared/python/imageset_loader.py`, which also reads `shared/things/concept_table.csv`.
- `shared/ledger/paper_numbers.csv`: the printed values and quotes behind `outputs/numbers.csv`.

## What the code does

**Participants and trials.** `extract_inversion.py` and `extract_datasets.py` apply the exclusion criteria
of the Methods (inversion experiment: 54 sessions, 4 excluded, 50 kept; image set experiments: 602
sessions, 152 excluded, 150 kept per set) and keep the test trials of 200 to 10,000 ms (3,981 of the 50
participants' 4,000 test trials and 282,100 of the 450 participants' 288,000). Responses 1 to 3 on the
6-point scale count as "old", and a test image counts as old when it was on that block's own study list.
`extract_inversion.py` stops unless the loader's table of the 40 critical items equals module 03's on
every column; the full-precision published scores of that table (the mean of the Kramer et al., 2023,
score over each item's test trials) set the rank order of items that share a score, as in Fig. 3e. An
image enters a set's table (an image-by-set cell) with at least 5 old and 5 new test trials there, which
gives 4,690 cells (1,600 maintained, 1,490 changed, 1,600 reduced) and 1,023 images in two or more sets
(900 in two, 123 in three).

**Features.** The late layer of each network (AlexNet avgpool, VGG16 as torchvision vgg16_bn avgpool,
ResNet18 avgpool, CLIP ViT-B/32 ln_post) is stored as 512 principal components (ResNet18's 512 values as
they are). Every column is standardized over the 26,107 photographs ((x - mean) / (SD + 1e-8)), the four
layers are put side by side (2,048 columns), and scikit-learn PCA (randomized solver, random_state 0)
reduces them to 49 components, which keep 17.7% of the variance. The components are computed in float32,
so on other hardware their last digits can differ.

**The model.** Each feature of each studied image is stored with probability 0.30. A stored feature is
copied correctly with probability 0.60 (the image's value plus Gaussian noise with SD = noise x the
feature's SD over THINGS); otherwise it takes, without noise, that feature's value in a THINGS photograph
drawn at random from all 26,107. The likelihood ratio of one test image against one trace is the product
over the stored features of 1 + 0.60 x (N(trace value; test value, noise SD) / f(trace value) - 1), each
factor at least 1e-12, where f is the feature's density over THINGS (a Gaussian kernel density with
bandwidth factor 0.2 from 6,000 photographs drawn at random, tabulated on 512 points over the range plus
15% on each side, at least 1e-9, and the end value beyond the grid). A test image's familiarity is the mean
of its likelihood ratios over the 40 traces of the block's study list, and the model says "old" when the
familiarity exceeds the criterion.

**The two settings.** The criterion is the quantile of the lure familiarities that gives the observed
false alarm rate, and the noise is the grid value whose hit rate comes closest to the observed hit rate
(rates pooled over all kept test trials, including the maximally dissimilar items of the inversion
experiment). The grids, as fractions of each feature's SD, are 0.05, 0.1, 0.2, 0.35, 0.6, 1.0 and 1.6
(inversion) and 0.1, 0.2, 0.35, 0.6 and 1.0 (image sets). Both runs choose 0.35, with a criterion of 1.75
(inversion) and 1.416 (image sets). A subject-block is one block of one participant: its study list and
its test trials. The inversion run simulates 120 participants per subject-block (400 subject-blocks). The
image set run sweeps the grid with 10 simulated participants on each of 900 randomly chosen subject-blocks
and then simulates 20 per subject-block on all 7,200 at the chosen noise.

**Crowding and the same-image contrast.** `bin_by_crowding.py` gives every cell its crowding, minus the
mean correlation distance (1 minus the Pearson correlation of two images' 49 values) to its 20 nearest
set-mates among the images tested in that set (column crowd_near). For each image in two or more sets,
`paired_crowding.py` takes the least crowded (more distinctive) minus the most crowded (less distinctive)
set, for hit rate, false alarm rate and d′ (rates kept between 1/(2N) and 1 - 1/(2N), with N the cell's
number of old or new trials), with images weighted equally, and computes the same contrast from the
model's predicted rates on the same pairs. The part of the d′ change that hits (or false alarms) carry
moves the hit rate (or false alarm rate) half its mean change either side of its mean over the paired
cells.

**Reliability.** `disattenuate_inversion.py` computes the reliability of the 40 items' observed scores as
1 - (mean over items of h(1 - h)/n_old + f(1 - f)/n_new) / (variance of the items' hit rate minus false
alarm rate, n - 1), treating only the observed scores as noisy, and divides the Fig. S8a rho by its
square root.

**Seeds and the order of the random draws.**

- `fit_inversion.py`: PCA random_state 0; numpy `default_rng(0)` draws the 6,000 photographs of the density
  estimates; then, for every noise value, a fresh `default_rng(0)` runs the subject-blocks in the order of
  participant id and block (blocks without test trials are skipped), drawing in each block the stored
  mask, the correct-copy mask, the Gaussian noise and then, feature by feature, the photographs for the
  wrongly copied values.
- `fit_datasets.py`: the same PCA and density draws; a separate `default_rng(0)` chooses the 900
  subject-blocks of the sweep (without replacement, over the subject-blocks in participant and block
  order); each noise value of the sweep and the full pass start a fresh `default_rng(0)`.
- `bin_by_crowding.py`: only the PCA.
- `paired_crowding.py`, one `default_rng(0)`, in this order: 10,000 image resamples of the observed
  changes; 2,000 draws of each set's participants (one multinomial per set, sets in the order changed,
  maintained, reduced) and 10,000 draws of the 1,023 images, whose values are not used; and 10,000 image
  resamples of the model's changes, whose intervals depend on the draws before them.
- `stats_figS8.py`, one `default_rng(20261013)`, 10,000 resamples each, in this order: the participant
  resamples of the published score against the observed scores (the 50 participants in id order, the
  same draws as module 03's test T06, whose interval the log compares; the observed rates are recomputed
  with the published scores fixed), the item resamples of T31 (each item keeps its scores and
  prediction) and the image resamples of T32.

**Checks.** `stats_figS8.py` stops unless the per-participant counts add up to the item table, the cells
reproduce every image's change in d′, and its reliability and d′ split equal those in
`outputs/disattenuation.txt` and `outputs/paired_crowding.txt`; it also reports whether its published-score
intervals equal module 03's test T06. `make_figureS8.py` checks every recomputed rho against the model run and the counts of images and participants across its inputs.

## Outputs

- `outputs/numbers.csv`: each number of the paper this module computes (29 rows: ledger ids L3-045,
  L7-001 to L7-017, L7-019, L7-020, L7-022, L7-026, L7-101 to L7-104 and ADD-014 to ADD-016), with the
  printed value, the computed value and whether they agree at the printed precision; sentences and figure
  panels without a printed number are listed as documented (columns in the top-level README).
- `outputs/tests.csv`: T31 holds the Spearman correlations over the 40 critical items (df = 38, two-sided
  p, 95% item interval) of the model with the observed memorability (hit rate minus false alarm rate;
  Fig. S8a), hit rate and false alarm rate, and of the published score with the model's and the observed
  memorability (the rho of Fig. 3e), plus the Fig. S8a rho corrected for reliability. T32 holds the
  same-image changes in d′, hit rate and false alarm rate, observed and model, with the image intervals of
  `paired_crowding.py` (seed 0; the error bars of Fig. S8b and c); the paired t test over images of the
  observed change in d′ (df = 1,022, t-based interval, Cohen's d_z); the model / observed ratio of the
  d′ change and the part of the observed d′ change that false alarms carry, each with its image interval
  from `stats_figS8.py`; and the numbers of images in two and in three sets. The model's rows of T32 have
  no p value (a simulation), and no correction for multiple comparisons is applied (statistics/README.md).
- `outputs/figures/FigS8_model_through_experiments.png` (300 dpi): Fig. S8a to c.
- `outputs/figure_data/FigS8a.csv` (the 40 points of panel a), `FigS8b.csv` (the two bars of panel b, their
  95% image intervals and the number of images) and `FigS8c.csv` (the four bars of panel c).
- `outputs/fit_inversion.txt` and `fit_inversion_items.csv`: the noise sweep and the chosen setting of the
  inversion run, and the 40 critical items with their observed and predicted rates.
- `outputs/disattenuation.txt`: the reliability of the observed scores and the corrected rho.
- `outputs/fit_datasets.txt` and `fit_datasets_cells.csv`: the noise sweep and the chosen setting of the
  image set run, and the observed and predicted rates of every image-by-set cell.
- `outputs/bin_by_crowding_cells.csv`: the crowding of every cell.
- `outputs/paired_crowding.txt`, `paired_crowding_summary.csv` and `paired_crowding_images.csv`: the
  same-image contrast of Fig. S8b and c (the report, the means with their image intervals, and one row
  per image).
- `outputs/model_inputs/`: `inversion_observed_items.csv` (the observed table of the 40 critical items),
  `datasets_observed.csv` (the observed table of the 4,690 cells), and `inversion_sample.csv` and
  `datasets_sample.csv` (sessions, exclusions, kept participants and trials). The per-trial tables the
  model reads (`inversion_study.csv`, `inversion_test.csv`, `datasets_study.csv`, `datasets_test.csv`;
  about 15 MB, copies of the module 03 and 06 data) are deleted at the end of `run_module.py`;
  `extract_inversion.py` and `extract_datasets.py` write them again in seconds.
- `outputs/pipeline_checks.csv`: the share of the variance that the 49 components keep (filled in when the
  run includes `fit_inversion.py`), the noise grids with the chosen values, and the numbers of cells and of
  images in two or more sets.

## Code

| File | What it does |
|---|---|
| `code/run_module.py` | runs the ten steps in order, keeps a log of each and deletes the per-trial tables at the end |
| `code/extract_inversion.py` | the 50 participants' study lists and kept test trials; the observed table of the 40 critical items |
| `code/fit_inversion.py` | the model through the inversion experiment: noise sweep, criterion, per-item predictions |
| `code/disattenuate_inversion.py` | the reliability of the observed item scores and the corrected rho |
| `code/extract_datasets.py` | the 450 participants' study lists and kept test trials; the observed table of every image-by-set cell |
| `code/fit_datasets.py` | the model through the three image sets |
| `code/bin_by_crowding.py` | the crowding of every image in each set it was tested in |
| `code/paired_crowding.py` | the same-image contrast of Fig. S8b and c, its intervals and the split into hits and false alarms |
| `code/stats_figS8.py` | tests T31 and T32 |
| `code/make_figureS8.py` | Fig. S8 and its plotted values |
| `code/write_numbers.py` | `outputs/numbers.csv` and `outputs/pipeline_checks.csv` |
