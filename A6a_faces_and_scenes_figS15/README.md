# A6a: Faces and scenes (SI Appendix, Fig. S15)

This module reproduces SI Appendix, Fig. S15 and the "Faces and Scenes" part of SI Appendix, Section 6
(Generality Across Databases and Embeddings), which the main-text section "The Similarity Structure of an
Image Set Determines What Will Be Memorable" cites. Summed similarity computed from deep-network features
predicts memorability (hit rate minus false alarm rate) in a face database (Bainbridge, Isola and Oliva,
2013) and in a scene database (Isola, Xiao, Torralba and Oliva, 2011). For each database the module computes
the held-out correlation of summed similarity with memorability, with all features weighted equally or with
weights fitted on other images (Fig. S15a), the whole-sample correlations with the hit rate and the false
alarm rate (Fig. S15b), and the split-half reliability of memorability (the dashed lines of Fig. S15a). It
writes the statistics T43 (Fig. S15a), T44 (Fig. S15b) and T45 (reliability), which
`statistics/all_tests.csv` collects. The license of the face database does not allow redistribution, so
the faces values come from saved aggregate results unless the database is present; the scene analyses run
on the public scene data and the embeddings in `shared/embeddings_isola/`.

## How to run

From the repository root:

    python A6a_faces_and_scenes_figS15/code/run_module.py

This takes about 3 minutes on the tested laptop (Apple M2 Pro, 16 GB), and the ridge step uses about 2 GB
of memory. With `--reuse-scene-outputs` the module skips the three scene analyses and recomputes the
comparison, the intervals, the statistics and the figure from the reports already in `outputs/scenes/`
(a few seconds). Each step runs as its own Python process, in a fixed order because later steps read the
reports of earlier ones, and its output is shown and saved in `outputs/logs/<step>.log`. The module needs
Python 3.11 or 3.12 with numpy, scipy, pandas, matplotlib, scikit-learn and h5py (versions in
`requirements.txt`); h5py reads the MATLAB v7.3 participant matrices.

## Inputs

- `data/scenes/image_order.csv`: the 10,442 scene images of Isola et al. (2011), 2,400 targets and 8,042
  fillers, one row per image, sorted by file name; every scene embedding lists its rows in this order.
  Columns: filename, has_scores (1 for every image), hit_rate, false_alarm_rate, corrected_recognition,
  kind (target or filler), in_cvpr_2222 (1 for the 2,222 targets that Isola et al. analyzed, the scored
  images of every analysis here; the other 178 targets and all fillers are context only), index_in_set
  (position in the package's list of targets or of fillers), hits, misses, false_alarms and
  correct_rejections.
- `data/scenes/isola_scores.csv`: the same public scores in the order of the package (the 2,400 targets,
  then the 8,042 fillers), with each image's path in the package (sun_filename), exported with
  `shared/embedding_extraction/export_isola.m`. `image_order.csv` is this table sorted by file name, which
  `run_module.py` checks.
- `data/scenes/subject_matrices.mat`: the 843 usable participant records of the public `subj_data.mat` as
  MATLAB v7.3 matrices (tdet and tfa for the 2,400 targets, fdet and ffa for the fillers; 1 = response,
  -1 = no response, 0 = not seen; no identifiers), exported with
  `shared/embedding_extraction/export_subjects.m`. The scene reliability over participants is computed
  from these matrices.
- `shared/embeddings_isola/regenerated_<network>_<layer>.npz`: the 14 scene spaces, 10,442 x 512, rows in
  the order of `image_order.csv` (`shared/embeddings_isola/README.md`).
- `data/paper_run_scenes/`: the saved scene run, the output the paper reports (`isola_crowding.txt`,
  `ridge_metric.txt`, `ridge_metric_folds.csv`, `noise_ceiling.txt`, `noise_ceiling.json`).
- `data/provided_faces/`: the saved faces results, aggregate only, in the same five files
  (`faces_crowding.txt` in place of `isola_crowding.txt`).
- `shared/ledger/paper_numbers.csv`: the printed values.
- The shared helpers `release_paths.py`, `numbers_table.py`, `tests_table.py`, `bootstrap.py` and
  `figure_style.py` in `shared/python/`.
- Not included: the 10k US Adult Faces Database (EXTERNAL_INPUTS.md, M4), which its authors share on
  request (wilmabainbridge.com/facememorability2.html) under a license that does not allow
  redistribution. Without it the module uses the saved results in `data/provided_faces/` and prints
  `PROVIDED OUTPUT: regenerating needs ...; see EXTERNAL_INPUTS.md`. To recompute the faces, put
  `10kfaces.zip` and `2kmemorability.zip` (or their unzipped folders) into `external/bainbridge_faces/`
  and run the module. `code/faces/regenerate_faces.py` then unzips them into
  `external/bainbridge_faces/derived/unzipped/`, builds the image list (reading `memorability-scores.xlsx`
  needs openpyxl), extracts the 14 face spaces into `derived/regenerated_embeddings/` with
  `shared/embedding_extraction/extract_embeddings.py` (PyTorch, torchvision, timm, the OpenAI clip package
  and cornet; about an hour; DINOv3 needs about 17 GB of memory in this extractor, and
  `shared/embedding_extraction/README.md` gives a low-memory route; spaces put into that folder by any
  route are used as they are) and runs the three faces analyses (a few minutes). Their aggregate reports go
  to `outputs/faces/` and are used in every table and in the figure, and `regenerate_faces.py` prints the
  recomputed values beside the saved ones. Everything per face stays in `external/bainbridge_faces/derived/`
  and must not be redistributed. A missing package or an image count other than 10,168 stops the run, and
  no model, embedding or image set is substituted.
- Optional: the Isola et al. (2011) package (EXTERNAL_INPUTS.md, M8). Without
  `data/scenes/subject_matrices.mat`, `run_module.py` rebuilds the matrices from `external/isola/Data/`
  (`python external/download_external.py --isola`, then `export_subjects.m` through MATLAB, found as
  `run_all.py` finds it); without the package or MATLAB, or with `--skip-matlab`, it reads the scene
  reliability from `data/paper_run_scenes/` and prints the provided-output message.

## What the code does

`run_module.py` checks the scene inputs (14 spaces; `image_order.csv` equal to `isola_scores.csv` sorted
by file name), runs the faces chain or reads the saved faces results, runs the three scene analyses, and
then `compare_scenes_runs.py`, `intervals.py`, `stats_figS15.py` and `make_figureS15.py`. The faces
scripts in `code/faces/` make the same computations as the scene scripts in `code/scenes/`.

- **Spaces.** Seven networks, each at an early and a late layer (AlexNet, VGG16-BN, ResNet18,
  InceptionV3, CORnet-S, DINOv3 ViT-B/16 and CLIP ViT-B/32), from the embedding pipeline of Brady and
  Robinson (2026); `shared/embedding_extraction/README.md` lists the layers, weights and preprocessing.
  Every space has 512 dimensions (larger layers are reduced to 512 principal components). Every image of a
  database is context (10,168 faces; 10,442 scenes, fillers included), and the correlations use the 2,222
  scored images (for the scenes, the targets with in_cvpr_2222 = 1).
- **Fig. S15b (`02_crowding_analysis.py`).** In each space, the correlation distance from every image to
  every other image is z-scored with the mean and SD of a 1,500-image sample of that space (numpy
  `default_rng(0)`, spaces in file-name order), and the 14 z-scored distances are averaged. Summed
  similarity with equal weights is here the negated mean of this ensemble distance to every other image of
  the database; for large tau, a sum of exp(-d / tau) over the other images orders the images in the same
  way. Its Spearman rho with the hit rate and with the false alarm rate over the 2,222 scored images (whole
  sample, not held out) is in the "Limiting case" block of the crowding report.
- **Fig. S15a (`04_ridge_metric.py`).** The 14 spaces are z-scored per dimension and joined (7,168
  dimensions). Summed similarity of an image is the log of the sum, over every other image, of
  exp(-(d - offset) / tau), with d the correlation distance in the joined space, z-scored with the mean and
  SD of a 1,000-image sample (`default_rng(0)`), and offset the 1st percentile of that sample's z-scored
  distances. The 2,222 scored images are split into five folds (`KFold(5, shuffle=True,
  random_state=0)`). On the four training folds, tau is chosen from 22 values (0.05 to 500, geometric) by
  the largest absolute Spearman rho with memorability. For fitted weights, a ridge regression of
  memorability on the 7,168 dimensions gives each dimension a weight, its squared coefficient rescaled to
  mean 1. Its penalty alpha (10, 100, 1,000, 10,000 or 100,000) is the one whose predictions have the
  highest mean Spearman rho with memorability over a `KFold(3, shuffle=True, random_state=0)` split of the
  training images. The dimensions are multiplied by the square root of their weights before summed
  similarity is recomputed. Each bar of Fig. S15a is the mean of the five held-out fold correlations (the
  rows "crowding, isotropic metric" and "crowding, ridge-weighted metric" of the report, which also lists
  the tau and alpha chosen in each fold). The p of each bar is the largest of its five single-fold p
  values (t approximation for Spearman's rho with 444 images).
- **Error bars (`intervals.py`).** 95% Fisher-z intervals with the Bonett and Wright (2000) standard error
  sqrt((1 + rho^2 / 2) / (n - 3)), n = 2,222, around the three-decimal rho of each report (for Fig. S15a,
  the mean of the five folds).
- **Reliability (`05_noise_ceiling.py`).** The Spearman rho between two random halves, averaged over 25
  splits (`default_rng(0)`) and uncorrected, gives the dashed lines of Fig. S15a. For the faces, each
  image's own old and new trials are split in half (hypergeometric draws), since the database has no
  per-participant responses; for the scenes, the 843 participants are split in half. The report also
  gives the Spearman-Brown value 2r / (1 + r) (faces 0.799, scenes 0.845); SI Appendix, Section 6
  reports their square roots, 0.89 and 0.92, as the highest correlation the scores allow.
- **Statistics (`stats_figS15.py`).** The script recomputes each fold correlation from
  `ridge_heldout_scores.csv` and each whole-sample correlation from `isola_crowding.npz`, and runs its own
  split-half code with seed 0 and 25 splits; it stops unless these agree with the reports (the split-half
  value must equal the scene reliability of `data/paper_run_scenes/`, 0.73161, exactly). There is no
  correction for multiple comparisons.

`code/read_results.py` reads every value from the reports by its row label, never by line number. The scene
values printed in the paper are those of the saved run in `data/paper_run_scenes/`, and this run rounds to
every one of them (`outputs/scenes_rerun_vs_paper_run.csv`).

## Outputs

- `outputs/numbers.csv`: each number of the paper this module computes (20 ledger rows), with the
  printed value, the computed value and whether they agree at the printed precision (columns in the
  top-level README).
- `outputs/tests.csv`: T43 (Fig. S15a; for each bar, the mean of the five fold correlations with its
  largest-fold p and Fisher-z interval) and T44 (Fig. S15b; the whole-sample rho with the hit rate and
  the false alarm rate, with its p and Fisher-z interval), from the saved runs in `data/provided_faces/`
  and `data/paper_run_scenes/`, and T45 (the 25-split reliabilities with their Spearman-Brown values,
  from the same sources as `outputs/numbers.csv`).
- `outputs/figures/FigS15_faces_scenes_generality.png`: Fig. S15a and b (faces from the saved results
  unless the face database is present, scenes from this run).
- `outputs/figure_data/FigS15a.csv`, `FigS15b.csv`: the plotted values and their intervals; the
  `paper_run_*` columns give the saved runs of `data/provided_faces/` and `data/paper_run_scenes/`.
- `outputs/intervals_figS15.csv`: the interval and p value of every correlation of Fig. S15 for each
  source of results.
- `outputs/scenes/`: this scene run, with the reports `isola_crowding.txt`, `ridge_metric.txt` and
  `noise_ceiling.txt`, `isola_crowding_meta.json` (the spaces and image counts), `isola_crowding.npz`
  (per-image crowding and scores), `ridge_metric_folds.csv` (the five held-out fold correlations),
  `ridge_heldout_scores.csv` (each scored image's fold and held-out summed similarity), `noise_ceiling.json`
  (the reliabilities at full precision) and `noise_ceiling_splits.csv` (the 25 split values).
- `outputs/faces/`: the aggregate faces reports, written only when the face database is present (M4).
- `outputs/scenes_rerun_vs_paper_run.csv`: every scene number of this run beside the saved run in
  `data/paper_run_scenes/`, with the difference and, for printed values, whether this run rounds to the
  printed value.
- `outputs/logs/`: the output of each step, written when the module runs.

## Code

| File | What it does |
|---|---|
| `code/run_module.py` | runs the steps above in order |
| `code/scenes/02_crowding_analysis.py` | ensemble distance and whole-sample crowding of the scenes (Fig. S15b) |
| `code/scenes/04_ridge_metric.py` | held-out summed similarity with equal and fitted weights, five folds (Fig. S15a) |
| `code/scenes/05_noise_ceiling.py` | split-half reliability over participants |
| `code/faces/00_build_image_order.py` | the face image list with the scores of the database (runs only with the database) |
| `code/faces/02_crowding_analysis.py`, `04_ridge_metric.py`, `05_noise_ceiling.py` | the same analyses for the faces (run only with the database) |
| `code/faces/regenerate_faces.py` | the faces chain: archives, image list, embeddings and the three analyses |
| `code/read_results.py` | where the reports are and how values are read from them, by row label |
| `code/compare_scenes_runs.py` | this scene run beside the saved run in `data/paper_run_scenes/` |
| `code/intervals.py` | the 95% intervals of Fig. S15 and the p value of each panel a bar |
| `code/stats_figS15.py` | `outputs/numbers.csv` and `outputs/tests.csv` |
| `code/make_figureS15.py` | Fig. S15 and its figure data |
