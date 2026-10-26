# 05: THINGS feature reanalysis (Fig. 5)

This module reproduces Fig. 5 (panels a to d) and the related numbers in the main-text section
"Representational Analyses: Memorability Follows the Structure of the Image Set" (most of them in its
subsection "The Similarity Structure of an Image Set Determines What Will Be Memorable"), in the
Abstract, the Introduction and the General Discussion, and in the Methods sections "Representational
Similarity Analysis" and "Experimental Procedures and Analysis" (Statistical Reporting). It
reanalyzes the THINGS memorability scores of Kramer et al. (2023) at the level of the 49 features of
the THINGS similarity embedding (SPoSE; Hebart et al., 2020). For each feature, the code computes two
Spearman correlations, one with concept memorability and one with the concept's summed similarity to
all other THINGS concepts, and then correlates the two lists across the features. It does this in the
human similarity space, with the 49 SPoSE features (Fig. 5c: r = -0.612, 95% CI [-0.677, -0.505],
p < 0.001), and in a monkey IT space, with 49 principal components of macaque IT responses (Fig. 5d:
r = -0.296, 95% CI [-0.389, -0.176], p < 0.001). The interval and the p value of each correlation
come from a bootstrap over concepts. The module's statistics are tests T10 (Fig. 5c), T11 (Fig. 5d)
and T53 (the variance the 49 IT components keep) in `statistics/all_tests.csv`.

## How to run

From the repository root:

    matlab -batch "restoredefaultpath; cd('05_things_reanalysis_fig5/code'); run_module"
    python 05_things_reanalysis_fig5/code/stats_fig5.py

The MATLAB command draws Figs. 5a to 5d, computes the module's numbers and writes them to
`outputs/numbers.csv`. The Python command computes the bootstrap statistics and writes
`outputs/tests.csv` and their rows of `outputs/numbers.csv`. Each command keeps the rows the other
wrote, so the two can run in either order. They need MATLAB R2023b with the Statistics and Machine
Learning Toolbox (no parallel pool) and Python 3 with numpy, scipy and pandas. On the tested laptop
(Apple M2 Pro, 16 GB), the MATLAB part takes about 20 seconds plus the start of MATLAB, and the Python
part about 6 minutes in one process, most of it for the bootstrap.

`stats_fig5.py` takes `--resamples N` (concept-bootstrap resamples, default 10,000). With any other
number it prints its results and writes no file; `--resamples 100` runs every step in seconds.

## Inputs

All inputs are in the repository.

- `shared/things/concept_info_human.mat`: the 1,854 THINGS concepts with their 49 SPoSE features and
  feature labels, concept memorability, summed similarity and image counts (`conceptCounts`). SPoSE
  embedding of Hebart et al. (2020), osf.io/f5rn6, CC BY 4.0; memorability from Kramer et al. (2023),
  osf.io/5a7z6.
- `shared/things/concept_info_monkey.mat`: the same concept table with the 49 monkey IT component
  scores as the features. The components come from the THINGS Ventral Stream Spiking Dataset (Papale
  et al., 2025; gin.g-node.org/paolo_papale/TVSD, CC BY 4.0); the PCA script is
  `shared/things/provenance/monkeyFeatGen.m`.
- `shared/things/things_images.csv`: the public per-image memorability scores of Kramer et al. (2023),
  one row per image (26,107 images); step 2 below checks the concept memorability against them.
- `shared/things/monkey_pca_explained.csv`: the 576 eigenvalues of the monkey PCA, one row per
  component.
- `shared/things/concept_table.csv` and `shared/things/monkey_features.csv`: the same concept data as
  CSV tables (the same values as the two .mat files), read by `stats_fig5.py`.
- `shared/ledger/paper_numbers.csv`: the values printed in the paper, which `outputs/numbers.csv` is
  judged against.
- `data/paper_run/paperStats_kramer.mat`: the r of Figs. 5c and 5d from the saved run, the output
  the paper reports.
- Shared helpers in `shared/matlab/` (`release_paths.m`, `load_concept_info.m`, `piyg_colormap.m`, the
  3-D PCA helpers with their camera table `pca3d_feature_space_view.csv`, `save_panel.m`,
  `write_numbers_csv.m`) and `shared/python/` (`release_paths.py`, `concept_tables.py`, `bootstrap.py`,
  `numbers_table.py`, `tests_table.py`).

The raw spiking data of the monkey dataset are not included. The module needs only the component
scores and the eigenvalues above.

## What the code does

`run_module.m` calls `things_reanalysis.m` and `monkey_variance_explained.m`, writes
`outputs/numbers.csv` (keeping the rows that `stats_fig5.py` wrote), and stops with an error if a
judged value does not agree with the printed value.
`things_reanalysis.m` draws no random numbers and runs these steps in order.

1. `shared/matlab/load_concept_info.m` loads the two concept tables and adds the 1,854 x 1,854 matrix
   of Pearson correlations between the concepts' feature rows, `corrcoef(features')`. The
   dissimilarity of the Methods is 1 minus this correlation.
2. The code checks that each concept's memorability is the mean of the public per-image
   corrected-recognition scores of its images. Of the 1,854 concepts, 1,853 have a score; goalpost has
   none, because one of its images (goalpost_08s.jpg) has no public score.
3. The summed similarity of every concept is rebuilt and checked against the stored values. For
   concept i it is the sum over every other concept j of normpdf(r_ij, 1, 2) x (n_j / 26,107), where
   r_ij is the correlation of the two feature rows and n_j is the image count of concept j in the
   concept table (`conceptCounts`; column concept_image_count_as_stored of
   `shared/things/concept_table.csv`). The Gaussian kernel is centered on a correlation of 1 (a
   dissimilarity of 0) with SD 2, concept i is left out, and the weights of the other concepts are not
   renormalized. The monkey space uses the same image-count weights. Steps 4 and 5 use the stored
   values. Because a dissimilarity lies between 0 and 2, the kernel gives every concept at least 0.61
   times the weight of a concept at a dissimilarity of 0; the code reports the range of the
   dissimilarities between two concepts and of their weights in each space.
4. Each feature gets a Spearman correlation with memorability (the 1,853 concepts with a score) and
   one with summed similarity (all 1,854 concepts); tied values take their average rank. The 49
   correlations with summed similarity come from one `corr` call, because a loop over the columns
   changes the last binary digit of the final r.
5. The two lists are correlated across the 49 features (Pearson r). In the human space this is
   Fig. 5c. Fig. 5d repeats steps 3 to 5 in the monkey space, with the 49 IT components as the
   features on both axes.
6. Each panel is saved with the numbers it plots.
   - Fig. 5a: the 49 SPoSE features ranked by their correlation with memorability, in PiYG colors by
     rank (green highest), with a tick label on every fifth feature.
   - Fig. 5b: the coordinates are the first three right singular vectors of the 49 x 1,854 feature
     matrix after each concept's 49 values are centered on their mean (`pcaCoeffsExplained(features',
     3)`). Color shows summed similarity (flipped parula, yellow for low summed similarity, color
     limits 0.174 and 0.185). Marker size is 20 + 200 x n / 1,854 points squared for the concept at
     position n of the alphabetical THINGS concept list. The camera angles come from
     `shared/matlab/pca3d_feature_space_view.csv`. The two concepts the caption names, face and
     chainsaw, are marked and named, and the code reports their summed similarity (THINGS photographs
     are not included in the repository).
   - Figs. 5c and 5d: PiYG colors ordered by each feature's correlation with memorability, x axis
     from -0.6 to 0.6, y axis from -0.4 to 0.4, a gray least-squares line, and r printed with three
     decimals.
7. The run stops if r differs from the saved run in `data/paper_run/` by more than 1e-10.

The monkey features are the first 49 principal components of a PCA of the 1,854 x 576 matrix of
multi-unit responses of the IT channels of the two monkeys (monkey F channels 513 to 832, monkey N
channels 769 to 1,024), averaged over the images of each concept in the dataset's training set.
`monkey_variance_explained.m` computes the percent of the variance that the first 49 components keep
from the 576 eigenvalues (84.6 percent; T53).

`stats_fig5.py` computes the statistics of `outputs/tests.csv` with seed 20261013
(`numpy.random.default_rng`, through `shared/python/bootstrap.py`).

- Concept bootstrap: 10,000 resamples. Each resample draws `integers(0, 1854, 1854)`, and both spaces
  use the same draws. In each resample all 49 correlations with memorability and all 49 with summed
  similarity are recomputed on the drawn concepts, and then r. Each concept keeps its stored summed
  similarity, and the features are never resampled. The interval runs from the 2.5th to the 97.5th
  percentile of the resampled r. The two-sided p is twice the share of resamples on the far side of
  zero (r of 0 or above, since both correlations are negative), at most 1.
- T53 from the eigenvalues in `shared/things/monkey_pca_explained.csv`.

The script stops if r differs from the saved run by more than 1e-12, or if its T53 value differs from
`outputs/monkey_variance_explained.csv` (when that file exists). It adds six rows to
`outputs/numbers.csv` (the r, interval and p of Figs. 5c and 5d in the caption and in the text, their
restatement in the General Discussion, and the number of resamples on the far side of zero), keeps the
rows of the MATLAB part, and stops with an error if one of its judged values does not agree with the
printed value.

## Outputs

- `outputs/numbers.csv`: each number of the paper this module computes (37 ledger items: L3-001 to
  L3-014, L3-019, L3-059 to L3-061, L3-101 to L3-109, L4-004, L4-013 and L5-045 to L5-052), with the
  printed value, the computed value and whether they agree at the printed precision (columns in the
  top-level README). Rows without a printed number (figure panels, design facts and statements) give
  the computed values. `stats_fig5.py` writes the six rows of the bootstrap statistics and the MATLAB
  part the others.
- `outputs/tests.csv`: T10 (Fig. 5c) and T11 (Fig. 5d), Pearson r with its 95% concept-bootstrap
  interval and bootstrap p, and T53, the percent of variance.
- `outputs/fig5_statistics.csv`: per space, the number of features and concepts, Pearson r, the slope
  and intercept of the least-squares line, the r of the saved run in `data/paper_run/`, and the text
  printed in the panel.
- `outputs/monkey_variance_explained.csv`: components kept (49), components in total (576) and the
  percent of the variance kept.
- `outputs/figures/fig5a_ranked_feature_memorability.png`, `fig5b_pca_feature_space.png`,
  `fig5c_human_features_similarity_vs_memorability.png` and
  `fig5d_monkey_features_similarity_vs_memorability.png`: Figs. 5a to 5d (300 dpi).
- `outputs/figure_data/<the same names>.csv`: the numbers each panel plots (Fig. 5a: per feature, its
  rank, its correlation with memorability, whether it has a tick label and its color; Fig. 5b: per
  concept, the three coordinates, summed similarity, marker size and whether it is one of the two
  named concepts; Figs. 5c and 5d: per feature, both correlations, their ranks and the color).
- `outputs/logs/run_module.log` and `outputs/logs/stats_fig5.log`: what the scripts print, written
  when they run (the repository does not include them; the quick check writes no log).

## Code

| File | What it does |
|---|---|
| `code/run_module.m` | Runs the MATLAB part: calls the two scripts below and writes `outputs/numbers.csv`. |
| `code/things_reanalysis.m` | Figs. 5a to 5d, their two correlations and the checks; writes the panels, their data and `outputs/fig5_statistics.csv`. |
| `code/monkey_variance_explained.m` | The percent of variance the 49 monkey IT components keep; writes `outputs/monkey_variance_explained.csv`. |
| `code/stats_fig5.py` | Bootstrap intervals and p values of Figs. 5c and 5d, and T53; writes `outputs/tests.csv` and its rows of `outputs/numbers.csv`. |
