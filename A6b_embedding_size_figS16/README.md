# A6b: Embedding size (SI Appendix, Fig. S16)

This module reproduces SI Appendix, Fig. S16 and the numbers on embedding size in SI Appendix, Section 6
(the opening paragraph of "Generality Across Databases and Embeddings" and its subsection "Embedding
Size"), together with the main-text sentence that points to them in "The Similarity Structure of an Image
Set Determines What Will Be Memorable". It repeats the Fig. 5c analysis and the Fig. 6 correlations in
the published 49-feature SPoSE embedding (Zheng et al., 2019; Hebart et al., 2020) and in 14 refits of
that model to the THINGS-data odd-one-out judgments (Hebart et al., 2023), which the sparsity penalty
pruned from 100 features to between 6 and 81. Its statistics are test T47 in `statistics/all_tests.csv`.

## How to run

From the repository root:

    python A6b_embedding_size_figS16/code/run_module.py

This runs the module's six steps one after the other and takes about 7 minutes on the tested laptop
(Apple M2 Pro, 16 GB), nearly all of them in step 4 (T47). It needs Python 3 with numpy, scipy, pandas and
matplotlib. Steps 1 to 5 can also be run one at a time, in this order, with `MPLBACKEND=Agg` and
`PYTHONDONTWRITEBYTECODE=1` set as `environment.md` describes (step 6, which writes `outputs/numbers.csv`,
is part of `run_module.py`):

    python A6b_embedding_size_figS16/code/extract_behavior.py
    python A6b_embedding_size_figS16/code/run_refit_sweep.py
    python A6b_embedding_size_figS16/code/gather_all_fits.py
    python A6b_embedding_size_figS16/code/stats_figS16.py
    python A6b_embedding_size_figS16/code/make_figureS16.py

| Option of `run_module.py` | What it does |
|---|---|
| `--resamples N`, `--published-resamples N` | concept-bootstrap resamples per refit (default 2,000) and for the published embedding (default 10,000); smaller numbers are for testing the code and overwrite `outputs/tests.csv`, `outputs/figS16_statistics.csv` and `outputs/numbers.csv` |
| `--refit` | fits the 14 refits again before the analysis (below) |
| `--device mps`, `cuda` or `cpu`, `--epochs N`, `--triplets-folder DIR` | settings of `--refit` (defaults: `cuda` or `mps` where PyTorch finds that GPU, else `cpu`; 400 epochs; `external/spose_triplets`); `cpu` runs the same model on the processor, much more slowly, and fewer epochs are for testing the code |
| `--skip-long`, `--provided-only` | accepted from `run_all.py`; they have no effect in this module |

Fitting the refits again is optional, since every reported number comes from the saved refits:

    python external/download_external.py --spose-triplets
    python A6b_embedding_size_figS16/code/run_module.py --refit

This needs PyTorch and the THINGS-data triplets (`trainset.txt` and `validationset.txt` of osf.io/f5rn6,
CC BY 4.0, about 60 MB; `EXTERNAL_INPUTS.md`, M9), which the repository does not include, and adds about
3 hours on the GPU of the tested laptop. It runs the four commands that made the saved refits
(`REFIT_COMMANDS` in `code/run_module.py`) into `outputs/refit_rerun/`, then the six steps on the saved
refits, and then step 2 on the new refits (`outputs/refit_rerun/refit_sweep.csv`). Training on a GPU
does not give exactly the same weights twice, so new refits differ somewhat from the saved ones.

## Inputs

- `data/refits/spose_lam<lambda>_seed<seed>_d<features>.txt`: the 14 saved refits made by
  `code/refit/spose.py`, one row per THINGS concept (1,854, in THINGS order) and one column per feature,
  with weights to 6 decimals.
- `data/refits/lamsweep_summary.json`, `lamfill_summary.json`, `sparse_s1_summary.json` and
  `sparse_s2_summary.json`: one summary file per refit command, with one record per fit (penalty
  strength `lmbda`, features kept `dim`, held-out cross-entropy and accuracy, stopping epoch, seed, file
  name, and the features kept and held-out fit at each evaluation). The lambda sweep also lists three
  fits that are not among the 14 refits and have no embedding file: at lambda 0.0035 and 0.004 the
  penalty removes no feature, and at 0.019 it removes every feature.
- `data/paper_run/all_fits.csv`: the saved run of this analysis, the output the paper reports, with one
  row per embedding. `k` is the number of features, `fig5c_r` and `fig5c_p` are the Fig. 5c statistic
  and its parametric p, and `maintain_vs_orig`, `maintain_sim`, `changed_vs_orig` and `changed_sim` are
  Figs. 6c, 6d, 6f and 6g (`TABLE_COLUMNS` in `code/gather_all_fits.py` pairs every column with its
  column in `outputs/refit_sweep.csv`).
- `shared/things/concept_table.csv` (the concept memorability of Kramer et al., 2023, and the number of
  THINGS images of each concept), `shared/things/spose_embedding_49d_sorted.txt` (the published
  embedding) and `shared/things/monkey_features.csv` (the 49 monkey IT components of Fig. 6).
- Read in place from module 06: `06_image_set_experiments_fig6_fig7/data/maintained_trials.csv` and
  `changed_trials.csv` (205 and 195 de-identified participants of the image-set experiments, one row per
  study or test trial), `sampled_image_sets.mat` (the number of images of each concept in each set) and
  `paper_run/paperStats_maintained_vs_changed.mat` (module 06's saved run of the six Fig. 6 correlations).
- `shared/ledger/paper_numbers.csv`: the values printed in the paper, which `outputs/numbers.csv` is
  judged against.
- Shared helpers in `shared/python/`: `release_paths.py`, `concept_tables.py`, `imageset_loader.py`,
  `bootstrap.py`, `tests_table.py`, `numbers_table.py` and `figure_style.py`.
- Not included: the THINGS-data triplets (`EXTERNAL_INPUTS.md`, M9), which only `--refit` needs.

## What the code does

Step 1 (`extract_behavior.py`) computes the memorability of each concept in the maintained and changed
image sets. The shared loader (`shared/python/imageset_loader.py`) applies the exclusions of the Methods
and keeps the first 150 remaining participants of each set in participant-id order (55 and 45
participants are excluded). A rating of 1 to 3 on the 6-point scale counts as an "old" response, and,
pooled over the kept participants, a concept's memorability is its hit rate minus its false alarm rate
(no score in a set where the concept has no old or no new test trial).

Step 2 (`run_refit_sweep.py`, with the functions of `analysis_core.py`) computes the Fig. 5c statistic
in each embedding:

1. the similarity of two concepts is the Pearson correlation of their feature rows;
2. the kernel similarity is normpdf(similarity, 1, 2);
3. the summed similarity of concept i is the sum, over every other concept j, of the kernel similarity
   times j's number of THINGS images divided by the total minus one (i's own count is reduced by one and
   its own term is left out);
4. each feature gets a Spearman correlation with memorability, over the 1,853 concepts with a score, and
   one with summed similarity, over all 1,854 (tied values take their average rank);
5. the two lists are correlated across the features: Pearson r, with a two-sided t-test p on the number
   of features minus 2 degrees of freedom.

The six Fig. 6 correlations repeat items 1 to 4 within each image set, with the memorability of step 1
and with summed similarity weighted by each concept's share of the images in that set (the concept's own
count is reduced by one and its own term is kept; concepts with no image in the set get no value). Figs.
6c and 6f correlate each feature's memorability correlation in all of THINGS with the one in the
maintained or changed set, 6d and 6g correlate the summed-similarity and memorability correlations within
each set, and 6e and 6h do the same with the 49 monkey IT components as the features. The monkey
components do not depend on the SPoSE embedding, so 6e and 6h take the same value in every refit (r =
-0.63 and -0.45, as in Fig. 6). The ranks are stored column by column in memory, which fixes the order
of later sums and so the last digit of every result.

Step 3 (`gather_all_fits.py`) writes the results in the layout of `data/paper_run/all_fits.csv`,
compares them value by value with that saved run, and compares the published embedding's six Fig. 6
correlations with module 06's saved run. It stops with an error if a row is missing from the saved run,
a text cell differs, or a number differs by more than 1e-10.

Step 4 (`stats_figS16.py`) computes T47 with seed 20261013 (`numpy.random.default_rng`, through
`shared/python/bootstrap.py`). Each embedding starts a new generator from the seed, so each row
reproduces on its own. Each bootstrap resample draws
`integers(0, 1854, 1854)` concepts, recomputes both per-feature correlations over them (each concept
keeps its summed similarity from the full embedding) and then r, and the interval runs from the 2.5th to
the 97.5th percentile of the resampled r. Each refit gets 2,000 resamples and keeps the parametric p of
step 2. The published embedding gets 10,000, the same draws as test T10 of module 05, and, as there, its
p is twice the share of resamples on the far side of zero, so both give the same interval and p. No
correction for multiple comparisons is applied.

Step 5 (`make_figureS16.py`) draws Fig. S16 from `outputs/all_fits.csv`, and step 6 writes
`outputs/numbers.csv`. `run_module.py` runs the steps serially and gives the numerical library one
processor core (it sets `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
`VECLIB_MAXIMUM_THREADS` to 1 when they are unset); the outputs are the same with more threads.

## Settings of the refits

`code/refit/spose.py` implements SPoSE in PyTorch. Each fit starts from 1,854 x 100 weights drawn
uniformly between 0 and 0.1. The similarity of two concepts is the dot product of their weights, the
choice probability is a softmax over the three pairwise similarities of a triplet, and the loss is the
mean cross-entropy of the observed odd-one-out choice plus (lambda / 1,854) x the sum of the absolute
weights. Training uses Adam (learning rate 0.001) on batches of 8,192 of the 4,120,663 training triplets,
in a new random order each epoch (503 batches; the 87 triplets left over are skipped in that epoch), and
clamps the weights at zero after every step. Every 5 epochs the fit is evaluated on the 453,642
validation triplets. Training stops once the last 12 evaluations show the same number of surviving
features (at the earliest after 65 epochs) or at 400 epochs, and it keeps the final weights rather than
the best-validation ones. A feature survives when at least one concept's weight on it is above 0.1, and
the surviving features are sorted by summed weight, largest first.

The four commands of `REFIT_COMMANDS` fit lambda 0.0035, 0.004, 0.0045, 0.005, 0.006, 0.007, 0.008,
0.009, 0.011, 0.013, 0.016 and 0.019 (seed 0), lambda 0.0135, 0.014 and 0.0145 (seed 0), and lambda
0.006 with seeds 1 and 2, on the Apple GPU (`mps`). The seed sets `torch.manual_seed` and a CPU
generator that draws the initial weights and the order of each epoch, and every lambda of a command
starts again from the seed. The 14 refits span lambda 0.0045 to 0.016, and their training stops after
65 to 400 epochs.

## Outputs

- `outputs/numbers.csv`: each number of the paper this module computes (15 ledger items: L3-015,
  L9-001, L9-025, L9-027 to L9-037 and L9-161), with the printed value, the computed value and
  whether they agree at the printed precision (columns in the top-level README). Rows without a printed
  number give the computed values.
- `outputs/tests.csv`: T47, one row for each of the 15 embeddings, with Pearson r, n (the number of
  features), the p (parametric with its df for the refits, bootstrap for the published embedding) and
  the 95% concept-bootstrap interval.
- `outputs/refit_sweep.csv`: one row per embedding (the published embedding first, then the refits from
  fewest to most features) with the fit's settings and held-out fit, its share of weights below 0.1,
  how its summed similarity correlates with the published embedding's and with memorability, and the
  Fig. 5c statistic with its parametric p and the six Fig. 6 correlations.
- `outputs/all_fits.csv`: the same results in the layout of `data/paper_run/all_fits.csv`, plus
  `maintain_sim_monkey` (Fig. 6e) and `changed_sim_monkey` (Fig. 6h).
- `outputs/all_fits_vs_paper_run.csv`: compares this run with the saved run in `data/paper_run/`, the
  output the paper reports (per column: values compared, values identical, largest absolute
  difference), and the published embedding's Fig. 6 correlations with module 06's saved run.
- `outputs/figS16_statistics.csv`: per embedding, r, n, df, the parametric p and the bootstrap
  interval, with its settings.
- `outputs/image_set_concept_memorability.csv`: per concept (1,854, THINGS order) and image set, the old
  and new test trials, hits, false alarms, hit rate, false alarm rate and memorability.
- `outputs/figures/FigS16_spose_dimensionality.png`: Fig. S16 (300 dpi), and
  `outputs/figure_data/FigS16_spose_dimensionality.csv`, the points it plots.
- `outputs/logs/`: what each step prints, written when the module runs.
- `outputs/refit_rerun/` (with `--refit` only): the new refits with their summary files and training
  logs, in the subfolders `lamsweep`, `lamfill` and `sparsityctl`, and `refit_sweep.csv`, their Fig. 5c
  and Fig. 6 statistics.

## Code

| File | What it does |
|---|---|
| `code/run_module.py` | Runs steps 1 to 5, writes `outputs/numbers.csv` (step 6), and holds and runs the refit commands for `--refit`. |
| `code/extract_behavior.py` | Step 1: the memorability of each concept in the maintained and changed image sets. |
| `code/run_refit_sweep.py` | Step 2: the Fig. 5c and Fig. 6 statistics in the published embedding and each refit. |
| `code/analysis_core.py` | The Fig. 5c and Fig. 6 computations in any embedding, and the loaders of their inputs. |
| `code/gather_all_fits.py` | Step 3: `outputs/all_fits.csv` and its comparison with the saved runs. |
| `code/stats_figS16.py` | Step 4: T47, the p and concept-bootstrap interval of each embedding. |
| `code/make_figureS16.py` | Step 5: Fig. S16 and its plotted numbers. |
| `code/refit/spose.py` | SPoSE in PyTorch, which made the refits (used only with `--refit`). |
