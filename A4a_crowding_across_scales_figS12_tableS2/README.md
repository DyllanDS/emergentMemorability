# A4a: One Measure of Distinctiveness Across Analyses (SI Appendix, Fig. S12 and Table S2)

This module reproduces SI Appendix, Section 4 (One Measure of Distinctiveness Across Analyses), SI Appendix, Fig. S12
and SI Appendix, Table S2, and the sentences of the main text and SI Appendix, Section 1 that refer to them. It
applies one crowding measure to seven datasets and varies only how local the measure is: the REM simulation of
Fig. 1, REM with SPoSE features (SI Appendix, Fig. S6), the hybrid model with rarity computed from the study list
(SI Appendix, Fig. S7), the THINGS memorability data of Kramer et al. (2023), and the maintained, changed and reduced
image sets (Figs. 6 and 7). The module writes the paper's numbers with ledger ids N4-001 to N4-026 and the statistics
of tests T58 to T63 (collected in `statistics/all_tests.csv`). Other modules compute two results of SI Appendix,
Section 4: the same image after a more similar stream (SI Appendix, Fig. S13; module A5) and the crowding of THINGS
concepts in the deep-network spaces (module A4b).

The THINGS results need the trial file of Kramer et al. (EXTERNAL_INPUTS.md, M1). Without it the module reads them
from the saved correlations in `data/provided/things_results.json`, and it computes everything else from files in
the repository.

## How to run

From the repository root:

    python A4a_crowding_across_scales_figS12_tableS2/code/run_module.py

The module takes about 10 seconds and under 1 GB of memory on the tested laptop (Apple M2 Pro, 16 GB). It needs
Python 3 with numpy, scipy, pandas and matplotlib (versions in `requirements.txt`). It reads files of modules 01,
06, A2e and A2f and, with M1, the trial table that module A5 writes, so `run_all.py` runs it after them.
`--provided-only` (which `run_all.py` passes on from its option of the same name) reads the THINGS results from
`data/provided/` even when the trial table is present.

## Inputs

- `01_rem_simulation_fig1/data/paper_run/fig1_paper_run.mat`: the 1,600 generated items of Fig. 1 with their hit
  and false alarm rates.
- `A2e_measured_representations_figS6/outputs/continuous_rem_items_sigma0.05.csv`: the hits, false alarms and
  test counts of REM with SPoSE features at retention noise 0.05, one row per concept (1,600 concepts).
- `A2f_hybrid_exemplar_model_figS7/outputs/core_items_basic.csv` (variant "Hybrid GCM, list") and
  `A2f_hybrid_exemplar_model_figS7/code/hgcm.py`: the hybrid model's counts for each item and the code that
  rebuilds its items.
- `06_image_set_experiments_fig6_fig7/data/` and
  `06_image_set_experiments_fig6_fig7/outputs/summed_similarity_by_concept.csv`: the image-set trials, read with
  `shared/python/imageset_loader.py` and its exclusions (150 participants kept in each set), and the slots of each
  concept in each set.
- `shared/things/concept_table.csv`: the 49 SPoSE features, the higher-level category (23 categories; 530 concepts
  have none), the image count and the published memorability score of each of the 1,854 THINGS concepts.
- `data/provided/things_results.json`: the saved THINGS correlations (crowding against each memory measure at
  each τ, overall and within categories; the THINGS rows of Table S2; the two category correlations; and the
  correlation with the published scores). It holds no value for a single trial, image or concept.
- Not included (EXTERNAL_INPUTS.md, M1): the trial file of Kramer et al., which module A5 turns into the trial
  table `external/kramer_trials/derived/trials.npz`. Without the trial table the module prints
  `PROVIDED OUTPUT: regenerating needs the Kramer et al. trial table that module A5 writes from the trial file
  (M1, expected at external/kramer_trials/derived/trials.npz); see EXTERNAL_INPUTS.md` and reads the THINGS
  results from `data/provided/things_results.json`. With it, the module computes them from the trials, writes them
  to `outputs/regenerated/things_results.json` and prints their largest difference from the saved file.

## What the code does

**Crowding.** For item i at falloff τ, crowding is sum_j w_j exp(-d_ij / τ) / sum_j w_j over the other items j of
a reference set, and a higher value means a more crowded item. For concepts, d is 1 minus the Pearson correlation
of the two concepts' 49 SPoSE values; for simulated items, it is the Euclidean distance divided by its median over
all pairs of items. The weights are the image counts of all 1,854 concepts for THINGS, the slots of each concept in
the set for the image sets, and 1 for each simulated item (for REM with SPoSE features, each of the 1,600
simulated concepts). τ takes the values 0.05, 0.1, 0.2, 0.4 and 1, and τ = 10^6 stands for the all-items limit, in
which every item of the reference set counts equally.

**Memory measures.** Each item has a hit rate, a false alarm rate, corrected recognition (hit rate minus false
alarm rate) and d′ = z(hit rate) - z(false alarm rate), computed with both rates clipped to [0.005, 0.995]. For
THINGS, a concept's hits are counted on target repeats (trial type 2) and its false alarms on first showings
(types 1 and 3), pooled over sessions, for the concepts with trials of both kinds. The module also correlates
their difference with the published concept scores (Spearman ρ). In the image sets, a concept's trials are pooled
over the participants of the set.

**SI Appendix, Fig. S12.** For each dataset and τ, the module computes the Spearman correlation across items
between crowding and each measure. For THINGS and the image sets it also computes the correlation within
higher-level categories: crowding and the measure are each centered on their category means, and concepts without
a category are left out.

**SI Appendix, Table S2.** For THINGS and the maintained and changed sets, at τ = 0.05, 0.2, 1 and all items, the
table gives the statistic of Fig. 5c: the Pearson correlation across the 49 features between each feature's
Spearman correlation with crowding and its Spearman correlation with corrected recognition or with d′.

**The same concept in two sets (T58 to T61).** Over the concepts tested in both the maintained and the reduced set
(559), and over those tested in both the maintained and the changed set (203), the module correlates (Spearman)
the change in crowding at τ = 0.05 with the change in false alarm rate and with the change in hit rate.

**THINGS categories (T62, T63).** Over the 1,324 THINGS concepts with a higher-level category, the module
correlates crowding at τ = 0.05 with hit rate, once using each concept's category means (between categories) and
once using its deviations from them (within categories). Every concept carries the means of its category, so the
between-category correlation rests on 23 distinct pairs of values and has no p value.

**The reduced set.** The spread of crowding is its SD across the concepts of a set, computed against the set's own
slots over the concepts with slots in it. The module divides the spread in the reduced set by that in the
maintained set at τ = 0.05, 0.2 and all items.

The hybrid model's items are rebuilt with the seed and draws of A2f's `01_core.py` (numpy `default_rng(1)`), and
the module stops when their crowding does not match `core_items_basic.csv`. No other step draws random numbers.

## Outputs

- `outputs/numbers.csv`: each number of the paper this module computes, with the printed value, the computed value
  and whether they agree at the printed precision (columns in the top-level README).
- `outputs/tests.csv`: T58 to T61 and T63, each a Spearman ρ with n, df, two-sided p and a 95% interval (Fisher z
  with the Bonett and Wright (2000) standard error), and T62, the between-category ρ with its n.
- `outputs/figures/FigS12_crowding_across_scales.png`: SI Appendix, Fig. S12, panels a to c (300 dpi).
- `outputs/figure_data/FigS12_crowding_across_scales.csv`: the correlations each panel plots (panel, measure,
  dataset, τ, ρ).
- `outputs/locality_profiles.csv`: the correlation of crowding with each of the four memory measures for every
  dataset and τ, and its within-category version (dataset, τ, scope, measure, ρ, n).
- `outputs/table_s2.csv`: SI Appendix, Table S2.
- `outputs/concept_changes.csv`, `outputs/things_categories.csv` and `outputs/reduced_set_spread.csv`: the last three
  analyses above.
- `outputs/locality_results.txt`: a text report of the analyses.
- `outputs/logs/run_module.log`: the screen output of the run.
- With M1: `outputs/regenerated/things_results.json`, the THINGS results computed from the trials, in the format of
  `data/provided/things_results.json`.

## Code

| File | What it does |
|---|---|
| `code/run_module.py` | chooses the source of the THINGS results, runs `locality_analyses.py` and `make_figureS12.py`, and writes `numbers.csv` and `tests.csv` |
| `code/locality_common.py` | the crowding measure and the seven datasets |
| `code/locality_analyses.py` | the correlations, Table S2, the concept changes, the THINGS categories and the spread in the reduced set |
| `code/make_figureS12.py` | SI Appendix, Fig. S12 |
