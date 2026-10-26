# A4b: Crowding among THINGS concepts (SI Appendix, Section 4)

This module reproduces the THINGS concepts sentence of SI Appendix, Section 4 (One Measure of Distinctiveness
Across Analyses, paragraph "Corrected recognition can reverse the sign"). It computes the crowding of each of
the 1,854 THINGS concepts (Hebart et al., 2019) among its five nearest concepts in the 14 deep-network spaces
of SI Appendix, Section 5 (A Similarity Space Without Human Judgments) and correlates crowding with each
concept's false alarm rate (ρ = 0.33), hit rate (ρ = 0.07), d′ (ρ = -0.18) and corrected recognition
(ρ = 0.03, p = 0.23) in the THINGS memorability data of Kramer et al. (2023). Crowding is computed from concept
centroids included in the repository. The concept memorability needs a per-image count table that the
repository does not include (input M2 of `EXTERNAL_INPUTS.md`), so without it the four correlations come from
the saved results in `data/provided/`. The module writes ledger items L9-020 to L9-024 in `outputs/numbers.csv`
and test T46 in `outputs/tests.csv`, which `statistics/all_tests.csv` collects. It draws no figure.

## How to run

From the repository root:

    python A4b_things_concepts/code/run_module.py

The module takes about 5 seconds on the tested laptop (Apple M2 Pro, 16 GB) and under 700 MB of memory, or
about 30 seconds with the count table (M2), most of it in the 10,000-resample concept bootstrap. It needs
Python 3 with numpy, scipy and pandas (versions in `requirements.txt`), runs in a single process and has no
options. `code/concept_level.py` also runs alone (`python A4b_things_concepts/code/concept_level.py`); it writes
the crowding table, and with the count table also the report.

## Inputs

- `shared/embeddings_things/concept_centroids/` (14 files, one per network and layer): each holds `centroids`
  (1,854 x 512, float32), `concepts` (the concept names, sorted by name) and `image_counts` (the number of
  photographs of each concept). `shared/embedding_extraction/make_concept_centroids.py` makes them from the
  photograph-level embeddings (see `shared/embeddings_things/README.md`).
- `shared/things/things_images.csv`: the concept, public image number and embedding row of every photograph,
  and the THINGS-order number of every concept.
- `shared/ledger/paper_numbers.csv`: the values printed in the paper, which the checks and `numbers.csv` use.
- `data/provided/concept_level.txt` and `concept_level.json`: the saved results of this analysis, used when the
  count table is not available. The txt is the report, and the json holds the same correlations at full
  precision (`by_k`: for k = 5, four [rho, p] pairs in the order corrected recognition, hit rate, false alarm
  rate, d′).
- The shared helpers `release_paths.py`, `concept_tables.py`, `numbers_table.py`, `tests_table.py` and
  `bootstrap.py` in `shared/python/`.
- Not included (`EXTERNAL_INPUTS.md`, input M2): the per-image count table `things_per_image_counts.csv`, which
  derives from the Kramer et al. trial data and is not public. It goes in `external/kramer_trials/`. Without it
  the module prints `PROVIDED OUTPUT: regenerating needs the per-image THINGS count table ... (M2, expected at
  external/kramer_trials/things_per_image_counts.csv); see EXTERNAL_INPUTS.md` and uses the saved results.

The count table has one row per THINGS photograph (26,107 rows) and the columns image_nr (the public image
number of `shared/things/things_images.csv`), n_old, hits, n_new and false_alarms. n_old counts hits and misses
over every old trial of the photograph, the short-lag vigilance repeats included, and n_new counts false alarms
and correct rejections. The code does not build this table from the Kramer et al. trial file (input M1), so M1
alone does not switch the module to regeneration. With the table in place, the same command computes the
concept memorability and the four correlations, writes `outputs/regenerated/` and adds the concept-bootstrap
intervals to T46.

## What the code does

**Concept centroids.** Every photograph is described in 14 spaces, an early and a late layer of AlexNet, CLIP
ViT-B/32, CORnet-S, DINOv3 ViT-B/16, InceptionV3, ResNet18 and VGG16-BN, each with 512 dimensions
(`shared/embedding_extraction/README.md`). In each space every column is standardized over the
26,107 photographs ((x - mean) / (SD + 1e-8)) and the photographs of each concept are averaged. The 14
centroids of a concept are placed side by side in sorted file-name order, which gives 7,168 values per concept,
all weighted equally.

**Distance and crowding.** The distance between two concepts is 1 minus the Pearson correlation of their
centroids. Each distance is scaled as (distance - mean) / SD, with the mean and SD (numpy `std`, ddof = 0, plus
1e-8) taken over the 499,500 distances among 1,000 concepts drawn with numpy
`default_rng(0).choice(1854, size=1000, replace=False)`, the only random draw of the analysis. The crowding of
a concept is minus the mean scaled distance to its 5 nearest other concepts (`REPORTED_NEIGHBORHOOD_SIZE` in
`code/concept_level.py`), so higher values mean more crowded concepts. The distances are computed in float32,
512 concepts at a time, and crowding values can differ between computers by about 1e-5.

**Concept memorability** (count table only). For each photograph the hit rate is hits / n_old, the false alarm
rate is false_alarms / n_new, and corrected recognition is the hit rate minus the false alarm rate. A concept's
three rates are the means over its photographs, summed in the row order of the embedding files, and a
photograph without old (or new) trials is left out of that mean. A concept's d′ is z(mean hit rate) minus
z(mean false alarm rate), with both rates clipped to 0.001 and 0.999, not the mean of per-photograph d′
values. The concept means stay in memory and are not written to disk, because they derive from the Kramer et
al. trial data, which are not public.

**Correlations.** Spearman ρ across the 1,854 concepts (sorted by name), with pairs that have a missing value
dropped and scipy's two-sided p from the t approximation (df = 1,852). The saved results give ρ = 0.3253 for the
false alarm rate, 0.072 for the hit rate, -0.177 for d′ and 0.028 (p = 0.234) for corrected recognition. The
txt gives p to one significant digit, so the module reads every p from the json.

**Checks.** Every run reads the saved report by its row labels and checks that every ρ and p of its k table
equals the json at the printed precision (4 cells), that the saved results count the same
photographs (26,107), concepts (1,854, median 13 photographs), dimensions (7,168) and spaces (14) as the
included centroids, and that the four correlations of the k = 5 row (and the p of corrected recognition)
round to the values printed in the paper. A failed check stops the run. With the count table,
the run also prints the largest differences between its k = 5 correlations and the saved ones.

**Statistics (T46).** For each correlation, in the order of the paper (false alarms, hits, d′, corrected
recognition), `code/stats_concepts.py` writes two rows. The first gives ρ, n = 1,854 concepts, df = 1,852, the
two-sided p at full precision and a 95% interval from the Fisher z transform with the Bonett and Wright (2000)
standard error for Spearman ρ, `sqrt((1 + rho^2 / 2) / (n - 3))`: false alarms 0.3253 [0.283, 0.366], hits 0.072
[0.026, 0.117], d′ -0.177 [-0.221, -0.132] and corrected recognition 0.028 [-0.018, 0.073]. Each p is also
recomputed from ρ and n as a check. The second row is a percentile bootstrap over concepts, computed only with
the count table. It draws 10,000 resamples with numpy `default_rng(20261013)`, calling `integers(0, 1854,
1854)` once per resample, and each concept keeps its crowding (computed once over all 1,854 concepts) and its
memorability. The interval runs from the 2.5th to the 97.5th percentile of the resampled ρ (numpy
`nanpercentile`). No correction for multiple comparisons is applied.

## Outputs

- `outputs/numbers.csv`: the numbers of the paper this module computes (L9-020 to L9-024), with the printed
  value, the computed value and whether they agree at the printed precision (columns in the top-level README).
  Values read from `data/provided/` carry the mode `provided`.
- `outputs/tests.csv`: test T46, two rows per correlation (8 rows), as described above.
- `outputs/concept_crowding.csv`: one row per concept (sorted by name) with concept,
  concept_number_things_order (the concept's row in `shared/things/unique_id.txt`), photographs and
  crowding_k5. Written on every run.
- `outputs/regenerated/concept_level.txt` and `.json`: the report and json of this run in the format of
  `data/provided/` (the k = 5 row), written only with the count table.
- `outputs/logs/run_module.log`: the printout of the run, written when the code runs.

## Code

| File | What it does |
|---|---|
| `code/run_module.py` | entry point: runs the analysis and the checks, writes `outputs/numbers.csv`, `outputs/tests.csv` and the log, and stops if a check of the saved results fails |
| `code/concept_level.py` | the analysis: concept centroids and crowding, and with the count table the concept memorability, the correlations and the report |
| `code/check_provided.py` | reads the saved results in `data/provided/` by row label and checks them; compares a run with the count table with the saved results |
| `code/stats_concepts.py` | test T46 (`outputs/tests.csv`) |
