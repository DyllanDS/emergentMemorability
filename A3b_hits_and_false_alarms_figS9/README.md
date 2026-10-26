# A3b: Hits and false alarms (SI Appendix, Fig. S9)

This module reproduces SI Appendix, Fig. S9 (panels a to c) with the numbers of its caption and of SI Appendix,
Section 3, "Similar Items Change Memory Mainly Through False Alarms". It also computes the numbers of the
paragraph of SI Appendix, Section 1 on false alarms near floor and of the sentences that restate it in "The
Similarity Structure of an Image Set Determines What Will Be Memorable", the Methods and the General Discussion.
Panels a and b score hits and false alarms separately for the 40 critical items of the memorability
inversion experiment (Fig. 3, module 03), over its 50 kept participants, and are computed from data in the
repository. Panel c correlates the memorability score of the original THINGS data (Kramer et al., 2023) with
the hit rate and with the false alarm rate across 26,107 photographs; it comes from saved results unless the
Kramer et al. trial records are added (input M1 of `EXTERNAL_INPUTS.md`). The module writes the numbers of
these passages to `outputs/numbers.csv` and tests T33 (Fig. S9a, b) and T34 (Fig. S9c) to `outputs/tests.csv`,
which `statistics/all_tests.csv` collects. The sentence of the Section 3 subsection on the image set
experiments (Fig. S8c) is computed by module A3a.

## How to run

From the repository root:

    python A3b_hits_and_false_alarms_figS9/code/run_module.py

The module takes about 20 seconds on the tested laptop (Apple M2 Pro, 16 GB), most of it in the statistics
step. It needs Python 3 with numpy, scipy, pandas and matplotlib (versions in `requirements.txt`),
no MATLAB, and the folders `03_inversion_experiment_fig3/data/` and `shared/` of the repository.
`--resamples N` sets the number of bootstrap resamples of the statistics step (default 10,000, the number
behind the reported intervals); a small number such as 200 checks the code in a few seconds.

`run_module.py` runs four steps, each as its own process, and stops at the first one that fails. The same
steps can be run one at a time, in this order:

    python A3b_hits_and_false_alarms_figS9/code/per_image_rates.py
    python A3b_hits_and_false_alarms_figS9/code/make_figureS9.py
    python A3b_hits_and_false_alarms_figS9/code/stats_figS9.py
    python A3b_hits_and_false_alarms_figS9/code/write_numbers.py

`write_numbers.py` reads the tables and figure data that `make_figureS9.py` writes and the intervals of the
three Spearman correlations that `stats_figS9.py` writes to `outputs/tests.csv`. With input M1 in place the
three later steps read the report that `per_image_rates.py` writes.

## Inputs

- `03_inversion_experiment_fig3/data/inversion_trials.csv` and `inversion_design.csv` (module 03): the trials of
  the 54 sessions of the memorability inversion experiment and its 40 critical items (block, memorability
  quintile, number of similar items, published score). `shared/python/inversion_loader.py` reads them and
  applies the exclusions described in the Methods (4 participants excluded, 50 kept). When module 03 has run,
  the 40-item table is also checked against its `outputs/inversion_items_50.csv` (counts and rates identical,
  published scores within 1e-12).
- `data/paper_run/figS9_numbers.txt`: the numbers of the saved run of Fig. S9, the output the paper reports
  (the per-level means with their error bars, the three Spearman correlations, the trial counts per item, the
  statistic printed on panel b and the values of panel c).
- `data/provided/per_image_rates.txt`: the saved report of `code/per_image_rates.py` on the Kramer et al.
  trial records (number of photographs, median numbers of old and new trials, mean and SD of each rate, and
  the two correlations of panel c), used when input M1 is not available. Module A2c also reads this file by
  its line labels.
- `data/provided/halves_check.txt`: saved results for the THINGS side of the halves comparison (the mean hit
  and false alarm rates of the more and the less memorable critical items in the original THINGS data), used
  when input M2 is not available.
- `shared/ledger/paper_numbers.csv` (the printed values), `shared/things/things_images.csv` (the image number
  of each THINGS photograph, read only with input M2), and the shared helpers `inversion_loader.py`, `sdt.py`,
  `bootstrap.py`, `numbers_table.py`, `tests_table.py`, `figure_style.py` and `release_paths.py`.
- Not included (`EXTERNAL_INPUTS.md`): the Kramer et al. (2023) trial file `imbysub-all-11-16-21.mat` (input
  M1, not public; its authors share it on request) and the per-image count table `things_per_image_counts.csv`
  (input M2, not public; columns image_nr, n_old, hits, n_new and false_alarms; module A4b uses the same
  file). Both go in `external/kramer_trials/`. Without them the module prints
  `PROVIDED OUTPUT: regenerating needs <input>; see EXTERNAL_INPUTS.md` and uses the saved results in
  `data/provided/`.

With M1 in place, module A5's extract step (`A5_things_context_effects_figS13_S14/code/extract_trials.py`,
which needs h5py and takes long) turns the trial file into `external/kramer_trials/derived/trials.npz`; step
1 runs it by itself when that table is missing. Step 1 then writes `outputs/regenerated/per_image_rates.txt`
and compares it with the saved report, and the figure, the statistics and `numbers.csv` use the regenerated
report. With M2 in place, step 2 computes the THINGS side of the halves comparison and compares it with the
saved results. M2 does not regenerate panel c, which counts only the long-lag target repeats of the trial
records.

## What the code does

**The 40 critical items.** An item's rates pool its test trials over the 50 kept participants. Each
participant was tested once on each critical item, as old or as new, so after the response-time filter (200
to 10,000 ms) every item has 48 to 50 test trials (at least 18 old and 17 new). A response of 1 to 3 on the
six-point confidence scale counts as "old". The hit rate is hits over old trials, the false alarm rate is
false alarms over new trials, and d′ is z((hits + 0.5) / (old trials + 1)) - z((false alarms + 0.5) / (new
trials + 1)), the log-linear correction (Hautus, 1995), computed item by item and then averaged over the items
of a level.

**Panels a and b** (`make_figureS9.py`). Each point is the mean over the 8 items studied with the same number
of similar items (0, 1, 2, 4 or 8). The error bars are the 2.5th and 97.5th percentiles (numpy's linear
interpolation) of the means of 10,000 resamples of those 8 items, drawn with numpy `default_rng(0)` as one
10,000 x 8 matrix of item positions per level, in the order 0, 1, 2, 4, 8. The Spearman correlations over the
40 items take the designed number of similar items as the predictor (5 values of 8 items each, so tied values
get their average rank), with scipy's two-sided p from the t approximation with 38 df. From 0 to 8 similar
items the false alarm rate rises from 0.051 to 0.570 (ρ = 0.719, p = 1.7e-07), the hit rate from 0.724 to
0.847 (ρ = 0.43498, p = 0.005), and d′ falls from 2.106 to 0.900 (ρ = -0.4946, p = 0.001). The script stops if
any of these numbers, the error bars, the trial counts or the statistic printed on panel b differs at the
printed digits from the saved run in `data/paper_run/`.

**The halves comparison.** The 40 items are sorted by their published score (a stable sort) and split into
the 20 more memorable items (published scores 0.7955 to 0.8889) and the 20 less memorable ones (0.5682 to
0.7895); the script stops if the 20th and 21st scores are tied. For each half it gives the mean hit and false
alarm rates in the original THINGS data and in this experiment. In the more memorable half the hit rate is
0.832 in THINGS and 0.839 in this experiment and the false alarm rate 0.0175 and 0.454; in the less memorable
half the hit rate is 0.762 and 0.747 and the false alarm rate 0.0214 and 0.189. The THINGS side comes from
the per-image count table (input M2), whose old trials include the short-lag vigilance repeats, or from the
saved values in `data/provided/halves_check.txt`. The script also gives the mean published score of the
items at each number of similar items (0.671, 0.742, 0.791, 0.843 and 0.874).

**Panel c** (`per_image_rates.py`). From the Kramer et al. trial records the script takes every THINGS
photograph with at least 30 old and 30 new trials (26,107 photographs, with medians of 40 old and 154 new
trials). Old trials are the long-lag target repeats (trial type 2; the short-lag vigilance repeats, type 4,
are not counted), new trials are first showings (types 1 and 3), a hit is response code 11 and a false
alarm is response code 13. Across photographs, the memorability score (hit rate minus false alarm rate)
correlates with the hit rate at r = 0.9873 and with the false alarm rate at r = -0.0773 (Pearson), with a
mean false alarm rate of 0.0201 (SD 0.0127) and a mean hit rate of 0.8080 (SD 0.0797). Without the trial
records the script reads these values from the saved report in `data/provided/`.

**Statistics** (`stats_figS9.py`). One generator, numpy `default_rng(20261013)`, draws 10,000 sets of 50
participant indices (`integers(0, 50, 50)`, not used further) and then 10,000 item resamples
(`integers(0, 40, 40)`). An item resample draws the 40 items with replacement, each keeping its level. The
95% intervals of the three correlations are [0.498, 0.851] for false alarms, [0.150, 0.654] for hits and
[-0.751, -0.1746] for d′. The mean of each measure at each level comes with the error bar of the figure. For
T34 the script reads the two Pearson correlations of panel c from the report and adds n = 26,107, df =
26,105, the two-sided p of the report (a p below double precision is written as 0) and the Fisher z 95%
interval ([0.987, 0.988] and [-0.089, -0.0652]). No correction for multiple comparisons is applied.

## Outputs

- `outputs/numbers.csv`: each number of the paper this module computes (24 rows), with the printed value, the
  computed value and whether they agree at the printed precision; statements and figure descriptions without
  a printed number are listed as documented (columns in the top-level README). Values read from
  `data/provided/` carry the mode `provided` (`restatement` for the three sentences outside the SI Appendix).
- `outputs/tests.csv`: 20 rows. T33 gives the three Spearman correlations (ρ, n = 40 items, df = 38,
  two-sided p, item interval) and the mean of each measure at each level with the error bar of the figure.
  T34 gives the two Pearson correlations of panel c.
- `outputs/figures/FigS9_hits_false_alarms.png`: Fig. S9 (300 dpi).
- `outputs/figure_data/FigS9a.csv`, `FigS9b.csv` and `FigS9c.csv`: the numbers each panel plots (x positions,
  means, interval ends, the statistics printed on panels b and c, and the file panel c was read from).
- `outputs/level_table.csv`: per level, the number of items, the mean hit rate, false alarm rate and d′ with
  the ends of their error bars, the hit rate minus the false alarm rate, and the mean published score.
- `outputs/level_correlations.csv`: the three Spearman correlations with their two-sided p.
- `outputs/halves.csv`: the halves comparison (range of published scores, rates in this experiment and in
  THINGS, and the source of the THINGS side).
- `outputs/regenerated/per_image_rates.txt`: the regenerated report of panel c, written only with input M1.
- `outputs/logs/`: the printout of each step, written when the code runs.

## Code

| File | What it does |
|---|---|
| `code/run_module.py` | runs the four steps in order and keeps each step's printout in `outputs/logs/` |
| `code/per_image_rates.py` | panel c: the per-image rates of the original THINGS data and their correlations with the memorability score |
| `code/make_figureS9.py` | Fig. S9 and its plotted numbers, the per-level table, the correlations, the halves comparison, and the comparison with the saved run |
| `code/stats_figS9.py` | tests T33 and T34 (`outputs/tests.csv`) |
| `code/write_numbers.py` | `outputs/numbers.csv` |
| `code/a3b_common.py` | what the scripts share: the 40-item table, the number of kept participants, the readers of the saved files and the switches to regeneration |
