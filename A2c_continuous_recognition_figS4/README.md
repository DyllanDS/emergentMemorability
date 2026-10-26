# A2c: Continuous Recognition (SI Appendix, Fig. S4)

This module reproduces SI Appendix, Section 2, Continuous Recognition, and SI Appendix, Fig. S4. It runs the REM
model trial by trial through the continuous recognition sequences of 400 participants of the THINGS memorability
study (Kramer et al., 2023), with simulated items that have no memorability of their own, and sets the model beside
the THINGS data. In both, d′ for the same item falls and false alarms rise when the items shown before it are more
similar. The module writes the paper's numbers with ledger ids L6-048 to L6-068, L6-261 and L6-262 and the
statistics of test T27 (collected in `statistics/all_tests.csv`).

Without the external inputs the module runs in provided mode. The trial structures the model runs on are not in
the repository (EXTERNAL_INPUTS.md, M6, built from the Kramer et al. trial file, M1), so the model numbers come from
the saved run in `data/provided/`, the run the paper reports, and the data points come from module A5's saved
output. The code checks each of these files against the text report of the same run before it uses their numbers.

## How to run

From the repository root:

    python A2c_continuous_recognition_figS4/code/run_module.py

Without options the module reruns the model when the trial structures are present and uses the saved run
otherwise. `--provided-only` and `--skip-long` (which `run_all.py` passes on from its options of the same names)
keep the saved run even when the trial structures are present. The module needs Python 3 with numpy, scipy and
matplotlib. In provided mode it takes about 3 s on an Apple M2 Pro (16 GB). With the trial structures, the model
rerun adds about 4 minutes (14,400 simulated sessions of 295 trials) and needs under 350 MB of memory.

`run_module.py` runs `check_model_mechanics.py`, `continuous_rem.py` (only with the trial structures),
`make_figureS4.py`, `stats_figS4.py` and `check_provided.py` in this order, each as
`python A2c_continuous_recognition_figS4/code/<script>` from the repository root, and stops at the first step that
fails. The last three take `--model-run provided`, or `--model-run regenerated` after a rerun.

## Inputs

- `data/provided/continuous_bins.npz`: the saved run of `continuous_rem.py` that the paper reports (seeds 7, 99
  and 1234). For each storage probability (keys `standard_*` and `accuracy-matched_*`) it holds the 20-bin curves
  of d′, hit rate, false alarm rate and criterion c, the Spearman correlations of the binned d′ and c with the bin
  number, the storage probability, the criterion, the overall rates and the correlations of corrected recognition
  with the per-item rates.
- `data/provided/continuous_rem.txt`: the text report of the same run; `figs4_inputs.py` checks every number it
  shares with the npz file.
- `../A5_things_context_effects_figS13_S14/data/provided/`: the data side, module A5's saved output (the
  whole-stream curves of SI Appendix, Fig. S13, open circles). `sdt_ctx_top10_target.npz` holds the 20 bins (hit
  rate, false alarm rate, d′ and c, each with its 95 percent limits from 400 resamples of the 21,137 participant
  sessions, and d_a), `sdt_results.txt` is its report, and `sdt_by_context_adjusted.txt` gives the across-image SD
  of d′ (0.379) used for one effect size.
- `../A3b_hits_and_false_alarms_figS9/data/provided/per_image_rates.txt`: the participants' mean per-image hit rate
  (0.808) and false alarm rate (0.0201) over 26,107 images with at least 30 old and 30 new trials.
- `shared/python/` (`release_paths`, `sdt`, `numbers_table`, `tests_table`, `bootstrap`, `figure_style`) and
  `shared/ledger/paper_numbers.csv`.

Not in the repository: the trial structures of the 400 sessions (EXTERNAL_INPUTS.md, M6), the structure-only file
`data/trial_templates_400.npz` (400 sessions x 295 trials; no responses, image numbers or identifiers). For each
trial it gives the slot (the distinct image of the session the trial shows, numbered in the order of the session's
image numbers) and the trial type (1 first showing of a target, 2 repeat of a target, 3 filler, 4 vigilance
repeat). The trial structures come from the Kramer et al. trial file (M1, `imbysub-all-11-16-21.mat`), which is not
public; its authors share it on request. With M1 in `external/kramer_trials/`, run
`python A5_things_context_effects_figS13_S14/code/run_module.py` first: its `extract_trials.py` writes the trial
table `external/kramer_trials/derived/trials.npz`, which this module then reads, and from which
`make_trial_templates.py` can write the template file. `continuous_rem.py` looks for the trial structures in this
order: the file given with `--templates`, `data/trial_templates_400.npz`, the file given with `--trial-table`,
`external/kramer_trials/derived/trials.npz`. Without any of them the module uses the saved run. The data points
always come from module A5's `data/provided/` folder.

## What the code does

**Model mechanics.** `check_model_mechanics.py` writes one session out trial by trial with explicit loops, as SI
Appendix, Section 2 describes the model, and runs it beside the model code on a made-up stream of 40 trials (28
items, 12 shown twice; no data needed) at storage 0.30 with criterion 5.236 and at 0.75 with 3.462. Both versions
must give the same response, context similarity and count of preceding trials on every trial with the same random
numbers (seed 20261013 for the 100 made-up items; 20261014, a fresh generator for each version and storage
probability, for the assignment of items to the stream and the encoding draws).

**The model.** `continuous_rem.py` uses REM with the standard values of Shiffrin and Steyvers (1997): g = 0.40, 20
features, copy fidelity c = 0.70 and storage probability u = 0.30. It generates 1,600 items with geometric feature
values on numpy's support {1, 2, ...} (no 1 is added), and each distinct image of a sequence is replaced by a
different item, drawn without replacement from the 1,600. On every trial the item on screen is compared with all
traces stored on earlier trials, and the model says "old" when the log of the mean likelihood ratio over those
traces exceeds the log of the criterion. The item is then stored as a new trace, whether it is a first showing or a
repeat: each feature is copied wrongly with probability 1 - c (a new geometric value) and stored with probability
u. No trace is removed or weakened.

**Criterion and storage.** The criterion at each storage probability comes from nine halvings on a logarithmic
scale between 0.2 and 2,000, each with 100 simulated participants (the first 100 sequences, with a fresh
`numpy.random.default_rng(99)` at every step), toward the participants' false alarm rate of 0.020 over first
showings (types 1 and 3); it is the geometric mean of the final bounds. The second storage probability is the
value of 0.30, 0.45, 0.60, 0.75 and 0.90 (tried in this order, each with its own criterion search) whose overall d′
is closest to 2.93, with hits over all repeats (types 2 and 4), 200 simulated participants on the first 200
sequences (`default_rng(1234)`) and rates kept within 0.0001 and 0.9999; a tie keeps the earlier value. The saved
run has 0.75, criteria of 5.236 (storage 0.30) and 3.462 (0.75), and false alarm rates of 0.0191 and 0.0204.

**Main runs.** Each storage probability runs 4,000 simulated participants, and simulated participant i follows
sequence i modulo 400. The sequences are the first 400 sessions with 295 trials, in the order of the trial table;
each has 187 distinct images, 50 targets shown twice and 137 fillers, 58 of which return as vigilance repeats.
`numpy.random.default_rng(7)` is one stream used in this order: the 1,600 x 20 item features, the standard run, then
the matched-accuracy run (for each participant the choice of items, then for each trial the copy draws, the new
values of wrongly copied features and the storage draws). The criterion search and the d′ check use their own
generators and leave this stream untouched, and everything runs serially in one process.

**Context and bins.** A trial's context is the mean similarity of the 10 most similar items shown before it, other
than the item itself (model: the proportion of features with the same value; data: the correlation of the two
concepts' 49 SPoSE dimensions, Hebart et al., 2020). A trial is binned once at least 50 earlier trials showed
other items in the model (20 in the data). Within each item, its n binned trials are ranked by context (rank r
from 0 to n - 1, ties in the order of the records) and go to bin floor(20 r / (n - 1)), capped at 19, so the
code's bins 0 to 19 are bins 1 to 20 of the figure; items with fewer than 20 such trials are left out. The ranks
run over the whole session, with no adjustment for serial position or lag, on both sides. False alarm trials
(first showings, types 1 and 3) and hit trials (target repeats, type 2; in the data, target repeats at a lag above
0) are binned separately. In each bin, hits and false alarms are pooled over participants, the rates are
log-linear corrected, (k + 0.5) / (n + 1), d′ = z(H) - z(F) and c = -(z(H) + z(F)) / 2 (`shared/python/sdt.py`,
the same functions as module A5). The Spearman correlations are over the 20 bins against the bin number.

**The report.** For each storage probability the report also lists the mean per-item hit and false alarm rates
(items with at least 30 trials of each kind) and the correlation of corrected recognition with each.

**Statistics (T27).** `stats_figS4.py` treats the contrasts between bins 20 and 1 as the inference (the change in
d′, the false alarm rate of bin 20 divided by that of bin 1, and the change in c) and the Spearman correlations over
the 20 bin means as descriptive. Module A5's saved file holds the 95 percent limits of each bin (400 resamples of
participant sessions, `numpy.random.default_rng(0)`) but not the resampled curves, so the data contrasts get no
interval here; test T38 of module A5 gives the interval of the same d′ contrast. The saved model run holds bin
means only; after a rerun, the model contrasts get percentile bootstrap intervals over the simulated participants
(10,000 resamples, `numpy.random.default_rng(20261013)`, one `integers(0, n, n)` draw per resample, in order). No
correction for multiple comparisons is applied.

## Outputs

- `outputs/numbers.csv`: each number of the paper that this module computes (23 rows: L6-048 to L6-068, L6-261 and
  L6-262), with the printed value, the computed value and whether they agree at the printed precision (columns in
  the top-level README).
- `outputs/tests.csv`: test T27, 16 rows (table below).
- `outputs/figures/FigS4_continuous_recognition.png`: SI Appendix, Fig. S4, panels a and b (300 dpi).
- `outputs/figure_data/FigS4a.csv`: for each bin, the data's d′, its change from bin 1 with the ends of the error
  bar, and the change in d′ of the two model lines. `FigS4b.csv`: for each bin, the data's false alarm rate with
  the ends of the error bar, and the two model lines.
- `outputs/model_mechanics_check.csv`: the checks of `check_model_mechanics.py`, one row per check and storage
  probability.
- After a rerun only: `outputs/continuous_rem.txt` and `continuous_bins.npz` (the new run, with the keys of
  `data/provided/`), `continuous_cell_counts.npz` (hits, hit trials, false alarms and false alarm trials per
  simulated participant and bin, the input of the bootstrap) and `equality_with_paper_run.csv`, which compares
  every array of the rerun with the saved run in `data/provided/`. A rerun with the trial structures of the same 400
  sessions reproduces every array of the saved run to within 1e-12.

| T27 row | Data (21,137 sessions) | Model, storage 0.30 | Model, storage 0.75 |
|---|---|---|---|
| d′, bin 20 minus bin 1 | -0.508 | -0.626 | -1.115 |
| Spearman ρ of the binned d′ with the bin number | -1.000 | -0.983 | -1.000 |
| false alarm rate, bin 20 / bin 1 (difference) | 2.13 (0.0153) | 2.43 (0.0183) | 10.18 (0.0524) |
| Spearman ρ of the binned false alarm rate | 0.998 | 0.977 | 1.000 |
| c, bin 20 minus bin 1 | -0.057 | | -0.401 |
| Spearman ρ of the binned c | -0.794 | | -0.997 |

The contrasts have n = 21,137 participant sessions or 4,000 simulated participants, the Spearman rows n = 20 bins.
The data's d′ contrast is -1.34 across-image SDs of d′ (SD 0.379 over 26,107 images).

## Code

| File | What it does |
|---|---|
| `code/run_module.py` | runs the steps in order and chooses the mode |
| `code/check_model_mechanics.py` | checks the model code against a version written with explicit loops from the description in SI Appendix, Section 2, on a made-up stream of 40 trials |
| `code/continuous_rem.py` | the REM model run trial by trial through the trial structures: the criterion search, the choice of the second storage probability, the two runs of 4,000 simulated participants and the 20-bin curves; writes the report, the curves and the counts per simulated participant |
| `code/make_trial_templates.py` | writes `data/trial_templates_400.npz` from a trial table and checks that it gives back the trial structures `continuous_rem.py` builds from that table |
| `code/figs4_inputs.py` | reads the model run and the data side and checks each file against its text report |
| `code/make_figureS4.py` | SI Appendix, Fig. S4 and the numbers it plots |
| `code/stats_figS4.py` | the statistics of test T27 |
| `code/check_provided.py` | every ledger number of the module, written to `numbers.csv` |
