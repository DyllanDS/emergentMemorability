# 03: Memorability inversion experiment (Fig. 3)

This module reproduces the memorability inversion experiment (the section "Inverting Memorability Through
Direct Manipulation of Distinctiveness" and the Methods sections "Behavioral Validation Experiments" and
"Experimental Procedures and Analysis"). From the deployed block lists and the de-identified trials of the
54 sessions it counts the design that the schematic of Fig. 3a to 3c shows (the module does not draw these
panels), applies the exclusion rules of the Methods (4 participants excluded, 50 kept) and draws Fig. 3d
and 3e. Its main statistic is the Spearman rank correlation between the published memorability of the 40
critical items (Kramer et al., 2023) and their corrected recognition in this task (rho = -0.50,
95% CI [-0.76, -0.18], p = 0.001), which the General Discussion restates. It also computes the uncertainty statistics of tests
T06 (Fig. 3e) and T54 (participant flow), which `statistics/all_tests.csv` collects. Modules A3a and A3b
read the same trials and check their item tables against `outputs/inversion_items_50.csv`, so this module
runs before them.

## How to run

From the repository root:

    matlab -batch "restoredefaultpath; cd('03_inversion_experiment_fig3/code'); run_module"
    python 03_inversion_experiment_fig3/code/stats_fig3.py

Run the MATLAB step first. It needs MATLAB R2023b with the Statistics and Machine Learning Toolbox (no
parallel pool) and takes under a minute on the tested laptop (Apple M2 Pro, 16 GB), about 15 s after
MATLAB starts. The Python step needs Python 3 with numpy, scipy and pandas (versions in `requirements.txt`)
and takes about 10 s; `--resamples N` sets the number of bootstrap resamples (default 10,000, the number
used for the paper), and a small number such as 200 checks the code in a few seconds. The Python step reads
`outputs/inversion_items_50.csv`, `outputs/participant_flow.csv` and `outputs/numbers.csv` from the MATLAB
step. These files are included in the repository, so it also runs without MATLAB (`python run_all.py --module 03 --skip-matlab`).

## Inputs

- `data/inversion_trials.csv`: every study and test trial of the 54 sessions, one row per trial (320
  study and 80 test rows per session). Columns: participant (S001 to S054), block (1 to 8), phase (study
  or test), trial_in_block, image_name, concept, condition (critical, controlled or maxDissimilar), and
  on test rows is_new, response_1to6 (1 = sure old to 6 = sure new) and rt_ms. Recruitment ids,
  timestamps, browser data and practice and instruction trials are not included. The study display
  time and interstimulus interval of the Methods are settings of the experiment software and are not
  in the file.
- `data/inversion_design.csv`: the 40 critical items: block, image_name, concept, memorability_quintile
  (1 = lowest 20 percent of the published scores), n_similar_items and kramer_cr (the published score of
  the image, as text from the public file of Kramer et al., 2023).
- `data/inversion_deployed_lists.csv`: the study and test lists of the 8 blocks as the experiment used
  them, 40 study and 10 test rows per block: block, phase, position, image_name, condition, concept,
  old_new and similar_to_critical_item (for a controlled item, the critical item it is similar to; none
  for the non-similar exemplars; empty on the rows of the critical and maximally dissimilar
  items). Which critical and maximally dissimilar items a participant studied was drawn per participant
  and block, so the old_new labels of these rows are one example draw; each participant's own draw is in
  `inversion_trials.csv`.
- `data/paper_run/paperStats_inversion.mat`: the saved run of this analysis, the output the paper
  reports. The struct `paperStats` holds rho (`r_memVsCR`), p (`p_memVsCR`) and the three participant
  counts (`nParticipantsFinal`, `nParticipantsBeforeExclusion`, `nExcluded`).
- `data/paper_run/published_scores.csv`: the published score of each critical item as the saved run
  used it (image_name and kramer_cr, the mean of the image's score over the item's test trials at full
  precision).
- `shared/things/things_images.csv` (the published per-image scores of Kramer et al., 2023, and the image
  paths), `shared/ledger/paper_numbers.csv` (the printed values), the MATLAB helpers `release_paths.m`,
  `load_inversion_trials.m`, `apply_participant_exclusions.m`, `piyg_colormap.m` and `write_numbers_csv.m`
  in `shared/matlab/`, and the Python helpers `release_paths.py`, `inversion_loader.py`, `bootstrap.py`,
  `numbers_table.py` and `tests_table.py` in `shared/python/`.
- Not included: the THINGS images (EXTERNAL_INPUTS.md, M7), whose license does not allow
  redistribution. They are optional: placed in `external/things_images/<concept>/<file>.jpg`, they are
  drawn as thumbnails in Fig. 3e. Without them the panel shows markers and concept names, and no number
  changes.

## What the code does

`run_module.m` runs four MATLAB functions in order; none of them draws random numbers.

1. `inversion_design_check.m` counts the design from the lists and from every block of the 54 sessions
   and records whether every block and session agrees:
   - 8 blocks. In each, participants studied 40 images and then made 10 recognition judgments (the 5
     critical items and 5 maximally dissimilar items of the block) on a 6-point scale from 1 (sure old)
     to 6 (sure new). Responses 1 to 3 count as "old".
   - One critical item per block from each quintile of the published scores of the 26,106 THINGS images
     that have one (MATLAB `prctile`; cuts 0.7250, 0.7750, 0.8182 and 0.8571; an image is in bin k when
     cut(k) <= score < cut(k + 1)): 40 critical items, with published scores from 0.568 to 0.889.
   - 0, 1, 2, 4 and 8 similar items studied with the critical items of quintiles 1 to 5. They include
     images of the item's own concept and of related concepts (for example 5 poles, a tee, a stake and a
     lamppost for the pole of block 1).
   - A study list of 35 controlled items (the 15 similar items and 20 non-similar exemplars) and
     5 further items, 2 or 3 of the block's critical items and 3 or 2 of its maximally dissimilar test
     items, drawn per participant and block. The critical and maximally dissimilar items a participant
     did not study are the lures of the test.
   - Each critical item was studied in 20 to 36 of the 54 sessions and was a lure in 18 to 34.

   The stimulus-selection code, `shared/things/provenance/SelectStimToFlipMemorability.m`, needs an
   image similarity file that is not included, so the module counts the design from
   `data/inversion_deployed_lists.csv` and does not repeat the selection.
2. `participant_flow.m` applies the exclusion rules of the Methods through
   `shared/matlab/apply_participant_exclusions.m`, the function `inversion_analysis.m` also uses. A
   participant is excluded when, over the 80 test trials, one response key was used on more than 80
   percent of trials (same key), more than 20 percent of trials were faster than 200 ms (too fast) or
   slower than 7,500 ms (too slow), or the hit rate minus the false alarm rate was below 0.05 (CR below
   0.05). Test trials faster than 200 ms or slower than 10,000 ms are then dropped (19 of the 4,000 test
   trials of the kept participants), and the first 50 participants are kept, which removes nobody. The
   function recomputes every participant's shares from the trials and stops if a flag does not follow
   these thresholds.

   | Sessions | Same key | Too fast | Too slow | CR below 0.05 | Excluded | Kept |
   |---|---|---|---|---|---|---|
   | 54 | 0 | 3 | 0 | 4 | 4 (S004, S005, S019, S026) | 50 |

   Three of the four excluded participants meet two rules (too fast and CR below 0.05).
3. `inversion_analysis.m` computes, for each critical item over the kept participants, the hit rate
   (hits over the item's old trials), the false alarm rate (false alarms over its new trials) and its
   memorability in this task, corrected recognition (CR = hit rate minus false alarm rate). Trials are
   pooled over participants (18 to 33 old and 17 to 31 new trials per item). The published memorability
   of an item is the mean of its image's Kramer et al. (2023) score over the item's test trials (every
   trial of an item shows the same image). Several critical items share a published score, and the
   full-precision means of the saved run fix the order of their ranks. The function reads them from
   `data/paper_run/published_scores.csv`, stops if one differs from the mean it computes by more than
   1e-12, and writes them to `outputs/inversion_items_50.csv` (column kramer_cr). The Fig. 3e statistic
   is the Spearman rank correlation over the 40 items (MATLAB `corr` with `'Type', 'Spearman'`: average
   ranks for ties, two-sided p from the t approximation with 38 df): rho = -0.5004, p = 0.0010. The
   function compares rho, p and the three participant counts with the saved run in `data/paper_run/`.
   It stops with an error when a printed value would change (rho at two decimals, p at three decimals
   and against 0.01, the counts exactly) and prints a warning when only the last digits differ.
4. `write_module_numbers.m` writes `outputs/numbers.csv`, to which `stats_fig3.py` adds the two rows
   that print the Fig. 3e interval (L2-014, L2-015).

`stats_fig3.py` checks that `outputs/inversion_items_50.csv` and `outputs/participant_flow.csv` equal
what the shared Python loader computes from `data/` (it stops if anything differs), and recomputes rho
and p with scipy, compared with the saved run in the same way. For the intervals it uses one generator,
numpy `default_rng(20261013)`, in this order:

1. 10,000 participant resamples: the 50 kept participants are drawn with replacement, and every item's
   hit rate, false alarm rate and CR are recomputed from the drawn participants' trials (a participant
   drawn twice counts twice), with the items and their published scores fixed. These give the
   additional participant interval of T06.
2. 10,000 item resamples: the 40 items are drawn with replacement, each keeping its published score and
   CR. These give the interval the paper reports for Fig. 3e.

The item interval comes from the second block of draws, so the participant resamples must be drawn
first. Module A3a repeats both blocks of draws for its test T31 and checks that its intervals equal these.
All intervals are 95% percentile bootstrap intervals; no correction for multiple comparisons is applied.

## Outputs

- `outputs/numbers.csv`: each number of the paper this module computes (26 rows), with the printed
  value, the computed value and whether they agree at the printed precision; statements and figure
  descriptions without a printed number are listed as documented (columns in the top-level README).
- `outputs/tests.csv`: T06, Spearman rho = -0.500 over 40 items (50 participants), df = 38, p = 0.0010
  (two-sided), with the 95% interval over items [-0.758, -0.180] (printed in the paper as
  [-0.76, -0.18]) and the additional interval over participants [-0.569, -0.318]; T54, the
  participant flow (54 sessions, 4 excluded, 50 kept).
- `outputs/inversion_items_50.csv`: the 40 critical items over the 50 kept participants, in image-name
  order: image_name, block, memorability_quintile, n_similar_items, kramer_cr (the published score as
  the analysis uses it, full precision), n_old, hits, n_new, false_alarms, hit_rate, fa_rate and cr.
- `outputs/fig3_statistics.csv`: rho, p and the three participant counts of this run beside the saved
  run in `data/paper_run/`, and their difference.
- `outputs/inversion_design_check.csv`: each design fact above and whether every block and session agrees.
- `outputs/participant_flow.csv`: the T54 row. `outputs/participant_exclusion_flags.csv`: one row per
  session, with the shares behind each rule and the flags.
- `outputs/figures/fig3d_rank_order_lines.png` (Fig. 3d, 300 dpi): the 40 items ranked by published
  memorability (left) and by CR in this task (right), one line per item, rank 1 at the top, colored by
  published rank on the PiYG scale of `piyg_colormap.m` (green for the highest, pink for the lowest),
  each end labeled with the concept name (and the image code where two items share a concept).
- `outputs/figures/fig3e_memorability_vs_cr.png` (Fig. 3e, 300 dpi): CR in this task against the
  published score, one marker per item in the color of its Fig. 3d line, labeled with its concept (or a
  thumbnail from `external/things_images/`), rho and p in the panel, no fitted line. The x axis runs from
  0.50 to 1.00 and the y axis from -0.2 to 1.00 (ticks 0.00, 0.50, 1.00), so every item is inside.
- `outputs/figure_data/fig3d_rank_order_lines.csv` and `fig3e_memorability_vs_cr.csv`: the numbers each
  panel plots (scores, CR, ranks, line heights and colors).

## Code

| File | What it does |
|---|---|
| `code/run_module.m` | runs the four MATLAB steps in order |
| `code/inversion_design_check.m` | counts the design from the lists and the trial records |
| `code/participant_flow.m` | applies the exclusion rules and writes the participant flow |
| `code/inversion_analysis.m` | item table, Fig. 3d and 3e, rank correlation, comparison with the saved run |
| `code/write_module_numbers.m` | writes `outputs/numbers.csv` |
| `code/stats_fig3.py` | tests T06 and T54 (`outputs/tests.csv`); rows L2-014 and L2-015 of `outputs/numbers.csv` |
