# shared/: what more than one module uses

This folder holds what the modules share: the loaders of the participant data, the signal-detection and
resampling helpers, the writers of `numbers.csv` and `tests.csv`, the figure style, the THINGS concept and
image tables with the SPoSE embedding, the deep-network embeddings and the scripts that make them, and the
ledger of the paper's numbers. Modules read these files in place through relative paths and keep no
copies of them.

A module script reaches the helpers through one relative path:

    # Python, in <module>/code/<script>.py
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
    import release_paths
    paths = release_paths.module_paths(__file__)      # data, outputs, figures, figure_data, logs of the module

    % MATLAB, in <module>/code/<script>.m
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared', 'matlab'));
    paths = release_paths(mfilename('fullpath'));

| Folder | What it holds | Used by |
|---|---|---|
| `python/` | paths, data loaders, signal detection, resampling statistics, the `numbers.csv` and `tests.csv` writers, the figure style | every Python step |
| `matlab/` | the MATLAB versions of the paths, loaders, exclusions and `numbers.csv` writer, the PiYG color map, the 3-D PCA panels | modules 01, 02, 03, 05, 06, A1, A2a, A2b, A2d, A2e |
| `things/` | the SPoSE embedding and the THINGS concept and image tables | most modules |
| `things/provenance/` | the scripts that made the THINGS tables and chose the stimulus sets | no module runs them |
| `embeddings_things/` | deep-network embeddings of the THINGS photographs | modules A3a, A4b (and A5, with M3) |
| `embeddings_isola/` | deep-network embeddings of the Isola et al. (2011) scenes | module A6a |
| `embedding_extraction/` | the scripts that make the embeddings | module A6a, with M4 (the face embeddings) |
| `memtoolbox2/` | the place for the MemToolbox2 Python package (not included) | module A3c |
| `ledger/` | every number of the paper with the module that computes it, and every figure and table with its script | every module, `collect_numbers.py` |

## matlab/

| File | What it does |
|---|---|
| `release_paths.m` | `paths = release_paths(mfilename('fullpath'))`: the release and module folders, found from the script's own location (the MATLAB version of `release_paths.py`); creates the module's output folders. |
| `load_concept_info.m` | `concInfo = load_concept_info('human')` or `('monkey')`: the concept struct of `things/concept_info_<space>.mat`, with `thingsSimRSM = corrcoef(features')` added (modules 05, 06, A1). |
| `load_inversion_trials.m` | `[T, design, studyTrials] = load_inversion_trials()`: the inversion experiment's test-trial table (subjectNumber 1..54 = S001..S054, block, trialNumber, imageName, concept, isCritical, isNew, response, rt, memorabilityScore, participant), the 40-item design table and the study rows (module 03). |
| `load_image_set_trials.m` | `T = load_image_set_trials({'maintained', 'changed'}, concInfo)`: the image-set test-trial table (subject, is_new, image_number, original_index, rt, response, sumSimToStudyItems1 and 2, filename, concept, dataset_index, dataset_name) for any of the three sets in any order; a third argument `false` leaves out the summed similarity (modules 06, A1). |
| `apply_participant_exclusions.m` | `[T, report] = apply_participant_exclusions(T, ...)`: the exclusions described below; for the inversion table pass `'subjectColumn', 'subjectNumber', 'isNewColumn', 'isNew', 'datasetColumn', '', 'cap', 50`. `report.flags` has one row per participant with every rule, and `report.badSubjects` lists the excluded participants (modules 03, 06, A1). |
| `sdt_rates.m` | `sdt = sdt_rates()`: hit and false alarm rates, corrected recognition, log-linear d′ and c (from counts), d′ with the 1/(2N) clamp, criterion c, d_a at a fixed z-ROC slope, and d_a from the six response counts (`daFromRatingCounts`; module A1). |
| `write_numbers_csv.m` | Writes a module's `outputs/numbers.csv` from a struct array of ledger ids and computed values, adds the printed value, location and quote of `ledger/paper_numbers.csv`, and judges each value by the rules of `numbers_table.py` (modules 01, 02, 03, 05, 06). |
| `save_panel.m` | Saves a figure panel as a 300 dpi PNG in `outputs/figures/` and the numbers it plots as a CSV in `outputs/figure_data/` (modules 05, 06). |
| `piyg_colormap.m` | The PiYG color map of the main-text figures: matplotlib's PiYG (ColorBrewer, Apache License 2.0) sampled at 256 points and interpolated linearly to n colors (modules 03, 05, 06). |
| `pcaCoeffsExplained.m`, `applyAxesPca3dView.m`, `stylePca3dAxes.m` | The 3-D PCA panels of Figs. 5b, 6a, 6b and 7a: principal directions and explained variance, camera view, axes and colorbar layout (modules 05, 06). |
| `readPca3dFeatureSpaceView.m`, `pca3d_feature_space_view.csv` | The camera angles of those panels. The function reads the CSV next to itself (its first argument is not used) and stops with an error when the CSV is missing. |

## python/

| File | What it does |
|---|---|
| `release_paths.py` | `module_paths(__file__)`, `release_root()`, `find_module('03')`, `shared_path(...)`, and the switch for inputs that are not included: `regeneration_mode([Input('M1', description, 'external/kramer_trials/...')])` returns `'regenerate'` when every input is present; otherwise it prints `PROVIDED OUTPUT: regenerating needs <input>; see EXTERNAL_INPUTS.md` and returns `'provided'`, and the module uses the saved results in its `data/provided/` folder. |
| `concept_tables.py` | Readers of the tables in `things/`: `load_concept_table()`, `spose_features()`, `load_monkey_features()`, `concept_similarity()` (correlation of the feature rows), `load_things_images(order='image_nr' or 'embedding')`, `load_concept_categories()`. |
| `kramer_public.py` | The public Kramer et al. (2023) data as the tables in `things/` carry them: per-image scores, concept means keyed by concept name (`skip_missing=False` gives NaN for goalpost, as `concept_table.csv` does), the category of each concept, and `verify_downloads()`, which checks those tables against the public files after `python external/download_external.py`. |
| `inversion_loader.py` | The inversion experiment: `load_trials()`, `participant_flags()`, `apply_exclusions()` and `items_table()` (the 40 critical items over the 50 kept participants: image_name, block, memorability_quintile, n_similar_items, kramer_cr, n_old, hits, n_new, false_alarms, hit_rate, fa_rate, cr; sorted by image name). |
| `imageset_loader.py` | The image-set experiments (the Python version of the two MATLAB loaders): `load_trials(sets)` (test trials with the summed similarity to the study lists of the same and of the previous block), `load_study_lists()`, `participant_flags()`, `apply_exclusions()`. |
| `face_loader.py` | The face study: `load_trials()`, `load_participants()`, `recompute_participants()` (the preregistered rule computed from the trials and checked against `face_participants.csv`), `registered_participants(order='load_order' or 'completion')`, `participant_rates()` (per registered participant: hit and false alarm rates of rare and ordinary faces, each against its own kind of lure, and the log-linear d′ and c). |
| `sdt.py` | The signal-detection definitions (rates, corrected recognition, log-linear d′ from rates or from counts, the 1/(2N) clamp, criterion c, d_a at a fixed slope, `da_from_rating_counts`), and the binned signal detection of the THINGS continuous-recognition analyses (`N_BINS`, `DA_SLOPE`, `within_image_bin()`, `cell_counts()`, `sdt()`) that modules A2c and A5 use. |
| `bootstrap.py` | `percentile_bootstrap`, `percentile_interval`, `matlab_prctile`, `permutation_test`, `split_half_reliability`, `spearman_brown`, `fisher_z_interval`, and `seeded_per_resample_mean_bootstrap` (the participant bootstrap of the face study, which seeds resample i with i). The other functions that draw random numbers take a seed, by default 20261013 (`RELEASE_SEED`). |
| `numbers_table.py` | `NumbersTable(script).add(ledger_id, computed_value, mode, note)` and `.write(outputs)`: a module's `numbers.csv`, with the printed value, location and quote read from `ledger/paper_numbers.csv` and the match judged by the item's match type (rules under ledger/ below). |
| `tests_table.py` | `TestsTable(script).add(test_id=..., ...)` and `.write(outputs)`: a module's `tests.csv` in the convention of `statistics/README.md`. |
| `figure_style.py` | The style of the SI Appendix figures and of Fig. 4 (fonts, colors, sizes, direct labels, the statistic on the plot). `save_figure(fig, name, paths['figures'])` writes a 300 dpi PNG (PDF or SVG on request). |

The numbers in the CSV tables of `things/` are written as the shortest decimal that reads back as the same
double. MATLAB `readtable` and Python `float()` read them exactly; pandas needs
`float_precision='round_trip'` (its default parser is off in the last digit for about 80 percent of the
SPoSE feature values). Every loader here passes it, and module code that reads these tables with pandas
should too.

## things/

| File | Contents |
|---|---|
| `spose_embedding_49d_sorted.txt`, `unique_id.txt`, `labels_short49.mat`, `LICENSE_spose.txt` | The 49-dimensional SPoSE embedding of the 1,854 THINGS concepts (Hebart et al. 2020, osf.io/f5rn6, CC BY 4.0), the concept order of its rows (the THINGS order), the feature labels and the license, as distributed. |
| `labels_short49.txt` | The 49 feature labels, one per line (the labels of `labels_short49.mat`). |
| `concept_table.csv` | 1,854 rows in THINGS order: concept_number, concept, category_number and category_label (module 06 counts the categories of its image sets with these), kramer_concept_cr (the concept's mean public memorability score; NaN for goalpost, whose image goalpost_08s.jpg has no score), concept_image_count_as_stored (the number of THINGS images of the concept), weighted_similarity_full_things_human and _monkey (summed similarity over the full THINGS set, measure (a) below), feature_01 to feature_49 (the SPoSE values). |
| `concept_categories.csv` | 1,854 rows in THINGS order: concept_number, concept, category_number and category_label from the public Kramer et al. `Concept_to_category_linking.csv`, joined by concept name. Module A5 (SI Appendix, Section 5) uses this file. |
| `monkey_features.csv` | 1,854 x 49 monkey IT principal-component scores (pc_01 to pc_49), rows in THINGS order: the multi-unit responses of the IT channels of the THINGS Ventral Stream Spiking Dataset (Papale et al. 2025; channels 513 to 832 of monkey F and 769 to 1,024 of monkey N, 576 in all), averaged over the images of each concept in the dataset's training set, then reduced to the first 49 principal components (`provenance/monkeyFeatGen.m`). |
| `monkey_pca_explained.csv` | The 576 eigenvalues of that PCA, the percent of variance of each component and the cumulative percent: 49 components keep 84.60 percent. |
| `concept_info_human.mat`, `concept_info_monkey.mat` | The two concept structs that the MATLAB scripts read through `matlab/load_concept_info.m` (MATLAB v7). |
| `things_images.csv` | 26,107 rows in the public image-number order: image_nr, filename, rel_path, concept, concept_number_things_order, concept_number_ascii, kramer_cr (the public per-image score; NaN for goalpost_08s.jpg), embedding_row (the 0-based row of the image in every file of `embeddings_things/`; those files are sorted by file name, which differs from image_nr order in 29 rows). |
| `things_image_mapping.csv` | The image-number table of the THINGS trial analyses (modules A3a and A5): new_fname (the image URL of the Kramer et al. experiment, `https://martin-hebart.com/turk/images500/<image_nr, 5 digits>.jpg`), orig_fname (the public file_path), concept_nr (concepts numbered in the order they first appear in the public score file, which is alphabetical), exemplar_nr, image_nr. `provenance/rebuild_things_image_mapping.py` writes it from the public score file. |
| `provenance/` | The scripts that made these tables and the scripts that chose the stimulus sets (see its README). |

## embeddings_things/, embeddings_isola/, embedding_extraction/

The four late-layer THINGS spaces that module A3a reads (SI Appendix, Fig. S8), the 14 THINGS
concept-centroid files that module A4b reads (SI Appendix, Section 4) and the 14 Isola scene spaces that
module A6a reads (SI Appendix, Fig. S15) are included, and `embedding_extraction/` holds the scripts that
make them. The other ten per-image THINGS spaces are not included (EXTERNAL_INPUTS.md, M3); module A5
needs them, with the Kramer et al. trial file (M1), only for the deep-network result of SI Appendix,
Section 5. Each folder has its own README.

## memtoolbox2/

The place for the MemToolbox2 Python package, with which module A3c fits the preregistered
unequal-variance model of the face study (EXTERNAL_INPUTS.md, M5). The package is not included; without
it, module A3c uses the fitted values saved in its `data/provided/` folder (see
`memtoolbox2/README.md`).

## ledger/

`paper_numbers.csv` lists every number and result of the paper and the SI Appendix that the code
computes or checks (648 items), one row each, with the columns:

| Column | Contents |
|---|---|
| id | the item's id (L1-003, ADD-002, ...); the modules' `numbers.csv` use these ids |
| location | the section of the paper (or of the SI Appendix) or the figure caption that prints the number |
| quote | the words that state the number |
| paper_value | the printed values, joined by "; "; for a figure panel, the values it shows; empty when the item states no number |
| type | statistic, design, parameter, count, mean, percentage, ci, exclusion, figure_panel or other |
| module, module_folder | the module that computes the item |
| release_mode | rerun (computed from the files in the repository), provided (read from the module's `data/provided/` folder because an input is not included), rerun_and_provided (both), restatement (a number that repeats another item, named in same_as) or documented (a statement or design fact checked against the data or the code) |
| same_as | a related item at another place in the paper, such as the same number, result or experiment (for a restatement, the item it repeats) |
| value_kind | numeric, text or none |
| match_type | how a computed value is judged (below) |

`numbers_table.py` and `write_numbers_csv.m` judge each computed value by its match type:

| match_type | match is yes when |
|---|---|
| round | the computed value rounds to the printed one: it lies within half a unit of the printed last digit (0.5 percentage points for 27%, 0.005 for -0.50) |
| exact, count | the computed value equals the printed one |
| less_than | the computed value is below the bound of a printed "< x" (every printed part that starts with "<" is judged this way) |
| range | both ends of a printed "a to b" pass as round |
| documented | not judged: the match column says documented, and the computed value and the note say what the code finds |

A printed value with several parts ("-0.50; < 0.01") is compared part by part with a computed list of the
same length.

`figure_map.csv` lists the figures and tables of the paper and the SI Appendix, one row for each group of
panels or table that one script makes with one mode: figure, panels, module, module_folder, script,
output_folder and mode (rerun, provided or rerun_and_provided; the last two followed in parentheses by the
EXTERNAL_INPUTS.md ids of the inputs that are not included), as a full run (`python run_all.py`) makes them.

## The participant data (in the module data folders)

The loaders read the participant data of the three experiments from the module data folders. Only the
sessions the paper reports before exclusions are included, and they are de-identified: there are no
SONA, Prolific or session ids, no timestamps, browser strings, screen sizes, interaction logs, device
checks, demographic answers or free text, and the maps to the original ids are not included.

| Folder | Files | Participants and order |
|---|---|---|
| `03_inversion_experiment_fig3/data/` | `inversion_trials.csv` (participant, block, phase, trial_in_block, image_name, concept, condition, is_new, response_1to6, rt_ms; 320 study and 80 test rows per session), `inversion_design.csv` (the 40 critical items: block, image_name, concept, memorability_quintile, n_similar_items, kramer_cr), `inversion_deployed_lists.csv` (the lists of the 8 blocks: 40 study and 10 test rows each, with the critical item each similar item belongs to; 8, 4, 2 and 1 similar items for the critical items of quintiles 5, 4, 3 and 2, then 20 additional non-similar exemplars; a similar item need not share the concept, for example tee, stake and lamppost for pole. Which 2 or 3 critical items a participant studied was drawn per participant, so the old and new labels of the critical and maximally dissimilar rows in these lists are one example draw; each participant's own draw is in `inversion_trials.csv`) | S001..S054: the 54 sessions, numbered in the alphabetical order of their session ids (the order in which the analysis loads them). 4 are excluded (S004, S005, S019, S026) and 50 kept. |
| `06_image_set_experiments_fig6_fig7/data/` | `maintained_trials.csv`, `changed_trials.csv`, `reduced_trials.csv` (participant, block 1 to 16, phase, trial_in_block, image_filename, concept, things_image_index = the public image number, image_number (the experiment's own index, not used), is_new, response_1to6, rt_ms; 640 study and 640 test rows per session) | maintained M001..M205, changed C001..C195, reduced R001..R202: the 602 sessions, numbered within each set in the alphabetical order of their session ids (the order in which the analyses load them, which sets the cap of 150 and the rng(42) resampling). The ids carry the set's letter so that they are unique across sets, because the MATLAB scripts pool two sets in one table before excluding. 55, 45 and 52 participants are excluded and 150 per set kept (the cap of 150 removes nobody). |
| `04_rare_feature_faces_fig4/data/` | `face_trials.csv` (participant, completion_rank, load_order, phase, trial_in_phase, image, kind, condition, feature, cell, is_repeat, pressed, oneback_correct, test_type, is_old, said_old, confidence_1to4, rt_ms, rt_confidence_ms, studied_image, role_swapped; 56 study and 36 recognition rows per session), `face_participants.csv` (participant, completion_rank, load_order, oneback_hits, oneback_false_presses, passes_rule, registered), `face_design.json` (the task constants without the completion code and the experimenter's name, the stimulus manifest of the synthetic faces, and counted design facts) | S001..S111 in completion order: the 111 sessions run. The first 100 sessions in completion order that pass the preregistered rule (at least 4 of the 6 one-back repeats detected, at most 5 presses on non-repeats) form the registered sample; 11 fail (8 with too many presses and 4 with too few repeats detected, 1 of them both). `load_order` is the order in which the registered analysis loads the sessions. Its seeded participant bootstrap depends on that order, so `face_loader.participant_rates()` returns the participants in load_order by default. |

**Exclusions** (Methods, Experimental Procedures and Analysis; `matlab/apply_participant_exclusions.m` and
the `apply_exclusions()` of the two Python loaders). A participant is excluded when one response key was
used on more than 80 percent of the test trials, when more than 20 percent of the test trials were faster
than 200 ms, when more than 20 percent were slower than 7,500 ms, or when hit rate minus false alarm rate
(old = response 1 to 3) was below 0.05. Trials faster than 200 ms or slower than 10,000 ms are then
dropped, and the first 150 remaining participants of each image set in sorted id order are kept (50 in the
inversion experiment); with these data the cap removes nobody. The face study uses its own preregistered
rule (above).

## Notes every module should know

**Concept order.** Every concept table here (`concept_table.csv`, `concept_categories.csv`,
`monkey_features.csv`, the two .mat files) is in the THINGS order of `unique_id.txt`, the row order of the
SPoSE embedding. The Kramer et al. files, `things_image_mapping.csv` (`concept_nr`) and the
`concept_number_ascii` column of `things_images.csv` use the alphabetical (ASCII) order of the concept
names, and the concept-centroid files of `embeddings_things/` are sorted by name. Join tables by concept
name, never by position.

**Three summed-similarity measures.** Each passes a similarity through the kernel
normpdf(similarity, 1, 2), where the similarity of two concepts is the correlation of their feature rows.
(a) Full THINGS (`weighted_similarity_full_things_*`, Fig. 5): for concept i, the sum over all other
concepts j of kernel(i, j) x count(j) / total count, with the image counts of `concept_table.csv` (the
weights of the other concepts are not renormalized after leaving i out). (b) Within an image set (Figs. 6
and 7): the same with the set's image counts, one image of concept i removed, and the weights renormalized
to sum to 1 (concept i itself keeps its remaining images). (c) Per study list (the trial-level
`sumSimToStudyItems`): the mean kernel similarity of the tested concept to the 40 concepts studied in the
same block (or in the block before), with one occurrence of the tested concept removed.
