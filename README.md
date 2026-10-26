# Memorability as an Emergent Property: code and data

Code and data for

> Simpson, D. D., Wixted, J. T., & Brady, T. F. (2026). *Memorability as an emergent property: How
> representational structure creates apparent intrinsic effects* [Manuscript submitted for publication].
> Department of Psychology, University of California, San Diego.

The repository holds the data and code behind the numbers, data figures and tables of the paper and its SI
Appendix (Sections 1 to 6, Figs. S1 to S16, Tables S1 and S2): the REM simulations (Figs. 1 and 2; SI Appendix,
Section 2), the memorability inversion experiment (Fig. 3), the preregistered rare-feature face study (Fig. 4;
SI Appendix, Section 3), the reanalysis of the THINGS memorability data (Fig. 5; SI Appendix, Section 5), the
three image-set experiments (Figs. 6 and 7; SI Appendix, Sections 1 and 3), a single crowding measure applied
to every dataset (SI Appendix, Section 4) and the analyses of faces, scenes and refit embeddings (SI Appendix,
Section 6). For every test the paper reports, the code also computes the statistic with its degrees of
freedom, effect size and 95% interval where these apply. Two tables connect the paper to the code.
`reported_numbers.csv` lists each number of the paper next to the value the code computes, and
`statistics/all_tests.csv` lists every statistical test. A few inputs cannot be included in the repository
(EXTERNAL_INPUTS.md); the results that need them are computed from saved results until the input is added.
Every path is relative to the repository, so the code runs from wherever it is cloned.

## Quick start

The code needs Python 3.11 or 3.12 with the packages pinned in `requirements.txt`, and MATLAB (R2023b tested)
with the Statistics and Machine Learning Toolbox for the MATLAB steps (modules 01, 02, 03, 05 and 06, and
parts of A1, A2a, A2b, A2d and A2e). `environment.md` gives the tested versions, the memory each part needs and
the settings the run scripts make. From the repository root (`python3.11` works in place of `python3.12`):

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python run_all.py

`run_all.py` runs the 24 modules one after the other, in an order that respects what each module reads from
the others, and then the two collectors that write `reported_numbers.csv` and `statistics/all_tests.csv`. For
each module it prints the mode, the runtime and a summary of the module's `outputs/numbers.csv`, and it ends
with exit code 1 if any step failed. It finds MATLAB through the environment variable `MATLAB` (the full path
of the `matlab` program or of the MATLAB folder), else `matlab` on the PATH, else a standard installation
folder. The MATLAB commands in this file and the other documentation start `matlab` by name; where it is not
on the PATH (as after a standard installation on a Mac), use its full path, for example
`/Applications/MATLAB_R2023b.app/bin/matlab`.

| Command | What it runs |
|---|---|
| `python run_all.py` | every module, then the two collectors |
| `python run_all.py --skip-long` | everything except the longest steps: module 02's simulation (Fig. 2 and its numbers then come from the saved run in `data/paper_run/`), the feature-count sweep of A2a, the 83-setting Bylinskii run of A2f and the two model fits of A3a (the fits in its `outputs/` are used). With the external inputs in place, A5 also leaves out its deep-network step, A3c its 243-setting refits, and A2c keeps its saved model run |
| `python run_all.py --module 03 A3b` | only the modules named, then the two collectors |
| `python run_all.py --skip-matlab` | everything that does not need MATLAB; the MATLAB steps are left out and their outputs in the repository are used |
| `python run_all.py --provided-only` | everything, with modules A2c, A3c, A4a and A5 using their saved results even when the external input is in place, and A2f running no simulation (its numbers come from its outputs or its saved run) |
| `python run_all.py --list` | nothing: it prints the modules and their commands |

On the tested laptop (Apple M2 Pro, 16 GB), `python run_all.py` takes about 1 hour 40 minutes,
`python run_all.py --skip-long` about 1 hour, `matlab -batch "run_all"` about 35 minutes and
`matlab -batch "run_all(true)"` about 15 minutes.

Each module also runs on its own, with the commands its README gives, for example

    matlab -batch "restoredefaultpath; cd('03_inversion_experiment_fig3/code'); run_module"
    python 03_inversion_experiment_fig3/code/stats_fig3.py
    python A3b_hits_and_false_alarms_figS9/code/run_module.py

after which the two tables are rebuilt with

    python collect_numbers.py
    python statistics/collect_all_tests.py

For readers who use MATLAB only, `matlab -batch "run_all"` (`run_all.m`) runs the MATLAB analyses: modules 01,
02, 03, 05 and 06 and the MATLAB steps of A1, A2a, A2b, A2d and A2e. It draws no SI Appendix figure and
computes none of the statistics, which are written in Python. `run_all(true)` leaves out module 02's
simulation.

## Repository layout

```
README.md             this file
EXTERNAL_INPUTS.md    the inputs that are not in the repository, and how to add them (M1 to M9)
LICENSE               license of the code (MIT)
LICENSE-DATA.md       license of the data, outputs and documentation (CC BY 4.0), and third-party files
CITATION.md           how to cite the paper and the data and models it uses
environment.md        software, versions, memory and runtimes
requirements.txt      Python packages
run_all.py            runs every module, then the two collectors
run_all.m             the MATLAB analyses only
collect_numbers.py    builds reported_numbers.csv from the modules' outputs/numbers.csv
reported_numbers.csv  every number of the paper next to the value the code computes
statistics/           all_tests.csv (every statistical test), participant_flow.csv and their collector
shared/               code and data that several modules use (shared/README.md)
external/             folders for the inputs of EXTERNAL_INPUTS.md, and a download script
01_... to A6b_...     one folder per figure, table or section of the paper (below)
```

| Module folder | Figures and tables | Section of the paper |
|---|---|---|
| `01_rem_simulation_fig1/` | Fig. 1 | Simulations of Memorability as an Emergent Property; Methods, Basic REM Simulation of Memorability |
| `02_bylinskii_simulation_fig2/` | Fig. 2 | Stable Memorability Across Contexts Does Not Require Intrinsic Memorability; Methods, Bylinskii et al. (2015) Context Manipulation Simulation |
| `03_inversion_experiment_fig3/` | Fig. 3 | Inverting Memorability Through Direct Manipulation of Distinctiveness; Methods, Behavioral Validation Experiments (Inversion Experiment) |
| `04_rare_feature_faces_fig4/` | Fig. 4a, 4b; SI Appendix, Fig. S11a | A Rare Feature's Advantage Depends on What the Face Is Tested Against; Methods, Behavioral Validation Experiments (Rare-Feature Face Experiment) |
| `04b_face_same_item/` | none | A Rare Feature's Advantage Depends on What the Face Is Tested Against (each face scored against itself) |
| `05_things_reanalysis_fig5/` | Fig. 5 | The Similarity Structure of an Image Set Determines What Will Be Memorable (the THINGS reanalysis); Methods, Representational Similarity Analysis |
| `06_image_set_experiments_fig6_fig7/` | Figs. 6 and 7 | The Similarity Structure of an Image Set Determines What Will Be Memorable; Reducing Variability In Distinctiveness Reduces Variability in Memorability; Methods, Image Set Structure Manipulations; Methods, Experimental Procedures and Analysis |
| `06b_reduced_set_variance_fig7/` | none | Reducing Variability In Distinctiveness Reduces Variability in Memorability and the Fig. 7 caption (the variance of memorability over the concepts in both sets, and the feature dose-response); Methods, Experimental Procedures and Analysis |
| `A1_measuring_memory_figS1/` | SI Appendix, Fig. S1 | SI Appendix, Section 1 (Measuring Memory) |
| `A2a_rem_parameter_sweep_figS2/` | SI Appendix, Fig. S2 | SI Appendix, Section 2, REM Parameters |
| `A2b_category_structure_criteria_figS3/` | SI Appendix, Fig. S3 | SI Appendix, Section 2, Category Structure and Decision Criteria |
| `A2c_continuous_recognition_figS4/` | SI Appendix, Fig. S4 | SI Appendix, Section 2, Continuous Recognition |
| `A2d_list_context_figS5/` | SI Appendix, Fig. S5 | SI Appendix, Section 2, List Context |
| `A2e_measured_representations_figS6/` | SI Appendix, Fig. S6 | SI Appendix, Section 2, Measured Representations |
| `A2f_hybrid_exemplar_model_figS7/` | SI Appendix, Fig. S7 | SI Appendix, Section 2, A Hybrid-Similarity Exemplar Model |
| `A3a_model_through_own_lists_figS8/` | SI Appendix, Fig. S8 | SI Appendix, Section 3, The Model Predicts Memory From Participants' Own Lists |
| `A3b_hits_and_false_alarms_figS9/` | SI Appendix, Fig. S9 | SI Appendix, Section 3, Similar Items Change Memory Mainly Through False Alarms; SI Appendix, Section 1 (the THINGS data of Fig. S9c) |
| `A3c_face_study_model_and_sdt_figS10_S11/` | Fig. 4c; SI Appendix, Figs. S10 and S11b | SI Appendix, Section 3, The Model Predicted Each Preregistered Face Result; The Preregistered Discriminability Analysis of the Face Study |
| `A3d_exclusions_tableS1/` | SI Appendix, Table S1 | SI Appendix, Section 3, The Image Set Results Do Not Depend on the Exclusions |
| `A4a_crowding_across_scales_figS12_tableS2/` | SI Appendix, Fig. S12 and Table S2 | SI Appendix, Section 4 (One Measure of Distinctiveness Across Analyses) |
| `A4b_things_concepts/` | none | SI Appendix, Section 4, the THINGS concepts in the deep-network spaces (paragraph "Corrected recognition can reverse the sign") |
| `A5_things_context_effects_figS13_S14/` | SI Appendix, Figs. S13 and S14 | SI Appendix, Section 5 (Context Effects in the Original THINGS Data) |
| `A6a_faces_and_scenes_figS15/` | SI Appendix, Fig. S15 | SI Appendix, Section 6, Faces and Scenes |
| `A6b_embedding_size_figS16/` | SI Appendix, Fig. S16 | SI Appendix, Section 6, Embedding Size |

Where a section repeats a number computed for another section (for example in the General Discussion),
`reported_numbers.csv` names the module and script that compute it.

Every module folder has the same layout. `README.md` says what the module reproduces, how to run it, its
inputs, what the code does (seeds, settings, exclusions) and its outputs. `data/` holds the inputs the module
ships; `data/paper_run/` holds the saved run whose output the paper reports, where a module compares a new run
with it, and `data/provided/` holds the saved results the module uses when an input of EXTERNAL_INPUTS.md is
absent. `code/` holds the scripts, with `run_module.py` or `run_module.m` as the entry point. `outputs/` holds
what the code writes (below); a `logs/` folder with the screen output of each step is added when the module
runs. Modules 04b, 06b, A3a and A3d have no `data/` folder, because they read the data of modules 03, 04 and 06
(and A3a also the THINGS embeddings in `shared/`).

## Figures and tables

Each data figure is drawn by its module's code from the module's outputs, into the module's `outputs/figures/`
(PNG); the numbers each panel plots are in `outputs/figure_data/` (CSV). `shared/ledger/figure_map.csv` holds
the same map, with the two SI Appendix tables and without the file names, as a CSV table.

| Figure | Panels | Module | Script (in the module's `code/`) | File (in the module's `outputs/figures/`) | Plotted values |
|---|---|---|---|---|---|
| Fig. 1 | a to d | 01 | `fig1_makeFigure1.m` | `fig1_rem_simulation.png` | computed |
| Fig. 2 | a, b | 02 | `bylinskii_makeFigure2.m` | `fig2a_memorability_within_vs_across.png`, `fig2b_summed_similarity_within_vs_across.png` | computed (with `--skip-long`, the saved run) |
| Fig. 3 | d, e (a to c show the design) | 03 | `inversion_analysis.m` | `fig3d_rank_order_lines.png`, `fig3e_memorability_vs_cr.png` | computed |
| Fig. 4 | a, b | 04 | `make_figure4.py` | `Fig4ab_rare_feature_faces.png` | computed; panel a shows the five synthetic faces in `data/fig4a_faces/` |
| Fig. 4 | c | A3c | `make_figureS11.py` | `Fig4c_confidence_roc.png` | points computed; fitted curves from saved results (M5) |
| Fig. 5 | a to d | 05 | `things_reanalysis.m` | one file per panel, `fig5a_ranked_feature_memorability.png` to `fig5d_monkey_features_similarity_vs_memorability.png` | computed |
| Fig. 6 | a to h | 06 | `maintained_vs_changed.m` | one file per panel, `fig6a_pca_maintained_set.png` to `fig6h_monkey_summed_similarity_vs_changed_profile.png` | computed |
| Fig. 7 | a | 06 | `make_figure7a_pca.m` | `fig7a_pca_reduced_set.png` | computed |
| Fig. 7 | b to d | 06 | `maintained_vs_reduced_variance.m` | `fig7b_reduced_set_histogram.png`, `fig7c_maintained_set_histogram.png`, `fig7d_summed_variance_explained.png` | computed |
| SI Appendix, Fig. S1 | | A1 | `make_figureS1.py` | `FigS1_criterion_dprime_da.png` | computed |
| SI Appendix, Fig. S2 | a, b | A2a | `make_figureS2.py` | `FigS2_rem_parameter_sweep.png` | computed |
| SI Appendix, Fig. S3 | | A2b | `make_figureS3.py` | `FigS3_bylinskii_robustness.png` | computed |
| SI Appendix, Fig. S4 | a, b | A2c | `make_figureS4.py` | `FigS4_continuous_recognition.png` | saved results (M1, M6) |
| SI Appendix, Fig. S5 | a, b | A2d | `make_figureS5.py` | `FigS5_list_context.png` | computed |
| SI Appendix, Fig. S6 | | A2e | `make_figureS6.py` | `FigS6_continuous_feature_rem.png` | computed |
| SI Appendix, Fig. S7 | a to c | A2f | `make_figureS7.py` | `FigS7_hybrid_exemplar_model.png` | computed |
| SI Appendix, Fig. S8 | a to c | A3a | `make_figureS8.py` | `FigS8_model_through_experiments.png` | computed |
| SI Appendix, Fig. S9 | a, b | A3b | `make_figureS9.py` | `FigS9_hits_false_alarms.png` | computed |
| SI Appendix, Fig. S9 | c | A3b | `make_figureS9.py` | `FigS9_hits_false_alarms.png` | saved results (M1) |
| SI Appendix, Fig. S10 | a | A3c | `make_figureS10.py` | `FigS10_rem_face_design.png` | computed |
| SI Appendix, Fig. S10 | b | A3c | `make_figureS10.py` | `FigS10_rem_face_design.png` | saved results (M5) |
| SI Appendix, Fig. S11 | a | 04 | `make_figure4.py` | `FigS11a_memorability_by_condition.png` | computed |
| SI Appendix, Fig. S11 | b | A3c | `make_figureS11.py` | `FigS11b_lure_set_and_feature_effects.png` | saved results (M5) |
| SI Appendix, Fig. S12 | a to c | A4a | `make_figureS12.py` | `FigS12_crowding_across_scales.png` | computed; the THINGS lines from saved results (M1) |
| SI Appendix, Fig. S13 | a to c | A5 | `make_figureS13.py` | `FigS13_things_trial_level_context.png` | saved results (M1) |
| SI Appendix, Fig. S14 | a, b | A5 | `make_figureS14.py` | `FigS14_kramer_quartile.png` | saved results (M1) |
| SI Appendix, Fig. S15 | a, b | A6a | `make_figureS15.py` | `FigS15_faces_scenes_generality.png` | scenes computed; faces from saved results (M4) |
| SI Appendix, Fig. S16 | | A6b | `make_figureS16.py` | `FigS16_spose_dimensionality.png` | computed from the saved refits in `data/refits/` |

SI Appendix, Tables S1 and S2 are CSV files in the `outputs/` folder of their module. Table S1 is
`A3d_exclusions_tableS1/outputs/tableS1.csv` (written by `make_tableS1.py`; computed), and Table S2 is
`A4a_crowding_across_scales_figS12_tableS2/outputs/table_s2.csv` (written by `locality_analyses.py`; computed,
the THINGS rows from saved results, M1).

"Saved results (Mn)" means that the plotted values come from saved results in the module's `data/provided/`
(for the data of SI Appendix, Fig. S4, module A5's), because input Mn of EXTERNAL_INPUTS.md is not in the
repository; EXTERNAL_INPUTS.md says what each input lets the code compute. The THINGS photographs that some
panels of the paper show as thumbnails are not in the repository (their license does not allow
redistribution); these panels show markers and concept names instead (EXTERNAL_INPUTS.md, M7).

## Outputs and the two tables

Each module writes to its `outputs/` folder: `numbers.csv` (every number of the paper the module computes,
with the columns of `reported_numbers.csv` below, `ledger_id` in place of `id` and without `same_as`),
`tests.csv` (its statistical tests, with the columns of `statistics/all_tests.csv`), `figures/` and
`figure_data/`, and further tables its README describes. The repository includes these outputs,
`reported_numbers.csv`, `statistics/all_tests.csv` and `statistics/participant_flow.csv`, so that they can
be read without running anything, and running the code writes them again.

**`reported_numbers.csv`** has one row for each number or result of the paper (the rows of
`shared/ledger/paper_numbers.csv`):

| Column | Content |
|---|---|
| `id` | the row's identifier in `shared/ledger/paper_numbers.csv` |
| `location` | where the paper reports it: the title of the section, or the figure caption |
| `quote` | the sentence or caption text that contains it |
| `paper_value` | the value as printed in the paper (several values joined by "; ") |
| `computed_value` | the value the code computes, or reads from saved results (see `mode`) |
| `match` | whether the two agree (below) |
| `match_type` | how the two are compared: `round` (the computed value is within half a unit of the last printed digit, so -0.50 accepts -0.505 to -0.495), `exact` and `count` (equal), `less_than` (below a printed bound such as p < 0.001), `range` (both ends of a printed "a to b"), `documented` (no single printed number) |
| `script` | the script that computes it, as a path from the repository root |
| `mode` | how the value is obtained (below) |
| `same_as` | the `id` of a related row at another place in the paper, such as the same number, result or experiment (for a `restatement`, the row it repeats) |
| `note` | what the computed value is |

`match` is `yes` when the computed value agrees with the printed value under its `match_type` and `no` when it
does not. `documented` marks a statement, design fact or figure panel without a single printed number; its
`computed_value` and `note` give what the code finds. `python collect_numbers.py` prints a summary of the
column and lists every row whose match is `no`.

The 95% confidence intervals the paper prints are part of the printed values that `reported_numbers.csv`
compares, and `statistics/all_tests.csv` gives each with its test and method (columns `ci_low`, `ci_high` and
`ci_method`).

| `mode` | Meaning |
|---|---|
| `rerun` | computed by the code from the data in the repository, in this run or, for a step the run leaves out (`--skip-long`, `--skip-matlab`, `--provided-only`), in the earlier full run whose results are in the module's `outputs/` |
| `saved_run` | read from the saved run in the module's `data/paper_run/`, which a full run of the simulation reproduces: in every run the correlations of Figs. 1 and 2 with their intervals and p values, and with `--skip-long` also the other numbers of module 02 |
| `provided` | computed from the saved results in the module's `data/provided/`, because an input of EXTERNAL_INPUTS.md is absent |
| `rerun_and_provided` | a value with parts of both kinds (for example SI Appendix, Fig. S15: scenes computed, faces from saved results) |
| `rerun_and_saved_run` | a value with parts computed by this run and parts read from the saved run in `data/paper_run/` (module A2f with `--skip-long` when its `outputs/` holds no full run) |
| `restatement` | a number the paper reports again in another place (for example the General Discussion), computed by the same code as its first place |
| `documented` | a design fact or the content of a figure panel, read from the module's data or drawn by its code |

**`statistics/all_tests.csv`** has one row per statistic, several per test: the test id (T01 to T63 and
T19b; there is no T23), the ids of the numbers the test concerns, the location, a description of the test, the
statistic and its name, n and its unit, df, p and its type, the effect size and its name, the 95% interval
with its method, the resampling unit, the number of resamples and the seed, and the script.
`statistics/README.md` lists the tests and gives the conventions: which unit is resampled for each kind of
data, how many resamples, which tests are one-sided, and how values are rounded.
**`statistics/participant_flow.csv`** gives, for each experiment, the sessions run, the exclusions by rule and
the participants kept.

## The data

**Participants.** The trial tables of the three experiments hold every session the paper counts, the excluded
ones included. Each participant has a session code; the tables hold no recruitment ids, dates or times, browser
or screen information, demographic answers or free text. The code applies the exclusion rules of the Methods
(shared loaders `shared/matlab/apply_participant_exclusions.m` and `shared/python/`).

| Experiment | Folder | Sessions | Codes | Exclusions (applied by the code) |
|---|---|---|---|---|
| Memorability inversion (Fig. 3) | `03_inversion_experiment_fig3/data/` | 54 | S001 to S054 | 4 excluded under the rules of the Methods; 50 kept |
| Rare-feature face study (Fig. 4) | `04_rare_feature_faces_fig4/data/` | 111 | S001 to S111, in order of completion | the preregistered rule: 11 sessions fail it, and the first 100 passing sessions in order of completion form the sample |
| Image sets (Figs. 6 and 7) | `06_image_set_experiments_fig6_fig7/data/` | 602 (205 maintained, 195 changed, 202 reduced) | M001 to M205, C001 to C195, R001 to R202 | 152 excluded (55, 45 and 52); 150 kept per set |

The codes of the inversion and image-set experiments follow the order in which the analyses read the
sessions, and the seeded resampling of participants depends on that order; the face study keeps the order of
its registered analysis in the column `load_order`. The folders also hold the design of each experiment: the
deployed lists and design table of the inversion experiment, the three sampled image sets, and the face study's
stimulus manifest and preregistration (AsPredicted #311640).

**THINGS.** The THINGS analyses use the identities of concepts and images and their published scores, not the
photographs. `shared/things/` holds the SPoSE embedding of the 1,854 THINGS concepts (Hebart et al., 2020),
the concept and image tables with the public per-image memorability scores and concept categories of Kramer et
al. (2023), the monkey IT component scores computed from the THINGS Ventral Stream Spiking Dataset (Papale et
al., 2025), and in `provenance/` the scripts that built these tables. `python external/download_external.py`
downloads the two public files of Kramer et al. and checks the tables against them.

**Deep-network embeddings.** `shared/embeddings_things/` holds the THINGS image embeddings the analyses read:
the late layers of AlexNet, VGG16-BN, ResNet18 and CLIP ViT-B/32 (26,107 photographs x 512 values each; SI
Appendix, Fig. S8) and, in `concept_centroids/`, the concept averages of all 14 spaces (seven networks, two
layers each; the THINGS concepts of SI Appendix, Section 4). The other ten per-image THINGS spaces are an
external input (M3). `shared/embeddings_isola/` holds the 14 spaces of the 10,442 scene images of Isola et al.
(2011), computed from the public images. `shared/embedding_extraction/` holds the extraction scripts and lists
the networks, layers, weights and preprocessing. Module A6a compares its scene values with the saved run in
`A6a_faces_and_scenes_figS15/data/paper_run_scenes/`, the output the paper reports; they agree at the printed
precision.

**Simulations.** The simulations generate their own items and run with fixed seeds. Modules 01, 02, A2a, A2b,
A2d, A2e, A2f and A3c run them again and compare the new run with the saved run in `data/paper_run/` (module
01, for example, writes the comparison to `01_rem_simulation_fig1/outputs/rerun_vs_paper_run.csv`).

## Summed similarity and crowding

The THINGS analyses (Figs. 5 to 7 and the SI Appendix) measure how crowded a concept is by its summed
similarity. The similarity of two concepts is the correlation of their 49 feature values (SPoSE or monkey IT),
each similarity passes through the kernel normpdf(similarity, 1, 2), and the kernel values are weighted in one
of three ways:

1. Over the full THINGS database (Fig. 5): for concept i, the sum over all other concepts j of kernel(i, j)
   times the number of images of j, divided by the 26,107 images.
2. Within an image set (Figs. 6 and 7; the feature level of SI Appendix, Table S1): the same with the image
   counts of the set, one image of concept i removed and the weights renormalized to sum to 1.
3. Per study list (the concept level of SI Appendix, Table S1, and the trial tables of the shared loaders): the
   mean kernel similarity of the tested concept to the 40 concepts studied in the same block, with the tested
   concept removed once when it was studied there.

SI Appendix, Section 4 applies one crowding measure to every dataset: the mean of exp(-d/τ) over the other
items of a reference set, weighted by the number of images of each concept in THINGS, the number of slots each
concept has in an image set, or equally for simulated items, at τ = 0.05, 0.1, 0.2, 0.4 and 1 and in the
all-items limit. Here d is 1 minus the Pearson correlation of two concepts' SPoSE values, or the Euclidean
distance between two simulated items divided by its median (module A4a README).

The REM simulations, whose items are vectors of feature values, measure crowding by Euclidean distance in
feature space. The r = -0.56 of the Fig. 1 caption uses 1 / mean distance (Pearson r), the x axis of Fig. 1b is
the sum over the other items of 1 / distance, and Fig. 2 and the parameter sweeps of SI Appendix, Section 2
use the negated mean distance (Spearman ρ); module 01 computes these measures side by side
(`01_rem_simulation_fig1/outputs/summed_similarity_measures.csv`). Figs. 1 and 2 come from the REM code of
modules 01 and 02. The parameter sweep of module A2a gives SI Appendix, Fig. S2, the ρ = -0.52 at REM's
standard parameter values in the Methods, and the REM ρ of -0.61 and d′ of 1.61 at the parameter values of
Fig. 1 in SI Appendix, Section 2, A Hybrid-Similarity Exemplar Model (module A2a README). It is a separate
implementation that generates its items and draws its random numbers differently, and it counts the new value
of a wrongly copied feature from 1 (module 01 counts it from 0 and does not store a 0).

## Inputs not included

Some inputs cannot be part of the repository, because they are not public, because their license does not
allow redistribution, or because they are large and few results need them. EXTERNAL_INPUTS.md describes each
one, who holds it, where to put it and the commands that use it, and `external/README.md` gives the download
and request steps.

| Id | Input | Held by | What needs it |
|---|---|---|---|
| M1 | THINGS trial file of Kramer et al. (2023) | the original authors, on request | SI Appendix, Section 5 (Figs. S13 and S14); the THINGS results of SI Appendix, Section 4 (Fig. S12, Table S2); SI Appendix, Figs. S4 (data side) and S9c |
| M2 | per-image THINGS count table | not public | SI Appendix, Section 4 (THINGS concepts, memorability side); the original THINGS hit and false alarm rates of the more and less memorable critical items of Fig. 3 (module A3b) |
| M3 | ten THINGS deep-network spaces | computed from the THINGS images (M7) | the deep-network result of SI Appendix, Section 5 (with M1) |
| M4 | 10k US Adult Faces Database (Bainbridge et al., 2013) | the original authors, on request | the faces half of SI Appendix, Fig. S15 |
| M5 | MemToolbox2 (Python) | not public | refitting the face study's d_a analysis (Fig. 4c; SI Appendix, Section 3) |
| M6 | trial structures of 400 THINGS sessions | built from M1 | the model side of SI Appendix, Fig. S4 |
| M7 | THINGS images | public download, after a license agreement | no reported number (thumbnails in Fig. 3e; extracting the THINGS embeddings) |
| M8 | scene memorability package of Isola et al. (2011) | public download | no reported number (extracting the scene embeddings; the scene participant matrices) |
| M9 | SPoSE triplets | public download | no reported number (refitting the embeddings of SI Appendix, Fig. S16) |

Every module runs without these inputs. Where a result needs one of them, the module computes the reported
numbers from the saved results of that analysis in its `data/provided/` folder (`data/provided_faces/` in
module A6a), prints

    PROVIDED OUTPUT: regenerating needs <input>; see EXTERNAL_INPUTS.md

and marks those numbers `provided` in the `mode` column. When the input is put where EXTERNAL_INPUTS.md says,
the same command computes the results from it.

## Reproducibility

Every random computation uses a fixed seed (MATLAB `rng(seed, 'twister')`, Python
`numpy.random.default_rng(seed)`, scikit-learn `random_state=seed` and PyTorch `torch.manual_seed(seed)`),
and the code keeps the order of the random draws behind each reported number, so a run on the same machine
gives the same numbers every time. Run the modules in the order of
`run_all.py`, because some modules read files that earlier modules write (for example, module 06b reads
outputs of module 06, and A4a outputs of modules 06, A2e and A2f and, with M1, the trial table that A5
writes). The simulations of modules A2a, A2b, A2d and A2e draw all their cells from one random stream in file
order, so they run serially, never in a parallel pool, and a single cell gives its reported values only within
the whole simulation. The module READMEs give each seed. The last digits of floating-point results can differ
between machines and numerical libraries (for example in p values far below any printed threshold); on the
tested setups these differences change no printed number, and the comparisons with saved runs allow for them.
`environment.md` lists these differences, including A2f's designed lists on processors with AVX-512.

## License

The code is licensed under the MIT License (`LICENSE`). The data, outputs, figures and documentation written
for this repository are licensed under CC BY 4.0 (`LICENSE-DATA.md`). Third-party files keep their own terms;
`LICENSE-DATA.md` lists them with their sources.

## Citation

If you use the code or data, please cite the paper (top of this page). `CITATION.md` gives the reference and
the references of the data and models of others that the analyses use.
