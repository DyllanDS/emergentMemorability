# shared/things/provenance/: how the THINGS tables and the stimulus sets were made

These scripts document how the tables in `shared/things/` were made and how the stimuli of the inversion
and image-set experiments were chosen. No module runs them, and most of them need files that are not
included: the full concept files with the THINGS thumbnails, the image file with the deep-network
similarity matrices (`ImageInfo.mat`, 2.7 GB) and the monkey spiking data. The tables they wrote are in
`shared/things/`, and the stimulus sets are in the module data folders. Paths in the scripts are relative to the
repository root, and a name in angle brackets, such as `<ImageInfo.mat>`, stands for a file that is not included.

## Scripts that write the tables of shared/things/

| Script | Input (not included unless stated) | Output |
|---|---|---|
| `strip_concept_info.m` | `ConceptInfo.mat` (132,630,398 bytes) and `ConceptMonkeyInfo.mat` (132,906,408 bytes), struct `concInfo` | `concept_info_human.mat`, `concept_info_monkey.mat`: the same struct without `images` (THINGS thumbnails), `thingsSimRSM` (computed again by `shared/matlab/load_concept_info.m`), `memorabilityItem`, `memorabilityWilma` (a copy of the `CR` column of the public `Concept_to_category_linking.csv`) and `resMemPredictions`, saved in MATLAB v7 (readable with scipy). It stops when `corrcoef(features')` differs from the stored `thingsSimRSM` by more than 1e-12, and it checks that every kept field is unchanged after saving. |
| `build_concept_tables.py` | the two files above, `spose_embedding_49d_sorted.txt`, `unique_id.txt` and `labels_short49.mat` (all included), and `monkeyPCA.mat` (the PCA of the monkey IT channels, made by `monkeyFeatGen.m`) | `concept_table.csv`, `monkey_features.csv`, `monkey_pca_explained.csv`, `labels_short49.txt`. It checks that the concept names follow `unique_id.txt`, that the human features equal the SPoSE text file row for row, and that the monkey features equal the first 49 PCA scores row for row. |
| `rebuild_things_image_mapping.py` | the public `THINGS_Memorability_Scores.csv` (in `external/kramer/` after `python external/download_external.py`) | `things_image_mapping.csv`; it runs in the repository once the public file is downloaded |
| `build_things_image_tables.py` | the two public Kramer et al. (2023) files, the image list of the image-set analyses (`ImageConceptList.mat`) and the image list of the THINGS embedding extraction (`image_order.csv`; only its file name, concept and path columns are read) | `things_images.csv`, `concept_categories.csv`. It checks the image list against `ImageConceptList.mat`, and that the category file has 1,854 concepts in 23 named categories plus NA. |

## Scripts that built the concept files and chose the stimuli

| Script | What it does | Inputs it needs |
|---|---|---|
| `CombineConceptLevelStuff.m` | builds the concept struct `concInfo` of the human space: SPoSE features (`spose_embedding_49d_sorted.txt`), concept names from `images1854.mat`, categories from `Concept_to_category_linking.csv`, concept memorability from `THINGS_Memorability_Scores.csv`, image counts, `thingsSimRSM = corrcoef(features')` and the full-THINGS summed similarity | the SPoSE files (`shared/things/`), `images1854.mat` (THINGS thumbnails), the two public Kramer et al. files, `ImageInfo.mat` |
| `CombineConceptMonkey.m` | the same struct for the monkey IT space, from the monkey PCA scores | `monkeyPCA.mat` and the inputs above |
| `monkeyFeatGen.m` | averages the IT channels of the THINGS Ventral Stream Spiking Dataset (Papale et al. 2025; channels 513 to 832 of monkey F and 769 to 1,024 of monkey N) within concept and runs the PCA saved as `monkeyPCA.mat` (49 components keep 84.6 percent of the variance: `shared/things/monkey_pca_explained.csv`) | the two normalized multi-unit files of the dataset (gin.g-node.org/paolo_papale/TVSD; the script reads them as `<TVSD download>/monkeyF.mat` and `monkeyN.mat`) and `Filtered_Final_THINGS_Memorability_Concepts.csv` (the THINGS concept number of each training image) |
| `SampleDatasetsThatFlip.m` | draws the three 1,600-image sets of the image-set experiments (`rng(6)`) and saves them as `SampledDatasets.mat`. The image sets of the experiments are in `06_image_set_experiments_fig6_fig7/data/sampled_image_sets.mat`, and module 06 (`code/reproduce_set_selection.m`) draws the concepts of the three sets again from included files and checks them against that file. | `ImageInfo.mat`, `ConceptInfo.mat` |
| `SelectStimToFlipMemorability.m` | chooses the critical items of the inversion experiment (one per Kramer et al. memorability quintile, in the same ResMem quintile) and their similar items (`reqSimilar = [0 1 2 4 8]`) from combined concept and deep-network similarities | `ImageInfo.mat` with the deep-network similarity matrix. The lists participants saw are in `03_inversion_experiment_fig3/data/inversion_deployed_lists.csv`. |
