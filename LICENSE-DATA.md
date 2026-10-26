# License of the data, outputs and documentation

The code of this repository (every `.py` and `.m` file) is licensed under the MIT License (`LICENSE`).

Everything else written for this repository is licensed under the Creative Commons Attribution 4.0
International License (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/; legal code:
https://creativecommons.org/licenses/by/4.0/legalcode): the de-identified data of the three experiments, the
tables and outputs the code writes (`numbers.csv`, `tests.csv`, `reported_numbers.csv`, the statistics tables
and the figure data), the figures, the embeddings computed for this repository, and the README files and other
documentation. When you use them, cite the paper (`CITATION.md`).

## Third-party files

The files below, or the columns named, come from other sources and keep their own terms. Cite the sources
(`CITATION.md`) when you use them.

| Files | Source | Terms |
|---|---|---|
| `shared/things/spose_embedding_49d_sorted.txt`, `unique_id.txt`, `labels_short49.mat` (and its text copy `labels_short49.txt`), and the SPoSE feature values in `shared/things/concept_table.csv` and `concept_info_human.mat` | the SPoSE embedding of the THINGS concepts (Hebart et al., 2020), as distributed with THINGS-data (Hebart et al., 2023; osf.io/f5rn6) | CC BY 4.0 (`shared/things/LICENSE_spose.txt`) |
| the per-image memorability scores in `shared/things/things_images.csv`, `03_inversion_experiment_fig3/data/inversion_design.csv` and `03_inversion_experiment_fig3/data/paper_run/published_scores.csv` (column `kramer_cr` of each); their concept means in `shared/things/concept_table.csv` (column `kramer_concept_cr`) and in `shared/things/concept_info_human.mat` and `concept_info_monkey.mat` (field `memorabilityCat`); the category labels in `shared/things/concept_categories.csv`, `concept_table.csv` (columns `category_number` and `category_label`) and the two MAT files (fields `categoryNumber` and `category`); and the copies of these scores in module outputs (columns `kramer_cr` and `kramer_concept_cr`) | the public THINGS memorability scores and concept categories of Kramer et al. (2023), osf.io/5a7z6 | the source declares no license; shared with citation |
| `shared/things/things_image_mapping.csv` | built from the public score file of Kramer et al. (2023): the THINGS file names, image numbers and image addresses listed there | the source declares no license; shared with citation |
| `shared/things/monkey_features.csv`, `monkey_pca_explained.csv`, `concept_info_monkey.mat` | principal components of the IT recordings of the THINGS Ventral Stream Spiking Dataset (Papale et al., 2025; gin.g-node.org/paolo_papale/TVSD), averaged within concept (the scripts that compute them are in `shared/things/provenance/`) | CC BY 4.0 |
| `A6a_faces_and_scenes_figS15/data/scenes/isola_scores.csv`, `image_order.csv`, `subject_matrices.mat` | the per-image scores and participant response matrices of the scene memorability data of Isola et al. (2011), re-encoded from the public package | the source states no license; shared with citation |
| the color values in `shared/matlab/piyg_colormap.m` | the PiYG scheme of ColorBrewer, copyright 2002 Cynthia Brewer, Mark Harrower and The Pennsylvania State University | Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0) |
| `04_rare_feature_faces_fig4/data/fig4a_faces/` (five images, drawn in Fig. 4a) | synthetic portraits from the stimulus set of the rare-feature face study: generated images that show no real person (Methods, Rare-Feature Face Experiment) | CC BY 4.0, with the data of the experiment |

The THINGS images, the 10k US Adult Faces Database, the scene images of Isola et al. (2011) and the trial-level
THINGS data of Kramer et al. (2023) are not included in the repository and are not covered by these licenses
(EXTERNAL_INPUTS.md).
