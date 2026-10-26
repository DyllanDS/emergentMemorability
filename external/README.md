# external/: inputs that are not in the repository

Nothing in this folder is needed for the default run. Every module either computes its results from the data
in the repository or, where an input cannot be included, computes them from its saved results in
`data/provided/` and says so (`PROVIDED OUTPUT: regenerating needs <input>; see EXTERNAL_INPUTS.md`). The
subfolders are where those inputs go, and a module uses an input as soon as it is in place. EXTERNAL_INPUTS.md
describes each input (M1 to M9) and the commands that use it.

`python external/download_external.py`, run from the repository root, downloads the two public files of Kramer
et al. (2023) into `external/kramer/`, checks the size and md5 of each (the script stops with an error if
either differs), checks the tables in `shared/things/` against them, and prints the request steps for the
inputs that cannot be downloaded. The downloads use the Python standard library, and checking the tables needs
numpy and pandas. With `--isola` the script also downloads the scene package of Isola et al. (2011) (about
1.8 GB, M8), with `--spose-triplets` the SPoSE triplet files (M9), and with `--instructions` it only prints the
request steps.

| Folder | What goes there | Source and terms | What it lets the code compute |
|---|---|---|---|
| `kramer/` | `THINGS_Memorability_Scores.csv` and `Concept_to_category_linking.csv` of Kramer, Hebart, Baker and Bainbridge (2023) | osf.io/5a7z6; downloads osf.io/download/z4ve9/ (1,675,292 bytes, md5 99e6f81616675a73b3ddf92058d1cf50) and osf.io/download/8b7hg/ (113,027 bytes, md5 4b42d2ea23005dad9bf4bddb41aeac68); the source declares no license | nothing new: `download_external.py` checks the scores and categories that the repository includes against them |
| `kramer_trials/` | M1, the trial file `imbysub-all-11-16-21.mat` (about 6.3 GB), and M2, the per-image count table `things_per_image_counts.csv` | not public; the original authors share the trial file on request | M1: SI Appendix, Section 5 (Figs. S13 and S14), the THINGS results of SI Appendix, Section 4 (Fig. S12, Table S2), SI Appendix, Fig. S9c and the model side of SI Appendix, Fig. S4; M2: the memorability side of the THINGS concepts analysis (SI Appendix, Section 4) and the original THINGS hit and false alarm rates of the more and less memorable critical items of Fig. 3 (module A3b) |
| `bainbridge_faces/` | M4, `10kfaces.zip` and `2kmemorability.zip` of the 10k US Adult Faces Database (Bainbridge, Isola and Oliva, 2013) | request form at wilmabainbridge.com/facememorability2.html; the images and per-image scores may not be redistributed | the faces half of SI Appendix, Fig. S15 |
| `things_images/` | M7, the THINGS images (Hebart et al., 2019), unzipped as `<concept>/<file>.jpg` | osf.io/jum2f, `images_THINGS.zip` (5.0 GB, md5 3c018a7c37ac41ff7d84d7683fea0630), password after the license agreement; research and educational use only, no redistribution | no reported number: thumbnails in Fig. 3e and the extraction of the THINGS embeddings |
| `isola/` | M8, the scene memorability package of Isola et al. (2011), `cvpr_memorability_data.zip` (about 1.8 GB) | web.mit.edu/phillipi/Public/WhatMakesAnImageMemorable/; the source states no license (the images come from the SUN database) | no reported number: the extraction of the scene embeddings and the participant matrices of module A6a |
| `spose_triplets/` | M9, `trainset.txt` and `validationset.txt` of THINGS-data (Hebart et al., 2023) | osf.io/f5rn6 (54.4 MB and 6.0 MB), CC BY 4.0 | no reported number: refitting the embeddings of SI Appendix, Fig. S16 |

The repository includes, with citation, the per-image memorability scores of Kramer et al. (2023)
(`shared/things/things_images.csv`, column `kramer_cr`, and the concept means in
`shared/things/concept_table.csv`) and their concept categories (`shared/things/concept_categories.csv`), so
that the analyses run without a download; cite Kramer et al. (2023) when you use them. Three other
third-party sources are included in the form the analyses read: the SPoSE embedding of Hebart et al. (2020)
(`shared/things/spose_embedding_49d_sorted.txt`, `unique_id.txt` and `labels_short49.mat`, from osf.io/f5rn6,
CC BY 4.0), the monkey IT component scores (`shared/things/monkey_features.csv`), computed from the THINGS
Ventral Stream Spiking Dataset of Papale et al. (2025) (gin.g-node.org/paolo_papale/TVSD, DOI
10.12751/g-node.hc7zlv, CC BY 4.0) with the scripts in `shared/things/provenance/`
(`python external/download_external.py --instructions` lists the files of that dataset), and the per-image
scores and participant response matrices of the scene memorability data of Isola et al. (2011)
(`A6a_faces_and_scenes_figS15/data/scenes/`). `LICENSE-DATA.md` lists every third-party file with its terms.

The per-trial and per-face files that the modules compute from the inputs that are not public stay in these
folders (`kramer_trials/derived/` and `bainbridge_faces/derived/`). The `.gitignore` of the repository excludes
the contents of every subfolder except its `README.txt`, so that none of these inputs or their per-trial and
per-face derivatives can be committed by mistake.
