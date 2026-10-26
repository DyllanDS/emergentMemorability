# External inputs (M1 to M9)

Some inputs of the analyses are not in the repository, because they are not public, because their license
does not allow redistribution, or because they are large and few results need them. Every module runs without
them. Where a result needs one of these inputs, the module computes the reported numbers from the saved results
of that analysis in its `data/provided/` folder (`data/provided_faces/` in module A6a), prints

    PROVIDED OUTPUT: regenerating needs <input>; see EXTERNAL_INPUTS.md

and marks those numbers `provided` in the `mode` column of `reported_numbers.csv`. When the input is put at the
path given below, the same command computes the results from it. No module uses a different input in place of
an absent one.

Run every command from the repository root. After computing results from an input, rebuild the two tables:

    python collect_numbers.py
    python statistics/collect_all_tests.py

| Id | Input | Held by | What it lets the code compute | Where it goes |
|---|---|---|---|---|
| M1 | THINGS trial file of Kramer et al. (2023), `imbysub-all-11-16-21.mat` | the original authors, on request | SI Appendix, Section 5 (Figs. S13 and S14); the THINGS results of SI Appendix, Section 4 (Fig. S12, Table S2); SI Appendix, Fig. S9c and its numbers in Section 1; the trial structures of M6 | `external/kramer_trials/` |
| M2 | per-image THINGS count table, `things_per_image_counts.csv` | not public | SI Appendix, Section 4 (THINGS concepts, memorability side); the original THINGS hit and false alarm rates of the more and less memorable critical items of Fig. 3 (module A3b) | `external/kramer_trials/` |
| M3 | the ten THINGS deep-network spaces that are not included | computed from the THINGS images (M7) | the deep-network result of SI Appendix, Section 5 (with M1) | `shared/embeddings_things/` |
| M4 | 10k US Adult Faces Database (Bainbridge et al., 2013) | the original authors, on request | the faces half of SI Appendix, Fig. S15 | `external/bainbridge_faces/` |
| M5 | MemToolbox2 (Python) | not public | refitting the face study's d_a analysis (Fig. 4c; SI Appendix, Section 3) | `shared/memtoolbox2/` |
| M6 | trial structures of 400 THINGS sessions | built from M1 | the model side of SI Appendix, Fig. S4 | `A2c_continuous_recognition_figS4/data/` |
| M7 | THINGS images (optional) | public download, after a license agreement | thumbnails in Fig. 3e; extracting the THINGS embeddings | `external/things_images/` |
| M8 | scene memorability package of Isola et al. (2011) (optional) | public download | extracting the scene embeddings; rebuilding the scene participant matrices | `external/isola/` |
| M9 | SPoSE triplets (optional) | public download (CC BY 4.0) | refitting the embeddings of SI Appendix, Fig. S16 | `external/spose_triplets/` |

## M1. The THINGS trial file of Kramer et al. (2023)

`imbysub-all-11-16-21.mat` (about 6.3 GB, MATLAB v7.3) holds every trial of the THINGS memorability study of
Kramer, Hebart, Baker and Bainbridge (2023): 21,137 participant sessions of continuous recognition. It is not
public, and the original authors share it on request. Their public page (osf.io/5a7z6) holds the per-image
scores and the concept categories, which the repository includes.

Put the file at `external/kramer_trials/imbysub-all-11-16-21.mat` (reading it needs h5py). Module A5 turns it
into a trial table, `external/kramer_trials/derived/trials.npz`, and its later steps write further per-trial
tables to the same folder; the trial table can also be put there directly in place of the raw file. These
tables hold per-trial data of a study that is not public, so do not share them (`.gitignore` keeps them out of
the repository).

With the file in place, module A5 computes SI Appendix, Section 5 (Figs. S13 and S14; its deep-network result
also needs M3) and writes the intervals that resample participant sessions to its
`data/provided/trial_intervals.csv`, the file it reads when the trial file is absent. Module A4a computes the
THINGS results of SI Appendix, Section 4 (Fig. S12 and Table S2), which it otherwise reads from its
`data/provided/things_results.json`. Module A3b computes SI Appendix, Fig. S9c and its numbers in Section 1,
and module A2c builds the trial structures of M6 from A5's trial table and runs its model through them (the
model side of SI Appendix, Fig. S4). The data side of Fig. S4 is the whole-stream analysis of SI Appendix,
Fig. S13 (its open circles), which module A5 computes; module A2c draws it from A5's saved results. Run the
four modules in this order, because A2c, A3b and A4a read the trial table that A5 writes:

    python A5_things_context_effects_figS13_S14/code/run_module.py
    python A2c_continuous_recognition_figS4/code/run_module.py
    python A3b_hits_and_false_alarms_figS9/code/run_module.py
    python A4a_crowding_across_scales_figS12_tableS2/code/run_module.py

Module A5 first converts the trial file, a step that reads the image name of every trial through HDF5
references and takes a long time. The analyses after it take about 8 minutes and about 1.6 GB of memory on
the tested laptop, and the deep-network step adds about 15 minutes and 5 GB (`--skip-deep-network` leaves it
out). The model run of A2c takes about 4 minutes. Each module compares its new results with the saved ones in
its `data/provided/` folder.

## M2. The per-image THINGS count table

`things_per_image_counts.csv` has one row per THINGS photograph (26,107 rows) and the columns `image_nr` (the
image number of `shared/things/things_images.csv`), `n_old`, `hits`, `n_new` and `false_alarms`, which give for
each photograph the numbers of old and new trials and of "old" responses to them in the trial file of Kramer et
al. (2023). The old trials of that file include every repeat of a photograph (median 88 old trials per
photograph). The table is not public. The repository has no script that builds it from M1, so M1 alone does
not replace it.

Put it at `external/kramer_trials/things_per_image_counts.csv`. With it, module A4b computes the memorability
side of the THINGS concepts analysis of SI Appendix, Section 4 (its correlations over concepts and their
intervals), and module A3b computes the THINGS side of its halves comparison: the mean hit and false alarm
rates of the more and the less memorable critical items of the inversion experiment in the original THINGS
data. SI Appendix, Fig. S9c needs M1.

    python A4b_things_concepts/code/run_module.py
    python A3b_hits_and_false_alarms_figS9/code/run_module.py

## M3. The ten THINGS deep-network spaces that are not included

The seven networks of SI Appendix, Section 5 (AlexNet, VGG16-BN, ResNet18, InceptionV3, CORnet-S, DINOv3
ViT-B/16 and CLIP ViT-B/32) give 14 THINGS image spaces, an early and a late layer of each network (26,107
photographs x 512 values per file). The repository includes the four late layers that SI Appendix, Fig. S8 needs
(`alexnet_late`, `vgg16_late`, `resnet18_late` and `clip_late` in `shared/embeddings_things/`) and the concept
averages of all 14 spaces (`shared/embeddings_things/concept_centroids/`). The other ten files
(`alexnet_early`, `vgg16_early`, `resnet18_early`, `clip_early`, `inception_early`, `inception_late`,
`cornet_early`, `cornet_late`, `dinov3_early` and `dinov3_late`) are needed only for the deep-network result of
SI Appendix, Section 5 (A Similarity Space Without Human Judgments), which also needs M1.

They are computed from the THINGS images (M7) with `shared/embedding_extraction/extract_embeddings.py`;
`shared/embedding_extraction/README.md` gives the commands, networks, layers, weights and preprocessing. Put
each file at `shared/embeddings_things/<network>_<layer>.npz` (for example
`shared/embeddings_things/inception_late.npz`) and run module A5 with M1 in place:

    python A5_things_context_effects_figS13_S14/code/run_module.py

## M4. The 10k US Adult Faces Database

The database holds the face images and memorability scores of Bainbridge, Isola and Oliva (2013). The original
authors share it on request (wilmabainbridge.com/facememorability2.html) under a license that does not allow
redistribution, so the repository holds nothing computed per face (no image list, scores, embeddings or
crowding values).

Put `10kfaces.zip` and `2kmemorability.zip` (or their unzipped folders) in `external/bainbridge_faces/` and run

    python A6a_faces_and_scenes_figS15/code/run_module.py

Module A6a unzips the archives into `external/bainbridge_faces/derived/`, extracts the 14 face spaces with
`shared/embedding_extraction/` (this needs PyTorch, torchvision, timm, the OpenAI CLIP package and CORnet, and
openpyxl for the score file; about an hour), and computes the faces half of SI Appendix, Fig. S15 (SI Appendix,
Section 6, Faces and Scenes). Everything per face stays in `external/bainbridge_faces/derived/`; do not share
it.

## M5. MemToolbox2 (Python)

MemToolbox2 is a Python package for fitting memory models; the registered unequal-variance signal-detection
fit of the face study uses its `UVSDModel`, `ByCell` and `MLE`. The package is not publicly available. The
registered fit, its resamples over participants and over the six rare features, the fits that leave out one
rare-feature pair and the fits of the REM run after data collection are saved in
`A3c_face_study_model_and_sdt_figS10_S11/data/provided/`, and module A3c computes every reported d_a value from
them (Fig. 4c; SI Appendix, Section 3, Figs. S10b and S11b).

With the package at `shared/memtoolbox2/memtoolbox/` (or at
`shared/memtoolbox2/python/memtoolbox/`; a copy elsewhere on the Python path is not used), the command

    python A3c_face_study_model_and_sdt_figS10_S11/code/run_module.py

also fits the registered model again (about 1.5 minutes), refits it with each rare-feature pair left out
(about 7 minutes) and fits the 243 settings of the REM run after data collection (hours; `--skip-long` leaves
out these 243 fits), and compares each new fit with the saved one. The reported numbers still come from the
saved fits and resamples. The two bootstraps behind the intervals, 10,000 resamples of participants and 1,000
resamples of the six rare features, take hours and are not run by `run_module.py`:

    python A3c_face_study_model_and_sdt_figS10_S11/code/bootstrap_parallel.py --workers 2
    python A3c_face_study_model_and_sdt_figS10_S11/code/bootstrap_items.py --workers 2

The participant bootstrap takes about 7.4 hours on 22 workers (`--workers 22`). Both write their resamples and
reports to the module's `outputs/` folder, where they can be compared with the saved ones in `data/provided/`.

## M6. The trial structures behind the model of SI Appendix, Fig. S4

`trial_templates_400.npz` gives, for 400 sessions of the study of Kramer et al. (2023), which of the session's
distinct images each of its 295 trials shows and the type of each trial (400 x 295; no responses, image numbers
or identifiers). The model of SI Appendix, Fig. S4 runs its simulated participants through these sequences.
The file is built from the trial table of M1 by `A2c_continuous_recognition_figS4/code/make_trial_templates.py`
and is not public.

Put it at `A2c_continuous_recognition_figS4/data/trial_templates_400.npz`, or provide M1 and run module A5
first (module A2c then reads A5's trial table), and run

    python A2c_continuous_recognition_figS4/code/run_module.py

The model run takes about 4 minutes, and the module compares it with the saved run in its `data/provided/`
folder.

## M7. THINGS images (optional)

The THINGS object images (Hebart et al., 2019) are distributed as `images_THINGS.zip` at osf.io/jum2f (5.0 GB,
md5 3c018a7c37ac41ff7d84d7683fea0630). The password comes with the license agreement, which allows research and
educational use only and no redistribution.

No reported number needs the images. Module 03 draws them as thumbnails in Fig. 3e when they are present
(otherwise the panel shows markers and concept names, as every THINGS panel in the repository does), and
`shared/embedding_extraction/` extracts the THINGS embeddings from them (M3). A Fig. 3e drawn with the
thumbnails contains THINGS photographs, so do not redistribute it.

Unzip the archive so that the concept folders sit directly in `external/things_images/`
(`external/things_images/<concept>/<file>.jpg`), and run module 03 to draw the thumbnails:

    matlab -batch "restoredefaultpath; cd('03_inversion_experiment_fig3/code'); run_module"

The extraction commands are in `shared/embedding_extraction/README.md`.

## M8. The scene memorability package of Isola et al. (2011) (optional)

Isola, Xiao, Torralba and Oliva (2011) distribute their scene memorability data publicly as
`cvpr_memorability_data.zip` (about 1.8 GB), linked from web.mit.edu/phillipi/Public/WhatMakesAnImageMemorable/.

No reported number needs the package, because the repository includes the 14 scene embeddings computed from its
images (`shared/embeddings_isola/`) and the participant response matrices that module A6a reads
(`A6a_faces_and_scenes_figS15/data/scenes/subject_matrices.mat`). The package is needed only to extract the
embeddings again or to rebuild the matrices. The command `python external/download_external.py --isola`
downloads it into `external/isola/`, unzips it and checks the files it uses by md5; the commands below expect
its `Data` folder at `external/isola/Data/`. To rebuild the participant matrices (module A6a uses this copy when
its own is absent):

    matlab -batch "addpath('shared/embedding_extraction'); export_subjects('external/isola/Data', 'external/isola/subject_matrices.mat')"

The extraction commands are in `shared/embedding_extraction/README.md`.

## M9. SPoSE triplets (optional)

The files `trainset.txt` and `validationset.txt` hold the odd-one-out judgments from which the SPoSE embedding
is learned (THINGS-data, Hebart et al., 2023; osf.io/f5rn6; CC BY 4.0).

No reported number needs them, because the repository includes the 14 refits of SI Appendix, Fig. S16
(`A6b_embedding_size_figS16/data/refits/`), from which the reported values come. With the triplets, module A6b
refits the embeddings and analyzes the new refits beside the saved ones:

    python external/download_external.py --spose-triplets
    python A6b_embedding_size_figS16/code/run_module.py --refit

The refits need PyTorch and take about 3 hours on the GPU of an Apple M2 Pro (`--device cpu` runs the same
model on the processor, much more slowly). A refit on a GPU does not reproduce the saved refits exactly.
