Folder for the scene memorability package of Isola et al. (2011) (EXTERNAL_INPUTS.md, M8; optional).
From the repository root,

    python external/download_external.py --isola

downloads cvpr_memorability_data.zip (about 1.8 GB) here, unzips it and checks the files it uses by md5.
No reported number needs the package: it is needed only to extract the scene embeddings again
(shared/embedding_extraction/) or to rebuild the scene participant matrices of module A6a with
shared/embedding_extraction/export_subjects.m, which expect the package's Data folder at
external/isola/Data/.
