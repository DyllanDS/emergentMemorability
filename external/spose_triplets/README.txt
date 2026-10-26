Folder for the SPoSE triplet files trainset.txt and validationset.txt (THINGS-data, Hebart et al., 2023;
osf.io/f5rn6; CC BY 4.0; EXTERNAL_INPUTS.md, M9; optional). From the repository root,

    python external/download_external.py --spose-triplets

downloads them here and checks their sizes and md5 sums. No reported number needs them: they are needed
only to refit the embeddings of SI Appendix, Fig. S16 (PyTorch; about 3 hours on an Apple M2 Pro GPU):

    python A6b_embedding_size_figS16/code/run_module.py --refit
