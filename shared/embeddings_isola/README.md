# shared/embeddings_isola/: deep-network embeddings of the 10,442 Isola scene images

Embeddings of the 10,442 images of the Isola et al. (2011) scene memorability package (2,400 targets and
8,042 fillers) from the same seven networks and layers as the THINGS embeddings, 512 dimensions per layer
(larger layers are reduced to 512 principal components). Module A6a reads them for the scenes half of SI
Appendix, Fig. S15 (SI Appendix, Section 6).

| File | Contents |
|---|---|
| `regenerated_<alexnet, vgg16, resnet18, inception, cornet, dinov3, clip>_<early, late>.npz` (14 files) | 10,442 x 512 (float32), rows in the order of `A6a_faces_and_scenes_figS15/data/scenes/image_order.csv` (the file names are stored as `filenames`) |
| `extraction_manifest.json` | shape, original dimension and cumulative explained variance of each layer |

Each `.npz` holds `features`, `filenames`, `network`, `layer_label`, `pca_applied`, `original_dim`,
`explained_variance_ratio` (layers reduced by PCA; ResNet18 late has 512 dimensions and no PCA) and, in the
two DINOv3 files, `pca_method`. Load with `numpy.load(path, allow_pickle=True)` (the `filenames` array is
stored as Python objects).

The twelve files of the other six networks come from `shared/embedding_extraction/extract_embeddings.py`.
The two DINOv3 files come from `shared/embedding_extraction/dinov3_lowmem.py`, which uses the same model,
layers and preprocessing and computes the PCA exactly through the image-by-image Gram matrix, because
`extract_embeddings.py` would hold about 17 GB of activations in memory for the two DINOv3 layers. PCA
signs are arbitrary. The scripts name their files `<network>_<layer>.npz`, and module A6a reads the files
of this folder whose names start with `regenerated_`. The commands are in
`shared/embedding_extraction/README.md`.
