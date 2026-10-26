# shared/embeddings_things/: deep-network embeddings of the 26,107 THINGS photographs

Embeddings of the 26,107 THINGS photographs from seven ImageNet- or web-trained networks, an early and a
late layer each, 512 dimensions per layer; larger layers are reduced to 512 principal components
(`shared/embedding_extraction/extract_embeddings.py`; models, layers and weights in
`shared/embedding_extraction/README.md`).

| File | Rows x columns | Used by |
|---|---|---|
| `alexnet_late.npz`, `vgg16_late.npz`, `resnet18_late.npz`, `clip_late.npz` | 26,107 x 512 (float32) | module A3a (SI Appendix, Fig. S8), which combines the four late layers and reduces them to the 49 features of its model |
| `concept_centroids/<network>_<early or late>.npz` (14 files) | 1,854 x 512 (float32) | module A4b (SI Appendix, Section 4): crowding among the THINGS concepts |

Each full `.npz` holds `features` (rows sorted by file name; the column `embedding_row` of
`shared/things/things_images.csv` gives each image's row), `filenames`, `network`, `layer_label`,
`pca_applied`, `original_dim` and `explained_variance_ratio` (not in `resnet18_late.npz`, whose 512
dimensions need no PCA). Load with `numpy.load(path, allow_pickle=True)` (the `filenames` array is stored as
Python objects).

## What is included

Module A3a needs the photograph-level embeddings of the four late layers above, and module A4b needs only
each layer's concept averages. The four full layers (about 190 MB) and the concept centroids of all 14
layers (about 47 MB) are included. The other ten full files (`alexnet_early`, `vgg16_early`,
`resnet18_early`, `clip_early`, `inception_early`, `inception_late`, `cornet_early`, `cornet_late`,
`dinov3_early`, `dinov3_late`) are needed only to compute the deep-network result of SI Appendix, Section 5
again, together with the Kramer et al. trial file (EXTERNAL_INPUTS.md, M3 and M1); they go in this folder
as `<network>_<layer>.npz`.

## Concept centroids

`concept_centroids/<network>_<layer>.npz` holds `centroids` (1,854 x 512), `concepts` (the concept names in
the row order of `centroids`: sorted by name, which is not the THINGS order of
`shared/things/unique_id.txt`), `image_counts` (photographs per concept) and `source` (the name of the full
file it was made from). `shared/embedding_extraction/make_concept_centroids.py` makes each one from the full
26,107 x 512 file of its layer: every column is standardized over the 26,107 photographs
((x - mean) / (sd + 1e-8), float32), then the rows are averaged within concept.
