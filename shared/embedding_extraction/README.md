# shared/embedding_extraction/: how the deep-network embeddings are made

The THINGS and scene embeddings that modules A3a, A6a and A4b read are included in
`shared/embeddings_things/` and `shared/embeddings_isola/`; the ten THINGS spaces that are not included
(EXTERNAL_INPUTS.md, M3) come from `extract_embeddings.py`. The face embeddings are not included either
(license of the face database; EXTERNAL_INPUTS.md, M4); with the face images in place, module A6a makes them
with `extract_embeddings.py` (Commands, below). The scripts need PyTorch and the images (EXTERNAL_INPUTS.md,
M4, M7 and M8; `external/README.md`).

| File | What it does |
|---|---|
| `extract_embeddings.py` | Extracts the early- and late-layer activations of seven networks for a list of images; larger layers are reduced to 512 principal components. The image folder, the image list and the output folder are required arguments, and `--only` runs some of the networks. A missing `cornet` or `clip` package, or a DINOv3 model without its 12 blocks or its final norm, stops the run with an error. |
| `dinov3_lowmem.py` | The same DINOv3 layers with less memory: the activations are written to disk and the PCA is computed exactly through the image-by-image Gram matrix. It makes the two DINOv3 files of `shared/embeddings_isola/`, for which `extract_embeddings.py` would hold about 17 GB of activations in memory. It imports `extract_embeddings.py` from this folder for the image loading. |
| `make_concept_centroids.py` | Averages each of the 14 THINGS spaces (26,107 images) within concept, giving the 14 files of `shared/embeddings_things/concept_centroids/` (the deep-network input of module A4b). |
| `export_isola.m`, `export_subjects.m` | MATLAB exports of the public Isola et al. (2011) package: the image folder and the score table (`export_isola`), and the participant response matrices for the scene noise ceiling (`export_subjects`; module A6a reads the same matrices from `A6a_faces_and_scenes_figS15/data/scenes/subject_matrices.mat`). Pass the input and output folders as arguments. |

## Models and layers (two layers per network: "early" and "late")

| Network | Weights | Early layer | Late layer |
|---|---|---|---|
| AlexNet | torchvision `IMAGENET1K_V1` | `features.3` | `avgpool` |
| VGG16-BN | torchvision `IMAGENET1K_V1` | `features.14` | `avgpool` |
| ResNet18 | torchvision `IMAGENET1K_V1` | `layer2` | `avgpool` (512 dimensions, no PCA) |
| InceptionV3 | torchvision `IMAGENET1K_V1` | `Mixed_5b` | `avgpool` |
| CORnet-S | `cornet.cornet_s(pretrained=True)` | `V2` | `IT` |
| DINOv3 ViT-B/16 | LVD-1689M | `blocks.6` without the 5 storage tokens (1 CLS + 4 registers) | final `norm` |
| CLIP ViT-B/32 | OpenAI `clip.load('ViT-B/32')` (QuickGELU) | `transformer.resblocks.6` | `ln_post` |

The two DINOv3 concept-centroid files of `shared/embeddings_things/` are averages of THINGS embeddings made
with the torch.hub weights `facebookresearch/dinov3` `dinov3_vitb16`; the two DINOv3 files of
`shared/embeddings_isola/`, and both scripts in this folder, use the timm port
`vit_base_patch16_dinov3.lvd1689m` of the same weights.

## Preprocessing and reduction

JPEG images are decoded in draft mode near 256 pixels. For AlexNet, VGG16-BN, ResNet18, InceptionV3 and
CORnet-S each image is padded to a square with white, resized to 256 x 256 (`transforms.Resize((256, 256))`)
and normalized with the ImageNet mean and SD. DINOv3 images are resized and normalized the same way,
without the padding. CLIP images are padded to a square and then go through CLIP's own preprocessing.
Activations of convolutional layers with more than 65,536 values per image are average-pooled to 8 x 8 per
channel before they are flattened (the early layers of AlexNet, VGG16-BN, ResNet18, InceptionV3 and
CORnet-S). Every layer with more than 512 values is then reduced to 512 with scikit-learn
`PCA(n_components=512, svd_solver='auto', random_state=0)`, fitted on all images of the set.

Each output `<network>_<early|late>.npz` holds `features` (images x 512, float32, rows in the order of the
image list), `filenames`, `network`, `layer_label`, `pca_applied`, `original_dim` and, for layers reduced by
PCA, `explained_variance_ratio` (`dinov3_lowmem.py` also stores `pca_method`); `extraction_manifest.json` in
the output folder lists every layer written.

## Devices

The scripts use CUDA when it is available, otherwise Apple MPS, otherwise the CPU (`dinov3_lowmem.py` uses
MPS or the CPU); the model, weights and arithmetic are the same on each. MPS cannot do adaptive average
pooling when the input size is not a multiple of the output size, so on MPS these steps run on the CPU:
the late-layer pooling of AlexNet (7 x 7 to 6 x 6) and VGG16-BN (8 x 8 to 7 x 7), and the 8 x 8 pooling of
the wide convolutional layers. The torch.hub download of the DINOv3 weights is gated, so both scripts load
timm's copy of the weights.

## Commands (from the repository root; MPLBACKEND=Agg)

THINGS (26,107 images; needs the THINGS images in `external/things_images/`, EXTERNAL_INPUTS.md, M7, and
the image list in embedding row order):

    python -c "import pandas as pd; t = pd.read_csv('shared/things/things_images.csv').sort_values('embedding_row'); t[['filename', 'concept', 'rel_path']].to_csv('external/things_images/image_order.csv', index=False)"
    python shared/embedding_extraction/extract_embeddings.py --images-dir external/things_images \
        --image-order external/things_images/image_order.csv --out-dir <output folder>
    python shared/embedding_extraction/make_concept_centroids.py --embeddings-folder <output folder> \
        --out-folder <folder for the concept centroids>

Isola scenes (10,442 images: 2,400 targets and 8,042 fillers; needs `external/isola/` from
`python external/download_external.py --isola`, EXTERNAL_INPUTS.md, M8, and then the image export):

    matlab -batch "addpath('shared/embedding_extraction'); export_isola('external/isola/Data', 'external/isola')"
    python shared/embedding_extraction/extract_embeddings.py --images-dir external/isola/images \
        --image-order A6a_faces_and_scenes_figS15/data/scenes/image_order.csv --out-dir <output folder> \
        --only alexnet vgg16 resnet18 inception cornet clip
    python shared/embedding_extraction/dinov3_lowmem.py --images-dir external/isola/images \
        --image-order A6a_faces_and_scenes_figS15/data/scenes/image_order.csv --out-dir <output folder> \
        --scratch-dir <folder with about 17 GB free>
    matlab -batch "addpath('shared/embedding_extraction'); export_subjects('external/isola/Data', 'external/isola/subject_matrices.mat')"

`A6a_faces_and_scenes_figS15/data/scenes/image_order.csv` is the scene image list: the row order of the
files in `shared/embeddings_isola/`, which also store the file names in `filenames`.

Faces (10,168 images of the 10k US Adult Faces Database; EXTERNAL_INPUTS.md, M4): module A6a
(`code/faces/regenerate_faces.py`) writes the face image list with `code/faces/00_build_image_order.py`
and runs `extract_embeddings.py` on the face images. The face images and everything computed per face
stay outside the repository.

## Software

Optional packages: torch, torchvision, timm, scikit-learn, the OpenAI `clip` package
(`pip install git+https://github.com/openai/CLIP.git`) and `cornet`
(`pip install git+https://github.com/dicarlolab/CORnet`; GPL-3), and MATLAB for the two export scripts.
The scripts never substitute other weights or layers: a model that fails to load, or lacks a layer it
should have, stops the run.
