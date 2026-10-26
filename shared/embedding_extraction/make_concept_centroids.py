"""
make_concept_centroids.py

WHAT IT DOES
Averages each of the 14 deep-network embeddings of the 26,107 THINGS photographs (26,107 x 512 each,
made by extract_embeddings.py) within the 1,854 THINGS concepts:
    1. every one of the 512 columns is standardized over the 26,107 images:
           (x - column mean) / (column standard deviation + 1e-8), in float32
    2. the standardized rows are averaged within concept (concepts in sorted name order, as
       pandas.factorize(sort=True) gives them; this is alphabetical, NOT the THINGS order of
       unique_id.txt)
The 14 concept-centroid files (1,854 x 512 each, about 47 MB in all) are the deep-network input of
module A4b (SI Appendix, Section 4); of the 14 full files, only the four late layers that module A3a
needs are included.

INPUTS
    --embeddings-folder   folder with the 14 files <net>_<early|late>.npz (array 'features',
                          26,107 x 512, rows in the order of shared/things/things_images.csv sorted by
                          embedding_row)
    shared/things/things_images.csv (the concept of every embedding row)
OUTPUTS
    --out-folder          <net>_<layer>.npz with arrays
                              centroids   1,854 x 512 float32
                              concepts    the 1,854 concept names in the row order of centroids
                              image_counts number of photographs averaged per concept
                              source      name of the full file it came from

RUN (from the repository root; reads one file at a time, about 60 MB of memory per file)
    python shared/embedding_extraction/make_concept_centroids.py \
        --embeddings-folder <folder with the 14 full files> --out-folder shared/embeddings_things/concept_centroids
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'python'))
import concept_tables   # noqa: E402

NETWORK_LAYERS = ['alexnet_early', 'alexnet_late', 'clip_early', 'clip_late', 'cornet_early', 'cornet_late',
                  'dinov3_early', 'dinov3_late', 'inception_early', 'inception_late', 'resnet18_early',
                  'resnet18_late', 'vgg16_early', 'vgg16_late']


def concept_codes():
    """Concept index of every embedding row (sorted-name order) and the concept names."""
    images = concept_tables.load_things_images(order='embedding')
    codes, concepts = pd.factorize(images['concept'], sort=True)
    return codes, np.asarray(concepts, dtype=str)


def centroids_of(features, codes, number_of_concepts):
    """Standardize every column over the images, then average the rows within concept."""
    standardised = (features - features.mean(0)) / (features.std(0) + 1e-8)
    counts = np.bincount(codes, minlength=number_of_concepts).astype(np.float32)
    sums = np.zeros((number_of_concepts, features.shape[1]), np.float32)
    np.add.at(sums, codes, standardised)
    return sums / counts[:, None], counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--embeddings-folder', required=True)
    parser.add_argument('--out-folder', required=True)
    args = parser.parse_args()

    codes, concepts = concept_codes()
    number_of_concepts = len(concepts)
    assert number_of_concepts == 1854
    os.makedirs(args.out_folder, exist_ok=True)
    for name in NETWORK_LAYERS:
        path = os.path.join(args.embeddings_folder, name + '.npz')
        loaded = np.load(path, allow_pickle=True)
        features = loaded['features'].astype(np.float32)
        assert features.shape[0] == len(codes) == 26107
        filenames = loaded['filenames'] if 'filenames' in loaded.files else None
        if filenames is not None:
            embedding_order = concept_tables.load_things_images(order='embedding')['filename'].to_numpy()
            assert np.array_equal(np.asarray(filenames, dtype=str), embedding_order.astype(str)), 'row order differs'
        centroids, counts = centroids_of(features, codes, number_of_concepts)
        np.savez_compressed(os.path.join(args.out_folder, name + '.npz'), centroids=centroids.astype(np.float32),
                            concepts=concepts, image_counts=counts.astype(np.int32), source=np.array(name + '.npz'))
        print('%s: %d x %d -> %d concept centroids' % (name, features.shape[0], features.shape[1], number_of_concepts))
        del features, loaded


if __name__ == '__main__':
    main()
