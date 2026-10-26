"""
build_concept_tables.py

WHAT IT DOES
Writes the plain-text concept tables of shared/things/ from the two stripped
MATLAB concept files (made by strip_concept_info.m) and from the monkey PCA file:

    concept_table.csv        one row per THINGS concept (1,854, THINGS order): the
                             category, the Kramer et al. (2023) concept memorability, the
                             image count, the two full-THINGS summed similarities and the
                             49 SPoSE feature values
    monkey_features.csv      the 49 monkey IT principal-component scores per concept (the
                             features of concept_info_monkey.mat)
    monkey_pca_explained.csv the eigenvalue of each of the 576 principal components of the
                             IT channel responses and the cumulative share of variance
    labels_short49.txt       the 49 SPoSE feature labels, one per line

Every number is written with Python's repr(), the shortest decimal string that turns back
into exactly the same double, so the CSV tables carry the same values as the .mat files.

It also checks that
    - the concept names equal unique_id.txt (THINGS order),
    - the human feature matrix equals spose_embedding_49d_sorted.txt row for row,
    - the stored monkey features equal the first 49 columns of the PCA scores row for row.

INPUTS
    shared/things/concept_info_human.mat, concept_info_monkey.mat,
    spose_embedding_49d_sorted.txt, unique_id.txt, labels_short49.mat (all included)
    monkeyPCA.mat (the PCA of the TVSD IT channels, Papale et al. 2025; made by
    monkeyFeatGen.m; not included: its path is the only argument)

OUTPUTS
    the four files above, in shared/things/

RUN (from the repository root)
    python shared/things/provenance/build_concept_tables.py <path to monkeyPCA.mat>
"""
import csv
import os
import sys

import numpy as np
import scipy.io as sio

THINGS_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # shared/things/
NUMBER_OF_CONCEPTS = 1854
NUMBER_OF_FEATURES = 49


def exact_text(value):
    """The shortest decimal text that reads back as exactly the same number ('NaN' for missing)."""
    value = float(value)
    if np.isnan(value):
        return 'NaN'
    return repr(value)


def load_concept_struct(file_name):
    """The struct concInfo of one stripped concept file."""
    path = os.path.join(THINGS_FOLDER, file_name)
    loaded = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    return loaded['concInfo']


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: python build_concept_tables.py <path to monkeyPCA.mat>')
    monkey_pca_file = sys.argv[1]

    human = load_concept_struct('concept_info_human.mat')
    monkey = load_concept_struct('concept_info_monkey.mat')

    concept_names = [str(name) for name in human.concepts]
    monkey_concept_names = [str(name) for name in monkey.concepts]
    human_features = np.asarray(human.features, dtype=float)
    monkey_features = np.asarray(monkey.features, dtype=float)

    # ---- checks ----
    assert len(concept_names) == NUMBER_OF_CONCEPTS
    assert concept_names == monkey_concept_names, 'the two concept files list the concepts differently'
    with open(os.path.join(THINGS_FOLDER, 'unique_id.txt')) as handle:
        things_order = [line.strip() for line in handle if line.strip()]
    assert concept_names == things_order, 'concept names are not in THINGS (unique_id.txt) order'
    spose_rows = np.loadtxt(os.path.join(THINGS_FOLDER, 'spose_embedding_49d_sorted.txt'))
    assert spose_rows.shape == (NUMBER_OF_CONCEPTS, NUMBER_OF_FEATURES)
    largest_difference = float(np.max(np.abs(spose_rows - human_features)))
    print('human features vs spose_embedding_49d_sorted.txt: max abs difference = %.3g' % largest_difference)
    assert largest_difference == 0.0, 'the stored features differ from the SPoSE text file'

    pca = sio.loadmat(monkey_pca_file, squeeze_me=True)
    pca_scores = np.asarray(pca['score'], dtype=float)
    pca_eigenvalues = np.asarray(pca['latent'], dtype=float).ravel()
    largest_difference = float(np.max(np.abs(pca_scores[:, :NUMBER_OF_FEATURES] - monkey_features)))
    print('stored monkey features vs PCA scores (first 49 columns): max abs difference = %.3g'
          % largest_difference)
    assert largest_difference == 0.0, 'the stored monkey features are not the first 49 PCA scores'

    # ---- concept_table.csv ----
    category_numbers = np.asarray(human.categoryNumber, dtype=int)
    category_labels = [str(label) for label in human.category]
    concept_numbers = np.asarray(human.conceptNumber, dtype=int)
    concept_memorability = np.asarray(human.memorabilityCat, dtype=float)
    image_counts = np.asarray(human.conceptCounts, dtype=int)
    summed_similarity_human = np.asarray(human.weightedSimByConc, dtype=float)
    summed_similarity_monkey = np.asarray(monkey.weightedSimByConc, dtype=float)

    header = ['concept_number', 'concept', 'category_number', 'category_label',
              'kramer_concept_cr', 'concept_image_count_as_stored',
              'weighted_similarity_full_things_human', 'weighted_similarity_full_things_monkey']
    for feature in range(NUMBER_OF_FEATURES):
        header.append('feature_%02d' % (feature + 1))
    with open(os.path.join(THINGS_FOLDER, 'concept_table.csv'), 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for concept in range(NUMBER_OF_CONCEPTS):
            row = [int(concept_numbers[concept]), concept_names[concept],
                   int(category_numbers[concept]), category_labels[concept],
                   exact_text(concept_memorability[concept]), int(image_counts[concept]),
                   exact_text(summed_similarity_human[concept]),
                   exact_text(summed_similarity_monkey[concept])]
            for feature in range(NUMBER_OF_FEATURES):
                row.append(exact_text(human_features[concept, feature]))
            writer.writerow(row)

    # ---- monkey_features.csv ----
    header = ['concept_number', 'concept']
    for component in range(NUMBER_OF_FEATURES):
        header.append('pc_%02d' % (component + 1))
    with open(os.path.join(THINGS_FOLDER, 'monkey_features.csv'), 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for concept in range(NUMBER_OF_CONCEPTS):
            row = [int(concept_numbers[concept]), concept_names[concept]]
            for component in range(NUMBER_OF_FEATURES):
                row.append(exact_text(monkey_features[concept, component]))
            writer.writerow(row)

    # ---- monkey_pca_explained.csv ----
    total_variance = float(np.sum(pca_eigenvalues))
    running_total = 0.0
    with open(os.path.join(THINGS_FOLDER, 'monkey_pca_explained.csv'), 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['component', 'eigenvalue', 'percent_of_variance', 'cumulative_percent'])
        for component in range(len(pca_eigenvalues)):
            running_total = running_total + pca_eigenvalues[component]
            writer.writerow([component + 1, exact_text(pca_eigenvalues[component]),
                             exact_text(100.0 * pca_eigenvalues[component] / total_variance),
                             exact_text(100.0 * running_total / total_variance)])
            if component + 1 == NUMBER_OF_FEATURES:
                print('cumulative share of IT variance in the first %d components: %.2f percent'
                      % (NUMBER_OF_FEATURES, 100.0 * running_total / total_variance))
    print('components in the monkey PCA: %d' % len(pca_eigenvalues))

    # ---- labels_short49.txt ----
    labels = sio.loadmat(os.path.join(THINGS_FOLDER, 'labels_short49.mat'), squeeze_me=True)['labels_short']
    labels = [str(label) for label in np.atleast_1d(labels)]
    assert len(labels) == NUMBER_OF_FEATURES
    assert labels == [str(label) for label in human.featureLabels], 'labels differ from the stored featureLabels'
    with open(os.path.join(THINGS_FOLDER, 'labels_short49.txt'), 'w') as handle:
        for label in labels:
            handle.write(label + '\n')
    print('wrote concept_table.csv, monkey_features.csv, monkey_pca_explained.csv, labels_short49.txt')


if __name__ == '__main__':
    main()
