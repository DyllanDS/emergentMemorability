"""
build_things_image_tables.py

WHAT IT DOES
Writes two THINGS tables of shared/things/:

  things_images.csv       one row per THINGS image (26,107), in the image-number order of the public
                          Kramer et al. (2023) file:
        image_nr                       the public image number (1 to 26,107); the image-set
                                       experiments store it as original_index
        filename                       e.g. aardvark_01b.jpg
        rel_path                       <concept>/<filename>, the path inside the THINGS image folder
        concept                        concept name
        concept_number_things_order    row of the concept in unique_id.txt and in every shared concept
                                       table (THINGS order, 1 to 1,854)
        concept_number_ascii           Kramer et al.'s concept number (alphabetical order; the
                                       concept_nr of things_image_mapping.csv)
        kramer_cr                      the public per-image memorability score (hit rate minus false
                                       alarm rate), copied as text from the public file; empty (NaN)
                                       for goalpost_08s.jpg, as in the public file
        embedding_row                  0-based row of this image in every deep-network embedding file
                                       of shared/embeddings_things/ (the order of the image list of the
                                       extraction, image_order.csv, which sorts by file name and differs
                                       from image_nr order in 29 rows around 'ice' / 'ice-cream_cone')

  concept_categories.csv  one row per concept (THINGS order): concept, category_number,
                          category_label, taken by concept name from the public
                          Concept_to_category_linking.csv (osf.io/5a7z6), the table module A5 reads
                          (SI Appendix, Section 5); concept_table.csv keeps the categories of
                          concept_info_human.mat.

Checks: the image list equals ImageConceptList.mat (file names, image numbers, THINGS concept
numbers, concept names); the extraction image list holds the same 26,107 files; concept numbers agree with
unique_id.txt and things_image_mapping.csv; 1,854 concepts in the category file with 23 named
categories plus NA.

INPUTS (the public files are in external/kramer/ after external/download_external.py; the other two
        are not included)
    THINGS_Memorability_Scores.csv, Concept_to_category_linking.csv   (public, Kramer et al. 2023)
    ImageConceptList.mat        (the image list of the image-set analyses)
    image_order.csv             (the image list of the THINGS embedding extraction, in embedding row
                                 order; only filename, concept, rel_path are read)
    shared/things/unique_id.txt, shared/things/things_image_mapping.csv (included)

OUTPUTS  shared/things/things_images.csv, shared/things/concept_categories.csv

RUN (from the repository root)
    python shared/things/provenance/build_things_image_tables.py <THINGS_Memorability_Scores.csv> \
        <Concept_to_category_linking.csv> <ImageConceptList.mat> <image_order.csv>
"""
import csv
import os
import sys

import numpy as np
import scipy.io as sio

THINGS_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    scores_file, category_file, image_concept_list_file, image_order_file = sys.argv[1:5]

    with open(os.path.join(THINGS_FOLDER, 'unique_id.txt')) as handle:
        things_order = [line.strip() for line in handle if line.strip()]
    things_number_of = {}
    for position, concept in enumerate(things_order):
        things_number_of[concept] = position + 1

    # ---- the public per-image scores (kept as text) ----
    with open(scores_file, newline='') as handle:
        score_rows = list(csv.DictReader(handle))
    assert len(score_rows) == 26107

    # ---- Kramer et al.'s alphabetical concept numbers, from things_image_mapping.csv ----
    with open(os.path.join(THINGS_FOLDER, 'things_image_mapping.csv'), newline='') as handle:
        mapping_rows = list(csv.DictReader(handle))
    ascii_number_of_image = {}
    for row in mapping_rows:
        ascii_number_of_image[int(row['image_nr'])] = int(row['concept_nr'])

    # ---- the image list of the image-set analyses ----
    image_list = sio.loadmat(image_concept_list_file, squeeze_me=True, struct_as_record=False)['imageList']
    list_files = [str(name) for name in image_list.filename]
    list_numbers = np.asarray(image_list.imageNumber, dtype=int)
    list_concept_numbers = np.asarray(image_list.conceptNumber, dtype=int)
    list_concepts = [str(name) for name in image_list.concepts]

    # ---- the embedding row order of the extraction image list (file names only) ----
    with open(image_order_file, newline='') as handle:
        order_rows = list(csv.DictReader(handle))
    assert len(order_rows) == 26107
    embedding_row_of = {}
    rel_path_of = {}
    for row_index, row in enumerate(order_rows):
        embedding_row_of[row['filename']] = row_index
        rel_path_of[row['filename']] = row['rel_path']
        assert row['rel_path'] == row['concept'] + '/' + row['filename']

    output_rows = []
    for index, row in enumerate(score_rows):
        image_number = int(row['image_nr'])
        filename = row['image_name']
        concept = row['file_path'].split('/')[1]
        # checks against the image list of the analyses
        assert list_files[index] == filename
        assert list_numbers[index] == image_number
        assert list_concepts[index] == concept
        assert list_concept_numbers[index] == things_number_of[concept]
        output_rows.append([image_number, filename, rel_path_of[filename], concept,
                            things_number_of[concept], ascii_number_of_image[image_number],
                            row['cr'], embedding_row_of[filename]])
    assert sorted(embedding_row_of.values()) == list(range(26107))
    rows_out_of_order = sum(1 for row in output_rows if row[7] != row[0] - 1)
    print('images: %d; rows where the embedding order differs from image_nr order: %d'
          % (len(output_rows), rows_out_of_order))
    missing_scores = [row[1] for row in output_rows if row[6] in ('', 'NaN', 'nan')]
    print('images without a public score: %s' % missing_scores)

    with open(os.path.join(THINGS_FOLDER, 'things_images.csv'), 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image_nr', 'filename', 'rel_path', 'concept', 'concept_number_things_order',
                         'concept_number_ascii', 'kramer_cr', 'embedding_row'])
        for row in output_rows:
            if row[6] == '':
                row[6] = 'NaN'
            writer.writerow(row)

    # ---- concept categories, joined by NAME ----
    with open(category_file, newline='', encoding='utf-8-sig') as handle:    # the file starts with a BOM
        category_rows = list(csv.DictReader(handle))
    category_of = {}
    for row in category_rows:
        category_of[row['concept']] = (int(row['category']), row['category_label'])
    assert len(category_of) == 1854
    assert set(category_of) == set(things_order)
    named = sorted(set(label for number, label in category_of.values() if label != 'NA'))
    print('concepts in the category file: %d; named categories: %d (+ NA); NA concepts: %d'
          % (len(category_of), len(named), sum(1 for n, label in category_of.values() if label == 'NA')))
    assert len(named) == 23
    with open(os.path.join(THINGS_FOLDER, 'concept_categories.csv'), 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['concept_number', 'concept', 'category_number', 'category_label'])
        for position, concept in enumerate(things_order):
            number, label = category_of[concept]
            writer.writerow([position + 1, concept, number, label])
    print('wrote things_images.csv and concept_categories.csv')


if __name__ == '__main__':
    main()
