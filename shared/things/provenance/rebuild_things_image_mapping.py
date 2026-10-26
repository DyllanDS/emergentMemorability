"""
rebuild_things_image_mapping.py

WHAT IT DOES
Writes shared/things/things_image_mapping.csv, the image-number table of the THINGS trial analyses
(modules A3a and A5; columns new_fname, orig_fname, concept_nr, exemplar_nr, image_nr), from the
public THINGS memorability file of Kramer et al. (2023), THINGS_Memorability_Scores.csv
(osf.io/5a7z6). Every column follows from the public file:
    image_nr     the image number of the public file (1 to 26,107)
    orig_fname   the public file_path column ('images/<concept>/<file>.jpg')
    new_fname    'https://martin-hebart.com/turk/images500/<image_nr, five digits>.jpg'
    concept_nr   the concept's position in the order concepts first appear in the public file
                 (alphabetical by concept name, the ASCII order of Kramer et al.'s files; this is
                 NOT the THINGS order of unique_id.txt; see the concept-order note in shared/README.md)
    exemplar_nr  the image's position within its concept, in file order
The file is written with Windows line ends and no final line end, the format of the table in
shared/things/.

INPUT   THINGS_Memorability_Scores.csv (public; external/download_external.py puts it in external/kramer/)
OUTPUT  things_image_mapping.csv

RUN (from the repository root; both arguments optional)
    python shared/things/provenance/rebuild_things_image_mapping.py \
        [external/kramer/THINGS_Memorability_Scores.csv] [shared/things/things_image_mapping.csv]
"""
import csv
import os
import sys

RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_INPUT = os.path.join(RELEASE_ROOT, 'external', 'kramer', 'THINGS_Memorability_Scores.csv')
DEFAULT_OUTPUT = os.path.join(RELEASE_ROOT, 'shared', 'things', 'things_image_mapping.csv')


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    output_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    if not os.path.exists(input_file):
        raise SystemExit('Public scores file not found: %s\n(run python external/download_external.py first)'
                         % input_file)

    with open(input_file, newline='') as handle:
        rows = list(csv.DictReader(handle))

    concept_number_of = {}
    images_seen_in_concept = {}
    lines = ['new_fname,orig_fname,concept_nr,exemplar_nr,image_nr']
    for row in rows:
        image_number = int(row['image_nr'])
        file_path = row['file_path']                      # images/<concept>/<file>.jpg
        concept = file_path.split('/')[1]
        if concept not in concept_number_of:
            concept_number_of[concept] = len(concept_number_of) + 1
            images_seen_in_concept[concept] = 0
        images_seen_in_concept[concept] = images_seen_in_concept[concept] + 1
        new_name = 'https://martin-hebart.com/turk/images500/%05d.jpg' % image_number
        lines.append('%s,%s,%d,%d,%d' % (new_name, file_path, concept_number_of[concept],
                                         images_seen_in_concept[concept], image_number))

    with open(output_file, 'w', newline='') as handle:
        handle.write('\r\n'.join(lines))                  # Windows line ends, no final line end
    print('wrote %s: %d images, %d concepts' % (output_file, len(rows), len(concept_number_of)))


if __name__ == '__main__':
    main()
