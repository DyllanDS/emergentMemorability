"""
kramer_public.py: the public THINGS memorability data of Kramer, Hebart, Baker and Bainbridge (2023,
Science Advances; osf.io/5a7z6), as the release uses them.

The release runs without downloading anything: two tables in shared/things/, derived from the public
files, carry what the code needs (citation and source: external/README.md):
    things_images.csv       kramer_cr = the public per-image memorability score (hit rate minus
                            false alarm rate), NaN for goalpost_08s.jpg as in the public file
    concept_categories.csv  each concept's higher-level category, joined by concept name

    image_scores()                       per-image table (image_nr order) with kramer_cr
    concept_mean_scores(skip_missing)    mean kramer_cr per concept, keyed by concept name;
                                         skip_missing=False (default) gives NaN for a concept with a
                                         missing image score (goalpost), as the field
                                         memorabilityCat of shared/things/concept_info_human.mat
                                         does; True skips it
    category_of_concept()                {concept name: category label}
    verify_downloads()                   if external/kramer/ holds the two public files (from
                                         external/download_external.py), checks their md5 sums and
                                         that the derived tables agree with them

The per-trial data of Kramer et al. are not included (EXTERNAL_INPUTS.md, M1 and M2).
"""
import csv
import hashlib
import os

import numpy as np

import concept_tables

_RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KRAMER_DOWNLOAD_FOLDER = os.path.join(_RELEASE_ROOT, 'external', 'kramer')
PUBLIC_FILES = {
    'THINGS_Memorability_Scores.csv': ('https://osf.io/download/z4ve9/', 1675292, '99e6f81616675a73b3ddf92058d1cf50'),
    'Concept_to_category_linking.csv': ('https://osf.io/download/8b7hg/', 113027, '4b42d2ea23005dad9bf4bddb41aeac68'),
}


def image_scores():
    """things_images.csv (26,107 rows, public image_nr order)."""
    return concept_tables.load_things_images(order='image_nr')


def concept_mean_scores(skip_missing=False):
    """{concept name: mean of kramer_cr over the concept's images}."""
    table = image_scores()
    sums = {}
    counts = {}
    has_missing = {}
    for concept, score in zip(table['concept'], table['kramer_cr']):
        if concept not in sums:
            sums[concept] = 0.0
            counts[concept] = 0
            has_missing[concept] = False
        if np.isfinite(score):
            sums[concept] = sums[concept] + score
            counts[concept] = counts[concept] + 1
        else:
            has_missing[concept] = True
    means = {}
    for concept in sums:
        if has_missing[concept] and not skip_missing:
            means[concept] = np.nan
        else:
            means[concept] = sums[concept] / counts[concept]
    return means


def category_of_concept():
    """{concept name: category label} from the public category file, joined by concept name."""
    table = concept_tables.load_concept_categories()
    return dict(zip(table['concept'], table['category_label']))


def _md5(path):
    digest = hashlib.md5()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_downloads():
    """Cross-check of the shipped derived tables against the public files, when downloaded.
    Returns True when the downloaded files (one or both) agree, None when neither is present;
    raises when a checksum or a value disagrees (never silently)."""
    present = [name for name in PUBLIC_FILES if os.path.exists(os.path.join(KRAMER_DOWNLOAD_FOLDER, name))]
    if not present:
        print('external/kramer/ is empty: nothing to cross-check (the release does not need the download)')
        return None
    for name in present:
        url, size, expected_md5 = PUBLIC_FILES[name]
        path = os.path.join(KRAMER_DOWNLOAD_FOLDER, name)
        if _md5(path) != expected_md5:
            raise ValueError('md5 of %s differs from the public file (%s)' % (path, url))
    if 'THINGS_Memorability_Scores.csv' in present:
        shipped = image_scores()
        with open(os.path.join(KRAMER_DOWNLOAD_FOLDER, 'THINGS_Memorability_Scores.csv'), newline='') as handle:
            public = list(csv.DictReader(handle))
        if len(public) != len(shipped):
            raise ValueError('the public score file has %d rows, the shipped table %d' % (len(public), len(shipped)))
        for row, (filename, score_text) in zip(public, zip(shipped['filename'], shipped['kramer_cr'])):
            if row['image_name'] != filename:
                raise ValueError('image order differs at %s / %s' % (row['image_name'], filename))
            public_score = float(row['cr']) if row['cr'] not in ('', 'NaN') else np.nan
            both_missing = np.isnan(public_score) and np.isnan(score_text)
            if not (both_missing or public_score == score_text):
                raise ValueError('score of %s differs from the public file' % filename)
    if 'Concept_to_category_linking.csv' in present:
        shipped = category_of_concept()
        with open(os.path.join(KRAMER_DOWNLOAD_FOLDER, 'Concept_to_category_linking.csv'), newline='',
                  encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle):
                if shipped[row['concept']] != row['category_label']:
                    raise ValueError('category of %s differs from the public file' % row['concept'])
    print('public Kramer et al. files present and identical to the shipped derived tables: %s' % ', '.join(present))
    return True
