"""
concept_tables.py: read the THINGS concept and image tables of shared/things/.

    load_concept_table()        concept_table.csv as a pandas table (1,854 concepts, THINGS order):
                                concept_number, concept, category_number, category_label (the
                                categories of concept_info_human.mat),
                                kramer_concept_cr, concept_image_count_as_stored,
                                weighted_similarity_full_things_human / _monkey, feature_01..feature_49
    spose_features(table)       the 1,854 x 49 SPoSE feature matrix (numpy, THINGS order)
    load_monkey_features()      the 1,854 x 49 monkey IT component scores (numpy, THINGS order)
    concept_similarity(F)       the concept-by-concept similarity used by every summed-similarity
                                measure: the Pearson correlation of the feature rows
                                (= corrcoef(features') in MATLAB, the thingsSimRSM that
                                shared/matlab/load_concept_info.m adds)
    load_things_images(order)   things_images.csv; order='image_nr' (default, the public numbering)
                                or order='embedding' (the row order of the embedding files)
    load_concept_categories()   concept_categories.csv (the public category file's labels, joined by
                                concept name; module A5, SI Appendix, Section 5, reads this table)

Notes
- The numbers in the CSV tables are written with Python's repr(), so they read back as exactly the
  doubles they were written from, PROVIDED the reader parses decimals exactly: pandas needs
  float_precision='round_trip' (its default parser is off by up to 4e-16 in 80 percent of the feature
  values); MATLAB readtable and Python float() are exact. Every loader here passes round_trip.
- concept_similarity() computed here with numpy differs from MATLAB's corrcoef by at most about
  1e-15 (rounding); the MATLAB loader (shared/matlab/load_concept_info.m) recomputes it in MATLAB.
"""
import os

import numpy as np
import pandas as pd

_THINGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'things')
NUMBER_OF_FEATURES = 49


def things_file(name):
    """Path of a file in shared/things/."""
    return os.path.join(_THINGS, name)


def load_concept_table():
    """concept_table.csv (1,854 rows, THINGS order)."""
    table = pd.read_csv(things_file('concept_table.csv'), keep_default_na=False, na_values=['NaN'],
                        float_precision='round_trip')
    assert len(table) == 1854
    return table


def spose_features(table=None):
    """The 49 SPoSE feature values of each concept as a 1,854 x 49 array (THINGS order)."""
    if table is None:
        table = load_concept_table()
    columns = ['feature_%02d' % (number + 1) for number in range(NUMBER_OF_FEATURES)]
    return table[columns].to_numpy(dtype=float)


def load_monkey_features():
    """The 49 monkey IT principal-component scores of each concept, 1,854 x 49 (THINGS order):
    the features of shared/things/concept_info_monkey.mat."""
    table = pd.read_csv(things_file('monkey_features.csv'), float_precision='round_trip')
    columns = ['pc_%02d' % (number + 1) for number in range(NUMBER_OF_FEATURES)]
    return table[columns].to_numpy(dtype=float)


def concept_similarity(features):
    """Concept-by-concept similarity: Pearson correlation between the feature rows."""
    return np.corrcoef(np.asarray(features, dtype=float))


def load_things_images(order='image_nr'):
    """things_images.csv. order='image_nr' keeps the public numbering (row i = image_nr i + 1);
    order='embedding' sorts the rows into the order of the embedding files (embedding_row 0, 1, ...).
    kramer_cr is NaN for goalpost_08s.jpg, as in the public file."""
    table = pd.read_csv(things_file('things_images.csv'), keep_default_na=False, na_values=['NaN'],
                        float_precision='round_trip')
    assert len(table) == 26107
    if order == 'embedding':
        table = table.sort_values('embedding_row').reset_index(drop=True)
        assert (table['embedding_row'].to_numpy() == np.arange(len(table))).all()
    elif order != 'image_nr':
        raise ValueError("order must be 'image_nr' or 'embedding'")
    return table


def load_concept_categories():
    """concept_categories.csv: concept, category_number, category_label from the public
    Concept_to_category_linking.csv, joined by concept NAME (1,854 rows, THINGS order).
    'NA' is a label here (concepts without a category), not a missing value."""
    table = pd.read_csv(things_file('concept_categories.csv'), keep_default_na=False)
    assert len(table) == 1854
    return table
