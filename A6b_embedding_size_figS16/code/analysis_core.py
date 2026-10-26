"""
analysis_core.py: the Fig. 5c and Fig. 6 statistics, computed in any embedding of the 1,854 THINGS
concepts (module A6b, SI Appendix, Section 6; Fig. S16).

WHAT IT DOES
Given an embedding (one row per THINGS concept in THINGS order, one column per feature), the
functions below compute the paper's feature-level statistics in that embedding's own features.

  Fig. 5c (all of THINGS)
    1. similarity of two concepts = the Pearson correlation of their feature rows
    2. kernel similarity = normpdf(similarity, 1, 2): a Gaussian with standard deviation 2 centered
       on a similarity of 1, so near-identical concepts count the most
    3. summed similarity of concept i = the sum over the other concepts j of
       kernel(i, j) x (number of THINGS images of j) / (number of THINGS images minus one);
       concept i's own image count is reduced by one (an image does not compete with itself) and
       its own kernel term is left out
    4. for every feature: its Spearman correlation with concept memorability (Kramer et al. 2023:
       hit rate minus false alarm rate of the concept's images; 1,853 concepts have a score) and
       its Spearman correlation with summed similarity (all 1,854 concepts)
    5. the Fig. 5c statistic: the Pearson correlation, across the features, between the two
       per-feature correlations of step 4, with its two-sided parametric p (t test on n - 2
       degrees of freedom, n = the number of features)

  Fig. 6c to 6h (the maintained and changed image sets of the image-set experiments)
    The same per-feature correlations within each image set. Summed similarity is weighted by the
    number of images of each concept in the set (the concept's own count reduced by one; concepts
    not in the set are left out), and memorability is the concept's hit rate minus false alarm
    rate in the experiment (extract_behavior.py).
      6c / 6f  every feature's memorability correlation in all of THINGS against its memorability
               correlation in the maintained / changed set
      6d / 6g  every feature's summed-similarity correlation against its memorability correlation,
               within the maintained / changed set
      6e / 6h  the same as 6d / 6g in the monkey IT space (the 49 principal components of the
               monkey IT responses, all 49 as in Fig. 6). The monkey space is not derived from the
               human similarity embedding, so 6e and 6h are the same for every embedding.

INPUTS (all inside the release; read by the loaders at the end of this file)
  shared/things/concept_table.csv               concept memorability (kramer_concept_cr) and the
                                                number of THINGS images of each concept
                                                (concept_image_count_as_stored), THINGS order
  shared/things/spose_embedding_49d_sorted.txt  the published 49-feature SPoSE embedding
  shared/things/monkey_features.csv             the monkey IT component scores (49 per concept)
  06_image_set_experiments_fig6_fig7/data/sampled_image_sets.mat
                                                the number of images of each concept in the
                                                maintained and changed image sets
  outputs/image_set_concept_memorability.csv    each concept's memorability in the two image sets
                                                (written by extract_behavior.py, step 1)
  data/refits/spose_*.txt                       the refits (one embedding per file)

OUTPUTS
  none; run_refit_sweep.py and stats_figS16.py call these functions (gather_all_fits.py compares
  the results with the saved run in data/paper_run/all_fits.csv on every run)
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import rankdata
from scipy.stats import t as t_distribution

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True     # no __pycache__ folders inside the release
sys.path.insert(0, os.path.join(CODE_FOLDER, '..', '..', 'shared', 'python'))
import release_paths      # noqa: E402  (shared helper: release folders)
import concept_tables   # noqa: E402  (shared helper: the THINGS concept tables)

# The kernel of summed similarity in Figs. 5 and 6: normpdf(similarity, 1, 2).
KERNEL_CENTRE = 1.0
KERNEL_SD = 2.0
SQUARE_ROOT_OF_TWO_PI = np.sqrt(2.0 * np.pi)

# The two image sets of Fig. 6 and their names inside sampled_image_sets.mat.
IMAGE_SETS = ('maintained', 'changed')
IMAGE_SET_STRUCT_NAMES = {'maintained': 'representativeDataset', 'changed': 'flipMemorabilityDataset'}
IMAGE_SET_FILE = os.path.join(release_paths.find_module('06'), 'data', 'sampled_image_sets.mat')
IMAGE_SET_MEMORABILITY_FILE = 'image_set_concept_memorability.csv'     # written by extract_behavior.py

# A weight counts as "used" by a feature when it is at least this large (Hebart et al.'s
# criterion, also used by the refits to decide which features survive the sparsity penalty).
WEIGHT_THRESHOLD = 0.1


# ----------------------------------------------------------------------------------- correlations
def zscore_columns(table):
    """Each column minus its mean, divided by its standard deviation (population SD). A column with
    no spread becomes all zeros, so its correlation with anything is 0."""
    centred = table - table.mean(0, keepdims=True)
    spread = centred.std(0, ddof=0, keepdims=True)
    spread[spread == 0] = np.inf
    return centred / spread


def pearson_with_each_column(values, table):
    """The Pearson correlation of one variable (one value per concept) with every column of a table
    (concepts x features): one correlation per feature, the mean product of the z-scores."""
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    table = np.asarray(table, dtype=float)
    return (zscore_columns(values) * zscore_columns(table)).mean(0)


def rank_columns(table):
    """The rank of every value within its column (1 = smallest; tied values share their average
    rank). The table is transposed first so that each column is ranked as one contiguous row, which
    gives the same ranks about twice as fast. The ranks are returned column by column in memory; this
    layout fixes the order in which later sums add up, and so the last digit of every result."""
    table_by_rows = np.ascontiguousarray(np.asarray(table, dtype=float).T)
    ranks_by_rows = rankdata(table_by_rows, axis=1)
    return ranks_by_rows.T


def spearman_with_each_column(values, table):
    """The Spearman correlation of one variable with every column of a table: the Pearson correlation
    of the ranks (tied values share their average rank; every column is ranked on its own)."""
    value_ranks = rankdata(values)
    table_ranks = rank_columns(table)
    return pearson_with_each_column(value_ranks, table_ranks)


def spearman_over_rows_with_values(values, table):
    """spearman_with_each_column over the rows (concepts) where values is not missing."""
    has_value = np.isfinite(values)
    return spearman_with_each_column(values[has_value], table[has_value])


def pearson_r_and_p(x, y):
    """The Pearson correlation of two variables over the pairs where both are present, with its
    two-sided parametric p: t = r sqrt((n - 2) / (1 - r^2)) on n - 2 degrees of freedom.
    Returns (r, p, n)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    both_present = np.isfinite(x) & np.isfinite(y)
    x = x[both_present]
    y = y[both_present]
    n = x.size
    if n < 3:
        return np.nan, np.nan, n
    r = float(np.corrcoef(x, y)[0, 1])
    if abs(r) >= 1.0:
        return r, 0.0, n
    t_value = r * np.sqrt((n - 2) / (1 - r ** 2))
    p = 2 * t_distribution.sf(abs(t_value), n - 2)
    return r, float(p), n


# ------------------------------------------------------------------------------ summed similarity
def concept_similarity(embedding):
    """The similarity of every pair of concepts: the Pearson correlation of their feature rows
    (1,854 x 1,854)."""
    return np.corrcoef(np.asarray(embedding, dtype=float))


def kernel_similarity(similarity, sd=KERNEL_SD):
    """normpdf(similarity, 1, sd), the same as MATLAB's normpdf(similarity, 1, sd)."""
    return np.exp(-0.5 * ((similarity - KERNEL_CENTRE) / sd) ** 2) / (sd * SQUARE_ROOT_OF_TWO_PI)


def summed_similarity_full_things(kernel, image_counts):
    """Summed similarity of each concept to all of THINGS (Fig. 5): for concept i, the kernel
    similarity to every other concept j, weighted by j's share of the THINGS images, where concept
    i's own image count is reduced by one and its own kernel term is left out."""
    number_of_concepts = kernel.shape[0]
    summed_similarity = np.empty(number_of_concepts)
    for concept in range(number_of_concepts):
        counts = image_counts.copy()
        counts[concept] -= 1.0
        weights = counts / counts.sum()
        kernel_row = kernel[concept].copy()
        kernel_row[concept] = 0.0
        summed_similarity[concept] = kernel_row @ weights
    return summed_similarity


def summed_similarity_in_image_set(kernel, set_counts):
    """Summed similarity of each concept within one image set (Fig. 6): for every concept with
    images in the set, the kernel similarity to every concept j (itself included) weighted by j's
    share of the set's images, with the concept's own count reduced by one. Concepts that are not
    in the set get no value (NaN)."""
    number_of_concepts = kernel.shape[0]
    summed_similarity = np.full(number_of_concepts, np.nan)
    for concept in range(number_of_concepts):
        if set_counts[concept] <= 0:
            continue
        counts = set_counts.copy()
        counts[concept] -= 1
        weights = counts / counts.sum()
        summed_similarity[concept] = kernel[concept] @ weights
    return summed_similarity


# ------------------------------------------------------------------------------- the two analyses
def fig5c_statistic(embedding, memorability, image_counts):
    """The Fig. 5c analysis in one embedding. Returns a dict with
        number_of_features, r (Pearson across features), p (two-sided parametric), number_of_points
        (features with both correlations), memorability_correlations and similarity_correlations
        (one per feature), summed_similarity (one per concept) and
        rho_summed_similarity_memorability (Spearman correlation of summed similarity with
        memorability across the 1,853 concepts with a score)."""
    embedding = np.asarray(embedding, dtype=float)
    kernel = kernel_similarity(concept_similarity(embedding))
    summed_similarity = summed_similarity_full_things(kernel, image_counts)

    has_score = np.isfinite(memorability)
    memorability_correlations = spearman_with_each_column(memorability[has_score], embedding[has_score])
    similarity_correlations = spearman_with_each_column(summed_similarity, embedding)
    r, p, number_of_points = pearson_r_and_p(similarity_correlations, memorability_correlations)

    concept_level = spearman_with_each_column(summed_similarity[has_score],
                                              memorability[has_score].reshape(-1, 1))
    return dict(number_of_features=embedding.shape[1], r=r, p=p, number_of_points=number_of_points,
                memorability_correlations=memorability_correlations,
                similarity_correlations=similarity_correlations,
                summed_similarity=summed_similarity,
                rho_summed_similarity_memorability=float(concept_level[0]))


def fig6_statistics(embedding, monkey_features, memorability, set_counts, set_memorability):
    """The six Fig. 6 correlations in one embedding. Returns a dict of (r, two-sided parametric p):
        maintained_vs_things (6c), maintained_human (6d), maintained_monkey (6e),
        changed_vs_things (6f), changed_human (6g), changed_monkey (6h)."""
    embedding = np.asarray(embedding, dtype=float)
    kernel_human = kernel_similarity(concept_similarity(embedding))
    kernel_monkey = kernel_similarity(concept_similarity(monkey_features))

    has_score = np.isfinite(memorability)
    memorability_correlations_things = spearman_with_each_column(memorability[has_score],
                                                                 embedding[has_score])
    results = {}
    for set_name in IMAGE_SETS:
        summed_similarity_human = summed_similarity_in_image_set(kernel_human, set_counts[set_name])
        summed_similarity_monkey = summed_similarity_in_image_set(kernel_monkey, set_counts[set_name])
        concept_memorability_in_set = set_memorability[set_name]

        memorability_correlations_set = spearman_over_rows_with_values(concept_memorability_in_set, embedding)
        similarity_correlations_set = spearman_over_rows_with_values(summed_similarity_human, embedding)
        memorability_correlations_monkey = spearman_over_rows_with_values(concept_memorability_in_set,
                                                                          monkey_features)
        similarity_correlations_monkey = spearman_over_rows_with_values(summed_similarity_monkey,
                                                                        monkey_features)

        results[set_name + '_vs_things'] = pearson_r_and_p(memorability_correlations_things,
                                                           memorability_correlations_set)[:2]
        results[set_name + '_human'] = pearson_r_and_p(similarity_correlations_set,
                                                       memorability_correlations_set)[:2]
        results[set_name + '_monkey'] = pearson_r_and_p(similarity_correlations_monkey,
                                                        memorability_correlations_monkey)[:2]
    return results


def share_of_small_weights(embedding):
    """The 'sparsity' of an embedding: the share of its weights below 0.1 (0.72 in the published
    49-feature embedding)."""
    return float((np.asarray(embedding, dtype=float) < WEIGHT_THRESHOLD).mean())


# ------------------------------------------------------------------------------------------ inputs
def load_things_inputs():
    """The concept-level inputs, in THINGS order (1,854 concepts): concept names, concept
    memorability (NaN for goalpost, whose only image has no score), the number of THINGS images of
    each concept (column concept_image_count_as_stored), the published 49-feature embedding and the
    monkey IT features."""
    table = concept_tables.load_concept_table()
    published_embedding = np.loadtxt(release_paths.shared_path('things', 'spose_embedding_49d_sorted.txt'))
    inputs = dict(concepts=list(table['concept']),
                  concept_numbers=table['concept_number'].to_numpy(dtype=int),
                  memorability=table['kramer_concept_cr'].to_numpy(dtype=float),
                  image_counts=table['concept_image_count_as_stored'].to_numpy(dtype=float),
                  published_embedding=published_embedding,
                  monkey_features=concept_tables.load_monkey_features())
    return inputs


def load_image_set_counts():
    """The number of images of each THINGS concept in the maintained and changed image sets
    (numImagesForConcept of the sampled sets; 1,600 images per set), THINGS order."""
    stored = scipy.io.loadmat(IMAGE_SET_FILE, squeeze_me=True, struct_as_record=False)
    counts = {}
    for set_name in IMAGE_SETS:
        image_set = stored[IMAGE_SET_STRUCT_NAMES[set_name]]
        counts[set_name] = np.asarray(image_set.numImagesForConcept, dtype=float)
    return counts


def load_image_set_memorability(outputs_folder, concepts):
    """The memorability of each concept in the maintained and changed sets, as written by
    extract_behavior.py (outputs/image_set_concept_memorability.csv), THINGS order."""
    path = os.path.join(outputs_folder, IMAGE_SET_MEMORABILITY_FILE)
    if not os.path.exists(path):
        raise SystemExit('%s is missing: run A6b_embedding_size_figS16/code/extract_behavior.py first'
                         % IMAGE_SET_MEMORABILITY_FILE)
    table = pd.read_csv(path, keep_default_na=False, na_values=['NaN'], float_precision='round_trip')
    if list(table['concept']) != list(concepts):
        raise SystemExit('%s is not in THINGS order' % IMAGE_SET_MEMORABILITY_FILE)
    memorability = {}
    for set_name in IMAGE_SETS:
        memorability[set_name] = table[set_name + '_memorability'].to_numpy(dtype=float)
    return memorability


def load_embedding(path, number_of_concepts=1854):
    """One embedding text file (one row per concept, one column per feature) as a 2-D array. A refit
    whose penalty removed every feature is saved as empty lines and comes back with 0 columns."""
    with open(path) as handle:
        has_numbers = any(line.strip() for line in handle)
    if not has_numbers:
        return np.zeros((number_of_concepts, 0))
    embedding = np.loadtxt(path)
    if embedding.ndim == 1:
        embedding = embedding[:, None]
    return embedding
