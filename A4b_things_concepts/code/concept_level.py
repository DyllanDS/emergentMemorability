"""
concept_level.py: crowding among the 1,854 THINGS concepts in the deep-network spaces, and how it
relates to the memorability of each concept (SI Appendix, Section 4, THINGS concepts).

WHAT IT DOES
  THINGS has 1,854 concepts with a median of 13 photographs each. The nearest neighbors of a
  photograph are mostly other photographs of the same object, so the analysis works at the level
  of concepts:
  1. Every photograph is described in 14 deep-network spaces (an early and a late layer of AlexNet,
     CLIP ViT-B/32, CORnet-S, DINOv3 ViT-B/16, InceptionV3, ResNet18 and VGG16-BN; 512 dimensions
     each; shared/embedding_extraction/README.md). Every column is standardized over the
     26,107 photographs and the photographs of each concept are averaged, which gives one centroid
     of 14 x 512 = 7,168 numbers per concept. These centroids are the files in
     shared/embeddings_things/concept_centroids/ (made from the photograph-level embeddings by
     shared/embedding_extraction/make_concept_centroids.py).
  2. The distance between two concepts is 1 minus the correlation of their centroids, put on a
     standard scale: minus the mean and divided by the SD of the distances among 1,000 randomly
     chosen concepts (SEED 0).
  3. The crowding of a concept is minus the mean scaled distance to its 5 nearest other concepts
     (higher = more crowded).
  4. The memorability of a concept is the mean over its photographs of the hit rate, the false
     alarm rate and corrected recognition (hit rate minus false alarm rate) in the THINGS
     memorability data of Kramer et al. (2023); d′ is z(mean hit rate) minus z(mean false alarm
     rate), with both rates clipped to 0.001 and 0.999.
  5. Spearman correlations across the 1,854 concepts between crowding and each memorability measure.

  Step 4 needs the hit and false alarm counts of every photograph, which come from the Kramer et
  al. trial data and are not included (EXTERNAL_INPUTS.md M2). Without them this script runs steps
  1 to 3 and writes the crowding of every concept; the four correlations printed in the paper are
  then read from the saved results (data/provided/, by check_provided.py). When the count table is
  in external/kramer_trials/, the script runs every step and writes the report.

INPUTS
  shared/embeddings_things/concept_centroids/<network>_<early or late>.npz   14 files, 1,854 x 512
  shared/things/things_images.csv     image number, file name, concept and embedding row of
                                      every photograph
  external/kramer_trials/things_per_image_counts.csv   only when present (EXTERNAL_INPUTS.md M2):
                                      image_nr, n_old, hits, n_new, false_alarms (the per-image
                                      counts described in the module README)

OUTPUTS
  outputs/concept_crowding.csv              the crowding of every concept (always written)
  outputs/regenerated/concept_level.txt     the report (only with M2)
  outputs/regenerated/concept_level.json    the correlations behind the report (only with M2)
  The concept means of the memorability measures stay in memory and are not written to disk,
  because they derive from the Kramer et al. trial data, which are not public.

SEED     0 (the 1,000 concepts that set the distance scale: numpy default_rng(0).choice, without
         replacement)
RUNTIME  about 5 seconds
RUN      from the repository root: python A4b_things_concepts/code/concept_level.py (the module's
         entry point, code/run_module.py, runs it and also writes numbers.csv and tests.csv)
"""
import csv
import json
import os
import sys

sys.dont_write_bytecode = True   # keep the release free of __pycache__ folders

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import concept_tables   # noqa: E402
import release_paths    # noqa: E402

# ---- settings ----
CONCEPTS_PER_CHUNK = 512        # rows of the distance matrix computed at once
SEED = 0                        # picks the 1,000 concepts that set the distance scale
SCALE_SAMPLE_SIZE = 1000
RATE_CLIP = 1e-3                # rates clipped to [0.001, 0.999] before z

# The neighborhood size of the crowding measure (SI Appendix, Section 4: "each THINGS concept's five
# nearest concepts").
REPORTED_NEIGHBORHOOD_SIZE = 5

# The 14 spaces, in sorted file-name order (the column order of the concept matrix).
NETWORK_LAYERS = ['alexnet_early', 'alexnet_late', 'clip_early', 'clip_late', 'cornet_early', 'cornet_late',
                  'dinov3_early', 'dinov3_late', 'inception_early', 'inception_late', 'resnet18_early',
                  'resnet18_late', 'vgg16_early', 'vgg16_late']

# The memorability measures, in the column order of the report and json.
MEMORABILITY_MEASURES = ['CR', 'hit rate', 'FA rate', 'd-prime']

# The input that turns on the memorability side (EXTERNAL_INPUTS.md M2).
COUNT_TABLE = release_paths.Input(
    'M2',
    'the per-image THINGS count table things_per_image_counts.csv (image_nr, n_old, hits, n_new, '
    'false_alarms) from the Kramer et al. (2023) trial data',
    'external/kramer_trials/things_per_image_counts.csv')
TRIAL_FILE_RELATIVE = 'external/kramer_trials/imbysub-all-11-16-21.mat'


def relative_to_release(path):
    """A path as printed in logs: relative to the repository root when it is inside the release."""
    root = release_paths.release_root()
    absolute = os.path.abspath(path)
    if absolute.startswith(root + os.sep):
        return os.path.relpath(absolute, root)
    return absolute


# ------------------------------------------------------------------------------------------------
# Step 1: the concept centroids
# ------------------------------------------------------------------------------------------------
def load_concept_centroids():
    """Reads the 14 concept-centroid files and places them side by side.

    Returns
      concept_matrix     1,854 x 7,168 (float32): one row per concept, 14 blocks of 512 columns in
                         the order of NETWORK_LAYERS (sorted file names)
      concept_names      the 1,854 concept names in the row order (sorted by name)
      photograph_counts  the number of photographs averaged in each concept
    """
    folder = release_paths.shared_path('embeddings_things', 'concept_centroids')
    expected_files = [layer + '.npz' for layer in NETWORK_LAYERS]
    files_present = sorted(name for name in os.listdir(folder) if name.endswith('.npz'))
    if files_present != expected_files:
        raise SystemExit('shared/embeddings_things/concept_centroids/ must hold exactly the 14 files %s; found %s'
                         % (', '.join(expected_files), ', '.join(files_present)))
    blocks = []
    concept_names = None
    photograph_counts = None
    for layer in NETWORK_LAYERS:
        loaded = np.load(os.path.join(folder, layer + '.npz'))
        centroids = loaded['centroids']
        if centroids.shape != (1854, 512) or centroids.dtype != np.float32:
            raise SystemExit('%s.npz: expected 1,854 x 512 float32 centroids, found %s %s'
                             % (layer, centroids.shape, centroids.dtype))
        if concept_names is None:
            concept_names = loaded['concepts']
            photograph_counts = loaded['image_counts']
        elif not (np.array_equal(loaded['concepts'], concept_names)
                  and np.array_equal(loaded['image_counts'], photograph_counts)):
            raise SystemExit('%s.npz lists other concepts or photograph counts than %s.npz'
                             % (layer, NETWORK_LAYERS[0]))
        blocks.append(centroids)
    concept_matrix = np.concatenate(blocks, axis=1)
    return concept_matrix, concept_names, photograph_counts


# ------------------------------------------------------------------------------------------------
# Steps 2 and 3: distances between concepts and crowding
# ------------------------------------------------------------------------------------------------
def compute_crowding(concept_matrix):
    """Crowding of every concept among all 1,854 concepts: minus the mean scaled distance to its
    REPORTED_NEIGHBORHOOD_SIZE (5) nearest other concepts. Returns an array of 1,854 (float32).
    The matrix products below are the one vectorized step (512 rows of the 1,854 x 1,854 distance
    matrix at a time).
    """
    number_of_concepts = len(concept_matrix)

    # Each centroid minus its own mean, divided by its own length: the product of two such rows is
    # the correlation of the two centroids.
    unit_rows = concept_matrix - concept_matrix.mean(1, keepdims=True)
    unit_rows /= np.maximum(np.linalg.norm(unit_rows, axis=1, keepdims=True), 1e-8)

    # The distance scale: mean and SD of the correlation distances (1 - correlation) among 1,000
    # randomly chosen concepts. This is the only random step (SEED 0).
    scale_sample = np.random.default_rng(SEED).choice(number_of_concepts,
                                                      size=min(SCALE_SAMPLE_SIZE, number_of_concepts),
                                                      replace=False)
    sample_correlations = unit_rows[scale_sample] @ unit_rows[scale_sample].T
    sample_distances = 1.0 - sample_correlations[np.triu_indices(len(scale_sample), 1)]
    distance_mean = float(sample_distances.mean())
    distance_sd = float(sample_distances.std() + 1e-8)

    crowding = np.zeros(number_of_concepts, np.float32)
    for first in range(0, number_of_concepts, CONCEPTS_PER_CHUNK):
        last = min(first + CONCEPTS_PER_CHUNK, number_of_concepts)
        # scaled distances from the concepts first .. last - 1 to every concept
        scaled_distances = (1.0 - (unit_rows[first:last] @ unit_rows.T) - distance_mean) / distance_sd
        # a concept is not its own neighbor
        scaled_distances[np.arange(last - first), np.arange(first, last)] = np.inf
        # crowding: minus the mean of the 5 smallest distances of each row
        sorted_distances = np.sort(scaled_distances, axis=1)
        crowding[first:last] = -sorted_distances[:, :REPORTED_NEIGHBORHOOD_SIZE].mean(1)
    return crowding


def write_crowding_table(path, concept_names, photograph_counts, crowding):
    """outputs/concept_crowding.csv: one row per concept (sorted by name, the analysis order).
    Columns: concept, concept_number_things_order (the row of the concept in shared/things/
    unique_id.txt), photographs, crowding_k5 (the crowding of compute_crowding). Values are written
    as the shortest decimal that reads back as the same number."""
    images = concept_tables.load_things_images(order='image_nr')
    things_number_of = {}
    for concept, number in zip(images['concept'], images['concept_number_things_order']):
        things_number_of[concept] = int(number)
    header = ['concept', 'concept_number_things_order', 'photographs', 'crowding_k%d' % REPORTED_NEIGHBORHOOD_SIZE]
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in range(len(concept_names)):
            name = str(concept_names[row])
            values = [name, things_number_of[name], int(photograph_counts[row]), repr(float(crowding[row]))]
            writer.writerow(values)


# ------------------------------------------------------------------------------------------------
# Step 4: the memorability of every concept (needs the count table, M2)
# ------------------------------------------------------------------------------------------------
def load_concept_memorability(count_table_path, concept_names):
    """Concept means of the hit rate, the false alarm rate and corrected recognition, and the concept
    d′, from the hit and false alarm counts of every photograph.

    The photographs are summed in the row order of the embedding files (embedding_row of
    shared/things/things_images.csv, sorted file names), which fixes the rounding of every concept
    mean. A photograph without old (or new) trials has no hit (or false alarm) rate and is left out
    of that mean. Returns {measure: array of 1,854} in the row order of concept_names. Nothing is
    written to disk.
    """
    counts = pd.read_csv(count_table_path)
    required_columns = ['image_nr', 'n_old', 'hits', 'n_new', 'false_alarms']
    missing_columns = [column for column in required_columns if column not in counts.columns]
    if missing_columns:
        raise SystemExit('%s lacks the columns %s' % (relative_to_release(count_table_path), ', '.join(missing_columns)))
    images = concept_tables.load_things_images(order='embedding')
    if len(counts) != len(images) or set(counts['image_nr']) != set(images['image_nr']):
        raise SystemExit('%s must have one row for each of the 26,107 THINGS images (image_nr 1 to 26,107)'
                         % relative_to_release(count_table_path))
    counts = counts.set_index('image_nr').loc[images['image_nr'].to_numpy()]

    concept_row = {}
    for row, name in enumerate(concept_names):
        concept_row[str(name)] = row
    number_of_concepts = len(concept_names)
    sum_of_rates = {'hit rate': np.zeros(number_of_concepts), 'FA rate': np.zeros(number_of_concepts),
                    'CR': np.zeros(number_of_concepts)}
    photographs_with_rate = {'hit rate': np.zeros(number_of_concepts), 'FA rate': np.zeros(number_of_concepts),
                             'CR': np.zeros(number_of_concepts)}
    concepts_of_photographs = images['concept'].to_numpy()
    old_trials = counts['n_old'].to_numpy(float)
    hits = counts['hits'].to_numpy(float)
    new_trials = counts['n_new'].to_numpy(float)
    false_alarms = counts['false_alarms'].to_numpy(float)
    for photograph in range(len(images)):
        concept = concept_row[str(concepts_of_photographs[photograph])]
        hit_rate = hits[photograph] / old_trials[photograph] if old_trials[photograph] > 0 else np.nan
        false_alarm_rate = false_alarms[photograph] / new_trials[photograph] if new_trials[photograph] > 0 else np.nan
        corrected_recognition = hit_rate - false_alarm_rate
        photograph_rates = {'hit rate': hit_rate, 'FA rate': false_alarm_rate, 'CR': corrected_recognition}
        for measure in photograph_rates:
            if np.isfinite(photograph_rates[measure]):
                sum_of_rates[measure][concept] += photograph_rates[measure]
                photographs_with_rate[measure][concept] += 1.0

    memorability = {}
    for measure in sum_of_rates:
        concept_means = np.full(number_of_concepts, np.nan)
        for concept in range(number_of_concepts):
            if photographs_with_rate[measure][concept] > 0:
                concept_means[concept] = sum_of_rates[measure][concept] / photographs_with_rate[measure][concept]
        memorability[measure] = concept_means
    # d′ of a concept: z of its mean hit rate minus z of its mean false alarm rate (rates clipped)
    z_hit = norm.ppf(np.clip(memorability['hit rate'], RATE_CLIP, 1 - RATE_CLIP))
    z_false_alarm = norm.ppf(np.clip(memorability['FA rate'], RATE_CLIP, 1 - RATE_CLIP))
    memorability['d-prime'] = z_hit - z_false_alarm
    return memorability


# ------------------------------------------------------------------------------------------------
# Step 5: the correlations and the report
# ------------------------------------------------------------------------------------------------
def spearman(crowding, memorability):
    """Spearman rho and its two-sided p over the concepts where both values exist."""
    usable = np.isfinite(crowding) & np.isfinite(memorability)
    result = spearmanr(crowding[usable], memorability[usable])
    return result.statistic, result.pvalue


def correlate_and_report(crowding, memorability, photograph_counts, report_folder):
    """The report: the Spearman correlations of crowding with the four memorability measures, in the
    format of data/provided/concept_level.txt (header and the nearest-neighbor table, here with the
    k = 5 row), written to report_folder/concept_level.txt and report_folder/concept_level.json.
    Returns the json content."""
    k = REPORTED_NEIGHBORHOOD_SIZE
    measures_in_order = []
    for measure in MEMORABILITY_MEASURES:
        measures_in_order.append(memorability[measure])

    os.makedirs(report_folder, exist_ok=True)
    report_lines = []

    def write(line=''):
        print(line, flush=True)
        report_lines.append(line)

    number_of_concepts = len(photograph_counts)
    number_of_photographs = int(photograph_counts.sum())
    write('CROWDING AMONG THINGS CONCEPTS')
    write('=' * 96)
    write(f'{number_of_photographs:,} photographs collapsed to {number_of_concepts:,} concepts '
          f'(median {int(np.median(photograph_counts))} photographs each).')
    write(f'{len(NETWORK_LAYERS) * 512:,} concatenated dimensions from {len(NETWORK_LAYERS)} network/layer '
          f'spaces, each z-scored.')
    write('Crowding is computed among concept centroids; memorability is averaged')
    write('within concept. Spearman rho throughout.')
    write()
    write('Photographs of one object lie close together in these spaces, so the')
    write('nearest neighbors of a photograph are mostly photographs of the same')
    write('object. Crowding is therefore measured among concepts, the level at')
    write('which the SPoSE embedding describes THINGS.')
    write()

    write('-' * 96)
    write('Nearest-neighbor crowding among concepts, by k')
    write('-' * 96)
    write(f"{'k':>5} {'CR':>18} {'hit rate':>18} {'FA rate':>18} {'d-prime':>18}")
    row = []
    for values in measures_in_order:
        row.append(spearman(crowding, values))
    cells = []
    for rho, p in row:
        cells.append(f'{rho:+.3f} (p={p:.1g})'.rjust(18))
    write(f'{k:>5} ' + ' '.join(cells))
    write()

    with open(os.path.join(report_folder, 'concept_level.txt'), 'w', encoding='utf-8') as handle:
        for line in report_lines:
            handle.write(line + '\n')
    # by_k[k]: the four [rho, p] pairs in the order of MEMORABILITY_MEASURES, as in the saved results
    results = {'by_k': {str(k): [(float(rho), float(p)) for rho, p in row]}}
    json_path = os.path.join(report_folder, 'concept_level.json')
    with open(json_path, 'w') as handle:
        json.dump(results, handle, indent=2)
    print('\nwrote %s' % relative_to_release(os.path.join(report_folder, 'concept_level.txt')))
    # returned as read back from the file (keys as text), the same form as the saved results
    with open(json_path) as handle:
        return json.load(handle)


# ------------------------------------------------------------------------------------------------
# The whole analysis
# ------------------------------------------------------------------------------------------------
def run(outputs_folder, count_table_path=None):
    """Runs the analysis. count_table_path: the count table (M2), or None to look for it in
    external/kramer_trials/ (provided-output mode when it is absent).

    Returns a dict with the mode ('regenerate' or 'provided'), the crowding of every concept, the
    concept names and photograph counts, the concept memorability (regenerate mode only, in memory)
    and the content of the report's json (regenerate mode only)."""
    if count_table_path is None:
        mode = release_paths.regeneration_mode([COUNT_TABLE])
        if mode == 'regenerate':
            count_table_path = os.path.join(release_paths.release_root(), COUNT_TABLE.relative_path)
        elif os.path.exists(os.path.join(release_paths.release_root(), TRIAL_FILE_RELATIVE)):
            print('The Kramer et al. trial file is in external/kramer_trials/; this module needs the per-image '
                  'count table instead (EXTERNAL_INPUTS.md M2), whose old trials include the short-lag vigilance '
                  'repeats as well as the long-lag target repeats. The code does not build that table from the '
                  'trial file (see this module\'s README).')
    else:
        mode = 'regenerate'
        print('count table given by the caller: %s' % relative_to_release(count_table_path))

    concept_matrix, concept_names, photograph_counts = load_concept_centroids()
    print('%s concepts x %s dimensions from %d spaces (%s photographs; median %d per concept)'
          % (format(concept_matrix.shape[0], ','), format(concept_matrix.shape[1], ','), len(NETWORK_LAYERS),
             format(int(photograph_counts.sum()), ','), int(np.median(photograph_counts))))

    crowding = compute_crowding(concept_matrix)
    os.makedirs(outputs_folder, exist_ok=True)
    crowding_table = os.path.join(outputs_folder, 'concept_crowding.csv')
    write_crowding_table(crowding_table, concept_names, photograph_counts, crowding)
    print('crowding at k = %d: mean %.4f, SD %.4f, from %.4f to %.4f; written to %s'
          % (REPORTED_NEIGHBORHOOD_SIZE, float(np.mean(crowding)), float(np.std(crowding, ddof=1)),
             float(np.min(crowding)), float(np.max(crowding)), relative_to_release(crowding_table)))

    result = {'mode': mode, 'concept_names': concept_names, 'photograph_counts': photograph_counts,
              'crowding': crowding, 'memorability': None, 'report': None,
              'number_of_dimensions': int(concept_matrix.shape[1])}
    if mode == 'regenerate':
        memorability = load_concept_memorability(count_table_path, concept_names)
        report_folder = os.path.join(outputs_folder, 'regenerated')
        report = correlate_and_report(crowding, memorability, photograph_counts, report_folder)
        result['memorability'] = memorability
        result['report'] = report
    return result


if __name__ == '__main__':
    # The analysis alone (crowding, and the report when the count table is present). The module's
    # entry point, code/run_module.py, also writes numbers.csv and tests.csv.
    module_paths = release_paths.module_paths(__file__, make_output_folders=False)
    run(module_paths['outputs'])
