"""
stats_figS15.py: the numbers of SI Appendix, Section 6 (Faces and Scenes), Fig. S15 and the main-text
sentence that cites them, and the uncertainty statistics T43 to T45.

What it does
    1. Counts the images each analysis ran over (ledger ADD-008) and lists the five held-out fold
       correlations behind each Fig. S15a value, with the ridge penalty alpha and the kernel width tau
       chosen in each fold (ADD-009).
    2. Writes outputs/numbers.csv: one row per ledger item of this module (20) with the value this
       code produced. Faces: the values are read from the saved faces results in data/provided_faces/,
       because the face database may not be redistributed (EXTERNAL_INPUTS.md, M4); when the database
       archives are in external/bainbridge_faces/, the faces are recomputed and these rows use the
       recomputed values. Scenes: the values come from this run (outputs/scenes/). The 95% intervals
       are those of outputs/intervals_figS15.csv (Fisher z, the error bars of Fig. S15).
    3. Writes outputs/tests.csv (the conventions of statistics/README.md), T43 and T44 from the saved
       runs (data/provided_faces/, data/paper_run_scenes/):
       T43  Fig. S15a, held-out correlations. Each value in the figure is the mean of five held-out
            fold correlations; its p ("all p < 0.001") is the largest single-fold p (t approximation,
            444 images); the interval is the Fisher-z interval on n = 2,222.
       T44  Fig. S15b, whole-sample correlations with the hit rate and the false alarm rate: rho and p
            as printed in the crowding report, with the Fisher-z interval.
       T45  split-half reliability of memorability, from the same sources as numbers.csv: the mean of
            the 25 random splits behind the dashed lines of Fig. S15a (faces over each image's own
            trials; scenes over the 843 participants) with its Spearman-Brown value. A correlation
            with these scores can be at most about the square root of the Spearman-Brown reliability.
Checks built in (the script stops if one fails)
    - each fold correlation recomputed from the per-image held-out table equals the fold table;
    - each whole-sample correlation recomputed from the per-image crowding equals the report;
    - the split-half code of this script, run with seed 0 and 25 splits, gives exactly the scene
      reliability over participants of data/paper_run_scenes/ (0.73161) and, when the faces were
      recomputed, the faces reliability over trials.
Inputs
    data/provided_faces/          the saved faces results (aggregate)
    outputs/scenes/               this scene run: isola_crowding.txt, isola_crowding.npz,
                                  ridge_metric.txt, ridge_metric_folds.csv, ridge_heldout_scores.csv,
                                  noise_ceiling.txt, noise_ceiling.json, noise_ceiling_splits.csv
    data/paper_run_scenes/        the saved scene run
    data/scenes/image_order.csv, data/scenes/subject_matrices.mat
    outputs/intervals_figS15.csv  (intervals.py)
    shared/embeddings_isola/ (the spaces), shared/ledger/paper_numbers.csv
Outputs
    outputs/numbers.csv, outputs/tests.csv
Seeds
    the analysis scripts (code/scenes/, code/faces/) and the built-in check use seed 0; this script
    draws no other random numbers.
Run (from the repository root; code/run_module.py runs it after the scene scripts and intervals.py)
    python A6a_faces_and_scenes_figS15/code/stats_figS15.py          (under a minute)
"""
import csv
import glob
import json
import math
import os
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import read_results  # noqa: E402  (also puts shared/python on the path)
import intervals  # noqa: E402
import release_paths  # noqa: E402
import bootstrap  # noqa: E402
import numbers_table  # noqa: E402
import tests_table  # noqa: E402

ANALYSIS_SEED = 0                        # the seed of the analysis scripts (code/scenes/, code/faces/)
N_SPLITS_PAPER = 25                      # random splits behind the reliabilities of Fig. S15a
N_SCORED = 2222                          # scored images per database
SMALLEST_FOLD = 444                      # 2,222 images in five folds of 444 or 445
MODULE = read_results.MODULE_FOLDER
OUTPUTS = os.path.join(MODULE, 'outputs')
SCENES_RERUN_FOLDER = read_results.SOURCES['scenes_rerun']['folder']
FACES_DERIVED_FOLDER = os.path.join(release_paths.release_root(), 'external', 'bainbridge_faces', 'derived')
SCRIPT = 'A6a_faces_and_scenes_figS15/code/stats_figS15.py'
DATABASE_NAMES = {'Faces': 'Bainbridge et al. (2013) faces', 'Scenes': 'Isola et al. (2011) scenes'}
WEIGHTING_TEXT = {'equal': 'all 7,168 network dimensions weighted equally',
                  'fitted': 'dimension weights fitted by ridge regression on the training folds'}
# the labels of the "Limiting case" block of the crowding reports, and the names used in this script
# (the third name is also the measure name in outputs/intervals_figS15.csv and the key of
# read_results.PRINTED)
PANEL_B_MEASURES = [('hit_rate', 'hit rate', 'hit rate'),
                    ('false_alarm_rate', 'FA rate', 'false alarm rate')]


# ============================================================================ small helpers
def spearman_rho(x, y):
    """Spearman rank correlation of two vectors (ties get average ranks)."""
    return float(spearmanr(x, y).statistic)


def spearman_p_t_approximation(rho, n):
    """Two-sided p of a Spearman rho with the t approximation (n - 2 degrees of freedom), as intervals.py."""
    t_value = rho * math.sqrt((n - 2) / (1 - rho ** 2))
    return float(2 * stats.t.sf(abs(t_value), n - 2))


def three_decimals(values):
    """'-0.483, -0.477, -0.439' for a list of numbers."""
    texts = []
    for value in values:
        texts.append('%.3f' % value)
    return ', '.join(texts)


def short_numbers(values):
    """'14.97, 9.65' for a list of numbers (no trailing zeros)."""
    texts = []
    for value in values:
        texts.append('%g' % value)
    return ', '.join(texts)


def read_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def scored_mask(order):
    """The 2,222 scored images: has_scores, and for the scenes the CVPR 2011 targets (as in the scripts
    in code/scenes/)."""
    mask = order['has_scores'].to_numpy() == 1
    if 'kind' in order.columns:
        mask = mask & (order['kind'].to_numpy() == 'target')
    if 'in_cvpr_2222' in order.columns:
        mask = mask & (order['in_cvpr_2222'].to_numpy() == 1)
    return mask


def interval_row(interval_rows, source, measure):
    """The row of outputs/intervals_figS15.csv for one source of results and one measure."""
    for row in interval_rows:
        if row['source'] == source and row['measure'] == measure:
            return row
    raise SystemExit('no interval row for %s, %s; run intervals.py first' % (source, measure))


# ============================================================================ reading the results
def count_images(faces_source, scenes_order):
    """ADD-008: the images each analysis ran over, from the reports and the shipped files."""
    counts = {}
    counts['faces_embedded'], counts['faces_scored'] = read_results.image_counts(faces_source)
    counts['scenes_embedded'] = len(scenes_order)
    counts['scenes_targets'] = int((scenes_order['kind'] == 'target').sum())
    counts['scenes_fillers'] = int((scenes_order['kind'] == 'filler').sum())
    counts['scenes_scored'] = int(scored_mask(scenes_order).sum())
    rerun_embedded, rerun_scored = read_results.image_counts('scenes_rerun')
    if (rerun_embedded, rerun_scored) != (counts['scenes_embedded'], counts['scenes_scored']):
        raise SystemExit('the scene crowding report counts %d / %d images, the image list %d / %d'
                         % (rerun_embedded, rerun_scored, counts['scenes_embedded'], counts['scenes_scored']))
    # the deep-network spaces: the 14 scene files (regenerated_<network>_<layer>.npz) and the number
    # each report names
    scene_space_files = sorted(glob.glob(release_paths.shared_path('embeddings_isola', 'regenerated_*.npz')))
    networks = []
    for path in scene_space_files:
        network = os.path.basename(path)[len('regenerated_'):].rsplit('_', 1)[0]
        if network not in networks:
            networks.append(network)
    counts['scene_space_files'] = len(scene_space_files)
    counts['networks'] = networks
    counts['faces_report_spaces'] = read_results.number_of_spaces(faces_source)
    counts['scenes_report_spaces'] = read_results.number_of_spaces('scenes_rerun')
    return counts


def fold_summary(source):
    """The five held-out fold correlations (equal and fitted weights) of one run, their means, the tau and
    alpha chosen in each fold, and the largest single-fold p of each bar."""
    settings = read_results.fold_settings(source)
    summary = dict(equal=read_results.fold_values(source, read_results.EQUAL_WEIGHTS_COLUMN),
                   fitted=read_results.fold_values(source, read_results.FITTED_WEIGHTS_COLUMN),
                   tau_equal=settings['tau_equal'], tau_fitted=settings['tau_fitted'], alpha=settings['alpha'])
    for weighting in ('equal', 'fitted'):
        summary[weighting + '_mean'] = float(np.mean(summary[weighting]))
        largest_p = 0.0
        for fold_rho in summary[weighting]:
            largest_p = max(largest_p, spearman_p_t_approximation(fold_rho, SMALLEST_FOLD))
        summary[weighting + '_largest_fold_p'] = largest_p
    return summary


def load_rerun_per_image(crowding_npz, heldout_csv, folds_csv):
    """The per-image results of one database's run:
    whole_sample  the limiting-case crowding (negated mean ensemble distance to every other image) of
                  every scored image with its hit rate and false alarm rate (Fig. S15b);
    heldout       each scored image's fold and held-out summed similarity under equal and fitted
                  weights (Fig. S15a).
    Checks that the fold correlations recomputed here equal the ones in the fold table."""
    crowding = np.load(crowding_npz, allow_pickle=True)
    scored = crowding['has_scores'].astype(bool)
    whole_sample = dict(crowding=crowding['mean_distance_crowding'][scored].astype(float),
                        hit_rate=crowding['hr'][scored],
                        false_alarm_rate=crowding['fa'][scored])
    heldout = pd.read_csv(heldout_csv, float_precision='round_trip')
    fold_table = pd.read_csv(folds_csv, float_precision='round_trip')
    for fold_number in range(1, 6):
        rows_of_fold = heldout[heldout['fold'] == fold_number]
        for column, table_column in (('summed_similarity_equal_weights', 'plain'),
                                     ('summed_similarity_fitted_weights', 'learned')):
            recomputed = spearman_rho(rows_of_fold[column], rows_of_fold['corrected_recognition'])
            saved = float(fold_table[table_column].iloc[fold_number - 1])
            if abs(recomputed - saved) > 1e-9:
                raise SystemExit('fold %d %s: recomputed %.10f, fold table %.10f' % (fold_number, column, recomputed, saved))
    return whole_sample, heldout


def whole_sample_correlations(whole_sample, source):
    """Fig. S15b at full precision (rho, exact p) from the per-image crowding, each checked against the
    3-decimal value in the crowding report of the same run."""
    correlations = {}
    for key, report_label, measure_name in PANEL_B_MEASURES:
        test = spearmanr(whole_sample['crowding'], whole_sample[key])
        report_rho, report_p = read_results.limiting_case(source, report_label)
        if abs(test.statistic - report_rho) > 0.0005 + 1e-12:
            raise SystemExit('%s: rho with the %s is %.5f, the report says %.3f'
                             % (source, measure_name, test.statistic, report_rho))
        correlations[key] = (float(test.statistic), float(test.pvalue))
    return correlations


def report_correlations(source):
    """Fig. S15b as printed in a crowding report (rho to 3 decimals, p as printed)."""
    correlations = {}
    for key, report_label, measure_name in PANEL_B_MEASURES:
        correlations[key] = read_results.limiting_case(source, report_label)
    return correlations


# ============================================================================ split-half checks
def participant_split_half(subject_matrices_file, image_order, n_splits, seed):
    """Split-half reliability of scene corrected recognition over participants: random halves of the 843
    participants (bootstrap.split_half_reliability: one permutation per split, first half = the first
    421), every target's hit rate and false alarm rate in each half from the response matrices (1 =
    response, -1 = no response, 0 = not seen, as in 05_noise_ceiling.py), Spearman correlation of the two
    halves' corrected recognition over the 2,222 scored targets."""
    import h5py
    with h5py.File(subject_matrices_file, 'r') as handle:
        detections = np.array(handle['tdet']).T       # participants x 2,400 targets (MATLAB v7.3 is transposed)
        false_alarms = np.array(handle['tfa']).T
    targets = image_order[(image_order['kind'] == 'target') & (image_order['in_cvpr_2222'] == 1)]
    target_columns = np.sort(targets['index_in_set'].to_numpy().astype(int) - 1)

    def corrected_recognition_of_half(participants):
        detections_half = detections[participants][:, target_columns]
        false_alarms_half = false_alarms[participants][:, target_columns]
        old_trials = (detections_half != 0).sum(0).astype(float)      # the target was seen as a repeat
        new_trials = (false_alarms_half != 0).sum(0).astype(float)    # the target was seen the first time
        hits = (detections_half == 1).sum(0).astype(float)
        false_alarm_count = (false_alarms_half == 1).sum(0).astype(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            hit_rate = np.where(old_trials > 0, hits / old_trials, np.nan)
            false_alarm_rate = np.where(new_trials > 0, false_alarm_count / new_trials, np.nan)
        return hit_rate - false_alarm_rate

    return bootstrap.split_half_reliability(corrected_recognition_of_half, detections.shape[0],
                                            n_splits=n_splits, seed=seed, method='spearman')


def trial_split_half(image_order, n_splits, seed):
    """Split-half reliability of corrected recognition over trials, as code/faces/05_noise_ceiling.py does:
    each scored image's own old trials and new trials are split in two at random
    (hypergeometric draws: hits first, then false alarms, in every split), each half gives a corrected
    recognition score, and the halves are correlated (Spearman) over the images. Needs the per-image
    counts, so it runs only when the faces were recomputed from the database (M4)."""
    generator = np.random.default_rng(seed)
    hits = image_order['hits'].to_numpy(float)
    misses = image_order['misses'].to_numpy(float)
    false_alarms = image_order['false_alarms'].to_numpy(float)
    correct_rejections = image_order['correct_rejections'].to_numpy(float)
    keep = scored_mask(image_order)
    keep = keep & np.isfinite(hits) & np.isfinite(misses) & np.isfinite(false_alarms) & np.isfinite(correct_rejections)
    keep = keep & (hits + misses >= 2) & (false_alarms + correct_rejections >= 2)
    hits = hits[keep]
    misses = misses[keep]
    false_alarms = false_alarms[keep]
    correct_rejections = correct_rejections[keep]
    old_trials = (hits + misses).astype(int)
    new_trials = (false_alarms + correct_rejections).astype(int)
    old_first = np.maximum(old_trials // 2, 1)
    new_first = np.maximum(new_trials // 2, 1)
    old_second = np.maximum(old_trials - old_first, 1)
    new_second = np.maximum(new_trials - new_first, 1)
    half_correlations = []
    for split in range(n_splits):
        hits_first = generator.hypergeometric(hits.astype(int), misses.astype(int), old_first)
        false_alarms_first = generator.hypergeometric(false_alarms.astype(int), correct_rejections.astype(int), new_first)
        corrected_first = hits_first / old_first - false_alarms_first / new_first
        corrected_second = (hits - hits_first) / old_second - (false_alarms - false_alarms_first) / new_second
        half_correlations.append(spearman_rho(corrected_first, corrected_second))
    half_correlations = np.asarray(half_correlations)
    corrected = bootstrap.spearman_brown(half_correlations)
    return dict(half_correlations=half_correlations, corrected=corrected,
                mean_half=float(np.mean(half_correlations)), mean_corrected=float(np.mean(corrected)),
                half_low=float(np.percentile(half_correlations, 2.5)),
                half_high=float(np.percentile(half_correlations, 97.5)),
                corrected_low=float(np.percentile(corrected, 2.5)),
                corrected_high=float(np.percentile(corrected, 97.5)),
                n_splits=n_splits, seed=seed)


# ============================================================================ outputs/numbers.csv
def with_interval(rho, row):
    """[rho, ci_low, ci_high] with the 95% interval of one row of outputs/intervals_figS15.csv."""
    return [rho, float(row['ci_low']), float(row['ci_high'])]


def write_numbers(results):
    """One row per ledger item of this module (20). Faces values come from the saved faces results (or the
    recomputed faces), scene values from this run; the 95% intervals are those of
    outputs/intervals_figS15.csv; the match is judged by the ledger's match type
    (shared/ledger/paper_numbers.csv)."""
    faces_recomputed = results['faces_recomputed']
    faces = results['faces_folds']
    scenes = results['scenes_folds']
    counts = results['counts']
    faces_panel_b = results['faces_panel_b']
    scenes_panel_b = results['scenes_panel_b']
    interval_rows = results['interval_rows']
    faces_source = results['faces_source']

    if faces_recomputed:
        faces_mode = 'rerun'
        mixed_mode = 'rerun'
        faces_files = 'A6a_faces_and_scenes_figS15/outputs/faces'
        faces_source_text = 'faces recomputed from the face database (outputs/faces/)'
    else:
        faces_mode = 'provided'
        mixed_mode = 'rerun_and_provided'
        faces_files = 'A6a_faces_and_scenes_figS15/data/provided_faces'
        faces_source_text = 'faces from the saved results in data/provided_faces/'
    scenes_mode = 'rerun'
    ridge_script = 'A6a_faces_and_scenes_figS15/code/scenes/04_ridge_metric.py; %s/ridge_metric_folds.csv' % faces_files
    crowding_script = 'A6a_faces_and_scenes_figS15/code/scenes/02_crowding_analysis.py; %s/faces_crowding.txt' % faces_files
    noise_script = 'A6a_faces_and_scenes_figS15/code/scenes/05_noise_ceiling.py; %s/noise_ceiling.json' % faces_files
    figure_script = 'A6a_faces_and_scenes_figS15/code/make_figureS15.py'
    intervals_script = 'A6a_faces_and_scenes_figS15/code/intervals.py'
    interval_text = '95% Fisher-z interval (Bonett-Wright standard error, n = 2,222) from outputs/intervals_figS15.csv'

    table = numbers_table.NumbersTable(script=SCRIPT)

    # main text: the four held-out correlations of Fig. S15a
    table.add('L3-016',
              computed_value='faces %.3f / %.3f; scenes %.3f / %.3f' % (faces['equal_mean'], faces['fitted_mean'],
                                                                     scenes['equal_mean'], scenes['fitted_mean']),
              mode=mixed_mode, script=ridge_script,
              note='mean held-out Spearman rho (five folds) between summed similarity and memorability (hit rate minus '
                   'false alarm rate), equal / fitted weights; %s, scenes from this run (SI Appendix, Fig. S15a)'
                   % faces_source_text)

    # Section 6 text: the 14 deep-network spaces
    table.add('L9-003', computed_value=counts['scene_space_files'], mode='rerun',
              script='A6a_faces_and_scenes_figS15/code/scenes/02_crowding_analysis.py',
              note='deep-network spaces in the ensemble (seven networks, an early and a late layer each): %d scene '
                   'files in shared/embeddings_isola/; the faces report names %d spaces'
                   % (counts['scene_space_files'], counts['faces_report_spaces']))

    # Section 6 text: fitted weights, faces then scenes, with their intervals
    table.add('L9-004',
              computed_value=with_interval(faces['fitted_mean'], interval_row(interval_rows, faces_source,
                                                                              'corrected recognition, fitted weights')),
              mode=faces_mode, script='%s/ridge_metric_folds.csv; %s' % (faces_files, intervals_script),
              note='faces, fitted weights: mean of the five held-out fold Spearman rhos (%s) between summed '
                   'similarity and memorability (hit rate minus false alarm rate), then its %s'
                   % (three_decimals(faces['fitted']), interval_text))
    table.add('L9-005',
              computed_value=with_interval(scenes['fitted_mean'], interval_row(interval_rows, 'scenes_rerun',
                                                                               'corrected recognition, fitted weights')),
              mode=scenes_mode, script='A6a_faces_and_scenes_figS15/code/scenes/04_ridge_metric.py; ' + intervals_script,
              note='scenes, fitted weights: mean of the five held-out fold Spearman rhos (%s) between summed '
                   'similarity and memorability (hit rate minus false alarm rate), this run, then its %s'
                   % (three_decimals(scenes['fitted']), interval_text))

    # Fig. S15 caption: split-half reliabilities, faces then scenes
    reliability_note = ('split-half reliability of memorability (hit rate minus false alarm rate), uncorrected, mean '
                        'of 25 random splits (seed 0): faces over each image\'s own trials %.5f (Spearman-Brown %.5f), '
                        'scenes over the 843 participants %.5f (Spearman-Brown %.5f); %s, scenes %s'
                        % (results['faces_reliability'], results['faces_spearman_brown'],
                           results['scenes_reliability'], results['scenes_spearman_brown'], faces_source_text,
                           results['scenes_reliability_source_text']))
    table.add('L9-006', computed_value=[results['faces_reliability'], results['scenes_reliability']],
              mode=mixed_mode, script=noise_script, note=reliability_note)

    # Section 6 text: crowded images got fewer hits and more false alarms (Fig. S15b)
    table.add('L9-007',
              computed_value='faces hits %.3f, false alarms %+.3f; scenes hits %.3f, false alarms %+.3f'
                             % (faces_panel_b['hit_rate'][0], faces_panel_b['false_alarm_rate'][0],
                                scenes_panel_b['hit_rate'][0], scenes_panel_b['false_alarm_rate'][0]),
              mode=mixed_mode, script=crowding_script,
              note='whole-sample Spearman rho of summed similarity (equal weights) with the hit rate and with the false '
                   'alarm rate over the 2,222 scored images of each database (Fig. S15b); %s, scenes from this run'
                   % faces_source_text)

    # Fig. S15 caption: the two panels
    table.add('L9-008', computed_value='A6a_faces_and_scenes_figS15/outputs/figures/FigS15_faces_scenes_generality.png, '
                                       'panel a (outputs/figure_data/FigS15a.csv)',
              mode=mixed_mode, script=figure_script,
              note='bars: faces %.3f / %.3f, scenes %.3f / %.3f (mean held-out rho, equal / fitted weights); dashed '
                   'lines at minus the uncorrected split-half reliabilities %.3f and %.3f'
                   % (results['faces_printed_heldout']['equal'], results['faces_printed_heldout']['fitted'],
                      results['scenes_printed_heldout']['equal'], results['scenes_printed_heldout']['fitted'],
                      results['faces_reliability'], results['scenes_reliability']))
    table.add('L9-009', computed_value='A6a_faces_and_scenes_figS15/outputs/figures/FigS15_faces_scenes_generality.png, '
                                       'panel b (outputs/figure_data/FigS15b.csv)',
              mode=mixed_mode, script=figure_script,
              note='bars: faces hits %.3f, false alarms %.3f; scenes hits %.3f, false alarms %.3f (whole-sample rho of '
                   'summed similarity with the hit rate and with the false alarm rate)'
                   % (faces_panel_b['hit_rate'][0], faces_panel_b['false_alarm_rate'][0],
                      scenes_panel_b['hit_rate'][0], scenes_panel_b['false_alarm_rate'][0]))

    # Fig. S15 caption: seven networks
    table.add('L9-010', computed_value=len(counts['networks']), mode='rerun',
              script='A6a_faces_and_scenes_figS15/code/scenes/02_crowding_analysis.py',
              note='networks in the ensemble, each at an early and a late layer: %s (AlexNet, CLIP ViT-B/32, CORnet-S, '
                   'DINOv3 ViT-B/16, InceptionV3, ResNet18, VGG16-BN; shared/embedding_extraction/README.md)'
                   % ', '.join(counts['networks']))

    # Section 6 text: equal weights, faces then scenes, with their intervals
    table.add('L9-011',
              computed_value=with_interval(faces['equal_mean'], interval_row(interval_rows, faces_source,
                                                                             'corrected recognition, equal weights')),
              mode=faces_mode, script='%s/ridge_metric_folds.csv; %s' % (faces_files, intervals_script),
              note='faces, equal weights: mean of the five held-out fold Spearman rhos (%s) between summed '
                   'similarity and memorability (hit rate minus false alarm rate), then its %s'
                   % (three_decimals(faces['equal']), interval_text))
    table.add('L9-012',
              computed_value=with_interval(scenes['equal_mean'], interval_row(interval_rows, 'scenes_rerun',
                                                                              'corrected recognition, equal weights')),
              mode=scenes_mode, script='A6a_faces_and_scenes_figS15/code/scenes/04_ridge_metric.py; ' + intervals_script,
              note='scenes, equal weights: mean of the five held-out fold Spearman rhos (%s) between summed '
                   'similarity and memorability (hit rate minus false alarm rate), this run, then its %s'
                   % (three_decimals(scenes['equal']), interval_text))

    # Section 6 text: all p < 0.001 (Fig. S15a)
    largest_panel_a_p = max(faces['equal_largest_fold_p'], faces['fitted_largest_fold_p'],
                            scenes['equal_largest_fold_p'], scenes['fitted_largest_fold_p'])
    table.add('L9-014', computed_value=largest_panel_a_p, mode=mixed_mode,
              script=intervals_script + '; ' + SCRIPT,
              note='largest two-sided p of the four Fig. S15a correlations; the p of each bar is the largest of its '
                   'five single-fold p values (t approximation, 444 images): faces %.3g / %.3g, scenes %.3g / %.3g '
                   '(equal / fitted weights)'
                   % (faces['equal_largest_fold_p'], faces['fitted_largest_fold_p'],
                      scenes['equal_largest_fold_p'], scenes['fitted_largest_fold_p']))

    # Fig. S15 caption: the dashed lines
    table.add('L9-015', computed_value=[results['faces_reliability'], results['scenes_reliability']],
              mode=mixed_mode, script=noise_script,
              note='uncorrected split-half reliability of memorability, mean of 25 random splits (seed 0), drawn with '
                   'sign reversed as the dashed lines of Fig. S15a: faces over each image\'s own trials, scenes over '
                   'the 843 participants; %s, scenes %s' % (faces_source_text, results['scenes_reliability_source_text']))

    # Section 6 text: hits and false alarms (Fig. S15b), with their intervals
    faces_hit = with_interval(faces_panel_b['hit_rate'][0], interval_row(interval_rows, faces_source, 'hit rate'))
    scenes_hit = with_interval(scenes_panel_b['hit_rate'][0], interval_row(interval_rows, 'scenes_rerun', 'hit rate'))
    table.add('L9-016', computed_value=faces_hit + scenes_hit,
              mode=mixed_mode, script=crowding_script + '; ' + intervals_script,
              note='whole-sample Spearman rho of summed similarity (equal weights) with the hit rate over the 2,222 '
                   'scored images, each followed by its %s, faces then scenes (p %.3g and %.3g); %s, scenes from '
                   'this run' % (interval_text, faces_panel_b['hit_rate'][1], scenes_panel_b['hit_rate'][1],
                                 faces_source_text))
    largest_panel_b_p = max(faces_panel_b['hit_rate'][1], faces_panel_b['false_alarm_rate'][1],
                            scenes_panel_b['hit_rate'][1], scenes_panel_b['false_alarm_rate'][1])
    faces_false_alarm = with_interval(faces_panel_b['false_alarm_rate'][0],
                                      interval_row(interval_rows, faces_source, 'false alarm rate'))
    scenes_false_alarm = with_interval(scenes_panel_b['false_alarm_rate'][0],
                                       interval_row(interval_rows, 'scenes_rerun', 'false alarm rate'))
    table.add('L9-017', computed_value=faces_false_alarm + scenes_false_alarm + [largest_panel_b_p],
              mode=mixed_mode, script=crowding_script + '; ' + intervals_script,
              note='whole-sample Spearman rho of summed similarity (equal weights) with the false alarm rate over the '
                   '2,222 scored images, each followed by its %s, faces then scenes (p %.3g and %.3g), then the '
                   'largest p of the four Fig. S15b correlations; %s, scenes from this run'
                   % (interval_text, faces_panel_b['false_alarm_rate'][1], scenes_panel_b['false_alarm_rate'][1],
                      faces_source_text))

    # Fig. S15 caption: 95% intervals on 2,222 images
    table.add('L9-018', computed_value=[intervals.CONFIDENCE_LEVEL, intervals.N_SCORED_IMAGES], mode=mixed_mode,
              script=intervals_script,
              note='confidence level (percent) and images per database of the Fisher-z intervals (Bonett-Wright '
                   'standard error) drawn in Fig. S15; scored images: faces %s of %s embedded, scenes %s of %s; every '
                   'interval is in outputs/intervals_figS15.csv'
                   % ('{:,}'.format(counts['faces_scored']), '{:,}'.format(counts['faces_embedded']),
                      '{:,}'.format(counts['scenes_scored']), '{:,}'.format(counts['scenes_embedded'])))

    # Section 6 text: the highest correlation the scores allow, the square root of the Spearman-Brown value
    faces_ceiling = math.sqrt(results['faces_spearman_brown'])
    scenes_ceiling = math.sqrt(results['scenes_spearman_brown'])
    table.add('L9-141', computed_value=[faces_ceiling, scenes_ceiling], mode=mixed_mode, script=noise_script,
              note='square root of the Spearman-Brown corrected split-half reliability of memorability (25 random '
                   'splits, seed 0): faces sqrt(%.5f), scenes sqrt(%.5f); %s, scenes %s'
                   % (results['faces_spearman_brown'], results['scenes_spearman_brown'], faces_source_text,
                      results['scenes_reliability_source_text']))

    # Section 6 text: the fitted-weight correlations reach about two thirds of that ceiling
    faces_share = abs(faces['fitted_mean']) / faces_ceiling
    scenes_share = abs(scenes['fitted_mean']) / scenes_ceiling
    table.add('L9-142',
              computed_value='faces %.3f, scenes %.3f' % (faces_share, scenes_share),
              mode=mixed_mode, script=ridge_script + '; ' + noise_script,
              note='fitted-weight held-out correlation (absolute value) divided by the square root of the '
                   'Spearman-Brown reliability: faces %.3f / %.3f, scenes %.3f / %.3f; %s, scenes from this run'
                   % (abs(faces['fitted_mean']), faces_ceiling, abs(scenes['fitted_mean']), scenes_ceiling,
                      faces_source_text))

    # ADD-008: the image counts
    computed_counts = [counts['faces_embedded'], counts['faces_scored'], counts['scenes_embedded'],
                       counts['scenes_targets'], counts['scenes_scored'], counts['scenes_fillers']]
    table.add('ADD-008', computed_value=computed_counts, mode=mixed_mode, script=SCRIPT,
              note='images embedded and scored: faces embedded, faces scored (read from the %s faces report); scenes '
                   'embedded, targets, scored targets, fillers (data/scenes/image_order.csv)'
                   % ('recomputed' if faces_recomputed else 'saved'))

    # ADD-009: the fold values behind each Fig. S15a value, and the settings chosen in the folds
    all_alphas = sorted(set(faces['alpha'] + scenes['alpha']))
    table.add('ADD-009',
              computed_value='faces equal %s; faces fitted %s; scenes equal %s; scenes fitted %s; ridge alpha %s in '
                             'every fold' % (three_decimals(faces['equal']), three_decimals(faces['fitted']),
                                             three_decimals(scenes['equal']), three_decimals(scenes['fitted']),
                                             short_numbers(all_alphas)),
              mode=mixed_mode, script=ridge_script,
              note='the five held-out fold Spearman rhos behind each Fig. S15a bar and the ridge penalty alpha chosen '
                   'in every fold (%s, scenes from this run); kernel width tau chosen in the training folds: faces '
                   'equal %s, fitted %s; scenes equal %s, fitted %s'
                   % (faces_source_text, short_numbers(faces['tau_equal']), short_numbers(faces['tau_fitted']),
                      short_numbers(scenes['tau_equal']), short_numbers(scenes['tau_fitted'])))
    table.write(OUTPUTS)


# ============================================================================ outputs/tests.csv
SECTION_6 = 'SI Appendix, Section 6, Faces and Scenes'


def ledger_ids_panel_a(database, weighting):
    """The ledger items that print one Fig. S15a correlation."""
    if database == 'Faces':
        return 'L3-016;L9-011;L9-014' if weighting == 'equal' else 'L3-016;L9-004;L9-014'
    return 'L3-016;L9-012;L9-014' if weighting == 'equal' else 'L3-016;L9-005;L9-014'


def add_t43_rows(tests, results):
    """T43: Fig. S15a, held-out correlations of summed similarity with memorability (the saved runs)."""
    interval_rows = results['interval_rows']
    for database in ('Faces', 'Scenes'):
        if database == 'Faces':
            paper_source = 'faces_provided'
            paper_folds = results['provided_faces_folds']
            paper_label = 'saved faces results (data/provided_faces/)'
            reliability_text = 'split-half reliability of face memorability over trials %.3f' % results['faces_reliability']
        else:
            paper_source = 'scenes_paper_run'
            paper_folds = results['paper_scenes_folds']
            paper_label = 'saved scene run (data/paper_run_scenes/)'
            reliability_text = ('split-half reliability of scene memorability over participants %.3f'
                                % results['scenes_reliability'])
        for weighting in ('equal', 'fitted'):
            ledger_ids = ledger_ids_panel_a(database, weighting)
            description = ('Fig. S15a, %s, %s weights (%s): Spearman rho between held-out summed similarity in the '
                           '14-space deep-network ensemble and memorability (hit rate minus false alarm rate) over the '
                           '2,222 scored images, the mean of five held-out folds (KFold seed 0); for comparison, %s'
                           % (DATABASE_NAMES[database], weighting, WEIGHTING_TEXT[weighting], reliability_text))
            measure = 'corrected recognition, %s weights' % weighting
            paper_interval = interval_row(interval_rows, paper_source, measure)
            paper_mean = paper_folds[weighting + '_mean']
            paper_p = paper_folds[weighting + '_largest_fold_p']
            tests.add(test_id='T43', ledger_ids=ledger_ids, location=SECTION_6 + ' (Fig. S15a)',
                      test=description + '; ' + paper_label + '; p = the largest single-fold p (t approximation, 444 images)',
                      statistic_name='Spearman rho', statistic=paper_mean, n=N_SCORED,
                      unit='scored images (five held-out folds of 444 or 445)', df='', p=paper_p,
                      p_type='parametric two-sided', effect_size_name='Spearman rho', effect_size=paper_mean,
                      ci_low=float(paper_interval['ci_low']), ci_high=float(paper_interval['ci_high']),
                      ci_method='Fisher z, Bonett-Wright standard error, n = 2,222 (the error bars of Fig. S15a, '
                                'around the mean of the five fold correlations)',
                      resampling_unit='', n_resamples='', seed=ANALYSIS_SEED)


def add_t44_rows(tests, results):
    """T44: Fig. S15b, whole-sample correlations with the hit rate and the false alarm rate (the saved runs)."""
    interval_rows = results['interval_rows']
    ledger_ids_of = {'hit_rate': 'L9-007;L9-016;L9-017', 'false_alarm_rate': 'L9-007;L9-017'}
    for database in ('Faces', 'Scenes'):
        if database == 'Faces':
            paper_source = 'faces_provided'
            paper_label = 'saved faces results (data/provided_faces/; rho to 3 decimals and p as printed in the report)'
            paper_values = results['provided_faces_panel_b']
        else:
            paper_source = 'scenes_paper_run'
            paper_label = 'saved scene run (data/paper_run_scenes/; rho to 3 decimals and p as printed in the report)'
            paper_values = results['paper_scenes_panel_b']
        for key, report_label, measure_name in PANEL_B_MEASURES:
            description = ('Fig. S15b, %s: Spearman rho between summed similarity (equal weights; the negated mean '
                           'ensemble distance to every other image of the database) and the %s over the 2,222 scored '
                           'images, whole sample (not held out)' % (DATABASE_NAMES[database], measure_name))
            paper_rho, paper_p = paper_values[key]
            paper_interval = interval_row(interval_rows, paper_source, measure_name)
            tests.add(test_id='T44', ledger_ids=ledger_ids_of[key], location=SECTION_6 + ' (Fig. S15b)',
                      test=description + '; ' + paper_label,
                      statistic_name='Spearman rho', statistic=paper_rho, n=N_SCORED, unit='scored images',
                      df=N_SCORED - 2, p=paper_p, p_type='parametric two-sided', effect_size_name='Spearman rho',
                      effect_size=paper_rho, ci_low=float(paper_interval['ci_low']),
                      ci_high=float(paper_interval['ci_high']),
                      ci_method='Fisher z, Bonett-Wright standard error, n = 2,222 (as the error bars of Fig. S15b)',
                      resampling_unit='', n_resamples='', seed='')


def add_t45_rows(tests, results):
    """T45: split-half reliability of memorability (the dashed lines of Fig. S15a; the square root of its
    Spearman-Brown value is the highest correlation the scores allow, SI Appendix, Section 6)."""
    # faces: the 25 splits over trials behind the dashed line
    tests.add(test_id='T45', ledger_ids='L9-006;L9-015;L9-141', location='SI Appendix, Fig. S15 caption (panel a, dashed lines)',
              test=('split-half reliability of face memorability (corrected recognition): Spearman rho between two '
                    'halves of each image\'s own trials (the face database has no per-participant data), over the '
                    '2,222 scored faces, mean of 25 random splits; %s'
                    % ('faces recomputed from the face database' if results['faces_recomputed']
                       else 'saved faces results (data/provided_faces/)')),
              statistic_name='split-half Spearman rho (uncorrected)', statistic=results['faces_reliability'], n=N_SCORED,
              unit='scored images; split over trials', df='', p='', p_type='none',
              effect_size_name='Spearman-Brown reliability (2r / (1 + r) of the mean)',
              effect_size=results['faces_spearman_brown'],
              ci_low=results['faces_paper_split_range'][0], ci_high=results['faces_paper_split_range'][1],
              ci_method=results['faces_paper_split_range_method'],
              resampling_unit='trials (25 random splits)', n_resamples=N_SPLITS_PAPER, seed=ANALYSIS_SEED)

    # scenes: the 25 splits over participants behind the dashed line
    tests.add(test_id='T45', ledger_ids='L9-006;L9-015;L9-141', location='SI Appendix, Fig. S15 caption (panel a, dashed lines)',
              test=('split-half reliability of scene memorability (corrected recognition): Spearman rho between two '
                    'random halves of the 843 participants over the 2,222 scored targets, mean of 25 random splits '
                    '(seed 0); %s' % results['scenes_reliability_source_text']),
              statistic_name='split-half Spearman rho (uncorrected)', statistic=results['scenes_reliability'], n=N_SCORED,
              unit='scored images; split over the 843 participants', df='', p='', p_type='none',
              effect_size_name='Spearman-Brown reliability (2r / (1 + r) of the mean)',
              effect_size=results['scenes_spearman_brown'],
              ci_low=results['scenes_paper_split_range'][0], ci_high=results['scenes_paper_split_range'][1],
              ci_method=results['scenes_paper_split_range_method'],
              resampling_unit='participants (25 random half splits)', n_resamples=N_SPLITS_PAPER, seed=ANALYSIS_SEED)


def write_tests(results):
    tests = tests_table.TestsTable(script=SCRIPT)
    add_t43_rows(tests, results)
    add_t44_rows(tests, results)
    add_t45_rows(tests, results)
    tests.write(OUTPUTS)
    return tests


# ============================================================================ the main computation
def main():
    print('stats_figS15.py: seed %d for the check against the saved scene run\n' % ANALYSIS_SEED)
    results = {}

    # ---------------------------------------------------------------- where the faces come from
    faces_source = read_results.faces_source()        # prints the provided-output message when needed
    faces_recomputed = (faces_source == 'faces_regenerated')
    results['faces_recomputed'] = faces_recomputed
    results['faces_source'] = faces_source
    scenes_order = pd.read_csv(os.path.join(MODULE, 'data', 'scenes', 'image_order.csv'), float_precision='round_trip')

    # ---------------------------------------------------------------- 1. image counts (ADD-008)
    counts = count_images(faces_source, scenes_order)
    results['counts'] = counts
    print('Images the analyses ran over (ADD-008)')
    print('  faces: %d images embedded, %d with memorability scores (%s)'
          % (counts['faces_embedded'], counts['faces_scored'], read_results.SOURCES[faces_source]['label']))
    print('  scenes: %d images embedded = %d targets + %d fillers; %d scored targets enter the correlations'
          % (counts['scenes_embedded'], counts['scenes_targets'], counts['scenes_fillers'], counts['scenes_scored']))
    print('  deep-network spaces: %d scene files (%d networks: %s); the faces and scene reports name %d and %d spaces\n'
          % (counts['scene_space_files'], len(counts['networks']), ', '.join(counts['networks']),
             counts['faces_report_spaces'], counts['scenes_report_spaces']))

    # ---------------------------------------------------------------- 2. folds and settings (ADD-009)
    results['faces_folds'] = fold_summary(faces_source)
    results['provided_faces_folds'] = fold_summary('faces_provided')
    results['scenes_folds'] = fold_summary('scenes_rerun')
    results['paper_scenes_folds'] = fold_summary('scenes_paper_run')
    for source, key in ((faces_source, 'faces_folds'), ('scenes_rerun', 'scenes_folds'),
                        ('scenes_paper_run', 'paper_scenes_folds')):
        summary = results[key]
        print('Held-out folds, %s' % read_results.SOURCES[source]['label'])
        print('  equal weights:  mean %+.4f; folds %s; tau %s'
              % (summary['equal_mean'], three_decimals(summary['equal']), short_numbers(summary['tau_equal'])))
        print('  fitted weights: mean %+.4f; folds %s; tau %s'
              % (summary['fitted_mean'], three_decimals(summary['fitted']), short_numbers(summary['tau_fitted'])))
        print('  ridge alpha: %s' % short_numbers(summary['alpha']))
    print()

    # the 3-decimal held-out values the reports print (the figure plots these)
    results['faces_printed_heldout'] = {}
    results['scenes_printed_heldout'] = {}
    for weighting, row_name in (('equal', read_results.EQUAL_WEIGHTS_ROW), ('fitted', read_results.FITTED_WEIGHTS_ROW)):
        results['faces_printed_heldout'][weighting] = read_results.heldout_rho(faces_source, row_name)
        results['scenes_printed_heldout'][weighting] = read_results.heldout_rho('scenes_rerun', row_name)

    # ---------------------------------------------------------------- per-image results (this scene run; recomputed faces)
    scenes_whole, _ = load_rerun_per_image(os.path.join(SCENES_RERUN_FOLDER, 'isola_crowding.npz'),
                                                        os.path.join(SCENES_RERUN_FOLDER, 'ridge_heldout_scores.csv'),
                                                        os.path.join(SCENES_RERUN_FOLDER, 'ridge_metric_folds.csv'))
    results['scenes_panel_b'] = whole_sample_correlations(scenes_whole, 'scenes_rerun')
    results['paper_scenes_panel_b'] = report_correlations('scenes_paper_run')
    results['provided_faces_panel_b'] = report_correlations('faces_provided')
    faces_order = None
    if faces_recomputed:
        faces_whole, _ = load_rerun_per_image(
            os.path.join(FACES_DERIVED_FOLDER, 'faces_crowding.npz'),
            os.path.join(FACES_DERIVED_FOLDER, 'ridge_heldout_scores.csv'),
            os.path.join(read_results.SOURCES['faces_regenerated']['folder'], 'ridge_metric_folds.csv'))
        faces_order = pd.read_csv(os.path.join(FACES_DERIVED_FOLDER, 'image_order.csv'), float_precision='round_trip')
        results['faces_panel_b'] = whole_sample_correlations(faces_whole, 'faces_regenerated')
    else:
        results['faces_panel_b'] = results['provided_faces_panel_b']

    # ---------------------------------------------------------------- reliabilities (the dashed lines)
    faces_noise = read_json(read_results.file_path(faces_source, 'noise').replace('.txt', '.json'))
    results['faces_reliability'], results['faces_spearman_brown'] = faces_noise['over trials']['cr']
    scenes_reliability_source = read_results.scene_reliability_source()
    scenes_noise = read_json(read_results.file_path(scenes_reliability_source, 'noise').replace('.txt', '.json'))
    results['scenes_reliability'], results['scenes_spearman_brown'] = scenes_noise['over participants']['cr']
    paper_scenes_noise = read_json(read_results.file_path('scenes_paper_run', 'noise').replace('.txt', '.json'))
    results['paper_scenes_reliability'] = paper_scenes_noise['over participants']['cr'][0]
    if scenes_reliability_source == 'scenes_rerun':
        results['scenes_reliability_source_text'] = 'computed in this run from the Isola participant matrices'
    else:
        results['scenes_reliability_source_text'] = ('read from the saved run in data/paper_run_scenes/ (no participant '
                                                     'matrices)')

    # the range of the 25 split values behind each reliability (noise_ceiling_splits.csv of a run)
    results['faces_paper_split_range'] = ('', '')
    results['faces_paper_split_range_method'] = 'none (data/provided_faces/ holds only the mean of the 25 splits)'
    if faces_recomputed:
        faces_splits = pd.read_csv(os.path.join(read_results.SOURCES['faces_regenerated']['folder'], 'noise_ceiling_splits.csv'))
        values = faces_splits[(faces_splits['unit'] == 'over trials') & (faces_splits['measure'] == 'cr')]['split_half_rho'].to_numpy()
        results['faces_paper_split_range'] = (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        results['faces_paper_split_range_method'] = '2.5 to 97.5 percent range of the 25 split values (not a confidence interval)'
    results['scenes_paper_split_range'] = ('', '')
    results['scenes_paper_split_range_method'] = 'none (the 25 split values need the participant matrices, M8)'
    if scenes_reliability_source == 'scenes_rerun':
        scene_splits = pd.read_csv(os.path.join(SCENES_RERUN_FOLDER, 'noise_ceiling_splits.csv'), float_precision='round_trip')
        values = scene_splits[(scene_splits['unit'] == 'over participants') & (scene_splits['measure'] == 'cr')]['split_half_rho'].to_numpy()
        if abs(np.mean(values) - results['scenes_reliability']) > 1e-12:
            raise SystemExit('the 25 scene split values do not average to the reliability in noise_ceiling.json')
        results['scenes_paper_split_range'] = (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        results['scenes_paper_split_range_method'] = ('2.5 to 97.5 percent range of the 25 split values (not a confidence '
                                                      'interval; SD over the splits %.4f)' % np.std(values, ddof=1))

    # ---------------------------------------------------------------- checks of the split-half code
    subjects_file = read_results.subject_matrices_path()
    if subjects_file is not None:
        check = participant_split_half(subjects_file, scenes_order, N_SPLITS_PAPER, ANALYSIS_SEED)
        if abs(check['mean_half'] - results['paper_scenes_reliability']) > 1e-12:
            raise SystemExit('the split-half code with seed 0 and 25 splits gives %.12f, the saved scene run %.12f'
                             % (check['mean_half'], results['paper_scenes_reliability']))
        print('check: the participant split-half code with seed 0 and 25 splits gives %.6f; saved scene run %.6f '
              '(equal)' % (check['mean_half'], results['paper_scenes_reliability']))
    if faces_recomputed:
        check = trial_split_half(faces_order, N_SPLITS_PAPER, ANALYSIS_SEED)
        if abs(check['mean_half'] - results['faces_reliability']) > 1e-12:
            raise SystemExit('the trial split-half code with seed 0 and 25 splits gives %.12f, the faces run %.12f'
                             % (check['mean_half'], results['faces_reliability']))
        print('check: the trial split-half code with seed 0 and 25 splits gives %.6f; recomputed faces reliability '
              '%.6f (equal)' % (check['mean_half'], results['faces_reliability']))

    # ---------------------------------------------------------------- the Fisher-z intervals (intervals.py)
    results['interval_rows'] = []
    with open(intervals.OUTPUT_FILE, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            results['interval_rows'].append(row)

    # ---------------------------------------------------------------- 3. numbers.csv and tests.csv
    write_numbers(results)
    write_tests(results)


if __name__ == '__main__':
    main()
