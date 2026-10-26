"""
read_results.py: where the Fig. S15 result files are, and how numbers are read out of them.

What it does
    The faces and scenes analyses write plain-text reports (a crowding report, a ridge-metric report
    with its fold table, and a noise-ceiling report). Every script of this module reads them through
    the functions below, which find each number by the label of its row (never by line number), so a
    report with an extra line still reads correctly.

The four sources of results (folders relative to the module folder)
    faces_provided       data/provided_faces/    the saved faces results. The face database
                                                  (Bainbridge, Isola and Oliva 2013) may not be
                                                  redistributed, so the repository holds these aggregate
                                                  reports (EXTERNAL_INPUTS.md, M4).
    faces_regenerated    outputs/faces/          written only by code/faces/regenerate_faces.py, from
                                                  the database archives put into
                                                  external/bainbridge_faces/
    scenes_rerun         outputs/scenes/         this module's scene run on the 14 spaces in
                                                  shared/embeddings_isola/
    scenes_paper_run     data/paper_run_scenes/  the saved scene run, the output the paper reports

Also here: the provided-output switch for the faces and the search for the Isola participant
matrices (they ship in data/scenes/; without them the scene reliability is read from
data/paper_run_scenes/).
"""
import csv
import os
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402

MODULE_FOLDER = release_paths.module_path(__file__)

# The file names of the four sources. The faces and scenes reports have different crowding-file
# names (the face scripts write faces_crowding.txt, the scene scripts isola_crowding.txt).
SOURCES = {
    'faces_provided': dict(folder=os.path.join(MODULE_FOLDER, 'data', 'provided_faces'),
                           crowding='faces_crowding.txt', database='Faces',
                           label='faces, saved results (data/provided_faces/)'),
    'faces_regenerated': dict(folder=os.path.join(MODULE_FOLDER, 'outputs', 'faces'),
                              crowding='faces_crowding.txt', database='Faces',
                              label='faces, recomputed from the face database (outputs/faces/)'),
    'scenes_rerun': dict(folder=os.path.join(MODULE_FOLDER, 'outputs', 'scenes'),
                         crowding='isola_crowding.txt', database='Scenes',
                         label='scenes, this run (outputs/scenes/)'),
    'scenes_paper_run': dict(folder=os.path.join(MODULE_FOLDER, 'data', 'paper_run_scenes'),
                             crowding='isola_crowding.txt', database='Scenes',
                             label='scenes, saved run (data/paper_run_scenes/)'),
}
RIDGE_REPORT = 'ridge_metric.txt'
FOLD_TABLE = 'ridge_metric_folds.csv'
NOISE_REPORT = 'noise_ceiling.txt'

# The faces need the archives of the face database (not in the repository); EXTERNAL_INPUTS.md calls
# this input M4.
FACES_INPUT = release_paths.Input('M4', 'the 10k US Adult Faces Database archives (Bainbridge, Isola and Oliva 2013)',
                                'external/bainbridge_faces')

# The rows of the HELD-OUT block of the ridge-metric report that the paper uses
EQUAL_WEIGHTS_ROW = 'crowding, isotropic metric'          # Fig. S15a, "Equal Weights"
FITTED_WEIGHTS_ROW = 'crowding, ridge-weighted metric'    # Fig. S15a, "Fitted Weights"
EQUAL_WEIGHTS_COLUMN = 'plain'                            # the same correlations per fold, in the fold table
FITTED_WEIGHTS_COLUMN = 'learned'


def printed_value(ledger_id, part=0):
    """The printed value of a ledger item (shared/ledger/paper_numbers.csv); items with several values
    ('-0.32; -0.36; -0.29; -0.51; -0.54; -0.47') are split at ';' and 'part' picks one."""
    import numbers_table
    parts = [piece.strip() for piece in numbers_table.load_ledger()[ledger_id]['paper_value'].split(';')]
    return parts[part]


# The ledger items that hold each Fig. S15 value, with the part for faces and for scenes (L9-016 and
# L9-017 print each rho followed by its interval, so the scene rho is part 3). The key
# 'false alarm rate' is also the measure name in outputs/intervals_figS15.csv and the figure data.
PRINTED = {('Faces', 'equal'): ('L9-011', 0), ('Scenes', 'equal'): ('L9-012', 0),
           ('Faces', 'fitted'): ('L9-004', 0), ('Scenes', 'fitted'): ('L9-005', 0),
           ('Faces', 'hit rate'): ('L9-016', 0), ('Scenes', 'hit rate'): ('L9-016', 3),
           ('Faces', 'false alarm rate'): ('L9-017', 0), ('Scenes', 'false alarm rate'): ('L9-017', 3),
           ('Faces', 'reliability'): ('L9-006', 0), ('Scenes', 'reliability'): ('L9-006', 1)}


def printed(database, quantity):
    """The printed value of one Fig. S15 quantity, e.g. printed('Scenes', 'fitted') -> '-0.64'."""
    ledger_id, part = PRINTED[(database, quantity)]
    return printed_value(ledger_id, part)


def file_path(source, which):
    """Full path of one report: which = 'crowding', 'ridge', 'folds' or 'noise'."""
    names = {'crowding': SOURCES[source]['crowding'], 'ridge': RIDGE_REPORT, 'folds': FOLD_TABLE,
             'noise': NOISE_REPORT}
    return os.path.join(SOURCES[source]['folder'], names[which])


def relative(path):
    """A path relative to the repository root, for printing (no absolute paths in logs)."""
    return os.path.relpath(path, release_paths.release_root())


def read_text(source, which):
    path = file_path(source, which)
    if not os.path.exists(path):
        raise SystemExit('missing %s (%s); run A6a_faces_and_scenes_figS15/code/run_module.py first'
                         % (relative(path), SOURCES[source]['label']))
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def heldout_rho(source, row_name):
    """The mean held-out Spearman rho printed on the row 'row_name' of the HELD-OUT block of the
    ridge-metric report (the mean of the five fold correlations, 3 decimals)."""
    text = read_text(source, 'ridge')
    block = text.split('HELD-OUT SPEARMAN RHO')[1]
    match = re.search(re.escape(row_name) + r'\s+([-+]\d\.\d+)', block)
    if match is None:
        raise SystemExit('could not find the row %r in %s' % (row_name, relative(file_path(source, 'ridge'))))
    return float(match.group(1))


def fold_values(source, column):
    """The five held-out fold correlations of one column of the fold table, at full precision."""
    path = file_path(source, 'folds')
    if not os.path.exists(path):
        raise SystemExit('missing %s; run A6a_faces_and_scenes_figS15/code/run_module.py first' % relative(path))
    values = []
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            values.append(float(row[column]))
    return values


def fold_settings(source):
    """The kernel width tau chosen inside each training fold (equal and fitted weights) and the
    ridge penalty alpha chosen in each fold, as printed at the end of the ridge-metric report."""
    text = read_text(source, 'ridge')
    tau_match = re.search(r"tau chosen: isotropic \[([^\]]*)\], learned \[([^\]]*)\]", text)
    alpha_match = re.search(r"ridge alpha chosen: \[([^\]]*)\]", text)
    if tau_match is None or alpha_match is None:
        raise SystemExit('could not find the chosen tau and alpha in %s' % relative(file_path(source, 'ridge')))
    tau_equal = [float(value.strip(" '")) for value in tau_match.group(1).split(',')]
    tau_fitted = [float(value.strip(" '")) for value in tau_match.group(2).split(',')]
    alphas = [float(value) for value in alpha_match.group(1).split(',')]
    return dict(tau_equal=tau_equal, tau_fitted=tau_fitted, alpha=alphas)


def limiting_case(source, measure_label):
    """rho and p of one measure in the 'Limiting case' block of the crowding report: the
    whole-sample Spearman correlation of the negated mean ensemble distance to every other image
    with the 'hit rate' or the 'FA rate' (Fig. S15b)."""
    text = read_text(source, 'crowding')
    block = text.split('Limiting case')[1]
    match = re.search(re.escape(measure_label) + r'\s+rho = ([-+]\d\.\d+) \(p = ([0-9.e+-]+)\)', block)
    if match is None:
        raise SystemExit('could not find %r in the limiting-case block of %s'
                         % (measure_label, relative(file_path(source, 'crowding'))))
    return float(match.group(1)), float(match.group(2))


def reliability(source, measure_label, unit):
    """Split-half reliability and its Spearman-Brown value for one measure (the row label of the
    report: 'corrected recognition') from the block 'Ceiling over <unit>' of the noise-ceiling report
    (unit = 'participants' or 'trials')."""
    text = read_text(source, 'noise')
    heading = 'Ceiling over ' + unit
    if heading not in text:
        raise SystemExit('no block %r in %s' % (heading, relative(file_path(source, 'noise'))))
    block = text.split(heading)[1]
    match = re.search(re.escape(measure_label) + r'\s+(\d\.\d+)\s+(\d\.\d+)', block)
    if match is None:
        raise SystemExit('could not find %r under %r in %s' % (measure_label, heading,
                                                               relative(file_path(source, 'noise'))))
    return float(match.group(1)), float(match.group(2))


def image_counts(source):
    """(images embedded, images with memorability scores) from the first lines of the crowding report,
    e.g. '10,442 images embedded; 2,222 carry memorability scores.'"""
    text = read_text(source, 'crowding')
    match = re.search(r'([\d,]+) (?:faces|images) embedded; ([\d,]+) carry memorability scores', text)
    if match is None:
        raise SystemExit('could not find the image counts in %s' % relative(file_path(source, 'crowding')))
    return int(match.group(1).replace(',', '')), int(match.group(2).replace(',', ''))


def number_of_spaces(source):
    """The number of network/layer spaces in the ensemble, from the crowding report."""
    text = read_text(source, 'crowding')
    match = re.search(r'Ensemble of (\d+) network/layer spaces', text)
    if match is None:
        raise SystemExit('could not find the number of spaces in %s' % relative(file_path(source, 'crowding')))
    return int(match.group(1))


def faces_source(quiet=False):
    """'faces_regenerated' when the face database archives are in external/bainbridge_faces/ (and the
    faces chain has written outputs/faces/), otherwise 'faces_provided', after printing
    'PROVIDED OUTPUT: regenerating needs ...; see EXTERNAL_INPUTS.md'."""
    if quiet:
        present = release_paths.input_is_present(FACES_INPUT)
        mode = 'regenerate' if present else 'provided'
    else:
        mode = release_paths.regeneration_mode([FACES_INPUT])
    if mode == 'provided':
        return 'faces_provided'
    if not os.path.exists(file_path('faces_regenerated', 'ridge')):
        raise SystemExit('The face database archives are in external/bainbridge_faces/ but outputs/faces/ has no '
                         'results yet: run python A6a_faces_and_scenes_figS15/code/run_module.py first.')
    return 'faces_regenerated'


def scene_reliability_source(quiet=False):
    """Where the scene reliability over participants (the printed 0.73) comes from: the noise-ceiling
    report of this run when it has the block over participants (the participant matrices were
    available), otherwise the saved report in data/paper_run_scenes/, after printing the
    provided-output message."""
    rerun_report = file_path('scenes_rerun', 'noise')
    if os.path.exists(rerun_report):
        with open(rerun_report, encoding='utf-8') as handle:
            if 'Ceiling over participants' in handle.read():
                return 'scenes_rerun'
    if not quiet:
        print(release_paths.PROVIDED_MESSAGE % ('the Isola participant matrices for the scene reliability over '
                                              'participants (data/scenes/subject_matrices.mat, or external/isola/Data/ '
                                              'to rebuild them; M8); the value is read from data/paper_run_scenes/'))
    return 'scenes_paper_run'


def subject_matrices_path():
    """The Isola participant response matrices (843 participants x 2,400 targets) for the scene
    reliability over participants: the shipped copy in data/scenes/, else a copy
    rebuilt from the public package into external/isola/ (code/run_module.py does that with
    shared/embedding_extraction/export_subjects.m when external/isola/Data/ is present), else None."""
    shipped = os.path.join(MODULE_FOLDER, 'data', 'scenes', 'subject_matrices.mat')
    rebuilt = os.path.join(release_paths.release_root(), 'external', 'isola', 'subject_matrices.mat')
    if os.path.exists(shipped):
        return shipped
    if os.path.exists(rebuilt):
        return rebuilt
    return None
