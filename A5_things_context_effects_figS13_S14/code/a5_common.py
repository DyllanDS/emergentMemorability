"""
a5_common.py: folders, inputs and concept tables shared by every script of module A5
(SI Appendix, Section 5, "Context Effects in the Original THINGS Data", Figs. S13 and S14).

WHAT IT PROVIDES
    MODULE, PROVIDED, REGENERATED, OUTPUTS ...   the module's folders
    DERIVED                                      where the trial tables go (see below)
    run_mode()                                   'provided', 'from_trial_table' or 'from_raw_file'
    report_folder(source)                        the folder the reports are read from
    trial_intervals(), write_trial_intervals()   data/provided/trial_intervals.csv (see below)
    concept_categories_alphabetical()            the category of each concept, in the concept
                                                 numbering of the Kramer et al. files
    spose_file(), image_mapping_file(), things_images_file()

THE THREE PLACES RESULTS LIVE
    data/provided/          the saved reports of the analyses (aggregate numbers only: per-bin rates,
                            per-category summaries, regression summaries). Used when the Kramer et
                            al. trial file is not available (EXTERNAL_INPUTS.md, M1). Its
                            trial_intervals.csv holds the 95% intervals that only the trial tables
                            give (one row per statistic); stats_section5.py writes it when the trial
                            file is available and reads it when it is not.
    outputs/regenerated/    the same reports, written by this module's code when the trial file is
                            available (aggregate numbers only, like data/provided/).
    external/kramer_trials/derived/
                            the trial tables made from the trial file (one row per trial, or per
                            participant): trials.npz, trials_with_context.npz, context_k.npz,
                            category_crowding.npz, kramer_quartile.npz and
                            fit_on_sequence_heldout_counts.npz. They stay next to the trial file and
                            are not distributed.

INPUTS (relative to the repository root)
    external/kramer_trials/imbysub-all-11-16-21.mat   the Kramer et al. (2023) trial file (M1)
    shared/things/spose_embedding_49d_sorted.txt       the 49-dimensional SPoSE embedding
    shared/things/things_image_mapping.csv             image number -> concept number (Kramer files)
    shared/things/things_images.csv                    image number -> file name, embedding row
    shared/things/concept_categories.csv               the higher-level category of each concept
    shared/embeddings_things/<network>_<layer>.npz     the 14 deep-network spaces (M3)
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
SHARED_PYTHON = CODE.parent.parent / 'shared' / 'python'
if str(SHARED_PYTHON) not in sys.path:
    sys.path.insert(0, str(SHARED_PYTHON))
import release_paths  # noqa: E402

PATHS = release_paths.module_paths(__file__)
RELEASE_ROOT = Path(release_paths.release_root())
MODULE = Path(PATHS['module'])
PROVIDED = Path(PATHS['provided'])
OUTPUTS = Path(PATHS['outputs'])
FIGURES = Path(PATHS['figures'])
FIGURE_DATA = Path(PATHS['figure_data'])
LOGS = Path(PATHS['logs'])
REGENERATED = OUTPUTS / 'regenerated'

RAW_TRIAL_FILE_RELATIVE = 'external/kramer_trials/imbysub-all-11-16-21.mat'
DERIVED_RELATIVE = 'external/kramer_trials/derived'
DERIVED = RELEASE_ROOT / DERIVED_RELATIVE
TRIAL_TABLE = DERIVED / 'trials.npz'

# The Kramer et al. trial file, as EXTERNAL_INPUTS.md names it.
RAW_TRIAL_INPUT = release_paths.Input(
    'M1', 'the Kramer et al. (2023) trial file imbysub-all-11-16-21.mat (from the original authors)',
    RAW_TRIAL_FILE_RELATIVE)

# The 14 deep-network spaces (seven networks, an early and a late layer each). This repository
# includes the four late layers that Fig. S8 needs; the other ten are input M3 (EXTERNAL_INPUTS.md).
NETWORK_SPACES = ['alexnet_early', 'alexnet_late', 'clip_early', 'clip_late', 'cornet_early',
                  'cornet_late', 'dinov3_early', 'dinov3_late', 'inception_early', 'inception_late',
                  'resnet18_early', 'resnet18_late', 'vgg16_early', 'vgg16_late']
EMBEDDING_INPUTS = [release_paths.Input('M3', 'the THINGS deep-network space %s' % name,
                                      'shared/embeddings_things/%s.npz' % name)
                    for name in NETWORK_SPACES]

# The saved report files; a run from the trial file writes the same names to outputs/regenerated/.
REPORT_FILES = ['sdt_results.txt', 'sdt_ctx_top10_target.npz', 'sdt_by_context_adjusted.txt',
                'sdt_by_context_adjusted.npz', 'kramer_quartile.txt', 'kramer_quartile_summary.csv',
                'crowding.txt', 'lone_exemplar.txt', 'fit_on_sequence.txt', 'fit_on_sequence.json']
DEEP_NETWORK_REPORTS = ['fit_on_sequence.txt', 'fit_on_sequence.json']

# The 95% intervals that only the trial tables give (stats_from_trials.py: 2,000 participant-session
# resamples, seed 20261013), one row per statistic of tests T38, T39, T41 and T42.
TRIAL_INTERVALS = PROVIDED / 'trial_intervals.csv'
TRIAL_INTERVAL_COLUMNS = ['test_id', 'key', 'statistic_name', 'statistic', 'ci_low', 'ci_high', 'n_resamples',
                          'seed']


def relative(path):
    """A path written relative to the repository root (for messages and logs)."""
    try:
        return str(Path(path).resolve().relative_to(RELEASE_ROOT.resolve()))
    except ValueError:
        return str(path)


def spose_file():
    return Path(release_paths.shared_path('things', 'spose_embedding_49d_sorted.txt'))


def image_mapping_file():
    return Path(release_paths.shared_path('things', 'things_image_mapping.csv'))


def things_images_file():
    return Path(release_paths.shared_path('things', 'things_images.csv'))


def embeddings_folder():
    return Path(release_paths.shared_path('embeddings_things'))


def make_derived_folder():
    DERIVED.mkdir(parents=True, exist_ok=True)
    return DERIVED


def make_regenerated_folder():
    REGENERATED.mkdir(parents=True, exist_ok=True)
    return REGENERATED


def run_mode(quiet=False):
    """How this module can run on this machine.
        'from_trial_table'  external/kramer_trials/derived/trials.npz exists (written by
                            extract_trials.py from the trial file): the analyses are recomputed
                            from it and extraction is skipped
        'from_raw_file'     the trial file is in external/kramer_trials/: extract_trials.py runs
                            first, then the analyses
        'provided'          neither is there: the saved reports in data/provided/ are used, and the
                            provided-output message is printed (never a silent fallback)"""
    if TRIAL_TABLE.exists():
        if not quiet:
            print('trial table found: %s (made by extract_trials.py from the Kramer et al. trial file)'
                  % relative(TRIAL_TABLE))
        return 'from_trial_table'
    if release_paths.regeneration_mode([RAW_TRIAL_INPUT], quiet=quiet) == 'regenerate':
        return 'from_raw_file'
    return 'provided'


def embeddings_present(quiet=False):
    """True when all 14 deep-network spaces are in shared/embeddings_things/
    (otherwise prints the provided-output message for each missing one)."""
    return release_paths.regeneration_mode(EMBEDDING_INPUTS, quiet=quiet) == 'regenerate'


def default_source():
    """'regenerated' when the trial data are available on this machine, else 'provided'."""
    if run_mode(quiet=True) == 'provided':
        return 'provided'
    return 'regenerated'


def report_folder(source):
    """The folder the Section 5 reports are read from: data/provided/ (the saved reports) or
    outputs/regenerated/ (this module's run from the trial file). Stops with a clear message when a
    regenerated report is missing (no silent fallback to the provided one)."""
    if source == 'provided':
        return PROVIDED
    if source != 'regenerated':
        raise ValueError("source must be 'provided' or 'regenerated', not %r" % (source,))
    missing = [name for name in REPORT_FILES if name not in DEEP_NETWORK_REPORTS
               and not (REGENERATED / name).exists()]
    if missing:
        raise SystemExit('outputs/regenerated/ lacks %s; run python %s/code/run_module.py first'
                         % (', '.join(missing), MODULE.name))
    return REGENERATED


def deep_network_report_folder(source):
    """The deep-network analysis needs the 14 deep-network spaces as well as the trial file (M1 and M3).
    Its report is read from outputs/regenerated/ only when that run exists; otherwise the provided one
    is used and the provided-output message says why."""
    if source == 'regenerated' and all((REGENERATED / name).exists() for name in DEEP_NETWORK_REPORTS):
        return REGENERATED
    if source == 'regenerated':
        print(release_paths.PROVIDED_MESSAGE % ('a run of fit_on_sequence.py with all 14 THINGS deep-network spaces '
                                              '(M3) for the deep-network analysis (outputs/regenerated/ has no '
                                              'deep-network report); the saved deep-network report in data/provided/ '
                                              'is used'))
    return PROVIDED


def write_trial_intervals(rows):
    """Write data/provided/trial_intervals.csv: one row per statistic (dictionaries with the keys of
    TRIAL_INTERVAL_COLUMNS), numbers with 6 significant digits."""
    with open(TRIAL_INTERVALS, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(TRIAL_INTERVAL_COLUMNS)
        for row in rows:
            cells = []
            for column in TRIAL_INTERVAL_COLUMNS:
                value = row[column]
                if isinstance(value, float):
                    value = '%.6g' % value
                cells.append(value)
            writer.writerow(cells)
    print('wrote %s' % relative(TRIAL_INTERVALS))


def trial_intervals():
    """data/provided/trial_intervals.csv as {(test id, key): {'statistic_name', 'statistic', 'ci_low',
    'ci_high', 'n_resamples', 'seed'}}. Stops with a message when the file is missing."""
    if not TRIAL_INTERVALS.exists():
        raise SystemExit('%s is missing: it holds the 95%% intervals that need the Kramer et al. trial file '
                         '(EXTERNAL_INPUTS.md, M1); with the trial file, stats_section5.py writes it'
                         % relative(TRIAL_INTERVALS))
    intervals = {}
    with open(TRIAL_INTERVALS, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            intervals[(row['test_id'], row['key'])] = dict(
                statistic_name=row['statistic_name'], statistic=float(row['statistic']),
                ci_low=float(row['ci_low']), ci_high=float(row['ci_high']),
                n_resamples=int(row['n_resamples']), seed=int(row['seed']))
    return intervals


def concept_categories_alphabetical():
    """The higher-level THINGS category of every concept, as a table with one row per concept in the
    concept numbering of the Kramer et al. files (row k = concept number k + 1 of
    things_image_mapping.csv, the alphabetical order of the concept names), with the columns the
    Section 5 scripts read:
        concept, category (0 = no named category, 'NA'), category_label
    Built from shared/things/concept_categories.csv by concept name; that file lists the concepts in
    the same order."""
    categories = pd.read_csv(release_paths.shared_path('things', 'concept_categories.csv'), keep_default_na=False)
    images = pd.read_csv(things_images_file(), keep_default_na=False, na_values=['NaN'])
    number_of_concept = {}
    for concept, number in zip(images['concept'], images['concept_number_ascii']):
        number_of_concept[concept] = int(number)
    rows = []
    for concept, category, label in zip(categories['concept'], categories['category_number'],
                                        categories['category_label']):
        rows.append((number_of_concept[concept], concept, int(category), label))
    rows.sort()
    numbers = [row[0] for row in rows]
    if numbers != list(range(1, len(rows) + 1)):
        raise SystemExit('the concept numbers of things_images.csv are not 1 to %d' % len(rows))
    table = pd.DataFrame({'concept': [row[1] for row in rows],
                          'category': np.array([row[2] for row in rows], dtype=np.int64),
                          'category_label': [row[3] for row in rows]})
    return table
