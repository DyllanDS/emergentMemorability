"""
a3b_common.py: what the scripts of module A3b share (SI Appendix, Fig. S9; Sections 1 and 3).

WHAT IT PROVIDES
    PATHS, RELEASE_ROOT        the module's folders (from shared/python/release_paths.py)
    LEVELS                     the designed numbers of similar items on the study list: 0, 1, 2, 4, 8
    critical_items()           the 40 critical items of the memorability inversion experiment over the
                               50 kept participants, with each item's hit rate, false alarm rate,
                               corrected recognition and d′
    number_of_kept_participants()
                               the number of participants kept after the exclusions (50)
    panel_c_mode()             how Fig. S9c can be made on this machine: 'regenerate' when the Kramer
                               et al. trial data are here (input M1 of EXTERNAL_INPUTS.md), else
                               'provided'
    panel_c_report_path(mode)  the per_image_rates.txt that Fig. S9c is read from
    read_per_image_rates(path) the numbers of a per_image_rates.txt, read by row label
    halves_mode()              how the THINGS side of the halves comparison can be made: 'regenerate'
                               when the per-image count table is here (input M2 of EXTERNAL_INPUTS.md),
                               else 'provided'
    read_halves_check()        the saved halves comparison (data/provided/halves_check.txt)
    things_rates_from_count_table(image_names)
                               the hit and false alarm rate of each named image in the original THINGS
                               data, from the per-image count table (M2)
    read_paper_run_numbers()   the numbers of the saved run of code/make_figureS9.py
                               (data/paper_run/figS9_numbers.txt)

WHERE THE INPUTS COME FROM (paths relative to the repository root)
    03_inversion_experiment_fig3/data/inversion_trials.csv and inversion_design.csv
        the 54 sessions of the inversion experiment (de-identified) and its 40 critical items, read
        through shared/python/inversion_loader.py, which applies the exclusions described in Methods
        (4 participants excluded, 50 kept)
    03_inversion_experiment_fig3/outputs/inversion_items_50.csv (only when module 03 has run)
        module 03's per-item table; critical_items() checks that it is identical
    A3b_hits_and_false_alarms_figS9/data/provided/per_image_rates.txt and halves_check.txt
    A3b_hits_and_false_alarms_figS9/data/paper_run/figS9_numbers.txt
    external/kramer_trials/ (empty in the repository; inputs M1 and M2 of EXTERNAL_INPUTS.md)
    shared/things/things_images.csv (image number of each THINGS photograph; used only with M2)

Nothing here draws random numbers.
"""
import os
import re
import sys

sys.dont_write_bytecode = True
CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
SHARED_PYTHON = os.path.join(CODE_FOLDER, '..', '..', 'shared', 'python')
if SHARED_PYTHON not in sys.path:
    sys.path.insert(0, SHARED_PYTHON)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import release_paths  # noqa: E402  (first of the shared helpers: it stops Python writing __pycache__ folders)
import inversion_loader  # noqa: E402
import sdt  # noqa: E402

PATHS = release_paths.module_paths(__file__)
RELEASE_ROOT = release_paths.release_root()

# The design: each block of the inversion experiment had five critical items, one from each fifth of
# the published memorability range, studied with 0, 1, 2, 4 or 8 similar items.
LEVELS = [0, 1, 2, 4, 8]
ITEMS_PER_LEVEL = 8                  # one critical item per level in each of the 8 blocks

# The Kramer et al. (2023) trial data (input M1 of EXTERNAL_INPUTS.md). Module A5's extract step
# (A5_things_context_effects_figS13_S14/code/extract_trials.py) turns the trial file into the trial
# table below; Fig. S9c is computed from that table.
TRIAL_FILE = release_paths.Input(
    'M1', 'the Kramer et al. (2023) trial file imbysub-all-11-16-21.mat (from the original authors)',
    'external/kramer_trials/imbysub-all-11-16-21.mat')
TRIAL_TABLE_RELATIVE = 'external/kramer_trials/derived/trials.npz'
EXTRACT_STEP_RELATIVE = 'A5_things_context_effects_figS13_S14/code/extract_trials.py'

# The per-image count table (input M2 of EXTERNAL_INPUTS.md), the same file module A4b uses: one row per
# THINGS photograph, with counts whose old trials include the short-lag vigilance repeats. It gives the
# THINGS side of the halves comparison; Fig. S9c counts the long-lag target repeats only.
COUNT_TABLE = release_paths.Input(
    'M2', 'the per-image THINGS count table things_per_image_counts.csv (image_nr, n_old, hits, n_new, '
          'false_alarms) from the Kramer et al. (2023) trial data',
    'external/kramer_trials/things_per_image_counts.csv')

MODULE_03_ITEMS_RELATIVE = os.path.join('03_inversion_experiment_fig3', 'outputs', 'inversion_items_50.csv')
PAPER_RUN_FILE = os.path.join(PATHS['paper_run'], 'figS9_numbers.txt')
HALVES_FILE = os.path.join(PATHS['provided'], 'halves_check.txt')
PROVIDED_RATES_FILE = os.path.join(PATHS['provided'], 'per_image_rates.txt')
REGENERATED_FOLDER = os.path.join(PATHS['outputs'], 'regenerated')
REGENERATED_RATES_FILE = os.path.join(REGENERATED_FOLDER, 'per_image_rates.txt')


def relative(path):
    """A path written relative to the repository root (for messages and logs)."""
    absolute_path = os.path.abspath(path)
    if absolute_path.startswith(RELEASE_ROOT + os.sep):
        return os.path.relpath(absolute_path, RELEASE_ROOT)
    return path


# ------------------------------------------------------------------------------------------------
# The 40 critical items of the inversion experiment
# ------------------------------------------------------------------------------------------------
def critical_items(check_against_module_03=True):
    """The 40 critical items over the 50 kept participants, sorted by image name (the order of module
    03's per-item table), with the columns of shared/python/inversion_loader.items_table():
        image_name, block, memorability_quintile, n_similar_items, kramer_cr (the published Kramer et
        al. score), n_old, hits, n_new, false_alarms, hit_rate, fa_rate, cr (hit rate minus false alarm
        rate)
    and one more:
        dprime   d′ from log-linear corrected rates:
                 z((hits + 0.5) / (n_old + 1)) - z((false alarms + 0.5) / (n_new + 1))
    'Old' is a response of 1 to 3 on the six-point confidence scale."""
    items = inversion_loader.items_table()
    dprime, _criterion = sdt.dprime_loglinear_counts(items['hits'].to_numpy(dtype=float),
                                                     items['n_old'].to_numpy(dtype=float),
                                                     items['false_alarms'].to_numpy(dtype=float),
                                                     items['n_new'].to_numpy(dtype=float))
    items['dprime'] = dprime

    # The design: 8 critical items at each of the 5 levels, every item both old and new for some
    # participants, every participant tested on every item once.
    if sorted(items['n_similar_items'].unique()) != LEVELS:
        raise SystemExit('the design table does not have the levels %s' % LEVELS)
    for level in LEVELS:
        if int(np.sum(items['n_similar_items'] == level)) != ITEMS_PER_LEVEL:
            raise SystemExit('level %d does not have %d critical items' % (level, ITEMS_PER_LEVEL))
    if items['n_old'].min() == 0 or items['n_new'].min() == 0:
        raise SystemExit('a critical item was never old or never new')

    if check_against_module_03:
        compare_with_module_03(items)
    return items


def compare_with_module_03(items):
    """When module 03 has run, its per-item table (written by MATLAB at full precision) must equal the
    table above: counts and rates exactly, the published score to within 1e-12 (module 03 stores the
    score as the mean over the item's test trials, which can differ in the last digit)."""
    path = os.path.join(RELEASE_ROOT, MODULE_03_ITEMS_RELATIVE)
    if not os.path.exists(path):
        print('%s not found (module 03 has not run here): the equality check with module 03 is skipped'
              % MODULE_03_ITEMS_RELATIVE)
        return
    module_03_items = pd.read_csv(path, float_precision='round_trip')
    if list(module_03_items['image_name']) != list(items['image_name']):
        raise SystemExit('the items of %s differ from the shared loader' % MODULE_03_ITEMS_RELATIVE)
    exact_columns = ['block', 'memorability_quintile', 'n_similar_items', 'n_old', 'hits', 'n_new', 'false_alarms',
                     'hit_rate', 'fa_rate', 'cr']
    for column in exact_columns:
        if not np.array_equal(module_03_items[column].to_numpy(dtype=float), items[column].to_numpy(dtype=float)):
            raise SystemExit('column %s of %s differs from the shared loader' % (column, MODULE_03_ITEMS_RELATIVE))
    largest_score_difference = float(np.max(np.abs(module_03_items['kramer_cr'].to_numpy(dtype=float)
                                                    - items['kramer_cr'].to_numpy(dtype=float))))
    if largest_score_difference > 1e-12:
        raise SystemExit('kramer_cr of %s differs from the published scores' % MODULE_03_ITEMS_RELATIVE)
    print('%s equals the shared loader\'s 40-item table (counts and rates identical; published score '
          'within %.1e)' % (MODULE_03_ITEMS_RELATIVE, largest_score_difference))


def number_of_kept_participants():
    """The number of participants kept after the exclusions described in Methods
    (shared/python/inversion_loader.py)."""
    test_trials = inversion_loader.load_trials('test')
    kept_trials, _flags = inversion_loader.apply_exclusions(test_trials)
    return len(kept_trials['participant'].unique())


# ------------------------------------------------------------------------------------------------
# Fig. S9c: the per-image rates of the original THINGS data
# ------------------------------------------------------------------------------------------------
def trial_table_path():
    return os.path.join(RELEASE_ROOT, TRIAL_TABLE_RELATIVE)


def panel_c_mode(quiet=False):
    """'regenerate' when the Kramer et al. trial data are on this machine: the trial table that module
    A5's extract step writes (external/kramer_trials/derived/trials.npz) or the trial file itself
    (external/kramer_trials/imbysub-all-11-16-21.mat, M1). Otherwise 'provided', after the
    provided-output message (never a silent fallback to another input).
    The per-image count table (M2) does not switch Fig. S9c to regeneration: its old trials include the
    short-lag vigilance repeats, while Fig. S9c counts the long-lag target repeats only (see README)."""
    if os.path.exists(trial_table_path()):
        if not quiet:
            print('trial table found: %s (made by module A5\'s extract step from the Kramer et al. trial file)'
                  % TRIAL_TABLE_RELATIVE)
        return 'regenerate'
    return release_paths.regeneration_mode([TRIAL_FILE], quiet=quiet)


def panel_c_report_path(mode):
    """The per_image_rates.txt that Fig. S9c is read from: the saved one in data/provided/ or the one
    code/per_image_rates.py regenerated from the trial data. Stops with a clear message when the trial
    data are here but the regenerated file is not (no silent fallback to the saved file)."""
    if mode == 'provided':
        return PROVIDED_RATES_FILE
    if not os.path.exists(REGENERATED_RATES_FILE):
        raise SystemExit('the Kramer et al. trial data are here but %s is missing: run '
                         'python A3b_hits_and_false_alarms_figS9/code/per_image_rates.py first'
                         % relative(REGENERATED_RATES_FILE))
    return REGENERATED_RATES_FILE


def _search(pattern, text, what, path):
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise SystemExit('could not find "%s" in %s' % (what, relative(path)))
    return match


def _p_value(text):
    """'6.15e-36' -> 6.15e-36; '< 1e-300' (a p below double precision) -> 0.0."""
    text = text.strip()
    if text.startswith('<'):
        return 0.0
    return float(text)


def read_per_image_rates(path):
    """The numbers of a per_image_rates.txt (the report of code/per_image_rates.py: the per-image rates
    of the original THINGS data), each read by its row label. Returns a dict."""
    with open(path, encoding='utf-8') as handle:
        text = handle.read()
    values = {}
    match = _search(r'observations:\s+([\d,]+)', text, 'observations', path)
    values['images'] = int(match.group(1).replace(',', ''))
    match = _search(r'(?:>=|at least )(\d+) old', text, 'minimum old observations', path)
    values['minimum_observations'] = int(match.group(1))
    match = _search(r'median opportunities per image:\s+(\d+) old,\s+(\d+) new', text, 'median opportunities', path)
    values['median_old'] = int(match.group(1))
    values['median_new'] = int(match.group(2))
    match = _search(r'^hit rate\s+mean ([\d.]+)\s+SD ([\d.]+)', text, 'hit rate mean', path)
    values['hit_mean'] = float(match.group(1))
    values['hit_sd'] = float(match.group(2))
    match = _search(r'^false alarm\s+mean ([\d.]+)\s+SD ([\d.]+)', text, 'false alarm mean', path)
    values['fa_mean'] = float(match.group(1))
    values['fa_sd'] = float(match.group(2))
    match = _search(r'^corrected rec mean ([\d.]+)\s+SD ([\d.]+)', text, 'corrected recognition mean', path)
    values['cr_mean'] = float(match.group(1))
    values['cr_sd'] = float(match.group(2))
    match = _search(r'vs hit rate:\s+r = ([-+][\d.]+)\s+p = (.+)$', text, 'r with the hit rate', path)
    values['r_cr_hit'] = float(match.group(1))
    values['p_cr_hit_text'] = match.group(2).strip()
    values['p_cr_hit'] = _p_value(match.group(2))
    match = _search(r'vs false alarm rate:\s+r = ([-+][\d.]+)\s+p = (.+)$', text, 'r with the false alarm rate', path)
    values['r_cr_fa'] = float(match.group(1))
    values['p_cr_fa_text'] = match.group(2).strip()
    values['p_cr_fa'] = _p_value(match.group(2))
    return values


# ------------------------------------------------------------------------------------------------
# The halves comparison (the THINGS side needs the per-image count table, M2)
# ------------------------------------------------------------------------------------------------
def halves_mode(quiet=False):
    """'regenerate' when the per-image count table (M2) is in external/kramer_trials/, else 'provided'
    (after the provided-output message)."""
    return release_paths.regeneration_mode([COUNT_TABLE], quiet=quiet)


def read_halves_check():
    """The saved halves comparison (data/provided/halves_check.txt), read by row label.
    Returns {'more memorable half': {...}, 'less memorable half': {...}} with the false alarm and hit
    rates in THINGS ('fa_things', 'hit_things') and in this experiment ('fa_here', 'hit_here')."""
    with open(HALVES_FILE, encoding='utf-8') as handle:
        text = handle.read()
    halves = {}
    for name in ('more memorable half', 'less memorable half'):
        pattern = (name + r': false alarms ([\d.]+) in THINGS -> ([\d.]+) here; '
                   r'hits ([\d.]+) in THINGS -> ([\d.]+) here')
        match = _search(pattern, text, name, HALVES_FILE)
        halves[name] = dict(fa_things=float(match.group(1)), fa_here=float(match.group(2)),
                            hit_things=float(match.group(3)), hit_here=float(match.group(4)),
                            fa_things_text=match.group(1), hit_things_text=match.group(3))
    return halves


def things_rates_from_count_table(image_names):
    """Hit rate (hits / n_old) and false alarm rate (false_alarms / n_new) of each named photograph in
    the original THINGS data, from the per-image count table (M2). The image number of each photograph
    comes from shared/things/things_images.csv. Returns (hit_rates, false_alarm_rates) as arrays."""
    count_table_path = os.path.join(RELEASE_ROOT, COUNT_TABLE.relative_path)
    counts = pd.read_csv(count_table_path)
    needed_columns = ['image_nr', 'n_old', 'hits', 'n_new', 'false_alarms']
    for column in needed_columns:
        if column not in counts.columns:
            raise SystemExit('%s has no column %s (needed: %s)' % (COUNT_TABLE.relative_path, column,
                                                                   ', '.join(needed_columns)))
    row_of_image_number = {}
    for row, image_number in enumerate(counts['image_nr']):
        row_of_image_number[int(image_number)] = row
    images = pd.read_csv(release_paths.shared_path('things', 'things_images.csv'), keep_default_na=False,
                         na_values=['NaN'])
    image_number_of_name = {}
    for name, image_number in zip(images['filename'], images['image_nr']):
        image_number_of_name[name] = int(image_number)
    hit_rates = []
    false_alarm_rates = []
    for name in image_names:
        row = row_of_image_number[image_number_of_name[name]]
        hit_rates.append(counts['hits'].iloc[row] / counts['n_old'].iloc[row])
        false_alarm_rates.append(counts['false_alarms'].iloc[row] / counts['n_new'].iloc[row])
    return np.array(hit_rates, dtype=float), np.array(false_alarm_rates, dtype=float)


# ------------------------------------------------------------------------------------------------
# The saved run of code/make_figureS9.py
# ------------------------------------------------------------------------------------------------
def read_paper_run_numbers():
    """The numbers of the saved run (data/paper_run/figS9_numbers.txt), each read by its row label,
    as printed there (so that a comparison checks the same digits). Returns a dict:
        levels        {level: [hit, hit low, hit high, FA, FA low, FA high, memorability, d′, d′ low, d′ high]}
                      (memorability = hit rate minus false alarm rate)
        spearman      {'false alarms': (rho, p), 'hits': ..., 'd-prime': ...}
        trials_line   (fewest and most test trials per item, fewest old, fewest new)
        panel_b_text  the statistic printed on panel b"""
    with open(PAPER_RUN_FILE, encoding='utf-8') as handle:
        text = handle.read()
    numbers = {'levels': {}, 'spearman': {}}
    number = r'(-?[\d.]+)'
    interval = r'\[(-?[\d.]+), (-?[\d.]+)\]'
    level_pattern = (r'^\s+(\d+)\s+' + number + r' ' + interval + r'\s+' + number + r' ' + interval + r'\s+'
                     + number + r'\s+' + number + r' ' + interval + r'\s*$')
    for match in re.finditer(level_pattern, text, flags=re.MULTILINE):
        numbers['levels'][int(match.group(1))] = [match.group(index) for index in range(2, 12)]
    if sorted(numbers['levels']) != LEVELS:
        raise SystemExit('could not read the level table of %s' % relative(PAPER_RUN_FILE))
    for name in ('false alarms', 'hits', 'd-prime'):
        match = _search(re.escape(name) + r' (-?[\d.]+) \(p = ([\d.e+-]+)\)', text, name, PAPER_RUN_FILE)
        numbers['spearman'][name] = (match.group(1), match.group(2))
    match = _search(r'test trials per item (\d+) to (\d+); at least (\d+) old and (\d+) new', text,
                    'test trials per item', PAPER_RUN_FILE)
    numbers['trials_line'] = tuple(match.group(index) for index in range(1, 5))
    match = _search(r'printed on panel: (.+)$', text, 'printed on panel', PAPER_RUN_FILE)
    numbers['panel_b_text'] = match.group(1).strip()
    return numbers
