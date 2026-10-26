"""
face_study_common.py: what the scripts of module A3c share (SI Appendix, Section 3, the rare-feature
face study: REM predictions and the preregistered d_a analysis; Fig. 4c and SI Appendix, Figs. S10
and S11b).

1. THE PARTICIPANTS' TEST TRIALS, AS THE FITTING SCRIPTS READ THEM
   The fitting scripts (sdt_shared_criteria.py, bootstrap_parallel.py, bootstrap_items.py,
   leave_one_pair_out.py, rem_like_data.py) import this file as F:
       subs = F.analysable(F.load())     one record per registered participant, in load_order;
                                         record['rec'] holds the 36 recognition trials as
                                         dictionaries, in the order they were run
       F.rating(trial)                   the old/new answer and the 1 to 4 confidence rating as one
                                         8-step rating (1 = surest new ... 8 = surest old)
   load(), analysable() and rating() read the de-identified table
   04_rare_feature_faces_fig4/data/face_trials.csv through shared/python/face_loader.py, with the
   participants in load_order (the column load_order, the participant order of the registered
   analysis; the registered bootstrap draws participants in this order) and the trials in the order
   they were run.
   Fields of each trial dictionary: task, practice, image, kind, condition ('rare' or 'filler'),
   feature and cell (the rare feature and its pair c1 to c6; '' for ordinary faces), test_type
   ('old', 'foil' or 'new'), is_old, said_old, confidence (1 to 4), and for rare-feature faces
   studied_image and role_swapped.

2. THE MEMTOOLBOX2 SWITCH
   The unequal-variance fit uses the MemToolbox2 Python package (UVSDModel, ByCell, MLE;
   EXTERNAL_INPUTS.md, item M5), which this repository does not include. import_memtoolbox() looks
   for it ONLY in shared/memtoolbox2/ (as shared/memtoolbox2/memtoolbox/ or
   shared/memtoolbox2/python/memtoolbox/). It returns the package, or None after printing
       MISSING PACKAGE: ...
       PROVIDED OUTPUT: regenerating needs ...; see EXTERNAL_INPUTS.md
   A memtoolbox package installed anywhere else is refused, never used in its place.
   When the environment variable A3c_PROVIDED_ONLY is 1 (run_module.py --provided-only sets it for
   every step it starts), import_memtoolbox() returns None even when the package is there, so every
   script uses the saved results in data/provided/.

3. READING THE SAVED TEXT REPORTS BY ROW LABEL
   find_row() and numbers_in() read a row of a saved report by the words it starts with, never by
   its line number.

Nothing here draws random numbers or writes files.
"""
import os
import re
import sys

sys.dont_write_bytecode = True

import numpy as np

_CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_CODE_FOLDER, '..', '..', 'shared', 'python'))
import release_paths   # noqa: E402
import face_loader   # noqa: E402

# ----------------------------------------------------------------------------- the registered rule
# The preregistered exclusion rule: at least 4 of the 6 one-back repeats detected, at most 5 presses
# on study faces that were not repeats, and 36 test trials.
MIN_ONEBACK_HITS = 4          # of the 6 one-back repeats
MAX_ONEBACK_FA = 5            # presses on study trials that were not repeats
EXPECTED_RECOGNITION_TRIALS = 36


def _text_or_empty(value):
    """A text cell of the CSV; an empty cell (read as NaN) becomes ''."""
    if isinstance(value, float) and np.isnan(value):
        return ''
    return str(value)


def load():
    """One record per registered participant, in load_order (the participant order of the registered
    analysis; the registered bootstrap draws participants in this order). Each record is a
    dictionary with
        participant  the participant id (S001 ... S111)
        rec          the participant's 36 recognition trials, as dictionaries, in the order run
        n_rec        the number of recognition trials
        ob_hits      one-back repeats detected (of n_reps)
        ob_fa        presses on study trials that were not repeats
        n_reps       one-back repeats shown (6)
    """
    all_trials = face_loader.load_trials('all')
    registered_in_load_order = face_loader.registered_participants(order='load_order')
    records = []
    for participant in registered_in_load_order:
        own_trials = all_trials[all_trials['participant'] == participant]
        study_trials = own_trials[own_trials['phase'] == 'study']
        test_trials = own_trials[own_trials['phase'] == 'recognition'].sort_values('trial_in_phase')

        # the one-back task during study (the exclusion rule)
        number_of_repeats = 0
        oneback_hits = 0
        oneback_false_presses = 0
        for _, study_trial in study_trials.iterrows():
            if study_trial['is_repeat'] == 1:
                number_of_repeats += 1
                if study_trial['oneback_correct'] == 1:
                    oneback_hits += 1
            elif study_trial['pressed'] == 1:
                oneback_false_presses += 1

        # the recognition trials, as dictionaries with the fields listed at the top of this file
        recognition_trials = []
        for _, test_trial in test_trials.iterrows():
            trial = {'task': 'recognition',
                     'practice': False,
                     'image': test_trial['image'],
                     'kind': test_trial['kind'],
                     'condition': test_trial['condition'],
                     'feature': _text_or_empty(test_trial['feature']),
                     'cell': _text_or_empty(test_trial['cell']),
                     'test_type': test_trial['test_type'],
                     'is_old': bool(int(test_trial['is_old'])),
                     'said_old': bool(int(test_trial['said_old'])),
                     'confidence': int(test_trial['confidence_1to4'])}
            if test_trial['condition'] == 'rare':
                trial['studied_image'] = test_trial['studied_image']
                trial['role_swapped'] = bool(int(test_trial['role_swapped']))
            recognition_trials.append(trial)

        records.append({'participant': participant,
                        'rec': recognition_trials,
                        'n_rec': len(recognition_trials),
                        'ob_hits': oneback_hits,
                        'ob_fa': oneback_false_presses,
                        'n_reps': number_of_repeats})
    return records


def exclusions(record):
    """Why a participant is excluded under the registered rule, or an empty list."""
    reasons = []
    if record['n_rec'] != EXPECTED_RECOGNITION_TRIALS:
        reasons.append('%d recognition trials, expected %d' % (record['n_rec'], EXPECTED_RECOGNITION_TRIALS))
    if record['ob_hits'] < MIN_ONEBACK_HITS:
        reasons.append('1-back hits %d of %d' % (record['ob_hits'], record['n_reps']))
    if record['ob_fa'] > MAX_ONEBACK_FA:
        reasons.append('1-back false alarms %d' % record['ob_fa'])
    return reasons


def analysable(records):
    """The records that pass the registered rule (all 100 registered participants do)."""
    kept = []
    for record in records:
        if not exclusions(record):
            kept.append(record)
    return kept


def rating(trial):
    """The old/new answer and the 1 to 4 confidence rating as one 8-step rating:
    an 'old' answer gives 5 (Guess) to 8 (Certain), a 'new' answer 4 (Guess) to 1 (Certain)."""
    if trial['said_old']:
        return 5 + (trial['confidence'] - 1)
    return 5 - trial['confidence']


# ----------------------------------------------------------------------------- the MemToolbox2 switch
MEMTOOLBOX_FOLDER = release_paths.shared_path('memtoolbox2')
MEMTOOLBOX_INPUT = release_paths.Input(
    'M5', 'the MemToolbox2 Python package (the parts UVSDModel, ByCell and MLE)',
    'shared/memtoolbox2/memtoolbox')
MISSING_PACKAGE_MESSAGE = ('MISSING PACKAGE: MemToolbox2 (Python; the unequal-variance '
                           'fit uses UVSDModel, ByCell and MLE) is not in shared/memtoolbox2/; '
                           'see EXTERNAL_INPUTS.md, item M5')
# run_module.py --provided-only sets this environment variable to 1 for every step it starts
PROVIDED_ONLY_VARIABLE = 'A3c_PROVIDED_ONLY'


def provided_only_asked():
    """True when the provided output was asked for (run_module.py --provided-only)."""
    return os.environ.get(PROVIDED_ONLY_VARIABLE, '') == '1'


def import_memtoolbox(quiet=False):
    """The memtoolbox package from shared/memtoolbox2/, or None when it is not there (or when
    the provided output was asked for: run_module.py --provided-only).
    A memtoolbox package found anywhere else is refused (never a silent substitute)."""
    if provided_only_asked():
        if not quiet:
            print('PROVIDED OUTPUT asked for (run_module.py --provided-only): MemToolbox2 is not used, even if it '
                  'is in shared/memtoolbox2/; the saved results in data/provided/ are used')
        return None
    candidate_folders = [MEMTOOLBOX_FOLDER, os.path.join(MEMTOOLBOX_FOLDER, 'python')]
    for folder in reversed(candidate_folders):
        if os.path.isdir(folder) and folder not in sys.path:
            sys.path.insert(0, folder)
    try:
        import memtoolbox
    except ImportError as import_error:
        if not quiet:
            # A memtoolbox folder that is there but does not import (for example, a module it needs is
            # missing) is named with the import error, rather than reported as absent.
            folders_with_the_package = []
            for folder in candidate_folders:
                if os.path.isdir(os.path.join(folder, 'memtoolbox')):
                    folders_with_the_package.append(folder)
            if folders_with_the_package:
                print('MISSING PACKAGE: shared/%s/memtoolbox/ is there but could not be imported (%s); see '
                      'EXTERNAL_INPUTS.md, item M5' % (os.path.relpath(folders_with_the_package[0],
                                                                      release_paths.shared_path()), import_error))
            else:
                print(MISSING_PACKAGE_MESSAGE)
            print(release_paths.PROVIDED_MESSAGE % ('%s (%s, expected at %s)' % (
                MEMTOOLBOX_INPUT.description, MEMTOOLBOX_INPUT.missing_id, MEMTOOLBOX_INPUT.relative_path)))
        return None
    package_file = os.path.realpath(getattr(memtoolbox, '__file__', '') or '')
    if not package_file.startswith(os.path.realpath(MEMTOOLBOX_FOLDER) + os.sep):
        # Another memtoolbox on the Python path is never used in place of the copy in shared/memtoolbox2/.
        del sys.modules['memtoolbox']
        if not quiet:
            print('A memtoolbox package was found at %s, outside shared/memtoolbox2/; it is not used '
                  '(only a copy in shared/memtoolbox2/ is used; EXTERNAL_INPUTS.md, item M5)'
                  % package_file)
            print(MISSING_PACKAGE_MESSAGE)
            print(release_paths.PROVIDED_MESSAGE % ('%s (%s, expected at %s)' % (
                MEMTOOLBOX_INPUT.description, MEMTOOLBOX_INPUT.missing_id, MEMTOOLBOX_INPUT.relative_path)))
        return None
    for needed in ('UVSDModel', 'ByCell', 'MLE'):
        if not hasattr(memtoolbox, needed):
            raise SystemExit('shared/memtoolbox2: the memtoolbox package has no %s; see EXTERNAL_INPUTS.md, '
                             'item M5' % needed)
    if not quiet:
        print('MemToolbox2 found: %s' % os.path.relpath(os.path.dirname(package_file), release_paths.release_root()))
    return memtoolbox


def report_provided_files(script_name, file_names, show_contents=False):
    """Says which saved files stand in for a run that needs MemToolbox2, and that they exist.
    With show_contents=True the text reports (.txt) among them are printed too."""
    paths = release_paths.module_paths(os.path.join(_CODE_FOLDER, script_name), make_output_folders=False)
    print('%s: the saved results in data/provided/ are used instead:' % script_name)
    for file_name in file_names:
        path = os.path.join(paths['provided'], file_name)
        if not os.path.exists(path):
            raise SystemExit('missing provided file data/provided/%s' % file_name)
        print('  data/provided/%s' % file_name)
        if show_contents and file_name.endswith('.txt'):
            with open(path, encoding='utf-8') as handle:
                for line in handle.read().splitlines():
                    print('    | ' + line)
    print('check_provided.py recomputes every number of these files from the saved fits and resamples.')


# ----------------------------------------------------------------------------- reading saved reports
NUMBER_PATTERN = r'[-+]?\d+(?:\.\d+)?'


def numbers_in(text):
    """Every number in a piece of text, in order ('+0.666  [+0.458, +0.917]' -> [0.666, 0.458, 0.917])."""
    return [float(number) for number in re.findall(NUMBER_PATTERN, text)]


def read_lines(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read().splitlines()


def find_line_containing(lines, phrase):
    """The first whole line that contains phrase (for a report's header sentence, which starts with a
    number rather than a label). Stops with a clear message when no line contains it."""
    for line in lines:
        if phrase in line:
            return line
    raise ValueError('no line contains %r' % phrase)


def find_row(lines, label, after=None):
    """The text of the first line that starts with label (leading spaces ignored), searching after the
    first line that starts with 'after' when it is given. Returns the rest of the line after the label.
    Stops with a clear message when no such line exists."""
    start = 0
    if after is not None:
        start = None
        for line_number, line in enumerate(lines):
            if line.strip().startswith(after):
                start = line_number + 1
                break
        if start is None:
            raise ValueError('no line starts with %r' % after)
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith(label):
            return stripped[len(label):]
    raise ValueError('no line starts with %r%s' % (label, '' if after is None else ' after %r' % after))
