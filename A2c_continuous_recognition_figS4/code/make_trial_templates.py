"""
make_trial_templates.py: writes the trial structures the model of Fig. S4 runs on, as a small
structure-only file (400 sessions x 295 trials), from a trial table of the Kramer et al. (2023) data.

WHAT THE FILE HOLDS
  For each of the 400 sessions continuous_rem.py uses (the first 400 sessions with 295 trials, in the
  order of the trial table) and for every one of its 295 trials:
    slot         which distinct image of the session the trial shows, numbered 0, 1, 2, ... in the
                 order of the image numbers within that session (a repeat has the slot of its first
                 showing). continuous_rem.py uses this numbering (np.unique); it decides which
                 generated item lands on which trial.
    trial_type   1 = first showing of a target, 2 = repeat of a target, 3 = filler (first showing),
                 4 = vigilance repeat
  Nothing else: no responses, no image or concept numbers, no participant or session identifiers.
  continuous_rem.py reads this file when it is at data/trial_templates_400.npz.

INPUT   a trial table with the arrays sub, image_nr and ttype (one row per trial, sorted by session), as
        module A5's extract_trials.py writes it from the Kramer et al. trial file (EXTERNAL_INPUTS.md,
        M1). Default: external/kramer_trials/derived/trials.npz (where module A5's extract_trials.py
        writes it; never part of the release).
OUTPUT  the template file (default: data/trial_templates_400.npz of this module), arrays
        slot (int16, 400 x 295), trial_type (int8, 400 x 295) and a one-line description.
CHECKS  every session has 295 trials, the trial types are 1 to 4, every slot is shown once or twice,
        and the file gives back exactly the templates continuous_rem.load_templates() builds from the
        same trial table.
No random numbers. Runtime: a few seconds.

RUN (from the repository root)
  python A2c_continuous_recognition_figS4/code/make_trial_templates.py
  python A2c_continuous_recognition_figS4/code/make_trial_templates.py --trial-table <file> --output <file>
"""
import argparse
import os
import sys

import numpy as np

sys.dont_write_bytecode = True   # write no __pycache__ folder into shared/python or code/ (set before the imports below)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402,F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import continuous_rem  # noqa: E402

DESCRIPTION = ('Trial structures of the 400 Kramer et al. (2023) sessions the continuous-recognition model of '
               'Fig. S4 runs on: slot = distinct image of the session shown on each trial (numbered in the order '
               'of the image numbers within the session), trial_type = 1 target first showing, 2 target repeat, '
               '3 filler, 4 vigilance repeat. No responses, image numbers or identifiers.')


def main():
    parser = argparse.ArgumentParser(description='Write the structure-only trial templates of Fig. S4.')
    parser.add_argument('--trial-table', default=continuous_rem.A5_TRIAL_TABLE,
                        help='trial table with sub, image_nr, ttype (default: external/kramer_trials/derived/'
                             'trials.npz, written by module A5)')
    parser.add_argument('--output', default=continuous_rem.TEMPLATE_FILE,
                        help='template file to write (default: data/trial_templates_400.npz of this module)')
    arguments = parser.parse_args()

    if not os.path.exists(arguments.trial_table):
        sys.exit('make_trial_templates.py: the trial table %s does not exist. It is built from the Kramer et al. '
                 'trial file (EXTERNAL_INPUTS.md, M1) by python A5_things_context_effects_figS13_S14/code/'
                 'run_module.py.' % continuous_rem.relative_to_release(arguments.trial_table))

    # the templates exactly as continuous_rem.py builds them from the trial table
    templates = continuous_rem.load_templates(arguments.trial_table, continuous_rem.N_TEMPLATES)
    number_of_sessions = len(templates)
    if number_of_sessions != continuous_rem.N_TEMPLATES:
        sys.exit('make_trial_templates.py: only %d sessions with %d trials in the trial table; %d are needed'
                 % (number_of_sessions, continuous_rem.TEMPLATE_LENGTH, continuous_rem.N_TEMPLATES))

    slot_table = np.zeros((number_of_sessions, continuous_rem.TEMPLATE_LENGTH), dtype=np.int16)
    trial_type_table = np.zeros((number_of_sessions, continuous_rem.TEMPLATE_LENGTH), dtype=np.int8)
    for session in range(number_of_sessions):
        slot, trial_type = templates[session]
        if len(slot) != continuous_rem.TEMPLATE_LENGTH:
            sys.exit('make_trial_templates.py: session %d of the templates has %d trials, not %d'
                     % (session, len(slot), continuous_rem.TEMPLATE_LENGTH))
        slot_table[session] = slot
        trial_type_table[session] = trial_type

    # checks: trial types, and the repeat structure (every slot shown once or twice)
    known_types = set(np.unique(trial_type_table).tolist())
    if not known_types <= {1, 2, 3, 4}:
        sys.exit('make_trial_templates.py: unexpected trial types %s' % sorted(known_types))
    for session in range(number_of_sessions):
        showings_per_slot = np.bincount(slot_table[session])
        if showings_per_slot.min() < 1 or showings_per_slot.max() > 2:
            sys.exit('make_trial_templates.py: session %d shows a slot %d times'
                     % (session, showings_per_slot.max()))

    output_folder = os.path.dirname(os.path.abspath(arguments.output))
    os.makedirs(output_folder, exist_ok=True)
    np.savez_compressed(arguments.output, slot=slot_table, trial_type=trial_type_table,
                        description=np.array(DESCRIPTION))

    # read the file back the way continuous_rem.py does and compare with the templates above
    templates_from_file = continuous_rem.load_templates_from_template_file(arguments.output)
    for session in range(number_of_sessions):
        same_slots = np.array_equal(templates_from_file[session][0], templates[session][0])
        same_types = np.array_equal(templates_from_file[session][1], templates[session][1])
        same_dtypes = (templates_from_file[session][0].dtype == templates[session][0].dtype and
                       templates_from_file[session][1].dtype == templates[session][1].dtype)
        if not (same_slots and same_types and same_dtypes):
            sys.exit('make_trial_templates.py: session %d of the written file differs from the trial table' % session)

    distinct_images = [int(slot_table[session].max()) + 1 for session in range(number_of_sessions)]
    print('wrote %s (%d bytes)' % (continuous_rem.relative_to_release(arguments.output), os.path.getsize(arguments.output)))
    print('  %d sessions x %d trials; distinct images per session %d to %d'
          % (number_of_sessions, continuous_rem.TEMPLATE_LENGTH, min(distinct_images), max(distinct_images)))
    for trial_type in (1, 2, 3, 4):
        per_session = (trial_type_table == trial_type).sum(axis=1)
        print('  trial type %d: %d to %d trials per session' % (trial_type, per_session.min(), per_session.max()))
    print('  read back as continuous_rem.py reads it: identical to the templates built from the trial table')


if __name__ == '__main__':
    main()
