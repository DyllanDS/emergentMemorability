"""
extract_datasets.py: the three image set experiments (Figs. 6 and 7), trial by trial, ready for the
model of Fig. S8b and c.

WHAT IT DOES
  The model of Fig. S8 is given the study lists and test trials each participant received. This
  script builds them for the participants the paper reports: the 602 sessions of
  06_image_set_experiments_fig6_fig7/data (205 maintained, 195 changed, 202 reduced), of which the
  exclusion criteria of the Methods remove 152, leaving 150 per set (450). It keeps the test trials
  the paper's analyses keep (200 to 10,000 ms).

  It then computes the observed memorability of every image in every set it was tested in (an
  image-by-set "cell"): the numbers of old and new test trials, the hit rate, the false alarm rate
  and corrected recognition (hit rate minus false alarm rate). Only cells with at least 5 old and
  at least 5 new trials are kept. Images tested in two or three sets are the ones the same-image
  comparison of Fig. S8b and c uses.

  Each participant studied 16 blocks of 40 images and after each block judged 40 test images (half
  studied, half new) on a 6-point scale from 1 (sure old) to 6 (sure new); 1 to 3 count as "old".
  A test image counts as old when it was on that block's own study list. Images are identified by
  their THINGS image number (things_image_index in the data files).

INPUTS (inside the repository)
  06_image_set_experiments_fig6_fig7/data/maintained_trials.csv, changed_trials.csv,
      reduced_trials.csv, read through shared/python/imageset_loader.py (which applies the
      exclusion criteria of the Methods)

OUTPUTS (in outputs/model_inputs/)
  datasets_study.csv     one row per study trial of the 450 participants: sub, cond, block, image_nr
  datasets_test.csv      one row per kept test trial: sub, cond, block, image_nr, response_1to6,
                         is_old, said_old
  datasets_observed.csv  one row per image-by-set cell with at least 5 old and 5 new trials:
                         cond, image_nr, n_old, n_new, hit, fa, cr
  datasets_sample.csv    the sample per set and in all: sessions, exclusions, kept participants,
                         trial counts, observed hit and false alarm rates, cells, shared images
  The two trial tables are removed again at the end of run_module.py (they are a copy of the module 06
  data); run this script to make them again (about 10 seconds).

RUN (from the repository root; about 10 seconds):  python A3a_model_through_own_lists_figS8/code/extract_datasets.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (first: it stops Python from writing __pycache__ folders)
import imageset_loader  # noqa: E402

PATHS = release_paths.module_paths(__file__)
MODEL_INPUTS = os.path.join(PATHS['outputs'], 'model_inputs')
MINIMUM_TRIALS_PER_CELL = 5        # old trials and new trials, each


def main():
    os.makedirs(MODEL_INPUTS, exist_ok=True)

    # ---- 1. the participants: 602 sessions, the exclusion criteria of the Methods, 150 kept per set
    test_trials = imageset_loader.load_trials(imageset_loader.SET_NAMES, with_summed_similarity=False)
    kept_test_trials, flags = imageset_loader.apply_exclusions(test_trials)
    kept_participants = set(kept_test_trials['participant'])
    for set_name in imageset_loader.SET_NAMES:
        set_flags = flags[flags['image_set'] == set_name]
        kept_in_set = kept_test_trials[kept_test_trials['image_set'] == set_name]['participant'].nunique()
        print('%-10s %d sessions, %d excluded, %d kept' % (set_name, len(set_flags), int(set_flags['excluded'].sum()),
                                                           kept_in_set))
    all_test_trials_of_kept = test_trials[test_trials['participant'].isin(kept_participants)]
    print('test trials of the kept participants: %d kept of %d (%d outside 200 to 10,000 ms)'
          % (len(kept_test_trials), len(all_test_trials_of_kept),
             len(all_test_trials_of_kept) - len(kept_test_trials)))

    # ---- 2. the study lists of the kept participants, in the order they were shown
    study_tables = []
    for set_name in imageset_loader.SET_NAMES:
        rows = imageset_loader.read_set_file(set_name)
        study_rows = rows[(rows['phase'] == 'study') & (rows['participant'].isin(kept_participants))]
        study_tables.append(pd.DataFrame({'sub': study_rows['participant'].to_numpy(),
                                          'cond': set_name,
                                          'block': study_rows['block'].astype(int).to_numpy(),
                                          'image_nr': study_rows['things_image_index'].astype(int).to_numpy()}))
    study = pd.concat(study_tables, ignore_index=True)

    # ---- 3. the kept test trials
    test = pd.DataFrame({'sub': kept_test_trials['participant'].to_numpy(),
                         'cond': kept_test_trials['image_set'].to_numpy(),
                         'block': kept_test_trials['block'].astype(int).to_numpy(),
                         'image_nr': kept_test_trials['things_image_index'].astype(int).to_numpy(),
                         'response_1to6': kept_test_trials['response_1to6'].astype(int).to_numpy(),
                         'said_old': (kept_test_trials['response_1to6'] <= 3).astype(int).to_numpy()})

    # Old or new: a test image is old when it was on that block's own study list.
    studied_in_block = {}
    for (participant, block), block_rows in study.groupby(['sub', 'block']):
        studied_in_block[(participant, block)] = set(block_rows['image_nr'])
    is_old = []
    for participant, block, image_number in zip(test['sub'], test['block'], test['image_nr']):
        is_old.append(int(image_number in studied_in_block[(participant, block)]))
    test.insert(5, 'is_old', is_old)
    print('old/new status from the study lists: %d old and %d new test trials'
          % (int(np.sum(test['is_old'])), int(np.sum(1 - test['is_old']))))

    # ---- 4. descriptive checks
    print('%d participants, %d study and %d test trials' % (study['sub'].nunique(), len(study), len(test)))
    for set_name, set_trials in test.groupby('cond'):
        hit_rate = set_trials[set_trials['is_old'] == 1]['said_old'].mean()
        false_alarm_rate = set_trials[set_trials['is_old'] == 0]['said_old'].mean()
        print('  %-10s n=%4d  hit %.3f  FA %.3f  CR %.3f' % (set_name, set_trials['sub'].nunique(), hit_rate,
                                                            false_alarm_rate, hit_rate - false_alarm_rate))
    list_lengths = study.groupby(['sub', 'block']).size()
    print('  study list length: median %d (%d to %d); %d subject-blocks'
          % (list_lengths.median(), list_lengths.min(), list_lengths.max(), len(list_lengths)))

    # ---- 5. observed memorability per image within each set, with trial counts
    rows = []
    for (set_name, image_number), cell_trials in test.groupby(['cond', 'image_nr']):
        old_trials = cell_trials[cell_trials['is_old'] == 1]
        new_trials = cell_trials[cell_trials['is_old'] == 0]
        if len(old_trials) < MINIMUM_TRIALS_PER_CELL or len(new_trials) < MINIMUM_TRIALS_PER_CELL:
            continue
        hit_rate = old_trials['said_old'].mean()
        false_alarm_rate = new_trials['said_old'].mean()
        rows.append((set_name, image_number, len(old_trials), len(new_trials), hit_rate, false_alarm_rate,
                     hit_rate - false_alarm_rate))
    observed = pd.DataFrame(rows, columns=['cond', 'image_nr', 'n_old', 'n_new', 'hit', 'fa', 'cr'])
    all_cells = test.groupby(['cond', 'image_nr']).ngroups
    print('\n  %d image-by-set cells with at least %d old and %d new trials (of %d cells tested)'
          % (len(observed), MINIMUM_TRIALS_PER_CELL, MINIMUM_TRIALS_PER_CELL, all_cells))
    for set_name, set_cells in observed.groupby('cond'):
        print('    %-10s %d cells' % (set_name, len(set_cells)))
    print('  median trials per cell: %d old, %d new' % (observed['n_old'].median(), observed['n_new'].median()))
    sets_per_image = observed.groupby('image_nr')['cond'].nunique()
    print('  images measured in 2+ sets: %d; in all three: %d'
          % (int(np.sum(sets_per_image >= 2)), int(np.sum(sets_per_image == 3))))

    # ---- 6. the sample, in one small table (kept when the trial tables are removed)
    sample_rows = []
    for set_name in list(imageset_loader.SET_NAMES) + ['all three sets']:
        if set_name == 'all three sets':
            set_flags = flags
            set_test = test
            set_cells = observed
            set_all_test = all_test_trials_of_kept
        else:
            set_flags = flags[flags['image_set'] == set_name]
            set_test = test[test['cond'] == set_name]
            set_cells = observed[observed['cond'] == set_name]
            set_all_test = all_test_trials_of_kept[all_test_trials_of_kept['image_set'] == set_name]
        sample_rows.append({'image_set': set_name,
                            'sessions': len(set_flags),
                            'excluded': int(set_flags['excluded'].sum()),
                            'kept_participants': set_test['sub'].nunique(),
                            'subject_blocks': set_test.groupby(['sub', 'block']).ngroups,
                            'test_trials_of_kept_participants': len(set_all_test),
                            'test_trials_kept': len(set_test),
                            'observed_hit_rate': set_test[set_test['is_old'] == 1]['said_old'].mean(),
                            'observed_fa_rate': set_test[set_test['is_old'] == 0]['said_old'].mean(),
                            'cells_with_5_old_and_5_new': len(set_cells),
                            'images_in_two_or_more_sets': (int(np.sum(sets_per_image >= 2))
                                                           if set_name == 'all three sets' else ''),
                            'images_in_three_sets': (int(np.sum(sets_per_image == 3))
                                                     if set_name == 'all three sets' else '')})
    sample = pd.DataFrame(sample_rows)

    # ---- 7. write
    study.to_csv(os.path.join(MODEL_INPUTS, 'datasets_study.csv'), index=False)
    test.to_csv(os.path.join(MODEL_INPUTS, 'datasets_test.csv'), index=False)
    observed.to_csv(os.path.join(MODEL_INPUTS, 'datasets_observed.csv'), index=False)
    sample.to_csv(os.path.join(MODEL_INPUTS, 'datasets_sample.csv'), index=False)
    print('\nwrote outputs/model_inputs/datasets_study.csv, datasets_test.csv, datasets_observed.csv, '
          'datasets_sample.csv')


if __name__ == '__main__':
    main()
