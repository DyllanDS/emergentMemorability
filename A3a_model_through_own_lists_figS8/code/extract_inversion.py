"""
extract_inversion.py: the memorability inversion experiment (Fig. 3), trial by trial, ready for the
model of Fig. S8a.

WHAT IT DOES
  The model of Fig. S8 is given the same study lists and test trials that each participant received.
  This script builds those lists for the participants the paper reports: the 54 sessions of
  03_inversion_experiment_fig3/data, of which the exclusion criteria of the Methods remove 4, leaving
  the 50 participants of Fig. 3e. It keeps the test trials the analysis of Fig. 3 keeps (200 to
  10,000 ms), so the model and the observed scores rest on the same trials.

  It also writes the observed table of the 40 critical items (hit rate, false alarm rate and corrected
  recognition, CR = hit rate minus false alarm rate, over the 50 participants, and the published
  memorability score of Kramer et al. 2023). The table comes from
  shared/python/inversion_loader.items_table(), the table of Fig. 3e.

  Responses: 1 (sure old) to 6 (sure new); 1 to 3 count as "old".

INPUTS (inside the repository)
  03_inversion_experiment_fig3/data/inversion_trials.csv and inversion_design.csv, read through
      shared/python/inversion_loader.py (which applies the exclusion criteria of the Methods)
  03_inversion_experiment_fig3/outputs/inversion_items_50.csv (module 03's item table, written by
      its MATLAB code): the two item tables must be equal. The published score of each item is the
      mean of that score over the item's test trials as module 03's MATLAB code computes it, taken
      from this file; where several items share a score, these full-precision values set their ranks,
      as in Fig. 3e (rho = -0.5004)

OUTPUTS (in outputs/model_inputs/)
  inversion_study.csv           one row per study trial of the 50 participants: sub, block, img
                                (image file name), concept, condition
  inversion_test.csv            one row per kept test trial: sub, block, img, concept, kind
                                (critical or maxdissimilar), response_1to6, is_old, said_old
  inversion_observed_items.csv  the 40 critical items: imageName, concept, nOld, nNew, hitRate,
                                faRate, cr, origMem (the published score)
  inversion_sample.csv          the sample: sessions, exclusions, kept participants, trial counts,
                                the observed hit and false alarm rates over all kept test trials
  The two trial tables are removed again at the end of run_module.py (they are a copy of the module 03
  data); run this script to make them again (a few seconds).

RUN (from the repository root; a few seconds):  python A3a_model_through_own_lists_figS8/code/extract_inversion.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (first: it stops Python from writing __pycache__ folders)
import inversion_loader  # noqa: E402

PATHS = release_paths.module_paths(__file__)
MODEL_INPUTS = os.path.join(PATHS['outputs'], 'model_inputs')
MODULE_03_ITEMS = os.path.join(release_paths.find_module('03'), 'outputs', 'inversion_items_50.csv')


def read_module_03_items():
    """Module 03's per-item table (MATLAB, full precision). Stops with a clear message if missing."""
    if not os.path.exists(MODULE_03_ITEMS):
        sys.exit('extract_inversion.py: %s is missing. Run module 03 first:\n'
                 '    matlab -batch "restoredefaultpath; cd(\'03_inversion_experiment_fig3/code\'); run_module"'
                 % os.path.relpath(MODULE_03_ITEMS, release_paths.release_root()))
    return pd.read_csv(MODULE_03_ITEMS, float_precision='round_trip')


def check_item_tables_are_equal(loader_items, module_03_items):
    """The shared loader's table and module 03's table must be the same table: every count and rate
    identical, the published score within 1e-12 (MATLAB's mean over the item's trials)."""
    if list(loader_items['image_name']) != list(module_03_items['image_name']):
        sys.exit('extract_inversion.py: the item order differs between inversion_loader.items_table() and '
                 'module 03 outputs/inversion_items_50.csv')
    for column in ['block', 'memorability_quintile', 'n_similar_items', 'n_old', 'hits', 'n_new',
                   'false_alarms', 'hit_rate', 'fa_rate', 'cr']:
        loader_values = loader_items[column].to_numpy(dtype=float)
        module_03_values = module_03_items[column].to_numpy(dtype=float)
        if not np.array_equal(loader_values, module_03_values):
            sys.exit('extract_inversion.py: column %s differs between the shared loader and module 03' % column)
    largest_score_difference = float(np.max(np.abs(loader_items['kramer_cr'] - module_03_items['kramer_cr'])))
    if largest_score_difference > 1e-12:
        sys.exit('extract_inversion.py: the published scores differ between the shared loader and module 03')
    print('item table: shared/python/inversion_loader.items_table() equals module 03 '
          'outputs/inversion_items_50.csv on every column (published score within %.1e)' % largest_score_difference)


def main():
    os.makedirs(MODEL_INPUTS, exist_ok=True)

    # ---- 1. the participants: 54 sessions, the exclusion criteria of the Methods, 50 kept
    all_rows = inversion_loader.load_trials('all')
    test_trials = inversion_loader.load_trials('test')
    kept_test_trials, flags = inversion_loader.apply_exclusions(test_trials)
    kept_participants = sorted(kept_test_trials['participant'].unique())
    excluded_participants = list(flags.loc[flags['excluded'], 'participant'])
    print('%d sessions; %d excluded (%s); %d participants kept'
          % (len(flags), len(excluded_participants), ', '.join(excluded_participants), len(kept_participants)))
    all_test_trials_of_kept = test_trials[test_trials['participant'].isin(kept_participants)]
    print('test trials of the kept participants: %d kept of %d (%d outside 200 to 10,000 ms)'
          % (len(kept_test_trials), len(all_test_trials_of_kept),
             len(all_test_trials_of_kept) - len(kept_test_trials)))

    # ---- 2. the study lists of the kept participants, in the order they were shown
    study_rows = all_rows[(all_rows['phase'] == 'study') & (all_rows['participant'].isin(kept_participants))]
    study = pd.DataFrame({'sub': study_rows['participant'].to_numpy(),
                          'block': study_rows['block'].astype(int).to_numpy(),
                          'img': study_rows['image_name'].to_numpy(),
                          'concept': study_rows['concept'].to_numpy(),
                          'condition': study_rows['condition'].to_numpy()})

    # ---- 3. the kept test trials
    kind = []
    for condition in kept_test_trials['condition']:
        if condition == 'critical':
            kind.append('critical')
        elif condition == 'maxDissimilar':
            kind.append('maxdissimilar')
        else:
            kind.append('other')
    test = pd.DataFrame({'sub': kept_test_trials['participant'].to_numpy(),
                         'block': kept_test_trials['block'].astype(int).to_numpy(),
                         'img': kept_test_trials['image_name'].to_numpy(),
                         'concept': kept_test_trials['concept'].to_numpy(),
                         'kind': kind,
                         'response_1to6': kept_test_trials['response_1to6'].astype(int).to_numpy(),
                         'is_old': (1 - kept_test_trials['is_new'].astype(int)).to_numpy(),
                         'said_old': (kept_test_trials['response_1to6'] <= 3).astype(int).to_numpy()})

    # Check: an item is old exactly when it was on that block's study list.
    studied_in_block = {}
    for (participant, block), block_rows in study.groupby(['sub', 'block']):
        studied_in_block[(participant, block)] = set(block_rows['img'])
    for participant, block, image, is_old in zip(test['sub'], test['block'], test['img'], test['is_old']):
        on_study_list = image in studied_in_block[(participant, block)]
        if int(on_study_list) != int(is_old):
            sys.exit('extract_inversion.py: is_new of %s %s block %d disagrees with the study list'
                     % (participant, image, block))
    print('old/new status of every test trial agrees with the block\'s own study list')

    # ---- 4. descriptive checks
    print('%d participants; %d study trials, %d test trials' % (study['sub'].nunique(), len(study), len(test)))
    list_lengths = study.groupby(['sub', 'block']).size()
    print('  study list length: median %d (min %d, max %d); %d subject-blocks'
          % (list_lengths.median(), list_lengths.min(), list_lengths.max(), len(list_lengths)))
    print('  test items by role: %s' % dict(test['kind'].value_counts()))
    print('  old / new test trials: %d / %d' % (int(test['is_old'].sum()), int((1 - test['is_old']).sum())))
    for role, role_trials in test.groupby('kind'):
        hit_rate = role_trials[role_trials['is_old'] == 1]['said_old'].mean()
        false_alarm_rate = role_trials[role_trials['is_old'] == 0]['said_old'].mean()
        print('    %-14s hit %.3f  FA %.3f  n=%d' % (role, hit_rate, false_alarm_rate, len(role_trials)))
    print('  all test trials: hit %.3f  FA %.3f'
          % (test[test['is_old'] == 1]['said_old'].mean(), test[test['is_old'] == 0]['said_old'].mean()))

    # ---- 5. the observed table of the 40 critical items (Fig. 3e's table)
    loader_items = inversion_loader.items_table()
    module_03_items = read_module_03_items()
    check_item_tables_are_equal(loader_items, module_03_items)
    design = inversion_loader.load_design().set_index('image_name')
    concept_of_item = []
    for image_name in loader_items['image_name']:
        concept_of_item.append(design.loc[image_name, 'concept'])
    observed_items = pd.DataFrame({'imageName': loader_items['image_name'],
                                   'concept': concept_of_item,
                                   'nOld': loader_items['n_old'],
                                   'nNew': loader_items['n_new'],
                                   'hitRate': loader_items['hit_rate'],
                                   'faRate': loader_items['fa_rate'],
                                   'cr': loader_items['cr'],
                                   'origMem': module_03_items['kramer_cr']})
    tests_per_item = observed_items['nOld'] + observed_items['nNew']
    print('observed critical items: %d; tests per item %d to %d (one per kept participant, less the trials '
          'outside 200 to 10,000 ms)' % (len(observed_items), tests_per_item.min(), tests_per_item.max()))

    # ---- 6. the sample, in one small table (kept when the trial tables are removed)
    sample = pd.DataFrame([{'experiment': 'memorability inversion experiment',
                            'sessions': len(flags),
                            'excluded': len(excluded_participants),
                            'excluded_participants': ' '.join(excluded_participants),
                            'kept_participants': len(kept_participants),
                            'subject_blocks': len(list_lengths),
                            'test_trials_of_kept_participants': len(all_test_trials_of_kept),
                            'test_trials_kept': len(test),
                            'critical_test_trials_kept': int(np.sum(test['kind'] == 'critical')),
                            'observed_hit_rate_all_test_trials': test[test['is_old'] == 1]['said_old'].mean(),
                            'observed_fa_rate_all_test_trials': test[test['is_old'] == 0]['said_old'].mean()}])

    # ---- 7. write
    study.to_csv(os.path.join(MODEL_INPUTS, 'inversion_study.csv'), index=False)
    test.to_csv(os.path.join(MODEL_INPUTS, 'inversion_test.csv'), index=False)
    observed_items.to_csv(os.path.join(MODEL_INPUTS, 'inversion_observed_items.csv'), index=False)
    sample.to_csv(os.path.join(MODEL_INPUTS, 'inversion_sample.csv'), index=False)
    print('wrote outputs/model_inputs/inversion_study.csv, inversion_test.csv, inversion_observed_items.csv, '
          'inversion_sample.csv')


if __name__ == '__main__':
    main()
