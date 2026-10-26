"""
extract_behavior.py: step 1 of module A6b. The memorability of every THINGS concept in the maintained
and changed image sets of the image-set experiments (paper Fig. 6), from the participants'
recognition responses. Nothing here depends on the embedding, so it is computed once and reused for
the published embedding and every refit.

WHAT IT DOES
  1. Loads the test trials of the maintained (205 sessions) and changed (195 sessions) image-set
     experiments with the release's shared loader (shared/python/imageset_loader.py).
  2. Applies the paper's exclusions (Methods), as implemented once in that loader: a participant is
     excluded when one response key was used on more than 80 percent of the test trials, when more
     than 20 percent of the responses were faster than 200 ms, when more than 20 percent were
     slower than 7,500 ms, or when the hit rate minus the false alarm rate was below 0.05; then
     trials faster than 200 ms or slower than 10,000 ms are dropped; then the first 150 remaining
     participants of each set (in participant-id order) are kept. The maintained set loses 55
     participants and the changed set 45, leaving 150 each.
  3. For every concept of each set it pools the kept participants' test trials: the hit rate is
     the share of the concept's studied (old) test images called "old" (a confidence rating of 1 to
     3 on the 6-point scale), the false alarm rate the same share for its new test images, and the
     concept's memorability (corrected recognition) is the hit rate minus the false alarm rate. A
     concept without an old or without a new test trial in a set gets no memorability there.

INPUTS
  06_image_set_experiments_fig6_fig7/data/maintained_trials.csv and changed_trials.csv
  (read through shared/python/imageset_loader.py), shared/things/concept_table.csv

OUTPUT
  outputs/image_set_concept_memorability.csv: one row per THINGS concept (1,854, THINGS order):
  concept_number, concept, and for each set (maintained_, changed_): old_trials, hits, new_trials,
  false_alarms, hit_rate, false_alarm_rate, memorability (empty cells written as NaN). Numbers are
  written as the shortest decimal that reads back as the same double.

RUN (from the repository root; a few seconds)
  python A6b_embedding_size_figS16/code/extract_behavior.py
"""
import csv
import os
import sys

import numpy as np

sys.dont_write_bytecode = True     # no __pycache__ folders inside the release
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths        # noqa: E402  (shared helper: release folders)
import concept_tables     # noqa: E402  (shared helper: the THINGS concept tables)
import imageset_loader    # noqa: E402  (shared helper: the image-set trials and the paper's exclusions)
import analysis_core      # noqa: E402  (this module: the set names and the output file name)

IMAGE_SETS = analysis_core.IMAGE_SETS                              # ('maintained', 'changed')
OUTPUT_FILE_NAME = analysis_core.IMAGE_SET_MEMORABILITY_FILE       # image_set_concept_memorability.csv


def exact_text(value):
    """The shortest decimal text that reads back as exactly the same number ('NaN' when missing)."""
    value = float(value)
    if np.isnan(value):
        return 'NaN'
    return repr(value)


def concept_counts_for_set(kept_trials, set_name, number_of_concepts):
    """Pooled over the kept participants of one set: for every concept (THINGS order) the number of
    old test trials, hits, new test trials and false alarms."""
    old_trials = np.zeros(number_of_concepts)
    hits = np.zeros(number_of_concepts)
    new_trials = np.zeros(number_of_concepts)
    false_alarms = np.zeros(number_of_concepts)
    set_trials = kept_trials[kept_trials['image_set'] == set_name]
    for concept_number, is_new, said_old in zip(set_trials['concept_number'], set_trials['is_new'],
                                                set_trials['said_old']):
        row = int(concept_number) - 1                  # concept numbers count from 1 in THINGS order
        if is_new == 0:
            old_trials[row] += 1
            hits[row] += bool(said_old)
        else:
            new_trials[row] += 1
            false_alarms[row] += bool(said_old)
    return old_trials, hits, new_trials, false_alarms


def concept_rates(old_trials, hits, new_trials, false_alarms):
    """Hit rate, false alarm rate and memorability (hit rate minus false alarm rate) per concept;
    missing (NaN) where the concept has no old or no new test trial."""
    with np.errstate(invalid='ignore', divide='ignore'):
        hit_rate = np.where(old_trials > 0, hits / np.maximum(old_trials, 1), np.nan)
        false_alarm_rate = np.where(new_trials > 0, false_alarms / np.maximum(new_trials, 1), np.nan)
    memorability = hit_rate - false_alarm_rate
    memorability[(old_trials == 0) | (new_trials == 0)] = np.nan
    return hit_rate, false_alarm_rate, memorability


def main():
    paths = release_paths.module_paths(__file__)
    concept_table = concept_tables.load_concept_table()
    number_of_concepts = len(concept_table)

    print('step 1: memorability of each concept in the maintained and changed image sets')
    trials = imageset_loader.load_trials(IMAGE_SETS, with_summed_similarity=False)
    kept_trials, participant_flags = imageset_loader.apply_exclusions(trials)

    columns = {}
    for set_name in IMAGE_SETS:
        flags = participant_flags[participant_flags['image_set'] == set_name]
        number_run = len(flags)
        number_excluded = int(flags['excluded'].sum())
        number_kept = kept_trials.loc[kept_trials['image_set'] == set_name, 'participant'].nunique()
        old_trials, hits, new_trials, false_alarms = concept_counts_for_set(kept_trials, set_name,
                                                                            number_of_concepts)
        hit_rate, false_alarm_rate, memorability = concept_rates(old_trials, hits, new_trials, false_alarms)
        print('  %-10s %d participants run, %d excluded, %d kept; %d concepts with a memorability score'
              % (set_name, number_run, number_excluded, number_kept, int(np.isfinite(memorability).sum())))
        columns[set_name] = dict(old_trials=old_trials, hits=hits, new_trials=new_trials,
                                 false_alarms=false_alarms, hit_rate=hit_rate,
                                 false_alarm_rate=false_alarm_rate, memorability=memorability)

    header = ['concept_number', 'concept']
    for set_name in IMAGE_SETS:
        for name in ('old_trials', 'hits', 'new_trials', 'false_alarms', 'hit_rate', 'false_alarm_rate',
                     'memorability'):
            header.append(set_name + '_' + name)
    output_path = os.path.join(paths['outputs'], OUTPUT_FILE_NAME)
    with open(output_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in range(number_of_concepts):
            values = [int(concept_table['concept_number'].iloc[row]), concept_table['concept'].iloc[row]]
            for set_name in IMAGE_SETS:
                values.append(int(columns[set_name]['old_trials'][row]))
                values.append(int(columns[set_name]['hits'][row]))
                values.append(int(columns[set_name]['new_trials'][row]))
                values.append(int(columns[set_name]['false_alarms'][row]))
                values.append(exact_text(columns[set_name]['hit_rate'][row]))
                values.append(exact_text(columns[set_name]['false_alarm_rate'][row]))
                values.append(exact_text(columns[set_name]['memorability'][row]))
            writer.writerow(values)
    print('  wrote', os.path.relpath(output_path, release_paths.release_root()))


if __name__ == '__main__':
    main()
