"""
imageset_loader.py: the three image-set experiments (Figs. 6 and 7) from their data files.

Python twin of shared/matlab/load_image_set_trials.m and apply_participant_exclusions.m. It builds
the same test-trial table, including the summed similarity of each tested concept to the 40 study
items of the same block and of the block before.

Reads 06_image_set_experiments_fig6_fig7/data/<set>_trials.csv (de-identified) and
shared/things/concept_table.csv. Participants are M001.. (maintained, 205), C001.. (changed, 195)
and R001.. (reduced, 202): numbered within each set in the alphabetical order of the session ids
(not included), the order in which the analyses load, cap and resample them.

    load_trials(sets)            test trials of the listed sets, in file order, columns:
        participant, image_set, block, trial_in_block, image_filename, concept, concept_number,
        things_image_index, image_number, is_new, response_1to6, rt_ms, said_old,
        summed_similarity_same_block, summed_similarity_previous_block,
        dataset_index, dataset_name   (dataset_index / dataset_name as load_image_set_trials.m sets them:
                                       maintained 1 'representativeImages', changed 0
                                       'flipMemorabilityImages', reduced 2 'equatedImages')
    load_study_lists(set_name)   {participant: {block: [concept numbers of the 40 study items]}}
    participant_flags(trials)    one row per participant: the four exclusion rules (Methods,
                                 Experimental Procedures and Analysis)
    apply_exclusions(trials)     (kept trials, flags): rules -> trial filter 200 to 10,000 ms ->
                                 the first 150 participants per set in id order (removes nobody)

Summed similarity: for a test trial of concept c in block b, take the concept numbers of the 40 study
items of block b, remove ONE occurrence of c if present, and average normpdf(similarity(c, j), 1, 2)
over the rest (similarity = correlation of the SPoSE feature rows); the same for block b - 1 (NaN in
block 1).
"""
import os

import numpy as np
import pandas as pd

import concept_tables

_RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FOLDER = os.path.join(_RELEASE_ROOT, '06_image_set_experiments_fig6_fig7', 'data')

SET_NAMES = ('maintained', 'changed', 'reduced')
DATASET_INDEX = {'maintained': 1, 'changed': 0, 'reduced': 2}
DATASET_NAME = {'maintained': 'representativeImages', 'changed': 'flipMemorabilityImages',
                'reduced': 'equatedImages'}
KERNEL_MEAN = 1.0
KERNEL_SD = 2.0
NUMBER_OF_BLOCKS = 16

FAST_RT_MS = 200
SLOW_RT_MS = 7500
TRIAL_FILTER_MIN_MS = 200
TRIAL_FILTER_MAX_MS = 10000
ONE_KEY_LIMIT = 0.8
RULE_SHARE_LIMIT = 0.2
MIN_HIT_MINUS_FA = 0.05
PARTICIPANTS_KEPT_PER_SET = 150


def normal_density(x, mean, sd):
    """The same as MATLAB normpdf(x, mean, sd)."""
    return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def read_set_file(set_name, data_folder=DATA_FOLDER):
    """All rows (study and test) of one set, in file order."""
    return pd.read_csv(os.path.join(data_folder, '%s_trials.csv' % set_name), float_precision='round_trip')


def concept_number_lookup():
    """{concept name: concept number in THINGS order, counted from 1} (first match, as MATLAB)."""
    table = concept_tables.load_concept_table()
    lookup = {}
    for number, name in zip(table['concept_number'], table['concept']):
        if name not in lookup:
            lookup[name] = int(number)
    return lookup


def load_study_lists(set_name, data_folder=DATA_FOLDER, rows=None, lookup=None):
    """{participant: {block: [concept numbers of the study items, in study order]}}."""
    if rows is None:
        rows = read_set_file(set_name, data_folder)
    if lookup is None:
        lookup = concept_number_lookup()
    study_rows = rows[rows['phase'] == 'study']
    lists = {}
    for participant, block, concept in zip(study_rows['participant'], study_rows['block'], study_rows['concept']):
        lists.setdefault(participant, {}).setdefault(int(block), []).append(lookup.get(concept, 0))
    return lists


def load_trials(sets=SET_NAMES, data_folder=DATA_FOLDER, with_summed_similarity=True):
    """Test trials of the listed sets (in the order given), with the MATLAB loader's columns."""
    if isinstance(sets, str):
        sets = (sets,)
    for set_name in sets:
        if set_name not in SET_NAMES:
            raise ValueError('unknown image set %r; the sets are %s' % (set_name, ', '.join(SET_NAMES)))
    lookup = concept_number_lookup()
    kernel = None
    if with_summed_similarity:
        features = concept_tables.spose_features()
        kernel = normal_density(concept_tables.concept_similarity(features), KERNEL_MEAN, KERNEL_SD)
    tables = []
    for set_name in sets:
        rows = read_set_file(set_name, data_folder)
        test = rows[rows['phase'] == 'test'].copy()
        for column in ('is_new', 'response_1to6'):             # whole numbers on test rows (study rows have blanks)
            test[column] = test[column].astype(int)
        test['image_set'] = set_name
        test['concept_number'] = [lookup[concept] for concept in test['concept']]
        test['said_old'] = test['response_1to6'] <= 3
        test['dataset_index'] = DATASET_INDEX[set_name]
        test['dataset_name'] = DATASET_NAME[set_name]
        if with_summed_similarity:
            study_lists = load_study_lists(set_name, data_folder, rows=rows, lookup=lookup)
            same_block = []
            previous_block = []
            for participant, block, concept_number in zip(test['participant'], test['block'],
                                                          test['concept_number']):
                values = []
                for blocks_back in (0, 1):
                    study_block = int(block) - blocks_back
                    if study_block < 1:
                        values.append(np.nan)
                        continue
                    study_list = list(study_lists[participant][study_block])
                    if concept_number in study_list:
                        study_list.remove(concept_number)            # the first occurrence only
                    if 0 in study_list:
                        raise ValueError('study item with a concept not in the concept table')
                    columns = np.array(study_list) - 1
                    values.append(float(np.mean(kernel[concept_number - 1, columns])))
                same_block.append(values[0])
                previous_block.append(values[1])
            test['summed_similarity_same_block'] = same_block
            test['summed_similarity_previous_block'] = previous_block
        tables.append(test)
    trials = pd.concat(tables, ignore_index=True)
    columns = ['participant', 'image_set', 'block', 'trial_in_block', 'image_filename', 'concept',
               'concept_number', 'things_image_index', 'image_number', 'is_new', 'response_1to6', 'rt_ms',
               'said_old']
    if with_summed_similarity:
        columns += ['summed_similarity_same_block', 'summed_similarity_previous_block']
    columns += ['dataset_index', 'dataset_name']
    return trials[columns]


def participant_flags(trials, slow_rt_ms=SLOW_RT_MS):
    """One row per participant (sets in the order they appear, participants in id order):
    the shares behind each rule, one column per rule and 'excluded'."""
    rows = []
    for set_name in trials['image_set'].unique():
        set_trials = trials[trials['image_set'] == set_name]
        for participant in sorted(set_trials['participant'].unique()):
            own = set_trials[set_trials['participant'] == participant]
            number_of_trials = len(own)
            key_counts = [np.sum(own['response_1to6'] == key) for key in range(1, 7)]
            one_key_share = max(key_counts) / number_of_trials
            fast_share = np.sum(own['rt_ms'] < FAST_RT_MS) / number_of_trials
            slow_share = np.sum(own['rt_ms'] > slow_rt_ms) / number_of_trials
            hit = np.mean(own.loc[own['is_new'] == 0, 'response_1to6'] <= 3)
            false_alarm = np.mean(own.loc[own['is_new'] == 1, 'response_1to6'] <= 3)
            row = dict(participant=participant, image_set=set_name, test_trials=number_of_trials,
                       one_key_share=one_key_share, fast_share=fast_share, slow_share=slow_share,
                       hit_minus_fa=hit - false_alarm, same_key=one_key_share > ONE_KEY_LIMIT,
                       too_fast=fast_share > RULE_SHARE_LIMIT, too_slow=slow_share > RULE_SHARE_LIMIT,
                       low_cr=(hit - false_alarm) < MIN_HIT_MINUS_FA)
            row['excluded'] = row['same_key'] or row['too_fast'] or row['too_slow'] or row['low_cr']
            rows.append(row)
    return pd.DataFrame(rows)


def apply_exclusions(trials, slow_rt_ms=SLOW_RT_MS, cap_per_set=PARTICIPANTS_KEPT_PER_SET):
    """Exclude participants by the rules, then drop trials outside 200 to 10,000 ms, then keep the
    first cap_per_set participants of each set in id order. Returns (kept trials, flags)."""
    flags = participant_flags(trials, slow_rt_ms=slow_rt_ms)
    excluded = set(flags.loc[flags['excluded'], 'participant'])
    kept = trials[~trials['participant'].isin(excluded)]
    kept = kept[(kept['rt_ms'] >= TRIAL_FILTER_MIN_MS) & (kept['rt_ms'] <= TRIAL_FILTER_MAX_MS)]
    keep_ids = []
    for set_name in kept['image_set'].unique():
        remaining = sorted(kept.loc[kept['image_set'] == set_name, 'participant'].unique())
        keep_ids.extend(remaining[:cap_per_set])
    kept = kept[kept['participant'].isin(set(keep_ids))]
    return kept.reset_index(drop=True), flags
