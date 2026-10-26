"""
inversion_loader.py: the memorability-inversion experiment (Fig. 3) from its data files.

Reads 03_inversion_experiment_fig3/data/ (de-identified):
    inversion_trials.csv   54 sessions (S001..S054, the order in which the analysis loads them),
                           320 study and 80 test rows each
    inversion_design.csv   the 40 critical items: block, quintile, number of similar items, Kramer score

    load_trials(phase='test')          the trial rows ('test', 'study' or 'all'), with is_critical
                                       and said_old (response 1 to 3 = old) added to test rows
    participant_flags(test_trials)     one row per participant: the four exclusion rules of the
                                       Methods (Experimental Procedures and Analysis) and whether
                                       the participant is excluded
    apply_exclusions(test_trials)      (kept test trials, flags): rules -> trial filter 200 to
                                       10,000 ms -> first 50 participants kept (no one is removed by
                                       this cap: 54 - 4 = 50)
    items_table()                      the 40 critical items over the 50 kept participants:
                                       image_name, block, memorability_quintile, n_similar_items,
                                       kramer_cr, n_old, hits, n_new, false_alarms, hit_rate, fa_rate, cr
                                       (rows sorted by image name, the order MATLAB's unique() gives)

Fig. 3e: Spearman's rho is -0.5003754106 (p = 0.0010088) with the per-item table that module 03
writes (03_inversion_experiment_fig3/outputs/inversion_items_50.csv, columns kramer_cr and cr). There
kramer_cr is the mean of the published score over the item's test trials as the saved run computed it
(module 03, data/paper_run/published_scores.csv). Module 03 ranks the items by these full-precision
values, which fix the order of items that share a published score. items_table() returns the published
score itself (inversion_design.csv) as kramer_cr, within 1e-12 of these means.

A participant is excluded if one response key takes more than 80 percent of the test trials, more
than 20 percent of the test trials are faster than 200 ms, more than 20 percent are slower than
7,500 ms, or hit rate minus false alarm rate (old = response 1 to 3) is below 0.05. The trial filter
that follows keeps trials with 200 <= rt <= 10,000 ms.
"""
import os

import numpy as np
import pandas as pd

_RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FOLDER = os.path.join(_RELEASE_ROOT, '03_inversion_experiment_fig3', 'data')

FAST_RT_MS = 200
SLOW_RT_MS = 7500                 # participant rule: share of trials slower than this
TRIAL_FILTER_MIN_MS = 200
TRIAL_FILTER_MAX_MS = 10000
ONE_KEY_LIMIT = 0.8
RULE_SHARE_LIMIT = 0.2
MIN_HIT_MINUS_FA = 0.05
PARTICIPANTS_KEPT = 50            # participants kept after the exclusions (the first 50 in S-id order)


def load_trials(phase='test', data_folder=DATA_FOLDER):
    """Trial rows in file order. phase: 'test' (80 per session), 'study' (320) or 'all'."""
    trials = pd.read_csv(os.path.join(data_folder, 'inversion_trials.csv'), float_precision='round_trip')
    if phase in ('test', 'study'):
        trials = trials[trials['phase'] == phase].reset_index(drop=True)
    elif phase != 'all':
        raise ValueError("phase must be 'test', 'study' or 'all'")
    if phase == 'test':
        for column in ('is_new', 'response_1to6'):             # whole numbers on test rows (study rows have blanks)
            trials[column] = trials[column].astype(int)
    test_rows = trials['phase'] == 'test'
    trials['is_critical'] = (trials['condition'] == 'critical') & test_rows
    trials['said_old'] = np.where(test_rows, trials['response_1to6'] <= 3, np.nan)
    return trials


def load_design(data_folder=DATA_FOLDER):
    """The 40 critical items of the deployed design."""
    return pd.read_csv(os.path.join(data_folder, 'inversion_design.csv'), float_precision='round_trip')


def participant_flags(test_trials, slow_rt_ms=SLOW_RT_MS):
    """One row per participant (in S001.. order) with the share of the most used key, the shares of
    fast and slow trials, hit rate minus false alarm rate over all 80 test trials, one column per rule
    and 'excluded' (any rule)."""
    rows = []
    for participant in sorted(test_trials['participant'].unique()):
        own = test_trials[test_trials['participant'] == participant]
        number_of_trials = len(own)
        key_counts = [np.sum(own['response_1to6'] == key) for key in range(1, 7)]
        one_key_share = max(key_counts) / number_of_trials
        fast_share = np.sum(own['rt_ms'] < FAST_RT_MS) / number_of_trials
        slow_share = np.sum(own['rt_ms'] > slow_rt_ms) / number_of_trials
        old_trials = own[own['is_new'] == 0]
        new_trials = own[own['is_new'] == 1]
        hit = np.mean(old_trials['response_1to6'] <= 3)
        false_alarm = np.mean(new_trials['response_1to6'] <= 3)
        row = dict(participant=participant, test_trials=number_of_trials, one_key_share=one_key_share,
                   fast_share=fast_share, slow_share=slow_share, hit_minus_fa=hit - false_alarm,
                   same_key=one_key_share > ONE_KEY_LIMIT, too_fast=fast_share > RULE_SHARE_LIMIT,
                   too_slow=slow_share > RULE_SHARE_LIMIT, low_cr=(hit - false_alarm) < MIN_HIT_MINUS_FA)
        row['excluded'] = row['same_key'] or row['too_fast'] or row['too_slow'] or row['low_cr']
        rows.append(row)
    return pd.DataFrame(rows)


def apply_exclusions(test_trials, slow_rt_ms=SLOW_RT_MS, cap=PARTICIPANTS_KEPT):
    """Exclude participants by the rules, then drop trials outside 200 to 10,000 ms, then keep the
    first `cap` remaining participants (S-id order). Returns (kept test trials, flags table)."""
    flags = participant_flags(test_trials, slow_rt_ms=slow_rt_ms)
    excluded = set(flags.loc[flags['excluded'], 'participant'])
    kept = test_trials[~test_trials['participant'].isin(excluded)]
    kept = kept[(kept['rt_ms'] >= TRIAL_FILTER_MIN_MS) & (kept['rt_ms'] <= TRIAL_FILTER_MAX_MS)]
    remaining = sorted(kept['participant'].unique())
    if len(remaining) > cap:
        kept = kept[kept['participant'].isin(remaining[:cap])]
    return kept.reset_index(drop=True), flags


def items_table(data_folder=DATA_FOLDER, slow_rt_ms=SLOW_RT_MS):
    """The 40 critical items over the kept participants (the per-item table of Fig. 3d, e).
    Old = response 1 to 3. cr = hit_rate - fa_rate (corrected recognition)."""
    test_trials = load_trials('test', data_folder)
    kept, _ = apply_exclusions(test_trials, slow_rt_ms=slow_rt_ms)
    design = load_design(data_folder).set_index('image_name')
    critical = kept[kept['is_critical']]
    rows = []
    for image_name in sorted(critical['image_name'].unique()):
        image_trials = critical[critical['image_name'] == image_name]
        old_trials = image_trials[image_trials['is_new'] == 0]
        new_trials = image_trials[image_trials['is_new'] == 1]
        hits = int(np.sum(old_trials['response_1to6'] <= 3))
        false_alarms = int(np.sum(new_trials['response_1to6'] <= 3))
        hit_rate = hits / len(old_trials)
        fa_rate = false_alarms / len(new_trials)
        rows.append(dict(image_name=image_name,
                         block=int(design.loc[image_name, 'block']),
                         memorability_quintile=int(design.loc[image_name, 'memorability_quintile']),
                         n_similar_items=int(design.loc[image_name, 'n_similar_items']),
                         kramer_cr=float(design.loc[image_name, 'kramer_cr']),
                         n_old=len(old_trials), hits=hits, n_new=len(new_trials), false_alarms=false_alarms,
                         hit_rate=hit_rate, fa_rate=fa_rate, cr=hit_rate - fa_rate))
    table = pd.DataFrame(rows)
    assert len(table) == 40
    return table
