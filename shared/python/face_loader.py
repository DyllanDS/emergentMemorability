"""
face_loader.py: the preregistered rare-feature face study (Fig. 4; SI Appendix, Section 3).

Reads 04_rare_feature_faces_fig4/data/ (de-identified):
    face_trials.csv        the 111 sessions run (S001..S111 in completion order): 56 study rows
                           (50 faces + 6 one-back repeats) and 36 recognition rows each
    face_participants.csv  one row per session: completion_rank, load_order, oneback_hits,
                           oneback_false_presses, passes_rule, registered

    load_trials(phase)             'study', 'recognition' or 'all'
    load_participants()            face_participants.csv
    recompute_participants()       the participant table recomputed from the trials (the registered
                                   rule), checked against face_participants.csv column by column
    registered_participants(order) the 100 registered participants, in load_order (default: the order
                                   in which the registered analysis loads the sessions, on which its
                                   seeded bootstrap depends) or in completion order
    participant_rates(order)       one row per registered participant: hit and false alarm rates of
                                   rare and ordinary faces, each against its own kind of lure, and the
                                   log-linear d' and c

The preregistered rule: 36 recognition trials, at least 4 of the 6 one-back repeats detected
(oneback_correct on repeat trials) and at most 5 presses on non-repeats; the first 100 passing
sessions in completion order form the registered sample (11 of the 111 fail: 8 with more than 5
false presses, 4 with fewer than 4 repeats detected, 1 both).

Rates: proportion of 'old' responses (said_old) among the recognition rows of a cell: rare faces
studied = condition 'rare', test_type 'old'; their look-alike foils = 'rare', 'foil'; ordinary faces
studied = 'filler', 'old'; never-studied faces = 'filler', 'new'.
"""
import os

import numpy as np
import pandas as pd

import sdt

_RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FOLDER = os.path.join(_RELEASE_ROOT, '04_rare_feature_faces_fig4', 'data')

MIN_ONEBACK_HITS = 4
MAX_ONEBACK_FA = 5
EXPECTED_RECOGNITION_TRIALS = 36
REGISTERED_SAMPLE_SIZE = 100


def load_trials(phase='all', data_folder=DATA_FOLDER):
    """Trial rows in the order they were run. phase: 'study', 'recognition' or 'all'."""
    trials = pd.read_csv(os.path.join(data_folder, 'face_trials.csv'), keep_default_na=False,
                         na_values=[''], float_precision='round_trip')
    if phase in ('study', 'recognition'):
        trials = trials[trials['phase'] == phase].reset_index(drop=True)
    elif phase != 'all':
        raise ValueError("phase must be 'study', 'recognition' or 'all'")
    return trials


def load_participants(data_folder=DATA_FOLDER):
    """face_participants.csv (111 rows, completion order)."""
    return pd.read_csv(os.path.join(data_folder, 'face_participants.csv'))


def recompute_participants(data_folder=DATA_FOLDER):
    """Recompute every column of face_participants.csv from the trials and check it matches."""
    trials = load_trials('all', data_folder)
    shipped = load_participants(data_folder)
    rows = []
    passing_so_far = 0
    for participant in shipped.sort_values('completion_rank')['participant']:
        own = trials[trials['participant'] == participant]
        study = own[own['phase'] == 'study']
        recognition = own[own['phase'] == 'recognition']
        repeats = study[study['is_repeat'] == 1]
        oneback_hits = int(np.sum(repeats['oneback_correct'] == 1))
        false_presses = int(np.sum((study['is_repeat'] == 0) & (study['pressed'] == 1)))
        passes = (len(recognition) == EXPECTED_RECOGNITION_TRIALS and oneback_hits >= MIN_ONEBACK_HITS
                  and false_presses <= MAX_ONEBACK_FA)
        registered = passes and passing_so_far < REGISTERED_SAMPLE_SIZE
        if passes:
            passing_so_far += 1
        rows.append(dict(participant=participant, completion_rank=int(own['completion_rank'].iloc[0]),
                         load_order=int(own['load_order'].iloc[0]), oneback_hits=oneback_hits,
                         oneback_false_presses=false_presses, passes_rule=int(passes), registered=int(registered)))
    recomputed = pd.DataFrame(rows)
    shipped_sorted = shipped.sort_values('completion_rank').reset_index(drop=True)
    for column in recomputed.columns:
        if not (recomputed[column].to_numpy() == shipped_sorted[column].to_numpy()).all():
            raise ValueError('face_participants.csv column %s differs from the recomputed value' % column)
    return recomputed


def registered_participants(order='load_order', data_folder=DATA_FOLDER):
    """The 100 registered participant ids, in load_order (default) or completion order."""
    table = recompute_participants(data_folder)
    registered = table[table['registered'] == 1]
    assert len(registered) == REGISTERED_SAMPLE_SIZE
    if order == 'load_order':
        registered = registered.sort_values('load_order')
    elif order == 'completion':
        registered = registered.sort_values('completion_rank')
    else:
        raise ValueError("order must be 'load_order' or 'completion'")
    return list(registered['participant'])


def _rate(recognition_rows, condition, test_type):
    cell = recognition_rows[(recognition_rows['condition'] == condition) & (recognition_rows['test_type'] == test_type)]
    if len(cell) == 0:
        return np.nan, 0
    return float(np.mean(cell['said_old'])), len(cell)


def participant_rates(order='load_order', data_folder=DATA_FOLDER):
    """One row per registered participant (in the given order) with
    hit_rare  (rare face studied, said old)        n_hr
    fa_rare   (its look-alike foil, said old)      n_fr
    hit_fill  (ordinary face studied, said old)    n_hf
    fa_new    (never-studied face, said old)       n_fn
    d_rare, c_rare, d_fill, c_fill: log-linear d' and criterion (sdt.dprime_loglinear), each kind of
    face against its own kind of lure."""
    recognition = load_trials('recognition', data_folder)
    rows = []
    for participant in registered_participants(order, data_folder):
        own = recognition[recognition['participant'] == participant]
        hit_rare, n_hr = _rate(own, 'rare', 'old')
        fa_rare, n_fr = _rate(own, 'rare', 'foil')
        hit_fill, n_hf = _rate(own, 'filler', 'old')
        fa_new, n_fn = _rate(own, 'filler', 'new')
        d_rare, c_rare = sdt.dprime_loglinear(hit_rare, fa_rare, n_hr, n_fr)
        d_fill, c_fill = sdt.dprime_loglinear(hit_fill, fa_new, n_hf, n_fn)
        rows.append(dict(participant=participant, hit_rare=hit_rare, n_hr=n_hr, fa_rare=fa_rare, n_fr=n_fr,
                         hit_fill=hit_fill, n_hf=n_hf, fa_new=fa_new, n_fn=n_fn,
                         d_rare=float(d_rare), c_rare=float(c_rare), d_fill=float(d_fill), c_fill=float(c_fill)))
    return pd.DataFrame(rows)
