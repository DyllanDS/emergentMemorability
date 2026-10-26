"""
same_face_scores.py: the rare-feature face study (Fig. 4) scored the way memorability is usually scored, each face
against itself across participants (main text, A Rare Feature's Advantage Depends on What the Face Is Tested
Against).

WHAT IT DOES
  Which member of each rare-feature pair was studied was drawn for each participant, and each ordinary face was
  studied by some participants and new for others, so each of the 12 rare-feature faces and 59 ordinary faces was
  tested both after it had been studied and when it had not. A new trial of a rare-feature face is always a trial
  on which the other member of its pair, carrying the same feature, had been studied (test_type 'foil'); the code
  stops if one is not.
  For each face, over the recognition trials of the 100 registered participants (module 04's data, through
  face_loader.py), the code counts the hits among the old trials and the false alarms among the new trials, and
  computes the hit rate, the false alarm rate, the hit rate minus the false alarm rate and the log-linear d'
  (sdt.dprime_loglinear_counts: 0.5 added to each count and 1 to each number of trials). It then averages each
  measure over the rare-feature faces and over the ordinary faces, each face weighted equally. No random numbers
  are drawn.

INPUTS   04_rare_feature_faces_fig4/data/face_trials.csv and face_participants.csv (through
         shared/python/face_loader.py)
OUTPUTS  outputs/same_face_items.csv     one row per face: image, rare, hit, fa, cr, dprime, n_old, n_new
         outputs/same_face_summary.csv   the mean of each measure over the rare-feature and the ordinary faces
RUNTIME  a few seconds
"""
import os
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import face_loader  # noqa: E402
import release_paths  # noqa: E402
import sdt  # noqa: E402

PATHS = release_paths.module_paths(__file__, make_output_folders=False)
MEASURES = ['hit rate', 'false alarm rate', 'hit rate minus false alarm rate', "d'"]
COLUMN_OF_MEASURE = {'hit rate': 'hit', 'false alarm rate': 'fa', 'hit rate minus false alarm rate': 'cr',
                     "d'": 'dprime'}


def score_each_face(recognition):
    """One row per tested face, sorted by image file: its trial counts in the given recognition rows and its
    four measures."""
    rows = []
    for image in sorted(recognition['image'].unique()):
        trials = recognition[recognition['image'] == image]
        old_trials = trials[trials['is_old'] == 1]
        new_trials = trials[trials['is_old'] == 0]
        is_rare = bool(trials['condition'].iloc[0] == 'rare')
        if is_rare and not (new_trials['test_type'] == 'foil').all():
            raise SystemExit('%s: a new trial of this rare-feature face is not a foil trial' % image)
        hits = int(old_trials['said_old'].sum())
        false_alarms = int(new_trials['said_old'].sum())
        n_old = len(old_trials)
        n_new = len(new_trials)
        hit = float(sdt.hit_rate(hits, n_old))
        false_alarm = float(sdt.false_alarm_rate(false_alarms, n_new))
        dprime, _ = sdt.dprime_loglinear_counts(hits, n_old, false_alarms, n_new)
        rows.append(dict(image=image.split('/')[-1], rare=is_rare, hit=hit, fa=false_alarm,
                         cr=float(sdt.corrected_recognition(hit, false_alarm)), dprime=float(dprime),
                         n_old=n_old, n_new=n_new))
    return pd.DataFrame(rows)


def main():
    """Scores every face, writes the two tables and returns the summary (columns measure, rare, ordinary)."""
    recognition = face_loader.load_trials('recognition')
    registered = face_loader.registered_participants(order='completion')
    recognition = recognition[recognition['participant'].isin(registered)]
    faces = score_each_face(recognition)
    is_rare = faces['rare'].to_numpy()
    print('Rare-feature face study, each face scored against itself: %d participants, %d rare-feature and %d '
          'ordinary faces, %d to %d old and %d to %d new trials per face'
          % (len(registered), is_rare.sum(), (~is_rare).sum(), faces['n_old'].min(), faces['n_old'].max(),
             faces['n_new'].min(), faces['n_new'].max()))
    summary_rows = []
    for measure in MEASURES:
        values = faces[COLUMN_OF_MEASURE[measure]].to_numpy()
        rare_mean = float(np.mean(values[is_rare]))
        ordinary_mean = float(np.mean(values[~is_rare]))
        summary_rows.append(dict(measure=measure, rare=rare_mean, ordinary=ordinary_mean))
        print('  %-32s rare-feature faces %.3f   ordinary faces %.3f' % (measure, rare_mean, ordinary_mean))
    summary = pd.DataFrame(summary_rows)
    os.makedirs(PATHS['outputs'], exist_ok=True)
    faces.to_csv(os.path.join(PATHS['outputs'], 'same_face_items.csv'), index=False)
    summary.to_csv(os.path.join(PATHS['outputs'], 'same_face_summary.csv'), index=False)
    return summary


if __name__ == '__main__':
    main()
