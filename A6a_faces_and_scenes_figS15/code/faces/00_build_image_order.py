"""
00_build_image_order.py (faces): the list of the 10,168 face images with the memorability scores of
the 2,222 scored faces, in the row order every face embedding follows.

This script runs only when the face database is available (EXTERNAL_INPUTS.md, M4): the images and the
per-face scores of Bainbridge, Isola and Oliva (2013) may not be redistributed, so neither this list
nor anything computed per face is in the repository. The database is shared on request
(https://wilmabainbridge.com/facememorability2.html). code/faces/regenerate_faces.py calls this script
when the archives are in external/bainbridge_faces/.
What it does
    Lists every face image on disk (all 10,168, unscored fillers included, because participants saw
    the fillers too and they crowd the scored faces), sorts the list by file name, and joins the
    scores of the 2,222 scored faces from the database's memorability-scores.xlsx by file name:
    hit rate, false alarm rate, corrected recognition (hit rate minus false alarm rate) and the hit,
    miss, false alarm and correct-rejection counts (the counts are needed by 05_noise_ceiling.py).
    Unscored faces get empty scores (NaN) and has_scores = 0. Files whose names start with "._"
    (macOS resource files in some zip archives) are not images and are skipped.
    Every image goes into the list, not only the 2,222 with scores: the scores are the outcome, but
    the predictor is how crowded a face's neighborhood is among the faces participants saw, and they
    saw fillers as well as scored targets. Crowding over the scored subset alone would measure density
    in an arbitrary sample rather than in the set the faces sat in.
Inputs (command line, all required)
    --images-dir    the folder of face images ("Face Images" of 10kfaces.zip)
    --scores-file   memorability-scores.xlsx (of 2kmemorability.zip; reading it needs openpyxl)
    --out           where to write image_order.csv (regenerate_faces.py: external/bainbridge_faces/derived/)
Output
    image_order.csv: filename, has_scores, hit_rate, false_alarm_rate, corrected_recognition, hits,
    misses, false_alarms, correct_rejections (one row per image, sorted by file name)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}

# Column names to look for in the score file (memorability-scores.xlsx); the
# first name of each list that the file has is used.
FILENAME_COL_CANDIDATES = ['Filename', 'filename', 'image', 'Image', 'File']
HR_COL_CANDIDATES = ['Hit Rate (HR)', 'HR', 'hit_rate', 'HitRate', 'Hit Rate']
FA_COL_CANDIDATES = ['False Alarm Rate (FAR)', 'FAR', 'FA', 'false_alarm_rate',
                     'FalseAlarmRate', 'False Alarm Rate']
CR_COL_CANDIDATES = ['CR', 'corrected_recognition', 'CorrectedRecognition',
                      'Corrected Recognition']
# the trial counts (needed by 05_noise_ceiling.py)
COUNT_COL_CANDIDATES = {
    'hits': ['Hits', 'hits', 'HITS'],
    'misses': ['Misses', 'misses', 'MISSES'],
    'false_alarms': ['False Alarms', 'false_alarms', 'FALSE ALARMS', 'FA count'],
    'correct_rejections': ['Correct Rejections', 'correct_rejections', 'CORRECT REJECTIONS'],
}


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_scores(scores_path):
    if scores_path.suffix.lower() == '.csv':
        df = pd.read_csv(scores_path)
    elif scores_path.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(scores_path)
    elif scores_path.suffix.lower() == '.mat':
        raise NotImplementedError(
            '.mat scores file given; use memorability-scores.xlsx of the same archive.')
    else:
        raise ValueError(f'Unrecognized scores file type: {scores_path}')
    return df


def main():
    ap = argparse.ArgumentParser(description='Build the face image list with the Bainbridge scores.')
    ap.add_argument('--images-dir', type=Path, required=True)
    ap.add_argument('--scores-file', type=Path, required=True,
                    help='memorability-scores.xlsx of the database (xlsx/csv).')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    if not args.images_dir.exists():
        sys.exit(f'ERROR: images dir not found: {args.images_dir}\n'
                 'Obtain and unzip the 10k US Adult Faces Database first '
                 '(see external/bainbridge_faces/README.txt).')
    if not args.scores_file.exists():
        sys.exit(f'ERROR: scores file not found: {args.scores_file}')

    on_disk = {p.name: p for p in args.images_dir.rglob('*')
               if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith('._')}
    print(f'Found {len(on_disk)} image files under the images folder')

    df = load_scores(args.scores_file)
    fn_col = find_col(df, FILENAME_COL_CANDIDATES)
    if fn_col is None:
        sys.exit(f'ERROR: could not find a filename column in {args.scores_file}. '
                 f'Columns present: {list(df.columns)}. '
                 f'Add the real column name to FILENAME_COL_CANDIDATES and run the script again.')
    hr_col = find_col(df, HR_COL_CANDIDATES)
    fa_col = find_col(df, FA_COL_CANDIDATES)
    cr_col = find_col(df, CR_COL_CANDIDATES)
    count_cols = {name: find_col(df, candidates) for name, candidates in COUNT_COL_CANDIDATES.items()}
    print(f'filename col: {fn_col!r}; score cols found: '
          f'HR={hr_col!r} FA={fa_col!r} CR={cr_col!r}; counts {count_cols}')
    if hr_col is None and fa_col is None and cr_col is None:
        sys.exit('ERROR: no memorability score column found. '
                 f'Columns present: {list(df.columns)}. '
                 'Add the real column name(s) to the *_COL_CANDIDATES lists and run the script again.')

    # Index the scores by filename, then write one row per image ON DISK. Every
    # image is kept, not only the scored ones: the scores are the outcome, but
    # the predictor is how crowded a face's neighborhood is among the faces
    # participants actually saw, and they saw unscored fillers too.
    scores = {}
    n_missing_image = 0
    for _, r in df.iterrows():
        fname = str(r[fn_col]).strip()
        if fname not in on_disk:
            n_missing_image += 1
            continue
        rec = {}
        if hr_col:
            rec['hit_rate'] = r[hr_col]
        if fa_col:
            rec['false_alarm_rate'] = r[fa_col]
        if cr_col:
            rec['corrected_recognition'] = r[cr_col]
        for name, column in count_cols.items():
            if column:
                rec[name] = r[column]
        scores[fname] = rec

    rows = []
    for fname in sorted(on_disk):
        row = {'filename': fname, 'has_scores': int(fname in scores)}
        row.update(scores.get(fname, {}))
        rows.append(row)

    out_df = pd.DataFrame(rows)
    # the archive holds hit and false alarm rates but not corrected recognition
    if ('corrected_recognition' not in out_df.columns
            and {'hit_rate', 'false_alarm_rate'} <= set(out_df.columns)):
        out_df['corrected_recognition'] = (out_df['hit_rate']
                                           - out_df['false_alarm_rate'])
    cols = ['filename', 'has_scores'] + [c for c in
            ('hit_rate', 'false_alarm_rate', 'corrected_recognition',
             'hits', 'misses', 'false_alarms', 'correct_rejections')
            if c in out_df.columns]
    out_df = out_df[cols]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    n_scored = int(out_df['has_scores'].sum())
    print(f'\nImages on disk:            {len(on_disk)}')
    print(f'Score rows in file:        {len(df)}')
    print(f'Score rows with no image:  {n_missing_image}')
    print(f'Images WITH scores:        {n_scored}')
    print(f'Images WITHOUT scores:     {len(out_df) - n_scored}  (kept as context)')
    print(f'Join rate against scores:  {n_scored / max(len(df), 1):.1%}')
    if n_scored == len(out_df):
        print('WARNING: every image is scored. If only the scored subset was')
        print('  downloaded, crowding will be computed over that subset rather')
        print('  than over the full set participants saw.')
    print('Wrote the face image list (image_order.csv)')


if __name__ == '__main__':
    main()
