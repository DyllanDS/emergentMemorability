"""
roc_points.py: the confidence ratings of the 100 registered participants of the rare-feature face
study, counted for each kind of test face, and the empirical ROC points of Fig. 4c (Fig. 4 caption:
"Each point is one criterion on the eight-step scale formed by the old or new answer and the
four-point confidence rating, pooled over participants").

WHAT IT DOES
Every test trial has an old/new answer and a confidence rating from 1 to 4 (Guess, Maybe, Probably,
Certain). As in the registered analysis, the two form one eight-step rating: 1 = surest "new" ... 4 =
least sure "new", 5 = least sure "old" ... 8 = surest "old" (face_study_common.rating). Trials are sorted
into the four kinds of test face of the fitted model (sdt_shared_criteria.item_type):
  studied_rare        a studied face carrying a rare feature (model cell rare_old)
  same_feature_lure   the never-studied partner carrying the same rare feature (rare_foil)
  studied_ordinary    a studied ordinary face (ordinary_old)
  never_studied       a never-studied ordinary face (unrelated; the anchor of the fit)
and the trials at each rating are counted over the 100 participants (pooled, as the fit pools them).
The seven operating points of a kind of face are the shares of its trials rated at or above 2, 3, ...,
8, from the most lenient criterion to the strictest. Fig. 4c plots a curve's points as (operating
points of its lure kind, operating points of its target kind).
Checks: 100 participants, every one passing the registered rule; 3, 3, 15 and 15 trials of the four
kinds for every participant; 3,600 trials in all.

INPUTS   04_rare_feature_faces_fig4/data/face_trials.csv (through face_study_common.py, the same records the
         unequal-variance fit reads)
OUTPUTS  outputs/rating_counts_by_face_kind.csv (counts of the eight ratings per kind of face)
         outputs/roc_operating_points.csv (the seven operating points per kind of face)
         outputs/rating_counts_check.txt (the counts, the checks and the rating use over all trials)
SEEDS    none (counting)
RUNTIME  about a second
RUN      python A3c_face_study_model_and_sdt_figS10_S11/code/roc_points.py
"""
import csv
import os
import sys

sys.dont_write_bytecode = True

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_FOLDER)
import face_study_common as F         # noqa: E402
import sdt_shared_criteria as S     # noqa: E402  (item_type: the four kinds of face of the fitted model)

PATHS = F.release_paths.module_paths(__file__)

# The four kinds of test face: the name used in the outputs, and the model's name for it
FACE_KINDS = [('studied_rare', 'rare_old'),
              ('same_feature_lure', 'rare_foil'),
              ('studied_ordinary', 'ordinary_old'),
              ('never_studied', 'unrelated')]
TRIALS_PER_PARTICIPANT = {'studied_rare': 3, 'same_feature_lure': 3, 'studied_ordinary': 15, 'never_studied': 15}
REGISTERED_SAMPLE_SIZE = 100


def main():
    model_name_to_kind = {model_name: kind for kind, model_name in FACE_KINDS}
    counts = {}
    for kind, _ in FACE_KINDS:
        counts[kind] = {rating: 0 for rating in range(1, 9)}
    overall_rating_use = {rating: 0 for rating in range(1, 9)}

    loaded = F.load()
    participants = F.analysable(loaded)
    every_participant_has_the_design = True
    for participant in participants:
        trials_of_each_kind = {kind: 0 for kind, _ in FACE_KINDS}
        for trial in participant['rec']:
            model_name = S.item_type(trial)
            if model_name is None:
                continue
            kind = model_name_to_kind[model_name]
            rating = F.rating(trial)
            counts[kind][rating] += 1
            overall_rating_use[rating] += 1
            trials_of_each_kind[kind] += 1
        if trials_of_each_kind != TRIALS_PER_PARTICIPANT:
            every_participant_has_the_design = False
            print('participant %s: trials per kind %s' % (participant['participant'], trials_of_each_kind))

    # ---------------------------------------------------------------- the counts
    with open(os.path.join(PATHS['outputs'], 'rating_counts_by_face_kind.csv'), 'w', newline='',
              encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['face_kind', 'model_cell'] + ['rating_%d' % rating for rating in range(1, 9)]
                        + ['total_trials'])
        for kind, model_name in FACE_KINDS:
            writer.writerow([kind, model_name] + [counts[kind][rating] for rating in range(1, 9)]
                            + [sum(counts[kind].values())])

    # ---------------------------------------------------------------- the operating points
    operating_points = {}
    for kind, _ in FACE_KINDS:
        total = sum(counts[kind].values())
        points = []
        for lowest_rating in range(2, 9):
            trials_at_or_above = 0
            for rating in range(lowest_rating, 9):
                trials_at_or_above += counts[kind][rating]
            points.append(trials_at_or_above / total)
        operating_points[kind] = points
    with open(os.path.join(PATHS['outputs'], 'roc_operating_points.csv'), 'w', newline='',
              encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['face_kind', 'model_cell'] + ['rated_%d_or_higher' % rating for rating in range(2, 9)])
        for kind, model_name in FACE_KINDS:
            writer.writerow([kind, model_name] + [repr(point) for point in operating_points[kind]])

    # ---------------------------------------------------------------- the report
    lines = ['Rating counts of the 100 registered participants', '================================================', '',
             'participants read: %d; passing the registered rule: %d (of them, the registered sample: %s)'
             % (len(loaded), len(participants), 'yes' if len(participants) == REGISTERED_SAMPLE_SIZE else 'NO'),
             'every participant has 3, 3, 15 and 15 trials of the four kinds: %s'
             % ('yes' if every_participant_has_the_design else 'NO'), '']
    for kind, model_name in FACE_KINDS:
        total = sum(counts[kind].values())
        proportion_old = sum(counts[kind][rating] for rating in range(5, 9)) / total
        lines.append('%-18s (%-12s) trials %5d   proportion called old %.4f   counts %s'
                     % (kind, model_name, total, proportion_old, [counts[kind][rating] for rating in range(1, 9)]))
    lines += ['', 'rating use over all trials: %s (total %d)'
              % ([overall_rating_use[rating] for rating in range(1, 9)], sum(overall_rating_use.values())), '',
              'operating points (share of trials rated 2 or higher, 3 or higher, ..., 8):']
    for kind, _ in FACE_KINDS:
        lines.append('  %-18s %s' % (kind, ' '.join('%.3f' % point for point in operating_points[kind])))
    report = '\n'.join(lines) + '\n'
    with open(os.path.join(PATHS['outputs'], 'rating_counts_check.txt'), 'w', encoding='utf-8') as handle:
        handle.write(report)
    print(report)
    if len(participants) != REGISTERED_SAMPLE_SIZE or not every_participant_has_the_design:
        sys.exit('roc_points.py: the data do not have the registered sample or design')


if __name__ == '__main__':
    main()
