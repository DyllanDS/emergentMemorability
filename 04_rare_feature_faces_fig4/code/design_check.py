"""
design_check.py: the design of the rare-feature face study, counted from the data.

WHAT IT DOES
Every design number of this experiment that the paper reports (the section "A Rare Feature's
Advantage Depends on What the Face Is Tested Against", the Fig. 4 caption and Methods, Behavioral
Validation Experiments) is counted here from the sessions themselves or read from the task
settings, and the script stops with an error if any session breaks the design. It checks:
  - the task settings (face_design.json, the settings of the experiment program): 3,000 ms per
    face at study, 500 ms between faces, 60,000 ms of arithmetic before the test
  - the stimulus set (the manifest in face_design.json): six rare-feature pairs (twelve faces) and
    59 ordinary faces
  - in every one of the 111 sessions: 50 different faces studied (44 ordinary + one member of each
    of the six rare-feature pairs), 6 one-back repeats, each shown directly after its own first
    showing and each an ordinary face that is never tested (so 56 study presentations), and 36 test
    trials: 3 studied rare faces, 3 new faces carrying a studied face's rare feature, 15 studied
    ordinary faces and 15 never-studied ordinary faces, each with a confidence rating from 1 to 4
  - which member of each pair was studied and which three pairs were tested with the studied face,
    over the 100 registered participants (both were drawn at random for each participant)

INPUTS   data/face_trials.csv, data/face_participants.csv (through shared/python/face_loader.py),
         data/face_design.json
OUTPUTS  outputs/design_check.csv  (fact, value, source)
         printed summary
SEEDS    none (nothing random)
RUN      python 04_rare_feature_faces_fig4/code/design_check.py   (from the repository root)
"""
import sys
sys.dont_write_bytecode = True
import csv
import json
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths
import face_loader

PATHS = release_paths.module_paths(__file__)

# ----------------------------------------------------------------------------- inputs
with open(os.path.join(PATHS['data'], 'face_design.json'), encoding='utf-8') as handle:
    design = json.load(handle)
task_settings = design['task_constants']
manifest = design['stimulus_manifest']

all_trials = face_loader.load_trials('all')
participant_table = face_loader.recompute_participants()          # also checks face_participants.csv
registered_ids = face_loader.registered_participants(order='completion')

facts = []      # one row per fact: (fact, value, source)


def add_fact(fact, value, source):
    facts.append((fact, value, source))
    print('  %-62s %s' % (fact, value))


def same_in_every_session(values_by_session, description):
    """All sessions must give the same value; return it (stop with an error otherwise)."""
    distinct_values = sorted(set(values_by_session.values()), key=str)
    if len(distinct_values) != 1:
        raise SystemExit('DESIGN CHECK FAILED: %s differs between sessions: %s' % (description, distinct_values))
    return distinct_values[0]


print('Design of the rare-feature face study, counted from the data')

# ----------------------------------------------------------------------------- 1. task settings
study_duration_ms = int(task_settings['study_ms'])
gap_between_faces_ms = int(task_settings['study_isi_ms'])
arithmetic_duration_ms = int(task_settings['filler_task_ms'])
add_fact('study time per face (ms)', study_duration_ms, 'face_design.json task_constants.study_ms')
add_fact('gap between study faces (ms)', gap_between_faces_ms, 'face_design.json task_constants.study_isi_ms')
add_fact('arithmetic filler task (ms)', arithmetic_duration_ms, 'face_design.json task_constants.filler_task_ms')

# ----------------------------------------------------------------------------- 2. stimulus set
number_of_pairs = len(manifest['pairs'])
rare_feature_faces = set()
features_of_pairs = []
for pair in manifest['pairs']:
    rare_feature_faces.add(pair['target'])
    rare_feature_faces.add(pair['foil'])
    features_of_pairs.append(pair['feature'])
number_of_ordinary_faces = len(manifest['ordinary'])
# The paper names two features (mohawk, beard) and calls the other four combinations of makeup,
# ink, piercings and eyewear.
features_other_than_mohawk_and_beard = [feature for feature in features_of_pairs
                                        if feature not in ('mohawk', 'beard')]
add_fact('rare-feature pairs in the stimulus set', number_of_pairs, 'face_design.json stimulus_manifest.pairs')
add_fact('rare-feature faces in the stimulus set', len(rare_feature_faces), 'face_design.json stimulus_manifest.pairs')
add_fact('rare features', ', '.join(features_of_pairs), 'face_design.json stimulus_manifest.pairs')
add_fact('rare features other than the mohawk and the beard', len(features_other_than_mohawk_and_beard),
         'face_design.json stimulus_manifest.pairs')
add_fact('ordinary faces in the stimulus set', number_of_ordinary_faces, 'face_design.json stimulus_manifest.ordinary')

# ----------------------------------------------------------------------------- 3. every session
distinct_faces_studied = {}
ordinary_faces_studied = {}
rare_faces_studied = {}
pairs_with_one_member_studied = {}
study_presentations = {}
number_of_repeats = {}
repeats_follow_their_first_showing = {}
repeats_are_untested_ordinary_faces = {}
test_trials = {}
studied_rare_tests = {}
same_feature_lure_tests = {}
studied_ordinary_tests = {}
never_studied_ordinary_tests = {}
confidence_outside_1_to_4 = {}
ordinary_images_used = set()
rare_images_used = set()

for participant in sorted(all_trials['participant'].unique()):
    own_trials = all_trials[all_trials['participant'] == participant]
    study_rows = own_trials[own_trials['phase'] == 'study'].sort_values('trial_in_phase')
    test_rows = own_trials[own_trials['phase'] == 'recognition']

    # study list: first showings and the one-back repeats
    first_showings = study_rows[study_rows['is_repeat'] == 0]
    repeat_rows = study_rows[study_rows['is_repeat'] == 1]
    distinct_faces_studied[participant] = first_showings['image'].nunique()
    ordinary_faces_studied[participant] = int((first_showings['condition'] == 'filler').sum())
    rare_faces_studied[participant] = int((first_showings['condition'] == 'rare').sum())
    study_presentations[participant] = len(study_rows)
    number_of_repeats[participant] = len(repeat_rows)

    # one member of each pair studied: the studied rare faces carry six different pair cells
    studied_rare_rows = first_showings[first_showings['condition'] == 'rare']
    pairs_with_one_member_studied[participant] = studied_rare_rows['cell'].nunique() == number_of_pairs \
        and len(studied_rare_rows) == number_of_pairs

    # each repeat directly follows the first showing of the same face (a 1-back repeat)
    images_in_study_order = list(study_rows['image'])
    repeat_flags_in_study_order = list(study_rows['is_repeat'])
    every_repeat_is_one_back = True
    for position in range(len(images_in_study_order)):
        if repeat_flags_in_study_order[position] == 1:
            if position == 0 or images_in_study_order[position - 1] != images_in_study_order[position]:
                every_repeat_is_one_back = False
    repeats_follow_their_first_showing[participant] = every_repeat_is_one_back

    # the repeated faces are ordinary faces that never appear at test
    tested_images = set(test_rows['image'])
    repeated_images_are_fine = True
    for _, repeat_row in repeat_rows.iterrows():
        if repeat_row['condition'] != 'filler' or repeat_row['image'] in tested_images:
            repeated_images_are_fine = False
    repeats_are_untested_ordinary_faces[participant] = repeated_images_are_fine

    # the test
    test_trials[participant] = len(test_rows)
    studied_rare_tests[participant] = int(((test_rows['condition'] == 'rare') & (test_rows['test_type'] == 'old')).sum())
    same_feature_lure_tests[participant] = int(((test_rows['condition'] == 'rare') & (test_rows['test_type'] == 'foil')).sum())
    studied_ordinary_tests[participant] = int(((test_rows['condition'] == 'filler') & (test_rows['test_type'] == 'old')).sum())
    never_studied_ordinary_tests[participant] = int(((test_rows['condition'] == 'filler') & (test_rows['test_type'] == 'new')).sum())
    confidence_outside_1_to_4[participant] = int((~test_rows['confidence_1to4'].isin([1, 2, 3, 4])).sum())

    # which images were used at all
    for _, row in own_trials.iterrows():
        image_name = row['image'].split('/')[-1]
        if row['condition'] == 'filler':
            ordinary_images_used.add(image_name)
        else:
            rare_images_used.add(image_name)

number_of_sessions = len(distinct_faces_studied)
source_sessions = 'face_trials.csv (the same in all %d sessions)' % number_of_sessions
add_fact('different faces studied per session', same_in_every_session(distinct_faces_studied, 'faces studied'), source_sessions)
add_fact('ordinary faces studied per session', same_in_every_session(ordinary_faces_studied, 'ordinary faces studied'), source_sessions)
add_fact('rare-feature faces studied per session', same_in_every_session(rare_faces_studied, 'rare faces studied'), source_sessions)
add_fact('one member of every rare-feature pair studied',
         'yes' if same_in_every_session(pairs_with_one_member_studied, 'one member per pair') else 'no', source_sessions)
add_fact('study presentations per session (faces + repeats)', same_in_every_session(study_presentations, 'study presentations'), source_sessions)
add_fact('one-back repeats per session', same_in_every_session(number_of_repeats, 'repeats'), source_sessions)
if not same_in_every_session(repeats_follow_their_first_showing, 'repeat position'):
    raise SystemExit('DESIGN CHECK FAILED: a repeat does not directly follow its own first showing')
add_fact('repeat task (each repeat directly follows its own first showing)', '1-back', source_sessions)
add_fact('repeated faces are ordinary faces never tested',
         'yes' if same_in_every_session(repeats_are_untested_ordinary_faces, 'repeated faces') else 'no', source_sessions)
add_fact('test trials per session', same_in_every_session(test_trials, 'test trials'), source_sessions)
add_fact('test: studied rare-feature faces', same_in_every_session(studied_rare_tests, 'studied rare tests'), source_sessions)
add_fact('test: new faces carrying a studied rare feature', same_in_every_session(same_feature_lure_tests, 'lure tests'), source_sessions)
add_fact('test: studied ordinary faces', same_in_every_session(studied_ordinary_tests, 'studied ordinary tests'), source_sessions)
add_fact('test: never-studied ordinary faces', same_in_every_session(never_studied_ordinary_tests, 'new ordinary tests'), source_sessions)
if same_in_every_session(confidence_outside_1_to_4, 'confidence range') != 0:
    raise SystemExit('DESIGN CHECK FAILED: a confidence rating outside 1 to 4')
highest_rating = int(all_trials.loc[all_trials['phase'] == 'recognition', 'confidence_1to4'].max())
lowest_rating = int(all_trials.loc[all_trials['phase'] == 'recognition', 'confidence_1to4'].min())
add_fact('confidence scale points (one rating after every test trial)', highest_rating - lowest_rating + 1, source_sessions)
ordinary_tests = studied_ordinary_tests[registered_ids[0]] + never_studied_ordinary_tests[registered_ids[0]]
if studied_ordinary_tests[registered_ids[0]] * 2 == ordinary_tests:
    share_of_ordinary_tests_studied = 'half'
else:
    share_of_ordinary_tests_studied = '%d of %d' % (studied_ordinary_tests[registered_ids[0]], ordinary_tests)
add_fact('share of the ordinary test faces that were studied', share_of_ordinary_tests_studied, source_sessions)
add_fact('different ordinary faces used over all sessions', len(ordinary_images_used), 'face_trials.csv')
add_fact('different rare-feature faces used over all sessions', len(rare_images_used), 'face_trials.csv')
if ordinary_images_used != set(manifest['ordinary']):
    raise SystemExit('DESIGN CHECK FAILED: the ordinary faces in the data are not the manifest list')

# ----------------------------------------------------------------------------- 4. random assignments
# Which member of each pair is studied: drawn for every pair of every participant (role_swapped = 1
# when the file named '_foil' was the studied one). Which three pairs are tested with the studied
# face: decided by one random draw per participant (pairs c1, c3, c5 or pairs c2, c4, c6).
swapped_assignments = 0
pair_assignments = 0
sessions_testing_odd_pairs_old = 0
for participant in registered_ids:
    own_trials = all_trials[all_trials['participant'] == participant]
    studied_rare_rows = own_trials[(own_trials['phase'] == 'study') & (own_trials['is_repeat'] == 0)
                                   & (own_trials['condition'] == 'rare')]
    swapped_assignments += int(studied_rare_rows['role_swapped'].sum())
    pair_assignments += len(studied_rare_rows)
    old_rare_cells = set(own_trials[(own_trials['phase'] == 'recognition') & (own_trials['condition'] == 'rare')
                                    & (own_trials['test_type'] == 'old')]['cell'])
    if old_rare_cells == {'c1', 'c3', 'c5'}:
        sessions_testing_odd_pairs_old += 1
    elif old_rare_cells != {'c2', 'c4', 'c6'}:
        raise SystemExit('DESIGN CHECK FAILED: unexpected set of pairs tested old: %s' % sorted(old_rare_cells))
add_fact('registered participants: pair members studied that were the file named _foil',
         '%d of %d' % (swapped_assignments, pair_assignments), 'face_trials.csv role_swapped (100 registered)')
add_fact('registered participants tested old on pairs c1, c3, c5 (others: c2, c4, c6)',
         '%d of %d' % (sessions_testing_odd_pairs_old, len(registered_ids)), 'face_trials.csv test_type (100 registered)')

# ----------------------------------------------------------------------------- write
output_file = os.path.join(PATHS['outputs'], 'design_check.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['fact', 'value', 'source'])
    for fact, value, source in facts:
        writer.writerow([fact, value, source])
print('wrote outputs/design_check.csv (%d facts)' % len(facts))
