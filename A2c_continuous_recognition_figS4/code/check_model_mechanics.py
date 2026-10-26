"""
check_model_mechanics.py: checks that the model code in continuous_rem.py does, trial by trial, what
SI Appendix, Section 2 says it does. It needs no data and runs in about a second, so it also runs
when the trial structures of Fig. S4 are not available.

WHAT THE MODEL DOES (SI Appendix, Section 2)
  1. Memory grows during the session: on each trial the item on screen is compared with every trace
     stored on earlier trials, using REM's likelihood ratio, and the model responds "old" when the
     AVERAGE likelihood ratio across those traces exceeds its criterion.
  2. After the response the item is stored as a new trace, whether it was shown for the first time
     or repeated; traces are never forgotten.
  3. The criterion is set to the participants' 2 percent false alarm rate (check_provided.py checks
     it on the model run itself).

HOW IT CHECKS
  A slow version of one simulated session is written below directly from sentences 1 and 2, with
  explicit loops: a list of stored traces that only ever grows, a likelihood ratio computed feature by
  feature (Shiffrin and Steyvers, 1997), their average, the comparison with the criterion, and the
  similarity of the item to the 10 most similar items shown before it. It draws its random numbers
  with the same calls, in the same order, as continuous_rem.run_subject(). Both are run on a made-up
  stream of 40 trials (28 items, 12 of them shown twice; not a THINGS sequence) with made-up items,
  at the two storage probabilities and criteria of the saved run in data/provided/ (0.30 with 5.236,
  0.75 with 3.462). The check passes when, on every trial, both give the same response, the same
  context similarity and the same count of preceding trials, and both used exactly the same random
  numbers.

OUTPUT  outputs/model_mechanics_check.csv (one row per check and storage probability: check,
        operating_point, passed, detail); exit code 1 when any check fails.
SEEDS   20261013 (the made-up items) and 20261014 (the assignment of items to the stream and the
        encoding draws; a fresh generator for each version at each storage probability). No input files.

RUN (from the repository root)
  python A2c_continuous_recognition_figS4/code/check_model_mechanics.py
"""
import csv
import math
import os
import sys

import numpy as np

sys.dont_write_bytecode = True   # write no __pycache__ folder into shared/python or code/ (set before the imports below)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import continuous_rem as model  # noqa: E402

SEED = 20261013
NUMBER_OF_MADE_UP_ITEMS = 100

# A made-up stream of 40 trials: the item (slot) shown on each trial. 28 items, the first 12 of
# them shown a second time, 4 to 29 trials after their first showing. Trial type 1 = first showing,
# 2 = repeat.
MADE_UP_SLOTS = [0, 1, 2, 3, 0, 4, 5, 2, 6, 7, 8, 9, 5, 10, 11, 3, 12, 13, 14, 7,
                 15, 16, 17, 10, 18, 19, 1, 20, 21, 22, 4, 23, 24, 25, 8, 26, 27, 6, 9, 11]

# the storage probabilities and criteria of the saved run (data/provided/continuous_rem.txt)
OPERATING_POINTS = [('standard storage', 0.30, 5.236), ('accuracy-matched storage', 0.75, 3.462)]


def made_up_template():
    """The made-up stream in the form continuous_rem.py uses: (slot of each trial, trial type)."""
    slots = np.array(MADE_UP_SLOTS, dtype=np.int32)
    trial_types = np.zeros(len(slots), dtype=np.int8)
    already_shown = set()
    for trial in range(len(slots)):
        if slots[trial] in already_shown:
            trial_types[trial] = 2
        else:
            trial_types[trial] = 1
            already_shown.add(slots[trial])
    return slots, trial_types


def likelihood_ratio(probe, trace, g, c):
    """REM's likelihood ratio of one probe against one stored trace (Shiffrin and Steyvers, 1997),
    feature by feature: a feature with nothing stored (value 0) gives 1; a stored feature that matches
    the probe gives (c + (1 - c) * p(v)) / p(v), where p(v) = g * (1 - g)^(v - 1) is how common the
    value v is; a stored feature that does not match gives 1 - c. The ratio is the product."""
    ratio = 1.0
    for feature in range(len(trace)):
        stored_value = int(trace[feature])
        if stored_value == 0:
            continue
        if stored_value == int(probe[feature]):
            how_common = g * (1.0 - g) ** (stored_value - 1)
            ratio = ratio * (c + (1.0 - c) * how_common) / how_common
        else:
            ratio = ratio * (1.0 - c)
    return ratio


def store_trace(item_features, g, u, c, generator):
    """One new trace of a presented item, with the same random draws as continuous_rem.encode():
    first one draw per feature for copying (wrong with probability 1 - c), then new geometric values
    for the wrongly copied features, then one draw per feature for storing (kept with probability u)."""
    number_of_features = len(item_features)
    copy_draws = generator.random((1, number_of_features))[0]
    wrongly_copied = []
    for feature in range(number_of_features):
        if copy_draws[feature] < (1.0 - c):
            wrongly_copied.append(feature)
    trace = [int(value) for value in item_features]
    if len(wrongly_copied) > 0:
        new_values = generator.geometric(g, len(wrongly_copied))
        for position in range(len(wrongly_copied)):
            trace[wrongly_copied[position]] = int(new_values[position])
    storage_draws = generator.random((1, number_of_features))[0]
    for feature in range(number_of_features):
        if storage_draws[feature] >= u:
            trace[feature] = 0
    return trace


def slow_session(database, template, criterion, generator, similarity, u):
    """One continuous recognition session written from the description in SI Appendix, Section 2, with
    explicit loops.
    Returns the response of each trial (True = old), the context similarity (mean similarity of the
    10 most similar items shown before, other than the item itself; NaN when fewer than 10) and the
    number of preceding trials that showed other items."""
    slots, trial_types = template
    number_of_slots = int(slots.max()) + 1
    items_of_slots = generator.choice(len(database), size=number_of_slots, replace=False)
    items_shown = [int(items_of_slots[slot]) for slot in slots]

    memory = []                      # the stored traces; traces are only ever added, never removed
    responses = []
    context = []
    preceding_other_items = []
    for trial in range(len(items_shown)):
        item = items_shown[trial]
        probe = database[item]

        # 1. compare the item on screen with EVERY trace stored on earlier trials; average the
        #    likelihood ratios; "old" when the average exceeds the criterion
        if len(memory) > 0:
            ratios = []
            for trace in memory:
                ratios.append(likelihood_ratio(probe, trace, model.G, model.C))
            average_ratio = sum(ratios) / len(ratios)
            responses.append(average_ratio > criterion)
        else:
            responses.append(False)

        # the context of this trial: similarities to the items shown before, other than itself
        similarities = []
        for earlier_trial in range(trial):
            if items_shown[earlier_trial] != item:
                similarities.append(float(similarity[item, items_shown[earlier_trial]]))
        preceding_other_items.append(len(similarities))
        if len(similarities) >= model.TOPK:
            most_similar = sorted(similarities, reverse=True)[:model.TOPK]
            context.append(sum(most_similar) / model.TOPK)
        else:
            context.append(float('nan'))

        # 2. after the response, store the item as a new trace, first showing or repeat
        memory.append(store_trace(probe, model.G, u, model.C, generator))
    return responses, context, preceding_other_items


def main():
    paths = release_paths.module_paths(__file__)
    setup_generator = np.random.default_rng(SEED)
    database = setup_generator.geometric(model.G, size=(NUMBER_OF_MADE_UP_ITEMS, model.W))
    similarity = model.build_similarity_matrix(database)
    template = made_up_template()
    print('made-up stream: %d trials, %d items, %d repeats; %d made-up items with %d features (seed %d)'
          % (len(template[0]), int(template[0].max()) + 1, int(np.sum(template[1] == 2)),
             NUMBER_OF_MADE_UP_ITEMS, model.W, SEED))

    rows = []
    all_passed = True
    for point_name, storage, criterion in OPERATING_POINTS:
        model_generator = np.random.default_rng(SEED + 1)
        slow_generator = np.random.default_rng(SEED + 1)
        _, _, model_responses, model_context, model_preceding = model.run_subject(
            database, template, criterion, model_generator, similarity, storage)
        slow_responses, slow_context, slow_preceding = slow_session(
            database, template, criterion, slow_generator, similarity, storage)

        same_responses = all(bool(model_responses[trial]) == slow_responses[trial]
                             for trial in range(len(slow_responses)))
        same_preceding = all(int(model_preceding[trial]) == slow_preceding[trial]
                             for trial in range(len(slow_preceding)))
        largest_context_difference = 0.0
        same_context_missing = True
        for trial in range(len(slow_context)):
            model_value = float(model_context[trial])
            slow_value = slow_context[trial]
            if math.isnan(model_value) or math.isnan(slow_value):
                if math.isnan(model_value) != math.isnan(slow_value):
                    same_context_missing = False
                continue
            largest_context_difference = max(largest_context_difference, abs(model_value - slow_value))
        same_context = same_context_missing and largest_context_difference < 1e-6
        # the same random numbers: the next draw of the two generators must be equal
        same_random_numbers = model_generator.random() == slow_generator.random()
        number_old = int(sum(slow_responses))

        checks = [
            ('1. each item compared with every earlier trace; "old" when the average likelihood ratio '
             'exceeds the criterion', same_responses,
             'same response on all %d trials (%d old, %d new)' % (len(slow_responses), number_old,
                                                                   len(slow_responses) - number_old)),
            ('2. every presentation (first showing or repeat) stored as a new trace, and nothing forgotten',
             same_responses and same_random_numbers,
             'same responses and the same random numbers as a memory that keeps every trace'),
            ('context = mean similarity of the 10 most similar items shown before (other than the item)',
             same_context and same_preceding,
             'largest difference %.2g; same count of preceding trials on every trial' % largest_context_difference),
        ]
        for what, passed, detail in checks:
            rows.append(dict(check=what, operating_point='%s (storage %.2f, criterion %.3f)'
                             % (point_name, storage, criterion),
                             passed='yes' if passed else 'no', detail=detail))
            all_passed = all_passed and passed
            print('%-4s %s | %s | %s' % ('ok' if passed else 'FAIL', point_name, what, detail))

    output_file = os.path.join(paths['outputs'], 'model_mechanics_check.csv')
    with open(output_file, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['check', 'operating_point', 'passed', 'detail'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print('wrote %s' % os.path.relpath(output_file, release_paths.release_root()))
    if not all_passed:
        sys.exit('check_model_mechanics.py: at least one check failed')
    print('all checks passed: the model code does what SI Appendix, Section 2 describes')


if __name__ == '__main__':
    main()
