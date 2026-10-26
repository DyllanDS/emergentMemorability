"""
collect_all_tests.py: the statistics tables of the release in one place.

What it does
  Every module writes <module>/outputs/tests.csv: for each statistical test or estimate of the paper
  (the test ids listed in statistics/README.md) its statistic, n, degrees of freedom, p, effect size and
  95% interval, with the method, the resampling unit, the number of resamples and the seed. This script

  1. joins them into statistics/all_tests.csv (sorted by test id, with T19b after T19; within a test, in
     the order the module wrote them), checks that every test id has at least one row, and writes every
     script path from the repository root;
  2. builds statistics/participant_flow.csv (test T54) from the participant-flow tables of modules 03
     (inversion experiment), 04 (face study) and 06 (image-set experiments).

Inputs
  <module>/outputs/tests.csv (24 modules), and 06_image_set_experiments_fig6_fig7/outputs/tests_shared_concepts.csv (T19b)
  03_inversion_experiment_fig3/outputs/participant_flow.csv, 04_rare_feature_faces_fig4/outputs/
  participant_flow.csv, 06_image_set_experiments_fig6_fig7/outputs/participant_flow.csv
Outputs
  statistics/all_tests.csv, statistics/participant_flow.csv

Values are copied as text, exactly as the modules wrote them (some p values are written below the
smallest number a computer stores, e.g. 5.81097e-682, and must not become 0).

Run (from the repository root, or from anywhere):
    python statistics/collect_all_tests.py
Python 3 standard library only. Runtime: about a second. The exit code is 1 when a check fails.
"""
import csv
import os
import re
import sys

# No __pycache__ folders in the release.
sys.dont_write_bytecode = True

STATISTICS_FOLDER = os.path.dirname(os.path.abspath(__file__))
RELEASE_ROOT = os.path.dirname(STATISTICS_FOLDER)
sys.path.insert(0, os.path.join(RELEASE_ROOT, 'shared', 'python'))
import release_paths  # noqa: E402  (the list of module folders)

TEST_COLUMNS = ['test_id', 'ledger_ids', 'location', 'test', 'statistic_name', 'statistic', 'n', 'unit', 'df', 'p',
                'p_type', 'effect_size_name', 'effect_size', 'ci_low', 'ci_high', 'ci_method', 'resampling_unit',
                'n_resamples', 'seed', 'script']
# The test ids: T01 to T63 (there is no T23), with T19b (the Fig. 7d reduction over the concepts in both image
# sets, module 06) right after T19
ALL_TEST_IDS = (['T%02d' % number for number in range(1, 20)] + ['T19b']
                + ['T%02d' % number for number in range(20, 64) if number != 23])

FLOW_MODULES = ['03', '04', '06']
FLOW_COLUMNS = ['experiment', 'image_set', 'sessions_before_exclusion',
                'excluded_same_key', 'excluded_fast', 'excluded_slow', 'excluded_low_cr',
                'excluded_more_than_5_false_presses', 'excluded_fewer_than_4_of_6_repeats_detected',
                'excluded_both_parts_of_the_rule', 'excluded_not_36_test_trials', 'excluded_more_than_one_rule',
                'excluded_total', 'removed_by_cap', 'passed_rule', 'passing_sessions_not_needed', 'kept', 'rules',
                'source']
# Module column name -> participant_flow.csv column name (modules 03 and 06 name this column differently).
FLOW_RENAMES = {'processed': 'sessions_before_exclusion'}


def read_csv_rows(path):
    """All rows of a CSV file as dicts of text (no number conversion: every value stays as written)."""
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(row)
        return reader.fieldnames, rows


def write_csv_rows(path, columns, rows):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def release_relative_script(script_text, module_folder):
    """'code/x.py' (written from the module folder) -> '<module>/code/x.py' (from the repository root)."""
    return re.sub(r'(?<![\w/])code/', module_folder + '/code/', script_text)


def test_number(test_id):
    """The place of a test id in ALL_TEST_IDS (for sorting): T19b comes after T19 and before T20."""
    return ALL_TEST_IDS.index(test_id)


# ---------------------------------------------------------------------------------------------------------
# 1. all_tests.csv
# ---------------------------------------------------------------------------------------------------------
def collect_tests(problems):
    collected = []          # (test number, module order, row order, row)
    modules_without_tests = []
    for module_order, (module_id, module_folder) in enumerate(release_paths.MODULE_FOLDERS.items()):
        tests_file = os.path.join(RELEASE_ROOT, module_folder, 'outputs', 'tests.csv')
        if not os.path.exists(tests_file):
            modules_without_tests.append(module_folder)
            continue
        columns, rows = read_csv_rows(tests_file)
        if columns != TEST_COLUMNS:
            problems.append('%s/outputs/tests.csv has the columns %s' % (module_folder, columns))
            continue
        # Module 06 also writes outputs/tests_shared_concepts.csv (T19b; code/shared_concepts_test.py)
        shared_concepts_file = os.path.join(RELEASE_ROOT, module_folder, 'outputs', 'tests_shared_concepts.csv')
        if os.path.exists(shared_concepts_file):
            extra_columns, extra_rows = read_csv_rows(shared_concepts_file)
            if extra_columns != TEST_COLUMNS:
                problems.append('%s/outputs/tests_shared_concepts.csv has the columns %s' % (module_folder, extra_columns))
            else:
                rows = rows + extra_rows
        for row_order, row in enumerate(rows):
            if row['test_id'] not in ALL_TEST_IDS:
                problems.append('%s/outputs/tests.csv: unknown test id %r' % (module_folder, row['test_id']))
                continue
            row['script'] = release_relative_script(row['script'], module_folder)
            collected.append((test_number(row['test_id']), module_order, row_order, row))
    collected.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    all_rows = [entry[3] for entry in collected]
    return all_rows, modules_without_tests


# ---------------------------------------------------------------------------------------------------------
# 2. participant_flow.csv
# ---------------------------------------------------------------------------------------------------------
def collect_participant_flow(problems):
    flow_rows = []
    for module_id in FLOW_MODULES:
        module_folder = release_paths.MODULE_FOLDERS[module_id]
        flow_file = os.path.join(RELEASE_ROOT, module_folder, 'outputs', 'participant_flow.csv')
        if not os.path.exists(flow_file):
            problems.append('%s/outputs/participant_flow.csv is missing (run the module)' % module_folder)
            continue
        columns, rows = read_csv_rows(flow_file)
        for row in rows:
            flow_row = {column: '' for column in FLOW_COLUMNS}
            for column, value in row.items():
                new_name = FLOW_RENAMES.get(column, column)
                if new_name not in flow_row:
                    problems.append('%s participant_flow.csv: column %s has no place in statistics/'
                                    'participant_flow.csv' % (module_folder, column))
                    continue
                flow_row[new_name] = value
            flow_row['source'] = module_folder + '/outputs/participant_flow.csv'
            flow_rows.append(flow_row)
    return flow_rows


# ---------------------------------------------------------------------------------------------------------
def main():
    problems = []

    all_rows, modules_without_tests = collect_tests(problems)
    write_csv_rows(os.path.join(STATISTICS_FOLDER, 'all_tests.csv'), TEST_COLUMNS, all_rows)

    rows_per_test = {}
    for row in all_rows:
        rows_per_test[row['test_id']] = rows_per_test.get(row['test_id'], 0) + 1
    missing_ids = [test_id for test_id in ALL_TEST_IDS if test_id not in rows_per_test]

    flow_rows = collect_participant_flow(problems)
    write_csv_rows(os.path.join(STATISTICS_FOLDER, 'participant_flow.csv'), FLOW_COLUMNS, flow_rows)

    # T54 (the participant flow) has rows from each of the three participant-flow modules.
    t54_scripts = set(row['script'].split('/')[0] for row in all_rows if row['test_id'] == 'T54')
    for module_id in FLOW_MODULES:
        if release_paths.MODULE_FOLDERS[module_id] not in t54_scripts:
            problems.append('T54 has no row from module %s' % module_id)

    # Rows without an interval because the interval needs an input that is not in the release; their
    # ci_method names the input (EXTERNAL_INPUTS.md).
    pending_rows = [row for row in all_rows if 'EXTERNAL_INPUTS.md' in row['ci_method']]

    print('collect_all_tests.py: statistics/all_tests.csv, %d rows, %d of %d test ids'
          % (len(all_rows), len(rows_per_test), len(ALL_TEST_IDS)))
    print('  rows per test: ' + ', '.join('%s %d' % (test_id, rows_per_test[test_id])
                                           for test_id in ALL_TEST_IDS if test_id in rows_per_test))
    print('  rows whose interval needs an external input (EXTERNAL_INPUTS.md): %d' % len(pending_rows))
    print('statistics/participant_flow.csv: %d rows' % len(flow_rows))
    if modules_without_tests:
        print('Modules without outputs/tests.csv (not run yet): %s' % ', '.join(modules_without_tests))
    if missing_ids:
        problems.append('test ids with no row: %s' % ', '.join(missing_ids))
    if problems:
        print('Problems:')
        for problem in problems:
            print('  ' + problem)
        sys.exit(1)


if __name__ == '__main__':
    main()
