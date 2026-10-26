"""
make_tableS1.py: SI Appendix, Table S1, the image-set results with and without the participant
exclusions (SI Appendix, Section 3).

WHAT THE TABLE SHOWS
Each row is one result of the three image-set experiments, computed on every participant who was
run (602; "Without exclusions") and after the exclusions of the Methods (450, 150 per set; "With
exclusions").
  Concept level (maintained, changed and reduced sets): Spearman rho, across concepts, between
    memorability (hit rate minus false alarm rate) and the mean summed similarity of the tested
    concept to the 40 items studied in its block, with its 95% interval over 10,000 concept
    resamples (T35).
  Feature level (maintained and changed sets; Fig. 6d and 6g): the correlation across the 49 SPoSE
    features between each feature's correlation with summed similarity and its correlation with
    memorability, with its 95% interval over 10,000 participant resamples (T36).
  Variance explained (maintained and reduced sets; Fig. 7d): the sum over the 49 features of the
    squared correlation with memorability, the mean and 95% interval over 1,000 participant
    resamples (seed 7; T37).
  Reduction, maintained to reduced: the mean percent reduction over the same resamples and its
    one-sided p (T37).

READS   outputs/tests.csv (written by code/stats_tableS1.py)
WRITES  outputs/tableS1.csv, one row per result. The columns without_exclusions and with_exclusions
        hold the cells as printed (the estimate and its 95% interval to two decimals; for the
        reduction, the percent and the one-sided p); the other columns hold the same values to six
        significant digits (for the reduction, the percent in the estimate columns).

RUN (from the repository root; seconds):
    python A3d_exclusions_tableS1/code/make_tableS1.py
"""
import csv
import os
import sys

sys.dont_write_bytecode = True   # no __pycache__ folder in shared/ when this step is run on its own
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (first: it stops Python from writing __pycache__ folders)

PATHS = release_paths.module_paths(__file__, make_output_folders=False)
TESTS_FILE = os.path.join(PATHS['outputs'], 'tests.csv')
TABLE_FILE = os.path.join(PATHS['outputs'], 'tableS1.csv')

# The two columns of Table S1 (prefix of the output columns, and the words that follow the row's name at
# the start of the test text that code/stats_tableS1.py writes to outputs/tests.csv)
TABLE_COLUMNS = [('without', 'without exclusions'), ('with', 'with exclusions')]

# The rows of Table S1: the label as printed, the test id, the start of the test text in outputs/tests.csv
# and the kind of cell (an estimate with its interval, or a percent with its one-sided p)
TABLE_ROWS = [
    ('Concept level, maintained set (\u03c1)', 'T35', 'Table S1, concept level, maintained set', 'interval'),
    ('Concept level, changed set (\u03c1)', 'T35', 'Table S1, concept level, changed set', 'interval'),
    ('Concept level, reduced set (\u03c1)', 'T35', 'Table S1, concept level, reduced set', 'interval'),
    ('Feature level, maintained set (r)', 'T36', 'Table S1, feature level, maintained set', 'interval'),
    ('Feature level, changed set (r)', 'T36', 'Table S1, feature level, changed set', 'interval'),
    ('Variance explained, maintained set', 'T37', 'Table S1, variance explained, maintained set', 'interval'),
    ('Variance explained, reduced set', 'T37', 'Table S1, variance explained, reduced set', 'interval'),
    ('Reduction, maintained to reduced', 'T37', 'Table S1, reduction', 'percent_and_p'),
]
OUTPUT_COLUMNS = ['result', 'without_exclusions', 'with_exclusions',
                  'without_estimate', 'without_ci_low', 'without_ci_high', 'without_one_sided_p',
                  'with_estimate', 'with_ci_low', 'with_ci_high', 'with_one_sided_p']


def tests_row(tests, test_id, text_start):
    """The one row of outputs/tests.csv with this test id whose test text starts with text_start."""
    found = [row for row in tests if row['test_id'] == test_id and row['test'].startswith(text_start)]
    if len(found) != 1:
        sys.exit('make_tableS1.py: %d rows of outputs/tests.csv start with "%s", expected 1; run '
                 'code/stats_tableS1.py first' % (len(found), text_start))
    return found[0]


def main():
    if not os.path.exists(TESTS_FILE):
        sys.exit('make_tableS1.py: outputs/tests.csv is missing; run code/stats_tableS1.py first')
    with open(TESTS_FILE, newline='', encoding='utf-8') as handle:
        tests = list(csv.DictReader(handle))

    table_rows = []
    for label, test_id, text_start, cell_kind in TABLE_ROWS:
        table_row = {'result': label}
        for prefix, column_words in TABLE_COLUMNS:
            row = tests_row(tests, test_id, '%s, %s' % (text_start, column_words))
            table_row[prefix + '_estimate'] = row['statistic']
            if cell_kind == 'interval':
                table_row[prefix + '_ci_low'] = row['ci_low']
                table_row[prefix + '_ci_high'] = row['ci_high']
                table_row[prefix + '_one_sided_p'] = ''
                table_row[prefix + '_exclusions'] = '%.2f [%.2f, %.2f]' % (
                    float(row['statistic']), float(row['ci_low']), float(row['ci_high']))
            else:
                table_row[prefix + '_ci_low'] = ''
                table_row[prefix + '_ci_high'] = ''
                table_row[prefix + '_one_sided_p'] = row['p']
                table_row[prefix + '_exclusions'] = '%.0f%% (one-sided p = %.2f)' % (
                    float(row['statistic']), float(row['p']))
        table_rows.append(table_row)

    with open(TABLE_FILE, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(table_rows)

    print('SI Appendix, Table S1. Image-set results with and without the exclusions')
    print('%-36s %-26s %-26s' % ('Result', 'Without exclusions', 'With exclusions'))
    for table_row in table_rows:
        print('%-36s %-26s %-26s' % (table_row['result'].replace('\u03c1', 'rho'), table_row['without_exclusions'],
                                     table_row['with_exclusions']))
    print('wrote', os.path.join('A3d_exclusions_tableS1', 'outputs', 'tableS1.csv'))


if __name__ == '__main__':
    main()
