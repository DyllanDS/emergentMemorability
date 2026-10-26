"""
check_against_paper_run.py: compare this module's run in outputs/ with the saved run in
data/paper_run/, the output the paper reports.

The saved run holds the outputs of the same three scripts (01_core.py, 02_sweep.py and
03_bylinskii.py). Every number of each pair of files is compared, and a difference larger than
TOLERANCE (1e-9) is reported. Text reports are compared line by line after removing carriage
returns.

Pairs (each one is checked only when the file in outputs/ exists):
  outputs/core.json      against data/paper_run/core.json    every value of the five model variants
  outputs/core.txt       against data/paper_run/core.txt     the text report, line by line
  outputs/sweep.csv      against data/paper_run/sweep.csv    every setting of the parameter sweep
  outputs/sweep.txt      against data/paper_run/sweep.txt
  outputs/bylinskii.csv  against data/paper_run/bylinskii.csv  every setting of the Bylinskii sweep
  outputs/bylinskii.txt  against data/paper_run/bylinskii.txt
  outputs/bylinskii_reference_cells.csv  the three settings run by bylinskii_reference_cells.py,
                         against their rows of data/paper_run/bylinskii.csv
  outputs/bylinskii_calibration.csv  the across-category d′ of each calibrated model, against the
                         'Width calibration' line of data/paper_run/bylinskii.txt (printed to 2 decimals)

Output: outputs/check_against_paper_run.csv (pair, item, quantity, rerun_value, paper_run_value,
        abs_difference, within_tolerance), and a summary on screen.
Run from the repository root:  python A2f_hybrid_exemplar_model_figS7/code/check_against_paper_run.py
"""
import sys
sys.dont_write_bytecode = True
import json
import math
import os
import re

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths

PATHS = release_paths.module_paths(__file__)
TOLERANCE = 1e-9
ROWS = []


def record(pair, item, quantity, rerun_value, paper_value, tolerance=TOLERANCE):
    """Add one compared value to the table; numbers are compared within the tolerance, text exactly."""
    if isinstance(rerun_value, (int, float)) and isinstance(paper_value, (int, float)) \
            and not isinstance(rerun_value, bool) and not isinstance(paper_value, bool):
        if math.isnan(rerun_value) and math.isnan(paper_value):
            difference = 0.0
        else:
            difference = abs(float(rerun_value) - float(paper_value))
        within = difference <= tolerance
    else:
        difference = ''
        within = str(rerun_value) == str(paper_value)
    ROWS.append(dict(pair=pair, item=item, quantity=quantity, rerun_value=rerun_value,
                     paper_run_value=paper_value, abs_difference=difference, within_tolerance=within))


def compare_json(pair, rerun, paper, path=''):
    """Walk two JSON structures side by side and record every leaf value."""
    if isinstance(paper, dict):
        for key in paper:
            if key not in rerun:
                record(pair, path, key, 'missing', 'present')
                continue
            compare_json(pair, rerun[key], paper[key], path + '/' + str(key))
        for key in rerun:
            if key not in paper:
                record(pair, path, key, 'present', 'missing')
    elif isinstance(paper, list):
        if len(rerun) != len(paper):
            record(pair, path, 'length', len(rerun), len(paper))
        for index in range(min(len(rerun), len(paper))):
            compare_json(pair, rerun[index], paper[index], path + '/' + str(index))
    else:
        item, _, quantity = path.rpartition('/')
        record(pair, item, quantity, rerun, paper)


def compare_text(pair, rerun_file, paper_file):
    """Line-by-line comparison; carriage returns are removed first."""
    with open(rerun_file, encoding='utf-8') as handle:
        rerun_lines = handle.read().replace('\r', '').split('\n')
    with open(paper_file, encoding='utf-8') as handle:
        paper_lines = handle.read().replace('\r', '').split('\n')
    if len(rerun_lines) != len(paper_lines):
        record(pair, 'whole file', 'number of lines', len(rerun_lines), len(paper_lines))
    different = 0
    for line_number in range(min(len(rerun_lines), len(paper_lines))):
        if rerun_lines[line_number] != paper_lines[line_number]:
            different += 1
            record(pair, 'line %d' % (line_number + 1), 'text', rerun_lines[line_number], paper_lines[line_number])
    record(pair, 'whole file', 'lines that differ', different, 0, tolerance=0)


def compare_table(pair, rerun_table, paper_table, key_column):
    """Row by row (matched on key_column), column by column."""
    paper_by_key = {row[key_column]: row for _, row in paper_table.iterrows()}
    for _, rerun_row in rerun_table.iterrows():
        key = rerun_row[key_column]
        if key not in paper_by_key:
            record(pair, key, 'row', 'present', 'missing')
            continue
        paper_row = paper_by_key[key]
        for column in paper_table.columns:
            if column == key_column:
                continue
            rerun_value = rerun_row[column]
            paper_value = paper_row[column]
            if isinstance(paper_value, (int, float)) and not isinstance(paper_value, bool):
                record(pair, key, column, float(rerun_value), float(paper_value))
            else:
                record(pair, key, column, str(rerun_value), str(paper_value))


def read_table(path):
    return pd.read_csv(path, float_precision='round_trip')


def main():
    outputs = PATHS['outputs']
    paper_run = PATHS['paper_run']
    checked = []

    rerun_file = os.path.join(outputs, 'core.json')
    if os.path.exists(rerun_file):
        with open(rerun_file, encoding='utf-8') as handle:
            rerun = json.load(handle)
        with open(os.path.join(paper_run, 'core.json'), encoding='utf-8') as handle:
            paper = json.load(handle)
        compare_json('core.json', rerun, paper)
        compare_text('core.txt', os.path.join(outputs, 'core.txt'), os.path.join(paper_run, 'core.txt'))
        checked.append('core')

    for name in ('sweep', 'bylinskii'):
        rerun_file = os.path.join(outputs, name + '.csv')
        if os.path.exists(rerun_file):
            compare_table(name + '.csv', read_table(rerun_file), read_table(os.path.join(paper_run, name + '.csv')), 'cell')
            compare_text(name + '.txt', os.path.join(outputs, name + '.txt'), os.path.join(paper_run, name + '.txt'))
            checked.append(name)

    rerun_file = os.path.join(outputs, 'bylinskii_reference_cells.csv')
    if os.path.exists(rerun_file):
        compare_table('bylinskii_reference_cells.csv', read_table(rerun_file),
                      read_table(os.path.join(paper_run, 'bylinskii.csv')), 'cell')
        checked.append('bylinskii_reference_cells')

    rerun_file = os.path.join(outputs, 'bylinskii_calibration.csv')
    if os.path.exists(rerun_file):
        # the saved run prints the calibrated d′ to 2 decimals on its 'Width calibration' line
        with open(os.path.join(paper_run, 'bylinskii.txt'), encoding='utf-8') as handle:
            text = handle.read().replace('\r', '')
        line = re.search(r'^Width calibration.*?:\s*(.*)$', text, re.M).group(1)
        printed = {name: float(value) for name, value in re.findall(r'(\w+) (-?\d+\.\d+)', line)}
        calibration = read_table(rerun_file)
        for _, row in calibration.iterrows():
            record('bylinskii_calibration.csv', row['model'], 'across-category d′ (printed to 2 decimals)',
                   round(float(row['across_category_dprime']), 2), printed[row['model']], tolerance=1e-12)
        checked.append('bylinskii_calibration')

    table = pd.DataFrame(ROWS, columns=['pair', 'item', 'quantity', 'rerun_value', 'paper_run_value',
                                       'abs_difference', 'within_tolerance'])
    table.to_csv(os.path.join(outputs, 'check_against_paper_run.csv'), index=False)

    print('compared with the saved run in data/paper_run/, tolerance %g:' % TOLERANCE)
    if not checked:
        print('  nothing to compare: no output of this module in outputs/ yet')
    for pair in table['pair'].unique():
        rows = table[table['pair'] == pair]
        numeric = pd.to_numeric(rows['abs_difference'], errors='coerce')
        largest = numeric.max() if numeric.notna().any() else float('nan')
        failed = int((~rows['within_tolerance'].astype(bool)).sum())
        print('  %-32s %5d values, %d outside the tolerance, largest numeric difference %.3g'
              % (pair, len(rows), failed, largest))
    failures = table[~table['within_tolerance'].astype(bool)]
    if len(failures):
        print('DIFFERENCES FOUND (%d); see outputs/check_against_paper_run.csv' % len(failures))
        print(failures.head(20).to_string(index=False))
    elif checked:
        print('ALL EQUAL within the tolerance')


if __name__ == '__main__':
    main()
