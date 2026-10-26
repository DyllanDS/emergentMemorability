"""
check_against_paper_run.py: compares the two printed tables of this run with the saved run in
data/paper_run/ (the output the paper reports), line by line.

WHAT IT DOES
  list_designed.m and list_binned.m print one table each (both parameter cells). The saved run's
  tables are in data/paper_run/; this run writes its tables to outputs/. Every line of each file is
  compared as text. CELL 2 (the parameter values of Fig. 1) is the simulation of SI Appendix,
  Fig. S5, so the check stops with an error when any line of a CELL 2 block differs. A difference
  elsewhere (the header, CELL 1 or the closing text) is listed but does not stop the run.
  A line that differs only in the p values of the within-item correlations, "(p = ...)", does not
  stop the run either. These p values print as 0 when another toolbox on the MATLAB path replaces
  MATLAB's tcdf.m (FieldTrip's does); run_module.py starts MATLAB with restoredefaultpath to avoid
  this, and the MATLAB scripts warn when it happens. Such lines are marked "only the p values
  differ" in the csv.

READS   data/paper_run/list_designed.txt, data/paper_run/list_binned.txt (the saved run)
        outputs/list_designed.txt, outputs/list_binned.txt (this run)
WRITES  outputs/equality_with_paper_run.csv: one row per line of each file (file, line_number, cell,
        identical = yes / no / only the p values differ, paper_run_line, this_run_line)

Run (from the repository root):
    python A2d_list_context_figS5/code/check_against_paper_run.py
"""
import csv
import os
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402

paths = release_paths.module_paths(__file__)
TABLE_FILES = ['list_designed.txt', 'list_binned.txt']


def read_lines(path):
    """The lines of a text file, without line endings."""
    with open(path, encoding='utf-8') as handle:
        return handle.read().splitlines()


def without_p_values(line):
    """The line with every '(p = ...)' replaced by '(p = ...)' literally, to see whether two lines differ
    only in their p values."""
    return re.sub(r'\(p = [^)]*\)', '(p = ...)', line)


def cell_of_each_line(lines):
    """For every line, the parameter cell it belongs to: 'header' before CELL 1, 'CELL 1', 'CELL 2',
    then 'reading this' for the closing text."""
    labels = []
    current = 'header'
    for line_index, line in enumerate(lines):
        cell_match = re.match(r'^CELL (\d+):', line)
        if cell_match:
            current = 'CELL %s' % cell_match.group(1)
        elif line.startswith('READING THIS'):
            current = 'reading this'
        labels.append(current)
    # the dashed line just above a 'CELL k:' or 'READING THIS' line belongs to that block
    for line_index in range(len(lines) - 1):
        if set(lines[line_index]) == {'-'} and labels[line_index + 1] != labels[line_index]:
            labels[line_index] = labels[line_index + 1]
    return labels


rows = []
cell2_differences = 0
other_differences = 0
p_value_only_differences = 0
for table_file in TABLE_FILES:
    paper_file = os.path.join(paths['paper_run'], table_file)
    run_file = os.path.join(paths['outputs'], table_file)
    if not os.path.exists(run_file):
        sys.exit('check_against_paper_run.py: outputs/%s is missing; run code/%s first'
                 % (table_file, table_file.replace('.txt', '.m')))
    paper_lines = read_lines(paper_file)
    run_lines = read_lines(run_file)
    paper_cells = cell_of_each_line(paper_lines)
    number_of_lines = max(len(paper_lines), len(run_lines))
    identical_count = 0
    for line_index in range(number_of_lines):
        paper_line = paper_lines[line_index] if line_index < len(paper_lines) else '(no line)'
        run_line = run_lines[line_index] if line_index < len(run_lines) else '(no line)'
        cell = paper_cells[line_index] if line_index < len(paper_cells) else 'after the end'
        identical = paper_line == run_line
        only_p_values_differ = (not identical) and without_p_values(paper_line) == without_p_values(run_line)
        if identical:
            identical_count += 1
            comparison = 'yes'
        elif only_p_values_differ:
            p_value_only_differences += 1
            comparison = 'only the p values differ'
            print('p values differ (%s, line %d, %s; within-item correlations):\n  saved run: %s\n  this run:  %s'
                  % (table_file, line_index + 1, cell, paper_line, run_line))
        elif cell == 'CELL 2':
            comparison = 'no'
            cell2_differences += 1
            print('DIFFERENT (%s, line %d, CELL 2):\n  saved run: %s\n  this run:  %s'
                  % (table_file, line_index + 1, paper_line, run_line))
        else:
            comparison = 'no'
            other_differences += 1
            print('different (%s, line %d, %s):\n  saved run: %s\n  this run:  %s'
                  % (table_file, line_index + 1, cell, paper_line, run_line))
        rows.append({'file': table_file, 'line_number': line_index + 1, 'cell': cell,
                     'identical': comparison, 'paper_run_line': paper_line, 'this_run_line': run_line})
    print('%s: %d of %d lines identical to the saved run' % (table_file, identical_count, number_of_lines))

output_file = os.path.join(paths['outputs'], 'equality_with_paper_run.csv')
with open(output_file, 'w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=['file', 'line_number', 'cell', 'identical', 'paper_run_line',
                                                'this_run_line'])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
print('wrote outputs/equality_with_paper_run.csv')

if p_value_only_differences:
    print('%d line(s) differ only in the p values of the within-item correlations; '
          'was MATLAB started with restoredefaultpath?' % p_value_only_differences)
if other_differences:
    print('%d line(s) outside CELL 2 differ; see outputs/equality_with_paper_run.csv' % other_differences)
if cell2_differences:
    sys.exit('check_against_paper_run.py: %d line(s) of CELL 2 differ from the saved run' % cell2_differences)
if p_value_only_differences:
    print('CELL 2 of both tables equals the saved run line by line, apart from the p values above')
else:
    print('CELL 2 of both tables is identical to the saved run, line by line')
