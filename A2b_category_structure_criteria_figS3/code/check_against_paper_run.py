"""
check_against_paper_run.py: compares this run of the 18 simulations (code/bylinskii_robustness.m) with the
saved run in data/paper_run/, the output the paper reports, and writes the module's outputs/numbers.csv (one
row per number of the paper this module reproduces).

What it checks
--------------
1. outputs/bylinskii_robustness.csv against data/paper_run/bylinskii_robustness.csv: the 18 cells in the
   same order with the same labels and settings, and every numeric column, as numbers (difference allowed:
   1e-6; for the tiny p values of the pRho column, 1e-6 of the value). It also counts the values that are
   identical to the last written digit (MATLAB's writetable writes 15 significant digits; the saved table
   has CRLF line endings, which are ignored).
2. The per-target file outputs/bylinskii_robustness_items.csv: every cell's rho, p and summed-similarity
   correlation recomputed from it (scipy) and compared with the table (1e-12; p 1e-6 of the value), and
   every cell's number of targets (categories x targets per category, all tested in both contexts).
Details go to outputs/equality_with_paper_run.csv (one row per cell and column, plus the per-target
checks).

Then it writes outputs/numbers.csv for the ledger items of this module (L6-036 to L6-044 and L6-046)
with the values of this run (mode rerun).

Options
    --paper-run   write numbers.csv from the saved run in data/paper_run/ (mode saved_run), for a
                  machine without MATLAB; the comparison is skipped

Run (from the repository root):
    python A2b_category_structure_criteria_figS3/code/check_against_paper_run.py
"""
import argparse
import csv
import os
import re
import sys

sys.dont_write_bytecode = True   # write no __pycache__ folder into shared/python (set before the import below)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402
import numbers_table  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robustness_tables  # noqa: E402  (this module's readers of the grid's tables)

SCRIPT = 'code/check_against_paper_run.py'
TOLERANCE = 1e-6

parser = argparse.ArgumentParser(description='Module A2b: comparison with the saved run, and numbers.csv.')
parser.add_argument('--paper-run', action='store_true',
                    help='write numbers.csv from the saved run (no MATLAB run needed); skips the comparison')
arguments = parser.parse_args()
paths = release_paths.module_paths(__file__)

this_run_table = os.path.join(paths['outputs'], 'bylinskii_robustness.csv')
this_run_items = os.path.join(paths['outputs'], 'bylinskii_robustness_items.csv')
paper_run_table = os.path.join(paths['paper_run'], 'bylinskii_robustness.csv')


def read_raw_rows(path):
    """The rows of a table as text, exactly as written (for the 'identical to the last digit' count)."""
    with open(path, newline='') as handle:
        return [row for row in csv.DictReader(handle)]


# ============================================================================ 1 and 2. the comparison
equality_rows = []
all_within_tolerance = True
if arguments.paper_run:
    print('--paper-run: numbers.csv from the saved run (data/paper_run/bylinskii_robustness.csv); '
          'comparison skipped')
    cells = robustness_tables.read_cells(paper_run_table)
    mode = 'saved_run'
    source_text = 'the saved run, data/paper_run/bylinskii_robustness.csv'
else:
    for needed_file in (this_run_table, this_run_items):
        if not os.path.exists(needed_file):
            sys.exit('check_against_paper_run.py: outputs/%s is missing; run code/bylinskii_robustness.m first '
                     '(python A2b_category_structure_criteria_figS3/code/run_module.py does it), or pass '
                     '--paper-run to write numbers.csv from the saved run' % os.path.basename(needed_file))
    cells = robustness_tables.read_cells(this_run_table)
    paper_cells = robustness_tables.read_cells(paper_run_table)
    mode = 'rerun'
    source_text = 'this run, outputs/bylinskii_robustness.csv'

    # 1. the table, cell by cell and column by column
    print('1. outputs/bylinskii_robustness.csv vs data/paper_run/bylinskii_robustness.csv')
    assert len(cells) == len(paper_cells) == 18, 'expected 18 cells in both tables'
    raw_this_run = read_raw_rows(this_run_table)
    raw_paper_run = read_raw_rows(paper_run_table)
    identical_text_count = 0
    value_count = 0
    largest_difference = 0.0
    for cell_index in range(18):
        cell_number = cell_index + 1
        this_cell = cells[cell_index]
        paper_cell = paper_cells[cell_index]
        same_label = this_cell['label'] == paper_cell['label']
        equality_rows.append({'check': 'table', 'cell_number': cell_number, 'label': paper_cell['label'],
                              'column': 'label', 'this_run': this_cell['label'], 'paper_run': paper_cell['label'],
                              'absolute_difference': '', 'relative_difference': '', 'tolerance': 'identical text',
                              'within_tolerance': 'yes' if same_label else 'no',
                              'identical_text': 'yes' if same_label else 'no'})
        if not same_label:
            all_within_tolerance = False
        for column in robustness_tables.NUMERIC_COLUMNS:
            this_value = this_cell[column]
            paper_value = paper_cell[column]
            absolute_difference = abs(this_value - paper_value)
            if paper_value != 0:
                relative_difference = absolute_difference / abs(paper_value)
            else:
                relative_difference = 0.0 if absolute_difference == 0 else float('inf')
            if column == 'pRho':
                within = relative_difference <= TOLERANCE
                tolerance_text = '%g of the value' % TOLERANCE
            else:
                within = absolute_difference <= TOLERANCE
                tolerance_text = '%g' % TOLERANCE
            identical_text = raw_this_run[cell_index][column] == raw_paper_run[cell_index][column]
            value_count += 1
            if identical_text:
                identical_text_count += 1
            if column != 'pRho':
                largest_difference = max(largest_difference, absolute_difference)
            if not within:
                all_within_tolerance = False
                print('   DIFFERS: cell %d "%s" %s: %r vs %r' % (cell_number, paper_cell['label'], column,
                                                                 this_value, paper_value))
            equality_rows.append({'check': 'table', 'cell_number': cell_number, 'label': paper_cell['label'],
                                  'column': column, 'this_run': repr(this_value), 'paper_run': repr(paper_value),
                                  'absolute_difference': '%.3g' % absolute_difference,
                                  'relative_difference': '%.3g' % relative_difference,
                                  'tolerance': tolerance_text, 'within_tolerance': 'yes' if within else 'no',
                                  'identical_text': 'yes' if identical_text else 'no'})
    not_identical = [row for row in equality_rows if row['check'] == 'table' and row['identical_text'] == 'no']
    print('   %d of %d values identical to the last written digit (15 significant digits); largest difference '
          'of the others %.2g; all within tolerance: %s' % (identical_text_count, value_count, largest_difference,
                                                             all_within_tolerance))
    for row in not_identical:
        print('   not identical in the last digit: cell %s "%s" %s %s vs %s'
              % (row['cell_number'], row['label'], row['column'], row['this_run'], row['paper_run']))

    # 2. the per-target file gives back the table
    print('2. outputs/bylinskii_robustness_items.csv gives back every cell of the table')
    items_by_cell = robustness_tables.read_items(this_run_items)
    for cell_index, cell in enumerate(cells):
        cell_number = cell_index + 1
        items = items_by_cell[cell_number]
        expected_targets = int(cell['nCat']) * robustness_tables.targets_per_category(cell)
        rho, rho_p = robustness_tables.spearman_rho_and_p(items['memorability_within'], items['memorability_across'])
        similarity_r, similarity_p = robustness_tables.pearson_r_and_p(items['summed_similarity_within'],
                                                                       items['summed_similarity_across'])
        checks = [('targets', items['n_targets'], expected_targets, items['n_targets'] == expected_targets),
                  ('targets left out', items['n_left_out'], 0, items['n_left_out'] == 0),
                  ('rho', rho, cell['rho'], abs(rho - cell['rho']) < 1e-12),
                  ('pRho', rho_p, cell['pRho'], abs(rho_p / cell['pRho'] - 1) < 1e-6),
                  ('simCorr', similarity_r, cell['simCorr'], abs(similarity_r - cell['simCorr']) < 1e-12)]
        for name, recomputed, in_table, passed in checks:
            if not passed:
                all_within_tolerance = False
                print('   DIFFERS: cell %d %s recomputed %r, table %r' % (cell_number, name, recomputed, in_table))
            equality_rows.append({'check': 'per-target file', 'cell_number': cell_number, 'label': cell['label'],
                                  'column': name, 'this_run': repr(recomputed), 'paper_run': repr(in_table),
                                  'absolute_difference': '%.3g' % abs(recomputed - in_table),
                                  'relative_difference': '', 'tolerance': '1e-12 (p: 1e-6 of the value; counts '
                                                                          'exact)',
                                  'within_tolerance': 'yes' if passed else 'no', 'identical_text': ''})
    print('   targets per cell: %s' % ', '.join(str(items_by_cell[number]['n_targets']) for number in range(1, 19)))

    equality_file = os.path.join(paths['outputs'], 'equality_with_paper_run.csv')
    with open(equality_file, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['check', 'cell_number', 'label', 'column', 'this_run', 'paper_run',
                                                    'absolute_difference', 'relative_difference', 'tolerance',
                                                    'within_tolerance', 'identical_text'])
        writer.writeheader()
        for row in equality_rows:
            writer.writerow(row)
    print('wrote outputs/equality_with_paper_run.csv; everything within tolerance: %s' % all_within_tolerance)
    if not all_within_tolerance:
        print('WARNING: this run differs from the saved run in data/paper_run/ beyond the tolerance; the paper '
              'reports the values of data/paper_run/bylinskii_robustness.csv. numbers.csv below holds this '
              'run\'s values.')

if arguments.paper_run:
    equality_note = ''
elif all_within_tolerance:
    equality_note = (' The table equals the saved run (data/paper_run/bylinskii_robustness.csv) within 1e-6 in '
                     'every column (outputs/equality_with_paper_run.csv).')
else:
    equality_note = (' The table is not within 1e-6 of the saved run (data/paper_run/bylinskii_robustness.csv) '
                     'in every column (outputs/equality_with_paper_run.csv).')

# ============================================================================ the numbers of the paper
assert len(cells) == 18
all_rho = [cell['rho'] for cell in cells]
all_p = [cell['pRho'] for cell in cells]
all_similarity_r = [cell['simCorr'] for cell in cells]
equal_cells = [cell for cell in cells if robustness_tables.has_equal_criteria(cell)]
equal_rho = [cell['rho'] for cell in equal_cells]
fig2a_cells = [cell for cell in cells if robustness_tables.has_fig2a_structure(cell)]
lowest = min(cells, key=lambda cell: cell['rho'])
highest = max(cells, key=lambda cell: cell['rho'])
lowest_similarity = min(cells, key=lambda cell: cell['simCorr'])
highest_similarity = max(cells, key=lambda cell: cell['simCorr'])


def criteria_text(cell):
    return '%g/%g' % (cell['critWithin'], cell['critAcross'])


# the design of the grid, read back from the table and from the script
criteria_cells = [cell for cell in cells if cell['label'].startswith('reference') or cell['label'].startswith('crit ')]
criteria_list = []
for cell in criteria_cells:
    pair = criteria_text(cell)
    if pair not in criteria_list:
        criteria_list.append(pair)
criteria_parts = []
for pair in criteria_list:
    repeats = sum(1 for cell in criteria_cells if criteria_text(cell) == pair)
    criteria_parts.append(pair + (' (x%d)' % repeats if repeats > 1 else ''))
feature_values = [int(cell['catFeatures']) for cell in cells if cell['label'].startswith('category-defining')]
category_values = [int(cell['nCat']) for cell in cells if cell['label'].startswith('categories')]
items_per_category_values = [int(cell['nPerCat']) for cell in cells if cell['label'].startswith('items per category')]

grid_source = open(os.path.join(paths['code'], 'bylinskii_robustness.m')).read()
reference_text = re.search(r"ref = struct\((.*?)\);", grid_source, re.S).group(1)
reference = {}
for name, value in re.findall(r"'(\w+)',\s*([0-9.]+)", reference_text):
    reference[name] = float(value)
seed = int(re.search(r"rng\((\d+), 'twister'\)", grid_source).group(1))
design_text = ('criteria within/across %s; category-defining features %s; categories %s; items per category %s; '
               'reference: g %.2f, w %d, u %.2f, c %.2f, %d categories x %d items, %d targets per category, '
               '%d category features, %d studied targets + %d fillers, %d within-category and {:,} across-category '
               'runs, rng(%d)'
               % (', '.join(criteria_parts), '/'.join(str(v) for v in feature_values),
                  '/'.join(str(v) for v in category_values), '/'.join(str(v) for v in items_per_category_values),
                  reference['g'], reference['w'], reference['u'], reference['c'], reference['nCat'],
                  reference['nPerCat'], reference['nTargetsPerCat'], reference['nCatFeatures'],
                  reference['nStudyTargets'], reference['nStudyFillers'], reference['nRunsWithin'], seed)
               ).format(int(reference['nRunsAcross']))

# the figure, as drawn by make_figureS3.py (outputs/figure_data/FigS3.csv)
figure_data_file = os.path.join(paths['figure_data'], 'FigS3%s.csv' % ('_paper_run' if arguments.paper_run else ''))
if os.path.exists(figure_data_file):
    with open(figure_data_file, newline='') as handle:
        plotted = [row for row in csv.DictReader(handle)]
    group_counts = []
    for group_name in robustness_tables.GROUP_NAMES:
        group_counts.append('%s %d' % (group_name, sum(1 for row in plotted if row['group'] == group_name)))
    black_rows = [row for row in plotted if row['marker'].startswith('black')]
    figure_text = ('outputs/figures/FigS3_bylinskii_robustness%s.png: %d rows in five groups (%s), one row per '
                   'simulation, labeled by the value of the factor varied (%s); x = Spearman rho, across target '
                   'items, of memorability (hit rate minus false alarm rate) between the within-category and '
                   'across-category contexts, 0 to 1, dashed zero line; %d black points (rho %s), the first '
                   'labeled "Fig. 2a Design"; no error bars'
                   % ('_paper_run' if arguments.paper_run else '', len(plotted), ', '.join(group_counts),
                      ', '.join(row['row_label'] for row in plotted), len(black_rows),
                      ', '.join('%.3f' % float(row['rho']) for row in black_rows)))
    figure_note = ('Fig. S3 as drawn by code/make_figureS3.py: one row per simulation, grouped by the factor varied, '
                   'at the Spearman rho of memorability across contexts; the five simulations with the category '
                   'structure and criteria of Fig. 2a in black (plotted values in outputs/figure_data/).')
else:
    figure_text = ('outputs/figure_data/%s not found: run code/make_figureS3.py%s first'
                   % (os.path.basename(figure_data_file), ' --paper-run' if arguments.paper_run else ''))
    figure_note = ''

# ============================================================================ numbers.csv
table = numbers_table.NumbersTable(script=SCRIPT)
grid_script = 'code/bylinskii_robustness.m'
cell_source = 'Values from %s.' % source_text

table.add('L6-036', len(cells), mode, script=grid_script,
          note='Number of simulations in the grid (rows of the grid table).' + equality_note)
table.add('L6-037', design_text, mode, script=grid_script,
          note='The values of each factor varied over the 18 simulations (from the labels and settings of the rows '
               'of the grid table) and the settings of the reference cell (from the reference struct and the rng '
               'call of code/bylinskii_robustness.m); every other setting of a simulation is that of the reference '
               'cell.')
table.add('L6-038', [min(all_rho), max(all_rho)], mode, script=grid_script,
          note='Smallest and largest Spearman rho over the 18 simulations, across target items, between memorability '
               '(hit rate minus false alarm rate) in the within-category and in the across-category context; '
               'smallest at "%s", largest at "%s"; %d of 18 above zero. %s'
               % (lowest['label'], highest['label'], sum(1 for rho in all_rho if rho > 0), cell_source))
table.add('L6-039', [min(equal_rho), max(equal_rho)], mode, script=grid_script,
          note='Smallest and largest Spearman rho of memorability across contexts among the %d simulations with the '
               'same criterion in both contexts (%s). %s'
               % (len(equal_cells), ', '.join('%s %.4f' % (criteria_text(cell), cell['rho']) for cell in equal_cells),
                  cell_source))
similarity_range = [min(all_similarity_r), max(all_similarity_r)]
table.add('L6-040', similarity_range, mode, script=grid_script,
          note='Smallest and largest Pearson r over the 18 simulations, across target items, between summed '
               'similarity within the category (minus the mean Euclidean distance to the other items of the '
               'category) and across the whole set (minus the mean Euclidean distance to all items); smallest at '
               '"%s", largest at "%s"; %d of 18 above zero. %s'
               % (lowest_similarity['label'], highest_similarity['label'],
                  sum(1 for r_value in all_similarity_r if r_value > 0), cell_source))
table.add('L6-041', figure_text, mode, script='code/make_figureS3.py', note=figure_note)
table.add('L6-042', len(cells), mode, script=grid_script,
          note='Number of simulations in the grid (rows of the grid table).')
table.add('L6-043', len(fig2a_cells), mode, script=grid_script,
          note='Number of simulations with the category structure and criteria of Fig. 2a (criteria 20 / 5, 10 '
               'category-defining features, 21 categories of 300 items), which differ only in their random draws: '
               '%s. %s'
               % ('; '.join('"%s" %.4f' % (cell['label'], cell['rho']) for cell in fig2a_cells), cell_source))
table.add('L6-044', 'g %.2f; copy fidelity c %.2f' % (reference['g'], reference['c']), mode, script=grid_script,
          note='g and copy fidelity c of the REM in code/bylinskii_robustness.m (its reference struct), the standard '
               'values of Shiffrin and Steyvers (1997).')
table.add('L6-046', max(all_p), mode, script=grid_script,
          note='Largest two-sided p of the Spearman rho of memorability across contexts over the 18 simulations '
               '(t approximation with n - 2 degrees of freedom, MATLAB corr; 750 to 3,150 target items per '
               'simulation). ' + cell_source)
table.write(paths['outputs'])

for row in table.rows:
    print('  %s  paper %-28s computed %-40s %s' % (row['ledger_id'], row['paper_value'][:28],
                                                    row['computed_value'][:40], row['match']))
