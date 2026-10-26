"""
gather_all_fits.py: step 3 of module A6b. The table behind Fig. S16, in the layout of the saved run
in data/paper_run/all_fits.csv, and its comparison with that saved run.

WHAT IT DOES
  1. Reads outputs/refit_sweep.csv (step 2) and writes outputs/all_fits.csv: one row per embedding
     (the published 49-feature embedding and the 14 refits) with the columns of
     data/paper_run/all_fits.csv. k is the number of features; fig5c_r and fig5c_p are the Fig. 5c
     statistic and its parametric p; maintain_vs_orig, maintain_sim, changed_vs_orig and
     changed_sim are Figs. 6c, 6d, 6f and 6g. Two more columns, maintain_sim_monkey and
     changed_sim_monkey (Figs. 6e and 6h, the monkey IT space), are not in the saved run.
  2. Compares this run with the saved run, row by row (rows matched by name), in each column the
     saved run has.
  3. Compares the six Fig. 6 correlations of the published embedding (the four human-space columns
     and the two monkey-space columns) with module 06's saved run of Fig. 6
     (06_image_set_experiments_fig6_fig7/data/paper_run/paperStats_maintained_vs_changed.mat).
  Both comparisons are written to outputs/all_fits_vs_paper_run.csv (per reference and column:
  values compared, values identical to the last digit, largest absolute difference). The script
  stops with an error when a row is missing from the saved run, a text cell differs, or a number
  differs by more than 1e-10.

INPUTS   outputs/refit_sweep.csv (step 2), data/paper_run/all_fits.csv (the saved run that the paper
         reports), 06_image_set_experiments_fig6_fig7/data/paper_run/paperStats_maintained_vs_changed.mat
OUTPUTS  outputs/all_fits.csv, outputs/all_fits_vs_paper_run.csv

RUN (from the repository root; a second)
  python A6b_embedding_size_figS16/code/gather_all_fits.py
"""
import csv
import os
import sys

import numpy as np
import pandas as pd
import scipy.io

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True     # no __pycache__ folders inside the release
sys.path.insert(0, os.path.join(CODE_FOLDER, '..', '..', 'shared', 'python'))
import release_paths   # noqa: E402  (shared helper: release folders)

# (column of outputs/all_fits.csv and data/paper_run/all_fits.csv, column of outputs/refit_sweep.csv)
TABLE_COLUMNS = [('source', 'source'),
                 ('name', 'embedding'),
                 ('subdir', 'arm'),
                 ('k', 'number_of_features'),
                 ('sparsity', 'share_of_weights_below_0_1'),
                 ('val_acc', 'held_out_accuracy'),
                 ('lmbda', 'penalty_lambda'),
                 ('init_dim', 'starting_features'),
                 ('penalty_bound', 'penalty_removed_features'),
                 ('ws_r_vs_published', 'summed_similarity_r_with_published'),
                 ('fig5c_r', 'fig5c_r'),
                 ('fig5c_p', 'fig5c_p'),
                 ('rho_ws_mem', 'rho_summed_similarity_memorability'),
                 ('maintain_sim', 'fig6d_maintained_human_r'),
                 ('maintain_vs_orig', 'fig6c_maintained_vs_things_r'),
                 ('changed_sim', 'fig6g_changed_human_r'),
                 ('changed_vs_orig', 'fig6f_changed_vs_things_r'),
                 ('maintain_sim_monkey', 'fig6e_maintained_monkey_r'),
                 ('changed_sim_monkey', 'fig6h_changed_monkey_r')]
TEXT_COLUMNS = ('source', 'name', 'subdir', 'penalty_bound')
NEW_COLUMNS = ('maintain_sim_monkey', 'changed_sim_monkey')       # not in data/paper_run/all_fits.csv
LARGEST_ALLOWED_DIFFERENCE = 1e-10
PUBLISHED_NAME = 'published 49-d'
PAPER_RUN_REFERENCE = 'data/paper_run/all_fits.csv'

# Module 06's saved run of Fig. 6 (the published embedding): (column of the table, field of paperStats)
MODULE_06_PAPER_RUN_FILE = os.path.join(release_paths.find_module('06'), 'data', 'paper_run',
                                        'paperStats_maintained_vs_changed.mat')
MODULE_06_REFERENCE = '06_image_set_experiments_fig6_fig7/data/paper_run/paperStats_maintained_vs_changed.mat'
FIG6_PAPER_RUN_FIELDS = [('maintain_vs_orig', 'r_origVsMaintain'),         # Fig. 6c
                         ('maintain_sim', 'r_sumSimVsMaintain'),           # Fig. 6d
                         ('maintain_sim_monkey', 'r_monkeyVsMaintain'),    # Fig. 6e
                         ('changed_vs_orig', 'r_origVsChanged'),           # Fig. 6f
                         ('changed_sim', 'r_sumSimVsChanged'),             # Fig. 6g
                         ('changed_sim_monkey', 'r_monkeyVsChanged')]      # Fig. 6h


def cell_text(value):
    """A cell of the table: empty for a missing value (as in data/paper_run/all_fits.csv), the
    shortest exact decimal for a number, whole numbers without a decimal point."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    return str(value)


def as_number(text):
    """A table cell as a number (NaN when empty)."""
    if text is None or str(text).strip() in ('', 'NaN', 'nan'):
        return np.nan
    return float(text)


def main():
    paths = release_paths.module_paths(__file__)
    sweep_file = os.path.join(paths['outputs'], 'refit_sweep.csv')
    if not os.path.exists(sweep_file):
        raise SystemExit('outputs/refit_sweep.csv is missing: run run_refit_sweep.py first')
    print('step 3: the Fig. S16 table and its comparison with the saved run')
    sweep = pd.read_csv(sweep_file, keep_default_na=False, na_values=['NaN'], float_precision='round_trip',
                        dtype={'arm': str, 'penalty_removed_features': str})

    # ---- 1. the table in the layout of data/paper_run/all_fits.csv ----
    table_rows = []
    for _, sweep_row in sweep.iterrows():
        table_row = {}
        for table_column, sweep_column in TABLE_COLUMNS:
            table_row[table_column] = sweep_row[sweep_column]
        table_rows.append(table_row)
    output_table = os.path.join(paths['outputs'], 'all_fits.csv')
    with open(output_table, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([column for column, _ in TABLE_COLUMNS])
        for table_row in table_rows:
            writer.writerow([cell_text(table_row[column]) for column, _ in TABLE_COLUMNS])
    print('  wrote %s (%d rows)' % (os.path.relpath(output_table, release_paths.release_root()), len(table_rows)))

    # ---- 2. comparison with the saved run ----
    saved = pd.read_csv(os.path.join(paths['paper_run'], 'all_fits.csv'), keep_default_na=False,
                        dtype=str).set_index('name')
    comparison_rows = []
    problems = []
    for table_column, _ in TABLE_COLUMNS:
        if table_column == 'name' or table_column in NEW_COLUMNS:
            continue
        rows_compared = 0
        rows_identical = 0
        largest_difference = 0.0
        for table_row in table_rows:
            name = table_row['name']
            if name not in saved.index:
                problems.append('row %s is not in the saved run' % name)
                continue
            saved_text = saved.loc[name, table_column]
            new_text = cell_text(table_row[table_column])
            rows_compared += 1
            if table_column in TEXT_COLUMNS:
                if saved_text == new_text:
                    rows_identical += 1
                else:
                    problems.append('%s, %s: saved %r, this run %r' % (name, table_column, saved_text, new_text))
                continue
            saved_value = as_number(saved_text)
            new_value = as_number(new_text)
            if np.isnan(saved_value) and np.isnan(new_value):
                rows_identical += 1
                continue
            difference = abs(new_value - saved_value)
            if np.isnan(difference) or difference > LARGEST_ALLOWED_DIFFERENCE:
                problems.append('%s, %s: saved %s, this run %s' % (name, table_column, saved_text, new_text))
            if not np.isnan(difference):
                largest_difference = max(largest_difference, difference)
            if new_value == saved_value:
                rows_identical += 1
        comparison_rows.append(dict(reference=PAPER_RUN_REFERENCE, column=table_column,
                                    values_compared=rows_compared, values_identical=rows_identical,
                                    largest_absolute_difference=largest_difference))

    # ---- 3. the published embedding's six Fig. 6 correlations against module 06's saved run ----
    module_06_run = scipy.io.loadmat(MODULE_06_PAPER_RUN_FILE, squeeze_me=True,
                                     struct_as_record=False)['paperStats']
    published_rows = [table_row for table_row in table_rows if table_row['name'] == PUBLISHED_NAME]
    if len(published_rows) != 1:
        raise SystemExit('the published embedding is missing from outputs/refit_sweep.csv')
    published_row = published_rows[0]
    for table_column, paper_run_field in FIG6_PAPER_RUN_FIELDS:
        module_06_value = float(getattr(module_06_run, paper_run_field))
        new_value = as_number(cell_text(published_row[table_column]))
        difference = abs(new_value - module_06_value)
        if np.isnan(difference) or difference > LARGEST_ALLOWED_DIFFERENCE:
            problems.append('%s, %s: module 06 saved run %r, this run %r'
                            % (PUBLISHED_NAME, table_column, module_06_value, new_value))
        comparison_rows.append(dict(reference=MODULE_06_REFERENCE + ' (' + paper_run_field + ')',
                                    column=table_column, values_compared=1,
                                    values_identical=1 if new_value == module_06_value else 0,
                                    largest_absolute_difference=difference))

    comparison_file = os.path.join(paths['outputs'], 'all_fits_vs_paper_run.csv')
    with open(comparison_file, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['reference', 'column', 'values_compared', 'values_identical',
                         'largest_absolute_difference'])
        for row in comparison_rows:
            writer.writerow([row['reference'], row['column'], row['values_compared'], row['values_identical'],
                             repr(float(row['largest_absolute_difference']))])
    for row in comparison_rows:
        if row['reference'] == PAPER_RUN_REFERENCE:
            print('  %-20s %2d of %2d values identical to the saved run (largest difference %.3g)'
                  % (row['column'], row['values_identical'], row['values_compared'],
                     row['largest_absolute_difference']))
        else:
            print('  %-20s published embedding vs Fig. 6 in module 06\'s saved run: difference %.3g'
                  % (row['column'], row['largest_absolute_difference']))
    print('  wrote', os.path.relpath(comparison_file, release_paths.release_root()))
    if problems:
        for problem in problems:
            print('  DIFFERENT FROM THE SAVED RUN:', problem)
        raise SystemExit('this run differs from the saved run (see above)')
    print('  every value of this run equals the saved run (data/paper_run/all_fits.csv; the six Fig. 6 '
          'values of the published embedding are within %g of module 06\'s saved run)' % LARGEST_ALLOWED_DIFFERENCE)


if __name__ == '__main__':
    main()
