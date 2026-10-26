"""
compare_scenes_runs.py: the scene results of this run next to the saved scene run in
data/paper_run_scenes/, the output the paper reports.

What it does
    Lines up every scene number of this run (outputs/scenes/) with the saved run: the held-out
    correlations of Fig. S15a and their five folds, the kernel width tau and the ridge penalty alpha
    chosen in each fold, the whole-sample correlations with their p values (Fig. S15b), the split-half
    reliability over participants with its Spearman-Brown value and the image counts. For each it gives
    the difference between the two runs, whether that difference is within 0.01 (correlations only), and,
    for the printed values (shared/ledger/paper_numbers.csv), whether this run rounds to them.
Inputs
    A6a_faces_and_scenes_figS15/outputs/scenes/         (written by code/scenes/02, 04 and 05)
    A6a_faces_and_scenes_figS15/data/paper_run_scenes/  (the saved scene run)
Output
    A6a_faces_and_scenes_figS15/outputs/scenes_rerun_vs_paper_run.csv
        quantity, printed_in_paper, paper_run (the saved run), rerun (this run), difference,
        within_0_01, rerun_rounds_to_printed
Run (from the repository root; code/run_module.py runs it)
    python A6a_faces_and_scenes_figS15/code/compare_scenes_runs.py
"""
import csv
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import read_results  # noqa: E402

OUTPUT_FILE = os.path.join(read_results.MODULE_FOLDER, 'outputs', 'scenes_rerun_vs_paper_run.csv')
TOLERANCE = 0.01
PAPER = 'scenes_paper_run'
RERUN = 'scenes_rerun'


def rounds_to(value, printed_text):
    """True when value rounds to the printed number at its printed number of decimals."""
    decimals = len(printed_text.split('.')[1]) if '.' in printed_text else 0
    return abs(value - float(printed_text)) <= 0.5 * 10 ** (-decimals) + 1e-12


def add_row(rows, quantity, paper_value, rerun_value, printed='', is_correlation=True, show_difference=True):
    """One comparison row. Only correlations get the 0.01 tolerance check; chosen settings and counts are
    listed with their difference, p values side by side only."""
    if show_difference:
        difference = rerun_value - paper_value
    else:
        difference = ''
    if is_correlation:
        within = 'yes' if abs(rerun_value - paper_value) <= TOLERANCE else 'no'
    else:
        within = ''
    if printed:
        rounds = 'yes' if rounds_to(rerun_value, printed) else 'no'
    else:
        rounds = ''
    rows.append(dict(quantity=quantity,
                     printed_in_paper=printed,
                     paper_run=paper_value,
                     rerun=rerun_value,
                     difference=difference,
                     within_0_01=within,
                     rerun_rounds_to_printed=rounds))


def reliability_values(source):
    """{measure: (split-half, Spearman-Brown)} over participants, at full precision, from noise_ceiling.json."""
    path = read_results.file_path(source, 'noise').replace('.txt', '.json')
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)['over participants']


def main():
    rows = []

    # Fig. S15a: held-out correlations (mean of five folds) and each fold
    panel_a = [('Fig. S15a equal weights', read_results.EQUAL_WEIGHTS_ROW, read_results.EQUAL_WEIGHTS_COLUMN,
                read_results.printed('Scenes', 'equal')),
               ('Fig. S15a fitted weights', read_results.FITTED_WEIGHTS_ROW, read_results.FITTED_WEIGHTS_COLUMN,
                read_results.printed('Scenes', 'fitted'))]
    for name, row_name, column, printed in panel_a:
        paper_folds = read_results.fold_values(PAPER, column)
        rerun_folds = read_results.fold_values(RERUN, column)
        paper_mean = sum(paper_folds) / len(paper_folds)
        rerun_mean = sum(rerun_folds) / len(rerun_folds)
        add_row(rows, name + ', held-out rho (mean of five folds)', paper_mean, rerun_mean, printed)
        for fold_number in range(len(paper_folds)):
            add_row(rows, '%s, held-out rho, fold %d' % (name, fold_number + 1),
                    paper_folds[fold_number], rerun_folds[fold_number])

    # the settings chosen inside the training folds
    paper_settings = read_results.fold_settings(PAPER)
    rerun_settings = read_results.fold_settings(RERUN)
    for key, label in (('tau_equal', 'kernel width tau chosen, equal weights'),
                       ('tau_fitted', 'kernel width tau chosen, fitted weights'),
                       ('alpha', 'ridge penalty alpha chosen')):
        for fold_number in range(len(paper_settings[key])):
            add_row(rows, '%s, fold %d' % (label, fold_number + 1),
                    paper_settings[key][fold_number], rerun_settings[key][fold_number], is_correlation=False)

    # whole-sample correlations and their p values (Fig. S15b)
    panel_b = [('the hit rate (Fig. S15b)', 'hit rate', read_results.printed('Scenes', 'hit rate')),
               ('the false alarm rate (Fig. S15b)', 'FA rate', read_results.printed('Scenes', 'false alarm rate'))]
    for measure, label, printed in panel_b:
        paper_rho, paper_p = read_results.limiting_case(PAPER, label)
        rerun_rho, rerun_p = read_results.limiting_case(RERUN, label)
        add_row(rows, 'whole-sample rho with %s, 3 decimals as the reports print it' % measure,
                paper_rho, rerun_rho, printed)
        add_row(rows, 'p value of the whole-sample rho with ' + measure, paper_p, rerun_p, is_correlation=False,
                show_difference=False)

    # split-half reliability over participants (computed from the participant matrices with seed 0, not
    # from the embeddings), at full precision from noise_ceiling.json
    measures = [('cr', 'corrected recognition')]
    if read_results.scene_reliability_source() != RERUN:
        print('this run has no reliability over participants (no participant matrices); not compared')
        measures = []
    if measures:
        paper_reliability = reliability_values(PAPER)
        rerun_reliability = reliability_values(RERUN)
    for key, measure in measures:
        printed = read_results.printed('Scenes', 'reliability') if key == 'cr' else ''
        add_row(rows, 'split-half reliability over participants, ' + measure,
                paper_reliability[key][0], rerun_reliability[key][0], printed)
        add_row(rows, 'Spearman-Brown reliability over participants, ' + measure,
                paper_reliability[key][1], rerun_reliability[key][1])

    # image counts
    paper_embedded, paper_scored = read_results.image_counts(PAPER)
    rerun_embedded, rerun_scored = read_results.image_counts(RERUN)
    add_row(rows, 'images embedded', paper_embedded, rerun_embedded, is_correlation=False)
    add_row(rows, 'scored images entering the correlations', paper_scored, rerun_scored, is_correlation=False)

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print('Scenes: this run against the saved run in data/paper_run_scenes/ (the values the paper reports)')
    print('%-86s %10s %10s %10s %6s %8s' % ('quantity', 'saved run', 'this run', 'diff', '<0.01', 'printed'))
    for row in rows:
        if row['difference'] == '':
            difference_text = '-'
        else:
            difference_text = '%+.4f' % row['difference']
        print('%-86s %10.6g %10.6g %10s %6s %8s' % (row['quantity'], row['paper_run'], row['rerun'], difference_text,
                                                  row['within_0_01'] or '-', row['rerun_rounds_to_printed'] or '-'))
    correlation_rows = []
    for row in rows:
        if row['within_0_01'] != '':
            correlation_rows.append(row)
    largest = 0.0
    not_within = []
    for row in correlation_rows:
        largest = max(largest, abs(row['rerun'] - row['paper_run']))
        if row['within_0_01'] == 'no':
            not_within.append(row['quantity'])
    printed_rows = []
    printed_mismatch = []
    for row in rows:
        if row['rerun_rounds_to_printed'] != '':
            printed_rows.append(row)
            if row['rerun_rounds_to_printed'] == 'no':
                printed_mismatch.append(row['quantity'])
    print('\nlargest difference between the runs over the correlations: %.4f' % largest)
    print('correlations differing by more than %.2f: %s' % (TOLERANCE, ', '.join(not_within) if not_within else 'none'))
    print('printed scene values this run does not round to: %s'
          % (', '.join(printed_mismatch) if printed_mismatch else 'none (%d checked)' % len(printed_rows)))
    print('wrote', read_results.relative(OUTPUT_FILE))


if __name__ == '__main__':
    main()
