"""
run_module.py: entry point of module A4b (SI Appendix, Section 4: crowding among the THINGS
concepts in the deep-network spaces of SI Appendix, Section 5 and the memorability of the concepts).

Run from the repository root:
    python A4b_things_concepts/code/run_module.py

WHAT IT DOES
  1. concept_level.py: the crowding of each of the 1,854 concept centroids among its 5 nearest
     concepts (always; outputs/concept_crowding.csv). When the per-image count table is in
     external/kramer_trials/ (EXTERNAL_INPUTS.md M2) it also computes the memorability of every
     concept, the correlations and the report (outputs/regenerated/); otherwise it prints
     "PROVIDED OUTPUT: regenerating needs ...; see EXTERNAL_INPUTS.md".
  2. check_provided.py: reads the saved results (data/provided/concept_level.txt and .json) by row
     label, checks the table of the txt against the json, checks that the four correlations printed
     in the paper come from the k = 5 row and, after a run with the count table, compares the new
     correlations with the saved results.
  3. outputs/numbers.csv: one row per ledger item of the THINGS concepts sentence (L9-020 to
     L9-024), from 1 and 2.
  4. stats_concepts.py: outputs/tests.csv, the uncertainty statistics of the four correlations (T46).
  5. outputs/logs/run_module.log: everything printed.
  Stops with an error if a check of the saved results fails.

OUTPUTS  outputs/concept_crowding.csv, outputs/numbers.csv, outputs/tests.csv,
         outputs/logs/run_module.log; with M2 also outputs/regenerated/concept_level.txt and .json
SEEDS    0 (the 1,000 concepts that set the distance scale, concept_level.py); 20261013 (the concept
         bootstrap of stats_concepts.py, only with M2)
RUNTIME  about 5 seconds; with M2 about 30 seconds (the 10,000-resample concept bootstrap); under
         700 MB of memory
"""
import os
import sys

sys.dont_write_bytecode = True   # keep the release free of __pycache__ folders

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import numbers_table   # noqa: E402
import release_paths   # noqa: E402
import tests_table     # noqa: E402

import check_provided  # noqa: E402
import concept_level   # noqa: E402
import stats_concepts  # noqa: E402


class PrintLog:
    """Sends everything printed to the screen and keeps a copy for outputs/logs/run_module.log."""

    def __init__(self, screen):
        self.screen = screen
        self.text = []

    def write(self, text):
        self.screen.write(text)
        self.text.append(text)

    def flush(self):
        self.screen.flush()


def write_table(table, folder):
    """Writes numbers.csv or tests.csv. The shared writers print the absolute path of the file; that
    message is dropped so the log holds only paths relative to the repository root."""
    printing_to = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        path = table.write(folder)
    finally:
        sys.stdout.close()
        sys.stdout = printing_to
    print('wrote %s (%d rows)' % (concept_level.relative_to_release(path), len(table.rows)))


def main(count_table_path=None, outputs_folder=None):
    """count_table_path: a count table to use instead of external/kramer_trials/ (None: look there).
    outputs_folder: where to write (None: the module's outputs/). Returns a summary dict."""
    module_paths = release_paths.module_paths(__file__, make_output_folders=False)
    outputs = outputs_folder if outputs_folder is not None else module_paths['outputs']
    logs = os.path.join(outputs, 'logs')
    os.makedirs(logs, exist_ok=True)
    log = PrintLog(sys.stdout)
    sys.stdout = log
    try:
        summary = run_everything(module_paths, outputs, count_table_path)
    finally:
        sys.stdout = log.screen
        with open(os.path.join(logs, 'run_module.log'), 'w', encoding='utf-8') as handle:
            handle.write(''.join(log.text))
    return summary


def run_everything(module_paths, outputs, count_table_path):
    print('MODULE A4b: crowding among THINGS concepts (SI Appendix, Section 4)')
    print('=' * 80)

    # ---- 1. the analysis ----
    analysis = concept_level.run(outputs, count_table_path)
    mode = analysis['mode']
    number_of_concepts = len(analysis['concept_names'])
    number_of_photographs = int(analysis['photograph_counts'].sum())
    print('mode: %s' % ('regenerate (count table present)' if mode == 'regenerate'
                        else 'provided output for the memorability side; crowding computed from the centroids'))
    print()

    # ---- 2. the saved results ----
    report, provided, checks, matching_k = check_provided.check(module_paths['provided'])
    checks.append(('the saved results count the same photographs, concepts and dimensions as the included centroids',
                   (report['photographs'], report['concepts'], report['median_photographs'], report['dimensions'],
                    report['spaces'])
                   == (number_of_photographs, number_of_concepts, int(np.median(analysis['photograph_counts'])),
                       analysis['number_of_dimensions'], len(concept_level.NETWORK_LAYERS)),
                   '%s photographs, %s concepts (median %d), %s dimensions, %d spaces'
                   % (format(report['photographs'], ','), format(report['concepts'], ','), report['median_photographs'],
                      format(report['dimensions'], ','), report['spaces'])))
    checks.append(('the four correlations printed in the paper come from the k = %d row'
                   % concept_level.REPORTED_NEIGHBORHOOD_SIZE,
                   matching_k == [concept_level.REPORTED_NEIGHBORHOOD_SIZE],
                   'rows whose four correlations round to the printed values: k = %s'
                   % ', '.join(str(k) for k in matching_k)))
    comparison = None
    if mode == 'regenerate':
        comparison = check_provided.compare_runs(analysis['report'], provided, concept_level.REPORTED_NEIGHBORHOOD_SIZE)
    print("checks of the saved results (data/provided/):")
    failed = []
    table_cells_checked = 0      # the rho and p cells of the k table in concept_level.txt (summarized in one line)
    table_cells_failed = 0
    for label, passed, detail in checks:
        is_table_cell = label.startswith('concept_level.txt k =')
        if is_table_cell:
            table_cells_checked += 1
            if not passed:
                table_cells_failed += 1
        else:
            print('  %s  %s  (%s)' % ('PASS' if passed else 'FAIL', label, detail))
        if not passed:
            failed.append(label)
    print('  %s  every rho and p of the k table in concept_level.txt equals the json value (%d cells)'
          % ('PASS' if table_cells_failed == 0 else 'FAIL', table_cells_checked))
    if failed:
        raise SystemExit('FAILED checks of the saved results: %s' % '; '.join(failed))
    if comparison is not None:
        print('this run against the saved results, k = %d row: largest rho difference %.2g, largest relative p '
              'difference %.2g.'
              % (concept_level.REPORTED_NEIGHBORHOOD_SIZE, comparison['largest_rho_difference_reported_row'],
                 comparison['largest_relative_p_difference_reported_row']))
        if comparison['largest_rho_difference_reported_row'] > 1e-9:
            print('WARNING: the k = %d correlations differ from the saved results. Check that the count table holds '
                  'the counts described in the module README (old trials include the short-lag vigilance repeats).'
                  % concept_level.REPORTED_NEIGHBORHOOD_SIZE)
    print()

    # ---- 3. numbers.csv ----
    k = concept_level.REPORTED_NEIGHBORHOOD_SIZE
    if mode == 'regenerate':
        source_results = analysis['report']
        source_text = "this module's run (outputs/regenerated/concept_level.json)"
    else:
        source_results = provided
        source_text = "the saved results (data/provided/concept_level.json)"
    correlations = {}
    for measure in check_provided.MEASURE_COLUMNS:
        correlations[measure] = check_provided.value_of(source_results, k, measure)

    numbers = numbers_table.NumbersTable(script='code/run_module.py')
    numbers.add('L9-020', k, mode='rerun', match_type='exact', script='code/concept_level.py',
                note='neighborhood size k of the crowding measure (REPORTED_NEIGHBORHOOD_SIZE in '
                     'code/concept_level.py): minus the mean scaled distance to the %d nearest other concepts '
                     '(outputs/concept_crowding.csv)' % k)
    for measure in stats_concepts.MEASURES_IN_TEXT_ORDER:
        ledger_id = check_provided.LEDGER_ITEM_OF_MEASURE[measure]
        rho, p = correlations[measure]
        printed = numbers.ledger[ledger_id]['paper_value']
        parts = [part for part in printed.split(';') if part.strip()]
        computed = [rho, p][:len(parts)]
        note = ('Spearman rho across the %s concepts between crowding at k = %d and %s in the THINGS memorability '
                'data of Kramer et al. (2023), two-sided p = %.2g; from %s'
                % (format(number_of_concepts, ','), k, stats_concepts.MEASURE_DESCRIPTION[measure], p, source_text))
        if mode == 'regenerate':
            rho_old, p_old = check_provided.value_of(provided, k, measure)
            note += '; saved results (data/provided/concept_level.json): rho %.6f, p %.4g' % (rho_old, p_old)
            numbers.add(ledger_id, computed, mode='rerun', match_type='round', script='code/concept_level.py',
                        note=note)
        else:
            note += ', used without the per-image count table (EXTERNAL_INPUTS.md M2)'
            numbers.add(ledger_id, computed, mode='provided', match_type='round', script='code/check_provided.py',
                        note=note)
    write_table(numbers, outputs)
    for row in numbers.rows:
        print('  %s  printed %s  computed %s  match %s (%s)'
              % (row['ledger_id'], row['paper_value'], row['computed_value'], row['match'], row['mode']))

    # ---- 4. tests.csv (T46) ----
    tests = tests_table.TestsTable(script=stats_concepts.SCRIPT)
    if mode == 'regenerate':
        check_lines = stats_concepts.add_tests(tests, correlations, number_of_concepts, 'regenerated',
                                               crowding=analysis['crowding'],
                                               memorability=analysis['memorability'])
    else:
        check_lines = stats_concepts.add_tests(tests, correlations, number_of_concepts, 'provided')
    print('T46: p values recomputed from rho and n (t approximation, df = %d):' % (number_of_concepts - 2))
    for line in check_lines:
        print('  ' + line)
    write_table(tests, outputs)
    return {'mode': mode, 'numbers': numbers.rows, 'tests': tests.rows, 'comparison': comparison,
            'matching_k': matching_k}


if __name__ == '__main__':
    main()
