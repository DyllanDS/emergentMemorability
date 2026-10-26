"""
check_provided.py: reads the saved results of this analysis (data/provided/concept_level.txt and
concept_level.json) and checks them.

WHAT IT DOES
  1. Reads the header lines and the nearest-neighbor table of concept_level.txt by their row
     labels (the k column) and checks every printed rho (3 decimals) and p value (1 significant
     digit) against the full-precision values in concept_level.json.
  2. Finds the k values of concept_level.json whose four correlations (and the p value of corrected
     recognition) round to the values printed in the paper (shared/ledger/paper_numbers.csv, items
     L9-021 to L9-024); run_module.py checks that the k = 5 row does.
  3. When the module has run the analysis with the count table (EXTERNAL_INPUTS.md M2), compares
     the new k = 5 correlations with the saved results.

INPUTS   data/provided/concept_level.txt, data/provided/concept_level.json (the saved results of
         this analysis), shared/ledger/paper_numbers.csv
OUTPUTS  none (run_module.py writes the tables from what these functions return)
"""
import json
import os
import re
import sys

sys.dont_write_bytecode = True

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import numbers_table   # noqa: E402

# The memorability measures in the column order of the report and json ...
MEASURE_COLUMNS = ['CR', 'hit rate', 'FA rate', 'd-prime']
# ... and the ledger item of each one.
LEDGER_ITEM_OF_MEASURE = {'FA rate': 'L9-021', 'd-prime': 'L9-022', 'hit rate': 'L9-023', 'CR': 'L9-024'}
TABLE_TITLE = 'Nearest-neighbor crowding among concepts, by k'


def read_provided_report(path):
    """The header counts and the nearest-neighbor table of concept_level.txt.

    Returns {'photographs', 'concepts', 'median_photographs', 'dimensions', 'spaces', 'k_table'},
    where k_table[k][measure] = (rho as printed, p as printed), both as text."""
    with open(path, encoding='utf-8') as handle:
        lines = [line.rstrip('\n') for line in handle]
    report = {'k_table': {}}
    for line in lines:
        match = re.match(r'^([\d,]+) photographs collapsed to ([\d,]+) concepts \(median (\d+) photographs each\)\.$',
                         line)
        if match:
            report['photographs'] = int(match.group(1).replace(',', ''))
            report['concepts'] = int(match.group(2).replace(',', ''))
            report['median_photographs'] = int(match.group(3))
        match = re.match(r'^([\d,]+) concatenated dimensions from (\d+) network/layer spaces', line)
        if match:
            report['dimensions'] = int(match.group(1).replace(',', ''))
            report['spaces'] = int(match.group(2))

    # the table: its title line, a rule, the column labels, then one row per k until a blank line
    title_line = lines.index(TABLE_TITLE)
    header_line = title_line + 2
    column_labels = re.split(r'\s{2,}', lines[header_line].strip())
    if column_labels[0] != 'k' or column_labels[1:] != MEASURE_COLUMNS:
        raise ValueError('unexpected column labels in %s: %s' % (path, column_labels))
    for line in lines[header_line + 1:]:
        if not line.strip():
            break
        match = re.match(r'^\s*(\d+)\s+(.*)$', line)
        cells = re.findall(r'([+-]\d+\.\d+) \(p=([^)]+)\)', match.group(2))
        if len(cells) != len(MEASURE_COLUMNS):
            raise ValueError('row k = %s of %s does not have %d cells' % (match.group(1), path, len(MEASURE_COLUMNS)))
        report['k_table'][int(match.group(1))] = dict(zip(MEASURE_COLUMNS, cells))
    return report


def read_provided_json(path):
    """concept_level.json as a dict. The module reads by_k: by_k[k] holds the four [rho, p] pairs of
    row k in MEASURE_COLUMNS order."""
    with open(path) as handle:
        return json.load(handle)


def value_of(results, k, measure):
    """(rho, p) of one measure at one k, from the json content."""
    rho, p = results['by_k'][str(k)][MEASURE_COLUMNS.index(measure)]
    return float(rho), float(p)


def check_report_against_json(report, results):
    """Every rho and p printed in the k table equals the json value at the printed precision.
    Returns a list of (label, passed, detail)."""
    checks = []
    for k in sorted(report['k_table']):
        for measure in MEASURE_COLUMNS:
            printed_rho, printed_p = report['k_table'][k][measure]
            rho, p = value_of(results, k, measure)
            passed = ('%+.3f' % rho == printed_rho) and (format(p, '.1g') == printed_p)
            checks.append(('concept_level.txt k = %d %s: rho %s (p=%s) equals the json value' % (k, measure, printed_rho, printed_p),
                           passed, 'json rho %.6f, p %.4g' % (rho, p)))
    json_ks = sorted(int(k) for k in results['by_k'])
    checks.append(('the txt and the json list the same k values', json_ks == sorted(report['k_table']),
                   'k = %s' % ', '.join(str(k) for k in json_ks)))
    return checks


def printed_values():
    """{measure: the value printed in the paper, as text}, from shared/ledger/paper_numbers.csv."""
    ledger = numbers_table.load_ledger()
    printed = {}
    for measure, ledger_id in LEDGER_ITEM_OF_MEASURE.items():
        printed[measure] = ledger[ledger_id]['paper_value']
    return printed


def rows_matching_printed(results):
    """The k values whose four correlations (and the CR p value) round to the values printed in the paper."""
    printed = printed_values()
    matching = []
    for k in sorted(int(key) for key in results['by_k']):
        all_round = True
        for measure in MEASURE_COLUMNS:
            rho, p = value_of(results, k, measure)
            parts = [part for part in printed[measure].split(';') if part.strip()]
            computed = [rho, p][:len(parts)]
            if numbers_table.judge(printed[measure], computed, 'round') != 'yes':
                all_round = False
        if all_round:
            matching.append(k)
    return matching


def compare_runs(regenerated, provided, reported_k):
    """Largest differences between the json of a run with the count table and the saved results, in
    the row the paper reports (reported_k): absolute for rho, relative for p."""
    largest_rho_reported_row = 0.0
    largest_relative_p = 0.0
    for (rho_new, p_new), (rho_old, p_old) in zip(regenerated['by_k'][str(reported_k)], provided['by_k'][str(reported_k)]):
        largest_rho_reported_row = max(largest_rho_reported_row, abs(rho_new - rho_old))
        largest_relative_p = max(largest_relative_p, abs(p_new - p_old) / p_old)
    return {'largest_rho_difference_reported_row': largest_rho_reported_row,
            'largest_relative_p_difference_reported_row': largest_relative_p}


def check(provided_folder):
    """Reads and checks the saved results. Returns (report, results, checks, matching_k)."""
    report = read_provided_report(os.path.join(provided_folder, 'concept_level.txt'))
    results = read_provided_json(os.path.join(provided_folder, 'concept_level.json'))
    checks = check_report_against_json(report, results)
    matching_k = rows_matching_printed(results)
    return report, results, checks, matching_k
