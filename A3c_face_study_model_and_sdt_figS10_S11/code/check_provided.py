"""
check_provided.py: reads every saved output of the face study's d_a analysis (the parts that need
MemToolbox2; data/provided/), recomputes each number its reports print from the saved fits and
resamples, and writes the tables the figures, the statistics and numbers.csv use (SI Appendix,
Section 3; Fig. 4c; SI Appendix, Figs. S10b and S11b).

WHAT IT CHECKS (every check is one row of outputs/provided_checks.csv)
  point_fit.json (the registered fit)
    - the five d_a quantities recomputed from the fitted means and SDs equal the stored ones;
    - the log likelihood of the shipped data at the stored parameters equals the stored log likelihood.
      This evaluates the fitted model at its saved parameters (never-studied faces N(0, 1); the seven
      criteria are c1, c1 + dc2, ..., c1 + dc2 + ... + dc7; the probability of rating r is the normal
      probability between criteria r - 1 and r); nothing is fitted. Equality shows that the shipped
      data are the data of the saved fit.
  bootstrap_raw.jsonl (10,000 participant resamples, each refitted) and sdt_shared_criteria.txt
    - 10,000 resamples, numbered 0 to 9,999, none failed;
    - every row of the registered report, read by its row label: the item-type means and SDs, the
      three d_a values with their 95% intervals (2.5th and 97.5th percentiles of the resamples, numpy
      percentile), and the two contrasts with interval and two-sided bootstrap p (twice the smaller of
      the shares of resamples at or below zero and at or above zero, the registered definition)
  bootstrap_items.jsonl (1,000 resamples of the faces) and bootstrap_items.txt: every interval and p
  leave_one_pair_out.json and .txt: the all-pairs fit equals the registered fit; every row and the
    two ranges over the six deletions
  rem_like_data.jsonl (the 243 settings scored like the data) and rem_like_data.txt: the settings are
    the grid, every setting was fitted; the counts of settings above zero (242 and 18 of 243) and the
    rows "read off the log-odds" and "fitted like the data"
  With MemToolbox2 (outputs/point_fit.json, leave_one_pair_out.json or rem_like_data.jsonl written by a
  new run) the new fits are compared with the saved ones as well: they agree when every d_a quantity
  is within 0.001 (the precision of the reports); a larger difference is reported, and does not stop
  the module, because the module's numbers always come from the saved fits.

INPUTS   data/provided/* (the saved fits and resamples); 04_rare_feature_faces_fig4/data/face_trials.csv
         (through face_study_common.py, for the likelihood check and the trial counts); new fits in
         outputs/, if any
OUTPUTS  outputs/provided_checks.csv          every check: what, file, recomputed, printed, agree, note
         outputs/registered_fit_summary.csv   the five d_a quantities: estimate, intervals over
                                              participants and over faces, bootstrap p,
                                              leave-one-pair-out range
         outputs/rem_243_summary.csv          the fitted scores of the 243-setting run: settings above
                                              zero, mean, median, range
         outputs/check_provided.txt           the same in words
SEEDS    none (reads saved runs)
RUNTIME  about 2 seconds
RUN      python A3c_face_study_model_and_sdt_figS10_S11/code/check_provided.py
"""
import csv
import itertools
import json
import os
import sys

sys.dont_write_bytecode = True

import numpy as np
from scipy.stats import norm

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_FOLDER)
import face_study_common as F          # noqa: E402
import sdt_shared_criteria as S      # noqa: E402  (item_type, the d_a formula and contrasts; no fitting)
import rem_final_spec as R           # noqa: E402  (the grid)

PATHS = F.release_paths.module_paths(__file__)
PROVIDED = PATHS['provided']
OUTPUTS = PATHS['outputs']

QUANTITIES = [('rare_unrel', 'd_a, rare faces against never-studied faces', 'rare vs never-studied'),
              ('rare_matched', 'd_a, rare faces against their same-feature lures', 'rare vs its own foil'),
              ('ordinary', 'd_a, ordinary faces against never-studied faces', 'ordinary studied vs never-studied'),
              ('interaction', 'effect of the lure set: d_a(rare vs never-studied) - d_a(rare vs same-feature lure)',
               'INTERACTION'),
              ('equivalence', 'effect of the feature itself: d_a(rare vs same-feature lure) - d_a(ordinary vs '
                              'never-studied)', 'WHAT THE FEATURE BUYS')]
CONTRASTS = ('interaction', 'equivalence')

checks = []
RERUN_TOLERANCE = 0.001     # a new fit agrees with the saved one when every d_a quantity is within 0.001 (the
                            # reports' precision); optimizer tolerance on another computer moves the 4th decimal


def add_check(what, file_name, recomputed, printed, agree, note=''):
    checks.append({'check': what, 'file': file_name, 'recomputed': recomputed, 'printed': printed,
                   'agree': 'yes' if agree else 'NO', 'note': note})


def read_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def percentile_interval(values):
    """The 2.5th and 97.5th percentiles (numpy percentile, linear interpolation), as the fitting scripts compute them."""
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def two_sided_bootstrap_p(values):
    """The registered p: twice the smaller of the shares at or below zero and at or above zero."""
    values = np.asarray(values, dtype=float)
    return float(2 * min(np.mean(values <= 0), np.mean(values >= 0)))


def fixed(value, decimals, signed=False):
    """A number as the reports print it ('%.3f' or '%+.3f')."""
    return ('%+.*f' if signed else '%.*f') % (decimals, value)


# ============================================================================ the registered fit
def check_point_fit():
    point_fit = json.load(open(os.path.join(PROVIDED, 'point_fit.json'), encoding='utf-8'))
    cells = {name: tuple(values) for name, values in point_fit['cells'].items()}

    # 1. the five quantities from the means and SDs
    recomputed = S.contrasts(cells)
    largest_difference = 0.0
    for name in recomputed:
        largest_difference = max(largest_difference, abs(float(recomputed[name]) - point_fit['contrasts'][name]))
    add_check('d_a quantities recomputed from the fitted means and SDs', 'point_fit.json',
              'largest difference %.2g' % largest_difference, 'contrasts', largest_difference < 1e-12)

    # 2. the cells repeat the parameters
    parameters = point_fit['params']
    same_cells = True
    for cell_number, cell_name in enumerate(S.CELL_NAMES):
        if (parameters['dprime_cell%d' % cell_number] != cells[cell_name][0]
                or parameters['sigma_cell%d' % cell_number] != cells[cell_name][1]):
            same_cells = False
    add_check('fitted means and SDs equal the named parameters', 'point_fit.json', 'cells', 'params', same_cells)

    # 3. the log likelihood of the shipped data at the saved parameters
    criteria = []
    running_total = 0.0
    for name in ('c1', 'dc2', 'dc3', 'dc4', 'dc5', 'dc6', 'dc7'):
        running_total += parameters[name]
        criteria.append(running_total)
    distributions = dict(cells)
    distributions['unrelated'] = (0.0, 1.0)            # the anchor
    rating_counts = {name: [0] * 9 for name in distributions}
    trial_counts = {name: 0 for name in distributions}
    participants = F.analysable(F.load())
    for participant in participants:
        for trial in participant['rec']:
            kind = S.item_type(trial)
            if kind is None:
                continue
            rating_counts[kind][F.rating(trial)] += 1
            trial_counts[kind] += 1
    log_likelihood = 0.0
    for kind, (mean, sd) in distributions.items():
        edges = [-np.inf] + criteria + [np.inf]
        for rating in range(1, 9):
            probability = (norm.cdf((edges[rating] - mean) / sd) - norm.cdf((edges[rating - 1] - mean) / sd))
            log_likelihood += rating_counts[kind][rating] * np.log(probability)
    add_check('log likelihood of the shipped data at the saved parameters (evaluated, not fitted)',
              'point_fit.json', '%.6f' % log_likelihood, '%.6f' % point_fit['loglike'],
              abs(log_likelihood - point_fit['loglike']) < 1e-6,
              '%d participants, %d trials: %s' % (len(participants), sum(trial_counts.values()),
                                                  ', '.join('%s %d' % item for item in trial_counts.items())))
    return point_fit, criteria, trial_counts, len(participants)


# ============================================================================ the participant bootstrap
def check_participant_bootstrap(point_fit, number_of_criteria, number_of_participants, trial_counts):
    rows = read_jsonl(os.path.join(PROVIDED, 'bootstrap_raw.jsonl'))
    numbers = sorted(row['i'] for row in rows)
    complete = numbers == list(range(10000))
    failed = sum(1 for row in rows if not row.get('c'))
    only_expected_fields = all(set(row) == {'i', 'c'} for row in rows)
    add_check('10,000 resamples numbered 0 to 9,999, none failed, fields i and c only', 'bootstrap_raw.jsonl',
              '%d resamples, %d failed' % (len(rows), failed), '10000', complete and failed == 0 and only_expected_fields)
    resampled = {}
    for name, _, _ in QUANTITIES:
        values = []
        for row in rows:
            if row.get('c') and np.isfinite(row['c'][name]):
                values.append(row['c'][name])
        resampled[name] = np.array(values, dtype=float)

    lines = F.read_lines(os.path.join(PROVIDED, 'sdt_shared_criteria.txt'))
    # the header sentence: "100 participants, 36 test trials each. ... share one set of 7 criteria"
    header_numbers = F.numbers_in(F.find_line_containing(lines, 'test trials each'))
    criteria_numbers = F.numbers_in(F.find_row(lines, 'placed on it, and all four share one set of'))
    trials_each = sum(trial_counts.values()) // number_of_participants
    add_check('report header: participants, test trials each, criteria', 'sdt_shared_criteria.txt',
              '%d, %d, %d' % (number_of_participants, trials_each, number_of_criteria),
              '%d, %d, %d' % (header_numbers[0], header_numbers[1], criteria_numbers[0]),
              header_numbers[:2] == [number_of_participants, trials_each]
              and criteria_numbers[0] == number_of_criteria)
    resample_numbers = F.numbers_in(F.find_row(lines, 'DISCRIMINABILITY (d_a), frequentist,'))
    add_check('report: number of participant resamples', 'sdt_shared_criteria.txt', '%d' % len(rows),
              '%d' % resample_numbers[0], resample_numbers[0] == len(rows))
    for cell_name in S.CELL_NAMES:
        printed = F.numbers_in(F.find_row(lines, cell_name, after='WHERE EACH ITEM TYPE SITS'))
        mean, sd = point_fit['cells'][cell_name]
        add_check('report: where %s sits (mean, SD)' % cell_name, 'sdt_shared_criteria.txt',
                  '%s, %s' % (fixed(mean, 3), fixed(sd, 3)), '%s, %s' % (fixed(printed[0], 3), fixed(printed[1], 3)),
                  fixed(mean, 3) == fixed(printed[0], 3) and fixed(sd, 3) == fixed(printed[1], 3))

    summary = {}
    for name, description, report_label in QUANTITIES:
        estimate = point_fit['contrasts'][name]
        low, high = percentile_interval(resampled[name])
        p_value = two_sided_bootstrap_p(resampled[name])
        summary[name] = {'quantity': name, 'description': description, 'estimate': estimate,
                         'participants_ci_low': low, 'participants_ci_high': high,
                         'participants_p_two_sided': p_value, 'participants_resamples': len(resampled[name]),
                         'participants_at_or_below_zero': int(np.sum(resampled[name] <= 0)),
                         'participants_at_or_above_zero': int(np.sum(resampled[name] >= 0))}
        printed = F.numbers_in(F.find_row(lines, report_label))
        signed = name in CONTRASTS
        recomputed_text = '%s [%s, %s]' % (fixed(estimate, 3, signed), fixed(low, 3, True), fixed(high, 3, True))
        printed_text = '%s [%s, %s]' % (fixed(printed[0], 3, signed), fixed(printed[1], 3, True),
                                        fixed(printed[2], 3, True))
        agree = recomputed_text == printed_text
        if signed:
            recomputed_text += ', p = %.4f' % p_value
            printed_text += ', p = %.4f' % printed[3]
            agree = agree and ('%.4f' % p_value == '%.4f' % printed[3])
        add_check('registered report row: %s' % report_label, 'sdt_shared_criteria.txt + bootstrap_raw.jsonl',
                  recomputed_text, printed_text, agree, 'estimate from point_fit.json; interval and p from the resamples')
    return summary


# ============================================================================ the item bootstrap
def check_item_bootstrap(summary):
    rows = read_jsonl(os.path.join(PROVIDED, 'bootstrap_items.jsonl'))
    by_scheme = {}
    errors = 0
    for row in rows:
        if row.get('err') is not None or not row.get('c'):
            errors += 1
            continue
        by_scheme.setdefault(row['mode'], []).append(row)
    numbers = sorted(row['i'] for row in by_scheme.get('items', []))
    complete = numbers == list(range(1000))
    add_check('1,000 resamples of the faces, numbered 0 to 999, no error', 'bootstrap_items.jsonl',
              'items %d, errors %d' % (len(by_scheme.get('items', [])), errors), 'items 1000',
              complete and errors == 0)

    lines = F.read_lines(os.path.join(PROVIDED, 'bootstrap_items.txt'))
    for name, _, _ in QUANTITIES:
        values = np.array([row['c'][name] for row in by_scheme['items'] if np.isfinite(row['c'][name])], dtype=float)
        low, high = percentile_interval(values)
        summary[name]['items_ci_low'] = low
        summary[name]['items_ci_high'] = high
        summary[name]['items_p_two_sided'] = two_sided_bootstrap_p(values)
        summary[name]['items_resamples'] = len(values)
        summary[name]['items_at_or_below_zero'] = int(np.sum(values <= 0))
    for name, heading in (('interaction', 'INTERACTION'), ('equivalence', 'WHAT THE FEATURE BUYS')):
        printed = F.numbers_in(F.find_row(lines, 'items', after=heading))   # n, low, high, p
        recomputed_text = 'n=%d [%s, %s] p = %.4f' % (summary[name]['items_resamples'],
                                                     fixed(summary[name]['items_ci_low'], 3, True),
                                                     fixed(summary[name]['items_ci_high'], 3, True),
                                                     summary[name]['items_p_two_sided'])
        printed_text = 'n=%d [%s, %s] p = %.4f' % (printed[0], fixed(printed[1], 3, True),
                                                  fixed(printed[2], 3, True), printed[3])
        add_check('item bootstrap report row: %s, items' % heading, 'bootstrap_items.txt + .jsonl',
                  recomputed_text, printed_text, recomputed_text == printed_text)


# ============================================================================ leave one pair out
def check_leave_one_pair_out(point_fit, summary):
    fits = json.load(open(os.path.join(PROVIDED, 'leave_one_pair_out.json'), encoding='utf-8'))
    pairs = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
    add_check('seven fits: all pairs and without each of c1 to c6', 'leave_one_pair_out.json',
              ', '.join(sorted(fits)), 'None, c1, ..., c6', sorted(fits) == sorted(['None'] + pairs))
    largest_difference = max(abs(fits['None'][name] - point_fit['contrasts'][name]) for name in point_fit['contrasts'])
    add_check('the all-pairs fit equals the registered fit', 'leave_one_pair_out.json + point_fit.json',
              'largest difference %.2g' % largest_difference, '0', largest_difference < 1e-12)

    lines = F.read_lines(os.path.join(PROVIDED, 'leave_one_pair_out.txt'))
    printed = F.numbers_in(F.find_row(lines, 'all pairs'))
    recomputed_text = '%s %s' % (fixed(fits['None']['interaction'], 3, True), fixed(fits['None']['equivalence'], 3, True))
    printed_text = '%s %s' % (fixed(printed[0], 3, True), fixed(printed[1], 3, True))
    add_check('report row: all pairs', 'leave_one_pair_out.txt', recomputed_text, printed_text,
              recomputed_text == printed_text)
    for pair in pairs:
        printed = F.numbers_in(F.find_row(lines, 'without ' + pair))
        recomputed_text = '%s %s' % (fixed(fits[pair]['interaction'], 3, True), fixed(fits[pair]['equivalence'], 3, True))
        printed_text = '%s %s' % (fixed(printed[-2], 3, True), fixed(printed[-1], 3, True))
        add_check('report row: without %s' % pair, 'leave_one_pair_out.txt', recomputed_text, printed_text,
                  recomputed_text == printed_text)
    for name, label in (('interaction', 'interaction  across deletions:'), ('equivalence', 'feature buys across deletions:')):
        values = [fits[pair][name] for pair in pairs]
        lowest_pair = pairs[int(np.argmin(values))]
        highest_pair = pairs[int(np.argmax(values))]
        printed = F.numbers_in(F.find_row(lines, label))
        recomputed_text = '%s to %s' % (fixed(min(values), 3, True), fixed(max(values), 3, True))
        printed_text = '%s to %s' % (fixed(printed[0], 3, True), fixed(printed[1], 3, True))
        add_check('report range over the six deletions: %s' % name, 'leave_one_pair_out.txt', recomputed_text,
                  printed_text, recomputed_text == printed_text)
        summary[name]['leave_one_pair_out_min'] = float(min(values))
        summary[name]['leave_one_pair_out_min_without'] = lowest_pair
        summary[name]['leave_one_pair_out_max'] = float(max(values))
        summary[name]['leave_one_pair_out_max_without'] = highest_pair
    for name in ('rare_unrel', 'rare_matched', 'ordinary'):
        values = [fits[pair][name] for pair in pairs]
        summary[name]['leave_one_pair_out_min'] = float(min(values))
        summary[name]['leave_one_pair_out_min_without'] = pairs[int(np.argmin(values))]
        summary[name]['leave_one_pair_out_max'] = float(max(values))
        summary[name]['leave_one_pair_out_max_without'] = pairs[int(np.argmax(values))]
    return fits


# ============================================================================ the 243-setting run
def check_rem_243():
    rows = read_jsonl(os.path.join(PROVIDED, 'rem_like_data.jsonl'))
    keys = ['g', 'w', 'u', 'c', 'nfeat']
    grid = list(itertools.product(*[R.GRID[key] for key in keys]))
    settings_are_the_grid = (sorted(row['i'] for row in rows) == list(range(len(grid))))
    for row in rows:
        if row['pars'] != dict(zip(keys, grid[row['i']])):
            settings_are_the_grid = False
    fitted = [row for row in rows if row.get('fitted') and row.get('err') is None]
    add_check('243 settings, the grid g x w x u x c x nfeat in order, every one fitted', 'rem_like_data.jsonl',
              '%d settings, %d fitted' % (len(rows), len(fitted)), '243', settings_are_the_grid and len(fitted) == 243)

    # the four columns: (name, scoring, the field of that scoring in the saved file)
    #   fitted: the unequal-variance fit of the simulated ratings (interaction, equivalence)
    #   direct: read off the evidence (interaction, buys); used only by the report-row checks below
    # rem_243_summary.csv holds the fitted scores only.
    column_definitions = [('fitted lure-set effect', 'fitted', 'interaction'),
                          ('fitted feature effect', 'fitted', 'equivalence'),
                          ('direct lure-set effect', 'direct', 'interaction'),
                          ('direct feature effect', 'direct', 'buys')]
    columns = {}
    summary_rows = []
    for name, scoring, field in column_definitions:
        values = np.array([row[scoring][field] for row in fitted], dtype=float)
        columns[name] = values
        if scoring == 'fitted':
            summary_rows.append({'scoring_and_quantity': name, 'settings': len(values),
                                 'above_zero': int(np.sum(values > 0)), 'mean': float(values.mean()),
                                 'median': float(np.median(values)), 'minimum': float(values.min()),
                                 'maximum': float(values.max())})

    # the report rows, read by label
    lines = F.read_lines(os.path.join(PROVIDED, 'rem_like_data.txt'))
    header = F.numbers_in(F.find_line_containing(lines, 'parameter combinations'))
    add_check('report header: settings and simulated participants per setting', 'rem_like_data.txt',
              '%d, 100' % len(fitted), '%d, %d' % (header[0], header[1]),
              header[0] == len(fitted) and header[1] == 100)
    for heading, name_prefix, quantity in (('WHAT THE FEATURE BUYS', 'feature effect', 'feature'),
                                           ('INTERACTION', 'lure-set effect', 'lure set')):
        for row_label, scoring in (('read off the log-odds', 'direct'), ('fitted like the data', 'fitted')):
            values = columns['%s %s' % (scoring, name_prefix)]
            printed = F.numbers_in(F.find_row(lines, row_label, after=heading))   # mean, low, high, percent
            recomputed_text = 'mean %s range %s to %s positive %d%%' % (
                fixed(values.mean(), 3, True), fixed(values.min(), 3, True), fixed(values.max(), 3, True),
                round(100 * (values > 0).mean()))
            printed_text = 'mean %s range %s to %s positive %d%%' % (
                fixed(printed[0], 3, True), fixed(printed[1], 3, True), fixed(printed[2], 3, True), printed[3])
            add_check('243-setting report row: %s, %s' % (heading, row_label), 'rem_like_data.txt + .jsonl',
                      recomputed_text, printed_text, recomputed_text == printed_text)

    # the model side, computed again by rem_like_data.py (outputs/rem_like_data_model_side.csv)
    model_side_file = os.path.join(OUTPUTS, 'rem_like_data_model_side.csv')
    if os.path.exists(model_side_file):
        largest_difference = 0.0
        with open(model_side_file, newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                largest_difference = max(largest_difference,
                                         abs(float(row['direct_lure_set_effect_rerun']) - float(row['direct_lure_set_effect_saved'])),
                                         abs(float(row['direct_feature_effect_rerun']) - float(row['direct_feature_effect_saved'])))
        add_check('REM evidence of the 243 settings computed again (direct scores, rem_like_data.py)',
                  'outputs/rem_like_data_model_side.csv', 'largest difference %.2g' % largest_difference, '0',
                  largest_difference < 1e-12)
    return summary_rows


# ============================================================================ new fits with MemToolbox2
def compare_reruns(point_fit, leave_one_pair_out_fits):
    rerun_file = os.path.join(OUTPUTS, 'point_fit.json')
    if os.path.exists(rerun_file):
        rerun = json.load(open(rerun_file, encoding='utf-8'))
        largest = max(abs(rerun['contrasts'][name] - point_fit['contrasts'][name]) for name in point_fit['contrasts'])
        add_check('new registered fit (outputs/point_fit.json) against the saved fit', 'point_fit.json',
                  'largest difference %.2g; log likelihood %.4f' % (largest, rerun['loglike']),
                  'log likelihood %.4f' % point_fit['loglike'], largest < RERUN_TOLERANCE, 'rerun')
    rerun_file = os.path.join(OUTPUTS, 'leave_one_pair_out.json')
    if os.path.exists(rerun_file):
        rerun = json.load(open(rerun_file, encoding='utf-8'))
        largest = 0.0
        for fit_name, values in leave_one_pair_out_fits.items():
            for name in values:
                largest = max(largest, abs(rerun[fit_name][name] - values[name]))
        add_check('new leave-one-pair-out fits against the saved fits', 'leave_one_pair_out.json',
                  'largest difference %.2g' % largest, '0', largest < RERUN_TOLERANCE, 'rerun')
    rerun_file = os.path.join(OUTPUTS, 'rem_like_data.jsonl')
    if os.path.exists(rerun_file):
        saved = {row['i']: row for row in read_jsonl(os.path.join(PROVIDED, 'rem_like_data.jsonl'))}
        largest = 0.0
        compared = 0
        for row in read_jsonl(rerun_file):
            if row.get('fitted') and row['i'] in saved:
                compared += 1
                for name in ('interaction', 'equivalence'):
                    largest = max(largest, abs(row['fitted'][name] - saved[row['i']]['fitted'][name]))
        add_check('new 243-setting fits against the saved fits', 'rem_like_data.jsonl',
                  '%d settings compared, largest difference %.2g' % (compared, largest), '0', largest < RERUN_TOLERANCE,
                  'rerun')


def write_csv(path, rows, columns=None):
    columns = columns or list(rows[0])
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: (repr(value) if isinstance(value, float) else value) for name, value in row.items()})


def main():
    point_fit, criteria, trial_counts, number_of_participants = check_point_fit()
    summary = check_participant_bootstrap(point_fit, len(criteria), number_of_participants, trial_counts)
    check_item_bootstrap(summary)
    leave_one_pair_out_fits = check_leave_one_pair_out(point_fit, summary)
    rem_243_rows = check_rem_243()
    compare_reruns(point_fit, leave_one_pair_out_fits)

    write_csv(os.path.join(OUTPUTS, 'provided_checks.csv'), checks)
    summary_rows = [summary[name] for name, _, _ in QUANTITIES]
    write_csv(os.path.join(OUTPUTS, 'registered_fit_summary.csv'), summary_rows)
    write_csv(os.path.join(OUTPUTS, 'rem_243_summary.csv'), rem_243_rows)

    lines = ['The saved d_a analysis of the face study (data/provided/), checked', '=' * 66, '',
             'fitted model: never-studied faces N(0, 1); %s; criteria %s'
             % ('; '.join('%s mean %.4f SD %.4f' % (name, values[0], values[1])
                          for name, values in point_fit['cells'].items()),
                ', '.join('%.3f' % value for value in criteria)),
             'trials pooled over %d participants: %s' % (number_of_participants,
                                                        ', '.join('%s %d' % item for item in trial_counts.items())),
             '']
    for row in summary_rows:
        lines.append('%s' % row['description'])
        lines.append('  estimate %+.5f; over participants [%+.3f, %+.3f] (%d resamples, two-sided p = %.4f); '
                     'over faces [%+.3f, %+.3f] (p = %.4f); leaving out one pair: '
                     '%+.5f (without %s) to %+.5f (without %s)'
                     % (row['estimate'], row['participants_ci_low'], row['participants_ci_high'],
                        row['participants_resamples'], row['participants_p_two_sided'], row['items_ci_low'],
                        row['items_ci_high'], row['items_p_two_sided'], row['leave_one_pair_out_min'],
                        row['leave_one_pair_out_min_without'],
                        row['leave_one_pair_out_max'], row['leave_one_pair_out_max_without']))
    lines.append('')
    for row in rem_243_rows:
        lines.append('243-setting run, %-24s above zero %3d of %d; median %+.3f; range %+.3f to %+.3f'
                     % (row['scoring_and_quantity'] + ':', row['above_zero'], row['settings'], row['median'],
                        row['minimum'], row['maximum']))
    lines += ['', 'checks (%d): %d agree, %d do not' % (len(checks), sum(1 for row in checks if row['agree'] == 'yes'),
                                                        sum(1 for row in checks if row['agree'] != 'yes'))]
    for row in checks:
        lines.append('  [%s] %s: %s (printed %s)' % (row['agree'], row['check'], row['recomputed'], row['printed']))
    report = '\n'.join(lines) + '\n'
    with open(os.path.join(OUTPUTS, 'check_provided.txt'), 'w', encoding='utf-8') as handle:
        handle.write(report)
    print(report)
    # A saved number that does not agree stops the module; a new MemToolbox2 fit that differs from the saved fit
    # is reported (the module's numbers always come from the saved fits).
    for row in checks:
        if row['note'] == 'rerun' and row['agree'] != 'yes':
            print('NOTE: the new MemToolbox2 fit differs from the saved fit by more than %g: %s' % (RERUN_TOLERANCE,
                                                                                                 row['check']))
    if any(row['agree'] != 'yes' and row['note'] != 'rerun' for row in checks):
        sys.exit('check_provided.py: at least one saved number does not agree; see outputs/provided_checks.csv')


if __name__ == '__main__':
    main()
