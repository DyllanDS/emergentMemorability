"""
check_against_paper_run.py: compares this run with the saved run in data/paper_run/, and writes
outputs/numbers.csv.

WHAT THIS DOES
    1. Compares the report of this run (outputs/continuous_rem_spose.txt, written by continuous_rem_spose.m)
       with the saved report of the run the paper reports (data/paper_run/continuous_rem_spose.txt):
       every number of the results table, row by row (rows found by their noise label) and column by
       column as printed; the design line, the storage line and the surprisal line (found by their
       first words); then the whole file, line by line.
    2. Checks that the point values the Python steps compute from the item files
       (outputs/continuous_rem_bootstrap_ci.csv) round to the same printed values (mean hit and
       false alarm rates, rho with local crowding, designed d′ change).
       Both comparisons go to outputs/equality_with_paper_run.csv.
    3. Writes outputs/numbers.csv: one row per ledger item of this module (20 items: L6-083 to L6-089,
       L6-091 to L6-097, L6-099, L6-100 and L6-301 to L6-304), each with the value this run computed and
       whether it agrees with the printed value (shared/python/numbers_table.py judges the match).
    Stops with an error (after writing both files) if a number of the saved report's results table,
    design line, storage line or surprisal line is not reproduced, or if a ledger number does not
    match. The line-by-line count of the whole report is written to the table and does not stop the
    script.

READS   outputs/continuous_rem_spose.txt, outputs/continuous_rem_settings.csv, outputs/concept_subset.csv,
        outputs/continuous_rem_bootstrap_ci.csv and .txt, outputs/figure_data/FigS6.csv,
        data/paper_run/continuous_rem_spose.txt,
        ../A2d_list_context_figS5/data/paper_run/list_designed.txt (CELL 2),
        ../01_rem_simulation_fig1/data/paper_run/fig1_paper_run.mat (the run of Fig. 1)
WRITES  outputs/equality_with_paper_run.csv, outputs/numbers.csv

Run (from the repository root; seconds; after the other steps):
    python A2e_measured_representations_figS6/code/check_against_paper_run.py
"""
import csv
import os
import re
import sys

import numpy as np
import scipy.io
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402
import numbers_table  # noqa: E402

paths = release_paths.module_paths(__file__)
OUTPUT_FOLDER = paths['outputs']
THIS_REPORT = os.path.join(OUTPUT_FOLDER, 'continuous_rem_spose.txt')
PAPER_REPORT = os.path.join(paths['paper_run'], 'continuous_rem_spose.txt')
REM_DESIGNED_FILE = os.path.join(release_paths.find_module('A2d'), 'data', 'paper_run', 'list_designed.txt')
FIG1_RUN_FILE = os.path.join(release_paths.find_module('01'), 'data', 'paper_run', 'fig1_paper_run.mat')

NOISE_VALUES_SHOWN = ['0.02', '0.03', '0.05']
TABLE_COLUMNS = ['hit', 'FA', "d'", 'rho vs global', 'rho vs local', "designed d'", 'hits-FA']
HEADER_LINE_STARTS = {'design line': r'^\d+ THINGS concepts', 'storage line': r'^storage ',
                      'surprisal line': r'^surprisal vs global crowding'}

SIMULATION_SCRIPT = 'code/continuous_rem_spose.m'
BOOTSTRAP_SCRIPT = 'code/continuous_rem_spose.m; code/bootstrap_ci.py'
FIGURE_SCRIPT = 'code/make_figureS6.py'
FIG1_RUN_SCRIPT = ('01_rem_simulation_fig1/code/fig1_remSim.m (its saved run, read by '
                   'code/check_against_paper_run.py)')
WHITE_BAR_SCRIPT = ('A2d_list_context_figS5/code/list_designed.m (its saved run, read by '
                    'code/check_against_paper_run.py)')


# ------------------------------------------------------------------------- reading
def thousands(number):
    """1600 -> '1,600' (counts in the notes are written as in the paper)."""
    return '{:,}'.format(int(number))


def read_report_table(path):
    """The results table of a report, by noise label: {'0.02': ['0.957', '0.040', ...], ...} (the 7
    numbers after the noise label, as printed)."""
    table = {}
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 8 and re.match(r'^0\.\d\d$', parts[0]):
                table[parts[0]] = parts[1:]
    return table


def read_report_line(path, pattern):
    """The first line of a report that matches the pattern (by its first words), without the newline."""
    with open(path) as handle:
        for line in handle:
            if re.match(pattern, line):
                return line.rstrip('\n')
    return None


def read_settings():
    """outputs/continuous_rem_settings.csv: {setting: list of numbers}."""
    settings = {}
    with open(os.path.join(OUTPUT_FOLDER, 'continuous_rem_settings.csv'), newline='') as handle:
        for row in csv.DictReader(handle):
            settings[row['setting']] = [float(value) for value in row['value'].split()]
    return settings


def read_bootstrap_rows():
    """outputs/continuous_rem_bootstrap_ci.csv by noise label, numbers as floats."""
    rows = {}
    with open(os.path.join(OUTPUT_FOLDER, 'continuous_rem_bootstrap_ci.csv'), newline='') as handle:
        for row in csv.DictReader(handle):
            rows[row['noise_fraction_of_sd']] = {name: float(value) for name, value in row.items()}
    return rows


def read_figure_data(file_name):
    """The rows of a figure_data file (one per bar)."""
    with open(os.path.join(paths['figure_data'], file_name), newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def describe_plotted_bars(figure_rows):
    """'gray 0.02 -0.414 [-0.502, -0.310]; ...; white -0.132 (no interval)' from a figure_data file."""
    parts = []
    for row in figure_rows:
        if row['bar'] == 'gray':
            parts.append('gray %s %.3f [%.3f, %.3f]' % (row['noise_fraction_of_sd'], float(row['value']),
                                                        float(row['ci_low']), float(row['ci_high'])))
        else:
            parts.append('white %.3f (no interval)' % float(row['value']))
    return '; '.join(parts)


def read_white_bar():
    """The white bar of Fig. S6: the d′ change of list_designed.txt CELL 2 (module A2d)."""
    white_change = None
    inside_cell_2 = False
    with open(REM_DESIGNED_FILE) as handle:
        for line in handle:
            if line.startswith('CELL 2'):
                inside_cell_2 = True
            if inside_cell_2 and '0 -> 16 similar items' in line:
                white_change = float(re.search(r'd-prime\s+([-+]?\d*\.\d+)', line).group(1))
                break
    if white_change is None:
        sys.exit('check_against_paper_run.py: the white-bar value could not be read from %s' % REM_DESIGNED_FILE)
    return white_change


def read_fig1_run_correlation():
    """The run of Fig. 1 (module 01, its saved run): Spearman rho across the simulated items between
    memorability (hit rate minus false alarm rate) and the negated mean Euclidean distance to the other
    items, over the items with both values, as in fig1_measures.m. Returns rho and the number of items."""
    figure_data = scipy.io.loadmat(FIG1_RUN_FILE, squeeze_me=True, struct_as_record=False)['figureData']
    memorability = np.asarray(figure_data.hitFARate, dtype=float)
    mean_distance = np.asarray(figure_data.avgDistances, dtype=float)
    has_both = np.isfinite(memorability) & np.isfinite(mean_distance)
    rho = spearmanr(-mean_distance[has_both], memorability[has_both])[0]
    return float(rho), int(np.sum(has_both))


# ------------------------------------------------------------------------- 1 and 2. equality
def compare_with_paper_run(bootstrap_rows):
    """Rows of the equality table and the number of printed values that differ."""
    equality_rows = []
    differences = 0

    paper_table = read_report_table(PAPER_REPORT)
    this_table = read_report_table(THIS_REPORT)
    for noise in sorted(paper_table):
        for column_index, column in enumerate(TABLE_COLUMNS):
            paper_value = paper_table[noise][column_index]
            this_value = this_table.get(noise, [''] * len(TABLE_COLUMNS))[column_index]
            equal = paper_value == this_value
            differences += 0 if equal else 1
            equality_rows.append(['report table (MATLAB)', noise, column, paper_value, this_value,
                                  'yes' if equal else 'no'])

    for label, pattern in HEADER_LINE_STARTS.items():
        paper_line = read_report_line(PAPER_REPORT, pattern)
        this_line = read_report_line(THIS_REPORT, pattern)
        equal = paper_line == this_line
        differences += 0 if equal else 1
        equality_rows.append(['report ' + label, '', 'whole line', paper_line, this_line, 'yes' if equal else 'no'])

    with open(PAPER_REPORT) as handle:
        paper_lines = handle.read().splitlines()
    with open(THIS_REPORT) as handle:
        this_lines = handle.read().splitlines()
    differing_lines = 0
    for line_index in range(max(len(paper_lines), len(this_lines))):
        paper_line = paper_lines[line_index] if line_index < len(paper_lines) else None
        this_line = this_lines[line_index] if line_index < len(this_lines) else None
        if paper_line != this_line:
            differing_lines += 1
    equality_rows.append(['report, whole file', '', 'lines that differ', '%d lines' % len(paper_lines),
                          '%d lines, %d differ' % (len(this_lines), differing_lines),
                          'yes' if differing_lines == 0 else 'no'])

    # the Python point values (from the item files) against the printed report
    python_columns = [('hit', 'mean_hit_rate'), ('FA', 'mean_false_alarm_rate'),
                      ('rho vs local', 'spearman_memorability_vs_local_crowding'),
                      ("designed d'", 'designed_d_prime_change')]
    for noise in sorted(paper_table):
        for column, bootstrap_column in python_columns:
            printed = paper_table[noise][TABLE_COLUMNS.index(column)]
            computed = bootstrap_rows[noise][bootstrap_column]
            equal = '%.3f' % computed == printed
            differences += 0 if equal else 1
            equality_rows.append(['item files (Python, bootstrap_ci.py)', noise, column, printed, repr(computed),
                                  'yes' if equal else 'no'])

    with open(os.path.join(OUTPUT_FOLDER, 'equality_with_paper_run.csv'), 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['what', 'noise_fraction_of_sd', 'column', 'paper_run', 'this_run', 'equal'])
        for row in equality_rows:
            writer.writerow(row)
    print('equality with the saved run: %d comparisons, %d differ (whole report: %d differing lines)'
          % (len(equality_rows), differences, differing_lines))
    print('wrote outputs/equality_with_paper_run.csv')
    return differences


# ------------------------------------------------------------------------- 3. numbers.csv
def write_numbers(bootstrap_rows):
    settings = read_settings()
    with open(os.path.join(OUTPUT_FOLDER, 'concept_subset.csv'), newline='') as handle:
        subset_rows = list(csv.DictReader(handle))
    number_of_concepts_used = len(set(row['concept_row_in_embedding'] for row in subset_rows))
    figure_rows = read_figure_data('FigS6.csv')
    white_change = read_white_bar()
    fig1_rho, fig1_items = read_fig1_run_correlation()
    with open(os.path.join(OUTPUT_FOLDER, 'continuous_rem_bootstrap_ci.txt')) as handle:
        bootstrap_text = handle.read()
    interval_level = int(re.search(r'(\d+)% BOOTSTRAP CONFIDENCE INTERVALS', bootstrap_text).group(1))
    number_of_resamples = int(re.search(r'(\d+) resamples', bootstrap_text).group(1))

    n_features = int(settings['n_features'][0])
    n_concepts_in_embedding = int(settings['n_concepts_in_embedding'][0])
    n_neighbours = int(settings['n_nearest_neighbors_crowding'][0])
    n_critical = int(settings['n_critical_items_designed'][0])
    n_participants = int(settings['n_participants_per_list_designed'][0])
    n_close = int(settings['n_close_neighbors_designed'][-1])
    n_experiments = int(settings['n_experiments_between_items'][0])
    noise_fractions = settings['noise_fractions']

    rho_by_noise = [bootstrap_rows[noise]['spearman_memorability_vs_local_crowding'] for noise in NOISE_VALUES_SHOWN]
    p_by_noise = [bootstrap_rows[noise]['spearman_p'] for noise in NOISE_VALUES_SHOWN]
    n_items_by_noise = [int(bootstrap_rows[noise]['n_items']) for noise in NOISE_VALUES_SHOWN]
    change_by_noise = [bootstrap_rows[noise]['designed_d_prime_change'] for noise in NOISE_VALUES_SHOWN]
    rho_parts = []
    change_parts = []
    for noise in NOISE_VALUES_SHOWN:
        row = bootstrap_rows[noise]
        rho_parts.append('noise %s %.4f' % (noise, row['spearman_memorability_vs_local_crowding']))
        change_parts.append('noise %s %.4f [%.3f, %.3f]' % (noise, row['designed_d_prime_change'],
                                                            row['designed_ci_low'], row['designed_ci_high']))
    rho_text = ', '.join(rho_parts)
    change_text = ', '.join(change_parts)
    gray_rows = [row for row in figure_rows if row['bar'] == 'gray']
    white_rows = [row for row in figure_rows if row['bar'] == 'white']
    shown_noise = [float(row['noise_fraction_of_sd']) for row in gray_rows]
    gray_upper_ends = [float(row['ci_high']) for row in gray_rows]
    for n_items in n_items_by_noise:
        if n_items != number_of_concepts_used:
            sys.exit('check_against_paper_run.py: %d concepts in the correlation, %d in concept_subset.csv'
                     % (n_items, number_of_concepts_used))

    table = numbers_table.NumbersTable(script=SIMULATION_SCRIPT)

    table.add('L6-083', n_features, 'rerun',
              note='SPoSE dimensions read by the simulation from shared/things/spose_embedding_49d_sorted.txt (%s '
                   'THINGS concepts x %d dimensions; Hebart et al., 2020; CC BY 4.0), used as the features of each '
                   'simulated item' % (thousands(n_concepts_in_embedding), n_features))
    table.add('L6-084', number_of_concepts_used, 'rerun',
              note="THINGS concepts in the simulation: %s of the %s, drawn with randperm as the first random draw "
                   "after rng(1, 'twister') (list in outputs/concept_subset.csv); the density of each SPoSE "
                   "dimension over these concepts (ksdensity, %d points) sets the rarity of a value"
                   % (thousands(number_of_concepts_used), thousands(n_concepts_in_embedding),
                      settings['density_grid_points'][0]))
    table.add('L6-085', noise_fractions, 'rerun',
              note='retention noise levels run by the simulation, each a fraction of a SPoSE dimension\'s standard '
                   'deviation across the %s concepts (noise SD of dimension j = fraction x SD of dimension j); '
                   'Fig. S6 shows 0.02, 0.03 and 0.05' % thousands(number_of_concepts_used))
    design_text = 'u %.2f; copy %.2f; %d studied; %d old + %d new tested; criterion %.1f' % (
        settings['storage_u'][0], settings['copy_c'][0], settings['n_studied'][0], settings['n_tested_old'][0],
        settings['n_tested_new'][0], settings['criterion'][0])
    table.add('L6-086', design_text, 'rerun',
              note='the settings of this run (outputs/continuous_rem_settings.csv), the values of Fig. 1: storage u, '
                   'copy c, concepts studied, old and new concepts tested per experiment, criterion; %s simulated '
                   'study-test experiments per noise level, so each concept is tested about %d times as old and %d '
                   'times as new' % (thousands(n_experiments),
                                     round(n_experiments * settings['n_tested_old'][0] / number_of_concepts_used),
                                     round(n_experiments * settings['n_tested_new'][0] / number_of_concepts_used)))
    table.add('L6-087', [min(rho_by_noise), max(rho_by_noise)], 'rerun', script=BOOTSTRAP_SCRIPT,
              note='Spearman rho across the %s concepts between memorability (hit rate minus false alarm rate) and '
                   'local crowding (minus the mean correlation distance to the %d nearest concepts), lowest and '
                   'highest over the three noise levels of Fig. S6 (%s); 95%% intervals from 10,000 resamples of the '
                   'concepts are in tests.csv (T29)' % (thousands(number_of_concepts_used), n_neighbours, rho_text))
    table.add('L6-088', [n_close, -max(change_by_noise), -min(change_by_noise)], 'rerun', script=BOOTSTRAP_SCRIPT,
              note='close neighbors on the crowded designed list, then the smallest and largest loss in d′ from 0 '
                   'to %d close neighbors over the three noise levels of Fig. S6 (%d critical concepts, %d simulated '
                   'participants per list, d′ from the mean hit and false alarm rates); d′ changes with the 95%% '
                   'bootstrap intervals drawn in Fig. S6: %s' % (n_close, n_critical, n_participants, change_text))
    table.add('L6-089', -white_change, 'rerun', script=WHITE_BAR_SCRIPT,
              note='loss in d′ from 0 to 16 similar items on designed lists for REM with generated features at the '
                   'Fig. 1 parameters (d′ change %.3f, the white bar of Fig. S6), read from '
                   '../A2d_list_context_figS5/data/paper_run/list_designed.txt (CELL 2), the saved run of module A2d'
                   % white_change)
    table.add('L6-091', describe_plotted_bars(figure_rows), 'rerun', script=FIGURE_SCRIPT,
              note='Fig. S6 as drawn by code/make_figureS6.py (outputs/figures/FigS6_continuous_feature_rem.png); '
                   'plotted values in outputs/figure_data/FigS6.csv: gray bars, d′ change from 0 to 16 close '
                   'neighbors at noise 0.02, 0.03 and 0.05 with their 95% bootstrap intervals; white bar, REM with '
                   'generated features at the Fig. 1 parameters, CELL 2 of list_designed.txt of module A2d (no '
                   'interval)')
    table.add('L6-092', [n_features, number_of_concepts_used], 'rerun',
              note='features per simulated item (the SPoSE dimensions) and THINGS concepts in the simulation '
                   '(outputs/concept_subset.csv)')
    table.add('L6-093', shown_noise, 'rerun', script=FIGURE_SCRIPT,
              note='retention noise levels of the gray bars of Fig. S6 (outputs/figure_data/FigS6.csv), each a '
                   'fraction of a SPoSE dimension\'s standard deviation across the concepts')
    table.add('L6-094', n_neighbours, 'rerun',
              note='nearest concepts in the local crowding measure: minus the mean correlation distance (1 minus the '
                   'Pearson r of two concepts\' SPoSE vectors) to the %d nearest of the %s concepts'
                   % (n_neighbours, thousands(number_of_concepts_used)))
    table.add('L6-095', [min(rho_by_noise), max(rho_by_noise), max(p_by_noise), number_of_concepts_used], 'rerun',
              script=BOOTSTRAP_SCRIPT,
              note='Spearman rho between memorability and local crowding, lowest and highest over noise 0.02, 0.03 '
                   'and 0.05; the largest of the three two-sided p values (scipy.stats.spearmanr, t approximation), '
                   '%.2g; %s concepts in each correlation' % (max(p_by_noise), thousands(number_of_concepts_used)))
    table.add('L6-096', fig1_rho, 'rerun', script=FIG1_RUN_SCRIPT,
              note='Spearman rho across the %s simulated items of the run of Fig. 1 between memorability (hit rate '
                   'minus false alarm rate) and the negated mean Euclidean distance to the other items (the -0.55 of '
                   'Methods, Basic REM Simulation of Memorability), read from '
                   '../01_rem_simulation_fig1/data/paper_run/fig1_paper_run.mat, the saved run of module 01'
                   % thousands(fig1_items))
    table.add('L6-097', [n_critical, n_close], 'rerun',
              note='%d critical concepts, each on a list with 0 and a list with %d close neighbors (the 2nd to %dth '
                   'nearest concepts), %d simulated participants per list; the nearest concept is the lure (never '
                   'studied); the other list items come from the farther half of the concepts'
                   % (n_critical, n_close, n_close + 1, n_participants))
    rate_parts = []
    for noise in NOISE_VALUES_SHOWN:
        row = bootstrap_rows[noise]
        rate_parts.append('noise %s: false alarms to the lure %.3f -> %.3f, hits %.3f -> %.3f'
                          % (noise, row['false_alarm_rate_0_neighbors'], row['false_alarm_rate_16_neighbors'],
                             row['hit_rate_0_neighbors'], row['hit_rate_16_neighbors']))
    false_alarm_text = '; '.join(rate_parts)
    table.add('L6-099', false_alarm_text, 'rerun', script=BOOTSTRAP_SCRIPT,
              note='mean false alarm rate to the lure (the nearest concept to the critical concept) and hit rate of '
                   'the critical concept across the %d critical concepts, with 0 and with 16 close neighbors, at each '
                   'noise level of Fig. S6; the intervals of the changes are in tests.csv (T29)' % n_critical)
    table.add('L6-100', interval_level, 'rerun', script='code/bootstrap_ci.py',
              note='confidence level of the error bars of Fig. S6: %d%% percentile bootstrap intervals, %s '
                   'resamples of the critical concepts, numpy default_rng(20261010); intervals from 10,000 resamples '
                   '(seed 20261013) are in tests.csv (T29)' % (interval_level, thousands(number_of_resamples)))
    table.add('L6-301', shown_noise, 'rerun',
              note='retention noise levels of the reported results (the gray bars of Fig. S6 and the ranges of SI '
                   'Appendix, Section 2), each set in settings.noiseFractions of code/continuous_rem_spose.m; no '
                   'value of the noise is fitted')
    table.add('L6-302', 'lure: the nearest concept, never studied; close neighbors: the 2nd to %dth nearest'
              % (n_close + 1), 'rerun',
              note='designed lists of code/continuous_rem_spose.m: the other concepts are ranked by correlation '
                   'distance to the critical concept; the nearest is tested as the lure and is on neither study '
                   'list, the next %d are the close neighbors, and the rest of each 40-concept list comes from the '
                   'farther half of the ranking' % n_close)
    table.add('L6-303', describe_plotted_bars(white_rows), 'rerun', script=FIGURE_SCRIPT,
              note='the white outlined bar of Fig. S6, labeled "Fig. 1 Values" (outputs/figure_data/FigS6.csv): REM '
                   'with generated features at the Fig. 1 parameters on designed lists, CELL 2 of '
                   '../A2d_list_context_figS5/data/paper_run/list_designed.txt, the saved run of module A2d')
    gray_bars_below_zero = sum(1 for upper_end in gray_upper_ends if upper_end < 0)
    table.add('L6-304', '%d of %d gray-bar intervals below 0 (upper ends %s)'
              % (gray_bars_below_zero, len(gray_upper_ends), ', '.join('%.3f' % value for value in gray_upper_ends)),
              'rerun', script=FIGURE_SCRIPT,
              note='the gray bars of Fig. S6 (outputs/figure_data/FigS6.csv): the d′ change from 0 to %d close '
                   'neighbors at noise 0.02, 0.03 and 0.05 and the upper ends of their 95%% bootstrap intervals (%s '
                   'resamples of the critical concepts)' % (n_close, thousands(number_of_resamples)))
    table.write(OUTPUT_FOLDER)
    return table


def main():
    for needed in (THIS_REPORT, PAPER_REPORT, REM_DESIGNED_FILE, FIG1_RUN_FILE):
        if not os.path.exists(needed):
            sys.exit('check_against_paper_run.py: %s is missing' % needed)
    bootstrap_rows = read_bootstrap_rows()
    differences = compare_with_paper_run(bootstrap_rows)
    table = write_numbers(bootstrap_rows)
    not_matching = [row['ledger_id'] for row in table.rows if row['match'] == 'no']
    if not_matching:
        print('ledger numbers that do not match the paper at its printed precision: %s' % ', '.join(not_matching))
    if differences > 0 or not_matching:
        sys.exit('check_against_paper_run.py: %d values of the saved report are not reproduced; %d ledger '
                 'numbers do not match (see outputs/equality_with_paper_run.csv and outputs/numbers.csv)'
                 % (differences, len(not_matching)))
    print('every value of the saved report is reproduced; every ledger number matches')


if __name__ == '__main__':
    main()
