"""
sweep_tables.py: readers of the tables the REM parameter sweep (rem_parameter_sweep.m) writes. Used by
stats_figS2.py, check_against_paper_run.py and make_figureS2.py of this module.

    read_cells(path)
        the 32 cells of rem_parameter_sweep.csv, in file order, as dicts with the label and the numeric
        columns (g, w, u, c, nStudy, crit, rhoItem, pItem, rFeat, pFeat, meanHit, meanFA, dprime)
    read_items(path)
        the per-item file rem_parameter_sweep_items.csv, as
        {cell number: {'memorability': array, 'negated_mean_distance': array,
                       'n_items': items in the cell, 'old_trials': total old trials,
                       'new_trials': total new trials}}
        for the items tested at least once as old and once as new, which are the items
        rem_parameter_sweep.m correlates:
          memorability           hit rate minus false alarm rate
          negated_mean_distance  minus the mean Euclidean distance from the item to the other items,
                                 the "summed similarity" of the Fig. S2 caption: a HIGHER value means an
                                 item that is closer to the others on average (more similar, less
                                 distinctive).
                                 A negative correlation with memorability therefore means that
                                 distinctive items are remembered better.
    spearman_rho_and_p(memorability, negated_mean_distance)
        Spearman rho and its two-sided p (t approximation with n - 2 degrees of freedom; scipy's
        spearmanr, the same formula as the pItem column of rem_parameter_sweep.csv)
    standard_value_cells(cells)
        the six cells with the Shiffrin and Steyvers values (g 0.40, 20 features, storage 0.30, copy
        0.70, list 40, criterion 1.0)
"""
import csv

import numpy as np
from scipy import stats

NUMERIC_COLUMNS = ['g', 'w', 'u', 'c', 'nStudy', 'crit', 'rhoItem', 'pItem', 'rFeat', 'pFeat', 'meanHit', 'meanFA',
                   'dprime']
STANDARD_VALUES = {'g': 0.40, 'w': 20, 'u': 0.30, 'c': 0.70, 'nStudy': 40, 'crit': 1.0}


def read_cells(path):
    """The cells of rem_parameter_sweep.csv, one dict per row, in file order."""
    cells = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            cell = {'label': row['label']}
            for column in NUMERIC_COLUMNS:
                cell[column] = float(row[column])
            cells.append(cell)
    return cells


def read_items(path):
    """Per cell: memorability and negated mean distance of the items the sweep correlated."""
    counts_by_cell = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            cell_number = int(row['cell_number'])
            if cell_number not in counts_by_cell:
                counts_by_cell[cell_number] = []
            counts_by_cell[cell_number].append((int(row['old_trials']), int(row['hits']), int(row['new_trials']),
                                                int(row['false_alarms']), float(row['neg_mean_distance'])))
    items_by_cell = {}
    for cell_number in sorted(counts_by_cell):
        memorability = []
        negated_mean_distance = []
        total_old_trials = 0
        total_new_trials = 0
        for old_trials, hits, new_trials, false_alarms, item_negated_mean_distance in counts_by_cell[cell_number]:
            total_old_trials += old_trials
            total_new_trials += new_trials
            if old_trials > 0 and new_trials > 0:
                # the same arithmetic as rem_parameter_sweep.m: hit rate minus false alarm rate
                hit_rate = hits / old_trials
                false_alarm_rate = false_alarms / new_trials
                memorability.append(hit_rate - false_alarm_rate)
                negated_mean_distance.append(item_negated_mean_distance)
        items_by_cell[cell_number] = {'memorability': np.array(memorability),
                                      'negated_mean_distance': np.array(negated_mean_distance),
                                      'n_items': len(counts_by_cell[cell_number]),
                                      'old_trials': total_old_trials, 'new_trials': total_new_trials}
    return items_by_cell


def spearman_rho_and_p(memorability, negated_mean_distance):
    """Spearman rho between the two lists and its two-sided p (t approximation, n - 2 df)."""
    result = stats.spearmanr(memorability, negated_mean_distance)
    return float(result.statistic), float(result.pvalue)


def standard_value_cells(cells):
    """The cells whose six parameters are the Shiffrin and Steyvers values."""
    found = []
    for cell in cells:
        same = True
        for column, value in STANDARD_VALUES.items():
            if abs(cell[column] - value) > 1e-9:
                same = False
        if same:
            found.append(cell)
    return found
