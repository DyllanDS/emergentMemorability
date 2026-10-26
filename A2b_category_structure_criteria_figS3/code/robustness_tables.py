"""
robustness_tables.py: readers of the tables that bylinskii_robustness.m writes, and the grouping of its
18 cells (simulations). Used by stats_figS3.py, make_figureS3.py and check_against_paper_run.py of this
module.

    read_cells(path)
        the 18 cells of bylinskii_robustness.csv, in file order, as dicts with the label and the
        numeric columns (critWithin, critAcross, catFeatures, nCat, nPerCat, rho, pRho, simCorr,
        hitWithin, faWithin, hitAcross, faAcross, dWithin, dAcross)
    read_items(path)
        the per-target file bylinskii_robustness_items.csv, as {cell number: dict of arrays} for the
        target items the grid correlated (tested at least once as old and once as new in both
        contexts), in the order of the file:
          memorability_within       hit rate minus false alarm rate in the within-category context
          memorability_across       the same in the across-category context
          summed_similarity_within  minus the mean Euclidean distance from the target to the other
                                    items of its own category
          summed_similarity_across  minus the mean Euclidean distance from the target to all items of
                                    the set (the target itself included, at distance 0)
          category                  the target's category (1 to the number of categories)
        plus n_targets (targets in the cell) and n_left_out (targets not tested in both contexts)
    targets_per_category(cell)
        75 targets per category, or a quarter of the items per category in the "items per category"
        cells (bylinskii_robustness.m: round(nPerCat / 4), which rounds 37.5 up to 38)
    has_fig2a_structure(cell)
        True for the cells with the category structure and criteria of Fig. 2a: criteria 20 / 5,
        10 category-defining features, 21 categories of 300 items
    has_equal_criteria(cell)
        True when both contexts use the same criterion
    group_of(cell)
        the factor the cell varies, as the five groups of Fig. S3 name it
    spearman_rho_and_p(x, y)
        Spearman rho and its two-sided p (t approximation with n - 2 degrees of freedom; scipy's
        spearmanr). MATLAB's corr uses the same t approximation when the data hold ties and n is 10
        or more, which is always the case here (hit and false alarm rates are counts over trials)
    pearson_r_and_p(x, y)
        Pearson r and its two-sided p (t test with n - 2 degrees of freedom)
"""
import csv
import math

import numpy as np
from scipy import stats

NUMERIC_COLUMNS = ['critWithin', 'critAcross', 'catFeatures', 'nCat', 'nPerCat', 'rho', 'pRho', 'simCorr',
                   'hitWithin', 'faWithin', 'hitAcross', 'faAcross', 'dWithin', 'dAcross']

# The five groups of Fig. S3, from top to bottom (group_of() gives the group of a cell from its label).
GROUP_NAMES = ['Unequal Criteria (Within / Across)', 'Equal Criteria', 'Category-Defining Features',
               'Categories', 'Items per Category']


def read_cells(path):
    """The cells of bylinskii_robustness.csv, one dict per row, in file order."""
    cells = []
    with open(path, newline='') as handle:
        for row in csv.DictReader(handle):
            cell = {'label': row['label']}
            for column in NUMERIC_COLUMNS:
                cell[column] = float(row[column])
            cells.append(cell)
    return cells


def read_items(path):
    """Per cell: memorability in each context, summed similarity in each context and category of the
    targets that bylinskii_robustness.m correlated (the same arithmetic as the MATLAB script)."""
    rows_by_cell = {}
    with open(path, newline='') as handle:
        for row in csv.DictReader(handle):
            cell_number = int(row['cell_number'])
            if cell_number not in rows_by_cell:
                rows_by_cell[cell_number] = []
            rows_by_cell[cell_number].append(row)

    items_by_cell = {}
    for cell_number in sorted(rows_by_cell):
        memorability_within = []
        memorability_across = []
        summed_similarity_within = []
        summed_similarity_across = []
        category = []
        left_out = 0
        for row in rows_by_cell[cell_number]:
            within_old_trials = int(row['within_old_trials'])
            within_new_trials = int(row['within_new_trials'])
            across_old_trials = int(row['across_old_trials'])
            across_new_trials = int(row['across_new_trials'])
            tested_in_both_contexts = (within_old_trials > 0 and within_new_trials > 0
                                       and across_old_trials > 0 and across_new_trials > 0)
            if not tested_in_both_contexts:
                left_out += 1
                continue
            # memorability = hit rate minus false alarm rate, as in bylinskii_robustness.m
            within_hit_rate = int(row['within_hits']) / within_old_trials
            within_false_alarm_rate = int(row['within_false_alarms']) / within_new_trials
            across_hit_rate = int(row['across_hits']) / across_old_trials
            across_false_alarm_rate = int(row['across_false_alarms']) / across_new_trials
            memorability_within.append(within_hit_rate - within_false_alarm_rate)
            memorability_across.append(across_hit_rate - across_false_alarm_rate)
            summed_similarity_within.append(float(row['summed_similarity_within']))
            summed_similarity_across.append(float(row['summed_similarity_across']))
            category.append(int(row['category']))
        items_by_cell[cell_number] = {
            'memorability_within': np.array(memorability_within),
            'memorability_across': np.array(memorability_across),
            'summed_similarity_within': np.array(summed_similarity_within),
            'summed_similarity_across': np.array(summed_similarity_across),
            'category': np.array(category),
            'n_targets': len(rows_by_cell[cell_number]),
            'n_left_out': left_out,
        }
    return items_by_cell


def targets_per_category(cell):
    """75 targets per category; in the 'items per category' cells a quarter of the items per category,
    rounded half away from zero as MATLAB's round (150 / 4 = 37.5 -> 38)."""
    if cell['label'].startswith('items per category'):
        return int(math.floor(cell['nPerCat'] / 4.0 + 0.5))
    return 75


def has_fig2a_structure(cell):
    """True for a cell with the category structure and criteria of Fig. 2a."""
    return (cell['critWithin'] == 20 and cell['critAcross'] == 5 and cell['catFeatures'] == 10
            and cell['nCat'] == 21 and cell['nPerCat'] == 300)


def has_equal_criteria(cell):
    """True when both contexts use the same decision criterion."""
    return cell['critWithin'] == cell['critAcross']


def group_of(cell):
    """The factor a cell varies, named as the five groups of Fig. S3."""
    label = cell['label']
    if label.startswith('reference') or label.startswith('crit '):
        if has_equal_criteria(cell):
            return 'Equal Criteria'
        return 'Unequal Criteria (Within / Across)'
    if label.startswith('category-defining features'):
        return 'Category-Defining Features'
    if label.startswith('categories'):
        return 'Categories'
    if label.startswith('items per category'):
        return 'Items per Category'
    raise ValueError('cell label %r belongs to no group' % label)


def spearman_rho_and_p(x_values, y_values):
    """Spearman rho and its two-sided p (t approximation with n - 2 degrees of freedom)."""
    result = stats.spearmanr(x_values, y_values)
    return float(result.statistic), float(result.pvalue)


def pearson_r_and_p(x_values, y_values):
    """Pearson r and its two-sided p (t test with n - 2 degrees of freedom)."""
    result = stats.pearsonr(x_values, y_values)
    return float(result.statistic), float(result.pvalue)
