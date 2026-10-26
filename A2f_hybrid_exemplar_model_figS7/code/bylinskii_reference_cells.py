"""
bylinskii_reference_cells.py: the Bylinskii context simulation of the hybrid model, run exactly
for the three settings behind the printed numbers, plus the per-target memorability of the
reference setting (for its bootstrap interval).

SI Appendix, Section 2 prints, for the hybrid model with rarity defined by each participant's own
study list: rho = 0.48 between an item's memorability within its category and across categories,
and rho = 0.15 to 0.74 over 19 settings. The full 19-setting run (times four reference
populations, plus 7 GCM settings) is code/03_bylinskii.py (about 17 min). This script runs, with
the same code and the same seeds, only:

  1. the width calibration of 03_bylinskii.py, exactly as its main block runs it (seed 11: the
     categorized item set, the median inter-item distance, then the calibration of the GCM and of
     the four reference populations world, set, list and mix, in that order);
  2. three settings of the [list] sweep, each with its own seed zlib.crc32(setting label), through
     03_bylinskii.run_cell(), exactly as the full run does:
        reference [list]                         the printed rho = 0.48
        rare-feature boost gamma = 1.0 [list]    the lowest of the 19 settings (0.15)
        equal criterion FA 0.3 [list]            the highest of the 19 settings (0.74)
  3. the steps of run_cell() for the reference [list] setting once more, keeping each target's
     hit and false alarm counts within category and across categories (the Spearman correlation
     of the two memorability scores is the printed 0.48; the script stops if it is not identical
     to step 2).

Inputs:  none (the items are generated); code/03_bylinskii.py and code/hgcm.py
Outputs: outputs/bylinskii_calibration.csv          calibrated width (as a fraction of the median
                                                     inter-item distance) and across-category d′ per model
         outputs/bylinskii_reference_cells.csv      the three settings, the columns of bylinskii.csv
         outputs/bylinskii_reference_list_targets.csv  one row per target of the reference [list] setting
Seeds:   11 (calibration); zlib.crc32(setting label) per setting
Runtime: about 2 min
Run from the repository root:  python A2f_hybrid_exemplar_model_figS7/code/bylinskii_reference_cells.py
"""
import sys
sys.dont_write_bytecode = True
import importlib
import os
import zlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_FOLDER)
sys.path.insert(0, os.path.join(CODE_FOLDER, '..', '..', 'shared', 'python'))
import release_paths
import hgcm
bylinskii = importlib.import_module('03_bylinskii')      # the file name starts with a digit

PATHS = release_paths.module_paths(__file__)
CALIBRATION_SEED = 11                                    # the seed of 03_bylinskii.py's main block
REFERENCE_LABEL = 'reference [list]'
CHECKED_LABELS = ['reference [list]', 'rare-feature boost gamma = 1.0 [list]', 'equal criterion FA 0.3 [list]']


def calibrate_widths():
    """The calibration of 03_bylinskii.py's main block, the same calls in the same order.
    Returns the fraction of the median distance chosen for each (rare-feature boost on, reference
    population) and the across-category d′ each fraction reached."""
    generator = np.random.default_rng(CALIBRATION_SEED)
    items, category, is_target = bylinskii.build(generator, bylinskii.REF)
    median_distance = bylinskii.median_distance(generator, items)
    reference_parameters = dict(bylinskii.REF, set_freq=hgcm.set_freq(items))
    fraction = {}
    dprime_reached = {}
    # the plain GCM first (gamma = 0), then the hybrid model for each reference population
    gcm_parameters = dict(reference_parameters, gamma=0.0)
    fraction[(False, 'set')], dprime_reached['GCM'] = bylinskii.calibrate(generator, items, is_target,
                                                                         gcm_parameters, median_distance)
    for population in bylinskii.REFS:
        hybrid_parameters = dict(reference_parameters, ref=population)
        fraction[(True, population)], dprime_reached[population] = bylinskii.calibrate(
            generator, items, is_target, hybrid_parameters, median_distance)
    return fraction, dprime_reached


def setting_change(label):
    """The parameter change of one setting, taken from 03_bylinskii.cells() (the list the full run uses)."""
    for cell_label, change in bylinskii.cells():
        if cell_label == label:
            return change
    raise KeyError('no setting called %r in 03_bylinskii.cells()' % label)


def target_memorability(change, fraction, seed):
    """The steps of 03_bylinskii.run_cell() up to the cross-context correlation, with the seed given
    (run_cell uses zlib.crc32(label)). Returns the per-target table and the Spearman rho."""
    generator = np.random.default_rng(seed)
    parameters = dict(bylinskii.REF, **change)
    items, category, is_target = bylinskii.build(generator, parameters)
    parameters['set_freq'] = hgcm.set_freq(items)
    width_fraction = fraction[(parameters['gamma'] > 0, parameters['ref'])]
    parameters['c'] = 1.0 / (width_fraction * bylinskii.median_distance(generator, items))
    within_targets, within_foils, across_targets, across_foils = bylinskii.contexts(
        generator, items, category, is_target, parameters)
    # criteria at the target false alarm rates, from the foils' familiarity in each context
    within_foil_familiarity = np.concatenate([familiarity.ravel() for _, familiarity in within_foils])
    across_foil_familiarity = np.concatenate([familiarity.ravel() for _, familiarity in across_foils])
    if parameters['equal'] is None:
        criterion_within = np.quantile(within_foil_familiarity, 1 - parameters['fa_within'])
        criterion_across = np.quantile(across_foil_familiarity, 1 - parameters['fa_across'])
    else:
        criterion_within = np.quantile(across_foil_familiarity, 1 - parameters['equal'])
        criterion_across = criterion_within
    n_items = len(items)
    counts_within = bylinskii.counts(within_targets, within_foils, criterion_within, n_items)
    counts_across = bylinskii.counts(across_targets, across_foils, criterion_across, n_items)
    usable = (is_target & (counts_within[1] > 0) & (counts_within[3] > 0)
              & (counts_across[1] > 0) & (counts_across[3] > 0))
    memorability_within = bylinskii.mem(counts_within, usable)
    memorability_across = bylinskii.mem(counts_across, usable)
    rho = float(spearmanr(memorability_within, memorability_across)[0])
    rows = []
    for item in np.flatnonzero(usable):
        rows.append(dict(target=int(item), category=int(category[item]),
                         hits_within=int(counts_within[0][item]), old_tests_within=int(counts_within[1][item]),
                         false_alarms_within=int(counts_within[2][item]), new_tests_within=int(counts_within[3][item]),
                         hits_across=int(counts_across[0][item]), old_tests_across=int(counts_across[1][item]),
                         false_alarms_across=int(counts_across[2][item]), new_tests_across=int(counts_across[3][item])))
    return pd.DataFrame(rows), rho


def main():
    print('Bylinskii context simulation: calibration and three settings, run exactly as in 03_bylinskii.py', flush=True)

    # ---- 1. calibration ----
    fraction, dprime_reached = calibrate_widths()
    calibration_rows = []
    for key, value in fraction.items():
        boosted, population = key
        model = population if boosted else 'GCM'
        calibration_rows.append(dict(model=model, rare_feature_boost=boosted, reference_population=population,
                                     fraction_of_median_distance=value, across_category_dprime=dprime_reached[model]))
        print('  calibration %-6s fraction %.2f of the median distance, across-category d\' %.4f'
              % (model, value, dprime_reached[model]), flush=True)
    pd.DataFrame(calibration_rows).to_csv(os.path.join(PATHS['outputs'], 'bylinskii_calibration.csv'), index=False)

    # ---- 2. the three settings through run_cell(), as the full run computes them ----
    setting_rows = []
    for label in CHECKED_LABELS:
        row = bylinskii.run_cell((label, setting_change(label), fraction))
        setting_rows.append(row)
        print('  %-40s rho = %+.6f' % (label, row['rho']), flush=True)
    settings = pd.DataFrame(setting_rows)
    settings.to_csv(os.path.join(PATHS['outputs'], 'bylinskii_reference_cells.csv'), index=False)

    # ---- 3. per-target memorability of the reference [list] setting ----
    reference_seed = zlib.crc32(REFERENCE_LABEL.encode())
    targets, reference_rho = target_memorability(setting_change(REFERENCE_LABEL), fraction, reference_seed)
    rho_from_run_cell = float(settings.loc[settings.cell == REFERENCE_LABEL, 'rho'].iloc[0])
    if reference_rho != rho_from_run_cell:
        raise RuntimeError('per-target table gives rho %.17g, run_cell gives %.17g' % (reference_rho, rho_from_run_cell))
    print('  per-target table: %d targets, rho %.6f (identical to run_cell)' % (len(targets), reference_rho), flush=True)
    targets.to_csv(os.path.join(PATHS['outputs'], 'bylinskii_reference_list_targets.csv'), index=False)
    print('done', flush=True)


if __name__ == '__main__':
    main()
