"""
accuracy_link.py: how the size of the distinctiveness effect in the REM parameter sweep tracks
overall accuracy (SI Appendix, Section 2, REM Parameters, and Fig. S2b).

What it does
------------
The REM parameter sweep (rem_parameter_sweep.m) runs 32 simulations ("cells"). For each cell it saves
  rhoItem : Spearman correlation, across 1,600 items, between item memorability (hit rate minus
            false alarm rate) and summed similarity (negated mean Euclidean distance to the other
            items)
  dprime  : overall d′ of the simulated participants (from the mean hit and false alarm rates)
This script answers two questions:
  1. How closely does the size of the item-level relationship track overall accuracy?
     Pearson r between d′ and rhoItem over all 32 cells (the r = 0.71 of SI Appendix, Section 2).
  2. How noisy is the standard Shiffrin and Steyvers cell? Six of the 32 cells use exactly the same
     parameters (g 0.40, w 20, u 0.30, c 0.70, list 40, criterion 1.0); they differ only in the
     random draw (the "-0.55 to -0.51" of SI Appendix, Section 2). Their values are listed.
It also records where the cell with the Fig. 1 values (g 0.65, w 49, c 0.60, labeled "paper values"
in the csv) falls among the 32 cells.

Reads   outputs/rem_parameter_sweep.csv (this run of rem_parameter_sweep.m), or with --paper-run
        data/paper_run/rem_parameter_sweep.csv (the saved run the paper reports)
Writes  outputs/accuracy_link.json (with --paper-run: outputs/accuracy_link_paper_run.json) and
        prints every value. The keys that start with fig1_values_ describe the cell with the Fig. 1
        values.

Run (from the repository root):
    python A2a_rem_parameter_sweep_figS2/code/accuracy_link.py
    python A2a_rem_parameter_sweep_figS2/code/accuracy_link.py --paper-run
"""
import argparse
import csv
import json
import os
import sys

from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (the module's folders, found from this file's location)

parser = argparse.ArgumentParser(description='r between d′ and the item-level correlation over the REM sweep.')
parser.add_argument('--paper-run', action='store_true',
                    help='read the saved run in data/paper_run/ instead of this run in outputs/')
arguments = parser.parse_args()

paths = release_paths.module_paths(__file__)
if arguments.paper_run:
    DATA_FILE = os.path.join(paths['paper_run'], 'rem_parameter_sweep.csv')
    OUTPUT_FILE = os.path.join(paths['outputs'], 'accuracy_link_paper_run.json')
    SOURCE = 'data/paper_run/rem_parameter_sweep.csv (the saved run the paper reports)'
else:
    DATA_FILE = os.path.join(paths['outputs'], 'rem_parameter_sweep.csv')
    OUTPUT_FILE = os.path.join(paths['outputs'], 'accuracy_link.json')
    SOURCE = 'outputs/rem_parameter_sweep.csv (this run of rem_parameter_sweep.m)'
if not os.path.exists(DATA_FILE):
    sys.exit('accuracy_link.py: %s is missing; run code/rem_parameter_sweep.m first '
             '(python A2a_rem_parameter_sweep_figS2/code/run_module.py does it)' % SOURCE)
print('reading', SOURCE)

# ---------------------------------------------------------------- read the 32 cells
cells = []
with open(DATA_FILE) as handle:
    for row in csv.DictReader(handle):
        cell = {'label': row['label']}
        for column in ['g', 'w', 'u', 'c', 'nStudy', 'crit', 'rhoItem', 'pItem', 'rFeat', 'pFeat',
                       'meanHit', 'meanFA', 'dprime']:
            cell[column] = float(row[column])
        cells.append(cell)
assert len(cells) == 32, 'expected 32 cells, found %d' % len(cells)

# ---------------------------------------------------------------- question 1: accuracy link
all_dprime = [cell['dprime'] for cell in cells]
all_rho = [cell['rhoItem'] for cell in cells]

r_all, p_all = stats.pearsonr(all_dprime, all_rho)
print("Pearson r between d′ and item-level rho, all 32 cells: %+.3f (p = %.2e), r squared %.2f"
      % (r_all, p_all, r_all ** 2))

# ---------------------------------------------------------------- question 2: the six standard runs
standard_parameters = (0.40, 20, 0.30, 0.70, 40, 1.0)
standard_runs = []
for cell in cells:
    parameters = (cell['g'], cell['w'], cell['u'], cell['c'], cell['nStudy'], cell['crit'])
    if all(abs(a - b) < 1e-9 for a, b in zip(parameters, standard_parameters)):
        standard_runs.append(cell)
print('\nRuns at the standard Shiffrin and Steyvers parameters: %d' % len(standard_runs))
for cell in standard_runs:
    print('  %-32s item rho %+.3f' % (cell['label'], cell['rhoItem']))
standard_item = [cell['rhoItem'] for cell in standard_runs]
print('  item level %.3f to %.3f, mean %.3f' % (min(standard_item), max(standard_item),
                                                sum(standard_item) / len(standard_item)))

# ---------------------------------------------------------------- where the Fig. 1 values cell falls
fig1_cell = None
for cell in cells:
    if cell['label'].startswith('paper values'):
        fig1_cell = cell
rank_of_fig1_cell = 1
for cell in cells:
    if cell['rhoItem'] < fig1_cell['rhoItem']:
        rank_of_fig1_cell += 1
dprime_sorted = sorted(all_dprime)
median_dprime = (dprime_sorted[15] + dprime_sorted[16]) / 2
print("\nFig. 1 values cell: item rho %.3f is number %d of 32 from the most negative; d′ %.2f "
      "(median d′ over 32 cells %.2f)" % (fig1_cell['rhoItem'], rank_of_fig1_cell,
                                          fig1_cell['dprime'], median_dprime))

# the four most negative cells and the most accurate and least accurate cells
by_rho = sorted(cells, key=lambda cell: cell['rhoItem'])
print('Most negative item rho:')
for cell in by_rho[:4]:
    print("  %-32s %+.3f  d′ %.2f  hit %.3f" % (cell['label'], cell['rhoItem'], cell['dprime'],
                                                cell['meanHit']))
by_dprime = sorted(cells, key=lambda cell: cell['dprime'])
print("Lowest d′: %s (%.2f, rho %.3f); highest d′: %s (%.2f, rho %.3f)" % (
    by_dprime[0]['label'], by_dprime[0]['dprime'], by_dprime[0]['rhoItem'],
    by_dprime[-1]['label'], by_dprime[-1]['dprime'], by_dprime[-1]['rhoItem']))
weakest_rank_of_lowest_dprime = 1
for cell in cells:
    if cell['rhoItem'] > by_dprime[0]['rhoItem']:
        weakest_rank_of_lowest_dprime += 1
print('  the lowest-d′ cell is number %d of 32 from the weakest' % weakest_rank_of_lowest_dprime)

results = {
    'source': SOURCE + ' (columns dprime, rhoItem)',
    'r_dprime_rho_all32': r_all, 'p_dprime_rho_all32': p_all,
    'r_squared_all32': r_all ** 2,
    'standard_cell_runs': [{'label': c['label'], 'rhoItem': c['rhoItem']} for c in standard_runs],
    'standard_item_min': min(standard_item), 'standard_item_max': max(standard_item),
    'standard_item_mean': sum(standard_item) / len(standard_item),
    'fig1_values_rhoItem': fig1_cell['rhoItem'], 'fig1_values_rank_most_negative_of_32': rank_of_fig1_cell,
    'fig1_values_dprime': fig1_cell['dprime'], 'median_dprime_32': median_dprime,
}
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as handle:
    json.dump(results, handle, indent=1)
print('\nwrote outputs/%s' % os.path.basename(OUTPUT_FILE))
