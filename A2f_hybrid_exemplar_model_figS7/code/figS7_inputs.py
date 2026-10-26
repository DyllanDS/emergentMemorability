"""
figS7_inputs.py: the readers that stats_figS7.py and make_figureS7.py share (module A2f).

Model results of this module (each from this module's run in outputs/ when the file exists, otherwise
from the saved run in data/paper_run/, which the reader says on screen):
    model_results()     core.json: parameters, basic simulation (A) and designed lists (D) of the
                        five model variants; the paper's hybrid model is 'Hybrid GCM, list' and the
                        paper's GCM is 'GCM'
    sweep_table()       sweep.csv: the parameter sweep (26 settings with gamma above 0, plus gamma 0)
                        for the four reference populations
    bylinskii_table()   bylinskii.csv: the Bylinskii context simulation and its 19-setting sweep

REM reference values beside the hybrid model in SI Appendix, Section 2 and Fig. S7. They come from the
saved REM runs of two sibling modules (read by row label):
    rem_basic()           the row labeled 'paper values' (the Fig. 1 values: g 0.65, w 49, u 0.30,
                          copy 0.60) of
                          A2a_rem_parameter_sweep_figS2/data/paper_run/rem_parameter_sweep.csv:
                          Spearman rho between memorability and crowding, hit and false alarm
                          rates and d′
    rem_designed_lists()  CELL 2, at the Fig. 1 values, of
                          A2d_list_context_figS5/data/paper_run/list_designed.txt: hit rate,
                          false alarm rate and d′ with 0, 2, 4, 8 and 16 close neighbors studied
"""
import sys
sys.dont_write_bytecode = True
import json
import os
import re

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths

PATHS = release_paths.module_paths(__file__)
HYBRID_MODEL = 'Hybrid GCM, list'        # the paper's hybrid model: rarity from the participant's own study list
GCM_MODEL = 'GCM'                        # the paper's GCM: the same script with gamma = 0
REM_SWEEP_FILE = os.path.join(release_paths.find_module('A2a'), 'data', 'paper_run', 'rem_parameter_sweep.csv')
REM_DESIGNED_FILE = os.path.join(release_paths.find_module('A2d'), 'data', 'paper_run', 'list_designed.txt')


def source_file(file_name, quiet=False):
    """(path, mode): outputs/<file_name> when this module's run wrote it ('rerun'), otherwise
    data/paper_run/<file_name> ('saved_run'), with a message on screen."""
    rerun_path = os.path.join(PATHS['outputs'], file_name)
    if os.path.exists(rerun_path):
        return rerun_path, 'rerun'
    saved_path = os.path.join(PATHS['paper_run'], file_name)
    if not quiet:
        print('NOTE: outputs/%s does not exist; reading the saved run data/paper_run/%s'
              % (file_name, file_name))
    return saved_path, 'saved_run'


def model_results(quiet=False):
    """(results of core.json as a dictionary, mode)."""
    path, mode = source_file('core.json', quiet)
    with open(path, encoding='utf-8') as handle:
        return json.load(handle), mode


def sweep_table(quiet=False):
    """(sweep.csv as a table, mode)."""
    path, mode = source_file('sweep.csv', quiet)
    return pd.read_csv(path, float_precision='round_trip'), mode


def bylinskii_table(quiet=False):
    """(bylinskii.csv as a table, mode)."""
    path, mode = source_file('bylinskii.csv', quiet)
    return pd.read_csv(path, float_precision='round_trip'), mode


def rem_basic():
    """The REM sweep's row at the Fig. 1 values (label 'paper values ...'): a dictionary with rho
    (Spearman, memorability vs crowding), hit, false_alarm and dprime, plus the row's label."""
    table = pd.read_csv(REM_SWEEP_FILE, float_precision='round_trip')
    rows = table[table['label'].str.startswith('paper values')]
    if len(rows) != 1:
        raise ValueError('expected one row labeled "paper values" in %s' % REM_SWEEP_FILE)
    row = rows.iloc[0]
    if (row['g'], row['w'], row['u'], row['c']) != (0.65, 49, 0.3, 0.6):
        raise ValueError('the row labeled "paper values" in the REM sweep does not have the Fig. 1 parameters')
    return dict(label=row['label'], rho=float(row['rhoItem']), hit=float(row['meanHit']),
                false_alarm=float(row['meanFA']), dprime=float(row['dprime']))


def rem_designed_lists():
    """REM on the designed lists (CELL 2 of list_designed.txt, at the Fig. 1 values), as a table with
    one row per number of neighbors: similar, hit, false_alarm, dprime (3 decimals, as printed)."""
    with open(REM_DESIGNED_FILE, encoding='utf-8') as handle:
        text = handle.read().replace('\r', '')
    block = text.split('CELL 2:')[1].split('0 -> 16')[0]
    if not re.search(r'g=0\.65 w=49 u=0\.30 c=0\.60', block):
        raise ValueError('CELL 2 of %s is not the cell at the Fig. 1 values' % REM_DESIGNED_FILE)
    rows = []
    # columns of the table: similar items, crowding, hit, FA, d-prime, criterion
    for match in re.finditer(r'^\s+(\d+)\s+(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$', block, re.M):
        rows.append(dict(similar=int(match.group(1)), hit=float(match.group(3)),
                         false_alarm=float(match.group(4)), dprime=float(match.group(5))))
    table = pd.DataFrame(rows)
    if list(table['similar']) != [0, 2, 4, 8, 16]:
        raise ValueError('unexpected rows in CELL 2 of %s' % REM_DESIGNED_FILE)
    return table


def designed_table(variant_results):
    """The designed-list block (D) of one model variant as a table: similar, hit, fa, hits_minus_fa,
    dprime, criterion (one row per number of neighbors)."""
    return pd.DataFrame(variant_results['D'])
