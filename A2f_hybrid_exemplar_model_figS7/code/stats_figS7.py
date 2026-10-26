"""
stats_figS7.py: the numbers of the hybrid-similarity exemplar model and the generalized context
model (GCM) in SI Appendix, Section 2 (A Hybrid-Similarity Exemplar Model) and Fig. S7, and in the
statements of the main text and the General Discussion that point to them, with their uncertainty.
Writes outputs/numbers.csv and outputs/tests.csv.

What it does
  1. Reads the model results through figS7_inputs.py:
       core.json      the basic simulation (A) and the designed lists (D) of the five model
                      variants; the paper's hybrid model is 'Hybrid GCM, list' and its GCM is 'GCM'
       sweep.csv      the parameter sweep (26 settings with gamma above 0), for four definitions
                      of rarity
       bylinskii.csv  the Bylinskii context simulation and its 19-setting sweep
     each from this module's run in outputs/ or, when a file is not in outputs/, from the saved run
     in data/paper_run/ (said on screen and in the mode column of numbers.csv). The REM values the
     paper prints beside the model (rho -0.61, d′ 1.61; Fig. S7b and c) come from the saved REM runs
     of modules A2a and A2d.
  2. Checks the model's rules against the paper's words: hgcm.familiarity() against the similarity
     rule written out with explicit loops, and the number of studied items that may share a value
     for it to count as rare ("three or fewer of the participant's 40 studied items").
  3. Checks that the per-item tables of the run (outputs/core_items_basic.csv and
     outputs/core_items_designed.csv, written by 01_core.py) give back core.json exactly.
  4. tests.csv, the uncertainty statistics of test T30:
       - 95% percentile bootstrap intervals (10,000 resamples, seed 20261013): over the 1,600
         simulated items for the item-level Spearman rho and the overall d′ of the hybrid model and
         of the GCM; over the 200 critical items for the change from 0 to 16 neighbors in hit rate,
         false alarm rate and d′; over the 1,575 targets for the Bylinskii cross-context rho;
       - the two sweeps as counts and ranges of settings, not as tests.
     No p values (a p value from simulated items reflects only how many items were simulated).
  5. numbers.csv: one row per ledger item of this module (29 rows) with the value the code produced
     and whether it agrees with the printed value (shared/python/numbers_table.py); the printed
     intervals come from step 4. Design facts the paper states in words, and the settings of the
     code (ADD-002 to ADD-006), are read from the scripts (their constants, default arguments and
     source text) and from the outputs.

Inputs:  outputs/ (above) or data/paper_run/; ../A2a_rem_parameter_sweep_figS2/data/paper_run/
         rem_parameter_sweep.csv; ../A2d_list_context_figS5/data/paper_run/list_designed.txt
Outputs: outputs/numbers.csv, outputs/tests.csv
Seed:    20261013 for every bootstrap and for the similarity-rule check (each bootstrap starts a new
         generator from it)
Runtime: about 25 seconds
Without the per-item tables (01_core.py not run) numbers.csv is written from the saved run with no
intervals, and tests.csv keeps only the rows that do not need them; the screen says so.
Run from the repository root:  python A2f_hybrid_exemplar_model_figS7/code/stats_figS7.py
"""
import sys
sys.dont_write_bytecode = True
import importlib
import inspect
import math
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_FOLDER)
sys.path.insert(0, os.path.join(CODE_FOLDER, '..', '..', 'shared', 'python'))
import release_paths
import numbers_table
import tests_table
import bootstrap
import hgcm
import figS7_inputs as inputs
core_script = importlib.import_module('01_core')          # file names that start with a digit
bylinskii_script = importlib.import_module('03_bylinskii')

PATHS = release_paths.module_paths(__file__)
SEED = bootstrap.RELEASE_SEED                             # 20261013
N_RESAMPLES = 10000
HYBRID = inputs.HYBRID_MODEL                              # 'Hybrid GCM, list'
GCM = inputs.GCM_MODEL                                    # 'GCM'
MODEL_NAMES = {HYBRID: 'hybrid-similarity exemplar model (gamma 2, rarity from the study list)',
               GCM: 'generalized context model (gamma 0)'}
BYLINSKII_REFERENCE = 'reference [list]'


# ============================================================================ small helpers
def signed(value, decimals=3):
    """'+0.263' or '-0.317' (never '-0.000')."""
    text = '%+.*f' % (decimals, value)
    if float(text) == 0:
        text = text.replace('-', '+')
    return text


def series(values):
    """'0.343, 0.332, 0.325' (three decimals)."""
    return ', '.join('%.3f' % value for value in values)


def read_source(script_name):
    """The text of one script of code/ (to read a setting the way the script states it)."""
    with open(os.path.join(CODE_FOLDER, script_name), encoding='utf-8') as handle:
        return handle.read()


def default_argument(function, name):
    """The default value of one argument of a function (the value the scripts run with)."""
    return inspect.signature(function).parameters[name].default


def interval(resampled_values):
    """The 2.5th and 97.5th percentiles of the resampled values."""
    low, high = bootstrap.percentile_interval(resampled_values)
    return float(low), float(high)


def designed_change(designed_table, column):
    """The change from 0 to 16 neighbors in one column (hit, fa or dprime) of a designed-list table."""
    return float(designed_table[column].iloc[-1] - designed_table[column].iloc[0])


# ============================================================================ 1. model results
results, core_mode = inputs.model_results()
hybrid = results[HYBRID]
gcm = results[GCM]
hybrid_designed = inputs.designed_table(hybrid)
gcm_designed = inputs.designed_table(gcm)
rem = inputs.rem_basic()
rem_designed = inputs.rem_designed_lists()
sweep, sweep_mode = inputs.sweep_table()
bylinskii, bylinskii_mode = inputs.bylinskii_table()
print('core.json: %s; sweep.csv: %s; bylinskii.csv: %s' % (core_mode, sweep_mode, bylinskii_mode))

# the 26-setting sweep of the paper's definition of rarity (the study list), settings with gamma > 0
sweep_list = sweep[(sweep['ref'] == 'list') & (sweep['gamma'] > 0)]
sweep_lowest = sweep_list.loc[sweep_list['rho_item'].idxmin()]
sweep_highest = sweep_list.loc[sweep_list['rho_item'].idxmax()]
sweep_n_negative = int((sweep_list['rho_item'] < 0).sum())
sweep_gamma0_list = sweep[(sweep['ref'] == 'list') & (sweep['gamma'] == 0)].iloc[0]
# rho at each weight on rare matches: the settings gamma 0, 1 and 3, and the reference setting (gamma 2, the
# first [list] setting of 02_sweep.py)
sweep_rho_by_gamma = [float(sweep.loc[(sweep['ref'] == 'list') & (sweep['gamma'] == gamma), 'rho_item'].iloc[0])
                      for gamma in (0.0, 1.0, 2.0, 3.0)]


def bylinskii_values():
    """The reference [list] rho, the lowest and the highest rho over the 19 [list] settings, the
    number above zero, and the mode. When outputs/bylinskii_reference_cells.csv exists, the three
    settings that bylinskii_reference_cells.py runs exactly give the values of those settings (after a
    full run they equal the rows of bylinskii.csv)."""
    list_rows = bylinskii[(bylinskii['model'] == 'hybrid') & (bylinskii['ref'] == 'list')]
    lowest_row = list_rows.loc[list_rows['rho'].idxmin()]
    highest_row = list_rows.loc[list_rows['rho'].idxmax()]
    reference_rho = float(list_rows.loc[list_rows['cell'] == BYLINSKII_REFERENCE, 'rho'].iloc[0])
    values = dict(reference=reference_rho,
                  lowest=float(lowest_row['rho']), lowest_cell=lowest_row['cell'],
                  highest=float(highest_row['rho']), highest_cell=highest_row['cell'],
                  n_settings=len(list_rows), n_positive=int((list_rows['rho'] > 0).sum()),
                  mode=bylinskii_mode, mode_note='')
    reference_cells_file = os.path.join(PATHS['outputs'], 'bylinskii_reference_cells.csv')
    if os.path.exists(reference_cells_file):
        reran = pd.read_csv(reference_cells_file, float_precision='round_trip')
        reran_rho = dict(zip(reran['cell'], reran['rho']))
        for key, label in (('reference', BYLINSKII_REFERENCE), ('lowest', values['lowest_cell']),
                           ('highest', values['highest_cell'])):
            if label in reran_rho:
                values[key] = float(reran_rho[label])
        if bylinskii_mode == 'saved_run':
            values['mode'] = 'rerun_and_saved_run'
            values['mode_note'] = ('; the reference, lowest and highest settings come from '
                                   'bylinskii_reference_cells.py and the count of settings from '
                                   'data/paper_run/bylinskii.csv')
    return values


bylinskii_numbers = bylinskii_values()


# ============================================================================ 2. the model's rules
def largest_distinctive_count(n_study, q):
    """The largest number of studied items that can share a value while the value still counts as
    distinctive. Found by running hgcm.ref_table() (rarity from the study list) on study lists of
    n_study items in which one value is carried by 1, 2, ..., n_study of the items."""
    largest = 0
    for count in range(1, n_study + 1):
        study_list = np.full((1, n_study, 1), 2, dtype=np.int16)   # one participant, one feature, value 2
        study_list[0, :count, 0] = 1                               # value 1 on 'count' of the studied items
        parameters = dict(ref='list', q=q, g=0.65, w=1)
        distinctive = hgcm.ref_table(parameters, study_list)       # [participant, feature, value]
        if distinctive[0, 0, 1]:
            largest = count
    return largest


def check_similarity_rule():
    """hgcm.familiarity() against the paper's rule written with explicit loops:
        familiarity = sum over the studied traces of exp(-c d) x exp(gamma m)
    where d is the Euclidean distance over the features the trace stored (rescaled to all w features,
    as hgcm.py does) and m is the number of stored features on which probe and trace share a
    distinctive value. A value is distinctive when fewer than q = 10% of the participant's 40 studied
    items (as studied, not as stored) carry it on that feature; the loops count this themselves rather
    than use hgcm.ref_table(). One simulated participant, 40 studied items, 10 studied and 10 new
    probes, the hybrid model's settings. Returns the largest relative difference."""
    generator = np.random.default_rng(SEED)
    n_features = core_script.BASE['w']
    g = core_script.BASE['g']
    n_study = core_script.BASE['n_study']
    q = core_script.VARIANTS[HYBRID]['q']
    items = hgcm.make_items(generator, n_study + 10, n_features, g)
    studied = items[None, :n_study]                                 # [participant, item, feature]
    traces, stored = hgcm.encode(generator, studied, core_script.BASE['u'], core_script.BASE['cenc'], g)
    probes = items[None, n_study - 10:]                             # the last 10 studied items, then 10 new ones
    parameters = dict(core_script.VARIANTS[HYBRID], c=hybrid['params']['c'], g=g, w=n_features)
    distinctive = hgcm.ref_table(parameters, studied)               # [participant, feature, value], used by the model
    model_familiarity = hgcm.familiarity(probes, traces, stored, parameters, distinctive)[0]
    largest_relative_difference = 0.0
    for probe_number in range(probes.shape[1]):
        probe = probes[0, probe_number]
        familiarity = 0.0
        for trace_number in range(n_study):
            trace = traces[0, trace_number]
            is_stored = stored[0, trace_number]
            n_stored = max(int(is_stored.sum()), 1)
            squared_difference = 0.0
            shared_distinctive_values = 0
            for feature in range(n_features):
                if not is_stored[feature]:
                    continue
                squared_difference += float(probe[feature] - trace[feature]) ** 2
                same_value = probe[feature] == trace[feature]
                # how many of the 40 studied items carry the probe's value on this feature
                items_with_value = 0
                for studied_item in range(n_study):
                    if studied[0, studied_item, feature] == probe[feature]:
                        items_with_value += 1
                value_is_distinctive = items_with_value / n_study < q
                if same_value and value_is_distinctive:
                    shared_distinctive_values += 1
            distance = math.sqrt(squared_difference / n_stored * n_features)
            similarity = math.exp(-parameters['c'] * distance) * math.exp(parameters['gamma'] * shared_distinctive_values)
            familiarity += similarity
        relative_difference = abs(float(model_familiarity[probe_number]) - familiarity) / familiarity
        largest_relative_difference = max(largest_relative_difference, relative_difference)
    return largest_relative_difference


similarity_rule_difference = check_similarity_rule()
print('similarity rule: hgcm.familiarity() and the explicit loops differ by at most %.1e (relative)'
      % similarity_rule_difference)
if similarity_rule_difference > 1e-9:
    raise RuntimeError('hgcm.familiarity() does not match the similarity rule written with explicit loops')
largest_count = largest_distinctive_count(core_script.BASE['n_study'], core_script.VARIANTS[HYBRID]['q'])


# ============================================================================ 3. per-item tables
def basic_item_rates(variant):
    """Hit rate, false alarm rate and crowding of each of the 1,600 items in the basic simulation of
    one model variant (outputs/core_items_basic.csv), plus the number of simulated experiments."""
    table = pd.read_csv(os.path.join(PATHS['outputs'], 'core_items_basic.csv'), float_precision='round_trip')
    rows = table[table['variant'] == variant].sort_values('item')
    hit_rate = rows['hits'].to_numpy() / rows['old_tests'].to_numpy()
    fa_rate = rows['false_alarms'].to_numpy() / rows['new_tests'].to_numpy()
    crowding = rows['crowding'].to_numpy()
    n_experiments = int(rows['old_tests'].sum()) // core_script.BASE['n_old']
    return hit_rate, fa_rate, crowding, n_experiments


def designed_item_rates(variant):
    """Hit and false alarm rate of each critical item (rows) at 0, 2, 4, 8 and 16 neighbors
    (columns), each over the simulated participants (outputs/core_items_designed.csv)."""
    table = pd.read_csv(os.path.join(PATHS['outputs'], 'core_items_designed.csv'))
    rows = table[table['variant'] == variant]
    levels = sorted(rows['n_similar'].unique())
    critical_items = list(dict.fromkeys(rows['critical_item']))      # in the order the script drew them
    hit_rate = np.zeros((len(critical_items), len(levels)))
    fa_rate = np.zeros((len(critical_items), len(levels)))
    for _, row in rows.iterrows():
        item_row = critical_items.index(row['critical_item'])
        level_column = levels.index(row['n_similar'])
        hit_rate[item_row, level_column] = row['hits'] / row['participants']
        fa_rate[item_row, level_column] = row['false_alarms'] / row['participants']
    n_participants = int(rows['participants'].iloc[0])
    return hit_rate, fa_rate, levels, n_participants


have_item_tables = (core_mode == 'rerun'
                    and os.path.exists(os.path.join(PATHS['outputs'], 'core_items_basic.csv'))
                    and os.path.exists(os.path.join(PATHS['outputs'], 'core_items_designed.csv')))
basic = {}
designed = {}
if have_item_tables:
    for variant in (HYBRID, GCM):
        hit_rate, fa_rate, crowding, n_experiments = basic_item_rates(variant)
        basic[variant] = dict(hit_rate=hit_rate, fa_rate=fa_rate, crowding=crowding, n_experiments=n_experiments)
        # the per-item table must give back the run's rho and d′ exactly
        rho_again = spearmanr(hit_rate - fa_rate, crowding)[0]
        dprime_again = float(hgcm.zc(hit_rate.mean()) - hgcm.zc(fa_rate.mean()))
        if abs(rho_again - results[variant]['A']['rho_item']) > 1e-12 or abs(dprime_again - results[variant]['A']['dprime']) > 1e-12:
            raise RuntimeError('core_items_basic.csv does not reproduce core.json for %s' % variant)
        hit_matrix, fa_matrix, levels, n_participants = designed_item_rates(variant)
        designed[variant] = dict(hit_rate=hit_matrix, fa_rate=fa_matrix, levels=levels, n_participants=n_participants)
        variant_designed = inputs.designed_table(results[variant])
        if (np.max(np.abs(hit_matrix.mean(axis=0) - variant_designed['hit'].to_numpy())) > 1e-12
                or np.max(np.abs(fa_matrix.mean(axis=0) - variant_designed['fa'].to_numpy())) > 1e-12):
            raise RuntimeError('core_items_designed.csv does not reproduce core.json for %s' % variant)
    print('per-item tables give back core.json (rho, d′ and designed-list rates)')
    n_items = len(basic[HYBRID]['crowding'])
    n_experiments = basic[HYBRID]['n_experiments']
    n_critical = designed[HYBRID]['hit_rate'].shape[0]
    n_participants_per_list = designed[HYBRID]['n_participants']
    neighbor_levels = designed[HYBRID]['levels']
    size_source = 'counted from the per-item tables of the run'
else:
    print('NOTE: the per-item tables are not in outputs/ (run code/01_core.py, about 3 min); '
          'numbers.csv takes the sizes from the code\'s settings, and tests.csv and numbers.csv have no bootstrap '
          'intervals (nan in numbers.csv)')
    n_items = core_script.BASE['n_db']
    n_experiments = default_argument(core_script.analysis_a, 'n_sim')
    n_critical = default_argument(core_script.analysis_d, 'n_crit')
    n_participants_per_list = default_argument(core_script.analysis_d, 'n_subj')
    neighbor_levels = list(default_argument(core_script.analysis_d, 'n_sim'))
    size_source = 'the default arguments of analysis_a and analysis_d in 01_core.py'


# ============================================================================ settings read from the scripts
core_source = read_source('01_core.py')
sweep_source = read_source('02_sweep.py')
bylinskii_source = read_source('03_bylinskii.py')
paper_seed = int(re.search(r'rng = np\.random\.default_rng\((\d+)\)', core_source).group(1))
calibration_fractions = re.search(r'for frac in \(([^)]*)\)', core_source).group(1)
sweep_runs = int(re.search(r'analysis_a\(rng, X, p, crowd, n_sim=(\d+)', sweep_source).group(1))
if 'default_rng(zlib.crc32(label.encode()))' in sweep_source:
    sweep_seed_rule = 'crc32(label)'
else:
    sweep_seed_rule = 'not found'
bylinskii_seed = int(re.search(r'default_rng\((\d+)\)', bylinskii_source).group(1))


# ============================================================================ 4. tests.csv (T30)
tests = tests_table.TestsTable(script='code/stats_figS7.py')
UNIT_ITEMS = 'simulated items (%s simulated experiments)' % format(n_experiments, ',')
NO_P = dict(df='', p='', p_type='none')
INTERVALS = {}      # (model variant or 'Bylinskii', statistic_name): (ci_low, ci_high), also printed in numbers.csv


def add_bootstrap_row(**fields):
    """One row with a 95% percentile bootstrap interval (seed 20261013, 10,000 resamples), no p value."""
    row = dict(test_id='T30', ci_method='percentile bootstrap', n_resamples=N_RESAMPLES, seed=SEED, **NO_P)
    row.update(fields)
    tests.add(**row)


# --- 4a. item-level rho and overall d′ of the basic simulation, bootstrap over the 1,600 items
def basic_statistics_for(variant):
    """A function of the resampled item positions that returns [Spearman rho, overall d′]."""
    hit_rate = basic[variant]['hit_rate']
    fa_rate = basic[variant]['fa_rate']
    crowding = basic[variant]['crowding']

    def statistics(chosen_items):
        memorability = hit_rate[chosen_items] - fa_rate[chosen_items]
        rho = spearmanr(memorability, crowding[chosen_items])[0]
        dprime = hgcm.zc(hit_rate[chosen_items].mean()) - hgcm.zc(fa_rate[chosen_items].mean())
        return [rho, dprime]
    return statistics


ITEM_RHO_TEST = ('Spearman correlation across the %s simulated items between memorability (hit rate minus '
                 'false alarm rate) and crowding (negated mean distance to the other items), %s')
for variant, ledger_ids in ((HYBRID, 'L6-107;L6-113'), (GCM, 'L6-112;L6-113')):
    rho = results[variant]['A']['rho_item']
    rho_fields = dict(ledger_ids=ledger_ids,
                      location='SI Appendix, Section 2, A Hybrid-Similarity Exemplar Model; Fig. S7a',
                      test=ITEM_RHO_TEST % (format(n_items, ','), MODEL_NAMES[variant]),
                      statistic_name='Spearman rho', statistic=rho, n=n_items, unit=UNIT_ITEMS,
                      effect_size_name='rho', effect_size=rho)
    dprime = results[variant]['A']['dprime']
    dprime_ledger_ids = 'L6-110;L6-119' if variant == HYBRID else 'L6-119'
    dprime_fields = dict(ledger_ids=dprime_ledger_ids,
                         location='SI Appendix, Section 2, A Hybrid-Similarity Exemplar Model',
                         test='Overall d′ of the basic simulation (from the mean hit and false alarm rates over the %s '
                              'items), %s' % (format(n_items, ','), MODEL_NAMES[variant]),
                         statistic_name='d′', statistic=dprime, n=n_items, unit=UNIT_ITEMS,
                         effect_size_name='d′', effect_size=dprime)
    if have_item_tables:
        print('bootstrap over items, basic simulation, %s (%d resamples)' % (variant, N_RESAMPLES), flush=True)
        resampled = bootstrap.percentile_bootstrap(basic_statistics_for(variant), n_items, N_RESAMPLES, SEED)
        rho_low, rho_high = interval(resampled[:, 0])
        dprime_low, dprime_high = interval(resampled[:, 1])
        INTERVALS[(variant, 'Spearman rho')] = (rho_low, rho_high)
        add_bootstrap_row(ci_low=rho_low, ci_high=rho_high, resampling_unit='simulated items', **rho_fields)
        add_bootstrap_row(ci_low=dprime_low, ci_high=dprime_high, resampling_unit='simulated items', **dprime_fields)


# --- 4b. designed lists, change from 0 to 16 neighbors, bootstrap over the critical items
def designed_statistics_for(variant):
    """A function of the resampled critical-item positions that returns the change from 0 to 16
    neighbors in [hit rate, false alarm rate, d′], d′ from the mean rates as analysis_d computes it."""
    hit_matrix = designed[variant]['hit_rate']
    fa_matrix = designed[variant]['fa_rate']
    lowest_rate = 1.0 / (2 * designed[variant]['n_participants'])      # the clamp of analysis_d

    def statistics(chosen_items):
        mean_hit = hit_matrix[chosen_items].mean(axis=0)
        mean_fa = fa_matrix[chosen_items].mean(axis=0)
        dprime = hgcm.zc(mean_hit, lowest_rate) - hgcm.zc(mean_fa, lowest_rate)
        return [mean_hit[-1] - mean_hit[0], mean_fa[-1] - mean_fa[0], dprime[-1] - dprime[0]]
    return statistics


DESIGNED_QUANTITIES = [('hit', 'hit rate', 'change in hit rate'),
                       ('fa', 'false alarm rate to the nearest neighbor', 'change in false alarm rate'),
                       ('dprime', 'd′', 'change in d′')]
for variant in (HYBRID, GCM):
    variant_designed = inputs.designed_table(results[variant])
    if have_item_tables:
        statistics = designed_statistics_for(variant)
        all_critical_items = np.arange(n_critical)
        if abs(statistics(all_critical_items)[2] - designed_change(variant_designed, 'dprime')) > 1e-12:
            raise RuntimeError('the designed-list d′ change does not reproduce core.json for %s' % variant)
        print('bootstrap over critical items, designed lists, %s (%d resamples)' % (variant, N_RESAMPLES), flush=True)
        resampled = bootstrap.percentile_bootstrap(statistics, n_critical, N_RESAMPLES, SEED)
    for position, (column, quantity_name, statistic_name) in enumerate(DESIGNED_QUANTITIES):
        change = designed_change(variant_designed, column)
        if variant == HYBRID:
            ledger_ids = 'L6-108;ADD-006'
            location = 'SI Appendix, Section 2, A Hybrid-Similarity Exemplar Model; Fig. S7b and c'
        else:
            ledger_ids = 'L6-112;ADD-006'
            location = 'SI Appendix, Section 2, A Hybrid-Similarity Exemplar Model; Fig. S7b and c'
        fields = dict(ledger_ids=ledger_ids, location=location,
                      test='Designed lists: %s of the same critical items studied with 16 rather than 0 close neighbors '
                           '(%d critical items x %d simulated participants per list), %s'
                           % (quantity_name, n_critical, n_participants_per_list, MODEL_NAMES[variant]),
                      statistic_name=statistic_name, statistic=change, n=n_critical,
                      unit='critical items (%d simulated participants each)' % n_participants_per_list,
                      effect_size_name=statistic_name, effect_size=change)
        if have_item_tables:
            low, high = interval(resampled[:, position])
            INTERVALS[(variant, statistic_name)] = (low, high)
            add_bootstrap_row(ci_low=low, ci_high=high, resampling_unit='critical items', **fields)


# --- 4c. Bylinskii context simulation, the reference setting: bootstrap over the targets
targets_file = os.path.join(PATHS['outputs'], 'bylinskii_reference_list_targets.csv')
n_bylinskii_targets = float('nan')      # the number of targets in the per-target table (nan without it)
if os.path.exists(targets_file):
    targets = pd.read_csv(targets_file)
    n_bylinskii_targets = len(targets)
    memorability_within = (targets['hits_within'].to_numpy() / targets['old_tests_within'].to_numpy()
                           - targets['false_alarms_within'].to_numpy() / targets['new_tests_within'].to_numpy())
    memorability_across = (targets['hits_across'].to_numpy() / targets['old_tests_across'].to_numpy()
                           - targets['false_alarms_across'].to_numpy() / targets['new_tests_across'].to_numpy())
    bylinskii_rho = spearmanr(memorability_within, memorability_across)[0]
    if abs(bylinskii_rho - bylinskii_numbers['reference']) > 1e-12:
        raise RuntimeError('bylinskii_reference_list_targets.csv does not reproduce the reference [list] rho')

    def bylinskii_statistic(chosen_targets):
        return spearmanr(memorability_within[chosen_targets], memorability_across[chosen_targets])[0]

    print('bootstrap over targets, Bylinskii reference [list] setting (%d resamples)' % N_RESAMPLES, flush=True)
    resampled = bootstrap.percentile_bootstrap(bylinskii_statistic, len(targets), N_RESAMPLES, SEED)
    low, high = interval(resampled)
    INTERVALS[('Bylinskii', 'Spearman rho')] = (low, high)
    fields = dict(ledger_ids='L6-109', location='SI Appendix, Section 2, A Hybrid-Similarity Exemplar Model',
                  test='Bylinskii et al. (2015) context simulation, hybrid model, reference setting: Spearman correlation '
                       'across the %s targets between memorability within category and across categories'
                       % format(len(targets), ','),
                  statistic_name='Spearman rho', statistic=bylinskii_rho, n=len(targets),
                  unit='simulated targets (%d categories)' % targets['category'].nunique(),
                  effect_size_name='rho', effect_size=bylinskii_rho)
    add_bootstrap_row(ci_low=low, ci_high=high, resampling_unit='simulated targets', **fields)
else:
    print('NOTE: outputs/bylinskii_reference_list_targets.csv does not exist (run code/bylinskii_reference_cells.py, '
          'about 2 min); the Bylinskii interval is left out of tests.csv, and it and the number of targets are nan '
          'in numbers.csv')


# --- 4d. the two sweeps, as counts and ranges of settings (not tests)
def add_sweep_rows(ledger_ids, test, count_name, count, n_settings, lowest, highest, unit):
    common = dict(test_id='T30', ledger_ids=ledger_ids,
                  location='SI Appendix, Section 2, A Hybrid-Similarity Exemplar Model', test=test, n=n_settings,
                  unit=unit, ci_low='', ci_high='', ci_method='', resampling_unit='', n_resamples='', seed='', **NO_P)
    tests.add(statistic_name=count_name, statistic=count, effect_size_name='', effect_size='', **common)
    tests.add(statistic_name='Spearman rho, lowest over the %d settings' % n_settings, statistic=lowest,
              effect_size_name='rho', effect_size=lowest, **common)
    tests.add(statistic_name='Spearman rho, highest over the %d settings' % n_settings, statistic=highest,
              effect_size_name='rho', effect_size=highest, **common)


add_sweep_rows('L6-111;ADD-004',
               'Parameter sweep of the hybrid model (rarity from the study list): item-level Spearman rho in each of the '
               '%d settings with a weight on rare matches (one change at a time from the reference, plus a grid of g and w)'
               % len(sweep_list),
               'settings with rho below zero', sweep_n_negative, len(sweep_list), float(sweep_lowest['rho_item']),
               float(sweep_highest['rho_item']),
               'parameter settings (%s simulated experiments each)' % format(sweep_runs, ','))
add_sweep_rows('L6-109;ADD-005',
               'Bylinskii context simulation of the hybrid model (rarity from the study list): cross-context Spearman rho '
               'in each of the %d settings (criteria, category structure and model parameters)'
               % bylinskii_numbers['n_settings'],
               'settings with rho above zero', bylinskii_numbers['n_positive'], bylinskii_numbers['n_settings'],
               bylinskii_numbers['lowest'], bylinskii_numbers['highest'],
               'parameter settings')

tests.write(PATHS['outputs'])
print('seed %d for every bootstrap; %d resamples' % (SEED, N_RESAMPLES))


# ============================================================================ 5. numbers.csv
numbers = numbers_table.NumbersTable(script='code/stats_figS7.py')
hybrid_rho = hybrid['A']['rho_item']
gcm_rho = gcm['A']['rho_item']
hybrid_dprime = hybrid['A']['dprime']
gcm_dprime = gcm['A']['dprime']
hybrid_hit_change = designed_change(hybrid_designed, 'hit')
hybrid_fa_change = designed_change(hybrid_designed, 'fa')
hybrid_dprime_change = designed_change(hybrid_designed, 'dprime')
gcm_dprime_change = designed_change(gcm_designed, 'dprime')
hybrid_settings = core_script.VARIANTS[HYBRID]
gcm_settings = core_script.VARIANTS[GCM]
base = core_script.BASE
CORE_SCRIPTS = 'code/01_core.py; code/hgcm.py'


def bootstrap_interval(variant, statistic_name):
    """(ci_low, ci_high) of one bootstrap of section 4; NaN when its per-item table is not in outputs/."""
    return list(INTERVALS.get((variant, statistic_name), (float('nan'), float('nan'))))


hybrid_rho_interval = bootstrap_interval(HYBRID, 'Spearman rho')
gcm_rho_interval = bootstrap_interval(GCM, 'Spearman rho')
hybrid_hit_interval = bootstrap_interval(HYBRID, 'change in hit rate')
hybrid_fa_interval = bootstrap_interval(HYBRID, 'change in false alarm rate')
hybrid_dprime_change_interval = bootstrap_interval(HYBRID, 'change in d′')
bylinskii_interval = bootstrap_interval('Bylinskii', 'Spearman rho')


# --- main text and General Discussion (no number printed: statements that point to SI Appendix, Section 2)
numbers.add('L1-042',
            'hybrid model (rarity from the study list): rho %s, d′ %.3f; designed lists, 0 to 16 neighbors: hits %s, '
            'false alarms %s, d′ %s; Bylinskii rho %s (above zero in %d of %d settings); sweep: rho below zero in %d of '
            '%d settings'
            % (signed(hybrid_rho), hybrid_dprime, signed(hybrid_hit_change), signed(hybrid_fa_change),
               signed(hybrid_dprime_change), signed(bylinskii_numbers['reference']), bylinskii_numbers['n_positive'],
               bylinskii_numbers['n_settings'], sweep_n_negative, len(sweep_list)),
            mode=core_mode, script=CORE_SCRIPTS,
            note='pointer to SI Appendix, Section 2. Values of the hybrid model with rarity from the study list: '
                 'item-level Spearman rho between memorability and crowding and overall d′ (01_core.py), the changes '
                 'from 0 to 16 close neighbors on designed lists, the Bylinskii cross-context rho with the number of '
                 'settings above zero (03_bylinskii.py), and the number of sweep settings with rho below zero '
                 '(02_sweep.py)')
numbers.add('L4-009',
            'without the weight on rare matches (gamma 0, the GCM) rho %s; with it (gamma 2) rho %s; REM rho %s; '
            'the gamma 0 setting of the [list] sweep: rho %s'
            % (signed(gcm_rho), signed(hybrid_rho), signed(rem['rho']), signed(sweep_gamma0_list['rho_item'])),
            mode=core_mode, script=CORE_SCRIPTS,
            note='item-level Spearman rho between memorability and crowding without extra weight on rare matches '
                 '(GCM, gamma 0), with it (hybrid model, gamma 2) and in REM at the Fig. 1 values (module A2a), and at '
                 'the gamma 0 setting of the parameter sweep with rarity from the study list')
numbers.add('L4-010',
            'rarity from each participant\'s own study list (ref = \'%s\' in hgcm.ref_table: a value is distinctive when '
            'fewer than %.0f%% of the %d studied items carry it); rho %s, designed-list d′ change %s, Bylinskii rho %s'
            % (hybrid_settings['ref'], 100 * hybrid_settings['q'], base['n_study'], signed(hybrid_rho),
               signed(hybrid_dprime_change), signed(bylinskii_numbers['reference'])),
            mode=core_mode, script=CORE_SCRIPTS,
            note='rarity from each participant\'s own study list (ref = \'list\' in hgcm.ref_table), counted over the '
                 'items as studied, not over the noisy traces (hgcm.list_freq); the item-level rho, the designed-list '
                 'change in d′ and the Bylinskii cross-context rho of this model')

# --- SI Appendix, Section 2, A Hybrid-Similarity Exemplar Model: the model
numbers.add('L6-102',
            'familiarity = sum over the studied traces of exp(-c d) x exp(gamma m - delta n_mismatch); d = Euclidean '
            'distance over the trace\'s stored features x sqrt(w / number stored); delta = %g in both models of the paper'
            % hybrid_settings['delta'],
            mode='rerun', script='code/hgcm.py',
            note='hgcm.familiarity(): familiarity is the sum over the studied traces of exp(-c d) x exp(gamma m), where '
                 'd is the Euclidean distance over the features the trace stored, multiplied by sqrt(w / number stored), '
                 'and m is the number of stored features with a shared distinctive value; the mismatch weight delta is '
                 '0 in both models. stats_figS7.py checks the function against the same rule written with explicit '
                 'loops (largest relative difference %.1e)' % similarity_rule_difference)
numbers.add('L6-103', round(100 * hybrid_settings['q'], 10), mode='rerun', script='code/01_core.py',
            note='rarity threshold q of the hybrid model in percent (q = %.2f, rarity from the study list; '
                 'VARIANTS["%s"] in 01_core.py)' % (hybrid_settings['q'], HYBRID))
numbers.add('L6-104', hybrid_settings['gamma'], mode='rerun', script='code/01_core.py',
            note='weight gamma on shared distinctive values in the hybrid model (VARIANTS["%s"]["gamma"] in '
                 '01_core.py); the mismatch weight delta is %g' % (HYBRID, hybrid_settings['delta']))
numbers.add('L6-105', base['target_fa'], mode='rerun', script='code/01_core.py',
            note='target false alarm rate of the criterion (BASE in 01_core.py); the criterion is the %.0fth percentile '
                 'of new-item familiarity in a 600-experiment calibration batch, and the false alarm rates in the basic '
                 'simulation are %.3f (hybrid model) and %.3f (GCM)'
                 % (100 * (1 - base['target_fa']), hybrid['A']['fa'], gcm['A']['fa']))
numbers.add('L6-106',
            'g %.2f, w %d, u %.2f, copy %.2f; %s items drawn from %s generated; %d studied; %d old and %d new tested'
            % (base['g'], base['w'], base['u'], base['cenc'], format(base['n_db'], ','), format(base['n_full'], ','),
               base['n_study'], base['n_old'], base['n_new']),
            mode='rerun', script='code/01_core.py',
            note='the design of Fig. 1 as set in BASE of 01_core.py; the items are a numpy draw (default_rng(1)) with '
                 'this design, not the item set of module 01')

# --- SI Appendix, Section 2, A Hybrid-Similarity Exemplar Model: simulation results
numbers.add('L6-322',
            'item-level rho %s; designed-list d′ change %s; Bylinskii rho %s'
            % (signed(hybrid_rho), signed(hybrid_dprime_change), signed(bylinskii_numbers['reference'])),
            mode=core_mode, script='code/01_core.py; code/03_bylinskii.py',
            note='the three simulation results of the hybrid model with rarity from the study list: the item-level '
                 'Spearman rho between memorability and crowding (below zero), the change in d′ from 0 to 16 close '
                 'neighbors on designed lists (below zero) and the Bylinskii cross-context rho (above zero)')
numbers.add('L6-107', [hybrid_rho] + hybrid_rho_interval + [rem['rho']], mode=core_mode, script='code/01_core.py',
            note='Spearman rho across the %s simulated items between memorability (hit rate minus false alarm rate) '
                 'and crowding (negated mean Euclidean distance to the other items) for the hybrid model (01_core.py), '
                 'its 95%% percentile bootstrap interval over the items (tests.csv, T30), and REM at the Fig. 1 values '
                 '(row "%s" of the REM parameter sweep, module A2a)' % (format(n_items, ','), rem['label']))
numbers.add('L6-108', [int(hybrid_designed['similar'].iloc[-1])] + [hybrid_hit_change] + hybrid_hit_interval
            + [hybrid_fa_change] + hybrid_fa_interval
            + [-hybrid_dprime_change, -hybrid_dprime_change_interval[1], -hybrid_dprime_change_interval[0]],
            mode=core_mode, script='code/01_core.py',
            note='designed lists, hybrid model: the largest number of close neighbors on a list, then each change from 0 '
                 'to that number with its 95%% percentile bootstrap interval over the critical items (tests.csv, T30): '
                 'hit rate of the critical items %.3f to %.3f, '
                 'false alarm rate to their nearest neighbor %.3f to %.3f, d′ %.3f to %.3f (the last three values are '
                 'the fall in d′ and its interval, from the mean rates over the critical items; the change is %s)'
                 % (hybrid_designed['hit'].iloc[0], hybrid_designed['hit'].iloc[-1], hybrid_designed['fa'].iloc[0],
                    hybrid_designed['fa'].iloc[-1], hybrid_designed['dprime'].iloc[0], hybrid_designed['dprime'].iloc[-1],
                    signed(hybrid_dprime_change, 4)))
numbers.add('L6-109', [bylinskii_numbers['reference']] + bylinskii_interval
            + [n_bylinskii_targets, bylinskii_numbers['n_positive'], bylinskii_numbers['lowest'],
               bylinskii_numbers['highest']],
            mode=bylinskii_numbers['mode'], script='code/03_bylinskii.py; code/bylinskii_reference_cells.py',
            note='Bylinskii context simulation of the hybrid model with rarity from the study list: Spearman rho across '
                 'the targets between memorability within category and across categories at the setting "%s" with its '
                 '95%% percentile bootstrap interval over the targets (tests.csv, T30) and the number of targets, the '
                 'number of settings with rho above zero (%d of %d), and the lowest ("%s") and highest ("%s") rho over '
                 'the settings%s'
                 % (BYLINSKII_REFERENCE, bylinskii_numbers['n_positive'], bylinskii_numbers['n_settings'],
                    bylinskii_numbers['lowest_cell'], bylinskii_numbers['highest_cell'], bylinskii_numbers['mode_note']))
numbers.add('L6-110', [hybrid_dprime, rem['dprime']], mode=core_mode, script='code/01_core.py',
            note='overall d′ of the basic simulation from the mean hit and false alarm rates over the items: hybrid '
                 'model (01_core.py) and REM at the Fig. 1 values (module A2a); the width c is the candidate that brings '
                 'd′ closest to REM\'s %.2f, and with it the hybrid model\'s d′ is %.2f' % (base['target_d'], hybrid_dprime))
numbers.add('L6-111', [sweep_n_negative, len(sweep_list), float(sweep_lowest['rho_item']), float(sweep_highest['rho_item'])],
            mode=sweep_mode, script='code/02_sweep.py',
            note='parameter sweep of the hybrid model with rarity from the study list (02_sweep.py, 5,000 simulated '
                 'experiments per setting): the number of the %d settings with gamma above 0 whose item-level rho is '
                 'below zero, the number of settings, and the lowest ("%s") and highest ("%s") rho; rho at gamma 0, '
                 '1, 2 (the reference setting) and 3: %s'
                 % (len(sweep_list), sweep_lowest['cell'], sweep_highest['cell'],
                    ', '.join(signed(value) for value in sweep_rho_by_gamma)))
numbers.add('L6-112', [gcm_settings['gamma'], gcm_rho] + gcm_rho_interval, mode=core_mode, script='code/01_core.py',
            note='the GCM (the hybrid model\'s code with gamma = 0, c = %.4f): gamma, and the Spearman rho between '
                 'memorability and crowding over the items with its 95%% percentile bootstrap interval over the items '
                 '(tests.csv, T30); its designed-list d′ falls by %.3f from 0 to 16 close '
                 'neighbors' % (gcm['params']['c'], -gcm_dprime_change))

# --- SI Appendix, Fig. S7 and its caption
numbers.add('L6-113', 'REM %s; GCM %s; hybrid %s' % (signed(rem['rho']), signed(gcm_rho), signed(hybrid_rho)),
            mode=core_mode, script='code/make_figureS7.py',
            note='Fig. S7a bars: item-level Spearman rho of REM at the Fig. 1 values (module A2a), the GCM and the '
                 'hybrid model (outputs/figure_data/FigS7a.csv)')
numbers.add('L6-114',
            'd′ at 0, 2, 4, 8, 16 neighbors: REM %s; GCM %s; hybrid %s'
            % (series(rem_designed['dprime']), series(gcm_designed['dprime']), series(hybrid_designed['dprime'])),
            mode=core_mode, script='code/make_figureS7.py',
            note='d′ with 0, 2, 4, 8 and 16 close neighbors studied: REM from CELL 2 of list_designed.txt (designed '
                 'lists of module A2d, 300 critical items), the GCM and the hybrid model from 01_core.py (%d critical '
                 'items); Fig. S7b plots the change from 0 neighbors (outputs/figure_data/FigS7b.csv)' % n_critical)
numbers.add('L6-115',
            'hits: REM %s; GCM %s; hybrid %s. false alarms: REM %s; GCM %s; hybrid %s'
            % (series(rem_designed['hit']), series(gcm_designed['hit']), series(hybrid_designed['hit']),
               series(rem_designed['false_alarm']), series(gcm_designed['fa']), series(hybrid_designed['fa'])),
            mode=core_mode, script='code/make_figureS7.py',
            note='hit and false alarm rates with 0, 2, 4, 8 and 16 close neighbors studied for REM (module A2d), the '
                 'GCM and the hybrid model; Fig. S7c plots the changes from 0 neighbors '
                 '(outputs/figure_data/FigS7c.csv)')
numbers.add('L6-116', [largest_count, base['n_study']], mode='rerun', script='code/hgcm.py',
            note='largest number of the %d studied items that can share a value while it still counts as distinctive, '
                 'found by running hgcm.ref_table() on study lists in which one value is carried by 1 to %d items: %d '
                 'or fewer (fewer than %.0f%%); and the length of the study list'
                 % (base['n_study'], base['n_study'], largest_count, 100 * hybrid_settings['q']))
numbers.add('L6-117', n_experiments, mode=core_mode, script='code/01_core.py',
            note='simulated experiments of the basic simulation, %s (the default n_sim of analysis_a is %d; the sweep '
                 'settings use 5,000)' % (size_source, default_argument(core_script.analysis_a, 'n_sim')))
numbers.add('L6-323', n_items, mode=core_mode, script='code/01_core.py',
            note='simulated items of the basic simulation (Fig. S7a), %s (n_db = %d in BASE of 01_core.py)'
                 % (size_source, base['n_db']))
numbers.add('L6-118', [int(hybrid_designed['similar'].min()), int(hybrid_designed['similar'].max())], mode=core_mode,
            script='code/01_core.py',
            note='fewest and most close neighbors on the designed lists (analysis_d in 01_core.py runs %s)'
                 % ', '.join(str(level) for level in hybrid_designed['similar']))
numbers.add('L6-119', [hybrid_dprime, gcm_dprime, rem['dprime']], mode=core_mode, script='code/01_core.py',
            note='overall d′ of the basic simulation for the hybrid model, the GCM and REM at the Fig. 1 values '
                 '(module A2a), from the mean hit and false alarm rates over the items')


numbers.add('ADD-002', [hybrid['params']['c'], gcm['params']['c']], mode=core_mode, script='code/01_core.py',
            note='similarity width c of the hybrid model (gamma 2) and the GCM (gamma 0), chosen in 01_core.py as '
                 'c = 1 / (f x median inter-item distance) with f from the candidates (%s): the one whose overall d′ '
                 'in a 600-experiment batch comes closest to REM\'s %.2f (the basic simulation then gives d′ %.2f for '
                 'the hybrid model and %.2f for the GCM). The criterion is the %.0fth percentile of new-item familiarity '
                 'in the same batch: %.4f (hybrid model) and %.5f (GCM); delta = %g; seed %d'
                 % (calibration_fractions, base['target_d'], hybrid_dprime, gcm_dprime, 100 * (1 - base['target_fa']),
                    hybrid['params']['crit'], gcm['params']['crit'], hybrid['params']['delta'], paper_seed))
numbers.add('L6-321', len(calibration_fractions.split(',')), mode='rerun', script='code/01_core.py',
            note='number of candidate widths c = 1 / (f x median inter-item distance) in the calibration of 01_core.py, '
                 'f = %s' % calibration_fractions)
numbers.add('ADD-003', n_critical, mode=core_mode, script='code/01_core.py',
            note='number of critical items on the designed lists (n_crit of analysis_d in 01_core.py, %s); each list '
                 'is studied by %d simulated participants, at %s close neighbors; the foil is each critical item\'s '
                 'nearest neighbor; the fillers come from the far half of the item\'s distance ranking; rates are '
                 'clamped to [1 / (2 x %d), 1 - 1 / (2 x %d)] before the z transform'
                 % (size_source, n_participants_per_list, ', '.join(str(level) for level in neighbor_levels),
                    n_participants_per_list, n_participants_per_list))
text_add004 = '%d settings; %s simulated experiments per setting; seed %s' % (len(sweep_list), format(sweep_runs, ','),
                                                                             sweep_seed_rule)
numbers.add('ADD-004', text_add004, mode=sweep_mode, script='code/02_sweep.py',
            note='settings of the parameter sweep (02_sweep.py): the settings with gamma above 0, the simulated '
                 'experiments per setting, and the seed of each setting, zlib.crc32 of its label')
reference = bylinskii_script.REF
text_add005 = ('g %.2f; w %d; u %.2f; copy %.2f; FA %.2f / %.2f; %d x %d; %d targets; %d category features; seed %d'
               % (reference['g'], reference['w'], reference['u'], reference['cenc'], reference['fa_within'],
                  reference['fa_across'], reference['n_cat'], reference['per_cat'], reference['n_targets'],
                  reference['cat_feat'], bylinskii_seed))
numbers.add('ADD-005', text_add005, mode=bylinskii_numbers['mode'], script='code/03_bylinskii.py',
            note='settings of the Bylinskii context simulation of the hybrid model (03_bylinskii.py): g, w, u, copy '
                 'fidelity, the false alarm rates within / across categories at which the criteria are set, categories '
                 'x items per category, targets per category and category-defining features; lists of %d targets and '
                 '%d fillers; %d within-category lists per category and %d across-category lists; seed %d is the seed '
                 'of the width calibration, and each setting has its own seed zlib.crc32(label)'
                 % (reference['n_study_t'], reference['n_study_f'], reference['runs_within'], reference['runs_across'],
                    bylinskii_seed))


def changes_text(designed_table, column, include_zero):
    """The changes from 0 neighbors at 2, 4, 8 and 16 neighbors as text."""
    values = designed_table[column].to_numpy()
    changes = values - values[0]
    parts = []
    if include_zero:
        parts.append('0')
    for change in changes[1:]:
        parts.append(signed(change))
    return ', '.join(parts)


text_add006 = ('GCM d′ change %s; hybrid d′ change %s; GCM hits %s; GCM FA %s; hybrid hits %s; hybrid FA %s'
               % (changes_text(gcm_designed, 'dprime', True), changes_text(hybrid_designed, 'dprime', True),
                  changes_text(gcm_designed, 'hit', False), changes_text(gcm_designed, 'fa', False),
                  changes_text(hybrid_designed, 'hit', False), changes_text(hybrid_designed, 'fa', False)))
numbers.add('ADD-006', text_add006, mode=core_mode, script='code/01_core.py; code/make_figureS7.py',
            note='changes from 0 close neighbors at 2, 4, 8 and 16 neighbors that Fig. S7b and c plot for the GCM and '
                 'the hybrid model, from the rates in core.json (outputs/figure_data/FigS7b.csv and FigS7c.csv)')

numbers.write(PATHS['outputs'])
