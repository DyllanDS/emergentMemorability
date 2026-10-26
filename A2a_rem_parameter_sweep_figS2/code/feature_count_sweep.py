"""
feature_count_sweep.py: the second REM sweep of SI Appendix, Section 2 (REM Parameters): the REM
simulation of Fig. 1 run with 5 to 120 features per item instead of 49.

What it does
------------
For every number of features w (18 values, 5 to 120) and every seed (12345, 777, 20260731) it runs
the REM simulation of Fig. 1 once:
  items       26,107 items (the number of THINGS images) with w features each; every feature value
              is a geometric draw with g = 0.65 (values 1, 2, 3, ...). A random 1,600 of them are
              the simulated image set.
  experiments 10,000 simulated recognition experiments. In each, a simulated participant studies 40
              of the 1,600 items and is tested on 20 of the studied items (old) and 20 items of the
              1,600 that were not studied (new).
  encoding    every feature of a studied item is stored with probability 0.30; a stored feature is
              copied correctly with probability 0.60, otherwise it gets a new geometric draw counted
              from 0 (as in the Fig. 1 script), and a value of 0 means "not stored".
  decision    the familiarity of a test item is the mean, over the 40 memory traces, of the REM
              likelihood ratio (Shiffrin and Steyvers, 1997); the item is called "old" when the
              familiarity is above 1.
For every item: hit rate, false alarm rate and memorability (hit rate minus false alarm rate). Then
  r_sim_spearman   Spearman correlation across the 1,600 items of memorability with summed
                   similarity, 1 / (mean Euclidean distance to the items of the set). For a rank
                   correlation this equals the correlation with the negated mean distance. SI
                   Appendix, Section 2 reports the smallest and largest of the 54 single runs
                   (rho = -0.66 to -0.35).
  r_dist_pearson   Pearson correlation of memorability with the mean distance itself
  r_meta           the feature-level analysis of Fig. 1d: across the w features, the Spearman
                   correlation between each feature's correlation with memorability and its
                   correlation with similarity

This is the REM simulation of Fig. 1 (module 01, fig1_remSim.m) in NumPy. It differs from
rem_parameter_sweep.m in this folder in its random numbers, in drawing the 1,600-item set from
26,107 generated items, and in drawing a wrongly copied feature from 0 (a 0 then counts as not
stored), as fig1_remSim.m does.

Inputs  none (the items are generated here)
Outputs outputs/sweep_rem.csv: one row per run (feature count x seed), in the order run. The first
        eight columns are those of the saved run (data/paper_run/sweep_rem.csv): w, seed,
        mean_cr, r_sim_spearman, p_sim_spearman, r_dist_pearson, r_meta, p_meta. The other columns
        (n_items, sd_cr, mean_hr, mean_fa) are the rest of what each run computes.
Seeds   one numpy generator per run, numpy.random.default_rng(seed), seeds 12345, 777 and 20260731.
        A run's results depend only on its own seed and feature count, not on the other runs. The
        random draws are made in this order: the 26,107 generated items, the permutation that picks
        the 1,600-item set, then per experiment the permutation of studied and new items, the
        stored and copied-correctly uniforms, and the wrong values.
Runtime about 3 minutes for the 54 runs on an Apple M2 Pro, up to about 12 minutes when other jobs
        share the processor.

Run (from the repository root):
    python A2a_rem_parameter_sweep_figS2/code/feature_count_sweep.py
A shorter test of the code (not the paper's numbers), e.g.:
    python A2a_rem_parameter_sweep_figS2/code/feature_count_sweep.py --nsim 200 --widths 5 49 --output sweep_rem_quick_test.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (the module's folders, found from this file's location)

# ----------------------------------------------------------------------------- REM settings (Fig. 1)
G = 0.65               # parameter of the geometric distribution of feature values
U = 0.30               # probability that a feature of a studied item is stored
C = 0.60               # probability that a stored feature is copied correctly
CRITERION = 1.0        # "old" when the mean likelihood ratio is above this
N_GENERATED = 26107    # items generated (the number of THINGS images)
N_ITEMS = 1600         # the simulated image set drawn from them
N_STUDIED = 40         # items studied per simulated experiment
N_OLD_TESTED = 20      # studied items tested
N_NEW_TESTED = 20      # unstudied items tested (foils)

# ----------------------------------------------------------------------------- the run the paper reports
PAPER_FEATURE_COUNTS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 49, 55, 60, 66, 70, 80, 90, 100, 120]
PAPER_SEEDS = [12345, 777, 20260731]

# The columns of the saved table (data/paper_run/sweep_rem.csv) come first, in its order; the rest follow.
OUTPUT_COLUMNS = ['w', 'seed', 'mean_cr', 'r_sim_spearman', 'p_sim_spearman', 'r_dist_pearson', 'r_meta',
                  'p_meta', 'n_items', 'sd_cr', 'mean_hr', 'mean_fa']


def run_rem(n_features, n_simulations=10000, seed=12345):
    """One REM run with n_features features per item. Returns a dict of the run's results."""
    random_generator = np.random.default_rng(seed)

    # Random draw 1: 26,107 items, every feature value a geometric draw (1, 2, 3, ...).
    generated_items = random_generator.geometric(G, size=(N_GENERATED, n_features))
    # Random draw 2: the 1,600 items of the simulated image set.
    item_set = generated_items[random_generator.permutation(N_GENERATED)[:N_ITEMS]]

    # The log likelihood ratio a matching stored feature with value v contributes (REM):
    #   log[(c + (1 - c) * P(v)) / P(v)], with P(v) = g * (1 - g)^(v - 1) the geometric probability
    # of the value. Rare (large) values are stronger evidence. A table for values 0 to largest_value.
    largest_value = int(max(item_set.max(), 200)) + 5
    feature_values = np.arange(largest_value)
    probability_of_value = G * (1 - G) ** np.maximum(feature_values - 1, 0)
    log_ratio_of_match = np.log((C + (1 - C) * probability_of_value) / probability_of_value)
    # Every stored feature that does not match multiplies the likelihood ratio by (1 - c).
    log_ratio_of_mismatch = np.log(1 - C)

    hit_count = np.zeros(N_ITEMS)       # times the item was called "old" when it was old
    old_trial_count = np.zeros(N_ITEMS)  # times the item was tested as old
    false_alarm_count = np.zeros(N_ITEMS)  # times the item was called "old" when it was new
    new_trial_count = np.zeros(N_ITEMS)  # times the item was tested as new

    for simulation in range(n_simulations):
        # Random draw: 40 studied items and 20 foils, all different.
        drawn = random_generator.permutation(N_ITEMS)[:N_STUDIED + N_NEW_TESTED]
        studied = drawn[:N_STUDIED]
        old_tested = studied[:N_OLD_TESTED]
        new_tested = drawn[N_STUDIED:N_STUDIED + N_NEW_TESTED]

        # Encoding (random draws, in this order: stored?, copied correctly?, the wrong values).
        studied_features = item_set[studied]                                  # 40 x w
        feature_is_stored = random_generator.random((N_STUDIED, n_features)) < U
        feature_is_correct = random_generator.random((N_STUDIED, n_features)) < C
        memory_traces = studied_features.copy()
        number_wrong = int((~feature_is_correct).sum())
        if number_wrong:
            # A wrong copy is a new geometric draw counted from 0 (MATLAB geornd), as in Fig. 1.
            memory_traces[~feature_is_correct] = random_generator.geometric(G, size=number_wrong) - 1
        memory_traces[~feature_is_stored] = 0                                 # 0 = not stored

        # Familiarity of the 40 test items: the one vectorized step of this script. For every test
        # item (probe) and every memory trace it counts the stored features that match and do not
        # match, adds up their log likelihood ratios, and averages the likelihood ratio over the 40
        # traces. The arithmetic runs over arrays of probes x traces x features.
        probes = item_set[np.concatenate([old_tested, new_tested])]          # 40 x w: 20 old, 20 new
        trace_feature_stored = memory_traces > 0                             # 40 traces x w
        same_value = probes[:, None, :] == memory_traces[None, :, :]         # probes x traces x w
        matches = same_value & trace_feature_stored[None, :, :]
        mismatches = (~same_value) & trace_feature_stored[None, :, :]
        log_ratio_of_stored_value = np.where(trace_feature_stored,
                                             log_ratio_of_match[np.clip(memory_traces, 0, largest_value - 1)],
                                             0.0)
        log_likelihood_ratio = (mismatches.sum(-1) * log_ratio_of_mismatch
                                + (matches * log_ratio_of_stored_value[None, :, :]).sum(-1))
        familiarity = np.exp(log_likelihood_ratio).mean(1)                   # one value per probe
        said_old = familiarity > CRITERION

        # Bookkeeping, one test item at a time.
        for position in range(N_OLD_TESTED):
            item = old_tested[position]
            hit_count[item] += said_old[position]
            old_trial_count[item] += 1
        for position in range(N_NEW_TESTED):
            item = new_tested[position]
            false_alarm_count[item] += said_old[N_OLD_TESTED + position]
            new_trial_count[item] += 1

    # Per-item rates; an item never tested as old (or as new) gets no memorability score.
    with np.errstate(invalid='ignore'):
        hit_rate = np.where(old_trial_count > 0, hit_count / np.maximum(old_trial_count, 1), np.nan)
        false_alarm_rate = np.where(new_trial_count > 0, false_alarm_count / np.maximum(new_trial_count, 1), np.nan)
    memorability = hit_rate - false_alarm_rate
    memorability[(old_trial_count == 0) | (new_trial_count == 0)] = np.nan

    # Distinctiveness: the mean Euclidean distance of each item to the items of the set (the mean
    # includes the item's zero distance to itself, as in the Fig. 1 script; a constant factor that
    # changes no correlation), and summed similarity = 1 / mean distance.
    mean_distance = cdist(item_set, item_set, 'euclidean').mean(1)
    similarity = 1.0 / mean_distance

    has_score = np.isfinite(memorability)
    rho_similarity, p_rho_similarity = spearmanr(similarity[has_score], memorability[has_score])
    r_distance = pearsonr(mean_distance[has_score], memorability[has_score])[0]

    # Feature level (Fig. 1d): each feature's Spearman correlation with memorability and with
    # similarity, then the Spearman correlation of those two lists across the features.
    feature_with_memorability = []
    feature_with_similarity = []
    for feature in range(n_features):
        feature_with_memorability.append(spearmanr(item_set[has_score, feature], memorability[has_score])[0])
        feature_with_similarity.append(spearmanr(item_set[:, feature], similarity)[0])
    feature_with_memorability = np.array(feature_with_memorability)
    feature_with_similarity = np.array(feature_with_similarity)
    rho_features, p_rho_features = spearmanr(feature_with_similarity, feature_with_memorability)

    return dict(w=int(n_features), n_items=int(has_score.sum()),
                mean_cr=float(np.nanmean(memorability)), sd_cr=float(np.nanstd(memorability, ddof=1)),
                mean_hr=float(np.nanmean(hit_rate)), mean_fa=float(np.nanmean(false_alarm_rate)),
                r_sim_spearman=float(rho_similarity), p_sim_spearman=float(p_rho_similarity),
                r_dist_pearson=float(r_distance),
                r_meta=float(rho_features), p_meta=float(p_rho_features))


def write_rows(rows, path):
    """The table so far, one row per run; numbers in Python's shortest exact form (as in the saved table)."""
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTPUT_COLUMNS)
        for row in rows:
            writer.writerow([str(row[column]) for column in OUTPUT_COLUMNS])


def main():
    parser = argparse.ArgumentParser(description='REM simulation of Fig. 1 at 5 to 120 features '
                                                 '(SI Appendix, Section 2).')
    parser.add_argument('--widths', type=int, nargs='+', default=PAPER_FEATURE_COUNTS,
                        help='numbers of features (default: the 18 of the paper)')
    parser.add_argument('--seeds', type=int, nargs='+', default=PAPER_SEEDS,
                        help='seeds (default: 12345 777 20260731, the paper)')
    parser.add_argument('--nsim', type=int, default=10000,
                        help='simulated experiments per run (default 10000, the paper)')
    parser.add_argument('--output', default='sweep_rem.csv',
                        help='file name inside this module\'s outputs/ folder (default sweep_rem.csv)')
    arguments = parser.parse_args()

    paths = release_paths.module_paths(__file__)
    output_path = os.path.join(paths['outputs'], arguments.output)
    is_paper_setting = (arguments.widths == PAPER_FEATURE_COUNTS and arguments.seeds == PAPER_SEEDS
                        and arguments.nsim == 10000)
    print('REM feature-count sweep, %d simulated experiments per run, %d feature counts x %d seeds%s'
          % (arguments.nsim, len(arguments.widths), len(arguments.seeds),
             ' (the paper\'s setting)' if is_paper_setting else ' (NOT the paper\'s setting: a test run)'))
    print('%5s %8s %11s %10s %12s %9s %10s' % ('w', 'meanCR', 'r(sim,mem)', 'p', 'r(dist,mem)', 'r_meta',
                                              'p_meta'))
    print('-' * 70)
    rows = []
    for n_features in arguments.widths:
        for seed in arguments.seeds:
            result = run_rem(n_features, n_simulations=arguments.nsim, seed=seed)
            result['seed'] = seed
            rows.append(result)
            print('%5d %8.3f %+11.3f %10.2e %+12.3f %+9.3f %10.2e   seed %d'
                  % (n_features, result['mean_cr'], result['r_sim_spearman'], result['p_sim_spearman'],
                     result['r_dist_pearson'], result['r_meta'], result['p_meta'], seed), flush=True)
            write_rows(rows, output_path)          # rewritten after every run, so a stopped run keeps its rows
    print('\nwrote outputs/%s (%d runs)' % (arguments.output, len(rows)))


if __name__ == '__main__':
    main()
