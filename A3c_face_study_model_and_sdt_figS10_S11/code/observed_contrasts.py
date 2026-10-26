"""
observed_contrasts.py: the observed differences of the rare-feature face study, from its data (100
registered participants; SI Appendix, Section 3).

WHAT IT DOES
The data points of SI Appendix, Fig. S10a (SI Appendix, Section 3: "All three predicted differences
appeared in the data"), the two registered rate predictions as paired differences in the proportion
called old:
  hits:         studied rare-feature faces minus studied ordinary faces            (prediction 1)
  false alarms: new faces carrying a studied face's rare feature minus
                never-studied ordinary faces                                       (prediction 2)
Each: the mean of the 100 per-participant differences, the 95% interval of the registered
participant bootstrap (resample i draws the participants with its own generator
numpy.random.default_rng(i), i = 0 to 9,999, participants in load_order; percentile 2.5 and
97.5), and the paired t test (scipy ttest_rel, two-sided).
These are the same computations as module 04 (Fig. 4b). The values are checked against module 04's
outputs (rate_contrasts.csv and the rows L2-024 and L2-025 of numbers.csv), and the script stops if
one of them is missing or differs.

INPUTS   04_rare_feature_faces_fig4/data/face_trials.csv and face_participants.csv (through
         shared/python/face_loader.py: participant_rates())
         04_rare_feature_faces_fig4/outputs/ (for the equality check; module 04 runs first)
OUTPUTS  outputs/observed_contrasts.csv (one row per difference: mean, interval, t, df, p, d_z)
SEEDS    the registered bootstrap: default_rng(i) for resample i
RUNTIME  about 5 seconds
RUN      python A3c_face_study_model_and_sdt_figS10_S11/code/observed_contrasts.py
"""
import csv
import os
import re
import sys

sys.dont_write_bytecode = True

import numpy as np
from scipy.stats import ttest_rel

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CODE_FOLDER, '..', '..', 'shared', 'python'))
import release_paths   # noqa: E402
import face_loader   # noqa: E402
import bootstrap     # noqa: E402

PATHS = release_paths.module_paths(__file__)
MODULE_04_OUTPUTS = os.path.join(release_paths.find_module('04'), 'outputs')
NUMBER_OF_RESAMPLES = 10000


def registered_bootstrap_interval(differences_in_load_order):
    """The registered participant bootstrap: resample i = default_rng(i).integers(0, n, n)."""
    resampled_means = bootstrap.seeded_per_resample_mean_bootstrap(differences_in_load_order, NUMBER_OF_RESAMPLES)
    low, high = np.percentile(resampled_means, [2.5, 97.5])
    return float(low), float(high), resampled_means


def describe(name, label, first, second, interval_function, interval_name):
    """One output row: the paired difference first - second, its interval and the paired t test."""
    differences = first - second
    t_value, p_value = ttest_rel(first, second)
    low, high, resampled = interval_function(differences)
    return {'contrast': name, 'label': label, 'participants': len(differences),
            'mean_first': float(np.mean(first)), 'mean_second': float(np.mean(second)),
            'mean_difference': float(np.mean(differences)), 'ci_low': low, 'ci_high': high,
            'resamples_at_or_below_zero': int(np.sum(resampled <= 0)),
            'resamples_at_or_above_zero': int(np.sum(resampled >= 0)),
            't': float(t_value), 'df': len(differences) - 1, 'p': float(p_value),
            'cohen_dz': float(np.mean(differences) / np.std(differences, ddof=1)),
            'interval': interval_name}


def same_number(a, b, tolerance=1e-12):
    return abs(float(a) - float(b)) <= tolerance


def check_against_module_04(rows):
    """Stops when one of module 04's outputs is missing or a value differs from it (module 04 runs first)."""
    rate_file = os.path.join(MODULE_04_OUTPUTS, 'rate_contrasts.csv')
    numbers_file = os.path.join(MODULE_04_OUTPUTS, 'numbers.csv')
    missing = [path for path in (rate_file, numbers_file) if not os.path.exists(path)]
    if missing:
        sys.exit('observed_contrasts.py: module 04 output missing (run module 04 first): %s'
                 % ', '.join(os.path.relpath(path, release_paths.release_root()) for path in missing))
    by_name = {row['contrast']: row for row in rows}
    problems = []

    with open(rate_file, newline='', encoding='utf-8') as handle:
        module_04_rates = {row['measure']: row for row in csv.DictReader(handle)}
    for our_name, their_name in (('hits', 'hit rate'), ('false_alarms', 'false alarm rate')):
        ours = by_name[our_name]
        theirs = module_04_rates[their_name]
        for our_column, their_column in (('mean_difference', 'difference'), ('ci_low', 'ci_low'),
                                         ('ci_high', 'ci_high'), ('t', 't'), ('p', 'p'), ('cohen_dz', 'cohen_dz')):
            if not same_number(ours[our_column], theirs[their_column]):
                problems.append('%s %s: %r here, %r in 04 rate_contrasts.csv'
                                % (our_name, our_column, ours[our_column], theirs[their_column]))
    print('checked against 04_rare_feature_faces_fig4/outputs/rate_contrasts.csv')

    with open(numbers_file, newline='', encoding='utf-8') as handle:
        module_04_numbers = {row['ledger_id']: row for row in csv.DictReader(handle)}
    # L2-024 and L2-025 print each difference as '+0.1047 [0.065, 0.143], t(99) = 5.29'
    for ledger_id, our_name in (('L2-024', 'hits'), ('L2-025', 'false_alarms')):
        text = module_04_numbers[ledger_id]['computed_value']
        match = re.search(r'([-+]\d+\.\d+) \[(-?\d+\.\d+), (-?\d+\.\d+)\], t\((\d+)\) = (-?\d+\.\d+)', text)
        ours = by_name[our_name]
        expected = ('%+.4f' % ours['mean_difference'], '%.3f' % ours['ci_low'], '%.3f' % ours['ci_high'],
                    '%d' % ours['df'], '%.2f' % ours['t'])
        if match is None or match.groups() != expected:
            problems.append('%s in 04 numbers.csv reads %r; here %s' % (ledger_id, text, expected))
    print('checked against 04_rare_feature_faces_fig4/outputs/numbers.csv (L2-024, L2-025)')

    if problems:
        for problem in problems:
            print('DIFFERS: ' + problem)
        sys.exit('observed_contrasts.py: the values differ from module 04')
    print('equal to module 04: yes')


def main():
    # ---------------------------------------------------------------- Fig. S10a, registered bootstrap
    rates_in_load_order = face_loader.participant_rates(order='load_order')
    hits = describe('hits', 'hit rate: studied rare-feature faces minus studied ordinary faces (prediction 1)',
                    rates_in_load_order['hit_rare'].to_numpy(dtype=float),
                    rates_in_load_order['hit_fill'].to_numpy(dtype=float),
                    registered_bootstrap_interval,
                    'percentile bootstrap over participants, 10,000 resamples, resample i drawn with numpy '
                    'default_rng(i), participants in load_order (the registered analysis)')
    false_alarms = describe('false_alarms', 'false alarm rate: new faces carrying a studied face\'s rare feature '
                            'minus never-studied ordinary faces (prediction 2)',
                            rates_in_load_order['fa_rare'].to_numpy(dtype=float),
                            rates_in_load_order['fa_new'].to_numpy(dtype=float),
                            registered_bootstrap_interval,
                            'percentile bootstrap over participants, 10,000 resamples, resample i drawn with numpy '
                            'default_rng(i), participants in load_order (the registered analysis)')

    rows = [hits, false_alarms]
    columns = list(rows[0])
    with open(os.path.join(PATHS['outputs'], 'observed_contrasts.csv'), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: (repr(value) if isinstance(value, float) else value) for name, value in row.items()})

    for row in rows:
        print('%-28s mean %.4f vs %.4f, difference %+.4f, 95%% interval [%+.3f, %+.3f], t(%d) = %.2f, p = %.3g, '
              'd_z = %.2f' % (row['contrast'], row['mean_first'], row['mean_second'], row['mean_difference'],
                              row['ci_low'], row['ci_high'], row['df'], row['t'], row['p'], row['cohen_dz']))
    check_against_module_04(rows)


if __name__ == '__main__':
    main()
