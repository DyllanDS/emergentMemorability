"""
rem_729_rates.py: REM's predictions at each of the 729 settings of the preregistered sweep, saved one
row per setting, so that SI Appendix, Fig. S10a can draw one point per setting (SI Appendix, Section 3).

WHAT IT DOES
The registered sweep (rem_final_spec.py, run before data collection) prints only counts ("hits:
rare > ordinary 712/729"). This script runs exactly the same simulation again: the same evidence()
function, imported from rem_final_spec.py, the same grid, and the same seed for each setting. For
every setting it saves
  the six rates (proportion of faces called old): rare_old, filler_old, ord_old, rare_foil,
      unrelated, ord_foil (labels as in rem_final_spec.py)
  prediction 1, hits: studied rare faces minus studied ordinary faces
  prediction 2, false alarms: the never-studied partner carrying a studied face's rare feature minus
      a never-studied unrelated face
  prediction 3, read off the model's evidence (d_a from the mean and SD of the log odds, as in
      rem_final_spec.py):
      lure-set effect, the registered two-term form: d_a(rare vs unrelated) - d_a(rare vs its own foil)
      interaction, the four-term form rem_final_spec.py prints:
          [d_a(rare vs unrelated) - d_a(ordinary vs unrelated)]
          - [d_a(rare vs its own foil) - d_a(ordinary vs its own foil)]
      what the feature buys (rem_final_spec.py): d_a(rare vs its own foil) - d_a(ordinary vs unrelated)
These d_a values come from the same simulated evidence; computing them draws no random numbers.
The script checks its counts and grid means against the registered report in data/paper_run/ and
lists the settings at which the hit prediction fails.

INPUTS   code/rem_final_spec.py (evidence(), the grid, the item labels)
         data/paper_run/rem_final_spec.txt (the registered counts, read by row label)
OUTPUTS  outputs/rem_729_rates.csv and outputs/rem_729_rates_check.txt
SEEDS    setting number i (0 to 728), as in rem_final_spec.py
RUNTIME  about 2 minutes on an Apple M2 Pro
RUN      python A3c_face_study_model_and_sdt_figS10_S11/code/rem_729_rates.py
"""
import argparse
import csv
import itertools
import os
import sys

sys.dont_write_bytecode = True

import numpy as np

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_FOLDER)
sys.path.insert(0, os.path.join(CODE_FOLDER, '..', '..', 'shared', 'python'))
import release_paths                  # noqa: E402
import rem_final_spec               # noqa: E402  (the registered evidence() and grid; importing it runs no sweep)
import face_study_common              # noqa: E402

PATHS = release_paths.module_paths(__file__)
evidence = rem_final_spec.evidence
GRID = rem_final_spec.GRID
ITEM_LABELS = rem_final_spec.LAB     # rare_old, filler_old, ord_old, rare_foil, unrelated, ord_foil


def d_a_from_evidence(odds, labels, kind_a, kind_b):
    """d_a of kind_a against kind_b read off REM's evidence, exactly as rem_final_spec.py computes it:
    (mean log odds of a - mean log odds of b) x sqrt(2 / (SD_a^2 + SD_b^2)), SDs with ddof = 1."""
    log_odds_a = np.log(odds[labels == kind_a])
    log_odds_b = np.log(odds[labels == kind_b])
    return ((log_odds_a.mean() - log_odds_b.mean())
            * np.sqrt(2.0 / (log_odds_a.std(ddof=1) ** 2 + log_odds_b.std(ddof=1) ** 2)))


def run_grid():
    """One row per setting: the parameters, the six rates and the predicted differences."""
    parameter_names = list(GRID)          # g, w, u, c, nfeat, fa (the order of the registered run)
    all_settings = list(itertools.product(*[GRID[name] for name in parameter_names]))
    rows = []
    for setting_number, setting_values in enumerate(all_settings):
        parameters = dict(zip(parameter_names, setting_values))
        # The registered run uses the setting's position in the grid as its random seed.
        seed = setting_number
        odds, item_label = evidence(parameters['g'], parameters['w'], parameters['u'],
                                    parameters['c'], parameters['nfeat'], seed=seed)
        # Criterion: placed so that a fixed share (fa) of never-studied faces is called old.
        criterion = np.quantile(odds[item_label == 'unrelated'], 1.0 - parameters['fa'])
        called_old = odds > criterion
        rates = {}
        for label in ITEM_LABELS:
            rates[label] = float(called_old[item_label == label].mean())

        rare_vs_unrelated = d_a_from_evidence(odds, item_label, 'rare_old', 'unrelated')
        rare_vs_own_foil = d_a_from_evidence(odds, item_label, 'rare_old', 'rare_foil')
        ordinary_vs_unrelated = d_a_from_evidence(odds, item_label, 'ord_old', 'unrelated')
        ordinary_vs_own_foil = d_a_from_evidence(odds, item_label, 'ord_old', 'ord_foil')

        row = {'setting_number': setting_number, 'seed': seed}
        row.update(parameters)
        for label in ITEM_LABELS:
            row['rate_' + label] = rates[label]
        # prediction 1: hits to studied rare faces minus hits to studied ordinary faces
        row['hit_difference_rare_minus_ordinary'] = rates['rare_old'] - rates['ord_old']
        # prediction 2: false alarms to the never-studied partner carrying the rare feature
        # minus false alarms to never-studied ordinary faces
        row['false_alarm_difference_feature_lure_minus_new'] = rates['rare_foil'] - rates['unrelated']
        # prediction 3 as registered: the same rare faces scored against two kinds of lure
        row['lure_set_effect_d_a'] = float(rare_vs_unrelated - rare_vs_own_foil)
        # prediction 3 in the four-term form of rem_final_spec.py (its 'interaction')
        row['interaction_four_term_d_a'] = float((rare_vs_unrelated - ordinary_vs_unrelated)
                                                 - (rare_vs_own_foil - ordinary_vs_own_foil))
        # what the feature buys (rem_final_spec.py): rare against its own foil minus ordinary against unrelated
        row['feature_effect_d_a'] = float(rare_vs_own_foil - ordinary_vs_unrelated)
        rows.append(row)
        if setting_number % 100 == 0:
            print('setting', setting_number, 'of', len(all_settings), flush=True)
    return rows, parameter_names


def write_rows(rows, parameter_names, file_path):
    column_names = (['setting_number', 'seed'] + parameter_names + ['rate_' + label for label in ITEM_LABELS]
                    + ['hit_difference_rare_minus_ordinary', 'false_alarm_difference_feature_lure_minus_new',
                       'lure_set_effect_d_a', 'interaction_four_term_d_a', 'feature_effect_d_a'])
    with open(file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=column_names)
        writer.writeheader()
        for row in rows:
            # numbers written as the shortest text that reads back as the same number
            writer.writerow({name: (repr(value) if isinstance(value, float) else value) for name, value in row.items()})


def count_and_total(text):
    """'712/729  98%' -> (712, 729)."""
    count_text, total_text = text.split()[0].split('/')
    return int(count_text), int(total_text)


def main():
    parser = argparse.ArgumentParser(description='REM rates at the 729 settings of the preregistered sweep.')
    parser.parse_args()
    output_stem = 'rem_729_rates'
    print('rem_729_rates.py: the registered seeds (seed = setting number)')

    rows, parameter_names = run_grid()
    write_rows(rows, parameter_names, os.path.join(PATHS['outputs'], output_stem + '.csv'))

    # ------------------------------------------------------------------ counts
    number_of_settings = len(rows)
    hit_holds = 0
    false_alarm_holds = 0
    lure_set_holds = 0
    interaction_holds = 0
    feature_holds = 0
    hit_failures = []
    for row in rows:
        if row['rate_rare_old'] > row['rate_ord_old']:
            hit_holds += 1
        else:
            hit_failures.append(row)
        if row['rate_rare_foil'] > row['rate_unrelated']:
            false_alarm_holds += 1
        if row['lure_set_effect_d_a'] > 0:
            lure_set_holds += 1
        if row['interaction_four_term_d_a'] > 0:
            interaction_holds += 1
        if row['feature_effect_d_a'] > 0:
            feature_holds += 1

    lines = ['REM at the 729 settings of the preregistered sweep, rates and d_a per setting',
             '============================================================================', '',
             'seeds: the registered ones (seed = setting number)',
             'settings: %d' % number_of_settings,
             'prediction 1, hits: rare > ordinary                        %d/%d' % (hit_holds, number_of_settings),
             'prediction 2, false alarms: rare foil > unrelated          %d/%d' % (false_alarm_holds, number_of_settings),
             'prediction 3, lure-set effect (registered two-term form)   %d/%d' % (lure_set_holds, number_of_settings),
             'prediction 3, interaction (four-term form of the report)   %d/%d' % (interaction_holds, number_of_settings),
             'feature effect (what the feature buys) > 0                 %d/%d' % (feature_holds, number_of_settings),
             '']
    mean_text = []
    for label in ITEM_LABELS:
        mean_text.append('%s %.3f' % (label, np.mean([row['rate_' + label] for row in rows])))
    lines.append('mean rates over the grid: ' + '  '.join(mean_text))
    for column, description in (('hit_difference_rare_minus_ordinary', 'hit difference (rare minus ordinary)'),
                                ('false_alarm_difference_feature_lure_minus_new', 'false alarm difference (lure minus new)'),
                                ('lure_set_effect_d_a', 'lure-set effect, d_a'),
                                ('interaction_four_term_d_a', 'interaction, four-term, d_a'),
                                ('feature_effect_d_a', 'feature effect, d_a')):
        values = np.array([row[column] for row in rows])
        lines.append('%-42s mean %+.3f, median %+.3f, range %+.3f to %+.3f'
                     % (description + ':', values.mean(), np.median(values), values.min(), values.max()))

    # ------------------------------------------------------------------ the settings where the hit prediction fails
    lines += ['', 'settings where hits do not favor rare faces: %d' % len(hit_failures)]
    for row in hit_failures:
        lines.append('  g %.1f  w %2d  u %.1f  c %.1f  nfeat %d  fa %.2f   rare %.3f  ordinary %.3f'
                     % (row['g'], row['w'], row['u'], row['c'], row['nfeat'], row['fa'],
                        row['rate_rare_old'], row['rate_ord_old']))
    if hit_failures:
        lowest_hit_rate = min(min(row['rate_rare_old'], row['rate_ord_old']) for row in hit_failures)
        highest_hit_rate = max(max(row['rate_rare_old'], row['rate_ord_old']) for row in hit_failures)
        failures_one_feature = sum(1 for row in hit_failures if row['nfeat'] == 1)
        failures_both_above = sum(1 for row in hit_failures
                                  if row['rate_rare_old'] > 0.89 and row['rate_ord_old'] > 0.89)
        largest_shortfall = max(row['rate_ord_old'] - row['rate_rare_old'] for row in hit_failures)
        lines += ['  hit rates of these settings: %.3f to %.3f (both rates); with the rare value on 1 feature: %d; '
                  'both hit rates above 0.89: %d; largest shortfall %.4f'
                  % (lowest_hit_rate, highest_hit_rate, failures_one_feature, failures_both_above, largest_shortfall)]

    # ------------------------------------------------------------------ checks against the registered report
    paper_lines = face_study_common.read_lines(os.path.join(PATHS['paper_run'], 'rem_final_spec.txt'))
    registered_hits = count_and_total(face_study_common.find_row(paper_lines, 'hits: rare > ordinary'))
    registered_false_alarms = count_and_total(face_study_common.find_row(paper_lines, 'FA: rare foil > unrelated'))
    registered_interaction = count_and_total(face_study_common.find_row(paper_lines, 'interaction > 0'))
    registered_buys = count_and_total(face_study_common.find_row(paper_lines, 'positive (a net benefit survives)'))
    registered_means = face_study_common.find_row(paper_lines, 'mean rates:').split()
    this_run_means = []
    for label in ITEM_LABELS:
        this_run_means.append('%.3f' % np.mean([row['rate_' + label] for row in rows]))
    checks = [('hits: rare > ordinary', (hit_holds, number_of_settings), registered_hits),
              ('FA: rare foil > unrelated', (false_alarm_holds, number_of_settings), registered_false_alarms),
              ('interaction > 0 (four-term)', (interaction_holds, number_of_settings), registered_interaction),
              ('feature effect > 0', (feature_holds, number_of_settings), registered_buys),
              ('mean rates', tuple(this_run_means), tuple(registered_means[1::2]))]
    lines += ['', 'checks against the registered report (data/paper_run/rem_final_spec.txt, read by row label):']
    all_equal = True
    for name, this_run, registered in checks:
        equal = this_run == registered
        all_equal = all_equal and equal
        lines.append('  %-30s this run %s   registered %s   %s' % (name, this_run, registered,
                                                                   'equal' if equal else 'DIFFERENT'))
    lines.append('  all equal: %s' % ('yes' if all_equal else 'NO'))

    report = '\n'.join(lines) + '\n'
    with open(os.path.join(PATHS['outputs'], output_stem + '_check.txt'), 'w', encoding='utf-8') as handle:
        handle.write(report)
    print(report)


if __name__ == '__main__':
    main()
