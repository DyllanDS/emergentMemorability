"""
check_provided.py: every number the paper reports about the continuous recognition simulation (SI
Appendix, Section 2, Continuous Recognition, and Fig. S4), computed from the model run and the THINGS
data side and written to outputs/numbers.csv (one row per ledger item).

INPUTS
  the model run: data/provided/continuous_bins.npz and continuous_rem.txt (the saved run the paper
      reports; the trial structures it needs are not in the release, EXTERNAL_INPUTS.md M6 and M1),
      or a run of continuous_rem.py in outputs/ (--model-run regenerated)
  the THINGS data side (provided output of module A5; regenerating it needs the Kramer et al. trial
      file, M1): ../A5_things_context_effects_figS13_S14/data/provided/sdt_ctx_top10_target.npz and
      sdt_results.txt, the participants' rates in
      ../A3b_hits_and_false_alarms_figS9/data/provided/per_image_rates.txt
  outputs/model_mechanics_check.csv (check_model_mechanics.py) and the figure files of
      make_figureS4.py (run both first; run_module.py does)
  shared/ledger/paper_numbers.csv (the paper's values)
Each npz is checked against its text twin (read by row label) before any number is used.

OUTPUTS
  outputs/numbers.csv                 ledger_id, location, quote, paper_value, computed_value, match,
                                      match_type, script, mode, note
  outputs/equality_with_paper_run.csv (only with --model-run regenerated) the new model run against
                                      the saved run in data/provided/, array by array
Each row is judged with the match type of the ledger (shared/python/numbers_table.py).
No random numbers. Runtime: about a second.

RUN (from the repository root)
  python A2c_continuous_recognition_figS4/code/check_provided.py
  python A2c_continuous_recognition_figS4/code/check_provided.py --model-run regenerated
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.stats import norm

sys.dont_write_bytecode = True   # write no __pycache__ folder into shared/python or code/ (set before the imports below)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402
import numbers_table  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import continuous_rem  # noqa: E402
import figs4_inputs  # noqa: E402

PATHS = release_paths.module_paths(__file__)
SCRIPT = 'code/check_provided.py'
FALSE_ALARM_TOLERANCE = 0.002   # a false alarm rate within 0.002 of the 0.020 target counts as on target
LEDGER = numbers_table.load_ledger()


def printed(ledger_id):
    """The paper's value for a ledger item (shared/ledger/paper_numbers.csv)."""
    return LEDGER[ledger_id]['paper_value']


def read_mechanics_check():
    """The rows of outputs/model_mechanics_check.csv (written by check_model_mechanics.py)."""
    path = os.path.join(PATHS['outputs'], 'model_mechanics_check.csv')
    if not os.path.exists(path):
        sys.exit('check_provided.py: outputs/model_mechanics_check.csv is missing; run '
                 'python A2c_continuous_recognition_figS4/code/check_model_mechanics.py first')
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def templates_loaded_by_the_run(model):
    """How many trial structures the model run used, and where the number comes from. For a run of
    continuous_rem.py in this release: the trial structures in the input it reads
    (continuous_rem.find_templates(): data/trial_templates_400.npz or module A5's trial table). For the
    saved run in data/provided/: the code's setting N_TEMPLATES."""
    if model['source'] == 'regenerated':
        templates, source = continuous_rem.find_templates()
        if templates is not None:
            return len(templates), 'the trial structures in the %s' % source
    return continuous_rem.N_TEMPLATES, ('continuous_rem.py, N_TEMPLATES: the first %d sessions with %d trials in the '
                                        'Kramer et al. (2023) trial data'
                                        % (continuous_rem.N_TEMPLATES, continuous_rem.TEMPLATE_LENGTH))


def compare_with_paper_run(model):
    """A new run against the saved run in data/provided/: every array and number of continuous_bins.npz."""
    paper = np.load(os.path.join(PATHS['provided'], 'continuous_bins.npz'))
    regenerated = np.load(model['files']['npz'])
    rows = []
    for key in sorted(paper.files):
        paper_values = np.atleast_1d(np.array(paper[key], dtype=float))
        if key not in regenerated.files:
            rows.append(dict(key=key, paper_run='', this_run='missing', largest_difference='', identical='no',
                             equal_within_1e_12='no'))
            continue
        new_values = np.atleast_1d(np.array(regenerated[key], dtype=float))
        same_shape = paper_values.shape == new_values.shape
        largest = float(np.max(np.abs(paper_values - new_values))) if same_shape else float('nan')
        identical = same_shape and np.array_equal(paper_values, new_values)
        close = same_shape and largest <= 1e-12
        rows.append(dict(key=key, paper_run=numbers_table.format_computed(paper_values.tolist() if paper_values.size > 1
                                                                          else float(paper_values[0])),
                         this_run=numbers_table.format_computed(new_values.tolist() if new_values.size > 1
                                                                else float(new_values[0])),
                         largest_difference='%.3g' % largest, identical='yes' if identical else 'no',
                         equal_within_1e_12='yes' if close else 'no'))
    path = os.path.join(PATHS['outputs'], 'equality_with_paper_run.csv')
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['key', 'paper_run', 'this_run', 'largest_difference', 'identical',
                                                    'equal_within_1e_12'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    number_identical = sum(1 for row in rows if row['identical'] == 'yes')
    number_close = sum(1 for row in rows if row['equal_within_1e_12'] == 'yes')
    print('this run against the saved run in data/provided/: %d of %d arrays identical, %d of %d equal within 1e-12 '
          '(%s)' % (number_identical, len(rows), number_close, len(rows),
                    os.path.relpath(path, release_paths.release_root())))
    return number_close == len(rows)


def main():
    parser = argparse.ArgumentParser(description='Numbers of SI Appendix, Section 2 (Continuous Recognition) and '
                                                 'Fig. S4 -> outputs/numbers.csv')
    parser.add_argument('--model-run', choices=['provided', 'regenerated', 'auto'], default='provided',
                        help="which model run to use: the saved run (data/provided/, default), a new run of "
                             "continuous_rem.py (outputs/), or auto (the new run when outputs/continuous_bins.npz "
                             "exists)")
    arguments = parser.parse_args()

    model = figs4_inputs.model_run(arguments.model_run)
    data = figs4_inputs.things_data()
    mechanics = read_mechanics_check()
    print('model run: %s' % model['label'])
    print('THINGS data side: provided output of module A5 (%s); regenerating it needs the Kramer et al. trial '
          'file (EXTERNAL_INPUTS.md M1)' % figs4_inputs.relative(data['files']['npz']))
    print('both files agree with their text reports (checked by row label)')

    standard = model['standard']
    matched = model['accuracy-matched']
    data_dprime = figs4_inputs.describe_curve(data['dprime'])
    data_false_alarms = figs4_inputs.describe_curve(data['F'])
    data_criterion = figs4_inputs.describe_curve(data['crit'])
    standard_dprime = figs4_inputs.describe_curve(standard['dprime'])
    matched_dprime = figs4_inputs.describe_curve(matched['dprime'])
    standard_false_alarms = figs4_inputs.describe_curve(standard['F'])
    matched_false_alarms = figs4_inputs.describe_curve(matched['F'])
    matched_criterion = figs4_inputs.describe_curve(matched['crit_bin'])
    number_of_bins = len(data['dprime'])

    regenerated = model['source'] == 'regenerated'
    model_mode = 'rerun' if regenerated else 'provided'
    mixed_mode = 'rerun_and_provided' if regenerated else 'provided'
    model_source = ('model: continuous_rem.py run in this release (outputs/)' if regenerated
                    else 'model: the saved run (data/provided/)')
    data_source = 'data: module A5\'s provided sdt_ctx_top10_target.npz and sdt_results.txt'
    table = numbers_table.NumbersTable(script=SCRIPT)

    # ---------------------------------------------------------------- SI Appendix, Section 2: the three changes
    change_1 = all(row['passed'] == 'yes' for row in mechanics if row['check'].startswith('1.'))
    change_2 = all(row['passed'] == 'yes' for row in mechanics if row['check'].startswith('2.'))
    # the searched criteria are not REM's usual 1.0, and the false alarm rates they produce are within
    # FALSE_ALARM_TOLERANCE of the 2 percent target (the search uses 100 simulated participants per step)
    criteria_not_usual = (standard['criterion'] != continuous_rem.USUAL_REM_CRITERION and
                          matched['criterion'] != continuous_rem.USUAL_REM_CRITERION)
    standard_distance = abs(standard['overall_fa'] - continuous_rem.TARGET_FALSE_ALARM_RATE)
    matched_distance = abs(matched['overall_fa'] - continuous_rem.TARGET_FALSE_ALARM_RATE)
    false_alarms_on_target = standard_distance < FALSE_ALARM_TOLERANCE and matched_distance < FALSE_ALARM_TOLERANCE
    change_3 = criteria_not_usual and false_alarms_on_target
    changes_verified = int(change_1) + int(change_2) + int(change_3)
    table.add('L6-048', changes_verified, mode='rerun' if regenerated else 'rerun_and_provided',
              script='code/check_model_mechanics.py; ' + SCRIPT,
              note='changes that adapt REM to continuous recognition and pass their check: (1) every earlier trace '
                   'is compared and the model says "old" when the average likelihood ratio exceeds the criterion: %s; '
                   '(2) every presentation is stored as a new trace: %s (both checked by check_model_mechanics.py '
                   'against a version with explicit loops, on a made-up stream; outputs/model_mechanics_check.csv); '
                   '(3) the criterion is set to the participants\' false alarm rate: %s (criteria %.3f at the standard '
                   'storage probability and %.3f at the matched one, not 1.0, giving false alarm rates of %.4f and '
                   '%.4f, within %.3f of the target %.3f; %s)'
                   % ('yes' if change_1 else 'no', 'yes' if change_2 else 'no', 'yes' if change_3 else 'no',
                      standard['criterion'], matched['criterion'], standard['overall_fa'], matched['overall_fa'],
                      FALSE_ALARM_TOLERANCE, continuous_rem.TARGET_FALSE_ALARM_RATE, model_source))
    table.add('L6-049', continuous_rem.USUAL_REM_CRITERION, mode='rerun', script='code/continuous_rem.py; ' + SCRIPT,
              note="REM's usual criterion (1.0, as in the study-test simulations); the continuous recognition runs "
                   'use the searched criteria %.3f (storage %.2f) and %.3f (storage %.2f), both different from 1.0: '
                   '%s (%s)'
                   % (standard['criterion'], standard['storage'], matched['criterion'], matched['storage'],
                      'yes' if criteria_not_usual else 'no', model_source))
    participants_percent = 100.0 * data['participants_fa_rate']
    table.add('L6-050', participants_percent, mode=mixed_mode,
              script=('code/continuous_rem.py; A3b_hits_and_false_alarms_figS9/data/provided/per_image_rates.txt; '
                      + SCRIPT),
              note="the participants' false alarm rate in percent: the mean per-image false alarm rate, %.4f, over "
                   '%s images with at least 30 old and 30 new trials (module A3b, data/provided/per_image_rates.txt); '
                   'the criterion search targets %.3f and the model reaches %.4f (storage %.2f) and %.4f (storage '
                   '%.2f)'
                   % (data['participants_fa_rate'], '{:,}'.format(data['per_image_n']),
                      continuous_rem.TARGET_FALSE_ALARM_RATE, standard['overall_fa'], standard['storage'],
                      matched['overall_fa'], matched['storage']))
    table.add('L6-051', 'traces never removed or weakened: %s' % ('yes' if change_2 else 'no'), mode='rerun',
              script='code/check_model_mechanics.py',
              note='continuous_rem.py never removes or weakens a trace: on a made-up test stream it gives the same '
                   'responses, with the same random numbers, as a version that keeps every trace '
                   '(check_model_mechanics.py, outputs/model_mechanics_check.csv)')
    table.add('L6-052', 'old when the average likelihood ratio over all earlier traces exceeds the criterion: %s'
              % ('yes' if change_1 else 'no'), mode='rerun', script='code/check_model_mechanics.py',
              note='continuous_rem.py says "old" when the log of the mean likelihood ratio over all earlier traces '
                   'exceeds the log of the criterion; on a made-up test stream it gives the same responses as a '
                   'feature-by-feature version at both storage probabilities and criteria (check_model_mechanics.py, '
                   'outputs/model_mechanics_check.csv)')

    # ---------------------------------------------------------------- SI Appendix, Section 2: the sequences
    table.add('L6-053', model['trials_per_sequence'], mode=model_mode,
              note='trials per simulated sequence (%d in the model run\'s report); the model uses sessions with %d '
                   'trials, the first %d in the order of the trial table'
                   % (model['trials_per_sequence'], continuous_rem.TEMPLATE_LENGTH, continuous_rem.N_TEMPLATES))
    templates_used, templates_where = templates_loaded_by_the_run(model)
    table.add('L6-054', templates_used, mode=model_mode, script='code/continuous_rem.py; ' + SCRIPT,
              note='trial sequences the model runs through (%s)'
                   % templates_where)
    per_sequence = model['simulated_participants'] / float(templates_used)
    table.add('L6-055', per_sequence, mode=model_mode, script='code/continuous_rem.py; ' + SCRIPT,
              note='simulated participants per sequence: %s simulated participants per storage probability (the run\'s '
                   'report) over %d sequences; simulated participant i follows sequence i modulo %d'
                   % ('{:,}'.format(model['simulated_participants']), templates_used, templates_used))
    table.add('L6-056', 'items: %s generated items of %d geometric features (g %.2f); similarity: proportion of features '
                        'with the same value; context: mean similarity of the %d most similar preceding items'
              % ('{:,}'.format(continuous_rem.N_DATABASE), continuous_rem.W, continuous_rem.G, continuous_rem.TOPK),
              mode=model_mode, script='code/continuous_rem.py',
              note='settings of continuous_rem.py (main, build_similarity_matrix, run_subject): each distinct image of '
                   'a sequence is replaced by a different item, drawn without replacement from the %s generated '
                   'items, so how similar the preceding items are is chance similarity among generated items'
                   % '{:,}'.format(continuous_rem.N_DATABASE))
    participants_dprime = norm.ppf(data['participants_hit_rate']) - norm.ppf(data['participants_fa_rate'])
    standard_overall_dprime = norm.ppf(standard['overall_hit']) - norm.ppf(standard['overall_fa'])
    matched_overall_dprime = norm.ppf(matched['overall_hit']) - norm.ppf(matched['overall_fa'])
    table.add('L6-057', [standard['storage'], matched['storage']], mode=model_mode,
              note='storage probabilities of the model run: the standard one, and the value of %s whose overall d\' in '
                   'a run of 200 simulated participants is closest to the target d\' of %.2f (TARGET_DPRIME); overall '
                   'd\' of the model run (aggregate rates) %.2f at storage %.2f and %.2f at storage %.2f; participants '
                   '%.2f (mean per-image hit rate %.4f, false alarm rate %.4f) (%s)'
                   % (', '.join('%.2f' % value for value in continuous_rem.STORAGE_GRID), continuous_rem.TARGET_DPRIME,
                      standard_overall_dprime, standard['storage'], matched_overall_dprime, matched['storage'],
                      participants_dprime, data['participants_hit_rate'], data['participants_fa_rate'],
                      model_source))
    declines = [-data_dprime['change'], -standard_dprime['change'], -matched_dprime['change']]
    table.add('L6-058', declines, mode=mixed_mode,
              script=('code/continuous_rem.py; A5_things_context_effects_figS13_S14/code/sdt_by_context.py '
                      '(saved results); ' + SCRIPT),
              note="fall in within-item d' from bin 1 (least similar preceding items) to bin 20 (most similar): data "
                   '%.3f to %.3f, model at storage %.2f %.3f to %.3f, model at storage %.2f %.3f to %.3f (%s; %s)'
                   % (data_dprime['first'], data_dprime['last'], standard['storage'], standard_dprime['first'],
                      standard_dprime['last'], matched['storage'], matched_dprime['first'], matched_dprime['last'],
                      model_source, data_source))
    criterion_rhos = [matched_criterion['rho'], data_criterion['rho']]
    table.add('L6-059', criterion_rhos, mode=mixed_mode,
              script=('code/continuous_rem.py; A5_things_context_effects_figS13_S14/code/sdt_by_context.py '
                      '(saved results); ' + SCRIPT),
              note='Spearman correlation of the binned criterion c with the bin number (%d bins): model at storage %.2f '
                   '%.4f (c from %+.3f to %+.3f), data %.4f (c from %+.3f to %+.3f)'
                   % (number_of_bins, matched['storage'], matched_criterion['rho'], matched_criterion['first'], matched_criterion['last'],
                      data_criterion['rho'], data_criterion['first'], data_criterion['last']))
    table.add('L6-261', 'one criterion for every trial at storage %.2f: %.3f; false alarm rate on first showings '
                        'from %.4f (bin 1) to %.4f (bin 20)'
              % (matched['storage'], matched['criterion'], matched_false_alarms['first'], matched_false_alarms['last']),
              mode=model_mode, script='code/continuous_rem.py; ' + SCRIPT,
              note='the model compares the log of the mean likelihood ratio with the log of one fixed criterion on '
                   'every trial (run_subject); the measured c falls from %+.3f to %+.3f because the false alarm rate '
                   'on first showings rises with the similarity of the preceding items (%s)'
                   % (matched_criterion['first'], matched_criterion['last'], model_source))
    table.add('L6-262', 'bins ranked over the whole session, with no adjustment for serial position or lag',
              mode=mixed_mode,
              script=('code/continuous_rem.py; A5_things_context_effects_figS13_S14/code/sdt_by_context.py '
                      '(saved results); ' + SCRIPT),
              note='model: continuous_rem.py bins each item\'s trials on context alone; data: the whole-stream '
                   'curves of module A5 (sdt_ctx_top10_target.npz, the open circles of SI Appendix, Fig. S13), '
                   'not its curves with serial position and lag held constant (sdt_by_context_adjusted.npz, the '
                   'filled circles)')

    # ---------------------------------------------------------------- Fig. S4
    figure_file = os.path.join(PATHS['figures'], 'FigS4_continuous_recognition.png')
    for ledger_id, panel, data_file in (('L6-060', 'a', 'FigS4a.csv'), ('L6-061', 'b', 'FigS4b.csv')):
        figure_present = os.path.exists(figure_file) and os.path.exists(os.path.join(PATHS['figure_data'], data_file))
        if not figure_present:
            sys.exit('check_provided.py: the figure files are missing; run '
                     'python A2c_continuous_recognition_figS4/code/make_figureS4.py first')
        table.add(ledger_id, 'outputs/figures/FigS4_continuous_recognition.png, panel %s (outputs/figure_data/%s)'
                  % (panel, data_file), mode=mixed_mode, script='code/make_figureS4.py',
                  note=('Fig. S4a: change in d\' from bin 1; data points with each bin\'s 95 percent participant-bootstrap '
                        "limits shifted by the bin 1 d'; model lines at storage 0.30 (solid) and 0.75 (dashed), "
                        'without intervals' if panel == 'a' else
                        'Fig. S4b: false alarm rate per bin; data points with their 95 percent participant-bootstrap '
                        'limits; model lines at storage 0.30 (solid) and 0.75 (dashed), without intervals') +
                  '; %s; %s' % (model_source, data_source))

    # ---------------------------------------------------------------- Fig. S4 caption
    table.add('L6-062', templates_used, mode=model_mode, script='code/continuous_rem.py; ' + SCRIPT,
              note='trial sequences the model runs through, the same as in SI Appendix, Section 2 (L6-054; %s)'
                   % templates_where)
    bins_and_neighbours = [len(standard['dprime']), continuous_rem.TOPK]
    same_bins_on_both_sides = (len(standard['dprime']) == len(matched['dprime']) == len(data['dprime'])
                               == data['n_bins'] == continuous_rem.N_BINS)
    table.add('L6-063', bins_and_neighbours, mode=mixed_mode,
              script=('code/continuous_rem.py; A5_things_context_effects_figS13_S14/code/compute_context.py and '
                      'sdt_by_context.py (saved results); ' + SCRIPT),
              note='%d bins within item on both sides (model run %d and %d bins, data %d; the same on both sides: %s); '
                   'context = mean similarity of the %d most similar preceding items (model TOPK = %d; data: the '
                   'top-10 context of module A5\'s file, sdt_ctx_top10); a trial is binned once at least %d earlier '
                   'trials showed other items in the model and %d in the data; similarity is the proportion of '
                   'shared feature values in the model and the correlation of the 49 SPoSE dimensions in the data'
                   % (continuous_rem.N_BINS, len(standard['dprime']), len(matched['dprime']), len(data['dprime']),
                      'yes' if same_bins_on_both_sides else 'no', continuous_rem.TOPK, continuous_rem.TOPK,
                      continuous_rem.MIN_PRIOR, figs4_inputs.DATA_MIN_PRIOR))
    table.add('L6-064', data['n_sessions'], mode='provided',
              script='A5_things_context_effects_figS13_S14/code/sdt_by_context.py (saved results); ' + SCRIPT,
              note='participant sessions in the Kramer et al. (2023) trial data behind the data points, from module '
                   'A5\'s data/provided/sdt_results.txt ("%s participants; context binned into %d within-image bins")'
                   % ('{:,}'.format(data['n_sessions']), data['n_bins']))
    table.add('L6-065', [standard['storage'], matched['storage']], mode=model_mode,
              note='storage probabilities of the solid ("Standard") and dashed ("Matched Accuracy") model lines, as in '
                   'L6-057')

    # ---------------------------------------------------------------- SI Appendix, Section 2: Spearman ranges
    dprime_rhos = [data_dprime['rho'], standard_dprime['rho'], matched_dprime['rho']]
    dprime_rho_range = [min(dprime_rhos), max(dprime_rhos)]
    table.add('L6-066', dprime_rho_range, mode=mixed_mode,
              script=('code/continuous_rem.py; A5_things_context_effects_figS13_S14/code/sdt_by_context.py '
                      '(saved results); ' + SCRIPT),
              note="the most and least negative Spearman correlation of the binned d' with the bin number (%d bins: "
                   'data %.4f, storage %.2f %.4f, storage %.2f %.4f)'
                   % (number_of_bins, data_dprime['rho'], standard['storage'], standard_dprime['rho'],
                      matched['storage'], matched_dprime['rho']))
    false_alarm_ratios = [data_false_alarms['ratio'], standard_false_alarms['ratio'], matched_false_alarms['ratio']]
    false_alarm_rhos = [data_false_alarms['rho'], standard_false_alarms['rho'], matched_false_alarms['rho']]
    false_alarm_summary = [min(false_alarm_ratios[0], false_alarm_ratios[1]), false_alarm_ratios[2],
                           min(false_alarm_rhos), max(false_alarm_rhos)]
    doubling = numbers_table.parse_number(printed('L6-067').split(';')[0])     # the 2 of "more than doubled"
    more_than_doubled = false_alarm_ratios[0] > doubling and false_alarm_ratios[1] > doubling
    table.add('L6-067', false_alarm_summary, mode=mixed_mode,
              script=('code/continuous_rem.py; A5_things_context_effects_figS13_S14/code/sdt_by_context.py '
                      '(saved results); ' + SCRIPT),
              note='false alarm rate in bin 20 divided by bin 1: data %.4f to %.4f (x%.2f), model at storage %.2f %.4f '
                   'to %.4f (x%.2f), model at storage %.2f %.4f to %.4f (x%.2f); the first value is the smaller of the '
                   'data and standard ratios (both above %g: %s), the second the matched-accuracy ratio; then the '
                   'lowest and highest Spearman correlation of the binned false alarm rate with the bin number (%d '
                   'bins: data %.4f, model %.4f and %.4f)'
                   % (data_false_alarms['first'], data_false_alarms['last'], false_alarm_ratios[0], standard['storage'],
                      standard_false_alarms['first'], standard_false_alarms['last'], false_alarm_ratios[1],
                      matched['storage'], matched_false_alarms['first'], matched_false_alarms['last'],
                      false_alarm_ratios[2], doubling, 'yes' if more_than_doubled else 'no', number_of_bins,
                      false_alarm_rhos[0],
                      false_alarm_rhos[1], false_alarm_rhos[2]))
    table.add('L6-068', data['ci_level'], mode='provided',
              script='A5_things_context_effects_figS13_S14/code/sdt_by_context.py (saved results); ' + SCRIPT,
              note='level of the data error bars: %d percent percentile intervals of %d bootstrap resamples of the %s '
                   'participant sessions (module A5, data/provided/sdt_results.txt), for example bin 1 d\' [%.3f, '
                   '%.3f]; the model lines have no intervals'
                   % (data['ci_level'], data['n_bootstraps'], '{:,}'.format(data['n_sessions']), data['dlo'][0],
                      data['dhi'][0]))

    table.write(PATHS['outputs'])
    if regenerated:
        compare_with_paper_run(model)

    # summary for the log: the paper value, the computed value and the match of every row
    print('\nledger item   paper value                     computed                         match')
    for row in table.rows:
        verdict = row['match']
        print('%-12s  %-30s  %-31s  %s' % (row['ledger_id'], row['paper_value'][:30], row['computed_value'][:31], verdict))


if __name__ == '__main__':
    main()
