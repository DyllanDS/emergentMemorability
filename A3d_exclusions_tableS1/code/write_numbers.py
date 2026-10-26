"""
write_numbers.py: outputs/numbers.csv of module A3d, one row for each of its 28 ledger items, from SI
Appendix, Section 3, "The Image Set Results Do Not Depend on the Exclusions" (Table S1), from the
main-text sentence on the exclusion criteria and from the Statistical Reporting paragraph of the
Methods.

Every computed value is read from what steps 1 to 3 (code/exclusion_regimes.py, code/stats_tableS1.py
and code/make_tableS1.py) wrote in outputs/ (run them first), counted from the data in the release or,
for the level of the concept-level intervals, read from shared/python/bootstrap.py;
the printed values, locations and quotes come from shared/ledger/paper_numbers.csv
(shared/python/numbers_table.py judges each match).

READS   outputs/participant_flags.csv, concept_level_by_regime.csv, feature_level_by_regime.csv,
        fig7d_reduction_by_regime.csv, reproduction_check.csv, exclusion_regimes.txt (step 1)
        outputs/tableS1.csv (step 3)
        06_image_set_experiments_fig6_fig7/data/<set>_trials.csv (the study lists, for the 40 items
        per block)
        shared/things/concept_table.csv (the number of SPoSE features)
        shared/python/bootstrap.py (the level of percentile_interval, with which step 2 takes the
        concept-level intervals)
WRITES  outputs/numbers.csv

RUN (from the repository root; seconds):
    python A3d_exclusions_tableS1/code/write_numbers.py
"""
import inspect
import os
import re
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True   # no __pycache__ folder in shared/ when this step is run on its own
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (first: it stops Python from writing __pycache__ folders)
import bootstrap  # noqa: E402
import concept_tables  # noqa: E402
import imageset_loader  # noqa: E402
import numbers_table  # noqa: E402

PATHS = release_paths.module_paths(__file__, make_output_folders=False)
OUTPUTS = PATHS['outputs']
SET_NAMES = ('maintained', 'changed', 'reduced')
SCRIPT_REGIMES = 'code/exclusion_regimes.py'
SCRIPT_STATS = 'code/stats_tableS1.py'
SCRIPT_TABLE = 'code/make_tableS1.py'

# The cells of SI Appendix, Table S1 with an estimate and a 95% interval in each column: ledger item, row of
# outputs/tableS1.csv, where the interval comes from and the script that computes the cells (make_tableS1.py
# only formats them)
TABLE_S1_INTERVAL_ROWS = [
    ('L7-056', 'Concept level, maintained set (\u03c1)', 'concept resamples, outputs/tests.csv T35', SCRIPT_STATS),
    ('L7-141', 'Concept level, changed set (\u03c1)', 'concept resamples, outputs/tests.csv T35', SCRIPT_STATS),
    ('L7-142', 'Concept level, reduced set (\u03c1)', 'concept resamples, outputs/tests.csv T35', SCRIPT_STATS),
    ('L7-058', 'Feature level, maintained set (r)', 'participant resamples, outputs/tests.csv T36', SCRIPT_STATS),
    ('L7-143', 'Feature level, changed set (r)', 'participant resamples, outputs/tests.csv T36', SCRIPT_STATS),
    ('L7-053', 'Variance explained, maintained set',
     'mean and percentiles over the 1,000 participant resamples of step 1, seed 7, outputs/tests.csv T37',
     SCRIPT_REGIMES),
    ('L7-144', 'Variance explained, reduced set',
     'mean and percentiles over the 1,000 participant resamples of step 1, seed 7, outputs/tests.csv T37',
     SCRIPT_REGIMES),
]
TABLE_S1_REDUCTION_ROW = 'Reduction, maintained to reduced'


def read_output(file_name, written_by=SCRIPT_REGIMES):
    path = os.path.join(OUTPUTS, file_name)
    if not os.path.exists(path):
        sys.exit('write_numbers.py: outputs/%s is missing; run %s first' % (file_name, written_by))
    return pd.read_csv(path)


def read_text(file_name, written_by):
    path = os.path.join(OUTPUTS, file_name)
    if not os.path.exists(path):
        sys.exit('write_numbers.py: outputs/%s is missing; run %s first' % (file_name, written_by))
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def one_value(table, set_name, regime, column):
    row = table[(table['image_set'] == set_name) & (table['regime'] == regime)]
    if len(row) != 1:
        raise ValueError('expected one row for %s / %s' % (set_name, regime))
    return float(row[column].iloc[0])


def main():
    flags = read_output('participant_flags.csv')
    concept_level = read_output('concept_level_by_regime.csv')
    feature_level = read_output('feature_level_by_regime.csv')
    reduction = read_output('fig7d_reduction_by_regime.csv').set_index('regime')
    reproduction = read_output('reproduction_check.csv').set_index('statistic')
    table_s1 = read_output('tableS1.csv', SCRIPT_TABLE).set_index('result')
    regimes_text = read_text('exclusion_regimes.txt', SCRIPT_REGIMES)
    table = numbers_table.NumbersTable(script=SCRIPT_REGIMES)

    # ---------------- participants ----------------
    run_per_set = {set_name: int(np.sum(flags['image_set'] == set_name)) for set_name in SET_NAMES}
    excluded_per_set = {set_name: int(np.sum((flags['image_set'] == set_name) & flags['excluded_005']))
                        for set_name in SET_NAMES}
    kept_per_set = {set_name: int(one_value(concept_level, set_name, 'reported', 'participants')) for set_name in SET_NAMES}
    trials_all_run = {set_name: int(one_value(concept_level, set_name, 'all_run', 'trials')) for set_name in SET_NAMES}
    total_run = sum(run_per_set.values())
    total_excluded = sum(excluded_per_set.values())
    total_kept = sum(kept_per_set.values())
    kept_each = kept_per_set['maintained'] if len(set(kept_per_set.values())) == 1 else list(kept_per_set.values())
    all_run_participants = sum(int(one_value(concept_level, set_name, 'all_run', 'participants')) for set_name in SET_NAMES)
    test_trials_per_session = sorted(set(int(number) for number in flags['test_trials']))
    per_set_text = ', '.join('%s %d run / %d excluded / %d kept' % (set_name, run_per_set[set_name],
                                                                      excluded_per_set[set_name], kept_per_set[set_name])
                             for set_name in SET_NAMES)
    table.add('L7-043', [total_excluded, total_run, kept_each], mode='rerun',
              note='participants excluded by the Methods rules over the three image sets; sessions run; participants '
                   'kept per set (%s)' % per_set_text)
    table.add('L7-044', all_run_participants, mode='rerun',
              note='sessions in the no-exclusion regime, every test trial kept (no response-time filter): %s '
                   'participants; %s test trials (%s per participant)' % (
                       ' / '.join(str(run_per_set[s]) for s in SET_NAMES),
                       ' / '.join('{:,}'.format(trials_all_run[s]) for s in SET_NAMES),
                       ' or '.join(str(number) for number in test_trials_per_session)))
    table.add('L7-054', [total_run, total_kept, kept_each], mode='rerun',
              note='sessions run (Table S1, without exclusions); participants kept after the exclusions (with '
                   'exclusions); participants kept per set')

    # ---------------- concept level (Table S1) ----------------
    rho_all = [one_value(concept_level, s, 'all_run', 'rho') for s in SET_NAMES]
    p_all = [one_value(concept_level, s, 'all_run', 'p') for s in SET_NAMES]
    rho_with = [one_value(concept_level, s, 'reported', 'rho') for s in SET_NAMES]
    rho_stricter = [one_value(concept_level, s, 'stricter_010', 'rho') for s in SET_NAMES]
    concepts = [int(one_value(concept_level, s, 'all_run', 'concepts')) for s in SET_NAMES]
    concepts_with = [int(one_value(concept_level, s, 'reported', 'concepts')) for s in SET_NAMES]
    changes = [with_value - without_value for with_value, without_value in zip(rho_with, rho_all)]
    largest_change = max(abs(change) for change in changes)
    all_stronger = all(change < 0 for change in changes)
    p_text = ', '.join('%.1e' % p for p in p_all)
    table.add('L7-045', max(p_all), mode='rerun',
              note='largest of the three two-sided p values of the concept-level Spearman rho without exclusions '
                   '(Table S1; over %s concepts): %s' % (' / '.join(str(n) for n in concepts), p_text))
    stronger_note = ('each more negative with the exclusions' if all_stronger
                     else 'not every change is negative')
    match = table.add('L7-046', largest_change, mode='rerun',
                      note='largest absolute change of the concept-level Spearman rho of Table S1 when the exclusions '
                           'are applied; changes (maintained, changed, reduced) %s (%s)'
                           % (', '.join('%+.4f' % change for change in changes), stronger_note))
    if not all_stronger and match == 'yes':
        table.rows[-1]['match'] = 'no'
    table.add('L7-148', concepts, mode='rerun',
              note='concepts with both values (CR and mean summed similarity) in the maintained, changed and reduced '
                   'sets without exclusions; with exclusions %s' % ' / '.join(str(n) for n in concepts_with))
    table.add('L7-051', 'Spearman rho without -> with exclusions: %s' % '; '.join(
        '%s %.4f -> %.4f' % (s, without_value, with_value) for s, without_value, with_value in
        zip(SET_NAMES, rho_all, rho_with)), mode='rerun',
              note='Table S1, concept level: Spearman rho across the concepts of each set between CR (hit rate minus '
                   'false alarm rate) and the mean summed similarity of the tested concept to the 40 items studied '
                   'in its block (outputs/concept_level_by_regime.csv)')

    # ---------------- feature level (Table S1, Fig. 6d and 6g; the six correlations of Fig. 6) ----------------
    feature_sets = ('maintained', 'changed')
    r_all = [one_value(feature_level, s, 'all_run', 'r_summed_similarity_human') for s in feature_sets]
    number_of_features = concept_tables.spose_features().shape[1]
    table.add('L7-052', number_of_features, mode='rerun',
              note='SPoSE features over which the feature-level correlation of Table S1 (Fig. 6d and 6g) is taken')
    fig6_correlations = [('6c', 'maintained', 'r_original_vs_set'), ('6d', 'maintained', 'r_summed_similarity_human'),
                         ('6e', 'maintained', 'r_summed_similarity_monkey'), ('6f', 'changed', 'r_original_vs_set'),
                         ('6g', 'changed', 'r_summed_similarity_human'), ('6h', 'changed', 'r_summed_similarity_monkey')]
    table.add('L7-047', 'Pearson r without -> with exclusions: %s' % '; '.join(
        'Fig. %s %.4f -> %.4f' % (panel, one_value(feature_level, set_name, 'all_run', column),
                                 one_value(feature_level, set_name, 'reported', column))
        for panel, set_name, column in fig6_correlations), mode='rerun',
              note='the six feature-level correlations of Fig. 6 (6c to 6h) without and with the exclusions; Table S1 '
                   'gives 6d and 6g with their intervals (outputs/tableS1.csv)')

    # ---------------- the cells of Table S1 ----------------
    for ledger_id, result, interval_source, computing_script in TABLE_S1_INTERVAL_ROWS:
        row = table_s1.loc[result]
        table.add(ledger_id, [row['without_estimate'], row['without_ci_low'], row['without_ci_high'],
                              row['with_estimate'], row['with_ci_low'], row['with_ci_high']],
                  mode='rerun', script=computing_script,
                  note='SI Appendix, Table S1 (outputs/tableS1.csv), %s: without exclusions %s, with exclusions %s; '
                       'estimate and 95%% interval (%s)' % (result, row['without_exclusions'],
                                                           row['with_exclusions'], interval_source))
    reduction_cells = table_s1.loc[TABLE_S1_REDUCTION_ROW]
    table.add('L7-060', [reduction_cells['without_estimate'], reduction_cells['without_one_sided_p'],
                         reduction_cells['with_estimate'], reduction_cells['with_one_sided_p']],
              mode='rerun', script=SCRIPT_REGIMES,
              note='SI Appendix, Table S1 (outputs/tableS1.csv), reduction from the maintained to the reduced set: '
                   'without exclusions %s, with exclusions %s (mean percent reduction and one-sided bootstrap p over '
                   'the 1,000 participant resamples of step 1, seed 7)'
                   % (reduction_cells['without_exclusions'], reduction_cells['with_exclusions']))

    # ---------------- the stricter floor ----------------
    # "changed none of these correlations by more than 0.02": the largest change over the concept-level rho
    # of Table S1 (three sets) and every feature-level correlation of Fig. 6 (6c to 6h; Table S1 gives 6d
    # and 6g)
    concept_rho_changes = [abs(stricter - reported) for stricter, reported in zip(rho_stricter, rho_with)]
    fig6_changes = []
    for panel, set_name, column in fig6_correlations:
        stricter_value = one_value(feature_level, set_name, 'stricter_010', column)
        reported_value = one_value(feature_level, set_name, 'reported', column)
        fig6_changes.append(abs(stricter_value - reported_value))
    largest_stricter_change = max(concept_rho_changes + fig6_changes)
    floor_found = re.search(r'stricter CR floor (\d+\.\d+)', regimes_text)
    if floor_found is None:
        raise RuntimeError('the stricter floor is not named in outputs/exclusion_regimes.txt')
    stricter_kept_text = '%d / %d / %d participants kept in the maintained, changed and reduced sets' % tuple(
        int(one_value(concept_level, s, 'stricter_010', 'participants')) for s in SET_NAMES)
    table.add('L7-048', float(floor_found.group(1)), mode='rerun',
              note='the stricter corrected-recognition floor (outputs/exclusion_regimes.txt); %s' % stricter_kept_text)
    table.add('L7-149', largest_stricter_change, mode='rerun',
              note='largest absolute change under the stricter floor (from the with-exclusion values) over the three '
                   'concept-level rho of Table S1 and the six feature-level r of Fig. 6c to 6h: concept rho '
                   '(maintained, changed, reduced) %s; Fig. 6c to 6h %s (%s)' % (
                       ', '.join('%.4f' % change for change in concept_rho_changes),
                       ', '.join('%.4f' % change for change in fig6_changes), stricter_kept_text))

    # ---------------- the Fig. 7d statistic (Table S1, variance explained and the reduction) ----------------
    without = reduction.loc['all_run']
    with_exclusions = reduction.loc['reported']
    resamples_found = set(re.findall(r'\((\d+) participant resamples per set', regimes_text))
    if len(resamples_found) != 1:
        raise RuntimeError('the number of participant resamples is not found in outputs/exclusion_regimes.txt')
    reduction_resamples = int(resamples_found.pop())
    table.add('L7-050', [float(without['percent_drop']), float(without['one_sided_p']),
                         float(with_exclusions['percent_drop']), float(with_exclusions['one_sided_p'])], mode='rerun',
              note='percent reduction of the summed squared feature-memorability correlation from the maintained to '
                   'the reduced set and its one-sided bootstrap p (%d participant resamples per set, seed 7): without '
                   'exclusions %.2f%%, p = %.3f; with exclusions %.2f%%, p = %.3f' % (
                       reduction_resamples, without['percent_drop'], without['one_sided_p'],
                       with_exclusions['percent_drop'], with_exclusions['one_sided_p']))
    fell = (float(without['maintained']) < float(with_exclusions['maintained'])
            and float(without['reduced']) < float(with_exclusions['reduced']))
    table.add('L7-049', 'summed squared correlation with -> without exclusions: maintained %.4f -> %.4f, reduced '
                        '%.4f -> %.4f; fell in both sets: %s' % (
                            with_exclusions['maintained'], without['maintained'], with_exclusions['reduced'],
                            without['reduced'], 'yes' if fell else 'no'), mode='rerun',
              note='summed squared feature-memorability correlation (sum over the 49 features of the squared Spearman '
                   'rho with concept corrected recognition) with and without exclusions, maintained and reduced sets; '
                   'bootstrap means over %d participant resamples (seed 7)' % reduction_resamples)
    table.add('L7-059', number_of_features, mode='rerun',
              note='number of SPoSE features in the sum (squared Spearman correlation of each feature with concept '
                   'corrected recognition)')
    table.add('L7-145', reduction_resamples, mode='rerun',
              note='participant resamples per set of the Fig. 7d bootstrap of step 1, without and with exclusions '
                   '(numpy default_rng(7), a new generator for each regime; outputs/exclusion_regimes.txt)')
    paper_run_percent = float(reproduction.loc['Fig. 7d percent drop (bootstrap mean)', 'saved_paper_run'])
    paper_run_p = float(reproduction.loc['Fig. 7d one-sided p', 'saved_paper_run'])
    table.add('L7-146', [paper_run_percent, paper_run_p], mode='restatement',
              script='06_image_set_experiments_fig6_fig7/code/maintained_vs_reduced_variance.m',
              note='the Fig. 7d statistic of the saved run of module 06 (MATLAB rng(42); '
                   '06_image_set_experiments_fig6_fig7/data/paper_run/fig7_variance_paper_run.mat), as listed in '
                   'outputs/reproduction_check.csv beside this module\'s run with the exclusions (numpy seed 7: '
                   '%.2f%%, one-sided p = %.3f)' % (with_exclusions['percent_drop'], with_exclusions['one_sided_p']))
    table.add('L7-147', 'reduction from the maintained to the reduced set, one-sided bootstrap p = %.3f without and '
                        '%.3f with exclusions (share of the %d resamples in which the reduced-set sum is at least the '
                        'maintained-set sum)' % (without['one_sided_p'], with_exclusions['one_sided_p'],
                                                 reduction_resamples), mode='rerun',
              note='the test of the predicted reduction in Table S1 is one-sided and reported by its p value '
                   '(outputs/tests.csv, T37, p_type bootstrap one-sided)')

    # ---------------- design facts ----------------
    study_items = set()
    for set_name in SET_NAMES:
        study_lists = imageset_loader.load_study_lists(set_name)
        for participant, blocks in study_lists.items():
            if len(blocks) != 16:
                raise RuntimeError('%s has %d blocks' % (participant, len(blocks)))
            for block, concepts_of_block in blocks.items():
                study_items.add(len(concepts_of_block))
    if len(study_items) != 1:
        raise RuntimeError('the number of study items per block differs between blocks: %s' % sorted(study_items))
    table.add('L7-055', study_items.pop(), mode='rerun', script='code/write_numbers.py',
              note='study items per block, counted in every block of the 602 sessions (16 blocks each); the summed '
                   'similarity of a test trial is the mean of normpdf(similarity, 1, 2) over them, the tested concept '
                   'removed once')
    # step 2 takes the concept-level intervals with bootstrap.percentile_interval at its default level
    concept_interval_level = 100 * inspect.signature(bootstrap.percentile_interval).parameters['level'].default
    participant_interval_levels = set(re.findall(r'(\d+)% bootstrap intervals', regimes_text))
    if len(participant_interval_levels) != 1:
        raise RuntimeError('the interval level is not found in outputs/exclusion_regimes.txt')
    table.add('L7-061', [int(round(concept_interval_level)), int(participant_interval_levels.pop()),
                         reduction_resamples], mode='rerun', script='%s; %s' % (SCRIPT_STATS, SCRIPT_REGIMES),
              note='intervals of Table S1: 2.5th and 97.5th percentiles of the resamples, over concepts (concept '
                   'level, outputs/tests.csv T35, the level of percentile_interval in shared/python/bootstrap.py) '
                   'and over %d participant resamples per set (variance explained, seed 7, '
                   'outputs/exclusion_regimes.txt)' % reduction_resamples)

    # ---------------- main-text sentence on the exclusion criteria (Table S1) ----------------
    table.add('L3-046', 'no exclusions: concept rho %s (largest p = %.1e); feature r %s' % (
                  ' / '.join('%.2f' % r for r in rho_all), max(p_all), ' / '.join('%.2f' % r for r in r_all)),
              mode='rerun',
              note='main-text sentence on the exclusion criteria (the results of SI Appendix, Table S1): without '
                   'exclusions, the concept-level Spearman rho (maintained, changed, reduced) and the feature-level '
                   'Pearson r of Fig. 6d and 6g')

    table.rows.sort(key=lambda row: row['ledger_id'])
    table.write(OUTPUTS)


if __name__ == '__main__':
    main()
