"""
check_provided.py: every number of SI Appendix, Section 5 that this module computes (and of the sentences of
the main text and of SI Appendix, Section 4 that refer to it), read from the Section 5 reports; writes
outputs/numbers.csv (one row per ledger item of this module).

WHERE THE NUMBERS COME FROM
    --source provided     the saved reports in data/provided/ (the default without the Kramer et al. trial
                          file, EXTERNAL_INPUTS.md M1). Each row is judged by the match type of its ledger
                          row.
    --source regenerated  the same reports written by this module's run from the trial file
                          (outputs/regenerated/); each row is judged by the check type given here.
    The 95% intervals that need the trial tables come from data/provided/trial_intervals.csv, which
    stats_section5.py writes when the trial file is available (run_module.py runs it first).
    Two numbers are always computed here: the 62 percent / 4 percent fact (neighbor_share.py, public
    SPoSE data) and the SPoSE five-bin drop of the deep-network comparison (spose_quintiles.py, from the
    20-bin file). The deep-network numbers come from the regenerated report only when the 14 deep-network
    spaces were there too (M3); otherwise from data/provided/.

INPUT   data/provided/ or outputs/regenerated/ (see read_reports.py), and data/provided/trial_intervals.csv;
        outputs/neighbor_share.txt is not read (neighbor_share.py is run from here)
OUTPUT  outputs/numbers.csv
SEEDS   none.  RUNTIME  seconds.
Run from the repository root:  python A5_things_context_effects_figS13_S14/code/check_provided.py [--source provided]
"""
import argparse

import numpy as np
from scipy.stats import spearmanr

import a5_common
import neighbor_share
import numbers_table
import read_reports
import spose_quintiles

MODULE = 'A5_things_context_effects_figS13_S14'
NUMBER_WORDS = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six'}
FIFTH = {5: 'fifth'}


class Rows:
    """Collects the numbers.csv rows. A row read from data/provided/ is judged by the match type of its
    ledger row; a row computed by this module's run is judged by check_type."""

    def __init__(self, source, provided_folder_name):
        self.source = source
        self.table = numbers_table.NumbersTable(script='')
        self.folder_name = provided_folder_name

    def add(self, ledger_id, computed, note, check_type, provided_source, regenerating_script, mode=None):
        entry = self.table.ledger[ledger_id]
        if (self.source == 'provided' and mode is None) or mode == 'provided':
            match_type = entry['match_type']
            row_mode = 'provided'
            script = '%s (%s)' % (provided_source, regenerating_script) if provided_source else regenerating_script
        else:
            match_type = check_type
            row_mode = mode or 'rerun'
            script = regenerating_script
        self.table.add(ledger_id, computed_value=computed, mode=row_mode, note=note, script=script,
                       match_type=match_type)


def main():
    parser = argparse.ArgumentParser(description='SI Appendix, Section 5 numbers -> outputs/numbers.csv')
    parser.add_argument('--source', choices=['provided', 'regenerated'], default=None,
                        help='read the reports from data/provided/ or outputs/regenerated/ '
                             '(default: regenerated when the trial data are on this machine)')
    args = parser.parse_args()
    source = args.source or a5_common.default_source()
    folder = a5_common.report_folder(source)
    fit_folder = a5_common.deep_network_report_folder(source)
    fit_regenerated = fit_folder == a5_common.REGENERATED
    reports = read_reports.load_all(folder)
    fit = read_reports.load_fit_on_sequence(fit_folder)
    spose = spose_quintiles.compute(folder)
    neighbors = neighbor_share.compute()
    intervals = a5_common.trial_intervals()

    folder_name = a5_common.relative(folder)
    fit_folder_name = a5_common.relative(fit_folder)
    intervals_file = a5_common.relative(a5_common.TRIAL_INTERVALS)
    code = MODULE + '/code/'
    trials_script = code + 'stats_from_trials.py'
    rows = Rows(source, folder_name)

    whole = reports['whole_stream']
    adjusted = reports['adjusted']
    sdt_text = reports['sdt_text']
    adjusted_text = reports['adjusted_text']
    summary = reports['quartile_summary']
    quartile_text = reports['quartile_text']
    crowding = reports['crowding']
    lone = reports['lone']

    whole_file = folder_name + '/sdt_ctx_top10_target.npz'
    adjusted_file = folder_name + '/sdt_by_context_adjusted.npz'
    adjusted_txt = folder_name + '/sdt_by_context_adjusted.txt'
    summary_file = folder_name + '/kramer_quartile_summary.csv'
    quartile_txt = folder_name + '/kramer_quartile.txt'

    def interval_note(statistic):
        return ('95%% interval [%.4f, %.4f] over %s participant-session resamples (seed %d; %s)'
                % (statistic['ci_low'], statistic['ci_high'], '{:,}'.format(statistic['n_resamples']),
                   statistic['seed'], intervals_file))

    # ------------------------------------------------------------------ The Same Image After Different Streams (Fig. S13)
    dprime = whole['dprime']
    whole_contrast = float(dprime[0] - dprime[19])
    monotone = bool(np.all(np.diff(dprime) < 0))
    adjusted_contrast = float(adjusted['contrast'])
    adjusted_interval = adjusted['contrast_ci']
    across_image_sd = float(adjusted['across_image_sd'])
    rho, rho_p = spearmanr(np.arange(1, 21), dprime)
    whole_interval = intervals[('T38', 'whole_stream_contrast')]
    adjusted_values = [adjusted_contrast, float(adjusted_interval[0]), float(adjusted_interval[1])]

    rows.add('L3-018', [round(whole_contrast, 4), round(adjusted_contrast, 4)],
             "declines in d' from bin 1 to bin 20 of the within-image bins (SI Appendix, Section 5), which this sentence "
             "refers to: whole stream %.3f -> %.3f (decline %.3f); position and lag held constant %.3f [%.3f, %.3f]"
             % (dprime[0], dprime[19], whole_contrast, adjusted_contrast, adjusted_interval[0],
                adjusted_interval[1]),
             'documented', whole_file + '; ' + adjusted_file, code + 'sdt_by_context.py; ' + code + 'sdt_by_context_adjusted.py')
    for ledger_id in ('L8-002', 'L8-102'):
        rows.add(ledger_id, [whole_contrast, whole_interval['ci_low'], whole_interval['ci_high']],
                 "d' of bin 1 minus bin 20 over the whole stream (ctx_top10, target repeats, 20 within-image bins): "
                 "%.4f -> %.4f; d' falls at every step: %s; %s"
                 % (dprime[0], dprime[19], 'yes' if monotone else 'no', interval_note(whole_interval)),
                 'round', whole_file + '; ' + intervals_file, code + 'sdt_by_context.py; ' + trials_script)
    for ledger_id in ('L8-003', 'L8-103'):
        rows.add(ledger_id, whole_contrast / across_image_sd,
                 "%.4f / %.4f: the whole-stream d' decline divided by the SD of d' across the %s images with at least "
                 "30 target repeats and 30 first showings"
                 % (whole_contrast, across_image_sd, '{:,}'.format(int(adjusted_text['across_image']['images']))),
                 'round', adjusted_file, code + 'sdt_by_context_adjusted.py')
    for ledger_id in ('L8-004', 'L8-013'):
        rows.add(ledger_id, adjusted_values,
                 "d' of bin 1 minus bin 20 with serial position and lag held constant: %.3f -> %.3f; 95%% percentile "
                 "interval [%.4f, %.4f] over %d participant-session resamples (seed 0)"
                 % (adjusted['dprime'][0], adjusted['dprime'][19], adjusted_interval[0], adjusted_interval[1],
                    adjusted_text['bootstraps']),
                 'round', adjusted_file, code + 'sdt_by_context_adjusted.py')
    hit_change = float(adjusted['H'][19] - adjusted['H'][0])
    fa_change = float(adjusted['F'][19] - adjusted['F'][0])
    held_constant_rates = ('hit rate %.4f -> %.4f, false alarm rate %.4f -> %.4f; z-scale parts of the %.3f d\' '
                           'contrast: hits %+.4f [%.4f, %.4f], false alarms %+.4f [%.4f, %.4f]'
                           % (adjusted['H'][0], adjusted['H'][19], adjusted['F'][0], adjusted['F'][19],
                              adjusted_contrast, adjusted['hit_part'], adjusted['hit_part_ci'][0],
                              adjusted['hit_part_ci'][1], adjusted['fa_part'], adjusted['fa_part_ci'][0],
                              adjusted['fa_part_ci'][1]))
    rows.add('L8-005', 'hit rate %+.4f; false alarm rate %+.4f' % (hit_change, fa_change),
             'change in hit rate and false alarm rate from bin 1 to bin 20 with position and lag held constant: '
             + held_constant_rates,
             'documented', adjusted_file, code + 'sdt_by_context_adjusted.py')
    rows.add('L8-107', 'held constant: false alarm rate %+.4f; hit rate %+.4f' % (fa_change, hit_change),
             'bin 1 to bin 20 with position and lag held constant (SI Appendix, Section 5): ' + held_constant_rates,
             'documented', adjusted_file, code + 'sdt_by_context_adjusted.py')
    rows.add('L8-006', 'c whole stream %.3f -> %.3f; held constant %.3f -> %.3f'
             % (whole['crit'][0], whole['crit'][19], adjusted['crit'][0], adjusted['crit'][19]),
             'criterion c = -(z(H) + z(F)) / 2 at bin 1 and bin 20, over the whole stream and with position and lag '
             'held constant',
             'documented', whole_file + '; ' + adjusted_file, code + 'sdt_by_context.py; ' + code + 'sdt_by_context_adjusted.py')
    measure, how_many = adjusted_text['context_measure']
    rows.add('L8-007', '%s: the mean of the %s highest SPoSE correlations with the items already shown' % (measure, how_many),
             "context measure: the mean of the ten highest SPoSE correlations between the probe and the items already "
             "shown (earlier showings of the probe left out); each image's trials ranked into %d within-image bins"
             % adjusted_text['bins'],
             'documented', adjusted_txt, code + 'compute_context.py')
    rows.add('L8-108', 'context minus its trend over %d serial-position strata (one trend for old and new trials); bins '
             're-weighted to %d position strata (false alarms) and %s position-by-lag strata (hits)'
             % (adjusted_text['detrend_strata'], adjusted_text['position_strata'],
                adjusted_text['position_by_lag_strata']),
             'settings of the held-constant analysis, from the header of sdt_by_context_adjusted.txt', 'documented',
             adjusted_txt, code + 'sdt_by_context_adjusted.py')
    whole_hit_change = float(whole['H'][19] - whole['H'][0])
    rows.add('L8-014', 'whole stream %.4f -> %.4f (%+.4f)' % (whole['H'][0], whole['H'][19], whole_hit_change),
             'hit rate at bin 1 and bin 20 over the whole stream',
             'documented', whole_file, code + 'sdt_by_context.py')
    whole_fa_ratio = float(whole['F'][19] / whole['F'][0])
    rows.add('L8-015', 'whole stream %.5f -> %.5f (ratio %.2f)' % (whole['F'][0], whole['F'][19], whole_fa_ratio),
             'false alarm rate at bin 1 and bin 20 over the whole stream, with the bin 20 / bin 1 ratio',
             'documented', whole_file, code + 'sdt_by_context.py')
    rows.add('L8-104', 'whole stream: false alarm part %.3f, hit part %.3f of the %.3f contrast; held constant: false '
             'alarm part %.3f, hit part %+.3f of the %.3f contrast'
             % (adjusted['fa_part_unadj'], adjusted['hit_part_unadj'], whole_contrast, adjusted['fa_part'],
                adjusted['hit_part'], adjusted_contrast),
             "z-scale parts of the bin 1 minus bin 20 d' contrast: false alarms z(F bin 20) - z(F bin 1), hits "
             "z(H bin 1) - z(H bin 20)",
             'documented', adjusted_file, code + 'sdt_by_context_adjusted.py')
    rows.add('L8-012', float(rho),
             "Spearman correlation of d' with bin number over the 20 whole-stream bins (descriptive: every bin holds "
             "the same images); the 'r with bin index' of sdt_results.txt (%.3f) is the Pearson correlation"
             % sdt_text['dprime']['pearson_r_with_bin'],
             'round', whole_file, code + 'sdt_by_context.py; ' + code + 'check_provided.py (Spearman rho)')

    # ------------------------------------------------------------------ SI Appendix, Fig. S13 caption
    rows.add('L8-008', [int(adjusted_text['trials']), int(adjusted_text['participants']), int(adjusted_text['images'])],
             'trials, participant sessions and images of the trial table (header of sdt_by_context_adjusted.txt); %s '
             'sessions in sdt_results.txt' % '{:,}'.format(int(sdt_text['participants'])),
             'count', adjusted_txt, code + 'extract_trials.py')
    spose_columns = np.loadtxt(a5_common.spose_file()).shape[1]
    words_to_number = {'ten': 10}
    rows.add('L8-009', [spose_columns, words_to_number.get(how_many, how_many)],
             'SPoSE dimensions (columns of shared/things/spose_embedding_49d_sorted.txt) and the number of most similar '
             'items averaged (the report: "the mean of the %s highest SPoSE correlations"); similarity = Pearson '
             'correlation of the SPoSE rows' % how_many,
             'exact', adjusted_txt, code + 'compute_context.py')
    rows.add('L8-010', sdt_text['bins'],
             'within-image bins (sdt_results.txt header); trials after at least %d preceding items'
             % adjusted_text['minimum_preceding_items'],
             'exact', folder_name + '/sdt_results.txt', code + 'sdt_by_context.py')
    rows.add('L8-011', 'detrended over %d position strata; %d position strata (false alarms), %s position-by-lag strata '
             '(hits); %d bins; %d resamples, seed 0'
             % (adjusted_text['detrend_strata'], adjusted_text['position_strata'], adjusted_text['position_by_lag_strata'],
                adjusted_text['bins'], adjusted_text['bootstraps']),
             'settings of the held-constant analysis, from the header of sdt_by_context_adjusted.txt', 'documented',
             adjusted_txt, code + 'sdt_by_context_adjusted.py')
    rows.add('L8-016', [sdt_text['interval_percent'], sdt_text['bootstraps']],
             'the %d%% interval of each bin: the 2.5 and 97.5 percentiles of %d participant resamples, over the whole '
             'stream (sdt_results.txt) and with position and lag held constant (%d, sdt_by_context_adjusted.txt)'
             % (sdt_text['interval_percent'], sdt_text['bootstraps'], adjusted_text['bootstraps']),
             'exact', folder_name + '/sdt_results.txt; ' + adjusted_txt,
             code + 'sdt_by_context.py; ' + code + 'sdt_by_context_adjusted.py')
    figure = MODULE + '/outputs/figures/FigS13_things_trial_level_context.png'
    for ledger_id, panel, data_file in [('L8-017', 'a', 'FigS13a_dprime.csv'), ('L8-018', 'b', 'FigS13b_hit_rate.csv'),
                                        ('L8-019', 'c', 'FigS13c_false_alarm_rate.csv')]:
        rows.add(ledger_id, '%s, panel %s' % (figure, panel),
                 'plotted values (both series, with their intervals): %s/outputs/figure_data/%s' % (MODULE, data_file),
                 'documented', whole_file + '; ' + adjusted_file, code + 'make_figureS13.py')

    # ------------------------------------------------------------------ Feature similarity and category membership
    model = crowding['false_alarms']['M3']
    ratio_interval = intervals[('T39', 'ratio')]
    rows.add('L8-020', [ratio_interval['statistic'], ratio_interval['ci_low'], ratio_interval['ci_high']],
             'false alarms, model M3 (both measures, z-scored and centered within image, with serial position): feature '
             'crowding b = %+.5f, same-category count b = %+.5f; trials after at least %d preceding items; the ratio of '
             'the unrounded coefficients is %.4f, with a %s'
             % (model['feature']['b'], model['category']['b'], crowding['minimum_preceding_items'],
                ratio_interval['statistic'], interval_note(ratio_interval)),
             'round', folder_name + '/crowding.txt; ' + intervals_file, code + 'crowding.py; ' + trials_script)
    rows.add('L8-021', [lone['minimum_preceding_items'], lone['lone_percent']],
             'first showings after at least %d preceding items: %s trials, %.1f%% with no item of the probe\'s '
             'higher-level category already shown (ALL CONCEPTS block of lone_exemplar.txt)'
             % (lone['minimum_preceding_items'], '{:,}'.format(int(lone['first_showings'])), lone['lone_percent']),
             'round', folder_name + '/lone_exemplar.txt', code + 'lone_exemplar.py')
    rows.add('L8-022', [100 * neighbors['share'], neighbor_share.NUMBER_OF_NEIGHBORS, 100 * neighbors['chance']],
             '%d of %d nearest SPoSE neighbors (ten per concept, Pearson correlation of SPoSE rows; %d concepts with a '
             'named higher-level category) share the concept\'s category = %.2f%%; chance = 1 of 27 higher-level THINGS '
             'categories = %.2f%%'
             % (neighbors['same_category_neighbors'], neighbors['concepts_with_a_named_category'] * 10,
                neighbors['concepts_with_a_named_category'], 100 * neighbors['share'], 100 * neighbors['chance']),
             'round', '', code + 'neighbor_share.py', mode='rerun')

    # ------------------------------------------------------------------ Kramer et al.'s Quartile Test (Fig. S14)
    animal = summary['animal']
    rows.add('L8-024', len(summary),
             '; '.join('%s (%d concepts, %s images, %s trials)' % (category, int(summary[category]['n_concepts']),
                                                                  '{:,}'.format(int(summary[category]['n_images'])),
                                                                  '{:,}'.format(int(summary[category]['n_trials'])))
                       for category in read_reports.CATEGORIES),
             'exact', summary_file, code + 'kramer_quartile.py')
    above = [category for category in read_reports.CATEGORIES if summary[category]['rho_cross'] > summary[category]['null_hi']]
    below = [category for category in read_reports.CATEGORIES if summary[category]['rho_cross'] < summary[category]['null_lo']]
    inside = len(read_reports.CATEGORIES) - len(above) - len(below)
    split_result = ('above the random-split range in %d of 6; inside %d; below %d (%s)'
                    % (len(above), inside, len(below), ', '.join(below) if below else 'none'))
    rows.add('L8-025', split_result,
             'number of categories whose cross-quartile Spearman correlation lies above, inside and below the middle '
             '95% of the random participant splits',
             'documented', summary_file, code + 'kramer_quartile.py')
    rows.add('L8-105', 'quartile test: ' + split_result,
             'cross-quartile Spearman correlation of the image scores against the middle 95% of the random participant '
             'splits, six categories (SI Appendix, Section 5)',
             'documented', summary_file, code + 'kramer_quartile.py')
    rows.add('L8-106', 'quartile test: %s; the same image after the most minus the least similar preceding stream: d\' '
             'lower by %.3f (whole stream) and %.3f (position and lag held constant)'
             % (split_result, whole_contrast, adjusted_contrast),
             'Kramer et al.\'s quartile test and the within-image context effect in the same trials (SI Appendix, '
             'Section 5)',
             'documented', summary_file + '; ' + whole_file + '; ' + adjusted_file,
             code + 'kramer_quartile.py; ' + code + 'sdt_by_context.py; ' + code + 'sdt_by_context_adjusted.py')
    animal_bins = quartile_text['categories']['animal']['context_bins']
    quartile_bins = quartile_text['categories']['animal']['quartile_bins']
    drop_20 = intervals[('T41', 'drop_20_bins')]
    rows.add('L8-026', [animal['dprime_ctx_drop'], drop_20['ci_low'], drop_20['ci_high'], animal_bins],
             "animal images, d' %.4f -> %.4f over the %d within-image bins; %s"
             % (animal['dprime_ctx_lo'], animal['dprime_ctx_hi'], animal_bins, interval_note(drop_20)),
             'round', summary_file + '; ' + quartile_txt + '; ' + intervals_file, code + 'kramer_quartile.py; ' + trials_script)
    drop_all = intervals[('T41', 'drop_5_bins_all')]
    rows.add('L8-036', [quartile_bins, animal['drop_all_5bin'], drop_all['ci_low'], drop_all['ci_high']],
             "animal images, %d within-image bins, all quartiles; %s" % (quartile_bins, interval_note(drop_all)),
             'round', summary_file + '; ' + intervals_file, code + 'kramer_quartile.py; ' + trials_script)
    drop_top = intervals[('T41', 'drop_5_bins_top')]
    drop_bottom = intervals[('T41', 'drop_5_bins_bottom')]
    rows.add('L8-027', [animal['drop_top_quartile'], drop_top['ci_low'], drop_top['ci_high'],
                        animal['drop_bottom_quartile'], drop_bottom['ci_low'], drop_bottom['ci_high']],
             "animal images, %d within-image bins inside the top quartile (%s) and inside the bottom quartile (%s)"
             % (quartile_bins, interval_note(drop_top), interval_note(drop_bottom)),
             'round', summary_file + '; ' + intervals_file, code + 'kramer_quartile.py; ' + trials_script)
    difference = intervals[('T41', 'quartile_difference')]
    rows.add('L8-028', [abs(animal['dprime_q_diff']), abs(difference['ci_high']), abs(difference['ci_low'])],
             "the size of the top-minus-bottom quartile difference in pooled animal d' (%.3f vs %.3f: %.4f), the dashed "
             "line of Fig. S14b; %s, given as sizes" % (animal['dprime_q_top'], animal['dprime_q_bottom'],
                                                        animal['dprime_q_diff'], interval_note(difference)),
             'round', summary_file + '; ' + intervals_file, code + 'kramer_quartile.py; ' + trials_script)
    split_share = [summary[category]['ctx_split_pct_of_range'] for category in read_reports.CATEGORIES]
    smallest = read_reports.CATEGORIES[int(np.argmin(split_share))]
    largest = read_reports.CATEGORIES[int(np.argmax(split_share))]
    rows.add('L8-029', [min(split_share), max(split_share)],
             'top-minus-bottom difference in mean crowding as a percentage of the range the 20 within-image bins span: '
             '%s smallest, %s largest' % (smallest, largest),
             'round', summary_file, code + 'kramer_quartile.py')
    rows.add('L8-030', [animal['dprime_q_diff_predicted'], animal['dprime_q_diff'], difference['ci_low'],
                        difference['ci_high']],
             "animals: predicted = slope of d' on mean crowding across the 20 bins x the quartiles' crowding difference "
             "(%.1f%% of the range); observed = pooled d' top minus bottom quartile, %s"
             % (animal['ctx_split_pct_of_range'], interval_note(difference)),
             'round', summary_file + '; ' + intervals_file, code + 'kramer_quartile.py; ' + trials_script)
    reliabilities = []
    for category in read_reports.CATEGORIES:
        reliabilities.append((summary[category]['rel_top'], category, 'top'))
        reliabilities.append((summary[category]['rel_bottom'], category, 'bottom'))
    lowest = min(reliabilities)
    highest = max(reliabilities)
    rows.add('L8-031', [lowest[0], highest[0]],
             'split-half reliability of the per-image scores inside a quartile (Spearman-Brown corrected, 50 random '
             'halves): lowest %s %s quartile, highest %s %s quartile' % (lowest[1], lowest[2], highest[1], highest[2]),
             'round', summary_file, code + 'kramer_quartile.py')
    food = summary['food']
    rows.add('L8-035', [inside, animal['rho_cross'], int(animal['n_images_scored']), animal['null_mean']],
             'categories inside the random-split range; animals rho %.4f over %s images scored vs random-split mean %.4f '
             '[%.4f, %.4f]; food %.4f below its range [%.4f, %.4f]'
             % (animal['rho_cross'], '{:,}'.format(int(animal['n_images_scored'])), animal['null_mean'],
                animal['null_lo'], animal['null_hi'], food['rho_cross'], food['null_lo'], food['null_hi']),
             'round', summary_file, code + 'kramer_quartile.py')

    # ------------------------------------------------------------------ SI Appendix, Fig. S14 caption
    quartile_sizes = sorted({int(quartile['participants']) for category in read_reports.CATEGORIES
                             for quartile in quartile_text['categories'][category]['quartiles'].values()})
    rows.add('L8-032', len(summary),
             'six categories; quartiles of %s participant sessions; animals %.1f vs %.1f distinct exemplars (bottom vs top '
             'quartile)' % (' or '.join('{:,}'.format(size) for size in quartile_sizes), animal['mean_seen_bottom'],
                            animal['mean_seen_top']),
             'exact', quartile_txt + '; ' + summary_file, code + 'kramer_quartile.py')
    rows.add('L8-033', '; '.join('%s %s of %s images' % (category, '{:,}'.format(int(summary[category]['n_images_scored'])),
                                                          '{:,}'.format(int(summary[category]['n_images'])))
                                 for category in read_reports.CATEGORIES),
             'images scored in both quartiles (at least 5 repeats and 5 first showings in each); corrected recognition = '
             'hit rate minus false alarm rate',
             'documented', summary_file, code + 'kramer_quartile.py')
    shuffles = quartile_text['categories']['animal']['shuffles']
    null_percent = quartile_text['categories']['animal']['null_interval_percent']
    rows.add('L8-034', [null_percent, shuffles],
             'random splits per category (report: "shuffled quartile assignment, %d permutations ... %d%% interval"); '
             'the interval is the 2.5 and 97.5 percentiles of the random splits' % (shuffles, null_percent),
             'exact', quartile_txt, code + 'kramer_quartile.py')
    figure = MODULE + '/outputs/figures/FigS14_kramer_quartile.png'
    rows.add('L8-039', '%s, panel a' % figure,
             'plotted values: %s/outputs/figure_data/FigS14a_cross_quartile_correlation.csv' % MODULE,
             'documented', summary_file, code + 'make_figureS14.py')
    rows.add('L8-040', '%s, panel b' % figure,
             'plotted values: %s/outputs/figure_data/FigS14b_animal_drops.csv' % MODULE,
             'documented', summary_file, code + 'make_figureS14.py')

    # ------------------------------------------------------------------ A Similarity Space Without Human Judgments
    fit_file = fit_folder_name + '/fit_on_sequence.txt; ' + fit_folder_name + '/fit_on_sequence.json'
    fit_mode = None if (source == 'provided' or fit_regenerated) else 'provided'
    held_out = fit['dprime']
    held_out_drop = held_out[0] - held_out[4]
    held_out_monotone = bool(np.all(np.diff(held_out) < 0))
    spose_interval = intervals[('T42', 'spose_five_bin_drop')]
    rows.add('L8-041', fit['spaces'],
             'network spaces in the report (%s)' % ', '.join(sorted(fit['space_share'])),
             'exact', fit_file, code + 'fit_on_sequence.py', mode=fit_mode)
    rows.add('L8-042', 'half of the sessions fit (numpy default_rng(0) permutation), the other half scored; logistic '
             'regression C = 1.0 on %d measures (%d spaces x %d falloffs)' % (fit['spaces'] * fit['falloffs'], fit['spaces'],
                                                                              fit['falloffs']),
             'fitting trials %s, held-out trials %s' % ('{:,}'.format(int(fit['fitting_trials'])),
                                                         '{:,}'.format(int(fit['held_out_trials']))),
             'documented', fit_file, code + 'fit_on_sequence.py', mode=fit_mode)
    rows.add('L8-043', held_out_drop,
             "held-out d' by fifth: %s; falls at every step: %s"
             % (', '.join('%.4f' % value for value in held_out), 'yes' if held_out_monotone else 'no'),
             'round', fit_file, code + 'fit_on_sequence.py', mode=fit_mode)
    rows.add('L8-044', [spose['drop'], spose_interval['ci_low'], spose_interval['ci_high']],
             "SPoSE whole stream on five bins, from the 20-bin rates (each fifth = mean of four bin rates): d' %s; %s"
             % (', '.join('%.3f' % value for value in spose['dprime']), interval_note(spose_interval)),
             'round', intervals_file, code + 'spose_quintiles.py (from %s); %s' % (whole_file, trials_script),
             mode='provided' if source == 'provided' else 'rerun')
    rows.add('L8-045', [int(fit['participants']), fit['spaces'] * fit['falloffs'],
                        NUMBER_WORDS.get(fit['falloffs'], fit['falloffs'])],
             'participant sessions, crowding measures (%d spaces x %d falloffs), falloffs' % (fit['spaces'], fit['falloffs']),
             'exact', fit_file, code + 'fit_on_sequence.py', mode=fit_mode)
    rows.add('L8-046', int(fit['held_out_trials']),
             'held-out trials (fitting trials %s)' % '{:,}'.format(int(fit['fitting_trials'])),
             'count', fit_file, code + 'fit_on_sequence.py', mode=fit_mode)
    rows.add('L8-048', [whole_contrast, sdt_text['bins']],
             "the whole-stream d' decline over the %d bins of Fig. S13 (as L8-002)" % sdt_text['bins'],
             'round', whole_file, code + 'sdt_by_context.py', mode='provided' if source == 'provided' else 'rerun')
    rows.add('L8-049', FIFTH.get(len(held_out), str(len(held_out))),
             '%d within-image bins of the held-out trials' % len(held_out),
             'exact', fit_file, code + 'fit_on_sequence.py', mode=fit_mode)
    rows.add('L8-101', "deep-network space: d' falls %.3f from the least to the most crowded fifth (held-out sessions); "
             "SPoSE: %.3f on the same five bins, %.3f over the 20 bins" % (held_out_drop, spose['drop'], whole_contrast),
             'the within-image context effect with similarity from human judgments (SPoSE) and from 14 deep-network '
             'spaces (SI Appendix, Section 5)',
             'documented', fit_file, code + 'fit_on_sequence.py; ' + code + 'spose_quintiles.py', mode=fit_mode)
    rows.add('L8-109', "five-bin d' drop: deep-network space %.3f (held-out sessions), SPoSE %.3f [%.3f, %.3f]"
             % (held_out_drop, spose['drop'], spose_interval['ci_low'], spose_interval['ci_high']),
             'the two five-bin drops of A Similarity Space Without Human Judgments',
             'documented', fit_file + '; ' + intervals_file, code + 'fit_on_sequence.py; ' + code + 'spose_quintiles.py',
             mode=fit_mode)

    # ------------------------------------------------------------------ SI Appendix, Fig. S14a, per category
    order = ['clothing', 'container', 'tool', 'vehicle', 'animal', 'food']
    rows.add('ADD-007', '; '.join('%s %.3f vs %.3f [%.3f, %.3f]' % (category, summary[category]['rho_cross'],
                                                                     summary[category]['null_mean'],
                                                                     summary[category]['null_lo'],
                                                                     summary[category]['null_hi'])
                                  for category in order),
             'Fig. S14a per category: observed cross-quartile Spearman rho vs the mean [middle 95%%] of the random '
             'participant splits; images scored %s'
             % ', '.join('{:,}'.format(int(summary[category]['n_images_scored'])) for category in order),
             'documented', summary_file, code + 'kramer_quartile.py')

    path = rows.table.write(str(a5_common.OUTPUTS))
    print('source of the reports: %s; deep-network report: %s; intervals: %s'
          % (folder_name, fit_folder_name, intervals_file))
    return path


if __name__ == '__main__':
    main()
