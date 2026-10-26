"""
stats_section5.py: the uncertainty statistics of SI Appendix, Section 5, tests T38 to T42
(statistics/README.md has the convention), written to outputs/tests.csv.

WHAT IT WRITES
    Every test first gets the numbers of the saved analyses (seed 0, 400 participant resamples, 200
    random splits), read from the reports. Where those reports hold an interval or an effect size, the
    row carries it. The intervals that need the trial tables (EXTERNAL_INPUTS.md, M1) come from
    stats_from_trials.py (2,000 participant resamples, seed 20261013): with the trial data on the
    machine this script computes them and writes them to data/provided/trial_intervals.csv (one row
    per statistic, aggregate numbers only); without the trial data it reads them from that file. The
    interval of the held-out deep-network drop also needs the ten deep-network spaces not included in
    this repository (M3); without them that row has no interval, and its ci_method names the inputs.

    T38  the within-image context effect of Fig. S13: whole stream (0.508; Spearman rho over the 20
         bins, descriptive) and with position and lag held constant (0.131 [0.101, 0.163]), and their
         split into hits and false alarms on the z scale
    T39  feature crowding against same-category count for false alarms (the 4.6 ratio)
    T40  Kramer et al.'s quartile test in six categories (Fig. S14a) and the split-half reliability
         of the image scores inside a quartile
    T41  the animal d' drops and the quartile difference (Fig. S14b)
    T42  the deep-network five-bin drop (0.512, held out) and the SPoSE five-bin drop (0.386)

    No correction for multiple comparisons is applied (statistics/README.md).

INPUT   the Section 5 reports in data/provided/ or outputs/regenerated/; with the trial data, the trial
        tables in external/kramer_trials/derived/, and without them data/provided/trial_intervals.csv
OUTPUT  outputs/tests.csv; with the trial data also data/provided/trial_intervals.csv
SEED    20261013 for every resample computed here (only with the trial data); none otherwise.
RUNTIME seconds without the trial data; at most 3 minutes with them (Apple M2 Pro).
Run from the repository root:  python A5_things_context_effects_figS13_S14/code/stats_section5.py [--source provided]
"""
import argparse

import numpy as np
from scipy.stats import spearmanr

import a5_common
import read_reports
import spose_quintiles
import tests_table

SCRIPT = 'A5_things_context_effects_figS13_S14/code/stats_section5.py'
NEEDS_M1_M3 = ('none without the Kramer et al. trial file and the ten deep-network spaces not included in '
               'this repository (EXTERNAL_INPUTS.md, M1 and M3)')
NEEDS_M3 = 'none without the ten deep-network spaces not included in this repository (EXTERNAL_INPUTS.md, M3)'
PAPER_RUN = 'the 400 participant resamples of sdt_by_context_adjusted.py (seed 0)'
TRIAL_RUN = ('2,000 participant-session resamples of stats_from_trials.py (seed 20261013; '
             'data/provided/trial_intervals.csv)')


def add_t38(tests, reports, intervals):
    whole = reports['whole_stream']
    adjusted = reports['adjusted']
    adjusted_text = reports['adjusted_text']
    sessions = int(reports['sdt_text']['participants'])
    across_image_sd = float(adjusted['across_image_sd'])
    unit = 'participant sessions (6,223,953 trials, 26,107 images, 20 within-image bins)'
    location = 'SI Appendix, Section 5, The Same Image After Different Streams'
    whole_contrast = float(whole['dprime'][0] - whole['dprime'][19])
    interval = intervals[('T38', 'whole_stream_contrast')]
    tests.add(test_id='T38', ledger_ids='L8-002;L8-003;L8-048;L8-102;L8-103;L3-018', location=location,
              test=("Fig. S13a, whole stream: d' of the least minus the most similar preceding stream, bin 1 minus bin "
                    "20 of each image's trials ranked within image by the mean SPoSE similarity of the ten most similar "
                    "items already shown (ctx_top10, target repeats, trials after at least 20 items); effect size in "
                    "across-image SDs of d' (SD %.3f over %s images); interval: %s"
                    % (across_image_sd, '{:,}'.format(int(adjusted_text['across_image']['images'])), TRIAL_RUN)),
              statistic_name="d' difference", statistic=whole_contrast, n=sessions, unit=unit, p_type='none',
              effect_size_name="d' difference / across-image SD of d'", effect_size=whole_contrast / across_image_sd,
              ci_low=interval['ci_low'], ci_high=interval['ci_high'], ci_method='percentile bootstrap',
              resampling_unit='participant sessions', n_resamples=interval['n_resamples'], seed=interval['seed'])

    # the Spearman correlation of d' with bin (SI Appendix, Section 5, The Same Image After Different Streams)
    rho = spearmanr(np.arange(1, 21), whole['dprime'])[0]
    tests.add(test_id='T38', ledger_ids='L8-012', location=location,
              test=("Fig. S13a, whole stream: Spearman rho between bin number and the d' of the 20 bins. Descriptive: "
                    "the 20 bins hold the same images and participants, and the inference is the participant "
                    "interval of the bin 1 minus bin 20 contrast"),
              statistic_name='Spearman rho', statistic=float(rho), n=20, unit='bins (not independent)', p_type='none',
              effect_size_name='Spearman rho', effect_size=float(rho), ci_method='none (descriptive)',
              resampling_unit='')

    # the whole-stream split into hits and false alarms on the z scale
    for part, key, ledger_ids in [('hits: z(H bin 1) - z(H bin 20)', 'hit_part', 'L8-014;L8-104'),
                                  ('false alarms: z(F bin 20) - z(F bin 1)', 'fa_part', 'L8-015;L8-104')]:
        value = float(adjusted[key + '_unadj'])
        interval = intervals[('T38', 'whole_stream_' + key)]
        tests.add(test_id='T38', ledger_ids=ledger_ids, location=location,
                  test=("Fig. S13b, c, whole stream: contribution of the %s to the %.3f d' contrast; interval: %s"
                        % (part, whole_contrast, TRIAL_RUN)),
                  statistic_name="z-scale contribution to the d' contrast", statistic=value, n=sessions, unit=unit,
                  p_type='none', effect_size_name="share of the d' contrast", effect_size=value / whole_contrast,
                  ci_low=interval['ci_low'], ci_high=interval['ci_high'], ci_method='percentile bootstrap',
                  resampling_unit='participant sessions', n_resamples=interval['n_resamples'], seed=interval['seed'])

    # position and lag held constant: the interval of the saved run
    adjusted_contrast = float(adjusted['contrast'])
    adjusted_interval = adjusted['contrast_ci']
    tests.add(test_id='T38', ledger_ids='L8-004;L8-013;L8-108;L3-018', location=location,
              test=("Fig. S13a, serial position and lag held constant: d' of bin 1 minus bin 20 after the context "
                    "measure is detrended over 50 serial-position strata and every bin is re-weighted to the same 10 "
                    "position strata (false alarms) and 10 x 10 position-by-lag strata (hits); effect size in "
                    "across-image SDs of d'; interval: " + PAPER_RUN),
              statistic_name="d' difference", statistic=adjusted_contrast, n=sessions, unit=unit, p_type='none',
              effect_size_name="d' difference / across-image SD of d'", effect_size=adjusted_contrast / across_image_sd,
              ci_low=float(adjusted_interval[0]), ci_high=float(adjusted_interval[1]),
              ci_method='percentile bootstrap (multinomial draw of participant sessions)',
              resampling_unit='participant sessions', n_resamples=int(adjusted_text['bootstraps']), seed=0)

    # held constant: the hit and false alarm parts on the z scale, with the intervals of the saved run
    for part, key, ledger_ids in [('hits: z(H bin 1) - z(H bin 20)', 'hit_part', 'L8-005;L8-107'),
                                  ('false alarms: z(F bin 20) - z(F bin 1)', 'fa_part', 'L8-005;L8-104;L8-107')]:
        value = float(adjusted[key])
        interval = adjusted[key + '_ci']
        tests.add(test_id='T38', ledger_ids=ledger_ids, location=location,
                  test=("Fig. S13b, c, position and lag held constant: contribution of the %s to the %.3f d' contrast; "
                        "interval: %s" % (part, adjusted_contrast, PAPER_RUN)),
                  statistic_name="z-scale contribution to the d' contrast", statistic=value, n=sessions, unit=unit,
                  p_type='none', effect_size_name="share of the d' contrast", effect_size=value / adjusted_contrast,
                  ci_low=float(interval[0]), ci_high=float(interval[1]),
                  ci_method='percentile bootstrap (multinomial draw of participant sessions)',
                  resampling_unit='participant sessions', n_resamples=int(adjusted_text['bootstraps']), seed=0)


def add_t39(tests, reports, intervals):
    model = reports['crowding']['false_alarms']['M3']
    ratio = intervals[('T39', 'ratio')]
    tests.add(test_id='T39', ledger_ids='L8-020',
              location='SI Appendix, Section 5, The Same Image After Different Streams',
              test=('False alarms, model M3 of crowding.py: ratio of the mean feature-crowding coefficient to the mean '
                    'same-category-count coefficient ("4.6 times more strongly"; unrounded coefficients); both '
                    'measures z-scored and centered within image, with serial position; interval: ' + TRIAL_RUN),
              statistic_name='ratio of mean coefficients', statistic=ratio['statistic'], n=model['feature']['df'] + 1,
              unit='participant sessions', p_type='none', ci_low=ratio['ci_low'], ci_high=ratio['ci_high'],
              ci_method='percentile bootstrap of the ratio of means', resampling_unit='participant sessions',
              n_resamples=ratio['n_resamples'], seed=ratio['seed'])


def add_t40(tests, reports):
    summary = reports['quartile_summary']
    quartile_text = reports['quartile_text']['categories']
    for category in read_reports.CATEGORIES:
        row = summary[category]
        text = quartile_text[category]
        images_scored = int(row['n_images_scored'])
        observed = row['rho_cross']
        null_mean = row['null_mean']
        tests.add(test_id='T40', ledger_ids='L8-025;L8-035;L8-105;L8-106;ADD-007', location='SI Appendix, Fig. S14a',
                  test=("Fig. S14a, %s (%d concepts, %s images), Kramer et al.'s quartile test: Spearman rho between the "
                        "images' corrected recognition in the top and in the bottom quartile of participants ranked by "
                        "how many %s exemplars they saw (%.1f vs %.1f distinct exemplars on average), against the same "
                        "correlation over %d random splits of the participants (mean %.3f; the interval columns hold "
                        "the middle 95%% of the random splits, not a confidence interval)"
                        % (category, int(row['n_concepts']), '{:,}'.format(int(row['n_images'])), category,
                           row['mean_seen_top'], row['mean_seen_bottom'], text['shuffles'], null_mean)),
                  statistic_name='Spearman rho', statistic=observed, n=images_scored,
                  unit='images scored in both quartiles (at least 5 repeats and 5 first showings each)', p_type='none',
                  effect_size_name='observed rho minus random-split mean', effect_size=observed - null_mean,
                  ci_low=row['null_lo'], ci_high=row['null_hi'],
                  ci_method="middle 95%% of %d random participant splits (the null range)" % text['shuffles'],
                  resampling_unit='participants (split assignment)', n_resamples=text['shuffles'], seed=0)
        for quartile, column in [('top', 'rel_top'), ('bottom', 'rel_bottom')]:
            tests.add(test_id='T40', ledger_ids='L8-031',
                      location="SI Appendix, Section 5, Kramer et al.'s Quartile Test",
                      test=("Fig. S14a, %s: split-half reliability of the per-image scores inside the %s quartile "
                            "(mean Spearman rho over 50 random halves of the quartile's participants, images with at "
                            "least 3 repeats and 3 first showings in both halves, Spearman-Brown corrected)"
                            % (category, quartile)),
                      statistic_name='split-half reliability (Spearman-Brown corrected)', statistic=row[column],
                      unit='images', p_type='none', ci_method='none', resampling_unit='participants (random halves)',
                      n_resamples=50, seed=0)


def add_t41(tests, reports, intervals):
    row = reports['quartile_summary']['animal']
    text = reports['quartile_text']['categories']['animal']
    sessions = int(reports['quartile_text']['participants'])
    quartile_sessions = '%s and %s' % ('{:,}'.format(int(text['quartiles'][4]['participants'])),
                                      '{:,}'.format(int(text['quartiles'][1]['participants'])))
    location = "SI Appendix, Section 5, Kramer et al.'s Quartile Test"
    entries = [
        ('drop_20_bins', row['dprime_ctx_drop'], 'L8-026',
         "animal images, d' of the least minus the most crowded of 20 within-image bins, all participants", sessions,
         'participant sessions'),
        ('drop_5_bins_all', row['drop_all_5bin'], 'L8-036',
         "Fig. S14b, animal images, d' of the least minus the most crowded fifth of each image's trials, all "
         "participants", sessions, 'participant sessions'),
        ('drop_5_bins_top', row['drop_top_quartile'], 'L8-027',
         "Fig. S14b, animal images, the same five-bin drop inside the top quartile alone",
         int(text['quartiles'][4]['participants']), 'participant sessions of the top quartile'),
        ('drop_5_bins_bottom', row['drop_bottom_quartile'], 'L8-027',
         "Fig. S14b, animal images, the same five-bin drop inside the bottom quartile alone",
         int(text['quartiles'][1]['participants']), 'participant sessions of the bottom quartile'),
        ('quartile_difference', row['dprime_q_diff'], 'L8-028;L8-030',
         "Fig. S14b, animals, pooled d' of the top minus the bottom quartile (predicted from the crowding the split "
         "moves: %.3f)" % row['dprime_q_diff_predicted'], sessions,
         'participant sessions (%s in the two quartiles)' % quartile_sessions),
    ]
    for key, value, ledger_ids, description, n, unit in entries:
        interval = intervals[('T41', key)]
        tests.add(test_id='T41', ledger_ids=ledger_ids, location=location,
                  test=('%s (%s animal images, binned within image over trials after at least 20 preceding items); '
                        'interval: %s' % (description, '{:,}'.format(int(row['n_images'])), TRIAL_RUN)),
                  statistic_name="d' difference", statistic=value, n=n, unit=unit, p_type='none',
                  effect_size_name="d' difference", effect_size=value, ci_low=interval['ci_low'],
                  ci_high=interval['ci_high'],
                  ci_method='percentile bootstrap (all participant sessions drawn; quartile membership fixed)',
                  resampling_unit='participant sessions', n_resamples=interval['n_resamples'], seed=interval['seed'])
    tests.add(test_id='T41', ledger_ids='L8-030', location=location,
              test=("Animals: top-minus-bottom quartile d' predicted from how far the split moves crowding (the slope of "
                    "d' on mean crowding across the 20 within-image bins times the quartiles' difference in mean "
                    "crowding, %.1f%% of the bins' range); a model-based prediction, no interval"
                    % row['ctx_split_pct_of_range']),
              statistic_name="predicted d' difference", statistic=row['dprime_q_diff_predicted'], n=sessions,
              unit='participant sessions', p_type='none', ci_method='none', resampling_unit='')


def add_t42(tests, reports, fit, spose, intervals, held_out, trial_data):
    location = 'SI Appendix, Section 5, A Similarity Space Without Human Judgments'
    held_out_drop = fit['dprime'][0] - fit['dprime'][4]
    test = ("Deep-network space: d' of the least minus the most crowded fifth of each image's held-out trials, "
            "crowding weighted over 14 network spaces x 5 falloffs by a logistic regression fitted on the other half of "
            "the sessions (%s fitting trials, %s held-out trials)"
            % ('{:,}'.format(int(fit['fitting_trials'])), '{:,}'.format(int(fit['held_out_trials']))))
    ledger_ids = 'L8-043;L8-046;L8-101;L8-109'
    if held_out is None:
        tests.add(test_id='T42', ledger_ids=ledger_ids, location=location, test=test,
                  statistic_name="d' difference", statistic=held_out_drop, n=int(fit['held_out_trials']),
                  unit='held-out trials', p_type='none', effect_size_name="d' difference", effect_size=held_out_drop,
                  ci_method=NEEDS_M3 if trial_data else NEEDS_M1_M3, resampling_unit='held-out participant sessions')
    else:
        tests.add(test_id='T42', ledger_ids=ledger_ids, location=location, test=test,
                  statistic_name="d' difference", statistic=held_out['point'], n=held_out['sessions'],
                  unit='held-out participant sessions (%s trials)' % '{:,}'.format(int(held_out['held_out_trials'])),
                  p_type='none', effect_size_name="d' difference", effect_size=held_out['point'],
                  ci_low=held_out['interval'][0], ci_high=held_out['interval'][1], ci_method='percentile bootstrap',
                  resampling_unit='held-out participant sessions', n_resamples=held_out['resamples'],
                  seed=held_out['seed'])
    interval = intervals[('T42', 'spose_five_bin_drop')]
    tests.add(test_id='T42', ledger_ids='L8-044;L8-109', location=location,
              test=("SPoSE space: d' of the least minus the most crowded fifth of each image's trials, all sessions, "
                    "from the 20 whole-stream bins of Fig. S13 pooled four at a time (spose_quintiles.py); interval: "
                    + TRIAL_RUN),
              statistic_name="d' difference", statistic=spose['drop'], n=int(reports['sdt_text']['participants']),
              unit='participant sessions', p_type='none', effect_size_name="d' difference", effect_size=spose['drop'],
              ci_low=interval['ci_low'], ci_high=interval['ci_high'], ci_method='percentile bootstrap',
              resampling_unit='participant sessions', n_resamples=interval['n_resamples'], seed=interval['seed'])


def trial_statistics(reports, fit, fit_source):
    """The statistics that need the trial tables (stats_from_trials.py)."""
    import stats_from_trials
    print('computing the trial-level intervals (stats_from_trials.py)', flush=True)
    trials = stats_from_trials.load_trials()
    results = {}
    results['t38_whole'] = stats_from_trials.t38_whole_stream(trials, reports['whole_stream'])
    print('  T38 done', flush=True)
    results['t39'] = stats_from_trials.t39_ratio(trials, reports['crowding'])
    print('  T39 done', flush=True)
    results['t41'] = stats_from_trials.t41_animal_drops(trials, reports['quartile_summary'])
    print('  T41 done', flush=True)
    results['t42_heldout'] = stats_from_trials.t42_heldout(fit) if fit_source == 'regenerated' else None
    print('  T42 done', flush=True)
    return results


def interval_rows(trial_results):
    """The rows of data/provided/trial_intervals.csv: each statistic of trial_statistics() with its
    interval over the 2,000 resamples."""
    rows = []
    whole = trial_results['t38_whole']
    for key, statistic_name in [('contrast', "d' difference"),
                                ('hit_part', "z-scale contribution to the d' contrast"),
                                ('fa_part', "z-scale contribution to the d' contrast")]:
        rows.append(dict(test_id='T38', key='whole_stream_' + key, statistic_name=statistic_name,
                         statistic=float(whole['point'][key]), ci_low=whole['release'][key][0],
                         ci_high=whole['release'][key][1], n_resamples=whole['release']['resamples'],
                         seed=whole['release']['seed']))
    ratio = trial_results['t39']
    rows.append(dict(test_id='T39', key='ratio', statistic_name='ratio of mean coefficients', statistic=ratio['ratio'],
                     ci_low=ratio['interval'][0], ci_high=ratio['interval'][1], n_resamples=ratio['resamples'],
                     seed=ratio['seed']))
    drops = trial_results['t41']
    for key in ['drop_20_bins', 'drop_5_bins_all', 'drop_5_bins_top', 'drop_5_bins_bottom', 'quartile_difference']:
        rows.append(dict(test_id='T41', key=key, statistic_name="d' difference", statistic=float(drops['point'][key]),
                         ci_low=drops['release'][key][0], ci_high=drops['release'][key][1],
                         n_resamples=drops['resamples'], seed=drops['seed']))
    rows.append(dict(test_id='T42', key='spose_five_bin_drop', statistic_name="d' difference",
                     statistic=float(whole['point']['five_bin_drop']), ci_low=whole['release']['five_bin_drop'][0],
                     ci_high=whole['release']['five_bin_drop'][1], n_resamples=whole['release']['resamples'],
                     seed=whole['release']['seed']))
    return rows


def main():
    parser = argparse.ArgumentParser(description='SI Appendix, Section 5 uncertainty statistics (T38 to T42)')
    parser.add_argument('--source', choices=['provided', 'regenerated'], default=None,
                        help='read the reports from data/provided/ or outputs/regenerated/ '
                             '(default: regenerated when the trial data are on this machine)')
    args = parser.parse_args()
    source = args.source or a5_common.default_source()
    folder = a5_common.report_folder(source)
    reports = read_reports.load_all(folder)
    fit_folder = a5_common.deep_network_report_folder(source)
    fit_source = 'regenerated' if fit_folder == a5_common.REGENERATED else 'provided'
    fit = read_reports.load_fit_on_sequence(fit_folder)
    spose = spose_quintiles.compute(folder)
    held_out = None
    if source == 'regenerated':
        trial_results = trial_statistics(reports, fit, fit_source)
        a5_common.write_trial_intervals(interval_rows(trial_results))
        held_out = trial_results['t42_heldout']
    intervals = a5_common.trial_intervals()

    tests = tests_table.TestsTable(script=SCRIPT)
    add_t38(tests, reports, intervals)
    add_t39(tests, reports, intervals)
    add_t40(tests, reports)
    add_t41(tests, reports, intervals)
    add_t42(tests, reports, fit, spose, intervals, held_out, source == 'regenerated')
    tests.write(str(a5_common.OUTPUTS))


if __name__ == '__main__':
    main()
