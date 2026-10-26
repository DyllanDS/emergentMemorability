"""
read_reports.py: reads the numbers of the SI Appendix, Section 5 reports (the saved reports in
data/provided/, or the same reports written by this module's run in outputs/regenerated/).

Every number is found by the label of its row (a regular expression on the row's words), never by its
line number, so a report with one more or one fewer line is still read correctly, and a missing row
stops the script with an error naming the row.

    reports = read_reports.load_all(folder)      # folder = a5_common.report_folder(source)
    reports['whole_stream']      per bin (20): dprime, crit, da, H, F and their 95% interval ends
    reports['adjusted']          the position-and-lag-adjusted curves, contrasts and spreads (npz)
    reports['sdt_text']          numbers printed in sdt_results.txt
    reports['adjusted_text']     numbers printed in sdt_by_context_adjusted.txt
    reports['quartile_summary']  kramer_quartile_summary.csv as {category: {column: value}}
    reports['quartile_text']     numbers printed in kramer_quartile.txt
    reports['crowding']          the regression rows of crowding.txt
    reports['lone']              the numbers of lone_exemplar.txt
    load_fit_on_sequence(folder) the deep-network report (txt and json)
"""
import csv
import json
import re

import numpy as np

CATEGORIES = ['animal', 'food', 'clothing', 'container', 'tool', 'vehicle']
NUMBER = r'[+-]?\d[\d,]*\.?\d*(?:e[+-]?\d+)?'


def to_number(text):
    """'21,137' -> 21137.0; '+0.00286' -> 0.00286; '3.64e-110' -> 3.64e-110."""
    return float(text.replace(',', ''))


def find(pattern, text, what, flags=re.S):
    """The groups of the first match of pattern in text; stops with an error naming what is missing."""
    match = re.search(pattern, text, flags)
    if match is None:
        raise SystemExit('could not find %s in the report (pattern %r)' % (what, pattern))
    return match.groups()


def read_text(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


# ----------------------------------------------------------------------------- whole stream (Fig. S13 open circles)
def load_whole_stream(folder):
    """sdt_ctx_top10_target.npz: the 20 within-image bins of ctx_top10, target repeats."""
    data = np.load(folder / 'sdt_ctx_top10_target.npz')
    curves = {key: np.asarray(data[key], dtype=float) for key in data.files}
    for key in ('dprime', 'crit', 'H', 'F', 'dlo', 'dhi', 'Hlo', 'Hhi', 'Flo', 'Fhi', 'clo', 'chi'):
        if key not in curves or len(curves[key]) != 20:
            raise SystemExit('sdt_ctx_top10_target.npz: %s is missing or does not have 20 bins' % key)
    return curves


def read_sdt_results(folder):
    """sdt_results.txt: the header and the summary lines of the ctx_top10 / target repeats block."""
    text = read_text(folder / 'sdt_results.txt')
    numbers = {}
    participants, n_bins = find(r'(' + NUMBER + r') participants; context binned into (\d+) within-image bins',
                                text, 'the participants and bins line')
    numbers['participants'] = to_number(participants)
    numbers['bins'] = int(n_bins)
    numbers['bootstraps'] = int(find(r'(\d+) participant bootstraps', text, 'the number of bootstraps')[0])
    block = text.split('ctx_top10   HITS FROM: target repeats (long lag)')[1].split('ctx_top10   HITS FROM: vigilance')[0]
    # the width of the per-bin intervals, from the table header ("d'  [95% CI]")
    numbers['interval_percent'] = int(find(r"d'\s+\[(\d+)% CI\]", block, 'the interval column of the bin table')[0])
    for name, label in [('dprime', "d'"), ('criterion', 'criterion c'), ('hit_rate', 'hit rate'), ('fa_rate', 'FA rate')]:
        first, last, change, monotone, pearson = find(
            r'\n  ' + re.escape(label) + r': bin1 (' + NUMBER + r') -> bin20 (' + NUMBER + r') \(change (' + NUMBER +
            r')\); monotone decreasing: (True|False); r with bin index = (' + NUMBER + r')', block, label + ' summary line')
        numbers[name] = dict(bin1=to_number(first), bin20=to_number(last), change=to_number(change),
                             monotone=(monotone == 'True'), pearson_r_with_bin=to_number(pearson))
    return numbers


# ----------------------------------------------------------------------------- adjusted (Fig. S13 filled circles)
def load_adjusted(folder):
    data = np.load(folder / 'sdt_by_context_adjusted.npz')
    return {key: np.asarray(data[key], dtype=float) for key in data.files}


def read_adjusted_text(folder):
    """sdt_by_context_adjusted.txt: header facts, the step 2 summary, the step 4 regressions, step 5."""
    text = read_text(folder / 'sdt_by_context_adjusted.txt')
    numbers = {}
    trials, participants, images = find(r'(' + NUMBER + r') trials, (' + NUMBER + r') participants, (' + NUMBER +
                                        r') images\.', text, 'the trials, participants and images line')
    numbers['trials'] = to_number(trials)
    numbers['participants'] = to_number(participants)
    numbers['images'] = to_number(images)
    numbers['context_measure'] = find(r'Context = (ctx_\w+): the mean of the (\w+) highest SPoSE correlations',
                                      text, 'the context measure line')
    numbers['minimum_preceding_items'] = int(find(r'Trials require at least (\d+) preceding items', text,
                                                  'the minimum number of preceding items')[0])
    numbers['bins'] = int(find(r'within-image rank bins\n\((\d+) of them\)', text, 'the number of bins')[0])
    position_strata, position_by_lag = find(r'Standardization: (\d+) serial-position strata for false alarms and (\d+x\d+) '
                                            r'position-by-lag strata', text, 'the standardization line')
    numbers['position_strata'] = int(position_strata)
    numbers['position_by_lag_strata'] = position_by_lag
    numbers['detrend_strata'] = int(find(r'detrended against (\d+) serial-position strata', text, 'the detrending line')[0])
    numbers['bootstraps'] = int(find(r'Confidence intervals are (\d+) bootstrap resamples of participants', text,
                                     'the number of bootstraps')[0])
    repeats, first_showings = find(r'STEP 2\..*?trials: (' + NUMBER + r') repeats, (' + NUMBER + r') first showings',
                                   text, 'the step 2 trial counts')
    numbers['repeats'] = to_number(repeats)
    numbers['first_showings'] = to_number(first_showings)
    # step 4: per-participant regressions
    regressions = {}
    for key, label in [('fa_adjusted_bin', 'false alarms, per adjusted bin step'),
                       ('hit_adjusted_bin', 'hits, per adjusted bin step'),
                       ('fa_descriptive_bin', 'false alarms, per descriptive bin step'),
                       ('hit_descriptive_bin', 'hits, per descriptive bin step'),
                       ('fa_per_sd', 'false alarms, per SD of raw context'),
                       ('hit_per_sd', 'hits, per SD of raw context')]:
        b, low, high, df, t, p, n = find(
            re.escape(label) + r'\s+b=(' + NUMBER + r') \[(' + NUMBER + r'), (' + NUMBER + r')\]\s+t\((\d+)\)=(' + NUMBER +
            r')\s+p=(' + NUMBER + r')\s+n=(' + NUMBER + r')', text, 'the regression row "%s"' % label)
        regressions[key] = dict(b=to_number(b), ci_low=to_number(low), ci_high=to_number(high), df=int(df),
                                t=to_number(t), p=to_number(p), n=to_number(n))
    numbers['regressions'] = regressions
    anchor_hit, anchor_fa = find(r'overall rates used as the anchor: hit (' + NUMBER + r'), FA (' + NUMBER + r')', text,
                                 'the overall rates of the analyzed trials')
    numbers['overall_rates'] = dict(hit=to_number(anchor_hit), false_alarm=to_number(anchor_fa))
    contrast, low, high = find(r'predicted bin1 - bin20 d\' contrast: (' + NUMBER + r') \[(' + NUMBER + r'), (' + NUMBER +
                               r')\]', text, 'the regression-route contrast')
    numbers['regression_route_contrast'] = dict(value=to_number(contrast), ci_low=to_number(low), ci_high=to_number(high))
    images_spread, sd, spread = find(r"across-image d' over (" + NUMBER + r') images: SD (' + NUMBER +
                                     r'), 5th-95th percentile range (' + NUMBER + r')', text, 'the across-image spread line')
    numbers['across_image'] = dict(images=to_number(images_spread), sd=to_number(sd), range_5_95=to_number(spread))
    return numbers


# ----------------------------------------------------------------------------- Fig. S14
def load_quartile_summary(folder):
    """kramer_quartile_summary.csv as {category: {column: number}} (category kept as text)."""
    rows = {}
    with open(folder / 'kramer_quartile_summary.csv', encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            values = {}
            for column, value in row.items():
                values[column] = value if column == 'category' else float(value)
            rows[row['category']] = values
    for category in CATEGORIES:
        if category not in rows:
            raise SystemExit('kramer_quartile_summary.csv has no row for %s' % category)
    return rows


def read_quartile_text(folder):
    """kramer_quartile.txt: per category, the header counts, the quartile table, the scoreable
    images, the shuffled splits and the Wilcoxon test; plus the whole-dataset block."""
    text = read_text(folder / 'kramer_quartile.txt')
    numbers = {}
    trials, participants, images = find(r'(' + NUMBER + r') trials, (' + NUMBER + r') participants, (' + NUMBER +
                                        r') images\.', text, 'the trials, participants and images line')
    numbers['trials'] = to_number(trials)
    numbers['participants'] = to_number(participants)
    numbers['images'] = to_number(images)
    hits_part, hits_percent, fa_part, fa_percent = find(
        r"of the " + NUMBER + r" drop in d', (" + NUMBER + r") \((\d+)%\) comes from hits\s+and (" + NUMBER +
        r") \((\d+)%\) from false alarms", text, 'the whole-dataset split of the drop')
    numbers['whole_dataset_split'] = dict(hits=to_number(hits_part), hits_percent=to_number(hits_percent),
                                          false_alarms=to_number(fa_part), false_alarms_percent=to_number(fa_percent))
    blocks = re.split(r'\nCATEGORY: ', text)[1:]
    per_category = {}
    for block in blocks:
        name = block.split()[0].lower()
        concepts, image_count, trial_count = find(r'\((\d+) concepts, (' + NUMBER + r') images, (' + NUMBER + r') trials\)',
                                                  block, 'the %s header' % name)
        quartiles = {}
        for quartile in range(1, 5):
            n_participants, mean_seen = find(r'\n  Q%d\s+(' % quartile + NUMBER + r')\s+(' + NUMBER + r')\s', block,
                                             'quartile Q%d of %s' % (quartile, name))
            quartiles[quartile] = dict(participants=to_number(n_participants), mean_exemplars=to_number(mean_seen))
        scored, total = find(r'images scoreable in both quartiles \(>= (?:\d+) trials of each kind\): (' + NUMBER +
                             r') of (' + NUMBER + r')', block, 'the scoreable images of %s' % name)
        n_null, null_mean, null_percent, null_low, null_high = find(
            r'shuffled quartile assignment, (\d+) permutations: mean rho (' + NUMBER + r')\s+(\d+)% interval \[(' +
            NUMBER + r'), (' + NUMBER + r')\]', block, 'the shuffled splits of %s' % name)
        w_statistic, w_p = find(r'Wilcoxon signed-rank on the paired per-image scores: W = (' + NUMBER + r'), p = (' +
                                NUMBER + r')', block, 'the Wilcoxon test of %s' % name)
        context_bins = int(find(r'preceding-stream crowding \((\d+) bins\)', block, 'the bins of %s' % name)[0])
        quartile_bins = int(find(r'the same effect inside each quartile \((\d+) bins', block,
                                 'the bins inside a quartile of %s' % name)[0])
        per_category[name] = dict(concepts=int(concepts), images=to_number(image_count), trials=to_number(trial_count),
                                  quartiles=quartiles, images_scored=to_number(scored), images_total=to_number(total),
                                  shuffles=int(n_null), null_mean=to_number(null_mean),
                                  null_interval_percent=int(null_percent), null_low=to_number(null_low),
                                  null_high=to_number(null_high), wilcoxon_W=to_number(w_statistic),
                                  wilcoxon_p=to_number(w_p), context_bins=context_bins, quartile_bins=quartile_bins)
    for category in CATEGORIES:
        if category not in per_category:
            raise SystemExit('kramer_quartile.txt has no block for %s' % category)
    numbers['categories'] = per_category
    return numbers


# ----------------------------------------------------------------------------- the 4.6 ratio and the 7.8 percent
def read_crowding(folder):
    """crowding.txt: per outcome (false alarms, hits), the trial count and every model row."""
    text = read_text(folder / 'crowding.txt')
    numbers = {}
    trials, participants, minimum = find(r'(' + NUMBER + r') trials, (' + NUMBER + r') participants\. Trials require at '
                                         r'least (\d+) preceding items', text, 'the header line')
    numbers['trials'] = to_number(trials)
    numbers['participants'] = to_number(participants)
    numbers['minimum_preceding_items'] = int(minimum)
    for outcome, start, end in [('false_alarms', 'FALSE ALARMS on new items', 'HITS on target repeats'),
                                ('hits', 'HITS on target repeats', 'READING THIS')]:
        block = text.split(start)[1].split(end)[0]
        outcome_numbers = dict(trials=to_number(find(r'n = (' + NUMBER + r') trials', block, start + ' trial count')[0]))
        for model, model_start, model_end in [('M1', 'M1  feature crowding alone', 'M2'),
                                              ('M2', 'M2  category crowding alone', 'M3'),
                                              ('M3', 'M3  both together', 'M4'),
                                              ('M4', 'M4  both, plus', '----')]:
            model_block = block.split(model_start)[1].split(model_end)[0]
            rows = {}
            for match in re.finditer(r'\n    (\w+)\s+b=(' + NUMBER + r') \[(' + NUMBER + r'), (' + NUMBER +
                                     r')\]\s+t\((\d+)\)=(' + NUMBER + r')\s+p=(' + NUMBER + r')\s+d=(' + NUMBER + r')',
                                     model_block):
                term, b, low, high, df, t, p, d = match.groups()
                rows[term] = dict(b=to_number(b), ci_low=to_number(low), ci_high=to_number(high), df=int(df),
                                  t=to_number(t), p=to_number(p), d=to_number(d))
            if not rows:
                raise SystemExit('crowding.txt: no rows for %s %s' % (start, model))
            outcome_numbers[model] = rows
        numbers[outcome] = outcome_numbers
    return numbers


def read_lone_exemplar(folder):
    text = read_text(folder / 'lone_exemplar.txt')
    minimum = int(find(r'trials require at\s+least (\d+) preceding items', text, 'the minimum number of preceding items')[0])
    first_showings, percent = find(r'ALL CONCEPTS.*?first showings: (' + NUMBER + r') trials, (' + NUMBER + r')% of them lone',
                                   text, 'the lone share of all concepts')
    return dict(minimum_preceding_items=minimum, first_showings=to_number(first_showings), lone_percent=to_number(percent))


# ----------------------------------------------------------------------------- deep-network spaces
def load_fit_on_sequence(folder):
    """fit_on_sequence.txt and .json: the held-out deep-network curve (5 bins)."""
    text = read_text(folder / 'fit_on_sequence.txt')
    with open(folder / 'fit_on_sequence.json', encoding='utf-8') as handle:
        fit = json.load(handle)
    numbers = {}
    trials, participants, spaces, falloffs = find(r'(' + NUMBER + r') trials, (' + NUMBER + r') participants, (\d+) '
                                                  r'network/layer spaces x (\d+) falloffs', text, 'the header line')
    numbers['trials'] = to_number(trials)
    numbers['participants'] = to_number(participants)
    numbers['spaces'] = int(spaces)
    numbers['falloffs'] = int(falloffs)
    fitting, held_out = find(r'(' + NUMBER + r') fitting trials, (' + NUMBER + r') held-out trials', text, 'the trial counts')
    numbers['fitting_trials'] = to_number(fitting)
    numbers['held_out_trials'] = to_number(held_out)
    curve = fit['results']['fitted combination']
    numbers['dprime'] = [float(value) for value in curve['dprime']]
    numbers['hit'] = [float(value) for value in curve['hit']]
    numbers['fa'] = [float(value) for value in curve['fa']]
    numbers['space_share'] = fit['space_share']
    numbers['tau_share'] = fit['tau_share']
    numbers['best_single'] = fit['best_single']
    best_key = 'best single space (%s, tau %g)' % (fit['best_single'][0], float(fit['best_single'][1]))
    numbers['best_single_dprime'] = [float(value) for value in fit['results'][best_key]['dprime']]
    printed = find(r'fitted combination d-prime\s+(' + NUMBER + r')\s+' + NUMBER + r'\s+' + NUMBER + r'\s+' + NUMBER +
                   r'\s+(' + NUMBER + r')\s+(' + NUMBER + r')', text, 'the printed fitted-combination row')
    # the json and the printed report must agree
    if abs(to_number(printed[0]) - numbers['dprime'][0]) > 0.0006 or abs(to_number(printed[1]) - numbers['dprime'][4]) > 0.0006:
        raise SystemExit('fit_on_sequence.txt and fit_on_sequence.json disagree')
    return numbers


def load_all(folder):
    """Every Section 5 report in one dictionary (except the deep-network report, see load_fit_on_sequence)."""
    return dict(whole_stream=load_whole_stream(folder), sdt_text=read_sdt_results(folder),
                adjusted=load_adjusted(folder), adjusted_text=read_adjusted_text(folder),
                quartile_summary=load_quartile_summary(folder), quartile_text=read_quartile_text(folder),
                crowding=read_crowding(folder), lone=read_lone_exemplar(folder))
