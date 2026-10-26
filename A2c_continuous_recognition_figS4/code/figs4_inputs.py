"""
figs4_inputs.py: reads the two sides of Fig. S4 and checks each file against its text twin. Used by
check_provided.py, stats_figS4.py and make_figureS4.py (this module only).

THE MODEL SIDE (REM run through the THINGS trial sequences; continuous_rem.py)
  model_run('provided')     the saved run the paper reports: data/provided/continuous_bins.npz and
                            continuous_rem.txt (seeds 7, 99 and 1234)
  model_run('regenerated')  a run of continuous_rem.py in this release: outputs/continuous_bins.npz,
                            continuous_rem.txt and continuous_cell_counts.npz
  model_run('auto')         'regenerated' when outputs/continuous_bins.npz exists, else 'provided'
  Each returns one dictionary per storage probability ('standard', 'accuracy-matched') with the 20-bin
  curves (dprime, H, F, crit_bin), the overall rates and the settings, after checking every number
  the report shares with the npz file (to the report's printed precision).

THE DATA SIDE (Kramer et al., 2023; provided output of module A5: regenerating it needs the trial
file, EXTERNAL_INPUTS.md M1)
  things_data()   ../A5_things_context_effects_figS13_S14/data/provided/sdt_ctx_top10_target.npz
                  (20 bins: hit and false alarm rates, d' and criterion c with their 95 percent
                  participant-bootstrap limits, and d_a) checked against sdt_results.txt (the same run's
                  report, block "ctx_top10   HITS FROM: target repeats (long lag)"), plus the number of
                  sessions and bootstraps from that report, the participants' overall rates from
                  ../A3b_hits_and_false_alarms_figS9/data/provided/per_image_rates.txt and the
                  across-image SD of d' from A5's sdt_by_context_adjusted.txt.

Every text file is read by the label of its rows (regular expressions), never by line number.
"""
import os
import re
import sys

import numpy as np
from scipy.stats import spearmanr

sys.dont_write_bytecode = True   # write no __pycache__ folder into shared/python or code/ (set before the imports below)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402

PATHS = release_paths.module_paths(__file__, make_output_folders=False)
RELEASE_ROOT = release_paths.release_root()
POINTS = ['standard', 'accuracy-matched']
POINT_NAMES = {'standard': 'standard storage', 'accuracy-matched': 'accuracy-matched storage'}

A5_PROVIDED = os.path.join(release_paths.find_module('A5'), 'data', 'provided')
A3B_PROVIDED = os.path.join(release_paths.find_module('A3b'), 'data', 'provided')
DATA_NPZ = os.path.join(A5_PROVIDED, 'sdt_ctx_top10_target.npz')
DATA_REPORT = os.path.join(A5_PROVIDED, 'sdt_results.txt')
ADJUSTED_REPORT = os.path.join(A5_PROVIDED, 'sdt_by_context_adjusted.txt')
PER_IMAGE_REPORT = os.path.join(A3B_PROVIDED, 'per_image_rates.txt')

# Settings of the data analysis (module A5's compute_context.py and sdt_by_context.py), for the
# descriptions only: a trial is binned once at least 20 earlier trials showed other images (MIN_PRIOR = 20).
DATA_MIN_PRIOR = 20


def relative(path):
    """A path relative to the repository root (for messages and tables)."""
    return os.path.relpath(os.path.abspath(path), RELEASE_ROOT)


def read_text(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def find_number(pattern, text, what, flags=re.M):
    """The groups of the first match of pattern in text, as text; stops with a message if absent."""
    match = re.search(pattern, text, flags)
    if match is None:
        raise ValueError('could not find %s (pattern %r)' % (what, pattern))
    return match.groups()


def to_number(text):
    return float(text.replace(',', ''))


def check_printed(value, printed_text, what, problems):
    """The npz value must round to the printed text at the printed precision."""
    decimals = len(printed_text.split('.')[1]) if '.' in printed_text else 0
    tolerance = 0.5 * 10 ** (-decimals) + 1e-12
    if abs(float(value) - to_number(printed_text)) > tolerance:
        problems.append('%s: file %.6g, report %s' % (what, float(value), printed_text))


def spearman_with_bin(values):
    """Spearman correlation of a binned value with the bin number (1 to 20), and its p (t approximation)."""
    bin_numbers = np.arange(1, len(values) + 1)
    result = spearmanr(bin_numbers, values)
    return float(result.statistic), float(result.pvalue)


# ----------------------------------------------------------------------------- the model side
def model_run_files(source):
    """The npz and report of the model run: source 'provided', 'regenerated' or 'auto'."""
    regenerated_npz = os.path.join(PATHS['outputs'], 'continuous_bins.npz')
    if source == 'auto':
        source = 'regenerated' if os.path.exists(regenerated_npz) else 'provided'
    if source == 'provided':
        folder = PATHS['provided']
        label = 'provided: the saved run, data/provided/'
    elif source == 'regenerated':
        folder = PATHS['outputs']
        label = 'regenerated: a run of continuous_rem.py in this release, outputs/'
    else:
        raise ValueError("source must be 'provided', 'regenerated' or 'auto'")
    files = dict(source=source, label=label,
                 npz=os.path.join(folder, 'continuous_bins.npz'),
                 report=os.path.join(folder, 'continuous_rem.txt'),
                 cell_counts=os.path.join(folder, 'continuous_cell_counts.npz'))
    for name in ('npz', 'report'):
        if not os.path.exists(files[name]):
            raise FileNotFoundError('model run (%s): %s is missing' % (source, relative(files[name])))
    return files


def model_run(source='auto'):
    """The model run of Fig. S4, checked against its report. Returns a dictionary:
    source, label, files, simulated_participants, trials_per_sequence, and per point ('standard',
    'accuracy-matched') a dictionary with storage, criterion, overall_hit, overall_fa, dprime, H, F,
    crit_bin, rho_d, rho_c, cr_H, cr_F (from the npz) and the report's printed text."""
    files = model_run_files(source)
    bins = np.load(files['npz'])
    report = read_text(files['report'])
    problems = []

    run = dict(source=files['source'], label=files['label'], files=files)
    run['simulated_participants'] = int(to_number(find_number(r'([\d,]+) simulated participants, each', report,
                                                              'the number of simulated participants')[0]))
    run['trials_per_sequence'] = int(find_number(r'inheriting the (\d+)-trial structure', report,
                                                 'the trials per sequence')[0])

    blocks = report.split('OPERATING POINT: ')[1:]
    for point in POINTS:
        block = None
        for candidate in blocks:
            if candidate.startswith(POINT_NAMES[point]):
                block = candidate
        if block is None:
            raise ValueError('no OPERATING POINT block %r in %s' % (POINT_NAMES[point], relative(files['report'])))
        series = dict(storage=float(bins[point + '_u']), criterion=float(bins[point + '_crit']),
                      overall_hit=float(bins[point + '_hit']), overall_fa=float(bins[point + '_fa']),
                      dprime=np.array(bins[point + '_dprime'], dtype=float),
                      H=np.array(bins[point + '_H'], dtype=float),
                      F=np.array(bins[point + '_F'], dtype=float),
                      crit_bin=np.array(bins[point + '_crit_bin'], dtype=float),
                      rho_d=float(bins[point + '_rho_d']), rho_c=float(bins[point + '_rho_c']),
                      cr_H=float(bins[point + '_cr_H']), cr_F=float(bins[point + '_cr_F']))

        # every number of the report block, against the npz
        storage_text, criterion_text = find_number(r'storage u = ([\d.]+), criterion = ([\d.]+)', block, 'storage')
        check_printed(series['storage'], storage_text, point + ' storage', problems)
        check_printed(series['criterion'], criterion_text, point + ' criterion', problems)
        hit_text, fa_text = find_number(r'aggregate hit rate ([\d.]+), false alarm rate ([\d.]+)', block, 'rates')
        check_printed(series['overall_hit'], hit_text, point + ' overall hit rate', problems)
        check_printed(series['overall_fa'], fa_text, point + ' overall false alarm rate', problems)
        cr_h_text, = find_number(r'corrected recognition vs hit rate\s+r = ([+-][\d.]+)', block, 'r(CR, hit)')
        cr_f_text, = find_number(r'corrected recognition vs FA rate\s+r = ([+-][\d.]+)', block, 'r(CR, FA)')
        check_printed(series['cr_H'], cr_h_text, point + ' r(corrected recognition, hit rate)', problems)
        check_printed(series['cr_F'], cr_f_text, point + ' r(corrected recognition, false alarm rate)', problems)
        d_first, d_last, d_change, d_rho = find_number(
            r"^\s+d'\s+([\d.]+) -> ([\d.]+)\s+\(change ([+-][\d.]+)\), rho = ([+-][\d.]+)", block, "binned d'")
        check_printed(series['dprime'][0], d_first, point + " d' bin 1", problems)
        check_printed(series['dprime'][-1], d_last, point + " d' bin 20", problems)
        check_printed(series['dprime'][-1] - series['dprime'][0], d_change, point + " d' change", problems)
        check_printed(series['rho_d'], d_rho, point + " Spearman rho of d' with bin", problems)
        h_first, h_last = find_number(r'^\s+hit rate\s+([\d.]+) -> ([\d.]+)\s*$', block, 'binned hit rate')
        check_printed(series['H'][0], h_first, point + ' hit rate bin 1', problems)
        check_printed(series['H'][-1], h_last, point + ' hit rate bin 20', problems)
        f_first, f_last = find_number(r'^\s+FA rate\s+([\d.]+) -> ([\d.]+)\s*$', block, 'binned false alarm rate')
        check_printed(series['F'][0], f_first, point + ' false alarm rate bin 1', problems)
        check_printed(series['F'][-1], f_last, point + ' false alarm rate bin 20', problems)
        c_first, c_last, c_rho = find_number(r'^\s+criterion ([+-][\d.]+) -> ([+-][\d.]+), rho = ([+-][\d.]+)',
                                             block, 'binned criterion')
        check_printed(series['crit_bin'][0], c_first, point + ' criterion bin 1', problems)
        check_printed(series['crit_bin'][-1], c_last, point + ' criterion bin 20', problems)
        check_printed(series['rho_c'], c_rho, point + ' Spearman rho of the criterion with bin', problems)

        # the stored Spearman correlations are those of the stored curves (bin numbers 0 to 19 or 1 to 20)
        if abs(spearman_with_bin(series['dprime'])[0] - series['rho_d']) > 1e-12:
            problems.append(point + " stored rho_d differs from the Spearman of the stored d' curve")
        if abs(spearman_with_bin(series['crit_bin'])[0] - series['rho_c']) > 1e-12:
            problems.append(point + ' stored rho_c differs from the Spearman of the stored criterion curve')
        series['report_text'] = dict(storage=storage_text, criterion=criterion_text, hit=hit_text, fa=fa_text,
                                     dprime_change=d_change, rho_d=d_rho, rho_c=c_rho)
        run[point] = series

    if problems:
        raise ValueError('model run (%s) disagrees with its report:\n  %s' % (files['source'], '\n  '.join(problems)))
    return run


# ----------------------------------------------------------------------------- the data side
def things_data():
    """The THINGS data side of Fig. S4 (module A5's provided output), checked against its report.
    Returns a dictionary with the 20-bin arrays (dprime, dlo, dhi, H, Hlo, Hhi, F, Flo, Fhi, crit, clo,
    chi, da), n_sessions, n_bins, n_bootstraps, ci_level, the Pearson correlations of the binned d',
    criterion and false alarm rate with the bin number that the report prints, the participants'
    overall hit and false alarm rates (per-image means, with the number of images), the across-image
    SD of d' (with its number of images) and the file paths."""
    for path in (DATA_NPZ, DATA_REPORT, ADJUSTED_REPORT, PER_IMAGE_REPORT):
        if not os.path.exists(path):
            raise FileNotFoundError('THINGS data side: %s is missing (it is a provided file of module %s)'
                                    % (relative(path), 'A3b' if path == PER_IMAGE_REPORT else 'A5'))
    stored = np.load(DATA_NPZ)
    data = dict(files=dict(npz=DATA_NPZ, report=DATA_REPORT, adjusted_report=ADJUSTED_REPORT,
                           per_image_report=PER_IMAGE_REPORT))
    for key in ('dprime', 'dlo', 'dhi', 'H', 'Hlo', 'Hhi', 'F', 'Flo', 'Fhi', 'crit', 'clo', 'chi', 'da'):
        data[key] = np.array(stored[key], dtype=float)

    report = read_text(DATA_REPORT)
    problems = []
    sessions_text, bins_text = find_number(r'([\d,]+) participants; context binned into (\d+) within-image bins',
                                           report, 'the number of sessions')
    data['n_sessions'] = int(to_number(sessions_text))
    data['n_bins'] = int(bins_text)
    data['n_bootstraps'] = int(find_number(r'(\d+) participant bootstraps', report, 'the number of bootstraps')[0])
    data['ci_level'] = int(find_number(r"d'\s+\[(\d+)% CI\]", report, 'the interval level')[0])

    block = report.split('ctx_top10   HITS FROM: target repeats (long lag)')[1].split('HITS FROM')[0]
    row_pattern = (r'^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) \[([\d.]+), ([\d.]+)\]\s+'
                   r'([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]\s+([\d.]+)\s*$')
    rows = re.findall(row_pattern, block, re.M)
    if len(rows) != data['n_bins'] or len(data['dprime']) != data['n_bins']:
        problems.append('%d bin rows in the report, %d bins in the file, %d announced'
                        % (len(rows), len(data['dprime']), data['n_bins']))
    for row in rows:
        index = int(row[0]) - 1
        label = 'data bin %d' % (index + 1)
        check_printed(data['H'][index], row[1], label + ' hit rate', problems)
        check_printed(data['F'][index], row[2], label + ' false alarm rate', problems)
        check_printed(data['dprime'][index], row[3], label + " d'", problems)
        check_printed(data['dlo'][index], row[4], label + " d' lower limit", problems)
        check_printed(data['dhi'][index], row[5], label + " d' upper limit", problems)
        check_printed(data['crit'][index], row[6], label + ' criterion', problems)
        check_printed(data['clo'][index], row[7], label + ' criterion lower limit', problems)
        check_printed(data['chi'][index], row[8], label + ' criterion upper limit', problems)
        check_printed(data['da'][index], row[9], label + ' d_a', problems)
    d_first, d_last, d_change, d_pearson = find_number(
        r"^\s+d': bin1 ([+-][\d.]+) -> bin20 ([+-][\d.]+) \(change ([+-][\d.]+)\);.*r with bin index = ([+-][\d.]+)",
        block, "the d' summary line")
    check_printed(data['dprime'][0], d_first, "data d' bin 1 (summary line)", problems)
    check_printed(data['dprime'][-1], d_last, "data d' bin 20 (summary line)", problems)
    check_printed(data['dprime'][-1] - data['dprime'][0], d_change, "data d' change", problems)
    data['dprime_pearson_printed'] = to_number(d_pearson)
    c_pearson, = find_number(r'^\s+criterion c: bin1 .*r with bin index = ([+-][\d.]+)', block,
                             'the criterion summary line')
    data['criterion_pearson_printed'] = to_number(c_pearson)
    f_pearson, = find_number(r'^\s+FA rate: bin1 .*r with bin index = ([+-][\d.]+)', block,
                             'the false alarm summary line')
    data['false_alarm_pearson_printed'] = to_number(f_pearson)

    per_image = read_text(PER_IMAGE_REPORT)
    data['participants_hit_rate'] = to_number(find_number(r'hit rate\s+mean ([\d.]+)', per_image, 'hit rate mean')[0])
    data['participants_fa_rate'] = to_number(find_number(r'false alarm\s+mean ([\d.]+)', per_image,
                                                         'false alarm mean')[0])
    data['per_image_n'] = int(to_number(find_number(r'images with >=30 old and >=30 new observations: ([\d,]+)',
                                                    per_image, 'the number of images')[0]))
    adjusted = read_text(ADJUSTED_REPORT)
    images_text, sd_text = find_number(r"across-image d' over ([\d,]+) images: SD ([\d.]+)", adjusted,
                                       "the across-image SD of d'")
    data['across_image_sd'] = to_number(sd_text)
    data['across_image_n'] = int(to_number(images_text))

    if problems:
        raise ValueError('THINGS data file disagrees with its report:\n  ' + '\n  '.join(problems))
    return data


def describe_curve(values):
    """Bin 1, bin 20, the change (bin 20 minus bin 1), the ratio (bin 20 / bin 1) and the Spearman
    correlation with the bin number (rho, p) of a 20-bin curve."""
    rho, p = spearman_with_bin(values)
    return dict(first=float(values[0]), last=float(values[-1]), change=float(values[-1] - values[0]),
                ratio=float(values[-1] / values[0]), rho=rho, p=p)
