"""
write_numbers.py: this module's numbers.csv (every number of SI Appendix, Section 3 that comes from the
model run through the participants' own lists, Fig. S8, and the main-text sentence that refers to it)
and the pipeline checks, from this module's run (the 50 participants of Fig. 3e and the 450 of
Figs. 6 and 7).

INPUTS (outputs/ of this module, written by the earlier steps of run_module.py)
  fit_inversion.txt, fit_inversion_items.csv, disattenuation.txt (the inversion run)
  fit_datasets.txt (the image set run)
  paired_crowding.txt, paired_crowding_summary.csv (the same-image contrast)
  tests.csv, written by stats_figS8.py (the 95% intervals and the paired t test whose p the paper prints)
  model_inputs/datasets_sample.csv (the image-by-set cells and the images in two or more sets)
  logs/fit_inversion.log, written by run_module.py when it runs fit_inversion.py (the share of the
      variance the 49 components keep, for the pipeline checks)
  shared/ledger/paper_numbers.csv (the printed values and quotes)
  Every text file is read by the labels of its rows, never by line number.

OUTPUTS (in outputs/)
  numbers.csv          the 29 ledger items of this module (mode rerun; paper_value = the printed value;
                       match by the rule of each ledger row)
  pipeline_checks.csv  further checks of the run: the explained variance of the 49 components, the
                       noise grids and the chosen values, the cells and the images in two or more sets

RUN (from the repository root; a second):  python A3a_model_through_own_lists_figS8/code/write_numbers.py
"""
import csv
import os
import re
import sys

sys.dont_write_bytecode = True   # no __pycache__ folders inside the repository (set before the imports below)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402
import numbers_table  # noqa: E402

PATHS = release_paths.module_paths(__file__)
RUN_FOLDER = PATHS['outputs']
MODE = 'rerun'


# ---------------------------------------------------------------------------------- reading
def read_text(folder, file_name):
    with open(os.path.join(folder, file_name), encoding='utf-8') as handle:
        return handle.read()


def find(pattern, text, what):
    """The first match of a row-label pattern; stops with a clear message if the row is missing."""
    match = re.search(pattern, text, re.M)
    if match is None:
        sys.exit('write_numbers.py: could not find %s' % what)
    return match


def inversion_fit(folder):
    """The numbers of the inversion run (the fit_inversion report and its item table)."""
    text = read_text(folder, 'fit_inversion.txt')
    run = {}
    blocks = find(r'(\d+) subject-blocks x (\d+) simulated participants', text, 'the subject-blocks line')
    run['subject_blocks'] = int(blocks.group(1))
    run['simulated_participants'] = int(blocks.group(2))
    space = find(r'space: (.+) reduced to (\d+) dimensions', text, 'the space line')
    run['networks'] = space.group(1).split(' + ')
    run['dimensions'] = int(space.group(2))
    storage = find(r'storage (\d+(?:\.\d+)?), stored-correctly (\d+(?:\.\d+)?)', text, 'the storage line')
    run['storage'] = float(storage.group(1))
    run['copied_correctly'] = float(storage.group(2))
    observed = find(r'^observed hit ([\d.]+), FA ([\d.]+), d\' ([-\d.]+)', text, 'the observed operating point')
    run['observed_hit'] = float(observed.group(1))
    run['observed_fa'] = float(observed.group(2))
    sweep = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 9 and re.match(r'^[\d.]+$', parts[0]):
            sweep.append(dict(sigma=float(parts[0]), criterion=float(parts[1]), hit=float(parts[2]),
                              fa=float(parts[3]), dprime=float(parts[4]), rho_cr=float(parts[5]),
                              rho_hit=float(parts[6]), rho_fa=float(parts[7]), rho_published=float(parts[8])))
    run['sweep'] = sweep
    chosen = find(r'closest to the observed operating point: sigma ([\d.]+), criterion ([-\d.e+]+), hit ([\d.]+), '
                  r'FA ([\d.]+)', text, 'the chosen operating point')
    run['sigma'] = float(chosen.group(1))
    run['criterion'] = float(chosen.group(2))
    run['model_hit'] = float(chosen.group(3))
    run['model_fa'] = float(chosen.group(4))
    items = pd.read_csv(os.path.join(folder, 'fit_inversion_items.csv'), float_precision='round_trip')
    run['items'] = items
    run['n_items'] = len(items)
    tests_per_item = items['nOld'] + items['nNew']
    run['tests_per_item'] = (int(tests_per_item.min()), int(tests_per_item.max()))
    for name, first, second in (('cr', 'pred_cr', 'cr'), ('hit', 'pred_hit', 'hitRate'), ('fa', 'pred_fa', 'faRate'),
                                ('model_published', 'pred_cr', 'origMem'), ('observed_published', 'cr', 'origMem')):
        result = spearmanr(items[first], items[second])
        run['rho_' + name] = float(result.statistic)
        run['p_' + name] = float(result.pvalue)
    return run


def reliability_of(folder, inversion):
    """The reliability of the observed item scores and the corrected rho (disattenuation.txt), with the
    reliability recomputed at full precision from the item table by the formula of
    disattenuate_inversion.py: 1 - (mean binomial sampling variance of CR) / (variance of CR over items)."""
    text = read_text(folder, 'disattenuation.txt')
    printed_reliability = float(find(r'reliability = 1 - share = ([\d.]+)', text, 'reliability').group(1))
    items = inversion['items']
    sampling_variance = (items['hitRate'] * (1 - items['hitRate']) / items['nOld']
                         + items['faRate'] * (1 - items['faRate']) / items['nNew'])
    reliability = 1 - float(sampling_variance.mean()) / float(items['cr'].var(ddof=1))
    if abs(reliability - printed_reliability) > 5e-5:
        sys.exit('write_numbers.py: the reliability does not match disattenuation.txt')
    return dict(reliability=reliability, median_old=float(items['nOld'].median()),
                median_new=float(items['nNew'].median()),
                rho_disattenuated=inversion['rho_cr'] / np.sqrt(reliability))


def datasets_fit(folder):
    """The numbers of the image set run (the fit_datasets report)."""
    text = read_text(folder, 'fit_datasets.txt')
    run = {}
    observed = find(r'observed overall rates \(hit ([\d.]+), FA ([\d.]+)\)', text, 'the observed rates')
    run['observed_hit'] = float(observed.group(1))
    run['observed_fa'] = float(observed.group(2))
    sweep = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4 and all(re.match(r'^-?[\d.]+$', part) for part in parts):
            sweep.append(dict(sigma=float(parts[0]), hit=float(parts[1]), fa=float(parts[2]), dprime=float(parts[3])))
    run['sweep'] = sweep
    chosen = find(r'chosen sigma ([\d.]+), criterion ([-\d.e+]+): model hit ([\d.]+), FA ([\d.]+)', text,
                  'the chosen operating point')
    run['sigma'] = float(chosen.group(1))
    run['criterion'] = float(chosen.group(2))
    run['model_hit'] = float(chosen.group(3))
    run['model_fa'] = float(chosen.group(4))
    return run


def paired_contrast(folder):
    """The paired same-image contrast (the paired_crowding report and summary)."""
    text = read_text(folder, 'paired_crowding.txt')
    summary = pd.read_csv(os.path.join(folder, 'paired_crowding_summary.csv'), float_precision='round_trip')
    run = {}
    counts = find(r'([\d,]+) images appear in two or more sets: ([\d,]+) in two sets, ([\d,]+) in three', text,
                  'the image counts')
    run['images'] = int(counts.group(1).replace(',', ''))
    run['images_two_sets'] = int(counts.group(2).replace(',', ''))
    run['images_three_sets'] = int(counts.group(3).replace(',', ''))
    for measure, key in (('d-prime', 'dprime'), ('hit rate', 'hit'), ('false alarm rate', 'fa')):
        for source in ('observed', 'model'):
            row = summary[(summary['measure'] == measure) & (summary['source'] == source)].iloc[0]
            run['%s_%s' % (source, key)] = (float(row['mean']), float(row['lo_images']), float(row['hi_images']))
    resamples = find(r'percentiles of ([\d,]+) image resamples', text, 'the number of resamples')
    run['image_resamples'] = int(resamples.group(1).replace(',', ''))
    split = find(r'Split of the paired d-prime effect: observed ([-\d.]+) from hits and ([-\d.]+) from false\s+alarms; '
                 r'model ([-\d.]+) from hits and ([-\d.]+) from false alarms', text, 'the d-prime split')
    run['split_observed'] = (float(split.group(1)), float(split.group(2)))
    run['split_model'] = (float(split.group(3)), float(split.group(4)))
    run['ratio_dprime'] = run['model_dprime'][0] / run['observed_dprime'][0]
    return run


def tests_row(tests, test_id, statistic_name, statistic=None):
    """The one row of this module's tests.csv written by stats_figS8.py with this test id and statistic
    name (and, when given, this statistic, to 6 significant digits); stops if there is not exactly one."""
    rows = tests[(tests['test_id'] == test_id) & (tests['statistic_name'] == statistic_name)
                 & (tests['script'] == 'code/stats_figS8.py')]
    if statistic is not None:
        rows = rows[(rows['statistic'] - statistic).abs() <= 1e-5 * max(1.0, abs(statistic))]
    if len(rows) != 1:
        sys.exit('write_numbers.py: expected one %s row "%s" in tests.csv, found %d'
                 % (test_id, statistic_name, len(rows)))
    return rows.iloc[0]


def log_value(pattern):
    """A value printed in this module's run log of fit_inversion.py (outputs/logs/), if the log exists."""
    path = os.path.join(PATHS['logs'], 'fit_inversion.log')
    if not os.path.exists(path):
        return None
    match = re.search(pattern, read_text(PATHS['logs'], 'fit_inversion.log'))
    return match.group(1) if match else None


# ---------------------------------------------------------------------------------- formatting
def signed(value, decimals=3):
    return '%+.*f' % (decimals, value)


def interval(low, high, decimals=3):
    return '[%s, %s]' % ('%.*f' % (decimals, low), '%.*f' % (decimals, high))


def main():
    inversion = inversion_fit(RUN_FOLDER)
    reliability = reliability_of(RUN_FOLDER, inversion)
    image_sets = datasets_fit(RUN_FOLDER)
    pairs = paired_contrast(RUN_FOLDER)
    datasets_sample = pd.read_csv(os.path.join(RUN_FOLDER, 'model_inputs', 'datasets_sample.csv'))
    datasets_all = datasets_sample.set_index('image_set').loc['all three sets']
    tests = pd.read_csv(os.path.join(RUN_FOLDER, 'tests.csv'), float_precision='round_trip')
    # item intervals of the correlations over the 40 critical items (T31)
    item_interval = {}
    for name in ('cr', 'hit', 'fa', 'model_published', 'observed_published'):
        row = tests_row(tests, 'T31', 'Spearman rho', inversion['rho_' + name])
        item_interval[name] = (float(row['ci_low']), float(row['ci_high']))
    # the paired t test of the observed change in d′ (its p) and the part of the observed change carried
    # by false alarms (image interval) (T32)
    dprime_t_test = tests_row(tests, 'T32', 't')
    false_alarm_part = tests_row(tests, 'T32', 'd′ change from false alarms')

    # ------------------------------------------------------------------ pipeline checks
    explained_variance = log_value(r'49 components keep ([\d.]+) of the variance')
    checks = [
        ('PCA space: share of the variance of the 2,048 standardized columns kept by 49 components',
         explained_variance if explained_variance is not None else 'not available: run fit_inversion.py '
         '(run_module.py without --skip-fits)',
         'printed by fit_inversion.py and fit_datasets.py'),
        ('inversion sigma grid (fraction of each feature\'s SD) and the chosen value',
         ', '.join('%g' % row['sigma'] for row in inversion['sweep']) + '; chosen %g' % inversion['sigma'],
         'the sigma whose hit rate is closest to the observed hit rate'),
        ('image set sigma grid and the chosen value',
         ', '.join('%g' % row['sigma'] for row in image_sets['sweep']) + '; chosen %g' % image_sets['sigma'],
         'sweep on 900 random subject-blocks, 10 simulated participants each'),
        ('cells and shared images of the image sets (at least 5 old and 5 new trials per cell)',
         '{:,} cells, {:,} images in two or more sets'.format(int(datasets_all['cells_with_5_old_and_5_new']),
                                                              int(datasets_all['images_in_two_or_more_sets'])),
         'SI Appendix, Section 3 and Fig. S8b: 1,023 images tested in more than one set'),
    ]
    with open(os.path.join(RUN_FOLDER, 'pipeline_checks.csv'), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['check', 'value', 'note'])
        for check in checks:
            writer.writerow(check)
    print('PIPELINE CHECKS')
    for check in checks:
        print('  %s: %s  (%s)' % check)

    # ------------------------------------------------------------------ numbers.csv (29 ledger items)
    numbers = numbers_table.NumbersTable(script='code/write_numbers.py')
    observed_dprime = pairs['observed_dprime']
    model_dprime = pairs['model_dprime']
    numbers.add('L3-045', computed_value='model against the published scores over the %d critical items: rho = %.4f '
                '(observed %.4f); same-image change in d′ over %s images, more minus less distinctive set: model %s '
                '%s, observed %s %s'
                % (inversion['n_items'], inversion['rho_model_published'], inversion['rho_observed_published'],
                   '{:,}'.format(pairs['images']), signed(model_dprime[0], 4), interval(*model_dprime[1:3], 4),
                   signed(observed_dprime[0], 4), interval(*observed_dprime[1:3], 4)),
                mode=MODE, note='Spearman rho over the 40 critical items between the published memorability score '
                '(Kramer et al., 2023) and the model\'s predicted hit rate minus false alarm rate (the observed rho in '
                'parentheses), and the mean same-image change in d′ between the set where an image\'s 20 nearest '
                'set-mates were least similar and the set where they were most similar, model and observed, with '
                '95% intervals over 10,000 image resamples', script='code/fit_inversion.py; code/paired_crowding.py')
    numbers.add('L7-001', computed_value=[inversion['dimensions'], len(inversion['networks'])], mode=MODE,
                note='Number of principal components and number of networks: the late layers %s (VGG16 as '
                'torchvision vgg16_bn, CLIP as ViT-B/32; 512 values each, principal components except for ResNet18, '
                'every column standardized, 2,048 columns) reduced to %d principal components'
                % (' + '.join(inversion['networks']), inversion['dimensions']), script='code/fit_inversion.py')
    numbers.add('L7-002', computed_value='inversion: sigma %g, criterion %.3g; image sets: sigma %g, criterion %.4g; '
                'storage %g, copied correctly %g' % (inversion['sigma'], inversion['criterion'], image_sets['sigma'],
                                                    image_sets['criterion'], inversion['storage'],
                                                    inversion['copied_correctly']),
                mode=MODE, note='The two settings chosen from each experiment\'s overall rates, the noise on stored '
                'features (sigma, a fraction of each feature\'s SD over THINGS) and the response criterion; the '
                'storage (0.30) and copy (0.60) probabilities are fixed',
                script='code/fit_inversion.py; code/fit_datasets.py')
    numbers.add('L7-003', computed_value='inversion: observed hit %.3f, FA %.3f, model hit %.3f, FA %.3f; image sets: '
                'observed hit %.3f, FA %.3f, model hit %.3f, FA %.3f'
                % (inversion['observed_hit'], inversion['observed_fa'], inversion['model_hit'], inversion['model_fa'],
                   image_sets['observed_hit'], image_sets['observed_fa'], image_sets['model_hit'],
                   image_sets['model_fa']),
                mode=MODE, note='Overall hit and false alarm rates over all kept test trials, observed and predicted '
                'by the model at the chosen noise and criterion, in the inversion experiment (50 participants) and '
                'the image set experiments (450 participants)',
                script='code/fit_inversion.py; code/fit_datasets.py')
    numbers.add('L7-004', computed_value=round(reliability['rho_disattenuated'], 4), mode=MODE,
                note='Spearman rho of Fig. S8a divided by the square root of the reliability of the observed hit rate '
                'minus false alarm rate (reliability %.4f: 1 minus the mean binomial sampling variance over the '
                'variance across the 40 items; median %g old and %g new trials per item; raw rho %.4f)'
                % (reliability['reliability'], reliability['median_old'], reliability['median_new'], inversion['rho_cr']),
                script='code/disattenuate_inversion.py')
    numbers.add('L7-005', computed_value=[inversion['rho_fa'], *item_interval['fa'], inversion['p_fa'],
                                          inversion['rho_hit'], *item_interval['hit'], inversion['p_hit']], mode=MODE,
                note='Spearman rho over the 40 critical items between the model\'s predicted and the observed false '
                'alarm rates, its 95% interval over 10,000 item resamples and two-sided p (df = 38), then the same for '
                'the hit rates (tests.csv, T31)',
                script='code/fit_inversion.py; code/stats_figS8.py')
    numbers.add('L7-006', computed_value=[inversion['rho_model_published'], *item_interval['model_published'],
                                          inversion['rho_observed_published'], *item_interval['observed_published'],
                                          inversion['p_observed_published']], mode=MODE,
                note='Spearman rho over the 40 critical items between the published memorability score (Kramer et al., '
                '2023) and the model\'s predicted hit rate minus false alarm rate, with its 95% interval over 10,000 '
                'item resamples, then the same for the observed hit rate minus false alarm rate of the same 50 '
                'participants (the rho of Fig. 3e) with its two-sided p (df = 38) (tests.csv, T31)',
                script='code/fit_inversion.py; code/stats_figS8.py')
    numbers.add('L7-007', computed_value=pairs['images'], mode=MODE,
                note='Images tested in two or more image sets with at least 5 old and 5 new trials in each set (%d in '
                'two sets, %d in three; %d image-by-set cells)' % (pairs['images_two_sets'], pairs['images_three_sets'],
                                                                 int(datasets_all['cells_with_5_old_and_5_new'])),
                script='code/extract_datasets.py; code/paired_crowding.py')
    numbers.add('L7-008', computed_value=round(observed_dprime[0], 4), mode=MODE,
                note='Mean over the images of the observed change in d′, the set where the image\'s 20 nearest '
                'set-mates were least similar minus the set where they were most similar (rates kept between 1/(2N) '
                'and 1 - 1/(2N)); 95%% interval %s over images (seed 0); paired t(%d) = %.2f, p = %.2g'
                % (interval(observed_dprime[1], observed_dprime[2], 4), int(dprime_t_test['df']),
                   float(dprime_t_test['statistic']), float(dprime_t_test['p'])),
                script='code/paired_crowding.py; code/stats_figS8.py')
    numbers.add('L7-009', computed_value=[round(pairs['ratio_dprime'], 3), round(model_dprime[0], 4)], mode=MODE,
                note='Ratio of the model\'s to the observed mean same-image change in d′, and the model\'s mean change '
                'in d′ on the same pairs (95%% interval %s over images, seed 0)'
                % interval(model_dprime[1], model_dprime[2], 4), script='code/paired_crowding.py')
    hit_fa_text = ('observed hits %s %s, FA %s %s; model hits %s %s, FA %s %s'
                   % (signed(pairs['observed_hit'][0], 4), interval(*pairs['observed_hit'][1:3], 4),
                      signed(pairs['observed_fa'][0], 4), interval(*pairs['observed_fa'][1:3], 4),
                      signed(pairs['model_hit'][0], 4), interval(*pairs['model_hit'][1:3], 4),
                      signed(pairs['model_fa'][0], 4), interval(*pairs['model_fa'][1:3], 4)))
    split_text = ('d′ change from hits / false alarms: observed %.3f / %.3f, model %.3f / %.3f'
                  % (pairs['split_observed'] + pairs['split_model']))
    numbers.add('L7-010', computed_value=hit_fa_text + '; ' + split_text + ' (each rate moved half its mean change '
                'either side of its mean over the paired cells)', mode=MODE,
                note='Mean same-image change in false alarm rate and in hit rate, model and observed, with 95% '
                'intervals over 10,000 image resamples (the bars of Fig. S8c), and the d′ change carried by the hits '
                'and by the false alarms', script='code/paired_crowding.py')
    numbers.add('L7-011', computed_value='rho = %.4f, p = %.4f; %d items; %d to %d tests per item'
                % (inversion['rho_cr'], inversion['p_cr'], inversion['n_items'], inversion['tests_per_item'][0],
                   inversion['tests_per_item'][1]), mode=MODE,
                note='Fig. S8a: Spearman rho and two-sided p over the 40 critical items between the model\'s predicted '
                'and the observed hit rate minus false alarm rate, the number of items and the tests per item',
                script='code/make_figureS8.py')
    numbers.add('L7-012', computed_value='observed %s %s; model %s %s'
                % (signed(observed_dprime[0], 4), interval(*observed_dprime[1:3], 4), signed(model_dprime[0], 4),
                   interval(*model_dprime[1:3], 4)), mode=MODE,
                note='Fig. S8b: mean same-image change in d′ (more minus less distinctive set), observed and model, '
                'with 95% percentile intervals over 10,000 image resamples (seed 0)', script='code/make_figureS8.py')
    numbers.add('L7-013', computed_value=hit_fa_text, mode=MODE,
                note='Fig. S8c: mean same-image change in hit rate and in false alarm rate, observed and model, with '
                '95% percentile intervals over 10,000 image resamples (seed 0)', script='code/make_figureS8.py')
    numbers.add('L7-014', computed_value=[inversion['dimensions'], len(inversion['networks'])], mode=MODE,
                note='Number of model features (principal components) and number of networks whose late layers '
                'build them (%s)' % ', '.join(inversion['networks']), script='code/fit_inversion.py')
    numbers.add('L7-015', computed_value='sigma %g, criterion %.3g (inversion); sigma %g, criterion %.4g (image sets)'
                % (inversion['sigma'], inversion['criterion'], image_sets['sigma'], image_sets['criterion']), mode=MODE,
                note='Noise (a fraction of each feature\'s SD over THINGS, from a grid) and criterion chosen for each '
                'experiment from its overall rates: the criterion gives the observed false alarm rate, the noise the '
                'closest hit rate', script='code/fit_inversion.py; code/fit_datasets.py')
    numbers.add('L7-016', computed_value=inversion['n_items'], mode=MODE,
                note='Critical items of the memorability inversion experiment in Fig. S8a (the items of Fig. 3e)',
                script='code/fit_inversion.py')
    numbers.add('L7-017', computed_value=[inversion['rho_cr'], *item_interval['cr'], inversion['p_cr']], mode=MODE,
                note='Fig. S8a: Spearman rho over the 40 critical items between the model\'s predicted and the observed '
                'hit rate minus false alarm rate, its 95% interval over 10,000 item resamples (tests.csv, T31) and '
                'two-sided p (df = 38)', script='code/fit_inversion.py; code/stats_figS8.py')
    numbers.add('L7-019', computed_value=pairs['images'], mode=MODE,
                note='Images tested in two or more image sets, each compared with itself in Fig. S8b and c',
                script='code/paired_crowding.py')
    numbers.add('L7-020', computed_value=[observed_dprime[0], observed_dprime[1], observed_dprime[2],
                                          float(dprime_t_test['p']), model_dprime[0], model_dprime[1],
                                          model_dprime[2]], mode=MODE,
                note='Mean same-image change in d′ (more minus less distinctive set): observed, with its 95%% '
                'interval over 10,000 image resamples and the two-sided p of the paired t test over images '
                '(t(%d) = %.2f; tests.csv, T32), then predicted by the model, with its 95%% interval over 10,000 '
                'image resamples; the two intervals are the error bars of Fig. S8b (seed 0)'
                % (int(dprime_t_test['df']), float(dprime_t_test['statistic'])),
                script='code/paired_crowding.py; code/stats_figS8.py')
    numbers.add('L7-022', computed_value='95%% percentile intervals over %s image resamples (seed 0)'
                % '{:,}'.format(pairs['image_resamples']),
                mode=MODE,
                note='Error bars of Fig. S8b and c: 2.5 and 97.5 percentiles of 10,000 resamples of the images '
                '(paired_crowding.py, seed 0)',
                script='code/paired_crowding.py')
    # the split of the same-image change in d′ into hits and false alarms (SI Appendix, Section 3)
    numbers.add('L7-026', computed_value=[float(false_alarm_part['statistic']), float(false_alarm_part['ci_low']),
                                          float(false_alarm_part['ci_high']), observed_dprime[0]], mode=MODE,
                note='The part of the observed same-image change in d′ carried by false alarms (each rate moved half '
                'its mean change either side of its mean over the paired cells), its 95% interval over 10,000 image '
                'resamples (tests.csv, T32), and the total observed change in d′',
                script='code/paired_crowding.py; code/stats_figS8.py')
    # statements of SI Appendix, Section 3 about how the model is run
    numbers.add('L7-101', computed_value='familiarity = mean over the traces of the study list of the product over '
                'stored features of 1 + %g x (N(trace value; test value, noise SD) / f(trace value) - 1), f = the '
                'feature\'s density over THINGS' % inversion['copied_correctly'], mode=MODE,
                note='The model\'s likelihood ratio divides by each feature\'s density over THINGS (a Gaussian kernel '
                'density from 6,000 photographs), so feature values common in THINGS count for less',
                script='code/fit_inversion.py; code/fit_datasets.py')
    numbers.add('L7-102', computed_value='inversion: sigma %g, criterion %.3g; image sets: sigma %g, criterion %.4g; no '
                'parameter per item' % (inversion['sigma'], inversion['criterion'], image_sets['sigma'],
                                        image_sets['criterion']), mode=MODE,
                note='The only settings fitted: one noise value and one criterion per experiment, from its overall hit '
                'and false alarm rates', script='code/fit_inversion.py; code/fit_datasets.py')
    numbers.add('L7-103', computed_value='observed %s, model %s' % (signed(observed_dprime[0], 4),
                                                                    signed(model_dprime[0], 4)), mode=MODE,
                note='Fig. S8b: the change in d′ from the set where an image\'s 20 nearest set-mates were most similar '
                '(most crowded) to the set where they were least similar (least crowded), observed (blue) and '
                'predicted by the model (gray)', script='code/paired_crowding.py; code/make_figureS8.py')
    numbers.add('L7-104', computed_value='inversion: %d subject-blocks x %d simulated participants; image sets: every '
                'subject-block of the 450 participants' % (inversion['subject_blocks'],
                                                           inversion['simulated_participants']), mode=MODE,
                note='The model receives the study list and the test trials of each block of each participant (a '
                'subject-block), in the inversion experiment and in the image set experiments',
                script='code/fit_inversion.py; code/fit_datasets.py')
    # three summary sentences of SI Appendix, Section 3 that rest on the values above
    fig_s8a_text = 'Fig. S8a: model against observed memorability, rho = %.4f, p = %.4f (40 items)' % (
        inversion['rho_cr'], inversion['p_cr'])
    same_image_text = ('same-image change in d′: observed %s, model %s (%.2f of the observed change)'
                       % (signed(observed_dprime[0]), signed(model_dprime[0]), pairs['ratio_dprime']))
    numbers.add('ADD-014', computed_value=fig_s8a_text, mode=MODE,
                note='The model run through the participants\' own study lists and test trials, compared with what '
                'they remembered: the Spearman rho of Fig. S8a over the 40 critical items',
                script='code/fit_inversion.py')
    numbers.add('ADD-015', computed_value='%s; against the published scores rho = %.4f (model) and %.4f (observed); %s'
                % (fig_s8a_text, inversion['rho_model_published'], inversion['rho_observed_published'], same_image_text),
                mode=MODE, note='The rho of Fig. S8a, the model\'s and the observed correlation with the published '
                'scores over the 40 critical items (the inversion), and the mean same-image change in d′, observed '
                'and model', script='code/fit_inversion.py; code/paired_crowding.py')
    numbers.add('ADD-016', computed_value='%s; %s' % (fig_s8a_text, same_image_text), mode=MODE,
                note='The rho of Fig. S8a and the mean same-image change in d′ between the more and the less '
                'distinctive set, observed and model, with the model\'s share of the observed change',
                script='code/fit_inversion.py; code/paired_crowding.py')
    numbers.write(RUN_FOLDER)


if __name__ == '__main__':
    main()
