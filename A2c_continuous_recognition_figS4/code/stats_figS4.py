"""
stats_figS4.py: the uncertainty statistics of Fig. S4 and SI Appendix, Section 2 (test T27 of
statistics/README.md), written to outputs/tests.csv.

WHAT IS TESTED (T27)
  Within each image, trials are ranked into 20 bins by how similar the preceding items were (bin 1 =
  least similar, bin 20 = most similar). The paper reports, for the THINGS data (21,137 participant
  sessions) and for the REM model run through 400 of their trial sequences (4,000 simulated
  participants at each storage probability):
    the d' contrast, bin 20 minus bin 1 (d' fell by 0.51, 0.63 and 1.12)
    the false alarm rate, bin 20 against bin 1 (more than doubled; tenfold)
    Spearman correlations of the binned d', false alarm rate and criterion with the bin number
  The Spearman correlations are over 20 bin means of the same items and participants, so they are
  descriptive. The inference is the contrast between bins 20 and 1, with its confidence interval from
  resampling the unit of replication: participant sessions for the data, simulated participants for
  the model (see INTERVALS for the rows that have one).

INTERVALS
  data side   module A5's saved file (sdt_ctx_top10_target.npz) holds the 95 percent limits of each
              bin from 400 resamples of participant sessions, not the resampled curves, so no interval
              of a contrast is computed here (test T38 of module A5 gives the interval of the same d'
              contrast).
  model side  the saved run in data/provided/ holds bin means only. After a run of continuous_rem.py
              (--model-run regenerated) the simulated participants are resampled: percentile
              bootstrap, 10,000 resamples, numpy.random.default_rng(20261013), one
              integers(0, n, n) draw per resample, in order.
  No multiplicity correction (statistics/README.md).

INPUTS   figs4_inputs.py (the model run and the data side, each checked against its report);
         outputs/continuous_cell_counts.npz (written by continuous_rem.py; regenerated runs only)
OUTPUT   outputs/tests.csv (columns of statistics/README.md)
SEED     20261013 (the bootstrap of the simulated participants). Runtime: about a second (provided),
         about 30 seconds (regenerated).

RUN (from the repository root)
  python A2c_continuous_recognition_figS4/code/stats_figS4.py
  python A2c_continuous_recognition_figS4/code/stats_figS4.py --model-run regenerated
"""
import argparse
import os
import sys

import numpy as np

sys.dont_write_bytecode = True   # write no __pycache__ folder into shared/python or code/ (set before the imports below)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402
import bootstrap  # noqa: E402
import sdt  # noqa: E402
import tests_table  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import continuous_rem  # noqa: E402
import figs4_inputs  # noqa: E402

PATHS = release_paths.module_paths(__file__)
SCRIPT = 'code/stats_figS4.py'
SEED = bootstrap.RELEASE_SEED          # 20261013
NUMBER_OF_RESAMPLES = 10000
LOCATION = 'SI Appendix, Section 2, Continuous Recognition'
DESCRIPTIVE = ('none: descriptive (20 bin means of the same items and participants); the inference is the '
               'contrast of bins 20 and 1')


def pooled_curves(weights, counts):
    """d', criterion, hit and false alarm rate per bin, pooled over the simulated participants with the
    given weights (how often each participant was drawn). One vectorized step: a weighted sum of the
    per-participant counts (participants x bins), then the log-linear SDT of shared/python/sdt.py."""
    hits = weights @ counts['hit_successes']
    hit_trials = weights @ counts['hit_trials']
    false_alarms = weights @ counts['false_alarm_successes']
    false_alarm_trials = weights @ counts['false_alarm_trials']
    dprime, criterion, _, hit_rate, false_alarm_rate = sdt.sdt(hits, hit_trials, false_alarms, false_alarm_trials)
    return dprime, criterion, hit_rate, false_alarm_rate


def bootstrap_model_contrasts(counts, label):
    """Percentile bootstrap over the simulated participants of one storage probability: the d'
    contrast (bin 20 minus bin 1), the false alarm ratio (bin 20 / bin 1) and the criterion change
    (bin 20 minus bin 1). Returns the point values (all participants once), the 95 percent intervals
    and the number of participants."""
    number_of_participants = counts['hit_successes'].shape[0]
    all_once = np.ones(number_of_participants)
    dprime, criterion, _, false_alarm_rate = pooled_curves(all_once, counts)
    point = np.array([dprime[-1] - dprime[0], false_alarm_rate[-1] / false_alarm_rate[0],
                      criterion[-1] - criterion[0]])

    def contrasts_of(resampled_participants):
        weights = np.bincount(resampled_participants, minlength=number_of_participants).astype(float)
        resampled_dprime, resampled_criterion, _, resampled_false_alarms = pooled_curves(weights, counts)
        return [resampled_dprime[-1] - resampled_dprime[0],
                resampled_false_alarms[-1] / resampled_false_alarms[0],
                resampled_criterion[-1] - resampled_criterion[0]]

    resampled = bootstrap.percentile_bootstrap(contrasts_of, number_of_participants, NUMBER_OF_RESAMPLES, SEED)
    low, high = bootstrap.percentile_interval(resampled)
    print('  %s: bootstrap over %d simulated participants, %d resamples, seed %d' % (label, number_of_participants,
                                                                                    NUMBER_OF_RESAMPLES, SEED))
    return point, low, high, number_of_participants


def load_cell_counts(model, point):
    """The per-participant counts of one storage probability from continuous_cell_counts.npz."""
    stored = np.load(model['files']['cell_counts'])
    counts = {}
    for name in ('hit_successes', 'hit_trials', 'false_alarm_successes', 'false_alarm_trials'):
        counts[name] = np.array(stored['%s_%s' % (point, name)], dtype=float)
    return counts


def main():
    parser = argparse.ArgumentParser(description='Uncertainty statistics of Fig. S4 (T27) -> outputs/tests.csv')
    parser.add_argument('--model-run', choices=['provided', 'regenerated', 'auto'], default='provided',
                        help='the saved model run (data/provided/, default), a new run of continuous_rem.py '
                             '(outputs/), or auto')
    arguments = parser.parse_args()

    model = figs4_inputs.model_run(arguments.model_run)
    data = figs4_inputs.things_data()
    regenerated = model['source'] == 'regenerated'
    if regenerated and not os.path.exists(model['files']['cell_counts']):
        sys.exit('stats_figS4.py: outputs/continuous_cell_counts.npz is missing; rerun continuous_rem.py')
    print('model run: %s' % model['label'])
    print('THINGS data side: provided output of module A5 (%s)' % figs4_inputs.relative(data['files']['npz']))

    tests = tests_table.TestsTable(script=SCRIPT)
    sessions = data['n_sessions']
    sessions_text = '{:,}'.format(sessions)
    data_unit = 'participant sessions (Kramer et al., 2023)'
    pending_data = ('none: module A5\'s saved file holds the per-bin 95 percent limits of %d participant-session '
                    'resamples (numpy default_rng(0)), not the resampled curves a contrast interval needs'
                    % data['n_bootstraps'])

    # ------------------------------------------------------------------ the THINGS data
    data_dprime = figs4_inputs.describe_curve(data['dprime'])
    data_false_alarms = figs4_inputs.describe_curve(data['F'])
    data_criterion = figs4_inputs.describe_curve(data['crit'])
    tests.add(test_id='T27', ledger_ids='L6-058',
              location=LOCATION,
              test="THINGS data: within-image d' in the most minus the least similar bin of preceding items (bin 20 "
                   "minus bin 1; 20 bins within image by the mean SPoSE similarity of the 10 most similar preceding "
                   "items; hits from target repeats, false alarms from first showings; log-linear d' of pooled counts). "
                   "Per-bin 95 percent limits: bin 1 [%.3f, %.3f], bin 20 [%.3f, %.3f] (not overlapping: %s)"
                   % (data['dlo'][0], data['dhi'][0], data['dlo'][-1], data['dhi'][-1],
                      'yes' if data['dhi'][-1] < data['dlo'][0] else 'no'),
              statistic_name="d' difference (bin 20 minus bin 1)", statistic=data_dprime['change'],
              n=sessions, unit=data_unit, df='', p='', p_type='none',
              effect_size_name="d' difference in across-image SDs of d' (SD %.3f over %s images)"
                               % (data['across_image_sd'], '{:,}'.format(data['across_image_n'])),
              effect_size=data_dprime['change'] / data['across_image_sd'],
              ci_low='', ci_high='',
              ci_method=pending_data + '; test T38 (module A5) gives the interval of the same d\' contrast',
              resampling_unit='participant sessions', n_resamples='', seed='')
    tests.add(test_id='T27', ledger_ids='L6-066', location=LOCATION,
              test="THINGS data: Spearman correlation of the binned d' with the bin number (1 to 20)",
              statistic_name='Spearman rho', statistic=data_dprime['rho'], n=data['n_bins'],
              unit='bins (descriptive; %s sessions)' % sessions_text, df='', p='',
              p_type='none', effect_size_name='rho', effect_size=data_dprime['rho'], ci_low='',
              ci_high='', ci_method=DESCRIPTIVE, resampling_unit='', n_resamples='', seed='')
    tests.add(test_id='T27', ledger_ids='L6-067', location=LOCATION,
              test='THINGS data: false alarm rate in the most against the least similar bin of preceding items (bin '
                   '20 / bin 1; first showings). Per-bin 95 percent limits: bin 1 [%.4f, %.4f], bin 20 [%.4f, %.4f] '
                   '(not overlapping: %s)' % (data['Flo'][0], data['Fhi'][0], data['Flo'][-1], data['Fhi'][-1],
                                              'yes' if data['Flo'][-1] > data['Fhi'][0] else 'no'),
              statistic_name='false alarm rate ratio (bin 20 / bin 1)', statistic=data_false_alarms['ratio'],
              n=sessions, unit=data_unit, df='', p='', p_type='none',
              effect_size_name='false alarm rate difference (bin 20 minus bin 1)',
              effect_size=data_false_alarms['change'], ci_low='', ci_high='', ci_method=pending_data,
              resampling_unit='participant sessions', n_resamples='', seed='')
    tests.add(test_id='T27', ledger_ids='L6-067', location=LOCATION,
              test='THINGS data: Spearman correlation of the binned false alarm rate with the bin number (1 to 20)',
              statistic_name='Spearman rho', statistic=data_false_alarms['rho'], n=data['n_bins'],
              unit='bins (descriptive; %s sessions)' % sessions_text, df='', p='',
              p_type='none', effect_size_name='rho', effect_size=data_false_alarms['rho'], ci_low='',
              ci_high='', ci_method=DESCRIPTIVE, resampling_unit='', n_resamples='', seed='')
    tests.add(test_id='T27', ledger_ids='L6-059', location=LOCATION,
              test='THINGS data: criterion c in the most minus the least similar bin of preceding items (bin 20 minus '
                   'bin 1; negative = more liberal). Per-bin 95 percent limits: bin 1 [%+.3f, %+.3f], bin 20 '
                   '[%+.3f, %+.3f] (not overlapping: %s)'
                   % (data['clo'][0], data['chi'][0], data['clo'][-1], data['chi'][-1],
                      'yes' if data['chi'][-1] < data['clo'][0] else 'no'),
              statistic_name='criterion change (bin 20 minus bin 1)', statistic=data_criterion['change'],
              n=sessions, unit=data_unit, df='', p='', p_type='none',
              effect_size_name='criterion change (bin 20 minus bin 1)', effect_size=data_criterion['change'],
              ci_low='', ci_high='', ci_method=pending_data, resampling_unit='participant sessions',
              n_resamples='', seed='')
    tests.add(test_id='T27', ledger_ids='L6-059', location=LOCATION,
              test='THINGS data: Spearman correlation of the binned criterion c with the bin number (1 to 20)',
              statistic_name='Spearman rho', statistic=data_criterion['rho'], n=data['n_bins'],
              unit='bins (descriptive; %s sessions)' % sessions_text, df='', p='',
              p_type='none', effect_size_name='rho', effect_size=data_criterion['rho'], ci_low='',
              ci_high='', ci_method=DESCRIPTIVE, resampling_unit='', n_resamples='', seed='')

    # ------------------------------------------------------------------ the model
    for point in figs4_inputs.POINTS:
        series = model[point]
        storage_text = 'storage %.2f' % series['storage']
        point_dprime = figs4_inputs.describe_curve(series['dprime'])
        point_false_alarms = figs4_inputs.describe_curve(series['F'])
        point_criterion = figs4_inputs.describe_curve(series['crit_bin'])
        model_unit = 'simulated participants (%s per storage probability, %d per THINGS trial structure)' % (
            '{:,}'.format(model['simulated_participants']),
            model['simulated_participants'] // continuous_rem.N_TEMPLATES)
        if regenerated:
            values, low, high, participants = bootstrap_model_contrasts(load_cell_counts(model, point), storage_text)
            same_as_curves = (abs(values[0] - point_dprime['change']) < 1e-9 and
                              abs(values[1] - point_false_alarms['ratio']) < 1e-9 and
                              abs(values[2] - point_criterion['change']) < 1e-9)
            if not same_as_curves:
                sys.exit('stats_figS4.py: the cell counts do not give the curves of continuous_bins.npz')
            interval = dict(ci_method='percentile bootstrap', resampling_unit='simulated participants',
                            n_resamples=NUMBER_OF_RESAMPLES, seed=SEED)
            intervals = [(low[0], high[0]), (low[1], high[1]), (low[2], high[2])]
        else:
            participants = model['simulated_participants']
            interval = dict(ci_method='none with the saved run (data/provided/ holds bin means only); a run of '
                                      'continuous_rem.py on the trial structures of the %d sessions (EXTERNAL_INPUTS.md '
                                      'M6, or the Kramer et al. trial file, M1) gives the percentile bootstrap over the '
                                      'simulated participants' % continuous_rem.N_TEMPLATES,
                            resampling_unit='simulated participants', n_resamples='', seed='')
            intervals = [('', ''), ('', ''), ('', '')]
        tests.add(test_id='T27', ledger_ids='L6-058',
                  location=LOCATION,
                  test="REM model run through the THINGS trial sequences, %s: within-item d' in the most minus the "
                       "least similar bin of preceding items (bin 20 minus bin 1; 20 bins within item by the mean "
                       "similarity of the 10 most similar preceding items; log-linear d' of pooled counts)" % storage_text,
                  statistic_name="d' difference (bin 20 minus bin 1)", statistic=point_dprime['change'],
                  n=participants, unit=model_unit, df='', p='', p_type='none',
                  effect_size_name="d' difference (bin 20 minus bin 1)", effect_size=point_dprime['change'],
                  ci_low=intervals[0][0], ci_high=intervals[0][1], **interval)
        tests.add(test_id='T27', ledger_ids='L6-066', location=LOCATION,
                  test="REM model, %s: Spearman correlation of the binned d' with the bin number (1 to 20)" % storage_text,
                  statistic_name='Spearman rho', statistic=point_dprime['rho'], n=len(series['dprime']),
                  unit='bins (descriptive; simulation)', df='', p='',
                  p_type='none', effect_size_name='rho', effect_size=point_dprime['rho'], ci_low='',
                  ci_high='', ci_method=DESCRIPTIVE, resampling_unit='', n_resamples='', seed='')
        tests.add(test_id='T27', ledger_ids='L6-067', location=LOCATION,
                  test='REM model, %s: false alarm rate in the most against the least similar bin of preceding items '
                       '(bin 20 / bin 1; first showings)' % storage_text,
                  statistic_name='false alarm rate ratio (bin 20 / bin 1)', statistic=point_false_alarms['ratio'],
                  n=participants, unit=model_unit, df='', p='', p_type='none',
                  effect_size_name='false alarm rate difference (bin 20 minus bin 1)',
                  effect_size=point_false_alarms['change'], ci_low=intervals[1][0], ci_high=intervals[1][1],
                  **interval)
        tests.add(test_id='T27', ledger_ids='L6-067', location=LOCATION,
                  test='REM model, %s: Spearman correlation of the binned false alarm rate with the bin number (1 to 20)'
                       % storage_text,
                  statistic_name='Spearman rho', statistic=point_false_alarms['rho'], n=len(series['F']),
                  unit='bins (descriptive; simulation)', df='', p='',
                  p_type='none', effect_size_name='rho', effect_size=point_false_alarms['rho'],
                  ci_low='', ci_high='', ci_method=DESCRIPTIVE, resampling_unit='', n_resamples='', seed='')
        if point == 'accuracy-matched':
            # SI Appendix, Section 2 reports the criterion at the higher storage probability only
            tests.add(test_id='T27', ledger_ids='L6-059', location=LOCATION,
                      test='REM model, %s: criterion c in the most minus the least similar bin of preceding items (bin '
                           '20 minus bin 1; negative = more liberal), although the model\'s criterion is fixed'
                           % storage_text,
                      statistic_name='criterion change (bin 20 minus bin 1)', statistic=point_criterion['change'],
                      n=participants, unit=model_unit, df='', p='', p_type='none',
                      effect_size_name='criterion change (bin 20 minus bin 1)', effect_size=point_criterion['change'],
                      ci_low=intervals[2][0], ci_high=intervals[2][1], **interval)
            tests.add(test_id='T27', ledger_ids='L6-059', location=LOCATION,
                      test='REM model, %s: Spearman correlation of the binned criterion c with the bin number (1 to 20)'
                           % storage_text,
                      statistic_name='Spearman rho', statistic=point_criterion['rho'], n=len(series['crit_bin']),
                      unit='bins (descriptive; simulation)', df='', p='',
                      p_type='none', effect_size_name='rho', effect_size=point_criterion['rho'],
                      ci_low='', ci_high='', ci_method=DESCRIPTIVE, resampling_unit='', n_resamples='', seed='')

    tests.write(PATHS['outputs'])
    print('no multiplicity correction (statistics/README.md); seed %d for every bootstrap of this script' % SEED)


if __name__ == '__main__':
    main()
