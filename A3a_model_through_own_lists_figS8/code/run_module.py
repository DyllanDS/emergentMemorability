"""
run_module.py: runs module A3a (SI Appendix, Section 3: the model run through the participants' own
lists, Fig. S8) from start to end.

Steps, in order (each a script in code/; each writes its log to outputs/logs/<step>.log)
   1. extract_inversion.py       the inversion experiment's lists and trials (50 participants)
   2. fit_inversion.py           the model through the inversion experiment (about 2 minutes)
   3. disattenuate_inversion.py  the reliability of the observed scores and the corrected rho
   4. extract_datasets.py        the image set experiments' lists and trials (450 participants)
   5. fit_datasets.py            the model through the image set experiments (about 3 minutes)
   6. bin_by_crowding.py         the crowding of every image in its set
   7. paired_crowding.py         the same image in two or three sets (Fig. S8b, c)
   8. stats_figS8.py             the uncertainty statistics T31 and T32 (under a minute)
   9. make_figureS8.py           Fig. S8 and its plotted values
  10. write_numbers.py           numbers.csv (each number of the paper this module computes) and the
                                 pipeline checks
  Then the four per-trial tables of outputs/model_inputs/ (a copy of the module 03 and 06 data) are
  removed; steps 1 and 4 make them again in seconds.
The run stops at the first step that fails.

Options
  --resamples N        bootstrap resamples of step 8 (the paper uses 10,000; a smaller number only
                       checks the code)
  --skip-fits          skip steps 2 and 5 and use the two model fits already in outputs/
                       (fit_inversion.* and fit_datasets.*, included in the repository); --skip-long is
                       the same option, for run_all.py
  --keep-model-inputs  keep the per-trial tables of outputs/model_inputs/

The model needs the four late-layer THINGS embeddings in shared/embeddings_things/ (alexnet_late.npz,
vgg16_late.npz, resnet18_late.npz, clip_late.npz). If one is missing, the module stops.

Every step runs with at most two processor threads for numpy and scikit-learn (the environment
variables OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and VECLIB_MAXIMUM_THREADS are set to
2 unless already set), so repeated runs on one machine give identical outputs.

Run (from the repository root):  python A3a_model_through_own_lists_figS8/code/run_module.py
Seeds: 0 for the model and for the intervals of paired_crowding.py; 20261013 for the intervals of
stats_figS8.py. Runtime on an Apple M2 Pro: about 6 minutes, about 1 minute with --skip-fits (a full run
writes its step times to outputs/logs/FULL_RUN_COMMAND.txt). Memory: about 1.1 GB at most (the
26,107 x 2,048 table and its principal components).
"""
import argparse
import os
import subprocess
import sys
import time

# Set before the first import from shared/python, so that Python does not write a __pycache__ folder
# there for release_paths itself (the steps below run with PYTHONDONTWRITEBYTECODE=1).
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (first: it stops Python from writing __pycache__ folders)

PATHS = release_paths.module_paths(__file__)
RELEASE_ROOT = release_paths.release_root()
CODE_FOLDER = PATHS['code']
MODULE_NAME = os.path.basename(PATHS['module'])
MODEL_INPUTS = os.path.join(PATHS['outputs'], 'model_inputs')
PER_TRIAL_TABLES = ['inversion_study.csv', 'inversion_test.csv', 'datasets_study.csv', 'datasets_test.csv']

EMBEDDINGS = [release_paths.shared_path('embeddings_things', name + '.npz')
              for name in ('alexnet_late', 'vgg16_late', 'resnet18_late', 'clip_late')]
OTHER_INPUTS = [
    release_paths.shared_path('things', 'things_images.csv'),
    release_paths.shared_path('things', 'things_image_mapping.csv'),
    release_paths.shared_path('ledger', 'paper_numbers.csv'),
    os.path.join(release_paths.find_module('03'), 'data', 'inversion_trials.csv'),
    os.path.join(release_paths.find_module('03'), 'data', 'inversion_design.csv'),
    os.path.join(release_paths.find_module('03'), 'outputs', 'inversion_items_50.csv'),
    os.path.join(release_paths.find_module('06'), 'data', 'maintained_trials.csv'),
    os.path.join(release_paths.find_module('06'), 'data', 'changed_trials.csv'),
    os.path.join(release_paths.find_module('06'), 'data', 'reduced_trials.csv'),
]
FIT_OUTPUTS = ['fit_inversion.txt', 'fit_inversion_items.csv', 'fit_datasets.txt', 'fit_datasets_cells.csv']


def run_step(name, command):
    """Runs one step, shows its output as it comes, and keeps a log in outputs/logs/<name>.log."""
    print('\n==== %s' % name, flush=True)
    environment = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    for thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        environment.setdefault(thread_setting, '2')
    start_time = time.time()
    log_lines = ['$ ' + ' '.join(command).replace(RELEASE_ROOT, '<release>')]
    process = subprocess.Popen(command, cwd=RELEASE_ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        line = line.rstrip('\n').replace(RELEASE_ROOT, '<release>')
        print(line, flush=True)
        log_lines.append(line)
    exit_code = process.wait()
    seconds = time.time() - start_time
    log_lines.append('[%s finished with exit code %d after %.0f s]' % (name, exit_code, seconds))
    with open(os.path.join(PATHS['logs'], name + '.log'), 'w') as handle:
        handle.write('\n'.join(log_lines) + '\n')
    if exit_code != 0:
        sys.exit('run_module.py: step %s failed (exit code %d); see outputs/logs/%s.log' % (name, exit_code, name))
    print('[%s done in %.0f s]' % (name, seconds), flush=True)
    return seconds


def write_full_run_note(runtimes):
    """outputs/logs/FULL_RUN_COMMAND.txt: the command of the full run and the measured step times."""
    lines = ['Module A3a (SI Appendix, Section 3: the model run through the participants\' own lists, Fig. S8):',
             'the full run, on the participants of Fig. 3e (50) and of Figs. 6 and 7 (450).',
             '',
             'Run from the repository root (one Python process per step, about 1 GB of memory at most):',
             '',
             '    python %s/code/run_module.py' % MODULE_NAME,
             '',
             'Steps and times of this full run (%s, two threads):' % time.strftime('%Y-%m-%d')]
    for name, seconds in runtimes.items():
        lines.append('  %s: %.0f s' % (name, seconds))
    lines.append('  Total: %.0f s' % sum(runtimes.values()))
    lines.extend(['',
                  'outputs/ holds the results of this run.',
                  '',
                  'Shorter runs (checks of the code):',
                  '    python %s/code/run_module.py --resamples 200' % MODULE_NAME,
                  '        the uncertainty statistics with 200 instead of 10,000 bootstrap resamples',
                  '    python %s/code/run_module.py --skip-fits' % MODULE_NAME,
                  '        reuses the two model fits in outputs/ (fit_inversion.*, fit_datasets.*)'])
    with open(os.path.join(PATHS['logs'], 'FULL_RUN_COMMAND.txt'), 'w') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Module A3a: the model run through the participants\' own lists '
                                                 '(Fig. S8).')
    parser.add_argument('--resamples', type=int, default=10000, help='bootstrap resamples of stats_figS8.py')
    parser.add_argument('--skip-fits', '--skip-long', dest='skip_fits', action='store_true',
                        help='reuse the model fits already in outputs/ (the same as --skip-long of run_all.py)')
    parser.add_argument('--keep-model-inputs', action='store_true',
                        help='keep the per-trial tables of outputs/model_inputs/')
    arguments = parser.parse_args()
    python = sys.executable
    runtimes = {}

    print('Module A3a: the model run through the participants\' own lists (SI Appendix, Section 3; Fig. S8)')
    print('participants: the 50 of the inversion experiment (Fig. 3e) and the 450 of the image set experiments '
          '(Figs. 6 and 7)')
    missing_embeddings = [path for path in EMBEDDINGS if not os.path.exists(path)]
    if missing_embeddings:
        sys.exit('run_module.py: the model needs the four late-layer THINGS embeddings, '
                 'and these are missing:\n  %s\nPut them in shared/embeddings_things/ (see '
                 'shared/embeddings_things/README.md); the module stops without them.'
                 % '\n  '.join(os.path.relpath(path, RELEASE_ROOT) for path in missing_embeddings))
    print('input found: the four late-layer embeddings in shared/embeddings_things/')
    missing = [path for path in OTHER_INPUTS if not os.path.exists(path)]
    if missing:
        sys.exit('run_module.py: these inputs are not found:\n  %s\n(module 03 writes outputs/inversion_items_50.csv: run it first, '
                 'see 03_inversion_experiment_fig3/README.md)'
                 % '\n  '.join(os.path.relpath(path, RELEASE_ROOT) for path in missing))

    def script(name):
        return [python, os.path.join(CODE_FOLDER, name + '.py')]

    runtimes['extract_inversion'] = run_step('extract_inversion', script('extract_inversion'))
    if arguments.skip_fits:
        for name in FIT_OUTPUTS:
            if not os.path.exists(os.path.join(PATHS['outputs'], name)):
                sys.exit('run_module.py --skip-fits: outputs/%s is missing; run without --skip-fits' % name)
        print('\nsteps 2 and 5 skipped (--skip-fits): using the model fits already in outputs/')
    else:
        runtimes['fit_inversion'] = run_step('fit_inversion', script('fit_inversion'))
    runtimes['disattenuate_inversion'] = run_step('disattenuate_inversion', script('disattenuate_inversion'))
    runtimes['extract_datasets'] = run_step('extract_datasets', script('extract_datasets'))
    if not arguments.skip_fits:
        runtimes['fit_datasets'] = run_step('fit_datasets', script('fit_datasets'))
    runtimes['bin_by_crowding'] = run_step('bin_by_crowding', script('bin_by_crowding'))
    runtimes['paired_crowding'] = run_step('paired_crowding', script('paired_crowding'))
    runtimes['stats_figS8'] = run_step('stats_figS8', script('stats_figS8') + ['--resamples', str(arguments.resamples)])
    runtimes['make_figureS8'] = run_step('make_figureS8', script('make_figureS8'))
    runtimes['write_numbers'] = run_step('write_numbers', script('write_numbers'))

    if arguments.keep_model_inputs:
        print('\nper-trial tables kept in outputs/model_inputs/ (--keep-model-inputs)')
    else:
        for name in PER_TRIAL_TABLES:
            path = os.path.join(MODEL_INPUTS, name)
            if os.path.exists(path):
                os.remove(path)
        print('\nremoved the per-trial tables of outputs/model_inputs/ (%s); code/extract_inversion.py and '
              'code/extract_datasets.py make them again' % ', '.join(PER_TRIAL_TABLES))

    if not arguments.skip_fits and arguments.resamples == 10000:
        write_full_run_note(runtimes)
    print('\nModule A3a finished. Runtimes: %s; total %.0f s'
          % (', '.join('%s %.0f s' % (name, seconds) for name, seconds in runtimes.items()), sum(runtimes.values())))


if __name__ == '__main__':
    main()
