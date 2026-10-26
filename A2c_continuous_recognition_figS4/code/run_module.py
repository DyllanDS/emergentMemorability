"""
run_module.py: runs module A2c (SI Appendix, Section 2, Continuous Recognition; Fig. S4) from start
to end.

MODE
  The model side needs the trial structures of 400 sessions of Kramer et al. (2023), and the data side
  needs their trial file; neither is part of the release (EXTERNAL_INPUTS.md, M6 and M1).
  - provided (the default, nothing to add): the model lines and numbers come from the saved run in
    data/provided/, the run the paper reports, and the data points from module A5's provided output.
    The script prints "PROVIDED OUTPUT: regenerating needs ..." and every number is read and checked
    from those files.
  - regenerate: as soon as the trial structures are there (data/trial_templates_400.npz, or the trial
    table module A5 builds from the trial file in external/kramer_trials/), the model is rerun
    (continuous_rem.py, about 4 minutes) and compared with the saved run in data/provided/; the data
    points stay module A5's (A5 regenerates them from the trial file).

STEPS (each writes its log to outputs/logs/<step>.log; the release folder is written as <release>)
  1. check_model_mechanics.py   the model code against a version written from the description in
                                SI Appendix, Section 2 (no data)
  2. continuous_rem.py          the model run (regenerate mode only)
  3. make_figureS4.py           Fig. S4 -> outputs/figures/, outputs/figure_data/
  4. stats_figS4.py             the uncertainty statistics T27 -> outputs/tests.csv
  5. check_provided.py          every ledger number -> outputs/numbers.csv
The run stops at the first step that fails.

OPTIONS
  --provided-only   use the saved run even if the trial structures are present (also --skip-long)

RUN (from the repository root)
  python A2c_continuous_recognition_figS4/code/run_module.py
Seeds: model 7, 99 and 1234 (continuous_rem.py); mechanics check 20261013 and 20261014; bootstrap 20261013.
Runtime: about 3 seconds in provided mode on an Apple M2 Pro; about 4 minutes more with the model rerun.
"""
import argparse
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True   # write no __pycache__ folder into shared/python or code/ (set before the imports below)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import continuous_rem  # noqa: E402

PATHS = release_paths.module_paths(__file__)
RELEASE_ROOT = release_paths.release_root()
CODE_FOLDER = PATHS['code']


def run_step(name, arguments):
    """Runs one script of code/ with this Python, shows its output as it comes, and keeps a log in
    outputs/logs/<name>.log. Stops the module when the step fails."""
    print('\n==== %s' % name, flush=True)
    command = [sys.executable, os.path.join(CODE_FOLDER, name + '.py')] + arguments
    environment = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    start_time = time.time()
    log_lines = ['$ python ' + ' '.join([os.path.relpath(command[1], RELEASE_ROOT)] + arguments)]
    process = subprocess.Popen(command, cwd=RELEASE_ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        line = line.rstrip('\n').replace(RELEASE_ROOT, '<release>')
        print(line, flush=True)
        log_lines.append(line)
    exit_code = process.wait()
    seconds = time.time() - start_time
    log_lines.append('[%s finished with exit code %d after %.0f s]' % (name, exit_code, seconds))
    with open(os.path.join(PATHS['logs'], name + '.log'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(log_lines) + '\n')
    if exit_code != 0:
        sys.exit('run_module.py: step %s failed (exit code %d); see outputs/logs/%s.log' % (name, exit_code, name))
    print('[%s done in %.0f s]' % (name, seconds), flush=True)
    return seconds


def remove_old_model_run():
    """In provided mode, model-run files left in outputs/ by an earlier regeneration are removed, so
    that no output mixes two runs (the saved run stays in data/provided/)."""
    for file_name in ('continuous_rem.txt', 'continuous_bins.npz', 'continuous_cell_counts.npz',
                      'equality_with_paper_run.csv'):
        path = os.path.join(PATHS['outputs'], file_name)
        if os.path.exists(path):
            os.remove(path)
            print('removed outputs/%s of an earlier regeneration (provided mode uses data/provided/)' % file_name)
    old_log = os.path.join(PATHS['logs'], 'continuous_rem.log')
    if os.path.exists(old_log):
        os.remove(old_log)


def main():
    parser = argparse.ArgumentParser(description='Module A2c: continuous recognition, Fig. S4.')
    parser.add_argument('--provided-only', action='store_true',
                        help='use the saved run in data/provided/ even when the trial structures are present')
    parser.add_argument('--skip-long', action='store_true', help='the same as --provided-only')
    arguments = parser.parse_args()

    print('Module A2c: REM in continuous recognition on the THINGS trial sequences (SI Appendix, Section 2, Fig. S4)')
    templates, source = continuous_rem.find_templates()
    if arguments.provided_only or arguments.skip_long:
        mode = 'provided'
        print('mode: provided (asked for); the model numbers come from the saved run in data/provided/')
    elif templates is not None:
        mode = 'regenerate'
        print('mode: regenerate the model run; trial structures from the %s' % source)
    else:
        mode = 'provided'
        print(release_paths.PROVIDED_MESSAGE % source)
        print('mode: provided; the model numbers come from the saved run in data/provided/')
    # the data side (module A5's output) is provided in this module in both modes
    if os.path.exists(continuous_rem.KRAMER_TRIAL_FILE):
        print('THINGS data side: module A5 regenerates it from the trial file; this module reads A5\'s data/provided/ copy')
    else:
        print(release_paths.PROVIDED_MESSAGE % 'the Kramer et al. trial file for the THINGS data side (M1, expected at '
                                              'external/kramer_trials/imbysub-all-11-16-21.mat)')

    runtimes = {}
    runtimes['check_model_mechanics'] = run_step('check_model_mechanics', [])
    if mode == 'regenerate':
        runtimes['continuous_rem'] = run_step('continuous_rem', [])
        model_run = ['--model-run', 'regenerated']
    else:
        remove_old_model_run()
        model_run = ['--model-run', 'provided']
    runtimes['make_figureS4'] = run_step('make_figureS4', model_run)
    runtimes['stats_figS4'] = run_step('stats_figS4', model_run)
    runtimes['check_provided'] = run_step('check_provided', model_run)

    print('\nModule A2c finished (mode %s). Runtimes: %s; total %.0f s'
          % (mode, ', '.join('%s %.0f s' % (name, seconds) for name, seconds in runtimes.items()),
             sum(runtimes.values())))


if __name__ == '__main__':
    main()
