"""
run_module.py: runs module A3b (SI Appendix, Fig. S9; Sections 1 and 3) from start to end.

Steps, in order (all Python; each runs as its own process from the repository root)
  1. code/per_image_rates.py (seconds): Fig. S9c, the per-image rates of the original THINGS data.
     Without the Kramer et al. trial records (input M1 of EXTERNAL_INPUTS.md) it prints the
     provided-output message and uses the saved results in data/provided/per_image_rates.txt. With the
     trial records in external/kramer_trials/ it regenerates the report into outputs/regenerated/ and
     compares it with the saved results.
  2. code/make_figureS9.py (seconds): panels a and b from the data of module 03 (the item bootstrap of
     the error bars, seed 0), panel c from step 1, the halves comparison; compares every number with
     the saved run in data/paper_run/ -> outputs/figures/FigS9_hits_false_alarms.png,
     outputs/figure_data/, outputs/level_table.csv, level_correlations.csv, halves.csv
  3. code/stats_figS9.py (about 15 seconds): statistics T33 and T34 (seed 20261013, 10,000
     resamples) -> outputs/tests.csv
  4. code/write_numbers.py (seconds): every ledger item of the module -> outputs/numbers.csv
Every step's printout is kept in outputs/logs/<step>.log (the repository folder written as <release>,
the home folder as ~, the Python interpreter as python). The run stops at the first step that fails.

Option (a quick check of the code only; the default is 10,000)
  --resamples N   bootstrap resamples of step 3 (default 10,000)

Run (from the repository root):
    python A3b_hits_and_false_alarms_figS9/code/run_module.py
Mode: panels a and b rerun; panel c and the THINGS side of the halves comparison from the saved results
in data/provided/ (they switch to regeneration when input M1 or M2 of EXTERNAL_INPUTS.md is in
external/kramer_trials/). Seeds: 0 (the error bars, step 2), 20261013 (the statistics, step 3).
Runtime: about 20 seconds.
"""
import argparse
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402

PATHS = release_paths.module_paths(__file__)
RELEASE_ROOT = release_paths.release_root()
CODE_FOLDER = PATHS['code']


def sanitise(text):
    """The repository folder written as <release> and the home folder as ~ (no machine paths in the logs)."""
    for root in sorted({os.path.realpath(RELEASE_ROOT), os.path.abspath(RELEASE_ROOT)}, key=len, reverse=True):
        text = text.replace(root, '<release>')
    home = os.path.expanduser('~')
    if home and home != '~':
        text = text.replace(home, '~')
    return text


def run_step(name, command):
    """Runs one step, shows its output as it comes, and keeps it in outputs/logs/<name>.log."""
    print('\n==== %s' % name, flush=True)
    environment = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    started = time.time()
    shown_command = ['python'] + [sanitise(part) for part in command[1:]]
    log_lines = ['$ ' + ' '.join(shown_command)]
    process = subprocess.Popen(command, cwd=RELEASE_ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        line = sanitise(line.rstrip('\n'))
        print(line, flush=True)
        log_lines.append(line)
    exit_code = process.wait()
    seconds = time.time() - started
    log_lines.append('[%s finished with exit code %d after %.0f s]' % (name, exit_code, seconds))
    with open(os.path.join(PATHS['logs'], name + '.log'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(log_lines) + '\n')
    if exit_code != 0:
        sys.exit('run_module.py: step %s failed (exit code %d); see outputs/logs/%s.log' % (name, exit_code, name))
    print('[%s done in %.0f s]' % (name, seconds), flush=True)
    return seconds


def main():
    parser = argparse.ArgumentParser(description='Module A3b: hits and false alarms (Fig. S9).')
    parser.add_argument('--resamples', type=int, default=10000,
                        help='bootstrap resamples of the statistics step (default 10,000)')
    arguments = parser.parse_args()
    python = sys.executable
    runtimes = {}
    print('Module A3b: hits and false alarms (SI Appendix, Fig. S9); mode: panels a and b rerun, '
          'panel c and the THINGS side of the halves comparison provided unless the Kramer et al. data are in '
          'external/kramer_trials/')
    runtimes['per_image_rates'] = run_step('per_image_rates', [python, os.path.join(CODE_FOLDER, 'per_image_rates.py')])
    runtimes['make_figureS9'] = run_step('make_figureS9', [python, os.path.join(CODE_FOLDER, 'make_figureS9.py')])
    runtimes['stats_figS9'] = run_step('stats_figS9', [python, os.path.join(CODE_FOLDER, 'stats_figS9.py'),
                                                       '--resamples', str(arguments.resamples)])
    runtimes['write_numbers'] = run_step('write_numbers', [python, os.path.join(CODE_FOLDER, 'write_numbers.py')])
    print('\nModule A3b finished. Runtimes: %s; total %.0f s'
          % (', '.join('%s %.0f s' % (name, seconds) for name, seconds in runtimes.items()), sum(runtimes.values())))


if __name__ == '__main__':
    main()
