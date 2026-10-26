"""
run_module.py: runs module A3d (SI Appendix, Section 3, "The Image Set Results Do Not Depend on the
Exclusions", Table S1) from start to end.

Steps, in order (all Python)
  1. code/exclusion_regimes.py (about 2 minutes): the image-set results on all 602 sessions run,
     with the exclusions of the Methods (450 participants kept) and with a stricter
     corrected-recognition floor; the Fig. 7d statistic without and with the exclusions (seed 7),
     and the comparison of the six Fig. 6 correlations and the participant counts with the saved
     run of module 06
     -> outputs/exclusion_regimes.txt, participant_flags.csv, concept_level_by_regime.csv,
        feature_level_by_regime.csv, fig7d_reduction_by_regime.csv, reproduction_check.csv
  2. code/stats_tableS1.py (about 30 seconds): uncertainty statistics T35 and T36 (seed 20261013;
     10,000 bootstrap resamples) and T37 (from step 1) -> outputs/tests.csv
  3. code/make_tableS1.py (seconds): SI Appendix, Table S1 -> outputs/tableS1.csv
  4. code/write_numbers.py (seconds): every ledger item of the module -> outputs/numbers.csv
Every step writes its log to outputs/logs/<step>.log (the release folder's own location is written
as <release> there, the Python interpreter as python). The run stops at the first step that fails.

Option (a quick check of the code only; the default gives the values in outputs/)
  --resamples N      bootstrap resamples of step 2 (default 10,000)

Run (from the repository root):
    python A3d_exclusions_tableS1/code/run_module.py
Seeds: 7 (step 1); 20261013 (step 2). Runtime: about 2.5 minutes in all.
"""
import argparse
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402

paths = release_paths.module_paths(__file__, make_output_folders=False)
RELEASE_ROOT = release_paths.release_root()
CODE_FOLDER = paths['code']


def run_step(name, command):
    """Runs one step, shows its output as it comes, and keeps a log in outputs/logs/<name>.log."""
    print('\n==== %s' % name, flush=True)
    environment = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    start_time = time.time()
    # the log shows the command as it is typed: "python" for the interpreter, <release> for the release folder
    shown_command = ['python'] + command[1:]
    log_lines = ['$ ' + ' '.join(shown_command).replace(RELEASE_ROOT, '<release>')]
    process = subprocess.Popen(command, cwd=RELEASE_ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        line = line.rstrip('\n').replace(RELEASE_ROOT, '<release>')
        print(line, flush=True)
        log_lines.append(line)
    exit_code = process.wait()
    seconds = time.time() - start_time
    log_lines.append('[%s finished with exit code %d after %.0f s]' % (name, exit_code, seconds))
    with open(os.path.join(paths['logs'], name + '.log'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(log_lines) + '\n')
    if exit_code != 0:
        sys.exit('run_module.py: step %s failed (exit code %d); see outputs/logs/%s.log' % (name, exit_code, name))
    print('[%s done in %.0f s]' % (name, seconds), flush=True)
    return seconds


def main():
    parser = argparse.ArgumentParser(description='Module A3d: the image-set results with and without the '
                                                 'exclusions (SI Appendix, Table S1).')
    parser.add_argument('--resamples', type=int, default=10000,
                        help='bootstrap resamples of the statistics step (default 10,000)')
    arguments = parser.parse_args()
    python = sys.executable
    runtimes = {}
    os.makedirs(paths['logs'], exist_ok=True)

    print('Module A3d: the image-set results with and without the exclusions (SI Appendix, Section 3, '
          'Table S1); mode: rerun')
    runtimes['exclusion_regimes'] = run_step('exclusion_regimes',
                                             [python, os.path.join(CODE_FOLDER, 'exclusion_regimes.py')])
    runtimes['stats_tableS1'] = run_step('stats_tableS1',
                                         [python, os.path.join(CODE_FOLDER, 'stats_tableS1.py'),
                                          '--resamples', str(arguments.resamples)])
    for step in ('make_tableS1', 'write_numbers'):
        runtimes[step] = run_step(step, [python, os.path.join(CODE_FOLDER, step + '.py')])

    print('\nModule A3d finished. Runtimes: %s; total %.0f s'
          % (', '.join('%s %.0f s' % (name, seconds) for name, seconds in runtimes.items()),
             sum(runtimes.values())))


if __name__ == '__main__':
    main()
