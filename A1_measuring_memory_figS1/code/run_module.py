"""
run_module.py: runs module A1 (SI Appendix, Section 1, Measuring Memory, Fig. S1) from start to end.

Steps, in order
  1. code/criterion_dprime_da.m (MATLAB, about 25 seconds): the analysis of the three image-set
     experiments (criterion c, corrected recognition against d' and d_a) and the REM check of the
     Fig. 1 run -> outputs/paperStats_criterionDpDa.mat
  2. code/criterion_dprime_da_checks.py (about 10 seconds): the same numbers in Python, the measured
     z-ROC slope and d_a at that slope, the criterion table, the checks against the MATLAB run and the
     saved run -> outputs/concept_scores.csv, criterion_table.csv, rem_item_scores.csv,
     criterion_dprime_da.txt, numbers.csv
  3. code/make_figureS1.py (seconds): Fig. S1 -> outputs/figures/, outputs/figure_data/
  4. code/stats_figS1.py (about 25 seconds): the uncertainty statistics T21 and T22 -> outputs/tests.csv,
     outputs/stats_figS1.txt, and the rows of outputs/numbers.csv that print a 95% interval
Every step also writes its printed output to outputs/logs/<step>.log, with the repository folder
written as <release>. The run stops at the first step that fails.

Options
  --skip-matlab      skip step 1; step 2 then takes every number from its own Python recomputation
                     (identical to the MATLAB run to about 1e-15; numbers.csv says which was used). A
                     MATLAB output already in outputs/ is still compared with it.
  --resamples N      bootstrap resamples of step 4 (default 10,000; a smaller N gives a quick check, and
                     the number used is written into tests.csv)
MATLAB is found from the environment variable MATLAB (the full path of the matlab executable), else
`matlab` on the PATH, else a standard installation (preferring R2023b, the version the code was tested
with). It needs the Statistics and Machine Learning Toolbox; no parallel pool is used. MATLAB starts
with restoredefaultpath (for that session only), so folders added to the MATLAB path cannot replace
MATLAB's own functions.

Run (from the repository root):
    python A1_measuring_memory_figS1/code/run_module.py
    python A1_measuring_memory_figS1/code/run_module.py --skip-matlab
Seeds: none in steps 1 to 3 (no random numbers); 20261013 in step 4. Full runtime about 1 minute on
an Apple M2 Pro (MATLAB step about 25 seconds, statistics about 25 seconds).
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402

paths = release_paths.module_paths(__file__)
RELEASE_ROOT = release_paths.release_root()
CODE_FOLDER = paths['code']


def find_matlab():
    """The MATLAB executable: $MATLAB, else matlab on the PATH, else a standard installation."""
    from_environment = os.environ.get('MATLAB')
    if from_environment:
        program = from_environment
        if os.path.isdir(program):
            # the MATLAB root folder (MATLAB's own meaning of this variable): the executable is in its bin folder
            program = os.path.join(program, 'bin', 'matlab.exe' if os.name == 'nt' else 'matlab')
        if not (os.path.isfile(program) and os.access(program, os.X_OK)):
            sys.exit('run_module.py: the MATLAB environment variable points to %s, which is neither the matlab '
                     'executable nor a MATLAB folder with bin/matlab' % from_environment)
        return program
    on_path = shutil.which('matlab')
    if on_path:
        return on_path
    installed = []
    for pattern in ('/Applications/MATLAB_R20*.app/bin/matlab', '/usr/local/MATLAB/R20*/bin/matlab',
                    'C:/Program Files/MATLAB/R20*/bin/matlab.exe'):
        installed.extend(sorted(glob.glob(pattern)))
    if not installed:
        sys.exit('run_module.py: MATLAB was not found. Set the environment variable MATLAB to the matlab '
                 'executable, or run with --skip-matlab (the Python steps then compute every number).')
    for candidate in installed:
        if 'R2023b' in candidate:
            return candidate
    return installed[-1]


def run_step(name, command):
    """Runs one step, shows its output as it comes, and keeps a log in outputs/logs/<name>.log."""
    print('\n==== %s' % name, flush=True)
    environment = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        environment.setdefault(variable, '2')           # two threads for numpy unless these variables are set
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
    with open(os.path.join(paths['logs'], name + '.log'), 'w') as handle:
        handle.write('\n'.join(log_lines) + '\n')
    if exit_code != 0:
        sys.exit('run_module.py: step %s failed (exit code %d); see outputs/logs/%s.log' % (name, exit_code, name))
    print('[%s done in %.0f s]' % (name, seconds), flush=True)
    return seconds


def main():
    parser = argparse.ArgumentParser(description='Module A1: criterion, d\' and d_a (SI Appendix, Section 1, Fig. S1).')
    parser.add_argument('--skip-matlab', action='store_true',
                        help='skip the MATLAB step; the Python steps compute every number')
    parser.add_argument('--resamples', type=int, default=10000,
                        help='bootstrap resamples of the statistics step (default 10,000)')
    arguments = parser.parse_args()
    python = sys.executable
    runtimes = {}

    print('Module A1: Measuring memory (SI Appendix, Section 1, Fig. S1); mode: rerun')
    if arguments.skip_matlab:
        print('step 1 skipped (--skip-matlab): every number comes from the Python recomputation')
    else:
        matlab = find_matlab()
        matlab_code_folder = CODE_FOLDER.replace("'", "''")
        runtimes['criterion_dprime_da'] = run_step(
            'criterion_dprime_da',
            [matlab, '-batch', "restoredefaultpath; cd('%s'); criterion_dprime_da" % matlab_code_folder])

    checks_command = [python, os.path.join(CODE_FOLDER, 'criterion_dprime_da_checks.py')]
    if arguments.skip_matlab:
        checks_command.append('--without-matlab')
    runtimes['criterion_dprime_da_checks'] = run_step('criterion_dprime_da_checks', checks_command)
    runtimes['make_figureS1'] = run_step('make_figureS1', [python, os.path.join(CODE_FOLDER, 'make_figureS1.py')])
    runtimes['stats_figS1'] = run_step('stats_figS1', [python, os.path.join(CODE_FOLDER, 'stats_figS1.py'),
                                                       '--resamples', str(arguments.resamples)])

    print('\nModule A1 finished. Runtimes: %s; total %.0f s'
          % (', '.join('%s %.0f s' % (name, seconds) for name, seconds in runtimes.items()), sum(runtimes.values())))


if __name__ == '__main__':
    main()
