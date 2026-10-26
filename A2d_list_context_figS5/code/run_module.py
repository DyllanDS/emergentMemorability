"""
run_module.py: runs module A2d (SI Appendix, Section 2, List Context: the same item on crowded and
uncrowded lists in REM, SI Appendix, Fig. S5) from start to end.

Steps, in order
  1. code/list_designed.m (MATLAB, about 1 minute): designed lists, 300 critical items each studied
     on lists with 0, 2, 4, 8 or 16 of its close neighbors -> outputs/list_designed.txt,
     outputs/list_designed_cell2.mat
  2. code/list_binned.m (MATLAB, about 35 seconds): 600 random lists, test items binned by their number
     of close neighbors on the list -> outputs/list_binned.txt, outputs/list_binned_cell2.mat
  3. code/check_against_paper_run.py (seconds): the two printed tables against the saved run
     (data/paper_run/), line by line -> outputs/equality_with_paper_run.csv; stops if a line of
     CELL 2 (the parameter values of Fig. 1) differs
  4. code/list_bootstrap.py (seconds): the changes and their 95% bootstrap intervals, the error bars
     of Fig. S5 (2,000 resamples, seed 20261010) -> outputs/list_bootstrap.txt,
     outputs/list_bootstrap.json
  5. code/stats_figS5.py (about 10 seconds): uncertainty statistics T28, with the intervals from
     2,000 resamples (list_bootstrap.py) and from 10,000 resamples (seed 20261013) -> outputs/tests.csv
  6. code/make_figureS5.py (seconds): Fig. S5 -> outputs/figures/FigS5_list_context.png,
     outputs/figure_data/FigS5a.csv, FigS5b.csv
  7. code/write_numbers.py (seconds): every ledger item of the module -> outputs/numbers.csv
Every step writes its log to outputs/logs/<step>.log (the repository's own location is written as
<release> there). The run stops at the first step that fails.

Options
  --skip-matlab  skip steps 1 and 2 and use the MATLAB outputs already in outputs/ (they ship with
                 the repository); stops if one of them is missing
MATLAB is found from the environment variable MATLAB (the full path of the matlab executable), else
`matlab` on the PATH, else a standard installation (preferring R2023b, the tested version). It
needs the Statistics and Machine Learning Toolbox; no parallel pool is used.
MATLAB starts with restoredefaultpath (for that session only), so that folders added to the MATLAB
path cannot replace MATLAB's own functions (FieldTrip's tcdf.m, for example, would print the p values
of the within-item correlations as 0).

Run (from the repository root):
    python A2d_list_context_figS5/code/run_module.py
    python A2d_list_context_figS5/code/run_module.py --skip-matlab
Seeds: MATLAB rng(1, 'twister') in each simulation (the Mersenne Twister); numpy 20261010 (2,000
resamples, the error bars of Fig. S5) and 20261013 (10,000 resamples). Runtime on an Apple M2 Pro:
about 2 minutes.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402

paths = release_paths.module_paths(__file__)
RELEASE_ROOT = release_paths.release_root()
CODE_FOLDER = paths['code']
MATLAB_OUTPUTS = ['list_designed.txt', 'list_designed_cell2.mat', 'list_binned.txt', 'list_binned_cell2.mat']


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
                 'executable, or run with --skip-matlab to use the MATLAB outputs already in outputs/.')
    for candidate in installed:
        if 'R2023b' in candidate:
            return candidate
    return installed[-1]


def run_step(name, command):
    """Runs one step, shows its output as it comes, and keeps a log in outputs/logs/<name>.log."""
    print('\n==== %s' % name, flush=True)
    environment = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
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
    with open(os.path.join(paths['logs'], name + '.log'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(log_lines) + '\n')
    if exit_code != 0:
        sys.exit('run_module.py: step %s failed (exit code %d); see outputs/logs/%s.log' % (name, exit_code, name))
    print('[%s done in %.0f s]' % (name, seconds), flush=True)
    return seconds


def main():
    parser = argparse.ArgumentParser(description='Module A2d: list context in REM, SI Appendix, Fig. S5.')
    parser.add_argument('--skip-matlab', action='store_true',
                        help='use the MATLAB outputs already in outputs/')
    arguments = parser.parse_args()
    python = sys.executable
    runtimes = {}

    print('Module A2d: the same item on crowded and uncrowded lists (SI Appendix, Section 2, Fig. S5)')
    if arguments.skip_matlab:
        for needed in MATLAB_OUTPUTS:
            if not os.path.exists(os.path.join(paths['outputs'], needed)):
                sys.exit('run_module.py --skip-matlab: outputs/%s is missing; run without --skip-matlab' % needed)
        print('steps 1 and 2 skipped (--skip-matlab): using the MATLAB outputs already in outputs/')
    else:
        matlab = find_matlab()
        matlab_code_folder = CODE_FOLDER.replace("'", "''")
        # steps 1 and 2: the two simulations (rng(1, 'twister'))
        for script in ('list_designed', 'list_binned'):
            # restoredefaultpath (for this MATLAB session only) removes folders added to the MATLAB path,
            # so that no other toolbox can replace a MATLAB function the simulation uses.
            runtimes[script] = run_step(script, [matlab, '-batch', "restoredefaultpath; cd('%s'); %s"
                                                 % (matlab_code_folder, script)])

    for step in ('check_against_paper_run', 'list_bootstrap', 'stats_figS5', 'make_figureS5', 'write_numbers'):
        runtimes[step] = run_step(step, [python, os.path.join(CODE_FOLDER, step + '.py')])

    print('\nModule A2d finished. Runtimes: %s; total %.0f s'
          % (', '.join('%s %.0f s' % (name, seconds) for name, seconds in runtimes.items()),
             sum(runtimes.values())))


if __name__ == '__main__':
    main()
