"""
run_module.py: runs module A2b (SI Appendix, Section 2, Category Structure and Decision Criteria; SI Appendix,
Fig. S3) from start to end.

Steps, in order
  1. code/bylinskii_robustness.m (MATLAB, about 3.5 minutes): the 18 REM simulations of the Bylinskii et al.
     (2015) context manipulation, one factor varied at a time
     -> outputs/bylinskii_robustness.csv, bylinskii_robustness.txt, bylinskii_robustness_items.csv,
        outputs/logs/bylinskii_progress.log
  2. code/stats_figS3.py (about 3 minutes): the statistics of test T26 -> outputs/tests.csv
  3. code/make_figureS3.py (seconds): Fig. S3 -> outputs/figures/, outputs/figure_data/
  4. code/check_against_paper_run.py (seconds): comparison with the saved run in data/paper_run/, the output
     the paper reports -> outputs/equality_with_paper_run.csv and outputs/numbers.csv
Every step writes its log to outputs/logs/<step>.log (the repository folder is written as <release> there).
The run stops at the first step that fails.

Options
  --skip-matlab  skip step 1 and use the grid outputs already in outputs/ (stops if they are missing)
  --skip-long    accepted because run_all.py --skip-long passes it to this module; this module has no long
                 step, so it changes nothing
MATLAB is found from the environment variable MATLAB (the full path of the matlab executable), else
`matlab` on the PATH, else a standard installation (preferring R2023b, the tested version). It needs
the Statistics and Machine Learning Toolbox; no parallel pool is used. MATLAB starts with
restoredefaultpath (for that session only), so that folders added to the MATLAB path cannot replace
MATLAB's own functions.

Without MATLAB, the figure and numbers.csv can be made from the saved run in data/paper_run/:
    python A2b_category_structure_criteria_figS3/code/make_figureS3.py --paper-run
    python A2b_category_structure_criteria_figS3/code/check_against_paper_run.py --paper-run
(the statistics of step 2 need the per-target file that only the MATLAB run writes).

Run (from the repository root):
    python A2b_category_structure_criteria_figS3/code/run_module.py
Seeds: MATLAB rng(11, 'twister') (the grid); numpy 20261013 (the statistics). Full runtime about 6.5 minutes on
an Apple M2 Pro (MATLAB about 3.5 minutes, statistics about 3 minutes, the other steps seconds).
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True   # write no __pycache__ folder into shared/python (set before the import below)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402

paths = release_paths.module_paths(__file__)
RELEASE_ROOT = release_paths.release_root()
CODE_FOLDER = paths['code']
GRID_OUTPUTS = ('bylinskii_robustness.csv', 'bylinskii_robustness.txt', 'bylinskii_robustness_items.csv')


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
                 'executable, or run with --skip-matlab to reuse the grid outputs in outputs/.')
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
    with open(os.path.join(paths['logs'], name + '.log'), 'w') as handle:
        handle.write('\n'.join(log_lines) + '\n')
    if exit_code != 0:
        sys.exit('run_module.py: step %s failed (exit code %d); see outputs/logs/%s.log' % (name, exit_code, name))
    print('[%s done in %.0f s]' % (name, seconds), flush=True)
    return seconds


def main():
    parser = argparse.ArgumentParser(description='Module A2b: category structure and decision criteria '
                                                 '(SI Appendix, Fig. S3).')
    parser.add_argument('--skip-matlab', action='store_true', help='reuse the MATLAB grid outputs in outputs/')
    parser.add_argument('--skip-long', action='store_true',
                        help='accepted for run_all.py; this module has no long step, so nothing is skipped')
    arguments = parser.parse_args()
    python = sys.executable
    runtimes = {}

    print('Module A2b: category structure and decision criteria (SI Appendix, Section 2, Fig. S3); mode: rerun')
    if arguments.skip_long:
        print('--skip-long: this module has no long step; everything runs')
    if arguments.skip_matlab:
        for needed in GRID_OUTPUTS:
            if not os.path.exists(os.path.join(paths['outputs'], needed)):
                sys.exit('run_module.py --skip-matlab: outputs/%s is missing; run without --skip-matlab' % needed)
        print('step 1 skipped (--skip-matlab): using the grid outputs already in outputs/')
    else:
        matlab = find_matlab()
        matlab_code_folder = CODE_FOLDER.replace("'", "''")
        # restoredefaultpath (for this MATLAB session only) removes folders added to the MATLAB path, so that no
        # other toolbox can replace a MATLAB function the grid uses (FieldTrip's external/stats/tcdf.m, for
        # example, returns 0 for p values below about 1e-16).
        runtimes['bylinskii_robustness'] = run_step(
            'bylinskii_robustness',
            [matlab, '-batch', "restoredefaultpath; cd('%s'); bylinskii_robustness(true)" % matlab_code_folder])

    runtimes['stats_figS3'] = run_step('stats_figS3', [python, os.path.join(CODE_FOLDER, 'stats_figS3.py')])
    runtimes['make_figureS3'] = run_step('make_figureS3', [python, os.path.join(CODE_FOLDER, 'make_figureS3.py')])
    runtimes['check_against_paper_run'] = run_step('check_against_paper_run',
                                                   [python, os.path.join(CODE_FOLDER, 'check_against_paper_run.py')])

    print('\nModule A2b finished. Runtimes: %s; total %.0f s'
          % (', '.join('%s %.0f s' % (name, seconds) for name, seconds in runtimes.items()), sum(runtimes.values())))


if __name__ == '__main__':
    main()
