"""
run_module.py: runs module A2a (SI Appendix, Section 2, the REM parameter sweeps, Fig. S2) from start to
end.

Steps, in order
  1. code/rem_parameter_sweep.m (MATLAB, about 2 to 3 minutes): the 32-cell REM parameter sweep
     -> outputs/rem_parameter_sweep.csv, rem_parameter_sweep.txt, rem_parameter_sweep_items.csv
  2. code/accuracy_link.py (seconds): r between d′ and rho over the cells -> outputs/accuracy_link.json
  3. code/feature_count_sweep.py (about 3 minutes, up to about 12 on a busy machine): the REM of Fig. 1
     at 5 to 120 features, 3 seeds -> outputs/sweep_rem.csv
  4. code/stats_figS2.py (about 3 to 4 minutes): the uncertainty statistics T20, T24, T25 -> outputs/tests.csv
  5. code/make_figureS2.py (seconds): Fig. S2 -> outputs/figures/, outputs/figure_data/
  6. code/check_against_paper_run.py (seconds): compares this run with the saved runs in data/paper_run/
     -> outputs/equality_with_paper_run.csv and outputs/numbers.csv
Every step writes its log to outputs/logs/<step>.log (the location of the repository folder is written as
<release> there, and the Python or MATLAB program by its name only). The run stops at the first step
that fails.

Options
  --skip-long    skip step 3 (the feature-count sweep). The feature-count numbers then
                 come from outputs/sweep_rem.csv if an earlier full run left it there, else from the
                 saved run, data/paper_run/sweep_rem.csv; numbers.csv says which (mode rerun or
                 saved_run).
  --skip-matlab  skip step 1 and use the sweep outputs an earlier run left in outputs/ (stops if they
                 are missing).
MATLAB is found from the environment variable MATLAB (the full path of the matlab executable), else
`matlab` on the PATH, else a standard installation (preferring R2023b, the tested version). It needs
the Statistics and Machine Learning Toolbox; no parallel pool is used. MATLAB starts with
restoredefaultpath (for that session only), so that folders added to the MATLAB path cannot replace
MATLAB's own functions.

Run (from the repository root):
    python A2a_rem_parameter_sweep_figS2/code/run_module.py
    python A2a_rem_parameter_sweep_figS2/code/run_module.py --skip-long
Seeds: MATLAB rng(1, 'twister') (the sweep); numpy 12345, 777, 20260731 (feature-count sweep); 20261013
(the statistics). Full runtime about 9 minutes on an Apple M2 Pro.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

# Do not write __pycache__ folders: switched off before the first import from shared/python (the steps
# below run with PYTHONDONTWRITEBYTECODE=1 for the same reason).
sys.dont_write_bytecode = True
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
                 'executable, or run with --skip-matlab to reuse the sweep outputs of an earlier run.')
    for candidate in installed:
        if 'R2023b' in candidate:
            return candidate
    return installed[-1]


def run_step(name, command):
    """Runs one step, shows its output as it comes, and keeps a log in outputs/logs/<name>.log."""
    print('\n==== %s' % name, flush=True)
    environment = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    start_time = time.time()
    # The log names the program (python3, matlab) without the folder it is installed in on this computer.
    shown_command = [os.path.basename(command[0])] + list(command[1:])
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
    with open(os.path.join(paths['logs'], name + '.log'), 'w') as handle:
        handle.write('\n'.join(log_lines) + '\n')
    if exit_code != 0:
        sys.exit('run_module.py: step %s failed (exit code %d); see outputs/logs/%s.log' % (name, exit_code, name))
    print('[%s done in %.0f s]' % (name, seconds), flush=True)
    return seconds


def main():
    parser = argparse.ArgumentParser(description='Module A2a: REM parameter sweeps and Fig. S2.')
    parser.add_argument('--skip-long', action='store_true', help='skip the feature-count sweep (step 3)')
    parser.add_argument('--skip-matlab', action='store_true', help='reuse the MATLAB sweep outputs in outputs/')
    arguments = parser.parse_args()
    python = sys.executable
    runtimes = {}

    print('Module A2a: REM parameter sweeps (SI Appendix, Section 2, Fig. S2); mode: rerun')
    if arguments.skip_matlab:
        for needed in ('rem_parameter_sweep.csv', 'rem_parameter_sweep.txt', 'rem_parameter_sweep_items.csv'):
            if not os.path.exists(os.path.join(paths['outputs'], needed)):
                sys.exit('run_module.py --skip-matlab: outputs/%s is missing; run without --skip-matlab' % needed)
        print('step 1 skipped (--skip-matlab): using the sweep outputs already in outputs/')
    else:
        matlab = find_matlab()
        matlab_code_folder = CODE_FOLDER.replace("'", "''")
        # restoredefaultpath (for this MATLAB session only) removes folders added to the MATLAB path, so that
        # no other toolbox can replace a MATLAB function the sweep uses (FieldTrip's external/stats/tcdf.m, for
        # example, returns 0 for p values below about 1e-16).
        runtimes['rem_parameter_sweep'] = run_step(
            'rem_parameter_sweep',
            [matlab, '-batch', "restoredefaultpath; cd('%s'); rem_parameter_sweep(true)" % matlab_code_folder])

    runtimes['accuracy_link'] = run_step('accuracy_link', [python, os.path.join(CODE_FOLDER, 'accuracy_link.py')])

    if arguments.skip_long:
        print('\nstep 3 skipped (--skip-long): the feature-count numbers come from outputs/sweep_rem.csv if an '
              'earlier full run left it, else from data/paper_run/sweep_rem.csv (the saved run)')
    else:
        runtimes['feature_count_sweep'] = run_step('feature_count_sweep',
                                                   [python, os.path.join(CODE_FOLDER, 'feature_count_sweep.py')])

    runtimes['stats_figS2'] = run_step('stats_figS2', [python, os.path.join(CODE_FOLDER, 'stats_figS2.py')])
    runtimes['make_figureS2'] = run_step('make_figureS2', [python, os.path.join(CODE_FOLDER, 'make_figureS2.py')])
    runtimes['check_against_paper_run'] = run_step('check_against_paper_run',
                                                   [python, os.path.join(CODE_FOLDER, 'check_against_paper_run.py')])

    print('\nModule A2a finished. Runtimes: %s; total %.0f s'
          % (', '.join('%s %.0f s' % (name, seconds) for name, seconds in runtimes.items()), sum(runtimes.values())))


if __name__ == '__main__':
    main()
