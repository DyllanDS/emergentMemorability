"""
run_module.py: runs module A2e (SI Appendix, Section 2, Measured Representations, Fig. S6) from start to
end.

Steps, in order
  1. code/continuous_rem_spose.m, seed 1 (MATLAB, about 1 minute): the simulation
     -> outputs/continuous_rem_spose.txt (the report), concept_subset.csv, continuous_rem_settings.csv,
        continuous_rem_items_sigma<noise>.csv, continuous_rem_designed_sigma<noise>.csv
  2. code/bootstrap_ci.py (seconds): the error bars of Fig. S6 (2,000 resamples, seed 20261010)
     -> outputs/continuous_rem_bootstrap_ci.csv and .txt
  3. code/stats_figS6.py (about 25 seconds): the uncertainty statistics T29 (10,000 resamples, seed 20261013)
     -> outputs/tests.csv
  4. code/make_figureS6.py (seconds): Fig. S6 -> outputs/figures/, outputs/figure_data/
  5. code/check_against_paper_run.py (seconds): comparison with the saved run in data/paper_run/, and
     numbers.csv -> outputs/equality_with_paper_run.csv, outputs/numbers.csv
Every step writes its log to outputs/logs/<step>.log (with the repository folder written as <release>).
The run stops at the first step that fails.

Option
  --skip-matlab  skip step 1 and use the MATLAB outputs already in outputs/ (stops if they are missing).
MATLAB is found from the environment variable MATLAB (the full path of the matlab executable), else
`matlab` on the PATH, else a standard installation (preferring R2023b, the tested version). It needs
the Statistics and Machine Learning Toolbox; no parallel pool is used. MATLAB starts with
restoredefaultpath (for that session only), so that folders added to the MATLAB path cannot replace
MATLAB's own functions.

Inputs besides this module: shared/things/ (the SPoSE embedding and the concept names), shared/matlab/,
shared/python/, shared/ledger/, and the saved REM runs of two other modules:
../A2d_list_context_figS5/data/paper_run/list_designed.txt (the white bar of Fig. S6) and
../01_rem_simulation_fig1/data/paper_run/fig1_paper_run.mat (the run of Fig. 1, for its rho of -0.55).

Run (from the repository root; about 1.5 minutes on an Apple M2 Pro):
    python A2e_measured_representations_figS6/code/run_module.py
Seeds: MATLAB rng(1, 'twister'); numpy default_rng(20261010) (the error bars of Fig. S6) and
default_rng(20261013) (the 10,000-resample intervals of tests.csv).
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

NEEDED_INPUTS = [
    release_paths.shared_path('things', 'spose_embedding_49d_sorted.txt'),
    release_paths.shared_path('things', 'unique_id.txt'),
    os.path.join(paths['paper_run'], 'continuous_rem_spose.txt'),
    os.path.join(release_paths.find_module('A2d'), 'data', 'paper_run', 'list_designed.txt'),
    os.path.join(release_paths.find_module('01'), 'data', 'paper_run', 'fig1_paper_run.mat'),
]
MATLAB_OUTPUTS = (['continuous_rem_spose.txt', 'concept_subset.csv', 'continuous_rem_settings.csv']
                  + ['continuous_rem_items_sigma%s.csv' % noise for noise in ('0.02', '0.03', '0.05', '0.08')]
                  + ['continuous_rem_designed_sigma%s.csv' % noise for noise in ('0.02', '0.03', '0.05', '0.08')])


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
    with open(os.path.join(paths['logs'], name + '.log'), 'w') as handle:
        handle.write('\n'.join(log_lines) + '\n')
    if exit_code != 0:
        sys.exit('run_module.py: step %s failed (exit code %d); see outputs/logs/%s.log' % (name, exit_code, name))
    print('[%s done in %.0f s]' % (name, seconds), flush=True)
    return seconds


def matlab_command(matlab, calls):
    """matlab -batch "restoredefaultpath; cd('<module>/code'); <calls>" (paths as found at run time)."""
    code_folder = CODE_FOLDER.replace("'", "''")
    return [matlab, '-batch', "restoredefaultpath; cd('%s'); %s" % (code_folder, calls)]


def main():
    parser = argparse.ArgumentParser(description='Module A2e: REM in the measured SPoSE space and Fig. S6.')
    parser.add_argument('--skip-matlab', action='store_true', help='reuse the MATLAB outputs in outputs/')
    arguments = parser.parse_args()
    python = sys.executable
    runtimes = {}

    print('Module A2e: REM in the measured SPoSE space (SI Appendix, Section 2, Fig. S6); mode: rerun')
    missing = [needed for needed in NEEDED_INPUTS if not os.path.exists(needed)]
    if missing:
        sys.exit('run_module.py: inputs not found:\n  %s\nThis module needs shared/things/ and the data/paper_run '
                 'folders of modules A2d (the white bar of Fig. S6) and 01 (the run of Fig. 1).'
                 % '\n  '.join(os.path.relpath(needed, RELEASE_ROOT) for needed in missing))

    if arguments.skip_matlab:
        for needed in MATLAB_OUTPUTS:
            if not os.path.exists(os.path.join(paths['outputs'], needed)):
                sys.exit('run_module.py --skip-matlab: outputs/%s is missing; run without --skip-matlab' % needed)
        print('step 1 skipped (--skip-matlab): using the MATLAB outputs already in outputs/')
    else:
        matlab = find_matlab()
        runtimes['continuous_rem_spose_seed1'] = run_step('continuous_rem_spose_seed1',
                                                          matlab_command(matlab, 'continuous_rem_spose(1)'))

    runtimes['bootstrap_ci'] = run_step('bootstrap_ci', [python, os.path.join(CODE_FOLDER, 'bootstrap_ci.py')])
    runtimes['stats_figS6'] = run_step('stats_figS6', [python, os.path.join(CODE_FOLDER, 'stats_figS6.py')])
    runtimes['make_figureS6'] = run_step('make_figureS6', [python, os.path.join(CODE_FOLDER, 'make_figureS6.py')])
    runtimes['check_against_paper_run'] = run_step('check_against_paper_run',
                                                   [python, os.path.join(CODE_FOLDER, 'check_against_paper_run.py')])

    print('\nModule A2e finished. Runtimes: %s; total %.0f s'
          % (', '.join('%s %.0f s' % (name, seconds) for name, seconds in runtimes.items()), sum(runtimes.values())))


if __name__ == '__main__':
    main()
