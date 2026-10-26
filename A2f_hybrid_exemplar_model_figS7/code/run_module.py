"""
run_module.py: runs module A2f (SI Appendix, Section 2, A Hybrid-Similarity Exemplar Model: the
hybrid-similarity exemplar model and the generalized context model, Fig. S7) from start to end.

Steps, each one a separate Python process with its log in outputs/logs/<step>.log:
  1. code/01_core.py                    the basic simulation and the designed lists for five model
                                        variants, seed 1
  2. code/02_sweep.py                   the parameter sweep (27 settings) for four definitions of
                                        rarity (2 worker processes)
  3. code/03_bylinskii.py               the Bylinskii context simulation, 83 settings
                                        (2 worker processes)                                [long]
  4. code/bylinskii_reference_cells.py  the Bylinskii calibration and the three settings whose values
                                        the paper reports, run exactly as in step 3; per-target data
  5. code/check_against_paper_run.py    every number of this run against the saved run in
                                        data/paper_run/
  6. code/stats_figS7.py                outputs/numbers.csv and outputs/tests.csv
  7. code/make_figureS7.py              Fig. S7 and its panel data
On an Apple M2 Pro (16 GB) the whole run takes about 25 minutes and --skip-long about 8 minutes
(the module README lists the steps).

Options
  --skip-long      skip step 3. The count of Bylinskii settings with a positive correlation
                   is then read from an earlier full run in outputs/bylinskii.csv if there is one
                   (mode rerun in numbers.csv), else from the saved run in
                   data/paper_run/bylinskii.csv (mode rerun_and_saved_run). The three Bylinskii
                   settings whose values the paper reports still run exactly in step 4.
  --provided-only  run no simulation (steps 5 to 7 only), from what outputs/ holds, otherwise from
                   the saved run in data/paper_run/.
Run from the repository root:
  python A2f_hybrid_exemplar_model_figS7/code/run_module.py                (everything)
  python A2f_hybrid_exemplar_model_figS7/code/run_module.py --skip-long    (without step 3)
The exit code is not zero when a step fails; the end of that step's log is printed.
outputs/logs/run_module_steps.log lists each step of the latest run, its exit code and its time.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import os
import subprocess
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths

PATHS = release_paths.module_paths(__file__)
RELEASE_ROOT = release_paths.release_root()
STEPS_LOG = os.path.join(PATHS['logs'], 'run_module_steps.log')


def remove_release_root(log_path):
    """Write paths in a log relative to the repository root, so no log holds a private absolute path."""
    with open(log_path, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    text = text.replace(RELEASE_ROOT + os.sep, '').replace(RELEASE_ROOT, '.')
    with open(log_path, 'w', encoding='utf-8') as handle:
        handle.write(text)


def run_step(log_name, script, arguments=()):
    """Run one script of code/ with this Python; its output goes to outputs/logs/<log_name>.log."""
    log_path = os.path.join(PATHS['logs'], log_name + '.log')
    command = [sys.executable, os.path.join(PATHS['code'], script)] + list(arguments)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    shown = ' '.join(['code/' + script] + list(arguments))
    print('%-46s ' % shown, end='', flush=True)
    started = time.time()
    with open(log_path, 'w', encoding='utf-8') as log:
        finished = subprocess.run(command, cwd=RELEASE_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    seconds = time.time() - started
    remove_release_root(log_path)
    if finished.returncode == 0:
        outcome = 'done'
    else:
        outcome = 'FAILED'
    print('%s in %4.0f s   (log: outputs/logs/%s.log)' % (outcome, seconds, log_name), flush=True)
    with open(STEPS_LOG, 'a', encoding='utf-8') as summary:
        summary.write('%-46s exit %d  %5.0f s\n' % (shown, finished.returncode, seconds))
    if finished.returncode != 0:
        with open(log_path, encoding='utf-8', errors='replace') as handle:
            print(''.join(handle.readlines()[-25:]))
        sys.exit(finished.returncode)


def main():
    parser = argparse.ArgumentParser(description='Module A2f: the hybrid-similarity exemplar model (Fig. S7).')
    parser.add_argument('--skip-long', action='store_true',
                        help='skip the 83-setting Bylinskii run')
    parser.add_argument('--provided-only', action='store_true',
                        help='run no simulation; use outputs/ or the saved run in data/paper_run/')
    options = parser.parse_args()

    if options.provided_only:
        mode = 'no simulation (--provided-only); results from outputs/ or data/paper_run/'
    elif options.skip_long:
        mode = 'rerun, without the full Bylinskii sweep (--skip-long)'
    else:
        mode = 'rerun of every simulation'
    print('module A2f, mode: ' + mode)
    with open(STEPS_LOG, 'w', encoding='utf-8') as summary:
        summary.write('module A2f, %s, mode: %s\n' % (time.strftime('%Y-%m-%d'), mode))
    started = time.time()

    if not options.provided_only:
        run_step('01_core', '01_core.py')
        run_step('02_sweep', '02_sweep.py')
        if not options.skip_long:
            run_step('03_bylinskii', '03_bylinskii.py')
        run_step('bylinskii_reference_cells', 'bylinskii_reference_cells.py')
    run_step('check_against_paper_run', 'check_against_paper_run.py')
    run_step('stats_figS7', 'stats_figS7.py')
    run_step('make_figureS7', 'make_figureS7.py')

    minutes = (time.time() - started) / 60.0
    with open(STEPS_LOG, 'a', encoding='utf-8') as summary:
        summary.write('all steps done in %.1f min\n' % minutes)
    # the comparison with the saved run, one line per file (the log check_against_paper_run.py wrote above)
    with open(os.path.join(PATHS['logs'], 'check_against_paper_run.log'), encoding='utf-8') as handle:
        print(handle.read().rstrip())
    print('all steps done in %.1f min' % minutes)
    print('outputs: outputs/numbers.csv, outputs/tests.csv, outputs/figures/FigS7_hybrid_exemplar_model.png')


if __name__ == '__main__':
    main()
