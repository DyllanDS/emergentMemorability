"""
run_module.py: runs module 04 (the preregistered rare-feature face study: Fig. 4a and 4b, and
SI Appendix, Fig. S11a) from start to end. Every number is computed from data in the repository;
the whole run takes about 10 s.

Steps, each one a separate Python process with its log in outputs/logs/<step>.log:
  1. code/design_check.py        the design counted from the data (50 faces, 6 repeats, 36 test trials, ...)
  2. code/face_analysis.py       the registered descriptive analysis: the participant flow (111
                                 sessions, 11 excluded, 100 kept), the rates and the rare against
                                 ordinary contrasts with the registered participant bootstrap
  3. code/memorability_check.py  the values of Fig. 4b and SI Appendix, Fig. S11a with their bootstrap
                                 intervals, and the two paired t tests
  4. code/make_figure4.py        Fig. 4a and 4b, and SI Appendix, Fig. S11a
  5. code/stats_fig4.py          the statistics of tests T07, T08, T09 and T54
  6. code/numbers_fig4.py        outputs/numbers.csv (every number of the paper this module computes)

Options --skip-long and --provided-only are accepted for run_all.py and change nothing here: the
module has no long step, and all of its inputs are in the repository.
Run from the repository root:
  python 04_rare_feature_faces_fig4/code/run_module.py
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
STEPS = ['design_check', 'face_analysis', 'memorability_check', 'make_figure4', 'stats_fig4', 'numbers_fig4']


def remove_release_root(log_path):
    """Write paths in a log relative to the repository root, so no log holds a private absolute path."""
    with open(log_path, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    text = text.replace(RELEASE_ROOT + os.sep, '').replace(RELEASE_ROOT, '.')
    with open(log_path, 'w', encoding='utf-8') as handle:
        handle.write(text)


def run_step(step_name):
    """Run code/<step_name>.py with this Python; its output goes to outputs/logs/<step_name>.log."""
    log_path = os.path.join(PATHS['logs'], step_name + '.log')
    command = [sys.executable, os.path.join(PATHS['code'], step_name + '.py')]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    print('%-28s ' % ('code/' + step_name + '.py'), end='', flush=True)
    started = time.time()
    with open(log_path, 'w', encoding='utf-8') as log:
        finished = subprocess.run(command, cwd=RELEASE_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    seconds = time.time() - started
    remove_release_root(log_path)
    if finished.returncode == 0:
        outcome = 'done'
    else:
        outcome = 'FAILED'
    print('%s in %3.0f s   (log: outputs/logs/%s.log)' % (outcome, seconds, step_name), flush=True)
    with open(STEPS_LOG, 'a', encoding='utf-8') as summary:
        summary.write('%-28s exit %d  %4.0f s\n' % ('code/' + step_name + '.py', finished.returncode, seconds))
    if finished.returncode != 0:
        with open(log_path, encoding='utf-8', errors='replace') as handle:
            print(''.join(handle.readlines()[-25:]))
        sys.exit(finished.returncode)


def main():
    parser = argparse.ArgumentParser(description='Module 04: the rare-feature face study (Fig. 4a and 4b, '
                                                 'SI Appendix, Fig. S11a).')
    parser.add_argument('--skip-long', action='store_true', help='accepted for run_all.py; nothing is long here')
    parser.add_argument('--provided-only', action='store_true',
                        help='accepted for run_all.py; every input is in the repository')
    parser.parse_args()

    mode = 'rerun (every number computed from data in the repository)'
    print('module 04, mode: ' + mode)
    with open(STEPS_LOG, 'w', encoding='utf-8') as summary:
        summary.write('module 04, %s, mode: %s\n' % (time.strftime('%Y-%m-%d'), mode))
    started = time.time()
    for step_name in STEPS:
        run_step(step_name)
    seconds = time.time() - started
    with open(STEPS_LOG, 'a', encoding='utf-8') as summary:
        summary.write('all steps done in %.0f s\n' % seconds)

    # the end of the numbers step: one line per ledger item and the match summary
    with open(os.path.join(PATHS['logs'], 'numbers_fig4.log'), encoding='utf-8') as handle:
        print(handle.read().rstrip())
    print('all steps done in %.0f s' % seconds)
    print('outputs: outputs/numbers.csv, outputs/tests.csv, outputs/figures/Fig4ab_rare_feature_faces.png, '
          'outputs/figures/FigS11a_memorability_by_condition.png')


if __name__ == '__main__':
    main()
