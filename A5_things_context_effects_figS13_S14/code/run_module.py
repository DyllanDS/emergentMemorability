"""
run_module.py: module A5, SI Appendix, Section 5 "Context Effects in the Original THINGS Data"
(SI Appendix, Figs. S13 and S14), and the sentences of the main text that refer to it.

WHAT IT RUNS
    The analyses of Section 5 need the trial-level data of the original THINGS memorability study
    (Kramer et al. 2023), which this repository does not include (EXTERNAL_INPUTS.md, M1). The module
    therefore runs in one of two modes, chosen by what is on the machine (it says which):

    PROVIDED OUTPUT (without the trial file): the provided-output message is printed, and
        1. neighbor_share.py    the 62 percent / 4 percent fact (public SPoSE data; always computed)
        2. spose_quintiles.py   the SPoSE five-bin drop of the deep-network comparison (from the saved
                                20-bin rates)
        3. stats_section5.py    outputs/tests.csv: tests T38 to T42 (the intervals that need the trial
                                file from data/provided/trial_intervals.csv)
        4. check_provided.py    outputs/numbers.csv: every number of the section, read from data/provided/
        5. make_figureS13.py, make_figureS14.py   the figures, from data/provided/

    REGENERATION, when the trial file is in external/kramer_trials/ (or the trial table that
    extract_trials.py makes from it is in external/kramer_trials/derived/):
        extract_trials.py -> compute_context.py -> sdt_by_context.py -> sdt_by_context_adjusted.py ->
        kramer_quartile.py -> k_robustness.py -> crowding.py -> lone_exemplar.py -> fit_on_sequence.py
        (the last one only when all 14 deep-network spaces are in shared/embeddings_things/, M3; this
        repository includes 4 of them), then steps 1 to 5 above on the reports in outputs/regenerated/;
        step 3 then computes the trial-level intervals and writes data/provided/trial_intervals.csv.
        The trial tables (one row per trial or participant) stay in external/kramer_trials/derived/ and
        are not distributed.

Run (from the repository root)
    python A5_things_context_effects_figS13_S14/code/run_module.py                  (seconds without the
                                                                                     trial file)
    python A5_things_context_effects_figS13_S14/code/run_module.py --provided-only  (the saved reports even
                                                                                     when the trial file is here)
    python A5_things_context_effects_figS13_S14/code/run_module.py --skip-deep-network
                                                         (with the trial file: leave out fit_on_sequence.py,
                                                         about 15 minutes and 5 GB on an Apple M2 Pro; the
                                                         deep-network drop then comes from data/provided/)
    python A5_things_context_effects_figS13_S14/code/run_module.py --reuse-regenerated
                                                         (skip the analyses and use outputs/regenerated/)
Each step runs as its own Python process; its output is shown and saved in outputs/logs/<step>.log, with the
repository folder written as "." and the home folder as "~". Exit code 0 when every step succeeded.
"""
import argparse
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_FOLDER)
import a5_common  # noqa: E402
import release_paths  # noqa: E402

RELEASE_ROOT = str(a5_common.RELEASE_ROOT)
STEP_TIMES = []
CHAIN = ['compute_context.py', 'sdt_by_context.py', 'sdt_by_context_adjusted.py', 'kramer_quartile.py',
         'k_robustness.py', 'crowding.py', 'lone_exemplar.py']


def sanitise(text):
    """Write the repository folder as '.' and the home folder as '~' (no private paths in the logs)."""
    for root in sorted({os.path.realpath(RELEASE_ROOT), os.path.abspath(RELEASE_ROOT)}, key=len, reverse=True):
        text = text.replace(root + os.sep, '').replace(root, '.')
    home = os.path.expanduser('~')
    if home and home != '~':
        text = text.replace(home, '~')
    return text


def write_summary():
    path = a5_common.LOGS / 'run_module.log'
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('A5_things_context_effects_figS13_S14/code/run_module.py\n')
        for name, minutes, return_code in STEP_TIMES:
            handle.write('%-28s %6.1f minutes   exit code %d\n' % (name, minutes, return_code))


def run_step(script, *arguments):
    """Run one script of code/ as its own process from the repository root; echo and save its output."""
    name = os.path.splitext(script)[0]
    command = [sys.executable, os.path.join(CODE_FOLDER, script)] + list(arguments)
    log_path = a5_common.LOGS / (name + '.log')
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    print('\n==== %s: %s' % (name, ' '.join(sanitise(part) for part in command[1:])))
    sys.stdout.flush()
    start = time.time()
    process = subprocess.Popen(command, cwd=RELEASE_ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    with open(log_path, 'w', encoding='utf-8') as log:
        log.write('$ python %s\n' % ' '.join(sanitise(part) for part in command[1:]))
        for line in process.stdout:
            line = sanitise(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = process.wait()
        minutes = (time.time() - start) / 60.0
        log.write('\nexit code %d after %.1f minutes\n' % (return_code, minutes))
    STEP_TIMES.append((name, minutes, return_code))
    if return_code != 0:
        write_summary()
        raise SystemExit('step %s failed (exit code %d); see %s' % (name, return_code, a5_common.relative(log_path)))


def main():
    parser = argparse.ArgumentParser(description='Module A5: SI Appendix, Section 5, Figs. S13 and S14')
    parser.add_argument('--provided-only', action='store_true',
                        help='use the saved reports in data/provided/ even when the trial file is on this machine')
    parser.add_argument('--skip-deep-network', action='store_true',
                        help='with the trial file: do not run fit_on_sequence.py (the deep-network drop from '
                             'data/provided/)')
    parser.add_argument('--reuse-regenerated', action='store_true',
                        help='skip the analyses and use the reports already in outputs/regenerated/')
    args = parser.parse_args()
    start = time.time()

    if args.provided_only:
        mode = 'provided'
        print('--provided-only: the saved reports in data/provided/ are used')
    else:
        mode = a5_common.run_mode()

    if mode == 'provided':
        print('SI Appendix, Section 5 runs on the saved reports in data/provided/ (the output of the analyses '
              'run on the Kramer et al. trial file).')
        print(release_paths.PROVIDED_MESSAGE % ('the 14 THINGS deep-network spaces as well, for the deep-network drop '
                                              '(M3: ten of them are not included; expected in '
                                              'shared/embeddings_things/)'))
        source = 'provided'
    else:
        source = 'regenerated'
        if args.reuse_regenerated:
            print('--reuse-regenerated: the reports already in outputs/regenerated/ are used')
            a5_common.report_folder('regenerated')          # stops with a message when one is missing
        else:
            if mode == 'from_raw_file':
                try:
                    import h5py  # noqa: F401
                except ImportError:
                    raise SystemExit('reading the Kramer et al. trial file (MATLAB v7.3) needs the Python package h5py '
                                     '(pip install h5py)')
                run_step('extract_trials.py')
            for script in CHAIN:
                run_step(script)
            if args.skip_deep_network:
                print('\n--skip-deep-network: fit_on_sequence.py is not run; the deep-network drop comes from '
                      'data/provided/')
            elif a5_common.embeddings_present():
                run_step('fit_on_sequence.py')
            else:
                print('The deep-network drop comes from data/provided/ (the 14 spaces are not all present).')

    run_step('neighbor_share.py')
    run_step('spose_quintiles.py', '--source', source)
    run_step('stats_section5.py', '--source', source)     # writes data/provided/trial_intervals.csv with the trial file
    run_step('check_provided.py', '--source', source)
    run_step('make_figureS13.py', '--source', source)
    run_step('make_figureS14.py', '--source', source)

    write_summary()
    print('\nA5 done in %.1f minutes (mode: %s); outputs in A5_things_context_effects_figS13_S14/outputs/ '
          '(numbers.csv, tests.csv, figures/, figure_data/, logs/)' % ((time.time() - start) / 60.0, mode))


if __name__ == '__main__':
    main()
