"""
run_module.py: runs module A3c (SI Appendix, Section 3, the rare-feature face study: REM's predictions
and the preregistered d_a analysis; Fig. 4c and SI Appendix, Figs. S10 and S11b) from start to end.

MODE
  The unequal-variance fits of the d_a analysis need the MemToolbox2 Python package, which this
  repository does not include (EXTERNAL_INPUTS.md, item M5; put it in shared/memtoolbox2/).
  - provided (without MemToolbox2): everything that needs MemToolbox2 comes from the saved fits and
    resamples in data/provided/; the scripts print "PROVIDED OUTPUT: regenerating needs ..." and
    check_provided.py recomputes every number of those files from the saved fits and resamples.
    Everything else runs: the 729-setting REM sweep, the model side of the 243-setting run and the
    data side of the figures.
  - regenerate: when the package is in shared/memtoolbox2/, the registered point fit (about
    1.5 minutes), the leave-one-pair-out refits (about 7 minutes) and the 243-setting run scored like
    the data (several hours; skipped with --skip-long) also run, into outputs/, and are compared with
    the saved results, which stay the source of the module's numbers, tables and figures. The two long
    bootstraps (10,000 participant refits, 7.4 hours on 22 workers; 1,000 refits over the faces, about
    2 hours on 20 workers) are not run here; their saved resamples are used in both modes.

STEPS (each runs in its own process and writes outputs/logs/<step>.log; the run stops at the first failure)
   1. rem_final_spec.py                 the registered 729-setting sweep (about 2 minutes)
   2. rem_729_rates.py                  its rates per setting, for SI Appendix, Fig. S10a (about 2 minutes)
   3. roc_points.py                     rating counts and ROC points of the data (Fig. 4c)
   4. observed_contrasts.py             the observed hit and false alarm differences (Fig. S10a)
   5. sdt_shared_criteria.py --boot 0   the registered fit (regenerate) or the saved fit
   6. leave_one_pair_out.py             per-pair rates, then the refits (regenerate) or the saved refits
   7. rem_like_data.py                  the 243-setting run: model side always; fits when regenerating
   8. check_provided.py                 every saved number recomputed and checked
   9. make_figureS10.py, 10. make_figureS11.py   the figures
  11. stats_face_study.py               tests T48 to T51 -> outputs/tests.csv
  12. numbers_face_study.py             the 44 ledger items -> outputs/numbers.csv

OPTIONS
  --provided-only   use the saved results even when MemToolbox2 is present (every step is told so
                    through the environment variable A3c_PROVIDED_ONLY=1; see
                    face_study_common.import_memtoolbox)
  --skip-long       with MemToolbox2: skip the 243-setting refits (the point fit and leave-one-pair-out
                    still run); without it, nothing changes

RUN (from the repository root)
  python A3c_face_study_model_and_sdt_figS10_S11/code/run_module.py
Seeds: the 729-setting sweep and the model side of the 243-setting run, seed = setting number; the
registered bootstrap of the rate differences, resample i drawn with default_rng(i); the sideways
spread of the gray points of Fig. S10, default_rng(1) to default_rng(4).
Runtime: about 4 to 5 minutes on an Apple M2 Pro in provided mode (two 2-minute simulations, the rest
seconds); at most about 170 MB of memory per step.
"""
import argparse
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True   # no __pycache__ in code/ or shared/ (set before the imports below)
CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_FOLDER)
import face_study_common  # noqa: E402

release_paths = face_study_common.release_paths
PATHS = release_paths.module_paths(__file__)
RELEASE_ROOT = release_paths.release_root()

FULL_RUN_TEXT = """Module A3c: the full run commands (from the repository root)

1. What run_module.py runs without MemToolbox2 (about 4 to 5 minutes on an Apple M2 Pro): the
   729-setting sweep, the model side of the 243-setting run, the data side, every check, the figures
   and the tables.
     python A3c_face_study_model_and_sdt_figS10_S11/code/run_module.py

2. With MemToolbox2 in shared/memtoolbox2/ (EXTERNAL_INPUTS.md, item M5), run_module.py also
   runs these (each can be run on its own):
     python A3c_face_study_model_and_sdt_figS10_S11/code/sdt_shared_criteria.py --boot 0
         the registered point fit (about 1.5 minutes)
     python A3c_face_study_model_and_sdt_figS10_S11/code/leave_one_pair_out.py
         seven fits on 2 workers (about 7 minutes)
     python A3c_face_study_model_and_sdt_figS10_S11/code/rem_like_data.py --workers 2
         243 fits (several hours); skipped by run_module.py --skip-long

3. Not run by run_module.py; the saved resamples in data/provided/ are used in both modes:
     python A3c_face_study_model_and_sdt_figS10_S11/code/bootstrap_parallel.py --workers 2
         10,000 participant refits (7.4 hours on 22 workers; about 3.4 days on 2)
     python A3c_face_study_model_and_sdt_figS10_S11/code/bootstrap_items.py --boot 1000 --workers 2 --modes items
         1,000 refits over the faces (about 2 hours on 20 workers)
   They write to outputs/ (bootstrap_raw.jsonl and sdt_shared_criteria.txt; bootstrap_items.jsonl and
   bootstrap_items.txt); compare those reports with the saved ones in data/provided/.

The module's numbers, tables and figures always come from the saved results in data/provided/ for
everything that needs MemToolbox2; a new fit in outputs/ is compared with them (check_provided.py),
never used in their place.
"""


def run_step(log_name, script_name, arguments):
    """Runs one script of code/ with this Python, shows its output as it comes, and keeps a log in
    outputs/logs/<log_name>.log. Stops the module when the step fails."""
    print('\n==== %s %s' % (script_name, ' '.join(arguments)), flush=True)
    command = [sys.executable, os.path.join(CODE_FOLDER, script_name)] + arguments
    environment = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2')
    start_time = time.time()
    log_lines = ['$ python ' + ' '.join([os.path.relpath(command[1], RELEASE_ROOT)] + arguments)]
    process = subprocess.Popen(command, cwd=RELEASE_ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        line = line.rstrip('\n').replace(RELEASE_ROOT, '<release>')
        if not line.startswith('setting '):          # rem_729_rates.py progress lines go to the screen only
            log_lines.append(line)
        print(line, flush=True)
    exit_code = process.wait()
    seconds = time.time() - start_time
    log_lines.append('[%s finished with exit code %d]' % (script_name, exit_code))
    with open(os.path.join(PATHS['logs'], log_name + '.log'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(log_lines) + '\n')
    if exit_code != 0:
        sys.exit('run_module.py: step %s failed (exit code %d); see outputs/logs/%s.log' % (script_name, exit_code, log_name))
    print('[%s done in %.0f s]' % (script_name, seconds), flush=True)
    return seconds


def main():
    parser = argparse.ArgumentParser(description='Module A3c: the rare-feature face study, REM predictions and the '
                                                 'preregistered d_a analysis (Fig. 4c; SI Appendix, Figs. S10 and '
                                                 'S11b).')
    parser.add_argument('--provided-only', action='store_true',
                        help='use the saved results even when MemToolbox2 is present')
    parser.add_argument('--skip-long', action='store_true', help='with MemToolbox2: skip the 243-setting refits')
    arguments = parser.parse_args()

    print('Module A3c: SI Appendix, Section 3, the rare-feature face study (Fig. 4c; SI Appendix, Figs. S10 and S11b)')
    if arguments.provided_only:
        # Every step started below inherits this setting, so each script's own MemToolbox2 import
        # (face_study_common.import_memtoolbox) returns None and the script takes its provided-output branch.
        os.environ[face_study_common.PROVIDED_ONLY_VARIABLE] = '1'
    memtoolbox = face_study_common.import_memtoolbox(quiet=True)
    if memtoolbox is not None:
        mode = 'regenerate'
        print('mode: regenerate; MemToolbox2 found in shared/memtoolbox2/: the registered fit and the '
              'leave-one-pair-out refits run again%s; the two long bootstraps come from the saved resamples'
              % ('' if arguments.skip_long else ', and the 243-setting refits'))
    else:
        mode = 'provided'
        if arguments.provided_only:
            print('mode: provided (asked for with --provided-only); MemToolbox2 is not used even if it is present')
        else:
            face_study_common.import_memtoolbox()      # prints MISSING PACKAGE and PROVIDED OUTPUT
        print('mode: provided; the d_a analysis comes from the saved results in data/provided/; everything else runs')
    with open(os.path.join(PATHS['logs'], 'FULL_RUN_COMMAND.txt'), 'w', encoding='utf-8') as handle:
        handle.write(FULL_RUN_TEXT)

    runtimes = {}
    runtimes['rem_final_spec'] = run_step('rem_final_spec', 'rem_final_spec.py', [])
    runtimes['rem_729_rates'] = run_step('rem_729_rates', 'rem_729_rates.py', [])
    runtimes['roc_points'] = run_step('roc_points', 'roc_points.py', [])
    runtimes['observed_contrasts'] = run_step('observed_contrasts', 'observed_contrasts.py', [])
    runtimes['sdt_shared_criteria'] = run_step('sdt_shared_criteria', 'sdt_shared_criteria.py', ['--boot', '0'])
    runtimes['leave_one_pair_out'] = run_step('leave_one_pair_out', 'leave_one_pair_out.py', [])
    if mode == 'regenerate' and not arguments.skip_long:
        runtimes['rem_like_data'] = run_step('rem_like_data', 'rem_like_data.py', ['--workers', '2'])
    elif mode == 'regenerate':
        runtimes['rem_like_data'] = run_step('rem_like_data', 'rem_like_data.py', ['--model-side-only'])
    else:
        # provided mode: the model side only (the 243 fits come from the saved results)
        runtimes['rem_like_data'] = run_step('rem_like_data', 'rem_like_data.py', ['--model-side-only'])
    runtimes['check_provided'] = run_step('check_provided', 'check_provided.py', [])
    runtimes['make_figureS10'] = run_step('make_figureS10', 'make_figureS10.py', [])
    runtimes['make_figureS11'] = run_step('make_figureS11', 'make_figureS11.py', [])
    runtimes['stats_face_study'] = run_step('stats_face_study', 'stats_face_study.py', [])
    runtimes['numbers_face_study'] = run_step('numbers_face_study', 'numbers_face_study.py', [])

    summary = ('Module A3c finished (mode %s). Runtimes: %s; total %.0f s'
               % (mode, ', '.join('%s %.0f s' % (name, seconds) for name, seconds in runtimes.items()),
                  sum(runtimes.values())))
    print('\n' + summary)


if __name__ == '__main__':
    main()
