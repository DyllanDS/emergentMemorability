"""
run_all.py: runs every module of the release, in an order that respects what each module reads,
then the two collectors that build the tables of reported numbers and statistics.

What it does
  For each module (01 to 06b for the main text, A1 to A6b for the SI Appendix) it runs the commands that
  the module's README gives, from the repository root, one after the other. MATLAB modules run through
  `matlab -batch`; MATLAB starts with restoredefaultpath (for that session only), so that folders a
  user added to the MATLAB path cannot replace MATLAB's own functions. After the modules it runs
      python collect_numbers.py                  -> reported_numbers.csv
      python statistics/collect_all_tests.py     -> statistics/all_tests.csv and statistics/participant_flow.csv
  It prints, for every module, the mode (rerun, provided output, ...), whether a module switched to
  provided output because an external input is missing (see EXTERNAL_INPUTS.md), the summary of its
  numbers.csv and its runtime. The exit code is 1 when any module or collector failed.

Order (later modules read files of earlier ones or check against them)
  01 02 03 04 04b 05 06 06b A1 A2a A2b A5 A2c A2d A2e A2f A4a A6a A4b A6b A3d A3a A3b A3c

Options
  --skip-long      leave out the long runs: module 02's simulation (Fig. 2 then comes from the saved run
                   in data/paper_run/), A2a's feature-count sweep, A2f's 83-setting Bylinskii run and
                   A3a's two model fits (the fits in outputs/ are used). When the external inputs are in
                   place, A5 also leaves out its deep-network step, A3c its 243-setting refits, and A2c
                   uses its provided output. About 1 hour instead of about 1 hour 40 minutes.
  --module ID ...  run only these modules (for example --module 03 A3b), then the collectors
  --provided-only  modules A2c, A2f, A3c, A4a and A5 use their provided outputs even when the external
                   input is there; for A2f this means no simulation at all (its check, statistics and
                   figure come from outputs/ or data/paper_run/). 04 and A6b accept the option and run as
                   usual, as do the other modules.
  --skip-matlab    run without MATLAB: the MATLAB steps of 01, 02, 03, 05 and 06 are left out (their
                   Python steps use the MATLAB outputs that ship in outputs/), and A1, A2a, A2b, A2d and
                   A2e use their shipped MATLAB outputs (A1 recomputes them in Python).
  --list           print the modules and the commands that would run, and stop.

MATLAB is found from the environment variable MATLAB (the full path of the matlab program or of the
MATLAB folder), else `matlab` on the PATH, else a standard installation (R2023b preferred, the tested
version). Python is the interpreter that runs this script.

Run (from the repository root):
    python run_all.py
    python run_all.py --skip-long
    python run_all.py --module 05 A6b
Requirements: environment.md. Runtimes are those of an Apple M2 Pro laptop (16 GB).
"""
import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import time

# No __pycache__ folders in the release: the modules' Python steps run with PYTHONDONTWRITEBYTECODE=1.
sys.dont_write_bytecode = True

RELEASE_ROOT = os.path.dirname(os.path.abspath(__file__))

# The modules, in the order they run. 'mode' is how the module obtains its numbers (as its README says);
# 'minutes' is the runtime of a full run (measured on an Apple M2 Pro with 16 GB) and the approximate runtime
# with --skip-long. 'matlab_minutes' is the part of these minutes spent in MATLAB (the step times of the
# module's README), left out of the expected runtime with --skip-matlab; 'provided_only_minutes' is A2f's
# runtime with --provided-only.
MODULES = [
    {'id': '01', 'folder': '01_rem_simulation_fig1', 'paper': 'Fig. 1',
     'mode': 'rerun (compared with the saved run in data/paper_run/)', 'minutes': (3.5, 3.5),
     'matlab_minutes': (2.5, 2.5)},
    {'id': '02', 'folder': '02_bylinskii_simulation_fig2', 'paper': 'Fig. 2',
     'mode': 'rerun (with --skip-long: the saved run in data/paper_run/)', 'minutes': (15, 1),
     'matlab_minutes': (15, 0.75)},
    {'id': '03', 'folder': '03_inversion_experiment_fig3', 'paper': 'Fig. 3',
     'mode': 'rerun', 'minutes': (0.6, 0.6), 'matlab_minutes': (0.4, 0.4)},
    {'id': '04', 'folder': '04_rare_feature_faces_fig4', 'paper': 'Fig. 4a, 4b; SI Appendix, Fig. S11a',
     'mode': 'rerun', 'minutes': (0.2, 0.2)},
    {'id': '04b', 'folder': '04b_face_same_item',
     'paper': "A Rare Feature's Advantage Depends on What the Face Is Tested Against, each face scored against itself",
     'mode': 'rerun', 'minutes': (0.1, 0.1)},
    {'id': '05', 'folder': '05_things_reanalysis_fig5', 'paper': 'Fig. 5',
     'mode': 'rerun', 'minutes': (6, 6), 'matlab_minutes': (0.5, 0.5)},
    {'id': '06', 'folder': '06_image_set_experiments_fig6_fig7', 'paper': 'Figs. 6 and 7',
     'mode': 'rerun', 'minutes': (2.5, 2.5), 'matlab_minutes': (2, 2)},
    {'id': '06b', 'folder': '06b_reduced_set_variance_fig7',
     'paper': ('Reducing Variability In Distinctiveness Reduces Variability in Memorability; Fig. 7 caption; '
               'Methods, Experimental Procedures and Analysis'),
     'mode': 'rerun', 'minutes': (0.2, 0.2)},
    {'id': 'A1', 'folder': 'A1_measuring_memory_figS1', 'paper': 'SI Appendix, Section 1, Fig. S1',
     'mode': 'rerun', 'minutes': (1, 1), 'matlab_minutes': (0.4, 0.4)},
    {'id': 'A2a', 'folder': 'A2a_rem_parameter_sweep_figS2', 'paper': 'SI Appendix, Section 2, Fig. S2',
     'mode': 'rerun', 'minutes': (8.5, 6), 'matlab_minutes': (2, 2)},
    {'id': 'A2b', 'folder': 'A2b_category_structure_criteria_figS3', 'paper': 'SI Appendix, Section 2, Fig. S3',
     'mode': 'rerun', 'minutes': (6, 6), 'matlab_minutes': (3.5, 3.5)},
    {'id': 'A5', 'folder': 'A5_things_context_effects_figS13_S14', 'paper': 'SI Appendix, Section 5, Figs. S13 and S14',
     'mode': 'provided output (M1; the deep-network results also M3); two results rerun', 'minutes': (0.1, 0.1)},
    {'id': 'A2c', 'folder': 'A2c_continuous_recognition_figS4', 'paper': 'SI Appendix, Section 2, Fig. S4',
     'mode': 'provided output (EXTERNAL_INPUTS.md M1, M6)', 'minutes': (0.1, 0.1)},
    {'id': 'A2d', 'folder': 'A2d_list_context_figS5', 'paper': 'SI Appendix, Section 2, Fig. S5',
     'mode': 'rerun', 'minutes': (1.7, 1.7), 'matlab_minutes': (1.4, 1.4)},
    {'id': 'A2e', 'folder': 'A2e_measured_representations_figS6', 'paper': 'SI Appendix, Section 2, Fig. S6',
     'mode': 'rerun', 'minutes': (1.5, 1.5), 'matlab_minutes': (1, 1)},
    {'id': 'A2f', 'folder': 'A2f_hybrid_exemplar_model_figS7', 'paper': 'SI Appendix, Section 2, Fig. S7',
     'mode': 'rerun (with --skip-long: the count over the 83 Bylinskii settings from outputs/ or the saved run)',
     'minutes': (25, 8), 'provided_only_minutes': 0.5},
    {'id': 'A4a', 'folder': 'A4a_crowding_across_scales_figS12_tableS2',
     'paper': 'SI Appendix, Section 4, Fig. S12 and Table S2',
     'mode': 'rerun; the THINGS results provided output (M1)', 'minutes': (0.3, 0.3)},
    {'id': 'A6a', 'folder': 'A6a_faces_and_scenes_figS15', 'paper': 'SI Appendix, Section 6, Fig. S15',
     'mode': 'faces: provided output (M4); scenes: rerun', 'minutes': (4, 4)},
    {'id': 'A4b', 'folder': 'A4b_things_concepts', 'paper': 'SI Appendix, Section 4, THINGS concepts',
     'mode': 'crowding rerun; memorability side provided output (M2)', 'minutes': (0.1, 0.1)},
    {'id': 'A6b', 'folder': 'A6b_embedding_size_figS16', 'paper': 'SI Appendix, Section 6, Fig. S16',
     'mode': 'rerun from the saved refits', 'minutes': (7.5, 7.5)},
    {'id': 'A3d', 'folder': 'A3d_exclusions_tableS1', 'paper': 'SI Appendix, Section 3, Table S1',
     'mode': 'rerun', 'minutes': (3, 3)},
    {'id': 'A3a', 'folder': 'A3a_model_through_own_lists_figS8', 'paper': 'SI Appendix, Section 3, Fig. S8',
     'mode': 'rerun', 'minutes': (6.5, 1)},
    {'id': 'A3b', 'folder': 'A3b_hits_and_false_alarms_figS9', 'paper': 'SI Appendix, Sections 1 and 3, Fig. S9',
     'mode': 'rerun (Fig. S9a, b); provided output (Fig. S9c, M1; the THINGS side of the halves comparison, M2)',
     'minutes': (0.5, 0.5)},
    {'id': 'A3c', 'folder': 'A3c_face_study_model_and_sdt_figS10_S11',
     'paper': 'SI Appendix, Section 3, Figs. S10 and S11b; Fig. 4c',
     'mode': 'rerun; the unequal-variance fits provided output (M5)', 'minutes': (4.5, 4.5)},
]


def python_step(script, *arguments):
    """A step that runs a Python script of the release (path from the repository root)."""
    return {'kind': 'python', 'script': script, 'arguments': list(arguments)}


def matlab_step(code_folder, call):
    """A step that runs one MATLAB call in a module's code folder."""
    return {'kind': 'matlab', 'folder': code_folder, 'call': call}


def module_steps(module, options):
    """The commands of one module, as its README gives them, for the chosen options."""
    folder = module['folder']
    code = folder + '/code'
    skip_long = options.skip_long
    provided_only = options.provided_only
    skip_matlab = options.skip_matlab
    steps = []
    module_id = module['id']

    if module_id == '01':
        if not skip_matlab:
            steps.append(matlab_step(code, 'run_module'))
        steps.append(python_step(code + '/stats_fig1.py'))
    elif module_id == '02':
        if not skip_matlab:
            if skip_long:
                steps.append(matlab_step(code, "run_module('skip_long')"))
            else:
                steps.append(matlab_step(code, "run_module('full')"))
        steps.append(python_step(code + '/stats_fig2.py'))
    elif module_id == '03':
        if not skip_matlab:
            steps.append(matlab_step(code, 'run_module'))
        steps.append(python_step(code + '/stats_fig3.py'))
    elif module_id == '05':
        if not skip_matlab:
            steps.append(matlab_step(code, 'run_module'))
        steps.append(python_step(code + '/stats_fig5.py'))
    elif module_id == '06':
        if not skip_matlab:
            steps.append(matlab_step(code, 'run_module'))
        steps.append(python_step(code + '/stats_fig6_fig7.py'))
        steps.append(python_step(code + '/shared_concepts_test.py'))
    else:
        # Modules 04, 04b, 06b and every SI Appendix module have one Python entry point, code/run_module.py.
        arguments = []
        accepts_skip_long = ('04', 'A2a', 'A2b', 'A2c', 'A2f', 'A3a', 'A6b', 'A3c')
        accepts_provided_only = ('04', 'A2c', 'A2f', 'A5', 'A4a', 'A6b', 'A3c')
        runs_matlab_itself = ('A1', 'A2a', 'A2b', 'A2d', 'A2e')
        accepts_skip_matlab = runs_matlab_itself + ('A6a',)   # A6a starts MATLAB only to rebuild optional inputs
        if skip_long and module_id in accepts_skip_long:
            arguments.append('--skip-long')
        if skip_long and module_id == 'A5':
            arguments.append('--skip-deep-network')
        if provided_only and module_id in accepts_provided_only:
            arguments.append('--provided-only')
        if skip_matlab and module_id in accepts_skip_matlab:
            arguments.append('--skip-matlab')
        steps.append(python_step(code + '/run_module.py', *arguments))
        # marked in --list: without --skip-matlab these five modules start MATLAB themselves
        steps[-1]['starts_matlab'] = module_id in runs_matlab_itself and not skip_matlab
    return steps


def find_matlab():
    """The MATLAB program: $MATLAB, else matlab on the PATH, else a standard installation; None if absent."""
    from_environment = os.environ.get('MATLAB')
    if from_environment:
        program = from_environment
        if os.path.isdir(program):
            # the MATLAB root folder (MATLAB's own meaning of this variable): the program is in its bin folder
            program = os.path.join(program, 'bin', 'matlab.exe' if os.name == 'nt' else 'matlab')
        if not (os.path.isfile(program) and os.access(program, os.X_OK)):
            sys.exit('run_all.py: the environment variable MATLAB points to %s, which is neither the matlab program '
                     'nor a MATLAB folder with bin/matlab' % from_environment)
        return program
    on_path = shutil.which('matlab')
    if on_path:
        return on_path
    installed = []
    for pattern in ('/Applications/MATLAB_R20*.app/bin/matlab', '/usr/local/MATLAB/R20*/bin/matlab',
                    'C:/Program Files/MATLAB/R20*/bin/matlab.exe'):
        installed.extend(sorted(glob.glob(pattern)))
    if not installed:
        return None
    for candidate in installed:
        if 'R2023b' in candidate:
            return candidate
    return installed[-1]


def command_of(step, matlab):
    """The command line of a step (a list), run with the repository root as the working folder."""
    if step['kind'] == 'python':
        return [sys.executable, step['script']] + step['arguments']
    # restoredefaultpath (for this MATLAB session only): no folder on the user's MATLAB path can replace
    # a MATLAB function the module uses. The code folder is given in full, so that a startup.m or the start
    # folder set in MATLAB's preferences cannot change where the call runs.
    folder = os.path.join(RELEASE_ROOT, step['folder']).replace("'", "''")
    call = "restoredefaultpath; cd('%s'); %s" % (folder, step['call'])
    return [matlab, '-batch', call]


def shown_command(step):
    """The command as a reader would type it from the repository root."""
    if step['kind'] == 'python':
        return ' '.join(['python', step['script']] + step['arguments'])
    return 'matlab -batch "restoredefaultpath; cd(\'%s\'); %s"' % (step['folder'], step['call'])


def run_step(step, matlab, environment):
    """Runs one step, showing its output as it comes. Returns (exit code, seconds, provided messages)."""
    print('\n$ ' + shown_command(step), flush=True)
    start_time = time.time()
    process = subprocess.Popen(command_of(step, matlab), cwd=RELEASE_ROOT, env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding='utf-8', errors='replace')
    provided_messages = []
    for line in process.stdout:
        line = line.rstrip('\n').replace(RELEASE_ROOT, '<release>')
        print('    ' + line, flush=True)
        if 'PROVIDED OUTPUT:' in line:
            provided_messages.append(line.strip())
    exit_code = process.wait()
    return exit_code, time.time() - start_time, provided_messages


def numbers_summary(module_folder):
    """'25 yes, 9 documented' from a module's outputs/numbers.csv (or a note that it is missing)."""
    numbers_file = os.path.join(RELEASE_ROOT, module_folder, 'outputs', 'numbers.csv')
    if not os.path.exists(numbers_file):
        return 'no outputs/numbers.csv'
    counts = {}
    with open(numbers_file, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            counts[row['match']] = counts.get(row['match'], 0) + 1
    return ', '.join('%d %s' % (counts[word], word) for word in sorted(counts))


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # the steps print d′, ρ and τ
    parser = argparse.ArgumentParser(description='Run every module of the release, then the collectors.')
    parser.add_argument('--skip-long', action='store_true', help='leave out the long runs')
    parser.add_argument('--module', nargs='+', metavar='ID', help='run only these modules (ids such as 03, A3b)')
    parser.add_argument('--provided-only', action='store_true',
                        help='modules A2c, A2f, A3c, A4a and A5 use their provided outputs even when the '
                             'external input is present')
    parser.add_argument('--skip-matlab', action='store_true', help='run without MATLAB (reuse shipped MATLAB outputs)')
    parser.add_argument('--list', action='store_true', help='print the commands and stop')
    options = parser.parse_args()

    known_ids = [module['id'] for module in MODULES]
    selected = MODULES
    if options.module:
        unknown = [module_id for module_id in options.module if module_id not in known_ids]
        if unknown:
            sys.exit('run_all.py: unknown module id(s) %s; known: %s' % (', '.join(unknown), ', '.join(known_ids)))
        selected = [module for module in MODULES if module['id'] in options.module]

    # Planned commands and the expected runtime.
    plan = []
    expected_minutes = 0
    column = 1 if options.skip_long else 0
    for module in selected:
        steps = module_steps(module, options)
        plan.append((module, steps))
        module_minutes = module['minutes'][column]
        if options.skip_matlab:
            module_minutes -= module.get('matlab_minutes', (0, 0))[column]
        if options.provided_only:
            module_minutes = module.get('provided_only_minutes', module_minutes)
        expected_minutes += module_minutes
    collectors = [python_step('collect_numbers.py'), python_step('statistics/collect_all_tests.py')]

    needs_matlab = any(step['kind'] == 'matlab' for module, steps in plan for step in steps)
    needs_matlab = needs_matlab or (not options.skip_matlab and
                                    any(module['id'] in ('A1', 'A2a', 'A2b', 'A2d', 'A2e') for module in selected))
    matlab = find_matlab() if needs_matlab else None
    if needs_matlab and matlab is None and not options.list:
        sys.exit('run_all.py: MATLAB was not found. Set the environment variable MATLAB to the matlab program, '
                 'or run with --skip-matlab (the MATLAB steps are then left out and their shipped outputs are used).')

    print('Release: Memorability as an Emergent Property (Simpson, Wixted and Brady), code and data')
    print('Options: %s' % (', '.join(name for name, value in (('--skip-long', options.skip_long),
                                                             ('--provided-only', options.provided_only),
                                                             ('--skip-matlab', options.skip_matlab)) if value)
                           or ('none' if options.module else 'none (full run)')))
    print('Modules: %s' % ' '.join(module['id'] for module in selected))
    if expected_minutes < 1:
        print('Expected runtime: under a minute on an Apple M2 Pro')
    else:
        rounded = '%.0f' % expected_minutes
        print('Expected runtime: about %s minute%s on an Apple M2 Pro' % (rounded, '' if rounded == '1' else 's'))
    if matlab:
        # the MATLAB installation folder name only (for example MATLAB_R2023b.app or R2023b), not the full path;
        # a link such as /usr/local/bin/matlab is followed to the installation first
        print('MATLAB: %s' % os.path.basename(os.path.dirname(os.path.dirname(os.path.realpath(matlab)))))
    if options.list:
        for module, steps in plan:
            print('\n%s  %s  (%s; mode: %s)' % (module['id'], module['folder'], module['paper'], module['mode']))
            for step in steps:
                print('    ' + shown_command(step) + ('  (starts MATLAB)' if step.get('starts_matlab') else ''))
        print('\nthen:')
        for step in collectors:
            print('    ' + shown_command(step))
        return

    # PYTHONUTF8: the steps print in UTF-8 on Windows too (its code pages lack the prime of d', rho and tau)
    environment = dict(os.environ, MPLBACKEND='Agg', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    if matlab:
        environment['MATLAB'] = matlab

    results = []
    run_start = time.time()
    for module, steps in plan:
        print('\n' + '=' * 100)
        print('%s  %s  (%s)' % (module['id'], module['folder'], module['paper']))
        print('mode: %s' % module['mode'])
        print('=' * 100, flush=True)
        module_start = time.time()
        failed_step = None
        provided_messages = []
        for step in steps:
            exit_code, seconds, messages = run_step(step, matlab, environment)
            provided_messages.extend(messages)
            if exit_code != 0:
                failed_step = '%s (exit code %d)' % (shown_command(step), exit_code)
                break
        minutes = (time.time() - module_start) / 60
        status = 'ok' if failed_step is None else 'FAILED: ' + failed_step
        results.append((module, status, minutes, len(provided_messages), numbers_summary(module['folder'])))
        print('\n[%s %s, %.1f min; %s; numbers.csv: %s]' % (module['id'], status, minutes,
                                                          '%d provided-output message(s)' % len(provided_messages)
                                                          if provided_messages else 'no provided output',
                                                          results[-1][4]), flush=True)

    print('\n' + '=' * 100)
    print('Collectors')
    print('=' * 100, flush=True)
    collector_failures = []
    for step in collectors:
        exit_code, seconds, messages = run_step(step, None, environment)
        if exit_code != 0:
            collector_failures.append(step['script'])

    print('\n' + '=' * 100)
    print('Summary (total %.1f min)' % ((time.time() - run_start) / 60))
    print('%-5s %-40s %-8s %-9s %s' % ('id', 'module', 'minutes', 'provided', 'status / numbers.csv'))
    for module, status, minutes, provided_count, summary in results:
        print('%-5s %-40s %8.1f %9s %s / %s' % (module['id'], module['folder'], minutes,
                                                 'yes' if provided_count else 'no', status, summary))
    for script in collector_failures:
        print('collector failed: %s' % script)
    any_failure = any(status != 'ok' for module, status, minutes, provided_count, summary in results)
    if any_failure or collector_failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
