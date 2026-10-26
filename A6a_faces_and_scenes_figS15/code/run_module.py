"""
run_module.py: module A6a, SI Appendix, Section 6 (Faces and Scenes) and Fig. S15.

What it runs, in order (each script as its own Python process; its output is shown and saved in
outputs/logs/<step>.log, with the repository folder written as "."):
    1. Checks the scene inputs: the 14 scene spaces in shared/embeddings_isola/ and the image list
       data/scenes/image_order.csv, which must equal data/scenes/isola_scores.csv sorted by file name.
    2. Faces. The face database may not be redistributed (EXTERNAL_INPUTS.md, M4). When its archives are
       in external/bainbridge_faces/, code/faces/regenerate_faces.py recomputes the faces reports into
       outputs/faces/; otherwise the provided-output message is printed and the saved faces results in
       data/provided_faces/ are used.
    3. Scenes: code/scenes/02_crowding_analysis.py, 04_ridge_metric.py (about 2 GB of memory) and
       05_noise_ceiling.py (seconds; its reliability over participants needs the Isola participant
       matrices, which ship in data/scenes/; without them this script rebuilds them from
       external/isola/Data/ with MATLAB, and failing that the later steps read the scene reliability from
       the saved scene run in data/paper_run_scenes/ and print the provided-output message).
    4. compare_scenes_runs.py (this run against the saved scene run in data/paper_run_scenes/),
       intervals.py (the error bars), stats_figS15.py (outputs/numbers.csv and outputs/tests.csv; under
       a minute) and make_figureS15.py (outputs/figures/, outputs/figure_data/).
Run (from the repository root)
    python A6a_faces_and_scenes_figS15/code/run_module.py                         (about 3 minutes)
    python A6a_faces_and_scenes_figS15/code/run_module.py --reuse-scene-outputs   (skips step 3 and uses the
                                                                                   scene reports already in
                                                                                   outputs/scenes/; a few seconds)
Exit code: 0 when every step succeeded, otherwise 1, with a message that names the failing step and its
exit code.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_FOLDER)
import read_results  # noqa: E402  (also puts shared/python on the path)
import release_paths  # noqa: E402

PATHS = release_paths.module_paths(__file__)
RELEASE_ROOT = release_paths.release_root()
N_SCENE_SPACES = 14
STEP_TIMES = []


def sanitise(text):
    """Write the repository folder as '.' and the home folder as '~', so the logs hold no absolute paths."""
    for root in sorted({os.path.realpath(RELEASE_ROOT), os.path.abspath(RELEASE_ROOT)}, key=len, reverse=True):
        text = text.replace(root, '.')
    home = os.path.expanduser('~')
    if home and home != '~':
        text = text.replace(home, '~')
    return text


def run_step(name, arguments):
    """Run one step as its own process from the repository root, echo and save its output, stop on failure."""
    log_path = os.path.join(PATHS['logs'], name + '.log')
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    print('\n==== %s: %s' % (name, ' '.join(sanitise(argument) for argument in arguments)))
    sys.stdout.flush()
    start = time.time()
    process = subprocess.Popen(arguments, cwd=RELEASE_ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    with open(log_path, 'w', encoding='utf-8') as log:
        log.write('$ %s\n' % ' '.join(sanitise(argument) for argument in arguments))
        for line in process.stdout:
            line = sanitise(line)
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
        minutes = (time.time() - start) / 60.0
        log.write('\nexit code %d after %.1f minutes\n' % (return_code, minutes))
    STEP_TIMES.append((name, minutes, return_code))
    if return_code != 0:
        write_summary()
        raise SystemExit('step %s failed (exit code %d); see %s' % (name, return_code, read_results.relative(log_path)))


def python_step(name, script, *arguments):
    run_step(name, [sys.executable, os.path.join(CODE_FOLDER, script)] + list(arguments))


def write_summary():
    path = os.path.join(PATHS['logs'], 'run_module.log')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('A6a_faces_and_scenes_figS15/code/run_module.py\n')
        for name, minutes, return_code in STEP_TIMES:
            handle.write('%-28s %6.1f minutes   exit code %d\n' % (name, minutes, return_code))


def check_scene_inputs():
    """Step 1: the scene embeddings (shared/embeddings_isola/regenerated_<network>_<layer>.npz) and the
    scene image list."""
    import numpy as np
    import pandas as pd
    space_files = sorted(name for name in os.listdir(release_paths.shared_path('embeddings_isola'))
                         if name.startswith('regenerated_') and name.endswith('.npz'))
    if len(space_files) != N_SCENE_SPACES:
        raise SystemExit('shared/embeddings_isola/ holds %d scene spaces, expected %d' % (len(space_files), N_SCENE_SPACES))
    order = pd.read_csv(os.path.join(PATHS['data'], 'scenes', 'image_order.csv'), float_precision='round_trip')
    export = pd.read_csv(os.path.join(PATHS['data'], 'scenes', 'isola_scores.csv'), float_precision='round_trip')
    # image_order.csv = isola_scores.csv sorted by file name, with has_scores = 1 for every image (all
    # 10,442 carry the scores of Isola et al.; the 2,222 CVPR 2011 targets are flagged by in_cvpr_2222)
    export_sorted = export.sort_values('filename').reset_index(drop=True)
    if list(export_sorted['filename']) != list(order['filename']):
        raise SystemExit('data/scenes/image_order.csv is not isola_scores.csv sorted by file name')
    for column in ('kind', 'in_cvpr_2222', 'index_in_set', 'hits', 'misses', 'false_alarms', 'correct_rejections'):
        if list(export_sorted[column]) != list(order[column]):
            raise SystemExit('image_order.csv and isola_scores.csv differ in column %s' % column)
    for column in ('hit_rate', 'false_alarm_rate', 'corrected_recognition'):
        largest = float(np.max(np.abs(export_sorted[column].to_numpy() - order[column].to_numpy())))
        if largest > 1e-12:
            raise SystemExit('image_order.csv and isola_scores.csv differ in column %s by %.3g' % (column, largest))
    if not (order['has_scores'] == 1).all():
        raise SystemExit('image_order.csv: has_scores should be 1 for every scene image')
    print('scene inputs: %d spaces in shared/embeddings_isola/; data/scenes/image_order.csv = '
          'isola_scores.csv sorted by file name (%d images: %d targets, %d fillers; %d CVPR targets)'
          % (len(space_files), len(order), int((order['kind'] == 'target').sum()), int((order['kind'] == 'filler').sum()),
             int(order['in_cvpr_2222'].sum())))


def find_matlab():
    """The MATLAB executable: $MATLAB, else matlab on the PATH, else a standard installation; None if absent."""
    from_environment = os.environ.get('MATLAB')
    if from_environment:
        program = from_environment
        if os.path.isdir(program):
            # the MATLAB root folder (MATLAB's own meaning of this variable): the executable is in its bin folder
            program = os.path.join(program, 'bin', 'matlab.exe' if os.name == 'nt' else 'matlab')
        if not (os.path.isfile(program) and os.access(program, os.X_OK)):
            raise SystemExit('run_module.py: the MATLAB environment variable points to %s, which is neither the '
                             'matlab executable nor a MATLAB folder with bin/matlab' % from_environment)
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


def scene_subject_matrices(skip_matlab=False):
    """The participant matrices for the scene reliability: shipped in data/scenes/, or rebuilt from the
    public package with MATLAB (the MATLAB variable, else matlab on the PATH, else a standard installation),
    or None (then the 0.73 is read from the saved scene run in data/paper_run_scenes/)."""
    path = read_results.subject_matrices_path()
    if path is not None:
        return path
    package = os.path.join(RELEASE_ROOT, 'external', 'isola', 'Data')
    rebuilt = os.path.join(RELEASE_ROOT, 'external', 'isola', 'subject_matrices.mat')
    has_package = os.path.exists(os.path.join(package, 'Experiment data', 'subj_data.mat'))
    matlab = find_matlab() if has_package and not skip_matlab else None
    if matlab is not None:
        # restoredefaultpath (this MATLAB session only): no folder on the user's MATLAB path replaces a MATLAB function
        command = ("restoredefaultpath; cd('%s'); addpath('shared/embedding_extraction'); "
                   "export_subjects('external/isola/Data', 'external/isola/subject_matrices.mat')"
                   % RELEASE_ROOT.replace("'", "''"))
        run_step('scenes_rebuild_participant_matrices', [matlab, '-batch', command])
        return rebuilt
    print(release_paths.PROVIDED_MESSAGE % ('the Isola participant matrices for the scene reliability over participants '
                                          '(data/scenes/subject_matrices.mat, or external/isola/Data/ and MATLAB to '
                                          'rebuild them; M8)'))
    return None


def main():
    parser = argparse.ArgumentParser(description='Module A6a: faces and scenes, Fig. S15.')
    parser.add_argument('--reuse-scene-outputs', action='store_true',
                        help='skip the scene analyses and use the reports already in outputs/scenes/')
    parser.add_argument('--skip-matlab', action='store_true',
                        help='never start MATLAB (the scene reliability then comes from the saved scene run '
                             'when data/scenes/subject_matrices.mat is missing)')
    args = parser.parse_args()
    start = time.time()

    # 1. scene inputs
    check_scene_inputs()

    # 2. faces
    faces_mode = release_paths.regeneration_mode([read_results.FACES_INPUT])
    if faces_mode == 'regenerate':
        python_step('faces_regenerate', os.path.join('faces', 'regenerate_faces.py'))
    else:
        print('faces: the saved results in data/provided_faces/ are used')

    # 3. scenes
    if args.reuse_scene_outputs:
        print('\nscenes: --reuse-scene-outputs, the reports already in outputs/scenes/ are used')
        for which in ('crowding', 'ridge', 'folds', 'noise'):
            if not os.path.exists(read_results.file_path('scenes_rerun', which)):
                raise SystemExit('outputs/scenes/ is incomplete; run without --reuse-scene-outputs')
    else:
        python_step('scenes_02_crowding', os.path.join('scenes', '02_crowding_analysis.py'))
        python_step('scenes_04_ridge_metric', os.path.join('scenes', '04_ridge_metric.py'))
        subject_matrices = scene_subject_matrices(args.skip_matlab)
        python_step('scenes_05_noise_ceiling', os.path.join('scenes', '05_noise_ceiling.py'),
                    '--subject-matrices', subject_matrices if subject_matrices is not None else 'none')

    # 4. comparison, intervals, statistics, figure
    python_step('compare_scenes_runs', 'compare_scenes_runs.py')
    python_step('intervals', 'intervals.py')
    python_step('stats_figS15', 'stats_figS15.py')
    python_step('make_figureS15', 'make_figureS15.py')

    write_summary()
    print('\nA6a done in %.1f minutes; outputs in A6a_faces_and_scenes_figS15/outputs/ '
          '(numbers.csv, tests.csv, figures/, figure_data/, logs/)' % ((time.time() - start) / 60.0))


if __name__ == '__main__':
    main()
