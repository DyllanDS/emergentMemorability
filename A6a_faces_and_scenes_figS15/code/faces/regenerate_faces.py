"""
regenerate_faces.py: the faces half of Fig. S15 from the face database itself (EXTERNAL_INPUTS.md, M4).

Why it exists
    The 10k US Adult Faces Database (Bainbridge, Isola and Oliva 2013) may not be redistributed, and
    nothing computed per face is in the repository. The module therefore holds the saved faces results
    (data/provided_faces/) and this chain, which recomputes them from the database for a reader who
    has obtained it (request form at wilmabainbridge.com/facememorability2.html).
What it does, in order (a step is skipped when its output already exists)
    1. Finds the face images and memorability-scores.xlsx in external/bainbridge_faces/, unzipping
       the archives there (10kfaces.zip, 2kmemorability.zip) into external/bainbridge_faces/derived/
       unzipped/ when they are not already unzipped. Stops unless it finds the 10,168 face images of
       10kfaces.zip.
    2. code/faces/00_build_image_order.py writes derived/image_order.csv (every face, with the scores
       and counts of the 2,222 scored faces).
    3. shared/embedding_extraction/extract_embeddings.py writes the 14 deep-network spaces to
       derived/regenerated_embeddings/<network>_<layer>.npz. This needs PyTorch, torchvision, timm, the
       OpenAI clip package and cornet (environment.md), takes about an hour, and its DINOv3 step needs
       about 17 GB of memory; shared/embedding_extraction/README.md describes the low-memory DINOv3 route.
       Files made another way and put into that folder are used as they are (the step is skipped when
       all 14 exist).
    4. code/faces/02_crowding_analysis.py, 04_ridge_metric.py and 05_noise_ceiling.py write the
       aggregate reports to A6a_faces_and_scenes_figS15/outputs/faces/ and the per-face values to
       external/bainbridge_faces/derived/ (about 5 minutes).
    5. Prints the recomputed faces numbers next to the saved ones in data/provided_faces/.
    Everything per face (the image list with scores, the embeddings, the per-face crowding values)
    stays in external/bainbridge_faces/derived/ and must not be redistributed.
Run (from the repository root): code/run_module.py runs it when external/bainbridge_faces/ holds the
archives; or directly
    python A6a_faces_and_scenes_figS15/code/faces/regenerate_faces.py
Options: --bainbridge-dir (where the archives or unzipped folders are), --derived-dir, --out-dir,
--embeddings-dir.
No model, embedding or image set is ever substituted: a missing package or an unexpected image count
stops the run with a message.
"""
import argparse
import os
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True
CODE_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_FOLDER)
import read_results  # noqa: E402  (also puts shared/python on the path)
import release_paths  # noqa: E402

FACES_FOLDER = os.path.dirname(os.path.abspath(__file__))
EXTRACTOR = release_paths.shared_path('embedding_extraction', 'extract_embeddings.py')
N_FACE_IMAGES = 10168             # the images of 10kfaces.zip (data/provided_faces/faces_crowding.txt)
SCORES_FILE_NAME = 'memorability-scores.xlsx'
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
SPACE_FILES = ['%s_%s.npz' % (network, layer)
               for network in ('alexnet', 'clip', 'cornet', 'dinov3', 'inception', 'resnet18', 'vgg16')
               for layer in ('early', 'late')]


def show(path):
    """A path relative to the repository root when it is inside the repository, for printing."""
    relative = os.path.relpath(path, release_paths.release_root())
    return path if relative.startswith('..') else relative


def is_resource_file(path):
    """macOS zip archives carry __MACOSX folders and '._' files that are not images."""
    return '__MACOSX' in path.split(os.sep) or os.path.basename(path).startswith('._')


def find_inputs(search_folders):
    """The folder holding the most face images and the first memorability-scores.xlsx, searching the
    given folders (subfolders named regenerated_embeddings or __MACOSX are skipped)."""
    best_folder, best_count, scores_file = None, 0, None
    for root_folder in search_folders:
        if not os.path.isdir(root_folder):
            continue
        for folder, subfolders, files in os.walk(root_folder):
            subfolders[:] = [name for name in subfolders if name not in ('regenerated_embeddings', '__MACOSX')]
            image_count = 0
            for name in files:
                path = os.path.join(folder, name)
                if is_resource_file(path):
                    continue
                if name.lower().endswith(IMAGE_EXTENSIONS):
                    image_count += 1
                if name.lower() == SCORES_FILE_NAME and scores_file is None:
                    scores_file = path
            if image_count > best_count:
                best_folder, best_count = folder, image_count
    return best_folder, best_count, scores_file


def safe_unzip(archive, target_folder):
    """Unzip one archive into target_folder, refusing any member path that would leave the folder."""
    target_real = os.path.realpath(target_folder)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.namelist():
            destination = os.path.realpath(os.path.join(target_folder, member))
            if not (destination == target_real or destination.startswith(target_real + os.sep)):
                raise SystemExit('refusing to unzip %s: the member %r points outside the folder' % (archive, member))
        handle.extractall(target_folder)


def locate_database(bainbridge_folder, derived_folder):
    """Step 1: the face image folder and the scores file, unzipping the archives when needed."""
    unzipped_folder = os.path.join(derived_folder, 'unzipped')
    search_folders = [bainbridge_folder, unzipped_folder]
    image_folder, image_count, scores_file = find_inputs(search_folders)
    if image_count < N_FACE_IMAGES or scores_file is None:
        archives = sorted(name for name in os.listdir(bainbridge_folder) if name.lower().endswith('.zip'))
        for archive in archives:
            target = os.path.join(unzipped_folder, os.path.splitext(archive)[0])
            if os.path.isdir(target):
                continue
            print('unzipping %s' % archive)
            os.makedirs(target, exist_ok=True)
            safe_unzip(os.path.join(bainbridge_folder, archive), target)
        image_folder, image_count, scores_file = find_inputs(search_folders)
    if scores_file is None:
        raise SystemExit('no %s found in %s (from 2kmemorability.zip of the database)' % (SCORES_FILE_NAME, show(bainbridge_folder)))
    if image_count != N_FACE_IMAGES:
        raise SystemExit('found %d face images (in %s); the analysis uses the %d images of 10kfaces.zip'
                         % (image_count, show(image_folder) if image_folder else 'no folder', N_FACE_IMAGES))
    print('face images: %d in %s' % (image_count, show(image_folder)))
    print('scores: %s' % show(scores_file))
    return image_folder, scores_file


def run_step(arguments):
    """Run one script as its own Python process (one job per process); stop if it fails."""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    print('\n$ python %s' % ' '.join(show(argument) if os.path.isabs(argument) else argument for argument in arguments))
    sys.stdout.flush()
    completed = subprocess.run([sys.executable] + arguments, env=environment)
    if completed.returncode != 0:
        raise SystemExit('step failed (exit code %d): %s' % (completed.returncode, os.path.basename(arguments[0])))


def extraction_packages_present():
    """True when the packages the extractor needs can be imported (checked in a separate process)."""
    check = subprocess.run([sys.executable, '-c', 'import torch, torchvision, timm, clip, cornet'],
                           capture_output=True)
    return check.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Recompute the faces half of Fig. S15 from the face database.')
    default_bainbridge = os.path.join(release_paths.release_root(), 'external', 'bainbridge_faces')
    parser.add_argument('--bainbridge-dir', default=default_bainbridge)
    parser.add_argument('--derived-dir', default=None, help='default: <bainbridge-dir>/derived')
    parser.add_argument('--out-dir', default=read_results.SOURCES['faces_regenerated']['folder'])
    parser.add_argument('--embeddings-dir', default=None, help='default: <derived-dir>/regenerated_embeddings')
    args = parser.parse_args()
    bainbridge_folder = os.path.abspath(args.bainbridge_dir)
    derived_folder = os.path.abspath(args.derived_dir or os.path.join(bainbridge_folder, 'derived'))
    out_folder = os.path.abspath(args.out_dir)
    embeddings_folder = os.path.abspath(args.embeddings_dir or os.path.join(derived_folder, 'regenerated_embeddings'))
    os.makedirs(derived_folder, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)

    # 1. the database
    image_folder, scores_file = locate_database(bainbridge_folder, derived_folder)

    # 2. the image list with the scores
    image_order = os.path.join(derived_folder, 'image_order.csv')
    if not os.path.exists(image_order):
        run_step([os.path.join(FACES_FOLDER, '00_build_image_order.py'), '--images-dir', image_folder,
                  '--scores-file', scores_file, '--out', image_order])

    # 3. the 14 deep-network spaces
    missing_spaces = [name for name in SPACE_FILES if not os.path.exists(os.path.join(embeddings_folder, name))]
    if missing_spaces:
        if not extraction_packages_present():
            raise SystemExit('The face embeddings are missing (%s) and the extraction packages (torch, torchvision, '
                             'timm, clip, cornet) cannot be imported. Install them (environment.md) or put the 14 '
                             'files into %s, then run again.' % (', '.join(missing_spaces), show(embeddings_folder)))
        run_step([EXTRACTOR, '--images-dir', image_folder, '--image-order', image_order, '--out-dir', embeddings_folder])
        missing_spaces = [name for name in SPACE_FILES if not os.path.exists(os.path.join(embeddings_folder, name))]
        if missing_spaces:
            raise SystemExit('the extraction did not write %s' % ', '.join(missing_spaces))
    else:
        print('the 14 face spaces are present in %s (extraction skipped)' % show(embeddings_folder))

    # 4. the three analyses
    run_step([os.path.join(FACES_FOLDER, '02_crowding_analysis.py'), '--embeddings-dir', embeddings_folder,
              '--image-order', image_order, '--out-dir', out_folder, '--per-image-dir', derived_folder])
    run_step([os.path.join(FACES_FOLDER, '04_ridge_metric.py'), '--embeddings-dir', embeddings_folder,
              '--image-order', image_order, '--out-dir', out_folder, '--per-image-dir', derived_folder])
    run_step([os.path.join(FACES_FOLDER, '05_noise_ceiling.py'), '--image-order', image_order,
              '--out-dir', out_folder])

    # 5. the recomputed numbers next to the saved ones
    print('\nFaces: recomputed (%s) against the saved results (data/provided_faces/)' % show(out_folder))
    original_folder = read_results.SOURCES['faces_regenerated']['folder']
    read_results.SOURCES['faces_regenerated']['folder'] = out_folder
    try:
        comparisons = [('held-out rho, equal weights', lambda source: read_results.heldout_rho(source, read_results.EQUAL_WEIGHTS_ROW)),
                       ('held-out rho, fitted weights', lambda source: read_results.heldout_rho(source, read_results.FITTED_WEIGHTS_ROW)),
                       ('whole-sample rho with the hit rate', lambda source: read_results.limiting_case(source, 'hit rate')[0]),
                       ('whole-sample rho with the false alarm rate', lambda source: read_results.limiting_case(source, 'FA rate')[0]),
                       ('split-half reliability over trials', lambda source: read_results.reliability(source, 'corrected recognition', 'trials')[0])]
        for label, reader in comparisons:
            regenerated = reader('faces_regenerated')
            provided = reader('faces_provided')
            print('  %-44s recomputed %+.3f   saved %+.3f   difference %+.3f' % (label, regenerated, provided, regenerated - provided))
    finally:
        read_results.SOURCES['faces_regenerated']['folder'] = original_folder


if __name__ == '__main__':
    main()
