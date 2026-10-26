"""
download_external.py: fetch or explain the third-party inputs the release does not redistribute.

The release runs without any download: the derived tables in shared/things/ carry what the code needs.
This script lets a reader (a) cross-check the shipped Kramer et al. tables against the public files,
(b) fetch the optional inputs that let more of the results be computed from their source data, and
(c) see how to request the inputs that cannot be downloaded. The downloads use the Python standard
library only; the cross-check of the Kramer et al. tables uses the release's shared/python helpers
(numpy, pandas). Every download is checked against its size and md5, and a file with another size or
md5 stops the script with an error.

RUN (from the repository root)
    python external/download_external.py                   # Kramer et al. public files -> external/kramer/
                                                           # then the cross-check, then the instructions
    python external/download_external.py --isola           # also the Isola et al. (2011) scene package
                                                           # (1.8 GB) -> external/isola/ (unzipped, checked)
    python external/download_external.py --spose-triplets  # also the SPoSE triplet files (60 MB)
                                                           # -> external/spose_triplets/ (the refits of
                                                           # SI Appendix, Fig. S16)
    python external/download_external.py --instructions    # only print the instructions

WHAT IT FETCHES
    external/kramer/THINGS_Memorability_Scores.csv   https://osf.io/download/z4ve9/  (Kramer et al. 2023, osf.io/5a7z6)
    external/kramer/Concept_to_category_linking.csv  https://osf.io/download/8b7hg/
    external/isola/ (optional)  https://people.csail.mit.edu/phillipi/Image%20memorability/cvpr_memorability_data.zip
    external/spose_triplets/ (optional)  trainset.txt and validationset.txt of osf.io/f5rn6 (THINGS-data, Hebart et al. 2023, CC BY 4.0)
"""
import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile

EXTERNAL = os.path.dirname(os.path.abspath(__file__))
RELEASE_ROOT = os.path.dirname(EXTERNAL)

KRAMER_FILES = [
    # (file name, url, size in bytes, md5)
    ('THINGS_Memorability_Scores.csv', 'https://osf.io/download/z4ve9/', 1675292, '99e6f81616675a73b3ddf92058d1cf50'),
    ('Concept_to_category_linking.csv', 'https://osf.io/download/8b7hg/', 113027, '4b42d2ea23005dad9bf4bddb41aeac68'),
]
# The data link of the project page web.mit.edu/phillipi/Public/WhatMakesAnImageMemorable/
ISOLA_URL = 'https://people.csail.mit.edu/phillipi/Image%20memorability/cvpr_memorability_data.zip'
ISOLA_INNER_FILES = [
    # (path inside the unzipped Data/ folder, size in bytes, md5)
    ('Image data/target_images.mat', 379916456, 'd622951bfcdb96af678d370aa365ac69'),
    ('Image data/filler_images.mat', 1267936634, '516ec26a211a2b384b19d42c9ca6eee6'),
    ('Experiment data/subj_data.mat', 2088008, '76325e095c847d869c6d27f820047840'),
    ('Experiment data/sorted_target_data.mat', 1281420, 'fa27ab9ab8f993b76ac89648d17c6f4b'),
    ('Experiment data/sorted_filler_data.mat', 2162456, '552325efac8e816ea55adf7d5588b49a'),
]
SPOSE_TRIPLET_FILES = [
    ('trainset.txt', 'https://osf.io/download/5mwts/', 54407680, 'f9a214d2cc631838c08f36e4e2ad66de'),
    ('validationset.txt', 'https://osf.io/download/b2a4j/', 5990048, '920638c5bbc905048ca583a6dcbdc016'),
]

INSTRUCTIONS = """
INPUTS THAT ARE NOT DOWNLOADED BY THIS SCRIPT (see EXTERNAL_INPUTS.md for which results need them)

  Kramer et al. (2023) trial-level data, imbysub-all-11-16-21.mat (about 6.3 GB; EXTERNAL_INPUTS M1)
      Not public. Request it from the original authors (W. A. Bainbridge, M. N. Hebart and
      colleagues). Put it in external/kramer_trials/. A per-image count table derived from it
      (things_per_image_counts.csv: image_nr, n_old, hits, n_new, false_alarms; M2) also goes there.

  Bainbridge, Isola and Oliva (2013) 10k US Adult Faces Database (EXTERNAL_INPUTS M4)
      Request access at https://wilmabainbridge.com/facememorability2.html (request form and license;
      the images and per-image scores may not be redistributed). Put 10kfaces.zip and
      2kmemorability.zip in external/bainbridge_faces/.

  THINGS images (Hebart et al. 2019; optional, EXTERNAL_INPUTS M7)
      https://osf.io/jum2f/ : images_THINGS.zip (5.0 GB, md5 3c018a7c37ac41ff7d84d7683fea0630), password
      after agreeing to the license (research and educational use only, no redistribution). Unzip so
      that external/things_images/<concept>/<file>.jpg exists. Needed only to re-extract the
      deep-network embeddings or to draw image thumbnails; no reported number needs it.

  THINGS Ventral Stream Spiking Dataset (Papale et al. 2025; optional)
      https://gin.g-node.org/paolo_papale/TVSD (DOI 10.12751/g-node.hc7zlv, CC BY 4.0):
      monkeyF/THINGS_normMUA.mat, monkeyN/THINGS_normMUA.mat and _logs/things_imgs.mat. Needed only
      to rebuild the monkey IT component scores from the spiking data with
      shared/things/provenance/monkeyFeatGen.m, which reads the two THINGS_normMUA.mat files as
      monkeyF.mat and monkeyN.mat (shared/things/provenance/README.md); the release ships the
      1,854 x 49 scores (shared/things/monkey_features.csv).

  SPoSE embedding (Hebart et al. 2020) and triplets (THINGS-data, Hebart et al. 2023)
      https://osf.io/f5rn6/ (CC BY 4.0). The 49-dimensional embedding ships in shared/things/. The
      triplet files (needed only for the refits of SI Appendix, Fig. S16) can be fetched with
      --spose-triplets (EXTERNAL_INPUTS M9).
"""


def md5_of(path):
    digest = hashlib.md5()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check_file(path, size, md5):
    """Stop with an error unless the file has the expected size and md5."""
    actual_size = os.path.getsize(path)
    if actual_size != size:
        raise SystemExit('SIZE MISMATCH for %s: %d bytes, expected %d' % (path, actual_size, size))
    actual_md5 = md5_of(path)
    if actual_md5 != md5:
        raise SystemExit('CHECKSUM MISMATCH for %s: md5 %s, expected %s' % (path, actual_md5, md5))
    print('  ok  %s (%d bytes, md5 %s)' % (os.path.relpath(path, RELEASE_ROOT), size, md5))


def download(url, path, size=None, md5=None):
    """Download url to path (skipped when a correct copy is already there), then check it."""
    if os.path.exists(path) and size is not None and os.path.getsize(path) == size and md5_of(path) == md5:
        print('  already present: %s' % os.path.relpath(path, RELEASE_ROOT))
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print('  downloading %s -> %s' % (url, os.path.relpath(path, RELEASE_ROOT)))
    temporary = path + '.part'
    with urllib.request.urlopen(url) as response, open(temporary, 'wb') as handle:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)
    os.replace(temporary, path)
    if size is not None:
        check_file(path, size, md5)


def get_kramer():
    print('Kramer et al. (2023) public files (cross-check of shared/things/things_images.csv and concept_categories.csv)')
    for name, url, size, md5 in KRAMER_FILES:
        download(url, os.path.join(EXTERNAL, 'kramer', name), size, md5)
    sys.dont_write_bytecode = True          # keep shared/python free of __pycache__ folders
    sys.path.insert(0, os.path.join(RELEASE_ROOT, 'shared', 'python'))
    import kramer_public
    kramer_public.verify_downloads()


def get_isola():
    print('Isola et al. (2011) scene memorability package (about 1.8 GB)')
    folder = os.path.join(EXTERNAL, 'isola')
    zip_path = os.path.join(folder, 'cvpr_memorability_data.zip')
    download(ISOLA_URL, zip_path)          # the page publishes no checksum; the inner files are checked below
    with zipfile.ZipFile(zip_path) as archive:
        # the package also holds macOS metadata copies (__MACOSX/..._*): not extracted
        members = [name for name in archive.namelist() if not name.startswith('__MACOSX/')]
        archive.extractall(folder, members=members)
    data_folder = None
    for root, folders, files in os.walk(folder):
        if '__MACOSX' in root.split(os.sep):
            continue
        if os.path.basename(root) == 'Data' and 'Image data' in folders:
            data_folder = root
            break
    if data_folder is None:
        raise SystemExit('the Isola package did not contain the expected Data/Image data folder')
    for relative, size, md5 in ISOLA_INNER_FILES:
        check_file(os.path.join(data_folder, relative), size, md5)


def get_spose_triplets():
    print('SPoSE triplets (THINGS-data, Hebart et al. 2023; SI Appendix, Fig. S16 refits only)')
    for name, url, size, md5 in SPOSE_TRIPLET_FILES:
        download(url, os.path.join(EXTERNAL, 'spose_triplets', name), size, md5)


def main():
    parser = argparse.ArgumentParser(description='Fetch or explain the third-party inputs of the release.')
    parser.add_argument('--isola', action='store_true', help='also fetch the Isola et al. scene package (1.8 GB)')
    parser.add_argument('--spose-triplets', action='store_true', help='also fetch the SPoSE triplet files (60 MB)')
    parser.add_argument('--instructions', action='store_true', help='only print the request instructions')
    args = parser.parse_args()
    if not args.instructions:
        get_kramer()
        if args.isola:
            get_isola()
        if args.spose_triplets:
            get_spose_triplets()
    print(INSTRUCTIONS)


if __name__ == '__main__':
    main()
