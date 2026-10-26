"""
release_paths.py: where things are in the release, and the provided-output switch.

Every Python script of the release finds its folders through this file, so no script contains an
absolute path and the release runs from wherever it is copied.

HOW A MODULE SCRIPT USES IT
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
    import release_paths

    paths = release_paths.module_paths(__file__)
    paths['module']        # the module folder, e.g. .../03_inversion_experiment_fig3
    paths['data']          # <module>/data
    paths['outputs']       # <module>/outputs (and 'figures', 'figure_data', 'logs' inside it)
    release_paths.release_root()             # the repository root folder
    release_paths.find_module('03')          # the folder of module 03, by its id
    release_paths.shared_path('things', 'concept_table.csv')

THE PROVIDED-OUTPUT SWITCH
Some results need an input that the release does not include (EXTERNAL_INPUTS.md lists them, M1 to M9).
A module declares the inputs it needs; when they are all present it regenerates the result, and
when one is missing it prints exactly

    PROVIDED OUTPUT: regenerating needs <input>; see EXTERNAL_INPUTS.md

and uses its data/provided/ files. It never falls back to a different input silently.

    needed = [release_paths.Input('M1', 'the Kramer et al. trial file imbysub-all-11-16-21.mat',
                                'external/kramer_trials/imbysub-all-11-16-21.mat')]
    mode = release_paths.regeneration_mode(needed)       # 'regenerate' or 'provided'
"""
import os
import sys
from collections import namedtuple

# Keep the shared folder free of __pycache__ folders: modules imported after this one (the other
# shared helpers) are not byte-compiled to disk. (Run with PYTHONDONTWRITEBYTECODE=1 to cover this
# file too.)
sys.dont_write_bytecode = True

# shared/python/release_paths.py -> shared/python -> shared -> the repository root
_RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODULE_FOLDERS = {
    '01': '01_rem_simulation_fig1',
    '02': '02_bylinskii_simulation_fig2',
    '03': '03_inversion_experiment_fig3',
    '04': '04_rare_feature_faces_fig4',
    '04b': '04b_face_same_item',
    '05': '05_things_reanalysis_fig5',
    '06': '06_image_set_experiments_fig6_fig7',
    '06b': '06b_reduced_set_variance_fig7',
    'A1': 'A1_measuring_memory_figS1',
    'A2a': 'A2a_rem_parameter_sweep_figS2',
    'A2b': 'A2b_category_structure_criteria_figS3',
    'A2c': 'A2c_continuous_recognition_figS4',
    'A2d': 'A2d_list_context_figS5',
    'A2e': 'A2e_measured_representations_figS6',
    'A2f': 'A2f_hybrid_exemplar_model_figS7',
    'A3a': 'A3a_model_through_own_lists_figS8',
    'A3b': 'A3b_hits_and_false_alarms_figS9',
    'A3c': 'A3c_face_study_model_and_sdt_figS10_S11',
    'A3d': 'A3d_exclusions_tableS1',
    'A4a': 'A4a_crowding_across_scales_figS12_tableS2',
    'A4b': 'A4b_things_concepts',
    'A5': 'A5_things_context_effects_figS13_S14',
    'A6a': 'A6a_faces_and_scenes_figS15',
    'A6b': 'A6b_embedding_size_figS16',
}

PROVIDED_MESSAGE = 'PROVIDED OUTPUT: regenerating needs %s; see EXTERNAL_INPUTS.md'

# One input a module needs: the EXTERNAL_INPUTS.md id (M1 to M9), a plain description, and the
# path relative to the repository root where it must be for the module to regenerate its results.
Input = namedtuple('Input', ['missing_id', 'description', 'relative_path'])


def release_root():
    """The folder that holds README.md, shared/ and the module folders."""
    return _RELEASE_ROOT


def shared_path(*parts):
    """A path inside shared/, e.g. shared_path('things', 'concept_table.csv')."""
    return os.path.join(_RELEASE_ROOT, 'shared', *parts)


def find_module(module_id):
    """The folder of a module, by its id ('03', 'A3a', ...)."""
    if module_id not in MODULE_FOLDERS:
        raise ValueError('unknown module id %r; known: %s' % (module_id, ', '.join(MODULE_FOLDERS)))
    return os.path.join(_RELEASE_ROOT, MODULE_FOLDERS[module_id])


def module_path(script_file):
    """The module folder of a script that sits in <module>/code/ (or deeper inside code/)."""
    folder = os.path.dirname(os.path.abspath(script_file))
    while True:
        parent = os.path.dirname(folder)
        if os.path.basename(folder) == 'code' and parent != folder:
            return parent
        if parent == folder:
            raise ValueError('%s is not inside a module code/ folder' % script_file)
        folder = parent


def module_paths(script_file, make_output_folders=True):
    """The standard folders of the module a script belongs to, as a dict:
    module, data, provided (data/provided), paper_run (data/paper_run), code, outputs,
    figures, figure_data, logs. The output folders are created unless make_output_folders=False."""
    module = module_path(script_file)
    paths = {
        'module': module,
        'data': os.path.join(module, 'data'),
        'provided': os.path.join(module, 'data', 'provided'),
        'paper_run': os.path.join(module, 'data', 'paper_run'),
        'code': os.path.join(module, 'code'),
        'outputs': os.path.join(module, 'outputs'),
        'figures': os.path.join(module, 'outputs', 'figures'),
        'figure_data': os.path.join(module, 'outputs', 'figure_data'),
        'logs': os.path.join(module, 'outputs', 'logs'),
    }
    if make_output_folders:
        for name in ('outputs', 'figures', 'figure_data', 'logs'):
            os.makedirs(paths[name], exist_ok=True)
    return paths


def input_is_present(needed_input):
    """True when the file or folder of one Input exists (a folder must not be empty)."""
    path = os.path.join(_RELEASE_ROOT, needed_input.relative_path)
    if os.path.isdir(path):
        visible = [name for name in os.listdir(path) if not name.startswith('.') and not name.startswith('README')]
        return len(visible) > 0
    return os.path.exists(path)


def regeneration_mode(needed_inputs, quiet=False):
    """'regenerate' when every needed input is present, otherwise 'provided'.
    For each input that is not present it prints the provided-output message (never a silent fallback)."""
    missing = [needed for needed in needed_inputs if not input_is_present(needed)]
    if not missing:
        if not quiet:
            for needed in needed_inputs:
                print('input found (%s): %s' % (needed.missing_id, needed.relative_path))
        return 'regenerate'
    for needed in missing:
        print(PROVIDED_MESSAGE % ('%s (%s, expected at %s)'
                                  % (needed.description, needed.missing_id, needed.relative_path)))
    return 'provided'
