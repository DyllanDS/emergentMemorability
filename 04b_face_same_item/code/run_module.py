"""
run_module.py: entry point of module 04b (the rare-feature face study scored each face against itself; main
text, A Rare Feature's Advantage Depends on What the Face Is Tested Against). It reads the data of module 04.

Run from the repository root:
    python 04b_face_same_item/code/run_module.py

WHAT IT DOES
  1. same_face_scores.py (see its docstring).
  2. outputs/numbers.csv (ledger items N2-001 and N2-002) and outputs/tests.csv (the header only: the module has
     no test).
  3. outputs/logs/run_module.log: everything printed, with paths relative to the repository root.

INPUTS   04_rare_feature_faces_fig4/data/face_trials.csv and face_participants.csv (through
         shared/python/face_loader.py)
OUTPUTS  outputs/same_face_items.csv, same_face_summary.csv, numbers.csv, tests.csv, logs/run_module.log
RUNTIME  a few seconds
"""
import os
import sys

sys.dont_write_bytecode = True   # keep the release free of __pycache__ folders
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, '..', '..', 'shared', 'python'))

import numbers_table  # noqa: E402
import release_paths  # noqa: E402
import tests_table  # noqa: E402

import same_face_scores  # noqa: E402

PATHS = release_paths.module_paths(__file__, make_output_folders=False)


class PrintLog:
    """Sends everything printed to the screen and keeps a copy for outputs/logs/run_module.log."""

    def __init__(self, screen):
        self.screen = screen
        self.text = []

    def write(self, text):
        self.screen.write(text)
        self.text.append(text)

    def flush(self):
        self.screen.flush()


def write_log(text):
    """Writes outputs/logs/run_module.log with every path relative to the repository root."""
    root = release_paths.release_root()
    text = text.replace(root + os.sep, '').replace(root, '.')
    os.makedirs(PATHS['logs'], exist_ok=True)
    with open(os.path.join(PATHS['logs'], 'run_module.log'), 'w', encoding='utf-8') as handle:
        handle.write(text)


def main():
    log = PrintLog(sys.stdout)
    sys.stdout = log
    try:
        summary = same_face_scores.main().set_index('measure')
        rare = summary['rare']
        ordinary = summary['ordinary']
        corrected = 'hit rate minus false alarm rate'
        numbers = numbers_table.NumbersTable('code/same_face_scores.py')
        numbers.add('N2-001', [rare[corrected], ordinary[corrected], rare["d'"], ordinary["d'"]], 'rerun',
                    note="mean over the 12 rare-feature and the 59 ordinary faces (each face weighted equally) of "
                         "hit rate minus false alarm rate, then of log-linear d'")
        numbers.add('N2-002', [rare['hit rate'], ordinary['hit rate'], rare['false alarm rate'],
                               ordinary['false alarm rate']], 'rerun',
                    note='mean hit rate (rare-feature, ordinary faces), then mean false alarm rate (rare-feature, '
                         'ordinary faces), each face scored against itself')
        numbers.write(PATHS['outputs'])
        for row in numbers.rows:
            print('  %s  printed %s  computed %s  match %s'
                  % (row['ledger_id'], row['paper_value'], row['computed_value'], row['match']))
        tests_table.TestsTable('code/same_face_scores.py').write(PATHS['outputs'])   # no test: the header only
    finally:
        sys.stdout = log.screen
        write_log(''.join(log.text))


if __name__ == '__main__':
    main()
