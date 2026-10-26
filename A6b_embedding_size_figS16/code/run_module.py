"""
run_module.py: entry point of module A6b (SI Appendix, Section 6, Embedding Size; Fig. S16).

WHAT IT DOES
Runs the module's steps in order. Each step is its own script and can also be run alone.
  1. extract_behavior.py  memorability of each THINGS concept in the maintained and changed image
                          sets (150 participants each, after the paper's exclusions)
  2. run_refit_sweep.py   the Fig. 5c statistic and the six Fig. 6 correlations in the published
                          49-feature SPoSE embedding and in each of the 14 saved refits
  3. gather_all_fits.py   the Fig. S16 table, its comparison with the saved run in
                          data/paper_run/all_fits.csv, and the comparison of the published
                          embedding's six Fig. 6 values with module 06's saved run
  4. stats_figS16.py      uncertainty statistics T47: n, p and concept-bootstrap interval for
                          every embedding
  5. make_figureS16.py    Fig. S16
  6. this script          outputs/numbers.csv: the module's 15 ledger numbers, from the outputs
The log of every step goes to outputs/logs/<step>.log.

THE REFITS
The 14 refits (data/refits/) are saved outputs of code/refit/spose.py (about 3 hours on an Apple
M2 Pro GPU). Training on a GPU does not give exactly the same weights twice, so the module analyzes
the saved files. With --refit it first fits them again from the SPoSE triplets with the four
commands that made them (REFIT_COMMANDS below), into outputs/refit_rerun/, then runs the steps
above, and at the end analyzes the new refits as well (outputs/refit_rerun/refit_sweep.csv); every
reported number still comes from the saved refits. --refit needs PyTorch and the triplet files in
external/spose_triplets/ (python external/download_external.py --spose-triplets;
EXTERNAL_INPUTS.md M9) and stops with a message if either is missing.

RUN (from the repository root)
  python A6b_embedding_size_figS16/code/run_module.py            about 7 minutes on an Apple M2 Pro
  python A6b_embedding_size_figS16/code/run_module.py --refit    about 3 more hours (GPU)
  options: --resamples <n>, --published-resamples <n>
                                             passed to stats_figS16.py (defaults 2000 per refit,
                                             10000 for the published embedding; smaller numbers
                                             only to test the code quickly)
           --device <mps, cuda or cpu>, --epochs <n>, --triplets-folder <folder>   for --refit (defaults:
                                             cuda or mps where PyTorch finds that GPU, else cpu; 400 epochs
                                             as in REFIT_COMMANDS; external/spose_triplets); cpu runs the
                                             same model and settings, much slower; fewer epochs only to
                                             test the code
           --skip-long, --provided-only      accepted from run_all.py; no effect here (the module
                                             has no long step to leave out and no data/provided/)

INPUTS   data/refits/ (14 refits and 4 summary files), data/paper_run/all_fits.csv, shared/things/,
         shared/ledger/paper_numbers.csv, 06_image_set_experiments_fig6_fig7/data/ (the image-set
         trials, the set composition and the saved run of Fig. 6)
OUTPUTS  outputs/numbers.csv, outputs/tests.csv, outputs/refit_sweep.csv, outputs/all_fits.csv,
         outputs/all_fits_vs_paper_run.csv, outputs/figS16_statistics.csv,
         outputs/image_set_concept_memorability.csv, outputs/figures/, outputs/figure_data/,
         outputs/logs/
"""
import argparse
import csv
import importlib.util
import os
import subprocess
import sys
import time

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True     # no __pycache__ folders inside the release
sys.path.insert(0, os.path.join(CODE_FOLDER, '..', '..', 'shared', 'python'))
import release_paths      # noqa: E402  (shared helper: release folders)
import numbers_table    # noqa: E402  (shared helper: the numbers.csv writer)

SWEEP_SCRIPT = 'code/run_refit_sweep.py'          # the script column of numbers.csv (module-relative)
STATISTICS_SCRIPT = 'code/stats_figS16.py'
FIGURE_SCRIPT = 'code/make_figureS16.py'
STEPS = [('extract_behavior', 'extract_behavior.py'),
         ('run_refit_sweep', 'run_refit_sweep.py'),
         ('gather_all_fits', 'gather_all_fits.py'),
         ('stats_figS16', 'stats_figS16.py'),
         ('make_figureS16', 'make_figureS16.py')]

# The four spose.py commands that made the 14 refits of Fig. S16, as (output subfolder, spose.py
# arguments): lambda lists in run order, seeds, --init-dim 100, at most 400 epochs, batches of
# 8,192, learning rate 1e-3 (the spose.py default), final (not best-validation) weights, device mps.
# The lambda sweep also gives two fits that keep all 100 features (lambda 0.0035 and 0.004) and one
# that loses every feature (0.019); these three are not among the 14 refits.
REFIT_COMMANDS = [
    ('lamsweep', ['--lambdas', '0.0035', '0.004', '0.0045', '0.005', '0.006', '0.007', '0.008', '0.009',
                  '0.011', '0.013', '0.016', '0.019', '--epochs', '400', '--batch-size', '8192',
                  '--init-dim', '100', '--seed', '0', '--tag', 'lamsweep']),
    ('lamfill', ['--lambdas', '0.0135', '0.014', '0.0145', '--epochs', '400', '--batch-size', '8192',
                 '--init-dim', '100', '--seed', '0', '--tag', 'lamfill']),
    ('sparsityctl', ['--lambdas', '0.006', '--epochs', '400', '--batch-size', '8192', '--init-dim', '100',
                     '--seed', '1', '--tag', 'sparse_s1']),
    ('sparsityctl', ['--lambdas', '0.006', '--epochs', '400', '--batch-size', '8192', '--init-dim', '100',
                     '--seed', '2', '--tag', 'sparse_s2']),
]


# ----------------------------------------------------------------------------------- running steps
def without_private_path(text, release_root):
    """text with the release folder's own location replaced by '.' (logs carry no private path)."""
    return text.replace(release_root + os.sep, '').replace(release_root, '.')


def run_logged(step_label, command, log_path, release_root, one_core=True):
    """Run one command from the repository root, show its output and save it to log_path (without the
    release folder's own location). Stops the module when the command fails."""
    environment = dict(os.environ)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['PYTHONUNBUFFERED'] = '1'          # show each line as soon as the step prints it
    environment['MPLBACKEND'] = 'Agg'
    if one_core:
        # One processor core for the numerical library: the matrices of the analysis are small, so
        # more threads do not make the steps faster, and every output is identical either way.
        # Not used for the refits, which PyTorch may spread over cores.
        for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            environment.setdefault(variable, '1')
    started = time.time()
    with open(log_path, 'w') as log:
        process = subprocess.Popen(command, cwd=release_root, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            line = without_private_path(line, release_root)
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return_code = process.wait()
        minutes = (time.time() - started) / 60
        log.write('\n[finished with exit code %d after %.1f minutes]\n' % (return_code, minutes))
    if return_code != 0:
        raise SystemExit('%s failed (exit code %d); see %s' % (step_label, return_code,
                                                               os.path.relpath(log_path, release_root)))


class LoggedOutput:
    """Shows what this script prints and copies it, without the release folder's location, to a log."""

    def __init__(self, log_path, release_root):
        self.log = open(log_path, 'w')
        self.release_root = release_root
        self.console = sys.stdout

    def write(self, text):
        text = without_private_path(text, self.release_root)
        self.console.write(text)
        self.log.write(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def refit_embeddings(arguments, paths, release_root):
    """--refit, first part: fit the 14 refits again into outputs/refit_rerun/ (analyse_new_refits
    analyzes them after the module's own steps, which write the inputs that analysis needs)."""
    if importlib.util.find_spec('torch') is None:
        raise SystemExit('--refit needs PyTorch (pip install torch); without --refit the module analyzes '
                         'the saved refits in data/refits/')
    triplets_folder = os.path.abspath(arguments.triplets_folder)
    for file_name in ('trainset.txt', 'validationset.txt'):
        if not os.path.exists(os.path.join(triplets_folder, file_name)):
            raise SystemExit('--refit needs the SPoSE triplets: %s is missing (python external/download_external.py '
                             '--spose-triplets; EXTERNAL_INPUTS.md M9)' % os.path.join(triplets_folder, file_name))
    if arguments.device in (None, 'mps'):
        import torch
        if arguments.device is None:
            # the first device PyTorch finds: an NVIDIA GPU, the Apple GPU (the device of the saved refits), the processor
            arguments.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        elif not torch.backends.mps.is_available():
            raise SystemExit('the Apple GPU (mps) is not available here; run with --device cpu (the same '
                             'model and settings, computed on the processor, much slower)')
    rerun_folder = os.path.join(paths['outputs'], 'refit_rerun')
    spose_program = os.path.join(CODE_FOLDER, 'refit', 'spose.py')
    print('refitting the 14 embeddings into %s (device %s)' % (os.path.relpath(rerun_folder, release_root),
                                                              arguments.device))
    for command_number, (subfolder, spose_arguments) in enumerate(REFIT_COMMANDS):
        spose_arguments = list(spose_arguments)
        if arguments.epochs is not None:
            spose_arguments[spose_arguments.index('--epochs') + 1] = str(arguments.epochs)
        command = ([sys.executable, '-u', spose_program] + spose_arguments
                   + ['--outdir', os.path.join(rerun_folder, subfolder), '--device', arguments.device,
                      '--triplets', triplets_folder])
        run_logged('refit command %d (%s)' % (command_number + 1, subfolder), command,
                   os.path.join(paths['logs'], 'refit_%d_%s.log' % (command_number + 1, subfolder)), release_root,
                   one_core=False)


def analyse_new_refits(paths, release_root):
    """--refit, second part (after step 1 has written outputs/image_set_concept_memorability.csv):
    the Fig. 5c and Fig. 6 statistics of the new refits, in outputs/refit_rerun/refit_sweep.csv."""
    rerun_folder = os.path.join(paths['outputs'], 'refit_rerun')
    run_logged('run_refit_sweep.py on the new refits',
               [sys.executable, os.path.join(CODE_FOLDER, 'run_refit_sweep.py'), '--refit-folder', rerun_folder,
                '--output', os.path.join(rerun_folder, 'refit_sweep.csv')],
               os.path.join(paths['logs'], 'refit_rerun_sweep.log'), release_root)
    print('the new refits are analyzed in outputs/refit_rerun/refit_sweep.csv; every reported number '
          '(numbers.csv, tests.csv, Fig. S16) comes from the saved refits in data/refits/')


# ---------------------------------------------------------------------------------- numbers.csv
def read_csv_rows(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def value_range(values, decimals=3):
    """'a to b' from the smallest to the largest value."""
    return '%.*f to %.*f' % (decimals, min(values), decimals, max(values))


def write_numbers_table(paths):
    """outputs/numbers.csv: one row per ledger item of the module, every value read from
    outputs/refit_sweep.csv (step 2) except the penalty strengths, which come from the refits'
    summary files, the published embedding's interval, from outputs/figS16_statistics.csv (step 4),
    and the black point of Fig. S16, from its plotted numbers (step 5)."""
    import run_refit_sweep      # this module: the published embedding's name and the summary-file reader
    print('step 6: numbers.csv')
    sweep = read_csv_rows(os.path.join(paths['outputs'], 'refit_sweep.csv'))
    published = [row for row in sweep if row['embedding'] == run_refit_sweep.PUBLISHED_NAME][0]
    refits = [row for row in sweep if row['embedding'] != run_refit_sweep.PUBLISHED_NAME]
    summaries = run_refit_sweep.read_summaries(os.path.join(paths['data'], 'refits'))
    statistics = read_csv_rows(os.path.join(paths['outputs'], 'figS16_statistics.csv'))
    published_statistics = [row for row in statistics if row['embedding'] == run_refit_sweep.PUBLISHED_NAME][0]

    features = [int(row['number_of_features']) for row in refits]
    correlations = [float(row['fig5c_r']) for row in refits]
    parametric_p = [float(row['fig5c_p']) for row in refits]
    starting = sorted(set(int(float(row['starting_features'])) for row in refits))
    pruned = [row for row in refits if row['penalty_removed_features'] == 'True']
    lambdas = [float(summaries[row['file']]['lmbda']) for row in refits]
    seeds = sorted(set(int(float(row['seed'])) for row in refits))
    number_of_refits = len(refits)
    number_below_zero = sum(1 for value in correlations if value < 0)
    smallest = features.index(min(features))
    largest = features.index(max(features))
    significant = [index for index in range(number_of_refits) if parametric_p[index] < 0.01]
    smallest_significant_size = min(features[index] for index in significant)
    largest_p_from_that_size = max(parametric_p[index] for index in range(number_of_refits)
                                   if features[index] >= smallest_significant_size)
    published_features = int(published['number_of_features'])
    published_r = float(published['fig5c_r'])
    published_interval = [float(published_statistics['bootstrap_low']), float(published_statistics['bootstrap_high'])]
    assert len(starting) == 1 and len(pruned) == number_of_refits
    # the starting size the analysis assumes must be the one the refit commands pass (--init-dim)
    starting_in_commands = sorted(set(int(spose_arguments[spose_arguments.index('--init-dim') + 1])
                                      for _, spose_arguments in REFIT_COMMANDS))
    assert starting_in_commands == starting, 'REFIT_COMMANDS and the analyzed starting size disagree'

    def column(name):
        return [float(row[name]) for row in refits]
    # the four Fig. 6 correlations that use SPoSE features, each with the sign it has in Fig. 6: in how
    # many refits it keeps that sign, and its range over the refits
    human_signs = []
    for label, name, sign in (('6c maintained vs THINGS', 'fig6c_maintained_vs_things_r', 1),
                              ('6d maintained, human', 'fig6d_maintained_human_r', -1),
                              ('6f changed vs THINGS', 'fig6f_changed_vs_things_r', -1),
                              ('6g changed, human', 'fig6g_changed_human_r', -1)):
        values = column(name)
        kept = sum(1 for value in values if value * sign > 0)
        if max(values) - min(values) < 1e-12:
            spread = 'r = %.3f in every refit' % values[0]
        else:
            spread = 'r = %s' % value_range(values)
        human_signs.append((label, kept, spread))
    # the refits in which all four keep their signs at once
    refits_keeping_human_signs = 0
    for row in refits:
        human_kept = (float(row['fig6c_maintained_vs_things_r']) > 0 and float(row['fig6d_maintained_human_r']) < 0
                      and float(row['fig6f_changed_vs_things_r']) < 0 and float(row['fig6g_changed_human_r']) < 0)
        if human_kept:
            refits_keeping_human_signs += 1

    table = numbers_table.NumbersTable(script=SWEEP_SCRIPT)
    table.add('L3-015', '%d refits, %d to %d features, r = %s (%d of %d below zero)'
              % (number_of_refits, min(features), max(features), value_range(correlations), number_below_zero,
                 number_of_refits), mode='rerun',
              note='Main-text sentence that points to SI Appendix, Section 6: the number of refits, their numbers '
                   'of features, and the range of the Fig. 5c correlation (Pearson r across features) over the '
                   'refits, with how many are below zero.')
    table.add('L9-001', published_features, mode='rerun',
              note='Number of features (columns) of the published SPoSE embedding, '
                   'shared/things/spose_embedding_49d_sorted.txt (Hebart et al., 2020).')
    table.add('L9-025', published_features, mode='rerun',
              note='Number of features (columns) of the published SPoSE embedding, '
                   'shared/things/spose_embedding_49d_sorted.txt (Hebart et al., 2020).')
    table.add('L9-027', starting[0], mode='rerun',
              note='Number of features every refit starts from (--init-dim %d in REFIT_COMMANDS, '
                   'code/run_module.py).' % starting[0])
    table.add('L9-028', 'lambda %s to %s (%d refits; seeds %s)' % ('%g' % min(lambdas), '%g' % max(lambdas),
                                                                 number_of_refits,
                                                                 ', '.join(str(seed) for seed in seeds)),
              mode='rerun',
              note='Sparsity penalty strengths (lambda) and random seeds of the %d refits, from the summary files '
                   'in data/refits/.' % number_of_refits)
    table.add('L9-029', [min(features), max(features)], mode='rerun',
              note='Fewest and most features (columns) among the %d saved refits in data/refits/.' % number_of_refits)
    table.add('L9-030', 'all %d refits below zero, r = %s' % (number_of_refits, value_range(correlations)),
              mode='rerun',
              note='Fig. 5c statistic in each refit (Pearson r across features between each feature\'s Spearman '
                   'correlation with summed similarity and its Spearman correlation with memorability): number '
                   'of refits below zero and range.')
    table.add('L9-031', 'the four Fig. 6 correlations that use SPoSE features keep their signs in %d of %d refits: %s'
              % (refits_keeping_human_signs, number_of_refits,
                 '; '.join('%s %s (%d of %d)' % (label, spread, kept, number_of_refits)
                           for label, kept, spread in human_signs)),
              mode='rerun',
              note='Signs and ranges over the refits of the four Fig. 6 correlations that use SPoSE features (6c, '
                   '6d, 6f, 6g).')
    table.add('L9-032', 'outputs/figures/FigS16_spose_dimensionality.png: %d refits (%d to %d features) and the '
                        'published embedding' % (number_of_refits, min(features), max(features)),
              mode='rerun', script=FIGURE_SCRIPT,
              note='Fig. S16 as drawn: blue points are the refits (number of features, Fig. 5c correlation), the '
                   'black point is the published 49-feature embedding and the dashed line is zero; plotted numbers '
                   'in outputs/figure_data/FigS16_spose_dimensionality.csv.')
    table.add('L9-033', [min(features), max(features), starting[0]], mode='rerun',
              note='Fewest and most features kept among the refits, then the number of features the refits start '
                   'from (REFIT_COMMANDS).')
    figure_rows = read_csv_rows(os.path.join(paths['figure_data'], 'FigS16_spose_dimensionality.csv'))
    black_point = [row for row in figure_rows if row['embedding'] == run_refit_sweep.PUBLISHED_NAME][0]
    table.add('L9-161', int(black_point['number_of_features']), mode='rerun', script=FIGURE_SCRIPT,
              note='Number of features of the black point of Fig. S16 (the published embedding), '
                   'outputs/figure_data/FigS16_spose_dimensionality.csv.')
    table.add('L9-034', [min(correlations), max(correlations), number_of_refits], mode='rerun',
              note='Lowest and highest Fig. 5c correlation (Pearson r across features) over the refits, then the '
                   'number of refits (%d of %d below zero).' % (number_below_zero, number_of_refits))
    from_that_size = [index for index in range(number_of_refits) if features[index] >= smallest_significant_size]
    table.add('L9-035', [largest_p_from_that_size, smallest_significant_size], mode='rerun',
              note='Largest two-sided parametric p (t test on the number of features minus 2 degrees of freedom) '
                   'among the refits with %d or more features (%d of %d below 0.01), then the fewest features of a '
                   'refit with p below 0.01.'
                   % (smallest_significant_size, sum(1 for index in from_that_size if parametric_p[index] < 0.01),
                      len(from_that_size)))
    table.add('L9-036', [parametric_p[smallest], features[smallest]], mode='rerun',
              note='Two-sided parametric p of the Fig. 5c correlation in the refit with the fewest features (%d '
                   'features; t test on the number of features minus 2 degrees of freedom), then its number of '
                   'features.' % features[smallest])
    table.add('L9-037', [published_r] + published_interval, mode='rerun',
              script=SWEEP_SCRIPT + '; ' + STATISTICS_SCRIPT,
              note='Fig. 5c correlation (Pearson r across features) in the published embedding, then its 95%% '
                   'concept-bootstrap interval (%s resamples, seed %s; outputs/figS16_statistics.csv).'
                   % ('{:,}'.format(int(published_statistics['number_of_resamples'])), published_statistics['seed']))
    table.write(paths['outputs'])
    print('  Fig. 5c r: refit with the most features (%d) %.4f, with the fewest (%d) %.4f, published %.4f'
          % (features[largest], correlations[largest], features[smallest], correlations[smallest], published_r))


def main():
    parser = argparse.ArgumentParser(description='Module A6b: Fig. S16 (SPoSE refits at 6 to 81 features)')
    parser.add_argument('--refit', action='store_true',
                        help='refit the 14 embeddings first (PyTorch, triplets, hours on a GPU)')
    parser.add_argument('--device', default=None, help='for --refit: cuda (NVIDIA GPU), mps (Apple GPU, the device of the '
                                                       'saved refits) or cpu; the default is the first of '
                                                       'these that PyTorch finds')
    parser.add_argument('--epochs', type=int, default=None, help='for --refit: override the 400-epoch cap '
                                                                   '(only to test the code)')
    parser.add_argument('--triplets-folder', default=os.path.join(release_paths.release_root(), 'external',
                                                                  'spose_triplets'))
    parser.add_argument('--resamples', type=int, default=None, help='bootstrap resamples per refit '
                                                                    '(default 2000)')
    parser.add_argument('--published-resamples', type=int, default=None,
                        help='bootstrap resamples for the published embedding (default 10000)')
    parser.add_argument('--skip-long', action='store_true', help='accepted from run_all.py; no effect here')
    parser.add_argument('--provided-only', action='store_true', help='accepted from run_all.py; no effect here')
    arguments = parser.parse_args()

    started = time.time()
    paths = release_paths.module_paths(__file__)
    release_root = release_paths.release_root()
    print('module A6b: SI Appendix, Section 6, embedding size (Fig. S16)')
    print('mode: every analysis runs on the saved refits in data/refits/ (fitting them again needs PyTorch and '
          'the SPoSE triplets, EXTERNAL_INPUTS.md M9: run with --refit)')
    if arguments.skip_long or arguments.provided_only:
        print('(--skip-long / --provided-only: no effect in this module; everything runs)')
    if arguments.refit:
        refit_embeddings(arguments, paths, release_root)

    for step_name, script in STEPS:
        command = [sys.executable, os.path.join(CODE_FOLDER, script)]
        if step_name == 'stats_figS16':
            if arguments.resamples is not None:
                command += ['--resamples', str(arguments.resamples)]
            if arguments.published_resamples is not None:
                command += ['--published-resamples', str(arguments.published_resamples)]
        run_logged(script, command, os.path.join(paths['logs'], step_name + '.log'), release_root)

    logged_output = LoggedOutput(os.path.join(paths['logs'], 'numbers.log'), release_root)
    sys.stdout = logged_output
    try:
        write_numbers_table(paths)
    finally:
        sys.stdout = logged_output.console
        logged_output.close()
    if arguments.refit:
        analyse_new_refits(paths, release_root)
    print('module A6b finished in %.1f minutes' % ((time.time() - started) / 60.0))


if __name__ == '__main__':
    main()
