"""
run_refit_sweep.py: step 2 of module A6b. The Fig. 5c and Fig. 6 statistics in the published
49-feature SPoSE embedding and in every saved SPoSE refit (SI Appendix, Section 6; Fig. S16).

WHAT IT DOES
For the published embedding (Hebart et al. 2020) and for every refit in the refit folder (by default
data/refits/: the 14 refits whose sparsity penalty removed features, keeping 6 to 81 of the 100
starting features), it computes with analysis_core.py
  - the Fig. 5c statistic: the Pearson correlation, across the embedding's features, between each
    feature's Spearman correlation with concept memorability and its Spearman correlation with
    summed similarity in the embedding's own features, with its two-sided parametric p
  - the six Fig. 6 correlations, 6c to 6h (6e and 6h are in the monkey IT space, which does not
    change with the embedding)
  - descriptions of each fit: the number of features, the penalty strength (lambda), the seed, the
    epoch at which training stopped, the held-out odd-one-out accuracy and cross-entropy (from the
    refit program's summary files), whether the penalty removed features (fewer than the 100
    starting features; left open, NaN, for a fit made by any command other than the four that
    made the 14 refits, since its summary file does not record its starting size), the share of
    weights below 0.1, and how closely the refit's summed similarity follows the published
    embedding's (Pearson r across the 1,854 concepts)

INPUTS
  data/refits/spose_lam<lambda>_seed<seed>_d<features>.txt   the saved refits (1,854 x features)
  data/refits/*_summary.json      the refit program's summaries (lambda, features kept, held-out fit,
                                  stopping epoch, seed, file name), one file per refit command
  outputs/image_set_concept_memorability.csv (step 1, extract_behavior.py)
  and the THINGS inputs of analysis_core.py (shared/things/, 06_.../data/sampled_image_sets.mat)

OUTPUT
  outputs/refit_sweep.csv: one row per embedding (the published embedding first, then the refits
  from fewest to most features); numbers written as the shortest decimal that reads back as the
  same double, NaN where a value does not apply

RUN (from the repository root; a few seconds)
  python A6b_embedding_size_figS16/code/run_refit_sweep.py
  options: --refit-folder <folder>   analyze the refits in another folder, searched with its
                                     subfolders (run_module.py --refit uses it for new refits)
           --output <file>           write the table to another file
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True     # no __pycache__ folders inside the release
sys.path.insert(0, os.path.join(CODE_FOLDER, '..', '..', 'shared', 'python'))
import release_paths     # noqa: E402  (shared helper: release folders)
import analysis_core   # noqa: E402  (this module: the Fig. 5c and Fig. 6 analyses)

# Every refit of Fig. S16 starts from 100 features: the commands that made them pass --init-dim 100
# (see code/run_module.py, REFIT_COMMANDS). A refit that still has 100 features never had a feature
# removed by its sparsity penalty.
STARTING_FEATURES = 100

# The refit program writes one summary file per command, named after the command's --tag. The
# source column labels the fits of lamsweep and lamfill (seed 0) "refit" and the two lambda 0.006
# fits with seeds 1 and 2 "control", as in data/paper_run/all_fits.csv.
SUMMARY_FILE_LABELS = {
    'lamsweep_summary.json': ('refit', 'lamsweep'),
    'lamfill_summary.json': ('refit', 'lamfill'),
    'sparse_s1_summary.json': ('control', 'sparsityctl'),
    'sparse_s2_summary.json': ('control', 'sparsityctl'),
}
PUBLISHED_NAME = 'published 49-d'

COLUMNS = ['embedding', 'source', 'arm', 'file', 'number_of_features', 'penalty_lambda', 'seed',
           'starting_features', 'penalty_removed_features', 'stop_epoch', 'held_out_accuracy',
           'held_out_cross_entropy', 'share_of_weights_below_0_1', 'summed_similarity_r_with_published',
           'fig5c_r', 'fig5c_p', 'fig5c_number_of_features', 'rho_summed_similarity_memorability',
           'fig6c_maintained_vs_things_r', 'fig6d_maintained_human_r', 'fig6e_maintained_monkey_r',
           'fig6f_changed_vs_things_r', 'fig6g_changed_human_r', 'fig6h_changed_monkey_r']

# (column of the table, key returned by analysis_core.fig6_statistics)
FIG6_COLUMNS = [('fig6c_maintained_vs_things_r', 'maintained_vs_things'),
                ('fig6d_maintained_human_r', 'maintained_human'),
                ('fig6e_maintained_monkey_r', 'maintained_monkey'),
                ('fig6f_changed_vs_things_r', 'changed_vs_things'),
                ('fig6g_changed_human_r', 'changed_human'),
                ('fig6h_changed_monkey_r', 'changed_monkey')]


def exact_text(value):
    """Numbers as the shortest decimal that reads back as the same double; NaN for missing values."""
    if value is None:
        return 'NaN'
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isnan(value):
            return 'NaN'
        return repr(value)
    return str(value)


def read_summaries(refit_folder):
    """{embedding file name: its summary record, plus the source and arm labels of
    data/paper_run/all_fits.csv} from every *_summary.json in the refit folder and its subfolders."""
    records = {}
    pattern = os.path.join(refit_folder, '**', '*_summary.json')
    for path in sorted(glob.glob(pattern, recursive=True)):
        file_name = os.path.basename(path)
        source, arm = SUMMARY_FILE_LABELS.get(file_name, ('refit', os.path.basename(os.path.dirname(path))))
        with open(path) as handle:
            for record in json.load(handle):
                record = dict(record)
                record['source'] = source
                record['arm'] = arm
                # True for the four commands that made the 14 refits (all start from 100 features)
                record['made_by_refit_commands'] = file_name in SUMMARY_FILE_LABELS
                records[record['file']] = record
    return records


def analyse_embedding(name, embedding, inputs, set_counts, set_memorability, published_summed_similarity):
    """All statistics of one embedding as a dict of table columns (fit descriptions filled later)."""
    fig5c = analysis_core.fig5c_statistic(embedding, inputs['memorability'], inputs['image_counts'])
    fig6 = analysis_core.fig6_statistics(embedding, inputs['monkey_features'], inputs['memorability'],
                                         set_counts, set_memorability)
    if published_summed_similarity is None:
        published_summed_similarity = fig5c['summed_similarity']
    row = dict(embedding=name,
               number_of_features=int(embedding.shape[1]),
               share_of_weights_below_0_1=analysis_core.share_of_small_weights(embedding),
               summed_similarity_r_with_published=float(np.corrcoef(fig5c['summed_similarity'],
                                                                    published_summed_similarity)[0, 1]),
               fig5c_r=fig5c['r'], fig5c_p=fig5c['p'], fig5c_number_of_features=fig5c['number_of_points'],
               rho_summed_similarity_memorability=fig5c['rho_summed_similarity_memorability'])
    for column, key in FIG6_COLUMNS:
        row[column] = fig6[key][0]          # r (fig6_statistics returns r and p)
    return row, fig5c['summed_similarity']


def main():
    paths = release_paths.module_paths(__file__)
    parser = argparse.ArgumentParser(description='Fig. 5c and Fig. 6 statistics in the published '
                                                 'embedding and the saved SPoSE refits')
    parser.add_argument('--refit-folder', default=os.path.join(paths['data'], 'refits'),
                        help='folder with the refits (spose_*.txt) and their *_summary.json files')
    parser.add_argument('--output', default=os.path.join(paths['outputs'], 'refit_sweep.csv'))
    arguments = parser.parse_args()

    print('step 2: Fig. 5c and Fig. 6 statistics in each embedding')
    inputs = analysis_core.load_things_inputs()
    set_counts = analysis_core.load_image_set_counts()
    set_memorability = analysis_core.load_image_set_memorability(paths['outputs'], inputs['concepts'])

    rows = []
    published_row, published_summed_similarity = analyse_embedding(
        PUBLISHED_NAME, inputs['published_embedding'], inputs, set_counts, set_memorability, None)
    published_row.update(source='published', arm='', file='shared/things/spose_embedding_49d_sorted.txt')
    rows.append(published_row)

    summaries = read_summaries(arguments.refit_folder)
    refit_files = sorted(glob.glob(os.path.join(arguments.refit_folder, '**', 'spose_*.txt'), recursive=True))
    print('  %d refit files in %s' % (len(refit_files), os.path.relpath(arguments.refit_folder,
                                                                         release_paths.release_root())))
    refit_rows = []
    for path in refit_files:
        file_name = os.path.basename(path)
        embedding = analysis_core.load_embedding(path)
        if embedding.shape[1] == 0:
            print('  skipping %s: the penalty removed every feature' % file_name)
            continue
        if file_name not in summaries:
            raise SystemExit('no summary record for %s in %s' % (file_name, arguments.refit_folder))
        summary = summaries[file_name]
        name = file_name.replace('spose_', '').replace('.txt', '')
        row, _ = analyse_embedding(name, embedding, inputs, set_counts, set_memorability,
                                   published_summed_similarity)
        if summary['made_by_refit_commands']:
            # one of the four commands of the 14 refits, which start from 100 features
            starting_features = STARTING_FEATURES
            penalty_removed_features = bool(embedding.shape[1] < STARTING_FEATURES)
        else:
            # a fit made by another command: its summary file does not record the starting size,
            # so whether the penalty removed features is left open (NaN in the table)
            starting_features = None
            penalty_removed_features = None
        row.update(source=summary['source'], arm=summary['arm'], file=file_name,
                   penalty_lambda=float(summary['lmbda']), seed=int(summary['seed']),
                   starting_features=starting_features,
                   penalty_removed_features=penalty_removed_features,
                   stop_epoch=int(summary['epoch']), held_out_accuracy=float(summary['val_acc']),
                   held_out_cross_entropy=float(summary['val_ce']))
        refit_rows.append(row)
    refit_rows.sort(key=lambda row: (row['number_of_features'], row['embedding']))
    rows.extend(refit_rows)

    print('  %-22s %4s %8s %7s %8s %10s | %7s %7s %7s %7s %7s %7s'
          % ('embedding', 'k', 'lambda', 'pruned', 'Fig 5c r', 'p', '6c', '6d', '6e', '6f', '6g', '6h'))
    for row in rows:
        if row.get('penalty_lambda') is None:          # the published embedding is not refit here
            lambda_text = '--'
            pruned_text = '--'
        else:
            lambda_text = '%.4g' % row['penalty_lambda']
            if row['penalty_removed_features'] is None:      # starting size not known (other commands)
                pruned_text = '?'
            elif row['penalty_removed_features']:
                pruned_text = 'yes'
            else:
                pruned_text = 'no'
        print('  %-22s %4d %8s %7s %+8.4f %10.3g | %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f %+7.3f'
              % (row['embedding'], row['number_of_features'], lambda_text, pruned_text, row['fig5c_r'],
                 row['fig5c_p'], row['fig6c_maintained_vs_things_r'], row['fig6d_maintained_human_r'],
                 row['fig6e_maintained_monkey_r'], row['fig6f_changed_vs_things_r'],
                 row['fig6g_changed_human_r'], row['fig6h_changed_monkey_r']))

    os.makedirs(os.path.dirname(os.path.abspath(arguments.output)), exist_ok=True)
    with open(arguments.output, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([exact_text(row.get(column)) for column in COLUMNS])
    print('  wrote', os.path.relpath(os.path.abspath(arguments.output), release_paths.release_root()))


if __name__ == '__main__':
    main()
