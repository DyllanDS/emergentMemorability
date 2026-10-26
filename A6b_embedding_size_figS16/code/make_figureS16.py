"""
make_figureS16.py: step 5 of module A6b. Fig. S16: the Fig. 5c relationship holds in embeddings refit
with 6 to 81 features.

What the figure shows
---------------------
One panel. Each blue point is one refit of the SPoSE similarity embedding to the THINGS odd-one-out
judgments (Hebart et al. 2023), starting from 100 features and pruned by its sparsity penalty to
between 6 and 81 features. x = the number of features the refit kept. y = the Fig. 5c statistic
computed in that refit: the correlation, across features, between each feature's correlation with
memorability and its correlation with summed similarity. The larger black point is the published
49-feature embedding the paper uses (Hebart et al. 2020). Black dashed line: zero.

The refits drawn are the rows with penalty_bound = True: the 14 refits whose sparsity penalty
removed features.

INPUT    outputs/all_fits.csv (step 3; columns used: name, k, init_dim, penalty_bound, fig5c_r,
         fig5c_p, where fig5c_r and fig5c_p are the Fig. 5c statistic and its parametric p)
OUTPUTS  outputs/figures/FigS16_spose_dimensionality.png (300 dpi)
         outputs/figure_data/FigS16_spose_dimensionality.csv (the plotted numbers)
RUN (from the repository root; seconds)
  python A6b_embedding_size_figS16/code/make_figureS16.py
"""
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')

sys.dont_write_bytecode = True     # no __pycache__ folders inside the release
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths          # noqa: E402  (shared helper: release folders)
import figure_style as ls  # noqa: E402  (shared helper: the SI Appendix figure style)

PUBLISHED_NAME = 'published 49-d'
STARTING_FEATURES = 100
FIGURE_NAME = 'FigS16_spose_dimensionality'


def main():
    paths = release_paths.module_paths(__file__)
    table_file = os.path.join(paths['outputs'], 'all_fits.csv')
    if not os.path.exists(table_file):
        raise SystemExit('outputs/all_fits.csv is missing: run gather_all_fits.py first')
    print('step 5: Fig. S16')

    # ------------------------------------------------------------------------- read the refits
    refit_names = []
    refit_features = []          # number of features each pruned refit kept
    refit_correlation = []       # its Fig. 5c correlation
    refit_p = []                 # its two-sided parametric p
    published_features = None
    published_correlation = None
    published_p = None
    with open(table_file, newline='') as handle:
        for row in csv.DictReader(handle):
            if row['name'] == PUBLISHED_NAME:
                published_features = int(row['k'])
                published_correlation = float(row['fig5c_r'])
                published_p = float(row['fig5c_p'])
            elif row['penalty_bound'] == 'True':
                starting_features = int(float(row['init_dim']))
                assert starting_features == STARTING_FEATURES, 'every drawn refit starts from 100 features'
                refit_names.append(row['name'])
                refit_features.append(int(row['k']))
                refit_correlation.append(float(row['fig5c_r']))
                refit_p.append(float(row['fig5c_p']))
    assert published_correlation is not None, 'the published embedding is missing from all_fits.csv'
    number_of_refits = len(refit_features)
    number_below_zero = 0
    for correlation in refit_correlation:
        if correlation < 0:
            number_below_zero += 1

    # ------------------------------------------------------------------------- the figure
    ls.apply_style('data')
    fig, axes = ls.new_figure(n_panels=1, width='single', height_cm=7.0)
    ax = axes[0]

    ax.set_xlim(0, 90)
    ax.set_ylim(-1, 0.2)
    ls.zero_line(ax)
    # The published embedding is drawn first, so the refit with the same number of features stays
    # whole on top of it (ls.highlight_point would draw it on top at zorder 6 and hide that refit;
    # same size and color as highlight_point).
    ax.plot([published_features], [published_correlation], linestyle='none', marker='o',
            markersize=ls.MARKER_SIZE + 2.5, markerfacecolor=ls.HIGHLIGHT_BLACK,
            markeredgecolor=ls.HIGHLIGHT_BLACK, zorder=2, clip_on=False)
    ls.scatter_points(ax, refit_features, refit_correlation, color=ls.DATA_BLUE, area=22, alpha=0.75)

    # direct labels
    published_label = 'Published\n%d Features' % published_features
    ls.direct_label(ax, published_features - 2, published_correlation, published_label,
                    color=ls.LABEL_GRAY, ha='right', va='top', dx_points=0, dy_points=-6)
    largest_refit = refit_features.index(max(refit_features))
    ls.direct_label(ax, refit_features[largest_refit], refit_correlation[largest_refit], 'Refits',
                    color=ls.DATA_BLUE, ha='center', va='bottom', dx_points=0, dy_points=6)
    ls.stat_text(ax, 'Below zero in %d of %d refits' % (number_below_zero, number_of_refits),
                 corner='upper right')

    ax.set_xticks([0, 30, 60, 90])
    ax.set_yticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2])
    ax.set_xlabel('Number of Features in the Embedding')
    ax.set_ylabel('Fig. 5c Correlation\n(Memorability vs Summed Similarity)')
    ls.plain_ticks(ax)

    ls.save_figure(fig, FIGURE_NAME, paths['figures'])          # prints the file it wrote

    # ------------------------------------------------------------------------- the plotted numbers
    data_file = os.path.join(paths['figure_data'], FIGURE_NAME + '.csv')
    with open(data_file, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['point', 'embedding', 'number_of_features', 'fig5c_r', 'fig5c_parametric_p'])
        writer.writerow(['published (black)', PUBLISHED_NAME, published_features, repr(published_correlation),
                         repr(published_p)])
        order = sorted(range(number_of_refits), key=lambda index: (refit_features[index], refit_names[index]))
        for index in order:
            writer.writerow(['refit (blue)', refit_names[index], refit_features[index],
                             repr(refit_correlation[index]), repr(refit_p[index])])
    print('  wrote', os.path.relpath(data_file, release_paths.release_root()))
    print('  %d refits drawn (%d to %d features), %d below zero; published %d features r = %.4f'
          % (number_of_refits, min(refit_features), max(refit_features), number_below_zero,
             published_features, published_correlation))


if __name__ == '__main__':
    main()
