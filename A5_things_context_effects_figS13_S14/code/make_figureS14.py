"""
make_figureS14.py: SI Appendix, Fig. S14, Kramer et al.'s (2023) quartile test of context, run on the
trial-level THINGS memorability data (SI Appendix, Section 5).

WHAT THE FIGURE SHOWS
    Kramer et al. split participants into quartiles by how many exemplars of a category they saw,
    scored each image separately in the top and the bottom quartile, and asked whether the ranking of
    images survived (a cross-quartile rank correlation compared with random splits).
    (a) That test for six higher-level THINGS categories. Blue point: the Spearman correlation between
        an image's corrected recognition in the top and in the bottom quartile. Gray point and gray
        bar: the mean and the middle 95% of the same correlation over 200 random reassignments of
        participants to quartiles.
    (b) Animals: in the same participants and images, the drop in d' from the least to the most crowded
        fifth of each image's trials (crowding = similarity of the preceding stream, ranked within
        image), for all participants and inside each quartile alone. Dashed line: how far the quartile
        split itself moves d' (bottom quartile minus top quartile). The bars have no error bars;
        outputs/tests.csv (T41) gives the participant-resample intervals of the drops.

INPUT   kramer_quartile_summary.csv (one row per category) and kramer_quartile.txt (the printed report,
        checked against the csv), from data/provided/ or outputs/regenerated/
OUTPUT  outputs/figures/FigS14_kramer_quartile.png (300 dpi)
        outputs/figure_data/FigS14a_cross_quartile_correlation.csv, FigS14b_animal_drops.csv
SEEDS   none.  RUNTIME  seconds.
Run from the repository root:  python A5_things_context_effects_figS13_S14/code/make_figureS14.py [--source provided]
"""
import argparse
import csv
import re
import sys

import matplotlib.pyplot as plt

import a5_common
import figure_style as ls

CATEGORIES = ['animal', 'food', 'clothing', 'container', 'tool', 'vehicle']
CATEGORY_LABELS = ['Animals', 'Food', 'Clothing', 'Containers', 'Tools', 'Vehicles']
OFFSET = 0.16               # observed point left of the category center, shuffled split right


def load_summary(folder):
    rows = {}
    with open(folder / 'kramer_quartile_summary.csv', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            rows[row['category']] = row
    for category in CATEGORIES:
        if category not in rows:
            sys.exit('missing category in kramer_quartile_summary.csv: ' + category)
    return rows


def check_against_report(folder, rows):
    """The animal numbers in the csv must match the printed report."""
    text = open(folder / 'kramer_quartile.txt', encoding='utf-8').read()
    animal_block = text.split('CATEGORY: ANIMAL')[1].split('CATEGORY: FOOD')[0]
    rho = float(re.search(r'Spearman rho between top- and bottom-quartile image scores: ([+-][\d.]+)',
                          animal_block).group(1))
    top_drop = float(re.search(r'top quartile\s+[\d.]+\s+[\d.]+\s+([+-][\d.]+)', animal_block).group(1))
    assert abs(rho - float(rows['animal']['rho_cross'])) < 0.0006
    assert abs(top_drop - float(rows['animal']['drop_top_quartile'])) < 0.0006


def new_figure_with_widths(width_ratios):
    """Two panels side by side at the PNAS double-column width, with unequal widths
    (panel a has six categories, panel b three bars)."""
    width_in = ls.WIDTHS_CM['double'] * ls.CM
    height_in = 6.6 * ls.CM
    fig, axes = plt.subplots(1, 2, figsize=(width_in, height_in), layout='constrained',
                             gridspec_kw={'width_ratios': width_ratios})
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.06, wspace=0.1)
    return fig, list(axes)


def main():
    parser = argparse.ArgumentParser(description='Fig. S14')
    parser.add_argument('--source', choices=['provided', 'regenerated'], default=None,
                        help='data/provided/ (the saved run) or outputs/regenerated/ (default: regenerated when the '
                             'trial data are on this machine)')
    args = parser.parse_args()
    folder = a5_common.report_folder(args.source or a5_common.default_source())
    rows = load_summary(folder)
    check_against_report(folder, rows)

    ls.apply_style('data')
    fig, (ax_rank, ax_drop) = new_figure_with_widths([1.9, 1])

    # ---------------- panel a: does the image ranking survive the quartile split?
    positions = ls.category_axis(ax_rank, CATEGORY_LABELS)
    panel_a_rows = []
    for index, category in enumerate(CATEGORIES):
        row = rows[category]
        observed = float(row['rho_cross'])
        shuffled_mean = float(row['null_mean'])
        shuffled_low = float(row['null_lo'])
        shuffled_high = float(row['null_hi'])
        x = positions[index]
        # shuffled split: gray point with gray capped bar over the middle 95% of 200 shuffles
        ax_rank.plot([x + OFFSET], [shuffled_mean], linestyle='none', marker='o',
                     markersize=ls.MARKER_SIZE + 1, color=ls.MODEL_GRAY, zorder=3)
        ls.error_bars(ax_rank, [x + OFFSET], [shuffled_mean], [shuffled_low], [shuffled_high],
                      color=ls.MODEL_GRAY)
        # observed split: blue point
        ax_rank.plot([x - OFFSET], [observed], linestyle='none', marker='o',
                     markersize=ls.MARKER_SIZE + 1, color=ls.DATA_BLUE, zorder=4)
        panel_a_rows.append([category, repr(observed), repr(shuffled_mean), repr(shuffled_low), repr(shuffled_high),
                             int(float(row['n_images_scored']))])
    ax_rank.set_ylim(-0.05, 0.25)
    ax_rank.set_yticks([0, 0.1, 0.2])
    ls.zero_line(ax_rank)
    ax_rank.set_ylabel('Cross-Quartile Correlation\n(Item Memorability)')
    ls.direct_label(ax_rank, 0.6, 0.225, 'Observed Split', ls.DATA_BLUE, ha='left', dx_points=0)
    ls.direct_label(ax_rank, 0.6, 0.195, 'Random Split (95% Range)', ls.MODEL_GRAY, ha='left', dx_points=0)
    ls.plain_ticks(ax_rank)
    ls.panel_label(ax_rank, 'a')

    # ---------------- panel b: animals, the context effect inside each quartile
    animal = rows['animal']
    bar_labels = ['All\nQuartiles', 'Top\nQuartile', 'Bottom\nQuartile']
    drop_values = [float(animal['drop_all_5bin']), float(animal['drop_top_quartile']),
                   float(animal['drop_bottom_quartile'])]
    ls.simple_bars(ax_drop, bar_labels, drop_values)
    split_effect = -float(animal['dprime_q_diff'])          # bottom-quartile d' minus top-quartile d'
    ls.reference_line(ax_drop, split_effect)
    # the dashed line is named in the free space above the bars, with its value
    ls.stat_text(ax_drop, 'Dashed line: top vs bottom\nquartile difference, %s' % ls.format_stat(split_effect),
                 corner='upper right')
    ax_drop.set_ylim(0, 0.8)
    ax_drop.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax_drop.set_ylabel('Drop in d′, Least to Most\nCrowded Stream (Animals)')
    ls.plain_ticks(ax_drop)
    ls.panel_label(ax_drop, 'b')

    ls.save_figure(fig, 'FigS14_kramer_quartile', str(a5_common.FIGURES))

    path = a5_common.FIGURE_DATA / 'FigS14a_cross_quartile_correlation.csv'
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['category', 'observed_spearman_rho', 'random_split_mean', 'random_split_low_2_5',
                         'random_split_high_97_5', 'images_scored'])
        writer.writerows(panel_a_rows)
    print('wrote', a5_common.relative(path))
    path = a5_common.FIGURE_DATA / 'FigS14b_animal_drops.csv'
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['bar', 'dprime_drop_least_to_most_crowded_fifth'])
        for label, value in zip(['all quartiles', 'top quartile', 'bottom quartile'], drop_values):
            writer.writerow([label, repr(value)])
        writer.writerow(['dashed line: bottom minus top quartile dprime', repr(split_effect)])
    print('wrote', a5_common.relative(path))
    print('Fig. S14 drawn from %s' % a5_common.relative(folder))


if __name__ == '__main__':
    main()
