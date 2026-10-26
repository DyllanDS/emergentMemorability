"""
make_figureS12.py: SI Appendix, Fig. S12 (Crowding effects across scales), from outputs/locality_profiles.csv.

Three panels (a false alarms, b hits, c d′), each the Spearman correlation across items between crowding and the
memory measure as a function of how local the crowding measure is (tau from 0.05 to the all-items limit), one line
per dataset: models dashed (the hybrid model dash-dot) in black and grays, people solid in color. Writes
outputs/figures/FigS12_crowding_across_scales.png and the plotted values to
outputs/figure_data/FigS12_crowding_across_scales.csv.

Run from the repository root (after locality_analyses.py):
    python A4a_crowding_across_scales_figS12_tableS2/code/make_figureS12.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import figure_style as ls  # noqa: E402
import release_paths  # noqa: E402

PATHS = release_paths.module_paths(__file__)
FIGURE_NAME = 'FigS12_crowding_across_scales'
ORDER = ['0.05', '0.1', '0.2', '0.4', '1', 'all']
SERIES = [  # dataset (as in outputs/locality_profiles.csv), legend label, line style
    ('REM-gen', 'REM, Generated Features', dict(color='black', linestyle='--', marker='o')),
    ('REM-SPoSE', 'REM, SPoSE Features', dict(color='0.55', linestyle='--', marker='o')),
    ('HGCM-list', 'Hybrid Model', dict(color='0.3', linestyle='-.', marker='s')),
    ('THINGS', 'THINGS', dict(color='#d5322f', linestyle='-', marker='o')),
    ('maintained', 'Maintained Set', dict(color='#2f5c9e', linestyle='-', marker='o')),
    ('changed', 'Changed Set', dict(color='#2f9e44', linestyle='-', marker='o')),
    ('reduced', 'Reduced Set', dict(color='#d87f2e', linestyle='-', marker='o')),
]
PANELS = [('a', 'fa', 'False Alarms'), ('b', 'hit', 'Hits'), ('c', 'dprime', 'd′')]
LINE_NAMES = {'--': 'dashed', '-.': 'dash-dot', '-': 'solid'}


def describe():
    """The figure file, its panels and the line style of each dataset (for outputs/numbers.csv)."""
    lines = {}
    for dataset, _, style in SERIES:
        lines.setdefault(LINE_NAMES[style['linestyle']], []).append(dataset)
    return '%s.png; panels %s; %s' % (FIGURE_NAME, ', '.join('%s %s' % (letter, title) for letter, _, title in PANELS),
                                      '; '.join('%s: %s' % (name, ', '.join(sets)) for name, sets in lines.items()))


def main():
    prof = pd.read_csv(os.path.join(PATHS['outputs'], 'locality_profiles.csv'), dtype={'tau': str})
    prof = prof[prof.scope == 'all']
    ls.apply_style('data')
    fig, axes = ls.new_figure(n_panels=3, width='double', height_cm=8.5)
    plotted = []
    for ax, (letter, measure, title) in zip(axes, PANELS):
        for dataset, label, style in SERIES:
            sub = prof[(prof.dataset == dataset) & (prof.measure == measure)].set_index('tau').reindex(ORDER)
            ax.plot(range(len(ORDER)), sub.rho, markersize=3.5, linewidth=1.2, label=label, **style)
            plotted += [dict(panel=letter, measure=measure, dataset=dataset, tau=tau, rho=rho)
                        for tau, rho in zip(ORDER, sub.rho)]
        ax.axhline(0, color='0.6', linewidth=0.8, linestyle=':', zorder=1)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(ORDER)
        ax.set_xlim(-0.3, len(ORDER) - 0.7)
        ax.set_ylim(-0.8, 0.8)
        ax.set_yticks([-0.5, 0, 0.5])
        ls.plain_ticks(ax)
        ax.set_title(title)
        ls.panel_label(ax, letter)
    axes[0].set_ylabel('Correlation With Crowding (ρ)')
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    axes[1].set_xlabel('Kernel Falloff τ (Local to All Items)')
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='outside lower center', ncols=3, frameon=False, handlelength=2.4,
               columnspacing=1.5)
    ls.save_figure(fig, FIGURE_NAME, PATHS['figures'])
    plt.close(fig)
    pd.DataFrame(plotted).to_csv(os.path.join(PATHS['figure_data'], FIGURE_NAME + '.csv'), index=False)
    print('figure data: outputs/figure_data/%s.csv' % FIGURE_NAME)


if __name__ == '__main__':
    main()
