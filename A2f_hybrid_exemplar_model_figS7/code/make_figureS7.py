"""
make_figureS7.py: SI Appendix, Fig. S7, the hybrid-similarity exemplar model and the generalized
context model (GCM) beside REM.

Panels (as in the caption of SI Appendix, Fig. S7):
  a  Spearman correlation between item memorability (hit rate minus false alarm rate) and
     crowding (the negated mean distance to the other items, the summed-similarity measure of these
     simulations) across 10,000 simulated experiments
  b  change in d′ for critical items studied with 0, 2, 4, 8 or 16 of their close neighbors, each
     tested against its nearest neighbor as the foil
  c  change in hit rate (circles, solid lines) and false alarm rate (squares, dashed lines) on the
     same lists
Colors as in the caption: red REM, gray GCM (gamma 0), green hybrid model (gamma 2, rarity from
the participant's own study list). The changes are taken from the 0-neighbor list of each model.

Inputs
  outputs/core.json   the run of 01_core.py (the saved run data/paper_run/core.json when
                      outputs/core.json does not exist, said on screen): variants 'GCM' and
                      'Hybrid GCM, list'
  REM (the saved REM runs of modules A2a and A2d, read through figS7_inputs.py):
      ../A2a_rem_parameter_sweep_figS2/data/paper_run/rem_parameter_sweep.csv, the row labeled
      'paper values', at the Fig. 1 values (a)
      ../A2d_list_context_figS5/data/paper_run/list_designed.txt, CELL 2, at the Fig. 1 values (b, c;
      the rates there are printed to 3 decimals, and the REM changes are taken between those values)
Outputs
  outputs/figures/FigS7_hybrid_exemplar_model.png (300 dpi)
  outputs/figure_data/FigS7a.csv, FigS7b.csv, FigS7c.csv (the plotted numbers)
Run from the repository root:  python A2f_hybrid_exemplar_model_figS7/code/make_figureS7.py
"""
import sys
sys.dont_write_bytecode = True
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

CODE_FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_FOLDER)
sys.path.insert(0, os.path.join(CODE_FOLDER, '..', '..', 'shared', 'python'))
import release_paths
import figure_style as ls
import figS7_inputs as inputs

PATHS = release_paths.module_paths(__file__)
REM_RED = (0.70, 0.13, 0.26)         # REM (the caption's red)
GCM_GRAY = ls.MODEL_GRAY             # the generalized context model (the caption's gray)
HYBRID_GREEN = (0.18, 0.58, 0.27)    # the hybrid model (the caption's green)
NEIGHBOR_TICKS = [0, 4, 8, 16]
REM_SWEEP_SOURCE = 'A2a_rem_parameter_sweep_figS2/data/paper_run/rem_parameter_sweep.csv (paper values)'
REM_DESIGNED_SOURCE = 'A2d_list_context_figS5/data/paper_run/list_designed.txt (CELL 2, paper values)'


def main():
    # ---- the numbers to plot ----
    results, core_mode = inputs.model_results()
    core_source = 'outputs/core.json' if core_mode == 'rerun' else 'data/paper_run/core.json'
    rem = inputs.rem_basic()
    rem_designed = inputs.rem_designed_lists()
    gcm_designed = inputs.designed_table(results[inputs.GCM_MODEL])
    hybrid_designed = inputs.designed_table(results[inputs.HYBRID_MODEL])

    # one table per model for panels b and c: neighbors, hit rate, false alarm rate, d′
    model_lists = [
        ('REM', REM_RED, pd.DataFrame(dict(similar=rem_designed['similar'], hit=rem_designed['hit'],
                                           fa=rem_designed['false_alarm'], dprime=rem_designed['dprime'])),
         REM_DESIGNED_SOURCE),
        ('GCM', GCM_GRAY, gcm_designed[['similar', 'hit', 'fa', 'dprime']], core_source + ' (GCM)'),
        ('Hybrid', HYBRID_GREEN, hybrid_designed[['similar', 'hit', 'fa', 'dprime']], core_source + ' (Hybrid GCM, list)'),
    ]
    bar_rows = [dict(model='REM', spearman_rho=rem['rho'], source=REM_SWEEP_SOURCE),
                dict(model='GCM', spearman_rho=results[inputs.GCM_MODEL]['A']['rho_item'], source=core_source + ' (GCM)'),
                dict(model='Hybrid', spearman_rho=results[inputs.HYBRID_MODEL]['A']['rho_item'],
                     source=core_source + ' (Hybrid GCM, list)')]

    # ---- the plotted numbers, saved as tables ----
    pd.DataFrame(bar_rows).to_csv(os.path.join(PATHS['figure_data'], 'FigS7a.csv'), index=False)
    line_rows_b = []
    line_rows_c = []
    for name, _, table, source in model_lists:
        for position in range(len(table)):
            line_rows_b.append(dict(model=name, n_similar=int(table['similar'].iloc[position]),
                                    dprime=float(table['dprime'].iloc[position]),
                                    dprime_change=float(table['dprime'].iloc[position] - table['dprime'].iloc[0]),
                                    source=source))
            line_rows_c.append(dict(model=name, n_similar=int(table['similar'].iloc[position]),
                                    hit_rate=float(table['hit'].iloc[position]),
                                    hit_change=float(table['hit'].iloc[position] - table['hit'].iloc[0]),
                                    false_alarm_rate=float(table['fa'].iloc[position]),
                                    false_alarm_change=float(table['fa'].iloc[position] - table['fa'].iloc[0]),
                                    source=source))
    panel_b = pd.DataFrame(line_rows_b)
    panel_c = pd.DataFrame(line_rows_c)
    panel_b.to_csv(os.path.join(PATHS['figure_data'], 'FigS7b.csv'), index=False)
    panel_c.to_csv(os.path.join(PATHS['figure_data'], 'FigS7c.csv'), index=False)

    # ---- the figure ----
    ls.apply_style('model')
    fig, axes = ls.new_figure(n_panels=3, width='double')

    # a: the item-level correlation in the three models
    ax = axes[0]
    names = [row['model'] for row in bar_rows]
    values = [row['spearman_rho'] for row in bar_rows]
    ls.simple_bars(ax, names, values, colors=[REM_RED, GCM_GRAY, HYBRID_GREEN])
    ls.zero_line(ax)
    for position, value in enumerate(values, start=1):
        if value < 0:
            ls.direct_label(ax, position, value, ls.format_stat(value), ha='center', va='top', dx_points=0, dy_points=-3)
        else:
            ls.direct_label(ax, position, value, ls.format_stat(value), ha='center', va='bottom', dx_points=0, dy_points=3)
    ax.set_ylim(-0.75, 0.45)
    ax.set_yticks([-0.6, -0.3, 0, 0.3])
    ax.set_ylabel('Memorability vs Crowding\n(Spearman ρ)')
    ls.plain_ticks(ax)
    ls.panel_label(ax, 'a')

    # b: change in d′ with the number of neighbors studied
    ax = axes[1]
    for name, color, _, _ in model_lists:
        rows = panel_b[panel_b['model'] == name]
        ax.plot(rows['n_similar'], rows['dprime_change'], color=color, linewidth=ls.LINE_WIDTH, marker='o',
                markersize=ls.MARKER_SIZE, markerfacecolor=color, markeredgecolor=color, zorder=3)
        ls.direct_label(ax, rows['n_similar'].iloc[-1], rows['dprime_change'].iloc[-1], name, color=color,
                        ha='left', dx_points=5)
    ls.zero_line(ax)
    ax.set_xlim(-1.2, 23)
    ax.set_xticks(NEIGHBOR_TICKS)
    ax.set_xlabel('Neighbors Studied')
    ax.set_ylabel('Change in d′')
    ls.plain_ticks(ax)
    ls.panel_label(ax, 'b')

    # c: change in hit rate (solid, circles) and false alarm rate (dashed, squares)
    ax = axes[2]
    for name, color, _, _ in model_lists:
        rows = panel_c[panel_c['model'] == name]
        ax.plot(rows['n_similar'], rows['hit_change'], color=color, linewidth=ls.LINE_WIDTH, linestyle='-',
                marker='o', markersize=ls.MARKER_SIZE, markerfacecolor=color, markeredgecolor=color, zorder=3)
        ax.plot(rows['n_similar'], rows['false_alarm_change'], color=color, linewidth=ls.LINE_WIDTH, linestyle='--',
                marker='s', markersize=ls.MARKER_SIZE - 0.5, markerfacecolor=color, markeredgecolor=color, zorder=3)
    ls.zero_line(ax)
    ax.set_xlim(-1.2, 17.2)
    ax.set_ylim(-0.008, 0.14)
    ax.set_xticks(NEIGHBOR_TICKS)
    ax.set_yticks([0, 0.05, 0.1])
    ax.set_xlabel('Neighbors Studied')
    ax.set_ylabel('Change in Rate')
    # which lines are which: false alarms above (upper left, empty), hits below (lower right, empty)
    ls.direct_label(ax, 0.0, 0.128, 'False Alarms\n(dashed, squares)', color=ls.LABEL_GRAY, ha='left', va='top',
                    dx_points=0)
    ls.direct_label(ax, 16.8, 0.003, 'Hits\n(solid, circles)', color=ls.LABEL_GRAY, ha='right', va='bottom',
                    dx_points=0)
    ls.plain_ticks(ax)
    ls.panel_label(ax, 'c')

    ls.save_figure(fig, 'FigS7_hybrid_exemplar_model', PATHS['figures'])
    plt.close(fig)
    print('figure data: outputs/figure_data/FigS7a.csv, FigS7b.csv, FigS7c.csv (model results: %s)' % core_source)


if __name__ == '__main__':
    main()
