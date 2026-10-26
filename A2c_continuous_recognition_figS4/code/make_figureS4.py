"""
make_figureS4.py: Fig. S4, the REM model run through continuous recognition on the THINGS trial
sequences, beside the THINGS data (SI Appendix, Section 2).

WHAT THE FIGURE SHOWS
  Two panels, side by side. In both, the x-axis is the similarity of the items that came before a
  trial, ranked within each item into 20 bins (bin 1 = least similar preceding items, bin 20 = most
  similar). Each item's own trials are compared with each other, so nothing about the item itself can
  produce a slope.
  (a) Change in d' from bin 1. Dark gray points with black error bars = the THINGS data (Kramer et al.,
      2023): each bin's d' minus the bin-1 d', with that bin's 95 percent participant-bootstrap limits
      shifted by the same amount. Gray lines = the REM model at two storage probabilities: solid =
      standard storage (0.30), dashed = the storage probability that brings the model's overall d' near
      the participants' (0.75, "Matched Accuracy").
  (b) False alarm rate in the same bins. Red points with black error bars = the THINGS data; the same
      two gray model lines.

INPUTS (read and checked by figs4_inputs.py)
  model lines: data/provided/continuous_bins.npz (the saved run; default) or outputs/continuous_bins.npz
               (a run of continuous_rem.py in this release, with --model-run regenerated)
  data points: ../A5_things_context_effects_figS13_S14/data/provided/sdt_ctx_top10_target.npz
               (provided output of module A5; regenerating it needs the Kramer et al. trial file, M1)
OUTPUTS
  outputs/figures/FigS4_continuous_recognition.png (300 dpi)
  outputs/figure_data/FigS4a.csv, FigS4b.csv (the plotted numbers)
No random numbers. Runtime: a few seconds.

RUN (from the repository root)
  python A2c_continuous_recognition_figS4/code/make_figureS4.py
  python A2c_continuous_recognition_figS4/code/make_figureS4.py --model-run regenerated
"""
import argparse
import csv
import io
import logging
import os
import sys

import numpy as np

sys.dont_write_bytecode = True   # write no __pycache__ folder into shared/python or code/ (set before the imports below)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402
import figure_style as ls  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figs4_inputs  # noqa: E402

PATHS = release_paths.module_paths(__file__)
FIGURE_NAME = 'FigS4_continuous_recognition'


def write_table(file_name, columns, rows):
    path = os.path.join(PATHS['figure_data'], file_name)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(['%.6g' % value if isinstance(value, float) else value for value in row])
    print('wrote %s' % os.path.relpath(path, release_paths.release_root()))


def main():
    parser = argparse.ArgumentParser(description='Fig. S4 (continuous recognition, model and THINGS data).')
    parser.add_argument('--model-run', choices=['provided', 'regenerated', 'auto'], default='provided',
                        help="model lines from the saved run (data/provided/, default), a new run of "
                             "continuous_rem.py (outputs/), or auto")
    arguments = parser.parse_args()

    model = figs4_inputs.model_run(arguments.model_run)
    data = figs4_inputs.things_data()
    print('model lines: %s' % model['label'])
    print('data points: provided output of module A5 (%s)' % figs4_inputs.relative(data['files']['npz']))

    data_series = {
        'name': 'THINGS data',
        'dprime': data['dprime'],
        'dprime_low': data['dlo'],
        'dprime_high': data['dhi'],
        'false_alarm': data['F'],
        'false_alarm_low': data['Flo'],
        'false_alarm_high': data['Fhi'],
    }
    model_series_list = []
    for point_name, plain_name, line_style in [('standard', 'Model, standard storage', '-'),
                                               ('accuracy-matched', 'Model, matched accuracy', '--')]:
        model_series_list.append({
            'name': plain_name,
            'point': point_name,
            'storage': model[point_name]['storage'],
            'dprime': model[point_name]['dprime'],
            'false_alarm': model[point_name]['F'],
            'line_style': line_style,
        })

    number_of_bins = len(data_series['dprime'])
    for model_series in model_series_list:
        if len(model_series['dprime']) != number_of_bins:
            sys.exit('make_figureS4.py: the model and the data must have the same bins')
    bin_numbers = np.arange(1, number_of_bins + 1)

    # --------------------------------------------------------------------------- the figure
    ls.apply_style('model')          # box and light grid, like the paper's REM figures (Fig. 1)
    fig, (axis_dprime, axis_fa) = ls.new_figure(n_panels=2, width='double', height_cm=7.2)

    data_dprime_color = ls.MEMORY_DARK   # d' shown next to a false alarm panel
    model_line_width = 1.2

    # ---- panel a: change in d' from bin 1 ----
    for model_series in model_series_list:
        change = model_series['dprime'] - model_series['dprime'][0]
        axis_dprime.plot(bin_numbers, change, color=ls.MODEL_GRAY, linestyle=model_series['line_style'],
                         linewidth=model_line_width, zorder=5, clip_on=False)
    bin1_dprime = data_series['dprime'][0]
    ls.line_series(axis_dprime, bin_numbers, data_series['dprime'] - bin1_dprime, color=data_dprime_color,
                   low=data_series['dprime_low'] - bin1_dprime, high=data_series['dprime_high'] - bin1_dprime)
    axis_dprime.set_ylim(-1.2, 0.1)
    axis_dprime.set_yticks([-1.2, -0.8, -0.4, 0])
    axis_dprime.set_ylabel('Change in d′')

    # direct labels, placed in empty parts of the panel
    ls.direct_label(axis_dprime, 11, -0.26, 'THINGS Data', data_dprime_color, ha='left', va='bottom', dx_points=0)
    ls.direct_label(axis_dprime, 20, -0.63, 'Model,\nStandard', ls.LABEL_GRAY, ha='right', va='top',
                    dx_points=0, dy_points=-6)
    ls.direct_label(axis_dprime, 9.5, -0.62, 'Model,\nMatched\nAccuracy', ls.LABEL_GRAY, ha='right', va='top',
                    dx_points=0, dy_points=-4)

    # ---- panel b: false alarm rate ----
    for model_series in model_series_list:
        axis_fa.plot(bin_numbers, model_series['false_alarm'], color=ls.MODEL_GRAY,
                     linestyle=model_series['line_style'], linewidth=model_line_width, zorder=5, clip_on=False)
    ls.line_series(axis_fa, bin_numbers, data_series['false_alarm'], color=ls.FA_RED,
                   low=data_series['false_alarm_low'], high=data_series['false_alarm_high'])
    axis_fa.set_ylim(0, 0.06)
    axis_fa.set_yticks([0, 0.02, 0.04, 0.06])
    axis_fa.set_ylabel('False Alarm Rate')
    ls.direct_label(axis_fa, 1.5, 0.021, 'THINGS Data', ls.FA_RED, ha='left', va='bottom', dx_points=0)
    ls.direct_label(axis_fa, 14.5, 0.0175, 'Model,\nStandard', ls.LABEL_GRAY, ha='center', va='top',
                    dx_points=0, dy_points=0)
    ls.direct_label(axis_fa, 15.5, 0.045, 'Model,\nMatched\nAccuracy', ls.LABEL_GRAY, ha='right', va='bottom',
                    dx_points=0, dy_points=0)

    for axis in (axis_dprime, axis_fa):
        axis.set_xlim(0, 21)
        axis.set_xticks([1, 5, 10, 15, 20])
        axis.set_xticklabels(['1\nLeast', '5', '10', '15', '20\nMost'])
        axis.set_xlabel('Similarity of Preceding Items\n(Bin Within Item)')
        ls.plain_ticks(axis)
    ls.panel_label(axis_dprime, 'a')
    ls.panel_label(axis_fa, 'b')

    # Each save lets matplotlib settle the text positions once more. The PDF and SVG versions are drawn
    # to memory (no files) before the PNG is saved, and this order gives the final layout of Fig. S4.
    logging.getLogger('fontTools').setLevel(logging.ERROR)   # quiet notes from the font subsetting of the PDF
    fig.savefig(io.BytesIO(), format='pdf')
    fig.savefig(io.BytesIO(), format='svg')
    ls.save_figure(fig, FIGURE_NAME, PATHS['figures'])

    # --------------------------------------------------------------------------- the plotted numbers
    standard_series = model_series_list[0]
    matched_series = model_series_list[1]
    rows_a = []
    rows_b = []
    for index in range(number_of_bins):
        rows_a.append([int(bin_numbers[index]),
                       float(data_series['dprime'][index]),
                       float(data_series['dprime'][index] - bin1_dprime),
                       float(data_series['dprime_low'][index] - bin1_dprime),
                       float(data_series['dprime_high'][index] - bin1_dprime),
                       float(standard_series['dprime'][index] - standard_series['dprime'][0]),
                       float(matched_series['dprime'][index] - matched_series['dprime'][0])])
        rows_b.append([int(bin_numbers[index]),
                       float(data_series['false_alarm'][index]),
                       float(data_series['false_alarm_low'][index]),
                       float(data_series['false_alarm_high'][index]),
                       float(standard_series['false_alarm'][index]),
                       float(matched_series['false_alarm'][index])])
    write_table('FigS4a.csv',
                ['bin', 'data_dprime', 'data_change_in_dprime', 'data_error_bar_low', 'data_error_bar_high',
                 'model_storage_%.2f_change_in_dprime' % standard_series['storage'],
                 'model_storage_%.2f_change_in_dprime' % matched_series['storage']], rows_a)
    write_table('FigS4b.csv',
                ['bin', 'data_false_alarm_rate', 'data_error_bar_low', 'data_error_bar_high',
                 'model_storage_%.2f_false_alarm_rate' % standard_series['storage'],
                 'model_storage_%.2f_false_alarm_rate' % matched_series['storage']], rows_b)


if __name__ == '__main__':
    main()
