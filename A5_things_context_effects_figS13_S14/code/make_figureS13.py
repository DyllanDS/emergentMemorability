"""
make_figureS13.py: SI Appendix, Fig. S13, the THINGS trial-level context analysis (SI Appendix, Section 5).

WHAT THE FIGURE SHOWS
    The same image is remembered less well when the images shown before it, in that participant's own
    stream, were more similar to it. Each image's trials are ranked into 20 bins by how similar the
    preceding images were (bin 1 = least similar, bin 20 = most similar), so every bin holds the same
    images and nothing intrinsic to an image can produce a slope.

    Two versions of every curve:
      open circles    whole stream: the bins as ranked, which carry both stream length and the
                      similarity of what was shown
      filled circles  serial position and lag held constant: similarity expressed relative to what is
                      typical at that point in the session, and each bin re-weighted to the same mix of
                      serial positions (and lags, for repeats)

    (a) d' across bins, dark gray.  (b) Hit rate across bins, blue.  (c) False alarm rate, red.
    Error bars: 95% intervals over 400 participant resamples, as the analysis scripts saved them.

INPUT   sdt_ctx_top10_target.npz     whole-stream curves (H, F, dprime, crit and their interval ends)
        sdt_by_context_adjusted.npz  held-constant curves, the bin 1 minus bin 20 contrast
        sdt_by_context_adjusted.txt  the printed report of the same run (checked against the npz)
        from data/provided/ (the saved run) or outputs/regenerated/ (a run from the trial file)
OUTPUT  outputs/figures/FigS13_things_trial_level_context.png (300 dpi)
        outputs/figure_data/FigS13a_dprime.csv, FigS13b_hit_rate.csv, FigS13c_false_alarm_rate.csv
SEEDS   none.  RUNTIME  seconds.
Run from the repository root:  python A5_things_context_effects_figS13_S14/code/make_figureS13.py [--source provided]
"""
import argparse
import csv
import re
import sys

import numpy as np

import a5_common
import figure_style as ls

SMALL_MARKER = 3.6          # 20 points per series, so the markers are a little smaller than usual
OPEN_FACE = 'white'          # whole-stream series: open circles


# ----------------------------------------------------------------------------- loading
def load_curves(folder):
    """Return two dictionaries (whole stream, held constant), each holding for 'dprime', 'hit', 'fa' and
    'criterion' a tuple (values, lower interval end, upper interval end), one value per bin."""
    whole_file = np.load(folder / 'sdt_ctx_top10_target.npz')
    held_file = np.load(folder / 'sdt_by_context_adjusted.npz')

    # The adjusted file carries its own copy of the whole-stream curves; they must agree.
    pairs_to_check = [('H_unadj', 'H'), ('F_unadj', 'F'), ('dprime_unadj', 'dprime')]
    for adjusted_key, whole_key in pairs_to_check:
        if not np.allclose(held_file[adjusted_key], whole_file[whole_key], atol=1e-6):
            sys.exit('whole-stream curves disagree between the two files: ' + adjusted_key)

    whole_stream = {}
    held_constant = {}
    for measure, value_key, low_key, high_key in [('dprime', 'dprime', 'dlo', 'dhi'),
                                                  ('hit', 'H', 'Hlo', 'Hhi'),
                                                  ('fa', 'F', 'Flo', 'Fhi'),
                                                  ('criterion', 'crit', 'clo', 'chi')]:
        whole_stream[measure] = (whole_file[value_key], whole_file[low_key], whole_file[high_key])
        held_constant[measure] = (held_file[value_key], held_file[low_key], held_file[high_key])
    return whole_stream, held_constant, held_file


def check_against_printed_report(folder, held_constant):
    """The held-constant d' at bins 1 and 20 must match the printed report (STEP 2 table)."""
    report_text = open(folder / 'sdt_by_context_adjusted.txt', encoding='utf-8').read()
    step2_text = report_text.split('STEP 2.')[1].split('STEP 3.')[0]
    bin_rows = re.findall(r'^\s+(\d+)\s+([\d.]+) \[.*?\]\s+([\d.]+) \[.*?\]\s+([\d.]+) \[', step2_text, re.M)
    first_row, last_row = bin_rows[0], bin_rows[19]
    assert first_row[0] == '1' and last_row[0] == '20'
    assert abs(float(first_row[3]) - held_constant['dprime'][0][0]) < 0.0006
    assert abs(float(last_row[3]) - held_constant['dprime'][0][19]) < 0.0006
    assert abs(float(first_row[1]) - held_constant['hit'][0][0]) < 0.00006
    assert abs(float(first_row[2]) - held_constant['fa'][0][0]) < 0.00006


# ----------------------------------------------------------------------------- drawing
def draw_series(ax, bins, values, low, high, color, filled):
    """One series across the 20 bins: thin line, circles (filled or open), black capped error bars."""
    if filled:
        face_color = color
    else:
        face_color = OPEN_FACE
    ls.error_bars(ax, bins, values, low, high)
    ax.plot(bins, values, color=color, linewidth=ls.LINE_WIDTH, marker='o', markersize=SMALL_MARKER,
            markerfacecolor=face_color, markeredgecolor=color, markeredgewidth=0.9, zorder=5,
            clip_on=False)


def bin_axis(ax):
    ax.set_xlim(0, 21)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel('Similarity of Preceding Items\n(Bin Within Image)')


def write_panel_data(name, bins, whole, held):
    """The plotted numbers of one panel: per bin, both series with their interval ends."""
    path = a5_common.FIGURE_DATA / name
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['bin', 'whole_stream', 'whole_stream_low', 'whole_stream_high',
                         'held_constant', 'held_constant_low', 'held_constant_high'])
        for index in range(len(bins)):
            writer.writerow([int(bins[index]), repr(float(whole[0][index])), repr(float(whole[1][index])),
                             repr(float(whole[2][index])), repr(float(held[0][index])), repr(float(held[1][index])),
                             repr(float(held[2][index]))])
    print('wrote', a5_common.relative(path))


def main():
    parser = argparse.ArgumentParser(description='Fig. S13')
    parser.add_argument('--source', choices=['provided', 'regenerated'], default=None,
                        help='data/provided/ (the saved run) or outputs/regenerated/ (default: regenerated when the '
                             'trial data are on this machine)')
    args = parser.parse_args()
    folder = a5_common.report_folder(args.source or a5_common.default_source())
    whole_stream, held_constant, held_file = load_curves(folder)
    check_against_printed_report(folder, held_constant)
    bins = np.arange(1, 21)

    ls.apply_style('data')
    fig, axes = ls.new_figure(n_panels=3, width='double', height_cm=6.4)
    ax_dprime, ax_hit, ax_fa = axes

    # Panel a: d'
    for curves, filled in [(whole_stream, False), (held_constant, True)]:
        values, low, high = curves['dprime']
        draw_series(ax_dprime, bins, values, low, high, ls.MEMORY_DARK, filled)
    ax_dprime.set_ylim(2.6, 3.3)
    ax_dprime.set_yticks([2.6, 2.8, 3.0, 3.2])
    ax_dprime.set_ylabel('d′')
    whole_drop = whole_stream['dprime'][0][0] - whole_stream['dprime'][0][19]
    held_drop = float(held_file['contrast'])
    # each series is named on the plot together with its bin 1 minus bin 20 decline
    ls.direct_label(ax_dprime, 2.2, 3.24, 'Whole Stream, Decline %s' % ls.format_stat(whole_drop),
                    ls.MEMORY_DARK, ha='left', va='center')
    ls.direct_label(ax_dprime, 0.6, 2.815, 'Position and Lag\nHeld Constant,\nDecline %s'
                    % ls.format_stat(held_drop), ls.MEMORY_DARK, ha='left', va='top', dx_points=0)

    # Panel b: hit rate
    for curves, filled in [(whole_stream, False), (held_constant, True)]:
        values, low, high = curves['hit']
        draw_series(ax_hit, bins, values, low, high, ls.HIT_BLUE, filled)
    ax_hit.set_ylim(0.76, 0.86)
    ax_hit.set_yticks([0.76, 0.78, 0.8, 0.82, 0.84, 0.86])
    ax_hit.set_ylabel('Hit Rate')
    ls.direct_label(ax_hit, 2.2, 0.8475, 'Whole Stream', ls.HIT_BLUE, ha='left', va='center')
    ls.direct_label(ax_hit, 20.5, 0.8185, 'Held Constant', ls.HIT_BLUE, ha='right', va='bottom', dx_points=0)

    # Panel c: false alarm rate
    for curves, filled in [(whole_stream, False), (held_constant, True)]:
        values, low, high = curves['fa']
        draw_series(ax_fa, bins, values, low, high, ls.FA_RED, filled)
    ax_fa.set_ylim(0, 0.035)
    ax_fa.set_yticks([0, 0.01, 0.02, 0.03])
    ax_fa.set_ylabel('False Alarm Rate')
    ls.direct_label(ax_fa, 15.5, 0.0305, 'Whole Stream', ls.FA_RED, ha='right', va='center', dx_points=0)
    ls.direct_label(ax_fa, 20.5, 0.0185, 'Held Constant', ls.FA_RED, ha='right', va='top', dx_points=0)

    for ax, letter in zip(axes, ['a', 'b', 'c']):
        bin_axis(ax)
        ls.plain_ticks(ax)
        ls.panel_label(ax, letter)

    # every plotted interval sits inside its axis
    for ax, measure in [(ax_dprime, 'dprime'), (ax_hit, 'hit'), (ax_fa, 'fa')]:
        for curves in (whole_stream, held_constant):
            values, low, high = curves[measure]
            assert low.min() >= ax.get_ylim()[0] and high.max() <= ax.get_ylim()[1], measure
            assert np.all(low <= values) and np.all(values <= high), measure

    ls.save_figure(fig, 'FigS13_things_trial_level_context', str(a5_common.FIGURES))
    write_panel_data('FigS13a_dprime.csv', bins, whole_stream['dprime'], held_constant['dprime'])
    write_panel_data('FigS13b_hit_rate.csv', bins, whole_stream['hit'], held_constant['hit'])
    write_panel_data('FigS13c_false_alarm_rate.csv', bins, whole_stream['fa'], held_constant['fa'])
    print('Fig. S13 drawn from %s' % a5_common.relative(folder))


if __name__ == '__main__':
    main()
