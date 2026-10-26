"""
figure_style.py: the style of the SI Appendix figures.

Every SI Appendix figure script (code/make_figureS<n>.py in its module) imports this module, and
so does module 04 for Fig. 4. The figures follow the look of the main-text figures (Helvetica,
MATLAB-like axes, hits blue and false alarms red, black capped error bars), with short direct labels
on the plot instead of legends, the key statistic printed on the plot, and plain axis labels. A
figure script gets its module's folders from shared/python/release_paths.py and passes the
outputs/figures folder to save_figure(), which writes a 300 dpi PNG (PDF and SVG on request).

How a figure script uses it
---------------------------
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
    import release_paths
    import figure_style as ls
    paths = release_paths.module_paths(__file__)     # data, outputs, figures, figure_data, logs of this module

    ls.apply_style('data')                         # 'data' (Figs. 3e, 5c look), 'model' or 'bars' (Figs. 1, 7d look)
    fig, axes = ls.new_figure(n_panels=2, width='double')
    ax = axes[0]
    ls.scatter_points(ax, x, y)                    # one point = one item
    ls.fit_line(ax, x, y)                          # gray regression line
    ls.stat_text(ax, ls.format_r_p(r, p, n=1600, unit='items'))
    ax.set_xlabel('Summed Similarity'); ax.set_ylabel('Item Memorability')
    ls.panel_label(ax, 'a')
    ls.hit_fa_bars(axes[1], ['Shown Alone', 'With 8 Similar Items'], hits, fas,
                   hit_low=..., hit_high=..., fa_low=..., fa_high=...)
    ls.save_figure(fig, 'example_figure', paths['figures'])

Every drawing helper takes an Axes as its first argument and returns what it drew.
"""
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import to_rgba


# ----------------------------------------------------------------------------- where files go
# No folder is derived from this file's location: each figure script passes its module's
# outputs/figures folder to save_figure() (see shared/python/release_paths.py).


# ----------------------------------------------------------------------------- colors
# Each color has one meaning across all figures. Never reuse a color for something else.
HIT_BLUE = (0.30, 0.60, 0.90)        # hit rate (the blue of Fig. 7d)
FA_RED = (0.90, 0.30, 0.30)          # false alarm rate (the red of Fig. 7d)
MEMORY_DARK = (0.25, 0.25, 0.25)     # memorability (hit rate minus false alarm rate) or d-prime,
                                     # when it is plotted next to hits and false alarms
DATA_BLUE = (0.0, 0.447, 0.741)      # any single series of observed data (Figs. 1 and 2: MATLAB blue)
MODEL_GRAY = (0.55, 0.55, 0.55)      # model predictions and simulations shown next to data
FIT_GRAY_DATA = (0.75, 0.75, 0.75)   # regression line on data scatters (Fig. 5c, d)
FIT_GRAY_MODEL = (0.50, 0.50, 0.50)  # regression line on model scatters (Fig. 1b, d)
HIGHLIGHT_BLACK = (0.0, 0.0, 0.0)    # the setting the paper reports (filled black point, larger)
AXIS_DARK = (0.15, 0.15, 0.15)       # axis lines and tick labels (MATLAB default, as in the main-text figures)
LABEL_GRAY = (0.35, 0.35, 0.35)      # direct labels for gray or black things ("Model", "Paper Setting")

# Two conditions that are not hits and false alarms (for example original set versus reduced
# set): the natural or original condition is red and the controlled or reduced one is blue,
# as in Fig. 7d. Use these names so the meaning is visible in the script.
NATURAL_RED = FA_RED
CONTROLLED_BLUE = HIT_BLUE

# ----------------------------------------------------------------------------- sizes
CM = 1 / 2.54                                   # inches per centimeter
WIDTHS_CM = {'single': 8.7, 'mid': 11.4, 'double': 17.8}   # PNAS column widths
FONT_SIZE = 9                                   # every text element at final print size
PANEL_LETTER_SIZE = 13                          # panel letters only
MARKER_SIZE = 5                                 # points in line plots and dot plots (pt diameter)
SCATTER_AREA = 14                               # scatter points (pt^2), as in the paper
LINE_WIDTH = 1.0                                # data lines

# Families follow the main-text figure a new figure extends:
#   'model' = REM simulations, like Fig. 1: box and light grid
#   'data'  = experiment or THINGS data, like Figs. 3e and 5c: open axes, no grid
#   'bars'  = bar charts, like Fig. 7d: box and light grid
FAMILIES = {'model': True, 'bars': True, 'data': False}     # value = box and grid on
_current = {'family': 'data'}


def apply_style(family='data'):
    """Set the global matplotlib look. Call once at the top of each script, before new_figure()."""
    if family not in FAMILIES:
        raise ValueError("family must be 'data', 'model' or 'bars', not %r" % (family,))
    _current['family'] = family
    box_and_grid = FAMILIES[family]
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'axes.titleweight': 'normal',
        'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'axes.edgecolor': AXIS_DARK,
        'xtick.color': AXIS_DARK,
        'ytick.color': AXIS_DARK,
        'axes.linewidth': 0.6,
        'axes.spines.top': box_and_grid,
        'axes.spines.right': box_and_grid,
        'xtick.top': box_and_grid,
        'ytick.right': box_and_grid,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.visible': False,
        'ytick.minor.visible': False,
        'axes.grid': box_and_grid,
        'axes.axisbelow': True,
        'grid.color': AXIS_DARK,
        'grid.alpha': 0.15,
        'grid.linewidth': 0.5,
        'axes.unicode_minus': False,            # hyphen-minus in tick labels, as the paper
        'axes.xmargin': 0.05,
        'axes.ymargin': 0.05,
        'errorbar.capsize': 2.5,
        'legend.frameon': False,
        'pdf.fonttype': 42,                     # editable text in PDF
        'ps.fonttype': 42,
        'svg.fonttype': 'none',                 # editable text in SVG
        'mathtext.default': 'regular',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


# ----------------------------------------------------------------------------- layout
def _width_in_inches(width):
    """'single', 'mid', 'double' or a number of centimeters -> inches."""
    if isinstance(width, str):
        if width not in WIDTHS_CM:
            raise ValueError("width must be 'single', 'mid', 'double' or centimeters, not %r" % (width,))
        return WIDTHS_CM[width] * CM
    return float(width) * CM


def new_figure(n_panels=1, width=None, height_cm=None):
    """A figure with n_panels side by side (at most 3), sized to a PNAS column.

    width: 'single' (8.7 cm), 'mid' (11.4 cm), 'double' (17.8 cm) or a number of cm.
           Default: single column for one panel, double column for two or three.
    height_cm: total height. Default makes each plot area about 1.25 wide to 1 high.
    Returns (fig, list_of_axes)."""
    if n_panels < 1 or n_panels > 3:
        raise ValueError('one message per panel and at most three panels; split the figure instead')
    if width is None:
        width = 'single' if n_panels == 1 else 'double'
    width_in = _width_in_inches(width)
    if height_cm is None:
        panel_width_in = width_in / n_panels
        height_in = panel_width_in / 1.25 + 0.15        # a little room for the panel letters
        height_in = max(height_in, 2.3)                 # never so short that two-line labels collide
    else:
        height_in = height_cm * CM
    fig, axes = plt.subplots(1, n_panels, figsize=(width_in, height_in), squeeze=False,
                             layout='constrained')
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.06, wspace=0.08)
    return fig, list(axes[0])


def panel_label(ax, letter):
    """Lower-case panel letter, 13 pt, regular weight, top-left outside the plot.
    Use only on figures with two or three panels."""
    return ax.annotate(letter, xy=(0, 1), xycoords='axes fraction',
                       xytext=(-30, 6), textcoords='offset points',
                       fontsize=PANEL_LETTER_SIZE, ha='left', va='bottom', color='black')


def plain_ticks(ax):
    """Tick labels like the paper: '0', '0.5', '-0.4', '1,600' (no trailing zeros)."""
    def format_number(value, _position=None):
        if abs(value) < 1e-12:
            return '0'
        if abs(value) >= 1000 and float(value).is_integer():
            return '{:,}'.format(int(value))
        return '{:g}'.format(value)
    for axis in (ax.xaxis, ax.yaxis):
        if type(axis.get_major_formatter()) is mticker.ScalarFormatter:
            axis.set_major_formatter(mticker.FuncFormatter(format_number))
    return ax


def category_axis(ax, labels, axis='x'):
    """Named categories at positions 1..n with limits 0.4..n+0.6. Returns the positions.
    Use short labels; a line break ('With 8\\nSimilar Items') is better than rotated text."""
    positions = np.arange(1, len(labels) + 1)
    if axis == 'x':
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlim(0.4, len(labels) + 0.6)
        ax.tick_params(axis='x', length=0)              # no tick marks under category names
    else:
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_ylim(0.4, len(labels) + 0.6)
        ax.tick_params(axis='y', length=0)
    return positions


# ----------------------------------------------------------------------------- reference lines
def zero_line(ax, axis='y'):
    """Black dashed line at zero. Draw it whenever the sign of the plotted value matters."""
    style = dict(color='black', linestyle='--', linewidth=0.8, zorder=1)
    if axis == 'y':
        return ax.axhline(0, **style)
    return ax.axvline(0, **style)


def reference_line(ax, value, axis='y', label=None, label_side='right'):
    """Black dashed line at another meaningful value (chance, a reliability ceiling, y = x is
    identity_line). Optional short gray label at one end."""
    style = dict(color='black', linestyle='--', linewidth=0.8, zorder=1)
    line = ax.axhline(value, **style) if axis == 'y' else ax.axvline(value, **style)
    if label:
        if axis == 'y':
            x_end = 1.0 if label_side == 'right' else 0.0
            ax.annotate(label, xy=(x_end, value), xycoords=('axes fraction', 'data'),
                        xytext=(-2 if label_side == 'right' else 2, 2), textcoords='offset points',
                        ha='right' if label_side == 'right' else 'left', va='bottom',
                        fontsize=FONT_SIZE, color=LABEL_GRAY)
        else:
            ax.annotate(label, xy=(value, 1.0), xycoords=('data', 'axes fraction'),
                        xytext=(2, -2), textcoords='offset points', ha='left', va='top',
                        fontsize=FONT_SIZE, color=LABEL_GRAY)
    return line


def identity_line(ax):
    """Black dashed y = x line across the current limits (set the limits first)."""
    low = min(ax.get_xlim()[0], ax.get_ylim()[0])
    high = max(ax.get_xlim()[1], ax.get_ylim()[1])
    return ax.plot([low, high], [low, high], color='black', linestyle='--', linewidth=0.8, zorder=1)[0]


# ----------------------------------------------------------------------------- text on the plot
def direct_label(ax, x, y, text, color='black', ha='left', va='center', dx_points=4, dy_points=0):
    """A short label (one to three words) written next to the thing it names, in that thing's
    color, in data coordinates. This replaces a legend. Example: direct_label(ax, 20.3, 2.7,
    'All Trials', FA_RED). Offsets are in points so the label never sits on the marker."""
    return ax.annotate(text, xy=(x, y), xycoords='data',
                       xytext=(dx_points, dy_points), textcoords='offset points',
                       ha=ha, va=va, fontsize=FONT_SIZE, color=color, annotation_clip=False)


def format_p(p):
    """'p < 0.001', 'p < 0.01', otherwise 'p = 0.036' (paper convention)."""
    if p < 0.001:
        return 'p < 0.001'
    if p < 0.01:
        return 'p < 0.01'
    return 'p = ' + ('%.3f' % p).rstrip('0').rstrip('.')


def format_stat(value, decimals=2):
    """Two decimals with a hyphen-minus, and no '-0.00'."""
    text = '%.*f' % (decimals, value)
    if text.startswith('-') and float(text) == 0:
        text = text[1:]
    return text


def format_r_p(r, p=None, n=None, unit=None, symbol='r'):
    """'r = -0.56, p < 0.001' and, when n is given, a second line 'n = 1,600 items'."""
    text = '%s = %s' % (symbol, format_stat(r))
    if p is not None:
        text += ', ' + format_p(p)
    if n is not None:
        text += '\nn = {:,}'.format(int(n)) + ((' ' + unit) if unit else '')
    return text


def stat_text(ax, text, corner='upper left'):
    """The statistic that carries the panel's point, inside the plot in a corner, 9 pt black on
    a white patch."""
    positions = {'upper left': (0.04, 0.96, 'left', 'top'),
                 'upper right': (0.96, 0.96, 'right', 'top'),
                 'lower left': (0.04, 0.04, 'left', 'bottom'),
                 'lower right': (0.96, 0.04, 'right', 'bottom')}
    x, y, ha, va = positions[corner]
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=FONT_SIZE,
                   color='black', linespacing=1.3,
                   bbox=dict(facecolor='white', edgecolor='none', pad=1.0), zorder=5)


# ----------------------------------------------------------------------------- marks
def scatter_points(ax, x, y, color=DATA_BLUE, area=SCATTER_AREA, alpha=0.7):
    """Filled circles, translucent face and solid edge of the same color (Fig. 1b).
    One point = one item, one feature or one participant."""
    face = list(to_rgba(color))
    face[3] = alpha
    return ax.scatter(x, y, s=area, facecolors=[tuple(face)], edgecolors=[color],
                      linewidths=0.5, zorder=3, clip_on=False)


def fit_line(ax, x, y, over='data'):
    """Least-squares line in gray. over='data' draws it over the data range;
    over='axis' across the whole x-axis (as in Fig. 5c; set the x-limits first).
    Light thick gray for the 'data' family, darker thin gray for 'model' and 'bars'."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    if over == 'axis':
        x_ends = np.array(ax.get_xlim())
    else:
        x_ends = np.array([x[keep].min(), x[keep].max()])
    if _current['family'] == 'data':
        color, width = FIT_GRAY_DATA, 1.8
    else:
        color, width = FIT_GRAY_MODEL, 1.0
    line = ax.plot(x_ends, slope * x_ends + intercept, color=color, linewidth=width, zorder=2)[0]
    if over == 'axis':
        ax.set_xlim(x_ends)
    return line


def error_bars(ax, x, y, low, high, color='black'):
    """Capped error bars from interval ENDS (low, high), not offsets. Black by default."""
    y = np.asarray(y, dtype=float)
    below = y - np.asarray(low, dtype=float)
    above = np.asarray(high, dtype=float) - y
    return ax.errorbar(x, y, yerr=[below, above], fmt='none', ecolor=color, elinewidth=0.75,
                       capsize=2.5, capthick=0.75, zorder=4, clip_on=False)


def line_series(ax, x, y, color=DATA_BLUE, low=None, high=None, label=None, label_at='end'):
    """One series across ordered levels (bins, list lengths, parameter values): filled circles
    joined by a thin line, optional black error bars, optional direct label at the last point
    (label_at='end') or first point (label_at='start')."""
    drawn = ax.plot(x, y, color=color, linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE,
                    markerfacecolor=color, markeredgecolor=color, zorder=3, clip_on=False)[0]
    if low is not None and high is not None:
        error_bars(ax, x, y, low, high)
    if label:
        if label_at == 'end':
            direct_label(ax, x[-1], y[-1], label, color=color, ha='left', dx_points=6)
        else:
            direct_label(ax, x[0], y[0], label, color=color, ha='right', dx_points=-6)
    return drawn


def dot_with_interval(ax, positions, values, low, high, color=DATA_BLUE):
    """Dot plot: one filled circle per condition with its interval.
    Good for comparing a handful of conditions on one number."""
    ax.plot(positions, values, linestyle='none', marker='o', markersize=MARKER_SIZE + 1,
            markerfacecolor=color, markeredgecolor=color, zorder=3, clip_on=False)
    return error_bars(ax, positions, values, low, high)


def highlight_point(ax, x, y, label='Paper Setting', label_side='right'):
    """Mark the setting the paper reports: a larger filled black circle plus a short gray label.
    No rings, no open markers."""
    ax.plot([x], [y], linestyle='none', marker='o', markersize=MARKER_SIZE + 2.5,
            markerfacecolor=HIGHLIGHT_BLACK, markeredgecolor=HIGHLIGHT_BLACK, zorder=6, clip_on=False)
    if label:
        if label_side == 'right':
            direct_label(ax, x, y, label, color=LABEL_GRAY, ha='left', va='bottom',
                         dx_points=5, dy_points=4)
        else:
            direct_label(ax, x, y, label, color=LABEL_GRAY, ha='right', va='bottom',
                         dx_points=-5, dy_points=4)


def simple_bars(ax, labels, values, colors=None, low=None, high=None, bar_width=0.6):
    """Plain bars, one per named condition, with black capped error bars if low/high are given.
    Colors default to DATA_BLUE; pass MODEL_GRAY for model bars."""
    positions = category_axis(ax, labels)
    if colors is None:
        colors = [DATA_BLUE] * len(values)
    bars = ax.bar(positions, values, width=bar_width, color=colors, edgecolor='none', zorder=2)
    if low is not None and high is not None:
        error_bars(ax, positions, values, low, high)
    return bars


def hit_fa_bars(ax, group_labels, hit_rates, fa_rates, hit_low=None, hit_high=None,
                fa_low=None, fa_high=None, label_bars=True, bar_width=0.36):
    """Paired bars for each group: hit rate in blue (left) and false alarm rate in red (right).
    The y-axis is 'Hit or False Alarm Rate' from 0 to 1.
    label_bars=True writes 'Hits' and 'False Alarms' inside the axes above the first pair,
    so no legend is needed. Interval ends (not offsets) go in *_low / *_high.
    Returns the list of group center positions."""
    group_positions = category_axis(ax, group_labels)
    hit_positions = group_positions - bar_width / 2
    fa_positions = group_positions + bar_width / 2
    ax.bar(hit_positions, hit_rates, width=bar_width, color=HIT_BLUE, edgecolor='none', zorder=2)
    ax.bar(fa_positions, fa_rates, width=bar_width, color=FA_RED, edgecolor='none', zorder=2)
    if hit_low is not None and hit_high is not None:
        error_bars(ax, hit_positions, hit_rates, hit_low, hit_high)
    if fa_low is not None and fa_high is not None:
        error_bars(ax, fa_positions, fa_rates, fa_low, fa_high)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    if not ax.get_ylabel():
        ax.set_ylabel('Hit or False Alarm Rate')
    if label_bars:
        # Each label names its bar in the first group. A label goes just above the bar (or its
        # interval) in the bar color; when that would run into the top of the axes (bar above
        # 0.75), it goes inside the bar, near its top, in white.
        bar_info = [(hit_positions[0], hit_rates[0], hit_high, 'Hits', HIT_BLUE),
                    (fa_positions[0], fa_rates[0], fa_high, 'False\nAlarms', FA_RED)]
        for position, height, high, text, color in bar_info:
            top = height if high is None else high[0]
            if top > 0.75:
                direct_label(ax, position, height, text, 'white', ha='center', va='top',
                             dx_points=0, dy_points=-4)
            else:
                direct_label(ax, position, top, text, color, ha='center', va='bottom',
                             dx_points=0, dy_points=3)
    plain_ticks(ax)
    return group_positions


# ----------------------------------------------------------------------------- saving
def save_figure(fig, name, figures_folder, formats=('png',)):
    """Write <figures_folder>/<name>.png at 300 dpi, plus .pdf and .svg when formats asks for them (vector,
    editable text). The page keeps the exact
    width set in new_figure() (no tight cropping). figures_folder is the calling module's
    outputs/figures folder (release_paths.module_paths(__file__)['figures']). Returns the paths."""
    os.makedirs(figures_folder, exist_ok=True)
    paths = []
    for file_format in formats:
        path = os.path.join(figures_folder, name + '.' + file_format)
        if file_format == 'png':
            fig.savefig(path, dpi=300)
        else:
            fig.savefig(path)
        paths.append(path)
    print('saved', *paths, sep='\n  ')
    return paths
