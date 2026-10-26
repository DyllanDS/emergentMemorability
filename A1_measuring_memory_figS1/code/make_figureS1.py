"""
make_figureS1.py: Fig. S1, corrected recognition against d_a in the three image-set experiments.

WHAT THE FIGURE SHOWS (SI Appendix, Section 1, Fig. S1)
One point per concept in one image set (maintained, changed or reduced): the 2,453 usable concept
scores of the 450 kept participants. x = memorability, corrected recognition (hit rate minus false
alarm rate; responses 1 to 3 of the six-point scale = "old"). y = d_a from the same trials'
confidence counts, with the z-ROC slope fixed at 0.80. The gray line is the least-squares fit. The
panel prints the Pearson r between the two scores, its p value and n.

INPUTS   A1_measuring_memory_figS1/outputs/concept_scores.csv (written by criterion_dprime_da_checks.py)
OUTPUTS  A1_measuring_memory_figS1/outputs/figures/FigS1_criterion_dprime_da.png (300 dpi)
         A1_measuring_memory_figS1/outputs/figure_data/FigS1_points.csv (every plotted point)
         A1_measuring_memory_figS1/outputs/figure_data/FigS1_fit_and_statistics.csv (the gray line and
             the printed statistics)
RUN (from the repository root)
    python A1_measuring_memory_figS1/code/make_figureS1.py
No random numbers; a few seconds.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared', 'python'))
import release_paths  # noqa: E402  (imported first: keeps the shared folder free of __pycache__)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import figure_style as ls  # noqa: E402

paths = release_paths.module_paths(__file__)

# ---- the concept scores
scores_file = os.path.join(paths['outputs'], 'concept_scores.csv')
if not os.path.exists(scores_file):
    sys.exit('make_figureS1.py: outputs/concept_scores.csv is missing; run criterion_dprime_da_checks.py first')
scores = pd.read_csv(scores_file, float_precision='round_trip')

corrected_recognition = scores['corrected_recognition'].to_numpy()
d_a = scores['d_a_slope080'].to_numpy()
number_of_points = len(scores)

# ---- the statistics the panel prints, and the gray least-squares line
r, p = stats.pearsonr(corrected_recognition, d_a)
fit_slope, fit_intercept = np.polyfit(corrected_recognition, d_a, 1)
annotation = ls.format_r_p(r, p, number_of_points, 'concept scores')
print('points %d, Pearson r = %.4f, p = %.3g' % (number_of_points, r, p))
print('x range %.3f to %.3f, y range %.3f to %.3f' % (corrected_recognition.min(), corrected_recognition.max(),
                                                      d_a.min(), d_a.max()))

# ---- the figure
ls.apply_style('data')
fig, axes = ls.new_figure(n_panels=1, width='single')
ax = axes[0]
# 2,453 points: smaller, lighter markers so the shape of the cloud stays visible
ls.scatter_points(ax, corrected_recognition, d_a, color=ls.DATA_BLUE, area=6, alpha=0.35)
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.5, 1])
ax.set_ylim(0, 4)
ax.set_yticks([0, 1, 2, 3, 4])
ls.fit_line(ax, corrected_recognition, d_a)
ls.stat_text(ax, annotation, corner='upper left')
ax.set_xlabel('Memorability\n(Hit Rate Minus False Alarm Rate)')
ax.set_ylabel('$d_a$')
ls.plain_ticks(ax)
ls.save_figure(fig, 'FigS1_criterion_dprime_da', paths['figures'])

# every point must lie inside the drawn axes (the markers are not clipped)
outside = np.sum((corrected_recognition < 0) | (corrected_recognition > 1) | (d_a < 0) | (d_a > 4))
if outside > 0:
    print('WARNING: %d points lie outside the axis limits of the panel' % outside)

# ---- the plotted numbers
points = scores[['image_set', 'concept_number', 'concept', 'corrected_recognition', 'd_a_slope080']]
points.to_csv(os.path.join(paths['figure_data'], 'FigS1_points.csv'), index=False)
fit_rows = [('number_of_points', number_of_points),
            ('pearson_r', r),
            ('p_value', p),
            ('fit_slope', fit_slope),
            ('fit_intercept', fit_intercept),
            ('fit_line_x_from', corrected_recognition.min()),
            ('fit_line_x_to', corrected_recognition.max()),
            ('annotation', annotation.replace('\n', ' / '))]
pd.DataFrame(fit_rows, columns=['quantity', 'value']).to_csv(
    os.path.join(paths['figure_data'], 'FigS1_fit_and_statistics.csv'), index=False)
print('wrote outputs/figure_data/FigS1_points.csv and FigS1_fit_and_statistics.csv')
