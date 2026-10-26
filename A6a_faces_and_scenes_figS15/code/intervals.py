"""
intervals.py: the 95% confidence intervals of Fig. S15 (the error bars) and the p value of each bar
of panel a.

What it does
    For every correlation Fig. S15 draws, it computes the 95% confidence interval the caption names,
    "Fisher transformation, 2,222 scored images per database": the interval is built on atanh(rho)
    with the Bonett and Wright (2000) standard error for Spearman's rho,
        standard error of atanh(rho) = sqrt((1 + rho^2 / 2) / (n - 3)),   n = 2,222,
    and transformed back with tanh.
    Panel a (held-out correlation with memorability): each plotted rho is the mean of five held-out
    fold correlations (444 or 445 images each), and its interval is computed for that mean with
    n = 2,222. The p value of each panel a bar is the largest of its five single-fold p values (t
    approximation for Spearman's rho with 444 images); "all p < 0.001" in SI Appendix, Section 6
    refers to these.
    Panel b (whole-sample correlation with the hit rate and the false alarm rate): one correlation
    over the 2,222 scored images; the p value is the one printed in the crowding report.
    It does this for each source of results (read_results.py): the saved faces results
    (data/provided_faces/), the faces recomputed from the database when its archives are present,
    this scene run (outputs/scenes/, which the figure plots) and the saved scene run
    (data/paper_run_scenes/).
Inputs
    the reports listed in read_results.py (data/provided_faces/, outputs/scenes/, data/paper_run_scenes/)
Output
    A6a_faces_and_scenes_figS15/outputs/intervals_figS15.csv
        one row per source and correlation: panel ('a' or 'b'), database, source, measure, rho, n,
        ci_low, ci_high, p, p_note, and the five fold correlations for the panel a rows
Run (from the repository root; code/run_module.py runs it)
    python A6a_faces_and_scenes_figS15/code/intervals.py
"""
import csv
import math
import os
import sys

sys.dont_write_bytecode = True
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import read_results  # noqa: E402

N_SCORED_IMAGES = 2222        # scored images per database (checked against each crowding report below)
SMALLEST_FOLD = 444           # 2,222 images split into five folds: 444 or 445 images each
CONFIDENCE_LEVEL = 95         # percent
Z_FOR_95_PERCENT = 1.959964   # the 97.5th percentile of the standard normal, to 6 decimals
OUTPUT_FILE = os.path.join(read_results.MODULE_FOLDER, 'outputs', 'intervals_figS15.csv')


def fisher_interval(rho, n):
    """95% interval for Spearman's rho, Fisher transformation with the Bonett-Wright standard error."""
    standard_error = math.sqrt((1 + rho ** 2 / 2) / (n - 3))
    centre = math.atanh(rho)
    low = math.tanh(centre - Z_FOR_95_PERCENT * standard_error)
    high = math.tanh(centre + Z_FOR_95_PERCENT * standard_error)
    return low, high


def spearman_p(rho, n):
    """Two-sided p for Spearman's rho with the t approximation (n - 2 degrees of freedom)."""
    t_value = rho * math.sqrt((n - 2) / (1 - rho ** 2))
    return 2 * stats.t.sf(abs(t_value), n - 2)


def rows_for_source(source):
    """All interval rows of one source of results."""
    database = read_results.SOURCES[source]['database']
    rows = []

    # the counts behind n = 2,222
    n_embedded, n_scored = read_results.image_counts(source)
    if n_scored != N_SCORED_IMAGES:
        raise SystemExit('%s: %d scored images, expected %d' % (source, n_scored, N_SCORED_IMAGES))

    # ---- panel a: held-out correlation with memorability, equal and fitted weights
    panel_a_rows = [('equal weights', read_results.EQUAL_WEIGHTS_ROW, read_results.EQUAL_WEIGHTS_COLUMN),
                    ('fitted weights', read_results.FITTED_WEIGHTS_ROW, read_results.FITTED_WEIGHTS_COLUMN)]
    for weighting, row_name, fold_column in panel_a_rows:
        printed_mean = read_results.heldout_rho(source, row_name)
        folds = read_results.fold_values(source, fold_column)
        fold_mean = sum(folds) / len(folds)
        # the report prints the fold mean to 3 decimals; the two must agree
        if abs(fold_mean - printed_mean) > 0.0006:
            raise SystemExit('%s: fold mean %.4f does not match the reported %.3f' % (source, fold_mean, printed_mean))
        largest_fold_p = 0.0
        for fold_rho in folds:
            largest_fold_p = max(largest_fold_p, spearman_p(fold_rho, SMALLEST_FOLD))
        # the interval is centered on the 3-decimal value printed in the report
        low, high = fisher_interval(printed_mean, N_SCORED_IMAGES)
        rows.append(dict(panel='a', database=database, source=source,
                         measure='corrected recognition, ' + weighting,
                         rho=printed_mean, rho_unrounded=fold_mean, n=N_SCORED_IMAGES,
                         ci_low=round(low, 4), ci_high=round(high, 4), p=largest_fold_p,
                         p_note='largest of five fold p values (t approximation, 444 images)',
                         fold_values='; '.join('%.6f' % value for value in folds)))

    # ---- panel b: whole-sample correlation with hits and false alarms. 'false alarm rate' is the measure
    # name that make_figureS15.py and stats_figS15.py look up.
    panel_b_rows = [('b', 'hit rate', 'hit rate'), ('b', 'false alarm rate', 'FA rate')]
    for panel, measure, label in panel_b_rows:
        value, p_value = read_results.limiting_case(source, label)
        low, high = fisher_interval(value, N_SCORED_IMAGES)
        rows.append(dict(panel=panel, database=database, source=source, measure=measure,
                         rho=value, rho_unrounded='', n=N_SCORED_IMAGES,
                         ci_low=round(low, 4), ci_high=round(high, 4), p=p_value,
                         p_note='as printed in the crowding report', fold_values=''))
    return rows


def main():
    faces = read_results.faces_source()       # prints the provided-output message when the faces are provided
    sources = ['faces_provided']
    if faces == 'faces_regenerated':
        sources.append('faces_regenerated')
    sources.append('scenes_rerun')
    sources.append('scenes_paper_run')
    interval_rows = []
    for source in sources:
        interval_rows.extend(rows_for_source(source))

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(interval_rows[0].keys()))
        writer.writeheader()
        for row in interval_rows:
            writer.writerow(row)

    print('Fig. S15 intervals: %d%% Fisher-z (Bonett-Wright) intervals, n = %d scored images per database'
          % (CONFIDENCE_LEVEL, N_SCORED_IMAGES))
    for row in interval_rows:
        print('%-9s %-17s %-40s rho %+.3f  95%% [%+.4f, %+.4f]  p %.3g' % (
            row['panel'], row['source'], row['measure'], row['rho'], row['ci_low'], row['ci_high'], row['p']))
    print('\nwrote', read_results.relative(OUTPUT_FILE))


if __name__ == '__main__':
    main()
