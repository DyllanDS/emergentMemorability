"""
tests_table.py: write a module's outputs/tests.csv (the uncertainty statistics, T01 to T63) in the
convention of statistics/README.md.

    import tests_table
    tests = tests_table.TestsTable(script='code/stats_fig3.py')
    tests.add(test_id='T06', ledger_ids='L2-009;L2-014;L2-015', location='Fig. 3e',
              test='Spearman correlation across the 40 critical items between the published memorability '
                   'score (Kramer et al. 2023) and corrected recognition in this task',
              statistic_name='Spearman rho', statistic=rho, n=40, unit='items (50 participants)',
              df=38, p=p_value, p_type='parametric two-sided', effect_size_name='rho', effect_size=rho,
              ci_low=interval_low, ci_high=interval_high, ci_method='percentile bootstrap',
              resampling_unit='items', n_resamples=10000, seed=20261013)
    (rho, p_value, interval_low and interval_high are the values the module computed)
    tests.write(paths['outputs'])                      # -> outputs/tests.csv

Columns (statistics/README.md): test_id, ledger_ids, location, test, statistic_name, statistic, n,
unit, df, p, p_type, effect_size_name, effect_size, ci_low, ci_high, ci_method, resampling_unit,
n_resamples, seed, script.
Numbers are stored with 6 significant digits ('NaN' for a missing value). format_p() and
format_value() round a p value or a statistic to 3 decimals for text.
"""
import csv
import os

COLUMNS = ['test_id', 'ledger_ids', 'location', 'test', 'statistic_name', 'statistic', 'n', 'unit', 'df', 'p',
           'p_type', 'effect_size_name', 'effect_size', 'ci_low', 'ci_high', 'ci_method', 'resampling_unit',
           'n_resamples', 'seed', 'script']
P_TYPES = ('parametric two-sided', 'parametric one-sided', 'permutation two-sided', 'bootstrap one-sided',
           'bootstrap two-sided', 'exact two-sided', 'rank test two-sided (Wilcoxon signed-rank)', 'none')


def _cell(value):
    if value is None:
        return ''
    if isinstance(value, float):
        if value != value:          # NaN
            return 'NaN'
        return '%.6g' % value
    return str(value)


class TestsTable:
    """Collects the tests.csv rows of one module."""

    def __init__(self, script):
        self.script = script
        self.rows = []

    def add(self, **fields):
        unknown = set(fields) - set(COLUMNS)
        if unknown:
            raise KeyError('not tests.csv columns: %s' % ', '.join(sorted(unknown)))
        if 'test_id' not in fields:
            raise KeyError('every row needs a test_id (T01 to T63)')
        if fields.get('p_type') and fields['p_type'] not in P_TYPES and not str(fields['p_type']).startswith('permutation'):
            raise ValueError('p_type %r is not one of %s' % (fields['p_type'], ', '.join(P_TYPES)))
        row = {column: '' for column in COLUMNS}
        row['script'] = self.script
        for name, value in fields.items():
            row[name] = _cell(value)
        self.rows.append(row)
        return row

    def write(self, outputs_folder, file_name='tests.csv'):
        os.makedirs(outputs_folder, exist_ok=True)
        path = os.path.join(outputs_folder, file_name)
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
        print('wrote %s: %d rows' % (path, len(self.rows)))
        return path


def format_p(p):
    """'p < 0.001' below 0.001, else 'p = 0.036' (3 decimals)."""
    p = float(p)
    if p < 0.001:
        return 'p < 0.001'
    return 'p = %.3f' % p


def format_value(value, decimals=3):
    """Fixed decimals with a hyphen-minus and no '-0.000'."""
    text = '%.*f' % (decimals, float(value))
    if text.startswith('-') and float(text) == 0:
        text = text[1:]
    return text

