"""
numbers_table.py: write a module's outputs/numbers.csv and judge each computed value against the paper.

Every module writes one row per ledger item it owns (and one per duplicate location), with the
value its code computed. The paper's printed value, the location and the quote come from
shared/ledger/paper_numbers.csv, so a module only supplies the ledger id and what it computed.

    import numbers_table
    table = numbers_table.NumbersTable(script='code/inversion_analysis.m')
    table.add('L2-014', computed_value=[-0.50038, 0.00101], mode='rerun',
              note='Spearman rho over the 40 critical items and its p')
    table.write(paths['outputs'])                      # -> outputs/numbers.csv

Columns: ledger_id, location, quote, paper_value, computed_value, match, match_type, script, mode, note

How a value is judged (match_type, from the ledger unless the module overrides it)
    round        the computed value rounds to the printed value at the printed precision:
                 |computed - printed| <= half a unit of the printed last digit (27% -> 0.5 percent
                 points; -0.50 -> 0.005)
    exact, count the computed value equals the printed one (numbers compared as numbers)
    less_than    a printed '< x' (for example 'p < .001'): computed < x
    range        a printed 'a to b': both ends judged as 'round'; the computed value is a pair
    sign         the computed value has the printed sign
    provided, pending, documented, none
                 not judged here; match is set to that word
A printed value with several parts ('-0.50; <0.01') is compared part by part with a computed list
of the same length; a part that starts with '<' is judged as less_than whatever the row's type.
match is 'yes' or 'no' (or the word above).
"""
import csv
import math
import os
import re

_RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEDGER_FILE = os.path.join(_RELEASE_ROOT, 'shared', 'ledger', 'paper_numbers.csv')
COLUMNS = ['ledger_id', 'location', 'quote', 'paper_value', 'computed_value', 'match', 'match_type',
           'script', 'mode', 'note']
NOT_JUDGED = ('provided', 'pending', 'documented', 'none')
NUMBER_WORDS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
                'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'fifteen': 15, 'twenty': 20}


def load_ledger():
    """{ledger id: row of shared/ledger/paper_numbers.csv}."""
    with open(LEDGER_FILE, newline='', encoding='utf-8') as handle:
        return {row['id']: row for row in csv.DictReader(handle)}


def _clean(text):
    text = str(text).strip()
    text = text.replace('\u2212', '-').replace('\u2013', '-').replace('\u2014', '-')   # minus, en dash, em dash
    return text


def parse_number(text):
    """'-0.50' -> -0.5; '27%' -> 27; '1,854' -> 1854; '.036' -> 0.036; 'five' -> 5; else None."""
    text = _clean(text).lower()
    text = re.sub(r'^(p|r|rho|d\'|d|t|n)\s*=\s*', '', text)
    text = text.replace(',', '').replace('%', '').strip()
    if text in NUMBER_WORDS:
        return float(NUMBER_WORDS[text])
    try:
        return float(text)
    except ValueError:
        return None


def printed_decimals(text):
    """Number of digits after the decimal point in a printed number ('27%' -> 0, '-0.50' -> 2)."""
    text = _clean(text).replace('%', '').replace(',', '')
    match = re.search(r'\.(\d+)', text)
    return len(match.group(1)) if match else 0


def judge_one(printed, computed, match_type):
    """True / False for one printed part against one computed value."""
    printed = _clean(printed)
    if printed.startswith('<') or printed.lower().startswith('p <') or printed.lower().startswith('p<'):
        threshold = parse_number(printed.split('<', 1)[1])
        return threshold is not None and float(computed) < threshold
    if match_type == 'less_than':
        threshold = parse_number(printed)
        return threshold is not None and float(computed) < threshold
    if match_type == 'range':
        ends = re.split(r'\s+to\s+', printed)
        if isinstance(computed, str):
            computed = re.split(r'\s+to\s+', _clean(computed))
        if len(ends) != 2 or len(computed) != 2:
            return False
        return judge_one(ends[0], computed[0], 'round') and judge_one(ends[1], computed[1], 'round')
    printed_number = parse_number(printed)
    if printed_number is None:
        return _clean(computed).lower() == printed.lower()
    try:
        computed_number = float(computed)
    except (TypeError, ValueError):
        computed_number = parse_number(computed)
    if computed_number is None or math.isnan(computed_number):
        return False
    if match_type == 'sign':
        return math.copysign(1, computed_number) == math.copysign(1, printed_number)
    if match_type in ('exact', 'count'):
        return computed_number == printed_number
    tolerance = 0.5 * 10 ** (-printed_decimals(printed)) + 1e-9
    return abs(computed_number - printed_number) <= tolerance


def judge(paper_value, computed_value, match_type):
    """'yes' / 'no', or the not-judged word. paper_value may hold several parts separated by ';'."""
    if match_type in NOT_JUDGED:
        return match_type
    if hasattr(computed_value, 'tolist'):            # a numpy array or number becomes a plain list or number
        computed_value = computed_value.tolist()
    printed_parts = [part for part in str(paper_value).split(';') if part.strip()]
    if isinstance(computed_value, (list, tuple)) and match_type != 'range':
        computed_parts = list(computed_value)
    elif isinstance(computed_value, str) and ';' in computed_value:
        computed_parts = [part.strip() for part in computed_value.split(';') if part.strip()]
    else:
        computed_parts = [computed_value]
    if match_type == 'range' and len(printed_parts) == 1:
        return 'yes' if judge_one(printed_parts[0], computed_value, 'range') else 'no'
    if len(printed_parts) != len(computed_parts):
        return 'no'
    for printed, computed in zip(printed_parts, computed_parts):
        if not judge_one(printed, computed, match_type):
            return 'no'
    return 'yes'


def format_computed(value):
    """Text for the computed_value column (lists joined with '; ', numbers with up to 6 significant digits)."""
    if hasattr(value, 'tolist'):                     # a numpy array or number becomes a plain list or number
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return '; '.join(format_computed(part) for part in value)
    if isinstance(value, float):
        return '%.6g' % value
    return str(value)


class NumbersTable:
    """Collects the numbers.csv rows of one module."""

    def __init__(self, script):
        self.script = script
        self.rows = []
        self.ledger = load_ledger()

    def add(self, ledger_id, computed_value, mode, note='', script=None, match_type=None,
            paper_value=None, location=None, quote=None):
        if ledger_id not in self.ledger:
            raise KeyError('unknown ledger id %s' % ledger_id)
        entry = self.ledger[ledger_id]
        match_type = match_type or entry['match_type']
        paper_value = entry['paper_value'] if paper_value is None else paper_value
        match = judge(paper_value, computed_value, match_type)
        self.rows.append(dict(ledger_id=ledger_id, location=location or entry['location'],
                              quote=quote or entry['quote'], paper_value=paper_value,
                              computed_value=format_computed(computed_value), match=match,
                              match_type=match_type, script=script or self.script, mode=mode, note=note))
        return match

    def write(self, outputs_folder, file_name='numbers.csv'):
        os.makedirs(outputs_folder, exist_ok=True)
        path = os.path.join(outputs_folder, file_name)
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
        summary = {}
        for row in self.rows:
            summary[row['match']] = summary.get(row['match'], 0) + 1
        print('wrote %s: %d rows %s' % (path, len(self.rows), summary))
        return path
