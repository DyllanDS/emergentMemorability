"""
collect_numbers.py: one table of every number reported in the paper, next to the value this release computes.

What it does
  shared/ledger/paper_numbers.csv lists the numbers and results the paper reports in the main text, the
  Methods and the SI Appendix (statistics, counts, parameters, percentages, figure values), one ledger
  item each. Each module writes <module>/outputs/numbers.csv with one row per ledger item it computes: the
  value its code computed and whether that value agrees with the printed one. This script joins the two
  into reported_numbers.csv at the repository root (one row per ledger item, in ledger order) and prints a
  summary that lists every row whose computed value does not agree with the printed one.

Inputs
  shared/ledger/paper_numbers.csv
  <module>/outputs/numbers.csv of the 24 modules (written when a module runs; the release ships them), and
  06_image_set_experiments_fig6_fig7/outputs/numbers_shared_concepts.csv (L3-065, test T19b)

Output
  reported_numbers.csv with the columns
    id              the ledger id (for example L1-003 or ADD-002)
    location        where the paper reports it (a section, or a figure caption such as "Fig. 1 caption" or
                    "SI Appendix, Fig. S1 caption")
    quote           the sentence or caption that reports it
    paper_value     the printed value the module judged against (parts joined by "; ")
    computed_value  the value the release computes (or reads from a saved output; see mode)
    match           yes or no (judged), or documented, pending or none (not judged; see Match words)
    match_type      how the value was judged (round, exact, count, less_than or range), or documented
    script          the script that computes it, as a path from the repository root
    mode            how the value is obtained (for example rerun, saved_run or provided)
    same_as         the id of a related row at another place in the paper, such as the same number,
                    result or experiment (for a restatement, the row it repeats); else empty
    note            what the computed value is (the module's note); for a row this script fills itself,
                    how it was filled

How each row is filled
  1. The row a module wrote for this ledger id (every module writes a row for every location it
     computes, also for a number printed at several locations). A judged row is judged again here with
     the same rules (shared/python/numbers_table.py) and listed on screen if the result differs; a row
     whose printed value is not the ledger's is reported on screen as a problem.
  2. An item with same_as filled that no module wrote a row for takes the computed value of the row
     that same_as names and is judged against its own printed value; the note says so.
  3. An item with no computation behind it (ledger module none): match none, empty computed value.
  4. An item of a module whose outputs/numbers.csv does not exist yet (the module has not been run):
     match pending.

Match words
  yes / no            the computed value was judged against the printed value at the printed precision,
                      also when it comes from the saved results in a module's data/provided/ folder (mode
                      provided)
  documented          a statement, design fact or figure panel without a printed number; the computed
                      value and the note give what the code finds
  pending             the module has not been run yet, or its outputs/numbers.csv has no row for the item
  none                no computation behind it (ledger module none)
A module can also pass the match type provided, which is not judged; when such a value prints as a number,
this script checks that it rounds to the printed value and lists it on screen if it does not.

Run (from the repository root, or from anywhere):
    python collect_numbers.py
Python 3 standard library only. Runtime: about a second. The exit code is 1 when a problem is found.
"""
import csv
import os
import re
import sys

# No __pycache__ folders in the release.
sys.dont_write_bytecode = True

RELEASE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RELEASE_ROOT, 'shared', 'python'))
import numbers_table  # noqa: E402  (the judging rules every module uses)
import release_paths  # noqa: E402  (the list of module folders)

LEDGER_FILE = os.path.join(RELEASE_ROOT, 'shared', 'ledger', 'paper_numbers.csv')
OUTPUT_FILE = os.path.join(RELEASE_ROOT, 'reported_numbers.csv')
OUTPUT_COLUMNS = ['id', 'location', 'quote', 'paper_value', 'computed_value', 'match', 'match_type', 'script',
                  'mode', 'same_as', 'note']
MODULE_COLUMNS = ['ledger_id', 'location', 'quote', 'paper_value', 'computed_value', 'match', 'match_type',
                  'script', 'mode', 'note']
JUDGED_MATCHES = ('yes', 'no')
SUMMARY_ORDER = ['yes', 'no', 'provided', 'documented', 'pending', 'none']


def read_csv_rows(path):
    """All rows of a CSV file as dicts of text (no number conversion, so every value stays as written)."""
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(row)
        return reader.fieldnames, rows


def release_relative_script(script_text, module_folder):
    """Script paths in a module's numbers.csv are written either from the module folder ('code/x.py')
    or from the repository root ('<module>/code/x.py'). Here every one is written from the repository root."""
    # 'code/' that does not already follow a folder name is the module's own code folder
    return re.sub(r'(?<![\w/])code/', module_folder + '/code/', script_text)


def every_printed_part_is_a_number(paper_value):
    """True when each '; '-separated part of a printed value is a number or a bound such as '< 0.001'."""
    parts = [part.strip() for part in paper_value.split(';') if part.strip()]
    if not parts:
        return False
    for part in parts:
        if part.startswith('<'):
            part = part[1:].strip()
        if numbers_table.parse_number(part) is None:
            return False
    return True


def main():
    # 1. The ledger: one row per printed number of the paper.
    ledger_columns, ledger_rows = read_csv_rows(LEDGER_FILE)
    ledger_by_id = {}
    for ledger_row in ledger_rows:
        ledger_by_id[ledger_row['id']] = ledger_row

    # 2. Every module's numbers.csv.
    module_rows_by_id = {}           # ledger id -> (module folder, row)
    modules_not_run = []
    problems = []
    for module_id, module_folder in release_paths.MODULE_FOLDERS.items():
        numbers_file = os.path.join(RELEASE_ROOT, module_folder, 'outputs', 'numbers.csv')
        if not os.path.exists(numbers_file):
            modules_not_run.append(module_folder)
            continue
        columns, rows = read_csv_rows(numbers_file)
        if columns != MODULE_COLUMNS:
            problems.append('%s/outputs/numbers.csv has the columns %s, expected %s'
                            % (module_folder, columns, MODULE_COLUMNS))
            continue
        # Module 06 also writes outputs/numbers_shared_concepts.csv (L3-065, test T19b; code/shared_concepts_test.py)
        shared_concepts_file = os.path.join(RELEASE_ROOT, module_folder, 'outputs', 'numbers_shared_concepts.csv')
        if os.path.exists(shared_concepts_file):
            extra_columns, extra_rows = read_csv_rows(shared_concepts_file)
            if extra_columns != MODULE_COLUMNS:
                problems.append('%s/outputs/numbers_shared_concepts.csv has the columns %s'
                                % (module_folder, extra_columns))
            else:
                rows = rows + extra_rows
        for row in rows:
            ledger_id = row['ledger_id']
            if ledger_id not in ledger_by_id:
                problems.append('%s/outputs/numbers.csv: %s is not a ledger id' % (module_folder, ledger_id))
                continue
            if ledger_id in module_rows_by_id:
                problems.append('%s is written by two modules: %s and %s'
                                % (ledger_id, module_rows_by_id[ledger_id][0], module_folder))
                continue
            module_rows_by_id[ledger_id] = (module_folder, row)

    # 3. One output row per ledger item, in ledger order.
    output_rows = []
    rejudged_differently = []
    provided_not_matching = []
    for ledger_row in ledger_rows:
        ledger_id = ledger_row['id']
        output_row = {
            'id': ledger_id,
            'location': ledger_row['location'],
            'quote': ledger_row['quote'],
            'paper_value': ledger_row['paper_value'],
            'computed_value': '',
            'match': '',
            'match_type': ledger_row['match_type'],
            'script': '',
            'mode': ledger_row['release_mode'],
            'same_as': ledger_row['same_as'],
            'note': '',
        }
        notes = []

        if ledger_id in module_rows_by_id:
            # 3a. The module's own row for this location.
            module_folder, module_row = module_rows_by_id[ledger_id]
            output_row['paper_value'] = module_row['paper_value']
            output_row['computed_value'] = module_row['computed_value']
            output_row['match'] = module_row['match']
            output_row['match_type'] = module_row['match_type']
            output_row['script'] = release_relative_script(module_row['script'], module_folder)
            output_row['mode'] = module_row['mode']
            if module_row['note']:
                notes.append(module_row['note'])
            # The module judges against the printed value of the ledger (numbers_table.py reads it there).
            if module_row['paper_value'] != ledger_row['paper_value'] and ledger_row['paper_value'] != '':
                problems.append('%s (%s): the module judged against "%s", the ledger has "%s"'
                                % (ledger_id, module_folder, module_row['paper_value'], ledger_row['paper_value']))
            # Judge again with the shared rules, as a check of the module's own judgment (listed on screen).
            if module_row['match'] in JUDGED_MATCHES:
                judged_here = numbers_table.judge(module_row['paper_value'], module_row['computed_value'],
                                                  module_row['match_type'])
                if judged_here != module_row['match']:
                    rejudged_differently.append((ledger_id, module_folder, module_row['match'], judged_here))
            # A provided value that prints as a number: does it round to the printed value? (listed on screen)
            if module_row['match'] == 'provided' and every_printed_part_is_a_number(module_row['paper_value']):
                judged_here = numbers_table.judge(module_row['paper_value'], module_row['computed_value'], 'round')
                if judged_here == 'no':
                    provided_not_matching.append((ledger_id, module_folder))

        elif ledger_row['module'] == 'none':
            # 3b. No computation behind it (a value from the literature, a registration number).
            output_row['match'] = 'none'
            output_row['match_type'] = 'none'
            output_row['mode'] = ledger_row['release_mode'] or 'none'
            notes.append('no computation behind it (a value from the literature, a registration number or a '
                         'summary statement)')

        elif ledger_row['module_folder'] in modules_not_run:
            # 3c. The module has not been run yet.
            output_row['match'] = 'pending'
            notes.append('%s has not written outputs/numbers.csv yet (run the module)' % ledger_row['module_folder'])

        elif ledger_row['same_as'] and ledger_row['same_as'] in module_rows_by_id:
            # 3d. An item no module wrote, with same_as filled: the computed value of that row (one level only).
            owner_folder, owner_row = module_rows_by_id[ledger_row['same_as']]
            output_row['computed_value'] = owner_row['computed_value']
            output_row['script'] = release_relative_script(owner_row['script'], owner_folder)
            output_row['match'] = numbers_table.judge(ledger_row['paper_value'], owner_row['computed_value'],
                                                      ledger_row['match_type'])
            notes.append('computed value of %s (the row named in same_as; module %s wrote no row for this item)'
                         % (ledger_row['same_as'], ledger_row['module_folder']))

        else:
            # 3e. The module ran but wrote no row for this item.
            output_row['match'] = 'pending'
            notes.append('%s/outputs/numbers.csv has no row for this item' % ledger_row['module_folder'])
            problems.append('%s (module %s) has no row in its numbers.csv' % (ledger_id, ledger_row['module_folder']))

        output_row['note'] = '; '.join(notes)
        output_rows.append(output_row)

    # 4. Write reported_numbers.csv.
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for output_row in output_rows:
            writer.writerow(output_row)

    # 5. Summary.
    counts = {}
    for output_row in output_rows:
        counts[output_row['match']] = counts.get(output_row['match'], 0) + 1
    print('collect_numbers.py: wrote reported_numbers.csv, %d ledger items' % len(output_rows))
    for match_word in SUMMARY_ORDER:
        print('  %-18s %4d' % (match_word, counts.get(match_word, 0)))
    for match_word in sorted(counts):
        if match_word not in SUMMARY_ORDER:
            print('  %-18s %4d   (unexpected match word)' % (match_word, counts[match_word]))

    print('\nRows judged no (the computed value does not agree with the printed one), with the module that '
          'computes them:')
    mismatch_count = 0
    for output_row in output_rows:
        if output_row['match'] != 'no':
            continue
        mismatch_count += 1
        module_folder = module_rows_by_id[output_row['id']][0] if output_row['id'] in module_rows_by_id else ''
        print('  %s  %s  printed "%s", computed "%s"' % (output_row['id'], module_folder,
                                                         output_row['paper_value'], output_row['computed_value']))
    if mismatch_count == 0:
        print('  none')

    print('\nProvided values that do not round to the printed value:')
    if provided_not_matching:
        for ledger_id, module_folder in provided_not_matching:
            print('  %s  %s' % (ledger_id, module_folder))
    else:
        print('  none')

    if rejudged_differently:
        print('\nRows judged differently here than by their module (check them):')
        for ledger_id, module_folder, module_match, match_here in rejudged_differently:
            print('  %s  %s  module: %s, here: %s' % (ledger_id, module_folder, module_match, match_here))
    if modules_not_run:
        print('\nModules without outputs/numbers.csv (not run yet): %s' % ', '.join(modules_not_run))
    if problems:
        print('\nProblems:')
        for problem in problems:
            print('  ' + problem)
        sys.exit(1)


if __name__ == '__main__':
    main()
