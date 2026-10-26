function rows = write_numbers_csv(rows, outputsFolder, defaultScript)
%WRITE_NUMBERS_CSV  Write a module's outputs/numbers.csv (MATLAB twin of shared/python/numbers_table.py).
%
%   rows = write_numbers_csv(rows, outputsFolder, defaultScript)
%
%   rows: struct array, one element per ledger item (and one per duplicate location), with fields
%       ledger_id       e.g. 'L2-014'
%       computed_value  a number, a vector (one value per printed part) or text, from the code
%       mode            'rerun', 'provided', ...
%       note            (optional) text
%       script, match_type, paper_value, location, quote   (optional; default from the ledger)
%   The paper's printed value, the location, the quote and the match type come from
%   shared/ledger/paper_numbers.csv. The match is judged as in numbers_table.py:
%       round        |computed - printed| <= half a unit of the printed last digit
%       exact, count computed == printed
%       less_than    printed '< x': computed < x (also any printed part starting with '<')
%       range        printed 'a to b': both ends judged as round (computed = [a b])
%       sign         same sign
%       provided, pending, documented, none: not judged (match = that word)
%   Writes <outputsFolder>/numbers.csv with the columns ledger_id, location, quote, paper_value,
%   computed_value, match, match_type, script, mode, note, and returns the completed rows.

sharedMatlabFolder = fileparts(mfilename('fullpath'));
ledgerFile = fullfile(sharedMatlabFolder, '..', 'ledger', 'paper_numbers.csv');
ledgerOptions = detectImportOptions(ledgerFile, 'Delimiter', ',', 'TextType', 'char');
ledgerOptions = setvartype(ledgerOptions, ledgerOptions.VariableNames, 'char');
ledger = readtable(ledgerFile, ledgerOptions);

for rowIndex = 1:numel(rows)
    row = rows(rowIndex);
    ledgerRow = find(strcmp(ledger.id, row.ledger_id), 1);
    if isempty(ledgerRow)
        error('write_numbers_csv:UnknownId', 'unknown ledger id %s', row.ledger_id);
    end
    rows(rowIndex).location = fieldOrDefault(row, 'location', ledger.location{ledgerRow});
    rows(rowIndex).quote = fieldOrDefault(row, 'quote', ledger.quote{ledgerRow});
    rows(rowIndex).paper_value = fieldOrDefault(row, 'paper_value', ledger.paper_value{ledgerRow});
    rows(rowIndex).match_type = fieldOrDefault(row, 'match_type', ledger.match_type{ledgerRow});
    rows(rowIndex).script = fieldOrDefault(row, 'script', defaultScript);
    rows(rowIndex).note = fieldOrDefault(row, 'note', '');
    rows(rowIndex).match = judgeValue(rows(rowIndex).paper_value, row.computed_value, rows(rowIndex).match_type);
    rows(rowIndex).computed_text = computedText(row.computed_value);
end

if ~exist(outputsFolder, 'dir')
    mkdir(outputsFolder);
end
outputFile = fullfile(outputsFolder, 'numbers.csv');
handle = fopen(outputFile, 'w', 'n', 'UTF-8');
fprintf(handle, 'ledger_id,location,quote,paper_value,computed_value,match,match_type,script,mode,note\n');
for rowIndex = 1:numel(rows)
    row = rows(rowIndex);
    fields = {row.ledger_id, row.location, row.quote, row.paper_value, row.computed_text, row.match, ...
        row.match_type, row.script, row.mode, row.note};
    for fieldIndex = 1:numel(fields)
        fields{fieldIndex} = csvField(fields{fieldIndex});
    end
    fprintf(handle, '%s\n', strjoin(fields, ','));
end
fclose(handle);
fprintf('wrote %s (%d rows)\n', outputFile, numel(rows));
end

function value = fieldOrDefault(row, name, default)
if isfield(row, name) && ~isempty(row.(name))
    value = row.(name);
else
    value = default;
end
end

function text = csvField(value)
text = char(string(value));
if any(text == ',') || any(text == '"') || any(text == newline)
    text = ['"' strrep(text, '"', '""') '"'];
end
end

function text = computedText(value)
if isnumeric(value) || islogical(value)
    parts = arrayfun(@(v) sprintf('%.6g', v), double(value), 'UniformOutput', false);
    text = strjoin(parts, '; ');
else
    text = char(string(value));
end
end

function match = judgeValue(paperValue, computedValue, matchType)
notJudged = {'provided', 'pending', 'documented', 'none'};
if any(strcmp(matchType, notJudged))
    match = matchType;
    return;
end
printedParts = strtrim(strsplit(char(paperValue), ';'));
printedParts = printedParts(~cellfun(@isempty, printedParts));
if strcmp(matchType, 'range') && numel(printedParts) == 1
    match = yesNo(judgeOne(printedParts{1}, computedValue, 'range'));
    return;
end
if isnumeric(computedValue) || islogical(computedValue)
    computedParts = num2cell(double(computedValue));
else
    computedParts = strtrim(strsplit(char(computedValue), ';'));
end
if numel(printedParts) ~= numel(computedParts)
    match = 'no';
    return;
end
allMatch = true;
for partIndex = 1:numel(printedParts)
    allMatch = allMatch && judgeOne(printedParts{partIndex}, computedParts{partIndex}, matchType);
end
match = yesNo(allMatch);
end

function text = yesNo(flag)
if flag
    text = 'yes';
else
    text = 'no';
end
end

function ok = judgeOne(printed, computed, matchType)
printed = strtrim(strrep(strrep(strrep(char(printed), char(8722), '-'), char(8211), '-'), char(8212), '-'));
lowerPrinted = lower(printed);
if startsWith(printed, '<') || startsWith(lowerPrinted, 'p <') || startsWith(lowerPrinted, 'p<')
    pieces = strsplit(printed, '<');
    ok = toNumber(computed) < parseNumber(pieces{2});
    return;
end
if strcmp(matchType, 'less_than')
    ok = toNumber(computed) < parseNumber(printed);
    return;
end
if strcmp(matchType, 'range')
    ends = strtrim(strsplit(printed, ' to '));
    if ischar(computed) || isstring(computed)
        computed = cellfun(@parseNumber, strtrim(strsplit(char(computed), ' to ')));
    end
    ok = numel(ends) == 2 && numel(computed) == 2 && judgeOne(ends{1}, computed(1), 'round') && ...
        judgeOne(ends{2}, computed(2), 'round');
    return;
end
printedNumber = parseNumber(printed);
if isnan(printedNumber)
    ok = strcmpi(strtrim(char(string(computed))), printed);
    return;
end
computedNumber = toNumber(computed);
if isnan(computedNumber)
    ok = false;
    return;
end
if strcmp(matchType, 'sign')
    ok = sign(computedNumber) == sign(printedNumber);
elseif any(strcmp(matchType, {'exact', 'count'}))
    ok = computedNumber == printedNumber;
else
    decimalPart = regexp(strrep(strrep(printed, '%', ''), ',', ''), '\.(\d+)', 'tokens', 'once');
    if isempty(decimalPart)
        decimals = 0;
    else
        decimals = numel(decimalPart{1});
    end
    ok = abs(computedNumber - printedNumber) <= 0.5 * 10 ^ (-decimals) + 1e-9;
end
end

function number = toNumber(value)
if isnumeric(value) || islogical(value)
    number = double(value);
else
    number = parseNumber(value);
end
end

function number = parseNumber(text)
text = lower(strtrim(char(string(text))));
text = regexprep(text, '^(p|r|rho|d''|d|t|n)\s*=\s*', '');
text = strtrim(strrep(strrep(text, ',', ''), '%', ''));
words = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', ...
    'eleven', 'twelve', 'fifteen', 'twenty'};
values = [0 1 2 3 4 5 6 7 8 9 10 11 12 15 20];
wordIndex = find(strcmp(words, text), 1);
if ~isempty(wordIndex)
    number = values(wordIndex);
    return;
end
number = str2double(text);
end
