function [T, report] = apply_participant_exclusions(T, varargin)
%APPLY_PARTICIPANT_EXCLUSIONS  The participant and trial exclusions of the Methods (Experimental
%   Procedures and Analysis).
%
%   [T, report] = apply_participant_exclusions(T)                     image-set table (subject, ...)
%   [T, report] = apply_participant_exclusions(T, 'subjectColumn', 'subjectNumber', ...
%       'isNewColumn', 'isNew', 'datasetColumn', '', 'cap', 50)      inversion table
%
%   The steps, in order:
%   1. A participant is excluded if, over all of their test trials,
%        one response key was used on more than 80 percent of trials       (same key)
%        more than 20 percent of trials were faster than 200 ms            (too fast)
%        more than 20 percent of trials were slower than 7,500 ms          (too slow)
%        hit rate minus false alarm rate was below 0.05, old = 1 to 3      (low CR)
%   2. Trials faster than 200 ms or slower than 10,000 ms are dropped (rt >= 200 & rt <= 10,000).
%   3. The first 'cap' remaining participants are kept, per dataset when datasetColumn is given
%      (unique() order = sorted ids). With the shipped participant sets this removes nobody.
%
%   Options (name, value):
%     'subjectColumn'   default 'subject'        ('subjectNumber' for the inversion table)
%     'isNewColumn'     default 'is_new'         ('isNew' for the inversion table)
%     'responseColumn'  default 'response'
%     'rtColumn'        default 'rt'
%     'datasetColumn'   default 'dataset_name'   ('' = one group)
%     'slowRtMs'        default 7500
%     'cap'             default 150              (50 for the inversion experiment)
%
%   report: struct with
%     flags            one row per participant: subject, dataset, oneKeyShare, fastShare, slowShare,
%                      hitMinusFa, sameKey, tooFast, tooSlow, lowCr, excluded
%     badSubjects      the excluded subjects (cell of char for string ids, numbers for numeric ids),
%                      in sorted id order (the order of unique())
%     removedByCap     subjects removed by the cap
%     nBefore, nExcluded, nAfter

options = struct('subjectColumn', 'subject', 'isNewColumn', 'is_new', 'responseColumn', 'response', ...
    'rtColumn', 'rt', 'datasetColumn', 'dataset_name', 'slowRtMs', 7500, 'cap', 150);
for argumentIndex = 1:2:numel(varargin)
    options.(varargin{argumentIndex}) = varargin{argumentIndex + 1};
end
minRT_excl = 200;
maxRT_excl = options.slowRtMs;
responseVariabilityThreshold = 0.8;
performanceThreshold = 0.05;

subjects = T.(options.subjectColumn);
uniqueSubjects = unique(subjects);
numberOfSubjects = numel(uniqueSubjects);
useStrings = ~isnumeric(uniqueSubjects);

% ---- step 1: the participant rules ----
flagRows = cell(numberOfSubjects, 11);
if useStrings
    badSubjects = {};
else
    badSubjects = [];
end
for subjectIndex = 1:numberOfSubjects
    subjectID = uniqueSubjects(subjectIndex);
    subjectTrials = T(subjects == subjectID, :);
    numberOfTrials = height(subjectTrials);

    responseDistribution = hist(subjectTrials.(options.responseColumn), 1:6) / numberOfTrials; %#ok<HIST>
    maxResponseProportion = max(responseDistribution);
    fastRTs = sum(subjectTrials.(options.rtColumn) < minRT_excl) / numberOfTrials;
    slowRTs = sum(subjectTrials.(options.rtColumn) > maxRT_excl) / numberOfTrials;

    oldTrials = subjectTrials(subjectTrials.(options.isNewColumn) == 0, :);
    newTrials = subjectTrials(subjectTrials.(options.isNewColumn) == 1, :);
    hitRate = mean(oldTrials.(options.responseColumn) <= 3, 'omitnan');
    faRate = mean(newTrials.(options.responseColumn) <= 3, 'omitnan');
    performance = hitRate - faRate;

    sameKey = maxResponseProportion > responseVariabilityThreshold;
    tooFast = fastRTs > 0.2;
    tooSlow = slowRTs > 0.2;
    lowCr = performance < performanceThreshold;
    isBad = sameKey || tooFast || tooSlow || lowCr;
    if isBad
        if useStrings
            badSubjects{end + 1} = char(subjectID); %#ok<AGROW>
        else
            badSubjects(end + 1) = subjectID; %#ok<AGROW>
        end
    end
    if isempty(options.datasetColumn)
        datasetName = "";
    else
        datasetName = string(subjectTrials.(options.datasetColumn)(1));
    end
    flagRows(subjectIndex, :) = {string(subjectID), datasetName, maxResponseProportion, fastRTs, slowRTs, ...
        performance, sameKey, tooFast, tooSlow, lowCr, isBad};
end
flags = cell2table(flagRows, 'VariableNames', {'subject', 'dataset', 'oneKeyShare', 'fastShare', ...
    'slowShare', 'hitMinusFa', 'sameKey', 'tooFast', 'tooSlow', 'lowCr', 'excluded'});

if ~isempty(badSubjects)
    badSubjectMask = false(height(T), 1);
    for k = 1:numel(badSubjects)
        if useStrings
            badSubjectMask = badSubjectMask | strcmp(T.(options.subjectColumn), badSubjects{k});
        else
            badSubjectMask = badSubjectMask | (T.(options.subjectColumn) == badSubjects(k));
        end
    end
    T(badSubjectMask, :) = [];
end

% ---- step 2: the trial filter ----
T = T(T.(options.rtColumn) >= 200 & T.(options.rtColumn) <= 10000, :);

% ---- step 3: the cap (sorted id order, per dataset) ----
if isempty(options.datasetColumn)
    groupNames = {''};
else
    groupNames = unique(cellstr(string(T.(options.datasetColumn))), 'stable');
end
if useStrings
    removedByCap = {};
else
    removedByCap = [];
end
for groupIndex = 1:numel(groupNames)
    if isempty(options.datasetColumn)
        inGroup = true(height(T), 1);
    else
        inGroup = string(T.(options.datasetColumn)) == string(groupNames{groupIndex});
    end
    uniqueSubjectsInGroup = unique(T.(options.subjectColumn)(inGroup));
    if numel(uniqueSubjectsInGroup) > options.cap
        subjectsExcludedByLimit = uniqueSubjectsInGroup(options.cap + 1:end);
        for j = 1:numel(subjectsExcludedByLimit)
            T(T.(options.subjectColumn) == subjectsExcludedByLimit(j), :) = [];
            if useStrings
                removedByCap{end + 1} = char(subjectsExcludedByLimit(j)); %#ok<AGROW>
            else
                removedByCap(end + 1) = subjectsExcludedByLimit(j); %#ok<AGROW>
            end
        end
    end
end

report = struct();
report.flags = flags;
report.badSubjects = badSubjects;
report.removedByCap = removedByCap;
report.nBefore = numberOfSubjects;
report.nExcluded = numel(badSubjects);
report.nAfter = numel(unique(T.(options.subjectColumn)));
fprintf('participants: %d before exclusion, %d excluded, %d removed by the cap of %d, %d kept\n', ...
    report.nBefore, report.nExcluded, numel(removedByCap), options.cap, report.nAfter);
end
