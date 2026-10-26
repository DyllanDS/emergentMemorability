function flow = participant_flow()
%PARTICIPANT_FLOW  How many sessions each image set had, who was excluded and why, who was kept.
%
%   What it does
%     Applies the exclusion criteria of the Methods to the 602 sessions of the three image sets
%     (maintained 205, changed 195, reduced 202) through shared/matlab/apply_participant_exclusions.m,
%     the same function the Fig. 6 and Fig. 7 analyses use:
%       (1) one response key on more than 80 percent of test trials (same key)
%       (2) more than 20 percent of test trials faster than 200 ms (too fast) or slower than
%           7,500 ms (too slow)
%       (3) hit rate minus false alarm rate below 0.05 ("old" = responses 1 to 3)
%     then drops trials faster than 200 ms or slower than 10,000 ms, then keeps the first 150
%     participants of each set in id order (with these sessions the cap removes nobody).
%     It also checks every participant's flags against the thresholds listed below, so the
%     numbers in outputs/numbers.csv are the thresholds the code applied.
%
%   Inputs:  data/maintained_trials.csv, changed_trials.csv, reduced_trials.csv
%   Outputs: outputs/participant_flow.csv (one row per set and a total row: sessions,
%            participants meeting each rule, meeting more than one rule, excluded, removed by the
%            cap, kept)
%            outputs/participant_exclusion_flags.csv (one row per session: the shares behind each
%            rule, one column per rule, excluded)
%   Returns a struct with the counts.

paths = release_paths(mfilename('fullpath'));

% The exclusion rules of the Methods, as the code applies them
rules = struct();
rules.oneKeyShareLimit = 0.80;      % (1) more than 80 percent of test trials with one key
rules.rtShareLimit = 0.20;          % (2) more than 20 percent of test trials too fast or too slow
rules.fastRtMs = 200;               %     too fast: under 200 ms
rules.slowRtMs = 7500;              %     too slow: over 7,500 ms
rules.minHitMinusFa = 0.05;         % (3) hit rate minus false alarm rate below 0.05
rules.trialMinRtMs = 200;           % trial filter after the exclusions: keep 200 to 10,000 ms
rules.trialMaxRtMs = 10000;
rules.capPerSet = 150;              % first 150 participants per set, in id order

concInfo = load_concept_info('human');
setNames = {'maintained', 'changed', 'reduced'};
datasetNames = {'representativeImages', 'flipMemorabilityImages', 'equatedImages'};
allTrials = load_image_set_trials(setNames, concInfo, false);
[keptTrials, report] = apply_participant_exclusions(allTrials, 'slowRtMs', rules.slowRtMs, 'cap', rules.capPerSet);
flags = report.flags;

%% Check the flags against the thresholds, recomputing the shares from the trials
numberOfSessions = height(flags);
for sessionIndex = 1:numberOfSessions
    own = allTrials(allTrials.subject == flags.subject(sessionIndex), :);
    fastShare = mean(own.rt < rules.fastRtMs);
    slowShare = mean(own.rt > rules.slowRtMs);
    keyShare = max(histcounts(own.response, 0.5:1:6.5)) / height(own);
    hitRate = mean(own.response(own.is_new == 0) <= 3);
    falseAlarmRate = mean(own.response(own.is_new == 1) <= 3);
    if abs(fastShare - flags.fastShare(sessionIndex)) > 1e-12 || abs(slowShare - flags.slowShare(sessionIndex)) > 1e-12 || ...
            abs(keyShare - flags.oneKeyShare(sessionIndex)) > 1e-12 || ...
            abs((hitRate - falseAlarmRate) - flags.hitMinusFa(sessionIndex)) > 1e-12
        error('participant_flow:Shares', 'session %s: the shares differ from the exclusion function', flags.subject(sessionIndex));
    end
end
if ~isequal(flags.sameKey, flags.oneKeyShare > rules.oneKeyShareLimit) || ...
        ~isequal(flags.tooFast, flags.fastShare > rules.rtShareLimit) || ...
        ~isequal(flags.tooSlow, flags.slowShare > rules.rtShareLimit) || ...
        ~isequal(flags.lowCr, flags.hitMinusFa < rules.minHitMinusFa)
    error('participant_flow:Rules', 'the exclusion flags do not follow the thresholds listed in this function');
end
% the trial filter: kept trials lie inside the window, and every kept participant's trials
% outside it were dropped
keptParticipants = unique(keptTrials.subject);
trialsOfKeptParticipants = allTrials(ismember(allTrials.subject, keptParticipants), :);
outsideWindow = trialsOfKeptParticipants.rt < rules.trialMinRtMs | trialsOfKeptParticipants.rt > rules.trialMaxRtMs;
if min(keptTrials.rt) < rules.trialMinRtMs || max(keptTrials.rt) > rules.trialMaxRtMs || ...
        height(keptTrials) ~= height(trialsOfKeptParticipants) - sum(outsideWindow)
    error('participant_flow:TrialFilter', 'the trial filter does not match the window listed in this function');
end
fprintf('flags of all %d sessions follow the listed thresholds; trial filter %d to %d ms removed %d of %d trials of the kept participants\n', ...
    numberOfSessions, rules.trialMinRtMs, rules.trialMaxRtMs, sum(outsideWindow), height(trialsOfKeptParticipants));

%% Counts per set
numberOfRulesMet = flags.sameKey + flags.tooFast + flags.tooSlow + flags.lowCr;
flowRows = cell(numel(setNames) + 1, 12);
flow = struct();
for setIndex = 1:numel(setNames) + 1
    if setIndex <= numel(setNames)
        inSet = flags.dataset == string(datasetNames{setIndex});
        setLabel = setNames{setIndex};
        keptInSet = numel(unique(keptTrials.subject(keptTrials.dataset_name == string(datasetNames{setIndex}))));
        removedByCapInSet = sum(ismember(string(report.removedByCap), flags.subject(inSet)));
    else
        inSet = true(numberOfSessions, 1);
        setLabel = 'all three sets';
        keptInSet = numel(keptParticipants);
        removedByCapInSet = numel(report.removedByCap);
    end
    counts = struct();
    counts.processed = sum(inSet);
    counts.sameKey = sum(flags.sameKey(inSet));
    counts.tooFast = sum(flags.tooFast(inSet));
    counts.tooSlow = sum(flags.tooSlow(inSet));
    counts.lowCr = sum(flags.lowCr(inSet));
    counts.moreThanOneRule = sum(numberOfRulesMet(inSet) > 1);
    counts.excluded = sum(flags.excluded(inSet));
    counts.removedByCap = removedByCapInSet;
    counts.kept = keptInSet;
    flowRows(setIndex, :) = {'image-set experiments', setLabel, counts.processed, counts.sameKey, counts.tooFast, ...
        counts.tooSlow, counts.lowCr, counts.moreThanOneRule, counts.excluded, counts.removedByCap, counts.kept, ...
        sprintf('exclusion rules of the Methods; trials kept %d to %d ms; first %d per set', rules.trialMinRtMs, ...
        rules.trialMaxRtMs, rules.capPerSet)};
    if setIndex <= numel(setNames)
        flow.(setNames{setIndex}) = counts;
    else
        flow.total = counts;
    end
    fprintf('%-15s %3d sessions: same key %d, too fast %d, too slow %d, CR < 0.05 %d (more than one rule %d); excluded %d, removed by the cap %d, kept %d\n', ...
        setLabel, counts.processed, counts.sameKey, counts.tooFast, counts.tooSlow, counts.lowCr, ...
        counts.moreThanOneRule, counts.excluded, counts.removedByCap, counts.kept);
end
flowTable = cell2table(flowRows, 'VariableNames', {'experiment', 'image_set', 'processed', 'excluded_same_key', ...
    'excluded_fast', 'excluded_slow', 'excluded_low_cr', 'excluded_more_than_one_rule', 'excluded_total', ...
    'removed_by_cap', 'kept', 'rules'});
writetable(flowTable, fullfile(paths.outputs, 'participant_flow.csv'));

% one row per session
setOfSession = strings(numberOfSessions, 1);
for setIndex = 1:numel(setNames)
    setOfSession(flags.dataset == string(datasetNames{setIndex})) = setNames{setIndex};
end
flagTable = table(flags.subject, setOfSession, flags.oneKeyShare, flags.fastShare, flags.slowShare, flags.hitMinusFa, ...
    flags.sameKey, flags.tooFast, flags.tooSlow, flags.lowCr, flags.excluded, ...
    'VariableNames', {'participant', 'image_set', 'share_one_key', 'share_faster_200ms', 'share_slower_7500ms', ...
    'hit_rate_minus_fa_rate', 'same_key', 'too_fast', 'too_slow', 'low_cr', 'excluded'});
writetable(flagTable, fullfile(paths.outputs, 'participant_exclusion_flags.csv'));

flow.rules = rules;
flow.keptPerSet = [flow.maintained.kept, flow.changed.kept, flow.reduced.kept];
end
