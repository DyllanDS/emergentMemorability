function flow = participant_flow()
%PARTICIPANT_FLOW  Sessions, exclusions by rule and kept participants of the memorability inversion
%   experiment (Methods, Experimental Procedures and Analysis: the exclusion criteria and the
%   participants; test T54).
%
%   What it does
%     Applies the exclusion rules of the Methods to the 54 sessions, through
%     shared/matlab/apply_participant_exclusions.m (the same function inversion_analysis.m uses):
%       (1) one response key on more than 80 percent of the 80 test trials          (same key)
%       (2) more than 20 percent of test trials faster than 200 ms                  (too fast)
%           or slower than 7,500 ms                                                 (too slow)
%       (3) hit rate minus false alarm rate below 0.05 ("old" = responses 1 to 3)  (low CR)
%     then drops test trials faster than 200 ms or slower than 10,000 ms, then keeps the first 50
%     participants (with these sessions the cap removes nobody: 54 - 4 = 50).
%     It recomputes every participant's shares from the trials and checks the flags against the
%     thresholds listed below, so these are the thresholds the exclusion function applied.
%
%   Inputs:  data/inversion_trials.csv (through shared/matlab/load_inversion_trials.m)
%   Outputs: outputs/participant_flow.csv (the T54 row: sessions before exclusion, participants
%            meeting each rule, meeting more than one rule, excluded, removed by the cap, kept)
%            outputs/participant_exclusion_flags.csv (one row per session: the shares behind each
%            rule, one column per rule, excluded)
%   Returns a struct with the counts and the rules.
%   No random numbers.

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
rules.cap = 50;                     % keep the first 50 participants after the exclusions

allTrials = load_inversion_trials();
exclusionOptions = {'subjectColumn', 'subjectNumber', 'isNewColumn', 'isNew', 'datasetColumn', '', 'cap', rules.cap};
[keptTrials, report] = apply_participant_exclusions(allTrials, exclusionOptions{:}, 'slowRtMs', rules.slowRtMs);
flags = report.flags;
numberOfSessions = height(flags);

%% Check the flags against the thresholds, recomputing the shares from the trials
for sessionIndex = 1:numberOfSessions
    own = allTrials(allTrials.subjectNumber == str2double(flags.subject(sessionIndex)), :);
    fastShare = mean(own.rt < rules.fastRtMs);
    slowShare = mean(own.rt > rules.slowRtMs);
    keyShare = max(histcounts(own.response, 0.5:1:6.5)) / height(own);
    hitRate = mean(own.response(own.isNew == 0) <= 3);
    falseAlarmRate = mean(own.response(own.isNew == 1) <= 3);
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
% the trial filter: kept trials lie inside the window, and every kept participant's trials outside
% it were dropped
keptParticipants = unique(keptTrials.subjectNumber);
trialsOfKeptParticipants = allTrials(ismember(allTrials.subjectNumber, keptParticipants), :);
outsideWindow = trialsOfKeptParticipants.rt < rules.trialMinRtMs | trialsOfKeptParticipants.rt > rules.trialMaxRtMs;
if min(keptTrials.rt) < rules.trialMinRtMs || max(keptTrials.rt) > rules.trialMaxRtMs || ...
        height(keptTrials) ~= height(trialsOfKeptParticipants) - sum(outsideWindow)
    error('participant_flow:TrialFilter', 'the trial filter does not match the window listed in this function');
end
fprintf('flags of all %d sessions follow the listed thresholds; the trial filter (%d to %d ms) removed %d of the %d test trials of the kept participants\n', ...
    numberOfSessions, rules.trialMinRtMs, rules.trialMaxRtMs, sum(outsideWindow), height(trialsOfKeptParticipants));

%% Counts
% the S-id of every session (flags.subject holds the subject number as text)
participantOfSession = strings(numberOfSessions, 1);
for sessionIndex = 1:numberOfSessions
    own = allTrials(allTrials.subjectNumber == str2double(flags.subject(sessionIndex)), :);
    participantOfSession(sessionIndex) = own.participant(1);
end
numberOfRulesMet = flags.sameKey + flags.tooFast + flags.tooSlow + flags.lowCr;
flow = struct();
flow.processed = numberOfSessions;
flow.sameKey = sum(flags.sameKey);
flow.tooFast = sum(flags.tooFast);
flow.tooSlow = sum(flags.tooSlow);
flow.lowCr = sum(flags.lowCr);
flow.moreThanOneRule = sum(numberOfRulesMet > 1);
flow.excluded = sum(flags.excluded);
flow.removedByCap = numel(report.removedByCap);
flow.kept = numel(keptParticipants);
flow.excludedParticipants = participantOfSession(flags.excluded)';
flow.trialsRemovedByFilter = sum(outsideWindow);
flow.trialsOfKeptParticipants = height(trialsOfKeptParticipants);
flow.rules = rules;
if flow.processed - flow.excluded - flow.removedByCap ~= flow.kept
    error('participant_flow:Counts', 'processed minus excluded minus removed by the cap is not the kept count');
end
fprintf(['memorability inversion experiment: %d sessions before exclusion; same key %d, too fast %d, too slow %d, ' ...
    'CR < 0.05 %d (more than one rule %d); excluded %d (%s), removed by the cap %d, kept %d\n'], ...
    flow.processed, flow.sameKey, flow.tooFast, flow.tooSlow, flow.lowCr, flow.moreThanOneRule, ...
    flow.excluded, strjoin(flow.excludedParticipants, ', '), flow.removedByCap, flow.kept);

%% outputs/participant_flow.csv (one row; the columns of the other experiments' flow tables)
ruleText = sprintf('exclusion rules of the Methods (slow rule %d ms); trials kept %d to %d ms; first %d participants kept', ...
    rules.slowRtMs, rules.trialMinRtMs, rules.trialMaxRtMs, rules.cap);
flowTable = table({'memorability inversion experiment'}, {''}, flow.processed, flow.sameKey, ...
    flow.tooFast, flow.tooSlow, flow.lowCr, flow.moreThanOneRule, flow.excluded, flow.removedByCap, flow.kept, {ruleText}, ...
    'VariableNames', {'experiment', 'image_set', 'processed', 'excluded_same_key', 'excluded_fast', ...
    'excluded_slow', 'excluded_low_cr', 'excluded_more_than_one_rule', 'excluded_total', 'removed_by_cap', 'kept', 'rules'});
writetable(flowTable, fullfile(paths.outputs, 'participant_flow.csv'));

%% outputs/participant_exclusion_flags.csv (one row per session)
flagTable = table(participantOfSession, flags.oneKeyShare, flags.fastShare, flags.slowShare, ...
    flags.hitMinusFa, flags.sameKey, flags.tooFast, flags.tooSlow, flags.lowCr, flags.excluded, ...
    'VariableNames', {'participant', 'share_one_key', 'share_faster_200ms', 'share_slower_7500ms', ...
    'hit_rate_minus_fa_rate', 'same_key', 'too_fast', 'too_slow', 'low_cr', 'excluded'});
writetable(flagTable, fullfile(paths.outputs, 'participant_exclusion_flags.csv'));
end
