function design = check_task_design()
%CHECK_TASK_DESIGN  The recognition task of the three image-set experiments, read from the trial files.
%
%   What it does
%     Counts, for every session of the three image sets, the study/test blocks, the study items
%     and test trials per block, the old and new test items per block, and the responses used on
%     the rating scale (1 = sure old to 6 = sure new). The Methods (Recognition Memory Task)
%     give 16 study/test blocks, each with a study list of 40 images and then 40 test trials
%     (20 old, 20 new), and a 6-point scale.
%
%   Inputs:  data/maintained_trials.csv, changed_trials.csv, reduced_trials.csv (all 602 sessions)
%   Output:  outputs/task_design_check.csv (one row per set: the smallest and largest count over
%            sessions and blocks)
%   Returns a struct with the counts (the same in every session and block, or an error).

paths = release_paths(mfilename('fullpath'));
setNames = {'maintained', 'changed', 'reduced'};

rows = cell(numel(setNames), 1);
allBlockCounts = [];
allStudyCounts = [];
allTestCounts = [];
allOldCounts = [];
allNewCounts = [];
allResponses = [];
for setIndex = 1:numel(setNames)
    trialFile = fullfile(paths.data, [setNames{setIndex} '_trials.csv']);
    options = detectImportOptions(trialFile, 'Delimiter', ',');
    options = setvartype(options, {'participant', 'phase', 'image_filename', 'concept'}, 'string');
    options = setvartype(options, {'block', 'trial_in_block', 'things_image_index', 'image_number', ...
        'is_new', 'response_1to6', 'rt_ms'}, 'double');
    trials = readtable(trialFile, options);

    participants = unique(trials.participant);
    blockCounts = zeros(numel(participants), 1);
    studyCounts = [];
    testCounts = [];
    oldCounts = [];
    newCounts = [];
    for participantIndex = 1:numel(participants)
        own = trials(trials.participant == participants(participantIndex), :);
        blocks = unique(own.block);
        blockCounts(participantIndex) = numel(blocks);
        for blockIndex = 1:numel(blocks)
            inBlock = own(own.block == blocks(blockIndex), :);
            isStudy = inBlock.phase == "study";
            isTest = inBlock.phase == "test";
            studyCounts(end + 1, 1) = sum(isStudy); %#ok<AGROW>
            testCounts(end + 1, 1) = sum(isTest); %#ok<AGROW>
            oldCounts(end + 1, 1) = sum(isTest & inBlock.is_new == 0); %#ok<AGROW>
            newCounts(end + 1, 1) = sum(isTest & inBlock.is_new == 1); %#ok<AGROW>
        end
    end
    responses = unique(trials.response_1to6(trials.phase == "test"));

    rows{setIndex} = {setNames{setIndex}, numel(participants), min(blockCounts), max(blockCounts), ...
        min(studyCounts), max(studyCounts), min(testCounts), max(testCounts), min(oldCounts), max(oldCounts), ...
        min(newCounts), max(newCounts), numel(responses), min(responses), max(responses)};
    fprintf('%-10s %d sessions: blocks %d to %d; per block study %d to %d, test %d to %d (old %d to %d, new %d to %d); responses %d to %d\n', ...
        setNames{setIndex}, numel(participants), min(blockCounts), max(blockCounts), min(studyCounts), max(studyCounts), ...
        min(testCounts), max(testCounts), min(oldCounts), max(oldCounts), min(newCounts), max(newCounts), ...
        min(responses), max(responses));

    allBlockCounts = [allBlockCounts; blockCounts]; %#ok<AGROW>
    allStudyCounts = [allStudyCounts; studyCounts]; %#ok<AGROW>
    allTestCounts = [allTestCounts; testCounts]; %#ok<AGROW>
    allOldCounts = [allOldCounts; oldCounts]; %#ok<AGROW>
    allNewCounts = [allNewCounts; newCounts]; %#ok<AGROW>
    allResponses = union(allResponses, responses);
end

designTable = cell2table(vertcat(rows{:}), 'VariableNames', {'image_set', 'sessions', 'blocks_min', 'blocks_max', ...
    'study_items_per_block_min', 'study_items_per_block_max', 'test_trials_per_block_min', 'test_trials_per_block_max', ...
    'old_test_items_per_block_min', 'old_test_items_per_block_max', 'new_test_items_per_block_min', ...
    'new_test_items_per_block_max', 'response_values_used', 'lowest_response', 'highest_response'});
writetable(designTable, fullfile(paths.outputs, 'task_design_check.csv'));

% the design must be the same in every session and every block
isConstant = @(counts) min(counts) == max(counts);
if ~(isConstant(allBlockCounts) && isConstant(allStudyCounts) && isConstant(allTestCounts) && ...
        isConstant(allOldCounts) && isConstant(allNewCounts))
    error('check_task_design:NotConstant', 'the design differs between sessions or blocks; see outputs/task_design_check.csv');
end
design = struct();
design.blocksPerSession = allBlockCounts(1);
design.studyItemsPerBlock = allStudyCounts(1);
design.testTrialsPerBlock = allTestCounts(1);
design.oldItemsPerBlock = allOldCounts(1);
design.newItemsPerBlock = allNewCounts(1);
design.ratingScalePoints = numel(allResponses);
design.numberOfSessions = numel(allBlockCounts);
end
