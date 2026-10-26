function design = inversion_design_check()
%INVERSION_DESIGN_CHECK  The design of the memorability inversion experiment, counted from the
%   block lists and trial records in data/ (Fig. 3a to 3c; Methods, Behavioral Validation Experiments).
%
%   What it does
%     Counts each design fact of Fig. 3a to 3c and of the Methods from the files, without assuming
%     any value:
%       - blocks per session; study items and test trials per block, in every block of every
%         session
%       - the make-up of a study list: the 35 controlled context items (the similar items of each
%         critical item, then the non-similar exemplars) and the 5 further studied items,
%         which are critical or maximally dissimilar items (which ones was drawn per participant
%         and block)
%       - the make-up of a test block: the 5 critical items and the 5 maximally dissimilar items
%       - the number of similar items studied with each critical item, from the deployed lists,
%         checked against inversion_design.csv in every block
%       - the quintile of each critical item's published score among the THINGS images that have
%         a score (26,106 of the 26,107 images), with the cuts and the bin rule of the selection code
%         shared/things/provenance/SelectStimToFlipMemorability.m (prctile at 0, 20, 40, 60, 80,
%         100 percent; an image is in bin k when cut(k) <= score < cut(k + 1))
%       - the response values used at test (the rating scale)
%       - how often each critical item was studied (old) and how often it was a lure (new)
%
%   Inputs:  data/inversion_deployed_lists.csv (the study and test lists of the 8 blocks as the
%            experiment used them), data/inversion_design.csv (the 40 critical items),
%            data/inversion_trials.csv (the 54 sessions), shared/things/things_images.csv (the
%            published per-image scores of Kramer et al. 2023)
%   Outputs: outputs/inversion_design_check.csv (one row per fact: the counted value and whether
%            every block and session agrees); returns a struct with the counts
%   No random numbers.

paths = release_paths(mfilename('fullpath'));

%% Read the files
lists = readCsvWithText(fullfile(paths.data, 'inversion_deployed_lists.csv'), ...
    {'phase', 'image_name', 'condition', 'concept', 'old_new', 'similar_to_critical_item'});
criticalItems = readCsvWithText(fullfile(paths.data, 'inversion_design.csv'), {'image_name', 'concept'});
trials = readCsvWithText(fullfile(paths.data, 'inversion_trials.csv'), ...
    {'participant', 'phase', 'image_name', 'concept', 'condition'});
images = readCsvWithText(fullfile(paths.sharedThings, 'things_images.csv'), {'filename', 'rel_path', 'concept'});

checks = cell(0, 4);       % fact, value, every block and session agrees, source

%% 1. The deployed lists: blocks, list lengths, the make-up of each list
blocks = unique(lists.block);
numberOfBlocks = numel(blocks);
studyRowsPerBlock = zeros(numberOfBlocks, 1);
testRowsPerBlock = zeros(numberOfBlocks, 1);
controlledPerBlock = zeros(numberOfBlocks, 1);
similarPerBlock = zeros(numberOfBlocks, 1);
fillersPerBlock = zeros(numberOfBlocks, 1);
furtherStudiedPerBlock = zeros(numberOfBlocks, 1);
criticalTestedPerBlock = zeros(numberOfBlocks, 1);
dissimilarTestedPerBlock = zeros(numberOfBlocks, 1);
similarCountMatchesDesign = true;
for blockIndex = 1:numberOfBlocks
    block = blocks(blockIndex);
    studyRows = lists(lists.block == block & lists.phase == "study", :);
    testRows = lists(lists.block == block & lists.phase == "test", :);
    studyRowsPerBlock(blockIndex) = height(studyRows);
    testRowsPerBlock(blockIndex) = height(testRows);

    controlledRows = studyRows(studyRows.condition == "controlled", :);
    controlledPerBlock(blockIndex) = height(controlledRows);
    similarPerBlock(blockIndex) = sum(controlledRows.similar_to_critical_item ~= "none");
    fillersPerBlock(blockIndex) = sum(controlledRows.similar_to_critical_item == "none");
    furtherStudiedPerBlock(blockIndex) = sum(studyRows.condition == "critical" | studyRows.condition == "maxDissimilar");

    criticalTestedPerBlock(blockIndex) = sum(testRows.condition == "critical");
    dissimilarTestedPerBlock(blockIndex) = sum(testRows.condition == "maxDissimilar");

    % the number of similar items of each critical item of this block, counted from the list
    criticalOfBlock = criticalItems(criticalItems.block == block, :);
    for itemIndex = 1:height(criticalOfBlock)
        countedSimilar = sum(controlledRows.similar_to_critical_item == criticalOfBlock.image_name(itemIndex));
        if countedSimilar ~= criticalOfBlock.n_similar_items(itemIndex)
            similarCountMatchesDesign = false;
        end
        % the similar items include at least one image of the critical item's own concept
        ownConcept = controlledRows.concept(controlledRows.similar_to_critical_item == criticalOfBlock.image_name(itemIndex));
        if countedSimilar > 0 && ~any(ownConcept == criticalOfBlock.concept(itemIndex))
            similarCountMatchesDesign = false;
        end
    end
    % the critical items tested in this block are the five of the design table
    if ~isequal(sort(testRows.image_name(testRows.condition == "critical")), sort(criticalOfBlock.image_name))
        error('inversion_design_check:Critical', 'block %d: the tested critical items differ from inversion_design.csv', block);
    end
end
if ~similarCountMatchesDesign
    error('inversion_design_check:Similar', 'the similar items counted in the deployed lists differ from inversion_design.csv');
end
checks(end + 1, :) = {'blocks in the deployed lists', numberOfBlocks, 'yes', 'inversion_deployed_lists.csv'};
checks(end + 1, :) = agreeRow('study items per block (deployed lists)', studyRowsPerBlock, 'inversion_deployed_lists.csv');
checks(end + 1, :) = agreeRow('test trials per block (deployed lists)', testRowsPerBlock, 'inversion_deployed_lists.csv');
checks(end + 1, :) = agreeRow('controlled context items per study list', controlledPerBlock, 'inversion_deployed_lists.csv');
checks(end + 1, :) = agreeRow('similar items per study list (all five critical items together)', similarPerBlock, 'inversion_deployed_lists.csv');
checks(end + 1, :) = agreeRow('non-similar exemplars among the controlled items', fillersPerBlock, 'inversion_deployed_lists.csv');
checks(end + 1, :) = agreeRow('further studied items (critical or maximally dissimilar) per study list', furtherStudiedPerBlock, 'inversion_deployed_lists.csv');
checks(end + 1, :) = agreeRow('critical items per test block', criticalTestedPerBlock, 'inversion_deployed_lists.csv');
checks(end + 1, :) = agreeRow('maximally dissimilar items per test block', dissimilarTestedPerBlock, 'inversion_deployed_lists.csv');

%% 2. The number of similar items by memorability quintile (the manipulation)
quintiles = unique(criticalItems.memorability_quintile);
similarByQuintile = zeros(1, numel(quintiles));
itemsPerQuintile = zeros(1, numel(quintiles));
for quintileIndex = 1:numel(quintiles)
    inQuintile = criticalItems.memorability_quintile == quintiles(quintileIndex);
    valuesInQuintile = unique(criticalItems.n_similar_items(inQuintile));
    if numel(valuesInQuintile) ~= 1
        error('inversion_design_check:Quintile', 'quintile %d has more than one number of similar items', quintiles(quintileIndex));
    end
    similarByQuintile(quintileIndex) = valuesInQuintile;
    itemsPerQuintile(quintileIndex) = sum(inQuintile);
end
checks(end + 1, :) = {'similar items per critical item, quintiles 1 (lowest) to 5 (highest), in all 8 blocks', ...
    strjoin(string(similarByQuintile), ' / '), yesNo(similarCountMatchesDesign), 'inversion_deployed_lists.csv, inversion_design.csv'};
checks(end + 1, :) = {'critical items per quintile', strjoin(string(itemsPerQuintile), ' / '), 'yes', 'inversion_design.csv'};

%% 3. The quintile of each critical item's published score among all THINGS images
allScores = images.kramer_cr;                      % 26,107 images; NaN for goalpost_08s.jpg
numberOfScoredImages = sum(~isnan(allScores));
numberOfBins = numel(quintiles);
binEdgesPercent = linspace(0, 100, numberOfBins + 1);
binWidthPercent = binEdgesPercent(2) - binEdgesPercent(1);
cuts = prctile(allScores, binEdgesPercent);       % prctile leaves out the NaN
quintileFromScores = zeros(height(criticalItems), 1);
for itemIndex = 1:height(criticalItems)
    score = criticalItems.kramer_cr(itemIndex);
    for binIndex = 1:numberOfBins
        if score >= cuts(binIndex) && score < cuts(binIndex + 1)
            quintileFromScores(itemIndex) = binIndex;
        end
    end
end
quintileMatches = isequal(quintileFromScores, criticalItems.memorability_quintile);
onePerQuintilePerBlock = true;
for blockIndex = 1:numberOfBlocks
    quintilesOfBlock = sort(criticalItems.memorability_quintile(criticalItems.block == blocks(blockIndex)))';
    if ~isequal(quintilesOfBlock, 1:numberOfBins)
        onePerQuintilePerBlock = false;
    end
end
if ~quintileMatches || ~onePerQuintilePerBlock
    error('inversion_design_check:Quintiles', 'the quintiles of the critical items do not follow the published scores');
end
checks(end + 1, :) = {'THINGS images with a published score', numberOfScoredImages, 'yes', 'things_images.csv'};
checks(end + 1, :) = {'quintile cuts of the published scores (20, 40, 60, 80 percent)', ...
    strjoin(compose('%.4f', cuts(2:end - 1)), ' / '), 'yes', 'things_images.csv'};
checks(end + 1, :) = {'every critical item lies in its quintile; one item per quintile in every block', ...
    sprintf('%d items', height(criticalItems)), yesNo(quintileMatches && onePerQuintilePerBlock), 'inversion_design.csv, things_images.csv'};
checks(end + 1, :) = {'published scores of the 40 critical items, lowest to highest', ...
    sprintf('%.4f to %.4f', min(criticalItems.kramer_cr), max(criticalItems.kramer_cr)), 'yes', 'inversion_design.csv'};

%% 4. The 54 sessions: blocks, list lengths, the studied items, the rating scale
studyTrials = trials(trials.phase == "study", :);
testTrials = trials(trials.phase == "test", :);
participants = unique(trials.participant);
numberOfSessions = numel(participants);
blocksPerSession = zeros(numberOfSessions, 1);
studyPerBlock = [];
testPerBlock = [];
controlledStudiedPerBlock = [];
furtherPerBlock = [];
criticalStudiedPerBlock = [];
criticalTestPerBlock = [];
dissimilarTestPerBlock = [];
controlledAreDeployed = true;
for sessionIndex = 1:numberOfSessions
    ownStudy = studyTrials(studyTrials.participant == participants(sessionIndex), :);
    ownTest = testTrials(testTrials.participant == participants(sessionIndex), :);
    sessionBlocks = unique([ownStudy.block; ownTest.block]);
    blocksPerSession(sessionIndex) = numel(sessionBlocks);
    for blockIndex = 1:numel(sessionBlocks)
        block = sessionBlocks(blockIndex);
        blockStudy = ownStudy(ownStudy.block == block, :);
        blockTest = ownTest(ownTest.block == block, :);
        studyPerBlock(end + 1, 1) = height(blockStudy); %#ok<AGROW>
        testPerBlock(end + 1, 1) = height(blockTest); %#ok<AGROW>
        controlledStudied = blockStudy.image_name(blockStudy.condition == "controlled");
        controlledStudiedPerBlock(end + 1, 1) = numel(controlledStudied); %#ok<AGROW>
        furtherPerBlock(end + 1, 1) = sum(blockStudy.condition == "critical" | blockStudy.condition == "maxDissimilar"); %#ok<AGROW>
        criticalStudiedPerBlock(end + 1, 1) = sum(blockStudy.condition == "critical"); %#ok<AGROW>
        criticalTestPerBlock(end + 1, 1) = sum(blockTest.condition == "critical"); %#ok<AGROW>
        dissimilarTestPerBlock(end + 1, 1) = sum(blockTest.condition == "maxDissimilar"); %#ok<AGROW>
        deployedControlled = lists.image_name(lists.block == block & lists.phase == "study" & lists.condition == "controlled");
        if ~isequal(sort(controlledStudied), sort(deployedControlled))
            controlledAreDeployed = false;
        end
    end
end
responseValues = unique(testTrials.response_1to6)';
checks(end + 1, :) = {'sessions (participants before exclusion)', numberOfSessions, 'yes', 'inversion_trials.csv'};
checks(end + 1, :) = agreeRow('blocks per session', blocksPerSession, 'inversion_trials.csv');
checks(end + 1, :) = agreeRow('study items per block (every block of every session)', studyPerBlock, 'inversion_trials.csv');
checks(end + 1, :) = agreeRow('test trials per block (every block of every session)', testPerBlock, 'inversion_trials.csv');
checks(end + 1, :) = agreeRow('controlled items studied per block (every block of every session)', controlledStudiedPerBlock, 'inversion_trials.csv');
checks(end + 1, :) = {'the controlled items each participant studied are the deployed ones', ...
    yesNo(controlledAreDeployed), yesNo(controlledAreDeployed), 'inversion_trials.csv, inversion_deployed_lists.csv'};
checks(end + 1, :) = agreeRow('further studied items (critical or maximally dissimilar) per block', furtherPerBlock, 'inversion_trials.csv');
checks(end + 1, :) = {'critical items among the further studied items of a block (drawn per participant and block)', ...
    strjoin(string(unique(criticalStudiedPerBlock))', ' or '), 'yes', 'inversion_trials.csv'};
checks(end + 1, :) = agreeRow('critical items per test block (every block of every session)', criticalTestPerBlock, 'inversion_trials.csv');
checks(end + 1, :) = agreeRow('maximally dissimilar items per test block (every block of every session)', dissimilarTestPerBlock, 'inversion_trials.csv');
checks(end + 1, :) = {'response values used at test (1 = sure old to 6 = sure new)', ...
    strjoin(string(responseValues), ' '), 'yes', 'inversion_trials.csv'};

%% 5. Old and new: each critical item was studied by some participants and a lure for others
criticalTests = testTrials(testTrials.condition == "critical", :);
timesOld = zeros(height(criticalItems), 1);
timesNew = zeros(height(criticalItems), 1);
for itemIndex = 1:height(criticalItems)
    itemTests = criticalTests(criticalTests.image_name == criticalItems.image_name(itemIndex), :);
    timesOld(itemIndex) = sum(itemTests.is_new == 0);
    timesNew(itemIndex) = sum(itemTests.is_new == 1);
end
everyItemBothWays = all(timesOld > 0) && all(timesNew > 0);
checks(end + 1, :) = {'sessions in which a critical item was studied (old), smallest to largest over the 40 items', ...
    sprintf('%d to %d', min(timesOld), max(timesOld)), yesNo(everyItemBothWays), 'inversion_trials.csv'};
checks(end + 1, :) = {'sessions in which a critical item was a lure (new), smallest to largest over the 40 items', ...
    sprintf('%d to %d', min(timesNew), max(timesNew)), yesNo(everyItemBothWays), 'inversion_trials.csv'};

%% Write the checks
checkTable = cell2table(checks, 'VariableNames', {'fact', 'value', 'every_block_and_session_agrees', 'source'});
checkTable.value = string(cellfun(@valueText, checkTable.value, 'UniformOutput', false));
writetable(checkTable, fullfile(paths.outputs, 'inversion_design_check.csv'));
for rowIndex = 1:height(checkTable)
    fprintf('%-95s %-28s %s\n', checkTable.fact{rowIndex}, checkTable.value(rowIndex), checkTable.every_block_and_session_agrees{rowIndex});
end

%% Return the counts
design = struct();
design.numberOfBlocks = numberOfBlocks;
design.blocksPerSession = singleValue(blocksPerSession);
design.studyItemsPerBlock = singleValue([studyRowsPerBlock; studyPerBlock]);
design.testTrialsPerBlock = singleValue([testRowsPerBlock; testPerBlock]);
design.controlledItemsPerList = singleValue([controlledPerBlock; controlledStudiedPerBlock]);
design.similarItemsPerList = singleValue(similarPerBlock);
design.fillerItemsPerList = singleValue(fillersPerBlock);
design.furtherStudiedItemsPerList = singleValue([furtherStudiedPerBlock; furtherPerBlock]);
design.criticalItemsPerBlock = singleValue([criticalTestedPerBlock; criticalTestPerBlock]);
design.dissimilarItemsPerTestBlock = singleValue([dissimilarTestedPerBlock; dissimilarTestPerBlock]);
design.similarItemsByQuintile = similarByQuintile;       % quintile 1 (lowest score) to 5 (highest)
design.itemsPerQuintile = itemsPerQuintile;
design.numberOfCriticalItems = height(criticalItems);
design.numberOfQuintiles = numberOfBins;
design.binWidthPercent = binWidthPercent;
design.quintileCuts = cuts(2:end - 1);
design.numberOfScoredImages = numberOfScoredImages;
design.quintilesFollowScores = quintileMatches && onePerQuintilePerBlock;
design.responseValues = responseValues;
design.numberOfResponseOptions = numel(responseValues);
design.numberOfSessions = numberOfSessions;
design.criticalStudiedPerBlock = unique(criticalStudiedPerBlock)';
design.timesOldRange = [min(timesOld), max(timesOld)];
design.timesNewRange = [min(timesNew), max(timesNew)];
design.everyItemOldAndNew = everyItemBothWays;
design.criticalScoreRange = [min(criticalItems.kramer_cr), max(criticalItems.kramer_cr)];
end


function table = readCsvWithText(fileName, textColumns)
% read a CSV file of the repository; the named columns as text, the others as numbers
options = detectImportOptions(fileName, 'Delimiter', ',');
options = setvartype(options, textColumns, 'string');
table = readtable(fileName, options);
end


function row = agreeRow(fact, values, source)
% one check row: the value when every block (or session) has the same one, else the range
if all(values == values(1))
    row = {fact, values(1), 'yes', source};
else
    row = {fact, sprintf('%g to %g', min(values), max(values)), 'no', source};
end
end


function value = singleValue(values)
% the one value every block and session shares (NaN when they differ)
if all(values == values(1))
    value = values(1);
else
    value = NaN;
end
end


function text = valueText(value)
if isnumeric(value)
    text = sprintf('%g', value);
else
    text = char(string(value));
end
end


function text = yesNo(flag)
if flag
    text = 'yes';
else
    text = 'no';
end
end
