function T = load_image_set_trials(setNames, concInfo, withSummedSimilarity)
%LOAD_IMAGE_SET_TRIALS  The image-set experiments (Figs. 6 and 7) as the analyses use them.
%
%   T = load_image_set_trials(setNames, concInfo)
%   T = load_image_set_trials(setNames, concInfo, false)   (skip the summed-similarity columns)
%
%   setNames  cell array with any of 'maintained', 'changed', 'reduced', in the order wanted
%             (module 06: {'maintained', 'changed'} in maintained_vs_changed.m,
%              {'maintained', 'reduced'} in maintained_vs_reduced_variance.m and all three in
%              participant_flow.m; module A1: all three)
%   concInfo  from load_concept_info('human') (needs concepts and thingsSimRSM)
%
%   Reads 06_image_set_experiments_fig6_fig7/data/<set>_trials.csv (the de-identified sessions:
%   maintained M001..M205, changed C001..C195, reduced R001..R202, numbered within each set in the
%   alphabetical order of the session ids) and builds the test-trial table:
%       subject              participant id (string)
%       is_new               1 = new item at test, 0 = studied
%       image_number         the experiment's own image index (carried, not analyzed)
%       original_index       the THINGS image number (Kramer et al. image_nr)
%       rt                   response time in ms
%       response             1 = sure old .. 6 = sure new
%       sumSimToStudyItems1  mean of normpdf(similarity(c, j), 1, 2) over the study items j of the
%                            same block, with one occurrence of the tested concept c removed
%       sumSimToStudyItems2  the same for the previous block (NaN in block 1)
%       filename, concept    (cell arrays of char)
%       dataset_index        maintained 1, changed 0, reduced 2
%       dataset_name         "representativeImages", "flipMemorabilityImages", "equatedImages"
%   Rows: sets in the order given, participants in id order, test trials in the order they were run.
%   The similarity is concInfo.thingsSimRSM; the kernel SD is 2 (stdForWeightedSim).

if nargin < 3
    withSummedSimilarity = true;
end
if ischar(setNames) || isstring(setNames)
    setNames = cellstr(setNames);
end

sharedMatlabFolder = fileparts(mfilename('fullpath'));
releaseRoot = fileparts(fileparts(sharedMatlabFolder));
dataFolder = fullfile(releaseRoot, '06_image_set_experiments_fig6_fig7', 'data');

datasetIndexOf = containers.Map({'maintained', 'changed', 'reduced'}, {1, 0, 2});
datasetNameOf = containers.Map({'maintained', 'changed', 'reduced'}, ...
    {'representativeImages', 'flipMemorabilityImages', 'equatedImages'});
stdForWeightedSim = 2;
numberOfBlocks = 16;

setTables = cell(numel(setNames), 1);
for setIndex = 1:numel(setNames)
    setName = setNames{setIndex};
    trialFile = fullfile(dataFolder, [setName '_trials.csv']);
    options = detectImportOptions(trialFile, 'Delimiter', ',');
    options = setvartype(options, {'participant', 'phase', 'image_filename', 'concept'}, 'string');
    options = setvartype(options, {'block', 'trial_in_block', 'things_image_index', 'image_number', ...
        'is_new', 'response_1to6', 'rt_ms'}, 'double');
    rows = readtable(trialFile, options);
    fprintf('Loading %d subjects from %s (%s)...\n', numel(unique(rows.participant)), ...
        datasetNameOf(setName), setName);

    isTest = rows.phase == "test";
    testRows = rows(isTest, :);
    numberOfTests = height(testRows);
    sumSimSameBlock = nan(numberOfTests, 1);
    sumSimPreviousBlock = nan(numberOfTests, 1);

    if withSummedSimilarity
        % concept number (THINGS order) of every row
        [isKnown, conceptNumberOfRow] = ismember(rows.concept, string(concInfo.concepts));
        if ~all(isKnown)
            error('load_image_set_trials:Concept', 'a concept is missing from the concept table');
        end
        [participants, ~, participantOfRow] = unique(rows.participant);   % participant number of each row
        isStudy = rows.phase == "study";
        testNumberOfRow = zeros(height(rows), 1);                        % position of a test row in testRows
        testNumberOfRow(isTest) = 1:numberOfTests;
        for participantIndex = 1:numel(participants)
            ownRows = find(participantOfRow == participantIndex);
            % the 40 study items of each block (study order)
            studyConceptNumbers = nan(numberOfBlocks, 40);
            for block = 1:numberOfBlocks
                studyOfBlock = ownRows(isStudy(ownRows) & rows.block(ownRows) == block);
                studyConceptNumbers(block, 1:numel(studyOfBlock)) = conceptNumberOfRow(studyOfBlock)';
            end
            % every test trial of this participant
            ownTests = ownRows(isTest(ownRows));
            for k = 1:numel(ownTests)
                rowIndex = ownTests(k);
                testIndex = testNumberOfRow(rowIndex);
                testedConcept = conceptNumberOfRow(rowIndex);
                testBlock = rows.block(rowIndex);
                sumSimToStudyItems = nan(1, 2);
                for blockBack = 0:1
                    if testBlock - blockBack > 0
                        studyConcepts = studyConceptNumbers(testBlock - blockBack, :);
                        studyConcepts(find(studyConcepts == testedConcept, 1)) = [];
                        sumSimToStudyItems(blockBack + 1) = mean(normpdf(concInfo.thingsSimRSM(testedConcept, ...
                            studyConcepts), 1, stdForWeightedSim));
                    end
                end
                sumSimSameBlock(testIndex) = sumSimToStudyItems(1);
                sumSimPreviousBlock(testIndex) = sumSimToStudyItems(2);
            end
        end
    end

    setTable = table();
    setTable.subject = testRows.participant;
    setTable.is_new = testRows.is_new;
    setTable.image_number = testRows.image_number;
    setTable.original_index = testRows.things_image_index;
    setTable.rt = testRows.rt_ms;
    setTable.response = testRows.response_1to6;
    if withSummedSimilarity
        setTable.sumSimToStudyItems1 = sumSimSameBlock;
        setTable.sumSimToStudyItems2 = sumSimPreviousBlock;
    end
    setTable.filename = cellstr(testRows.image_filename);
    setTable.concept = cellstr(testRows.concept);
    setTable.dataset_index = repmat(datasetIndexOf(setName), numberOfTests, 1);
    setTable.dataset_name = repmat(string(datasetNameOf(setName)), numberOfTests, 1);
    setTables{setIndex} = setTable;
end

T = vertcat(setTables{:});
fprintf('Building trial table (%d rows)...\n', height(T));
end
