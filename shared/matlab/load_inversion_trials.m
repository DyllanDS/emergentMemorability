function [T, design, studyTrials] = load_inversion_trials()
%LOAD_INVERSION_TRIALS  The memorability-inversion experiment (Fig. 3) as the analysis uses it.
%
%   [T, design, studyTrials] = load_inversion_trials()
%
%   Reads 03_inversion_experiment_fig3/data/inversion_trials.csv (54 de-identified sessions,
%   S001..S054) and inversion_design.csv (the 40 critical items), and builds the test-trial table:
%       subjectNumber   1..54, in S001..S054 order (the alphabetical order of the session ids)
%       block, trialNumber (1..80 within the participant), imageName, concept, isCritical, isNew,
%       response (1 = sure old .. 6 = sure new), rt (ms), memorabilityScore (the public Kramer et
%       al. score of the image, from shared/things/things_images.csv)
%       participant     the S-id
%   All 54 sessions are used.
%
%   design:      the 40 critical items (block, image_name, concept, memorability_quintile,
%                n_similar_items, kramer_cr)
%   studyTrials: the study rows (participant, block, trial_in_block, image_name, concept, condition)

sharedMatlabFolder = fileparts(mfilename('fullpath'));
releaseRoot = fileparts(fileparts(sharedMatlabFolder));
dataFolder = fullfile(releaseRoot, '03_inversion_experiment_fig3', 'data');

% ---- read the trial file with explicit column types ----
trialFile = fullfile(dataFolder, 'inversion_trials.csv');
options = detectImportOptions(trialFile, 'Delimiter', ',');
options = setvartype(options, {'participant', 'phase', 'image_name', 'concept', 'condition'}, 'string');
options = setvartype(options, {'block', 'trial_in_block', 'is_new', 'response_1to6', 'rt_ms'}, 'double');
allRows = readtable(trialFile, options);

% ---- the public memorability score of every image ----
imageFile = fullfile(releaseRoot, 'shared', 'things', 'things_images.csv');
imageOptions = detectImportOptions(imageFile, 'Delimiter', ',');
imageOptions = setvartype(imageOptions, {'filename', 'rel_path', 'concept'}, 'string');
imageOptions = setvartype(imageOptions, 'kramer_cr', 'double');
images = readtable(imageFile, imageOptions);

% ---- participants in S001..S054 order ----
participants = unique(allRows.participant(allRows.phase == "test"));     % sorted S001, S002, ...
numberOfParticipants = numel(participants);

testRows = allRows(allRows.phase == "test", :);
numberOfTrials = height(testRows);
subjectNumber = zeros(numberOfTrials, 1);
trialNumber = zeros(numberOfTrials, 1);
for participantIndex = 1:numberOfParticipants
    isThisParticipant = testRows.participant == participants(participantIndex);
    subjectNumber(isThisParticipant) = participantIndex;
    trialNumber(isThisParticipant) = (1:sum(isThisParticipant))';     % order of the test trials
end

[found, imageRow] = ismember(testRows.image_name, images.filename);
if ~all(found)
    error('load_inversion_trials:Image', 'a tested image is missing from things_images.csv');
end

T = table();
T.subjectNumber = subjectNumber;
T.block = testRows.block;
T.trialNumber = trialNumber;
T.imageName = testRows.image_name;
T.concept = testRows.concept;
T.isCritical = testRows.condition == "critical";
T.isNew = testRows.is_new == 1;
T.response = testRows.response_1to6;
T.rt = testRows.rt_ms;
T.memorabilityScore = images.kramer_cr(imageRow);
T.participant = testRows.participant;

% ---- the design table and the study rows ----
designFile = fullfile(dataFolder, 'inversion_design.csv');
designOptions = detectImportOptions(designFile, 'Delimiter', ',');
designOptions = setvartype(designOptions, {'image_name', 'concept'}, 'string');
design = readtable(designFile, designOptions);

studyTrials = allRows(allRows.phase == "study", {'participant', 'block', 'trial_in_block', 'image_name', ...
    'concept', 'condition'});
end
