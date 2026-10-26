function strip_concept_info(humanSourceFile, monkeySourceFile, outputFolder)
%STRIP_CONCEPT_INFO  Make the two concept files of shared/things/.
%
% WHAT IT DOES
%   ConceptInfo.mat (the human SPoSE space, Hebart et al. 2020) and
%   ConceptMonkeyInfo.mat (the monkey IT space, Papale et al. 2025) each hold
%   one struct, concInfo, with one row per THINGS concept (1,854 concepts,
%   THINGS order). This function removes the fields that no reported number
%   needs, or that may not be shared, and saves what is left in MATLAB v7
%   format (which Python's scipy.io.loadmat can also read):
%       images             concept thumbnails (THINGS pixels; not ours to share)
%       thingsSimRSM       the 1,854 x 1,854 concept similarity matrix,
%                          corrcoef(features') (CombineConceptLevelStuff.m line 11,
%                          CombineConceptMonkey.m line 20), which the shared
%                          loader load_concept_info.m recomputes
%       memorabilityItem   unused by any reported analysis
%       memorabilityWilma  a copy of the CR column of the public
%                          Concept_to_category_linking.csv (unused by any
%                          reported analysis)
%       resMemPredictions  ResMem predictions (unused by any reported analysis)
%   Kept unchanged: features, featureLabels, concepts, conceptNumber,
%   categoryNumber, category, memorabilityCat, weightedSimByConc, conceptCounts.
%
%   Before dropping thingsSimRSM it checks that corrcoef(features') equals the
%   stored matrix (it prints the largest absolute difference and whether the two
%   are exactly equal). After saving it reloads each file and checks that every kept
%   field is identical to the input.
%
% INPUTS
%   humanSourceFile   path to ConceptInfo.mat
%   monkeySourceFile  path to ConceptMonkeyInfo.mat
%   outputFolder      the folder shared/things/
%
% OUTPUTS (in outputFolder)
%   concept_info_human.mat, concept_info_monkey.mat (struct concInfo, -v7)
%
% It needs the two input files, which are not included. Run it from the
% repository root as
%   matlab -batch "addpath('shared/things/provenance'); strip_concept_info('<ConceptInfo.mat>', '<ConceptMonkeyInfo.mat>', 'shared/things')"

fieldsToRemove = {'images', 'thingsSimRSM', 'memorabilityItem', 'memorabilityWilma', 'resMemPredictions'};
sourceFiles = {humanSourceFile, monkeySourceFile};
outputNames = {'concept_info_human.mat', 'concept_info_monkey.mat'};
spaceNames = {'human (SPoSE)', 'monkey (IT)'};

for fileIndex = 1:2
    fprintf('\n=== %s space: %s ===\n', spaceNames{fileIndex}, sourceFiles{fileIndex});
    loaded = load(sourceFiles{fileIndex}, 'concInfo');
    original = loaded.concInfo;

    % The stored similarity matrix must be exactly the correlation of the feature rows.
    recomputedSimilarity = corrcoef(original.features');
    storedSimilarity = original.thingsSimRSM;
    maximumAbsoluteDifference = max(abs(recomputedSimilarity(:) - storedSimilarity(:)));
    bitIdentical = isequal(recomputedSimilarity, storedSimilarity);
    fprintf('corrcoef(features'') vs stored thingsSimRSM: max abs difference = %.3g, exactly equal = %d\n', ...
        maximumAbsoluteDifference, bitIdentical);
    if maximumAbsoluteDifference > 1e-12
        error('strip_concept_info:RSM', ...
            'The recomputed similarity matrix differs from the stored one by %.3g (> 1e-12); keep the stored matrix.', ...
            maximumAbsoluteDifference);
    end

    % Remove the unneeded fields and save the rest in v7 format.
    concInfo = rmfield(original, fieldsToRemove); %#ok<NASGU>
    outputFile = fullfile(outputFolder, outputNames{fileIndex});
    save(outputFile, 'concInfo', '-v7');
    fileInfo = dir(outputFile);
    fprintf('saved %s (%d bytes)\n', outputNames{fileIndex}, fileInfo.bytes);

    % Reload and compare every kept field with the input.
    reloaded = load(outputFile, 'concInfo');
    keptFields = fieldnames(reloaded.concInfo);
    for fieldIndex = 1:numel(keptFields)
        fieldName = keptFields{fieldIndex};
        if ~isequaln(reloaded.concInfo.(fieldName), original.(fieldName))   % isequaln: NaN counts as equal to NaN
            error('strip_concept_info:Reload', 'Field %s changed on save.', fieldName);
        end
    end
    fprintf('kept fields (%d), all identical to the input after reloading: %s\n', ...
        numel(keptFields), strjoin(keptFields', ', '));
end
end
