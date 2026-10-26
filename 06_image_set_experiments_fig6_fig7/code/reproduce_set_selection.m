function selection = reproduce_set_selection()
%REPRODUCE_SET_SELECTION  Redraw the concepts of the three image sets (Methods, Image Set
%   Structure Manipulations).
%
%   What it does
%     Draws the concepts of the three 1,600-slot image sets with rng(6), in the order below, and
%     checks that they equal the concepts of the sets in data/sampled_image_sets.mat. The
%     assignment of THINGS images to the drawn concepts is not repeated here;
%     data/sampled_image_sets.mat holds the images, and
%     shared/things/provenance/SampleDatasetsThatFlip.m (which needs files that are not
%     included) shows that step.
%
%     Changed set: concept score = - sum over features positively correlated with
%       memorability of (feature value x squared correlation) + the same sum over the
%       negatively correlated features. From the 500 highest-scoring concepts (high on
%       features that go with poor memorability) 160 are chosen greedily: start with the top
%       concept, then repeatedly add the candidate with the lowest mean similarity
%       (normpdf(similarity, 1, 2)) to the concepts already chosen. Then 1,440 more concepts
%       are drawn with replacement, with probability proportional to the squared positive
%       part of minus the score.
%     Maintained set: representativeness score = sum over features of feature value x
%       sign(r) x r^2, where r is the Spearman correlation of the feature with the full-THINGS
%       summed similarity; scores shifted to start at 0 and normalized; 1,600 concepts drawn
%       with replacement.
%     Reduced set: minus the changed-set score, rescaled to 0 to 1, raised to an exponent
%       (0.01 to 0.5 in steps of 0.01, each repeated 10 times: 500 candidates) and
%       normalized; 1,600 concepts drawn with replacement per candidate. For each candidate the
%       summed similarity of every concept to the candidate set is computed and correlated
%       (Spearman, over the 1,600 slots) with each of the 49 features. The composite score of a
%       candidate is the mean of its 49 correlations plus their maximum; the candidate with the
%       lowest composite score is kept.
%
%   Inputs
%     shared/things/concept_info_human.mat (via load_concept_info.m): SPoSE features, the
%       Kramer et al. concept memorability, the full-THINGS summed similarity, the similarity
%       matrix (recomputed as corrcoef of the feature rows)
%     data/sampled_image_sets.mat: the three image sets
%
%   Output
%     outputs/set_selection_check.csv: the design constants used and whether each drawn set
%     equals the set in data/sampled_image_sets.mat
%   Returns a struct with the same values (used by write_module_numbers.m).
%
%   Seed: rng(6), set once. The random draws come in this order: the 1,440 weighted draws of the
%   changed set (the 160 greedy choices use no random numbers), the 1,600 draws of the maintained
%   set, then the 500 reduced-set candidates of 1,600 draws each. Run the function whole, in
%   this order. Runtime about 30 seconds.

paths = release_paths(mfilename('fullpath'));
concInfo = load_concept_info('human');
shipped = load(fullfile(paths.data, 'sampled_image_sets.mat'));

% Design constants (Methods, Image Set Structure Manipulations)
numberOfCandidateConcepts = 500;      % top-scoring concepts considered for the sparse part of the changed set
numberOfDissimilarConcepts = 160;     % chosen greedily from those 500
numToSample = 1440;                   % further concepts of the changed set
numToSampleRep = 1600;                % concepts of the maintained set
numToSampleEquated = 1600;            % concepts of each reduced-set candidate
expToTry = repmat(0.01:0.01:0.5, [1, 10]);   % 50 exponents, each tried 10 times
stdForWeightedSim = 2;                % SD of the similarity kernel normpdf(similarity, 1, 2)

rng(6, 'twister');   % the seed of the set selection

nConcepts = length(concInfo.concepts);
nFeatures = size(concInfo.features, 2);

% Similarity kernel between every pair of concepts
for i = 1:nConcepts
    concInfo.simMatrix(i, :) = normpdf(concInfo.thingsSimRSM(i, :), 1, stdForWeightedSim);
end

% Which features predict memorability in the full THINGS data (Spearman, across concepts)
featureCorrs = zeros(nFeatures, 1);
for i = 1:nFeatures
    featureCorrs(i) = corr(concInfo.features(:, i), concInfo.memorabilityCat, 'Type', 'Spearman', 'rows', 'complete');
end
% Which features go with summed similarity in the full THINGS data (Spearman, across concepts)
featureCorrsSim = corr(concInfo.weightedSimByConc, concInfo.features, 'type', 'Spearman');

%% Changed set, part 1: 160 dissimilar concepts that are high on "poor memory" features
posFeatures = find(featureCorrs > 0);
negFeatures = find(featureCorrs < 0);
conceptScores = zeros(nConcepts, 1);
for i = 1:nConcepts
    % subtract the weighted values of the features that go with good memory
    conceptScores(i) = -sum(concInfo.features(i, posFeatures) .* (featureCorrs(posFeatures) .^ 2)');
    % add the weighted values of the features that go with poor memory
    conceptScores(i) = conceptScores(i) + sum(concInfo.features(i, negFeatures) .* (featureCorrs(negFeatures) .^ 2)');
end
[~, sortedConceptIdx] = sort(conceptScores, 'descend');
candidateConceptIdx = sortedConceptIdx(1:numberOfCandidateConcepts);

selectedConceptIndices = candidateConceptIdx(1);
remainingCandidates = candidateConceptIdx(2:end);
while length(selectedConceptIndices) < numberOfDissimilarConcepts
    % mean similarity of each remaining candidate to the concepts chosen so far
    avgSimilarity = zeros(length(remainingCandidates), 1);
    for i = 1:length(remainingCandidates)
        candidateIdx = remainingCandidates(i);
        similarities = zeros(length(selectedConceptIndices), 1);
        for j = 1:length(selectedConceptIndices)
            similarities(j) = concInfo.simMatrix(candidateIdx, selectedConceptIndices(j));
        end
        avgSimilarity(i) = mean(similarities);
    end
    % add the least similar candidate
    [~, minSimIdx] = min(avgSimilarity);
    selectedConceptIndices(end + 1, 1) = remainingCandidates(minSimIdx); %#ok<AGROW>
    remainingCandidates(minSimIdx) = [];
end

%% Changed set, part 2: 1,440 concepts drawn in proportion to their squared "good memory" score
conceptScoresPositive = -conceptScores;
conceptScoresPositive(conceptScoresPositive < 0) = 0;
conceptScoresPositive = conceptScoresPositive .^ 2;
conceptScoresPositive = conceptScoresPositive ./ sum(conceptScoresPositive);
sampledIndices = randsample(length(conceptScoresPositive), numToSample, true, conceptScoresPositive);
changedSetConcepts = sort([selectedConceptIndices; sampledIndices]);

%% Maintained set: 1,600 concepts drawn in proportion to their representativeness score
representativeScores = zeros(nConcepts, 1);
for i = 1:nConcepts
    featureSimilarityMatch = 0;
    for j = 1:nFeatures
        featureSimilarityMatch = featureSimilarityMatch + ...
            (concInfo.features(i, j) * sign(featureCorrsSim(j)) * abs(featureCorrsSim(j)) ^ 2);
    end
    representativeScores(i) = featureSimilarityMatch;
end
representativeScores = representativeScores - min(representativeScores);
representativeScores = representativeScores ./ sum(representativeScores);
representativeDatasetIndices = randsample(nConcepts, numToSampleRep, true, representativeScores);
maintainedSetConcepts = sort(representativeDatasetIndices);

%% Reduced set: 500 candidate sets, keep the one whose summed similarity is least tied to the features
normalizedConceptScores = (-conceptScores - min(-conceptScores)) / (max(-conceptScores) - min(-conceptScores));
numberOfCandidates = length(expToTry);
equatedDatasetIndices = zeros(numberOfCandidates, numToSampleEquated);
scoreValue = zeros(numberOfCandidates, 1);   % composite score: mean + max of the 49 correlations
for n = 1:numberOfCandidates
    equatedScores = normalizedConceptScores .^ (expToTry(n));
    equatedScores = equatedScores ./ sum(equatedScores);
    equatedDatasetIndices(n, :) = randsample(nConcepts, numToSampleEquated, true, equatedScores);

    % summed similarity of every concept to this candidate set (weights = slot counts / 1,600,
    % the concept itself left out)
    equatedWeights = zeros(nConcepts, 1);
    for i = 1:nConcepts
        equatedWeights(i) = sum(equatedDatasetIndices(n, :) == i);
    end
    equatedWeights = equatedWeights ./ sum(equatedWeights);
    candidateWeightedSim = zeros(1, nConcepts);
    for i = 1:nConcepts
        otherConcepts = true(nConcepts, 1);
        otherConcepts(i) = false;
        candidateWeightedSim(i) = normpdf(concInfo.thingsSimRSM(i, otherConcepts), 1, stdForWeightedSim) * equatedWeights(otherConcepts);
    end

    % Spearman correlation of each feature with summed similarity, over the 1,600 slots
    candidateCorrs = zeros(1, nFeatures);
    for i = 1:nFeatures
        candidateCorrs(i) = corr(candidateWeightedSim(equatedDatasetIndices(n, :))', ...
            concInfo.features(equatedDatasetIndices(n, :), i), 'type', 'Spearman');
    end
    scoreValue(n) = mean(candidateCorrs) + max(candidateCorrs);
end
[~, minCorrIdx] = min(scoreValue);
reducedSetConcepts = sort(equatedDatasetIndices(minCorrIdx, :))';

%% Compare with the sets in data/sampled_image_sets.mat
changedEqual = isequal(changedSetConcepts(:), shipped.flipMemorabilityDataset.conceptIndex(:));
maintainedEqual = isequal(maintainedSetConcepts(:), shipped.representativeDataset.conceptIndex(:));
reducedEqual = isequal(reducedSetConcepts(:), shipped.equatedDataset.conceptIndex(:));

exponentSteps = unique(round(diff(unique(expToTry)), 10));
selection = struct();
selection.numberOfCandidateConcepts = numberOfCandidateConcepts;
selection.numberOfDissimilarConcepts = numel(selectedConceptIndices);
selection.numberOfFurtherChangedConcepts = numel(sampledIndices);
selection.numberOfMaintainedDraws = numel(representativeDatasetIndices);
selection.numberOfReducedDraws = size(equatedDatasetIndices, 2);
selection.numberOfExponents = numel(expToTry);
selection.smallestExponent = min(expToTry);
selection.largestExponent = max(expToTry);
selection.exponentStep = exponentSteps;
selection.repetitionsPerExponent = numel(expToTry) / numel(unique(expToTry));
selection.normalizedScoreRange = [min(normalizedConceptScores), max(normalizedConceptScores)];
selection.numberOfCorrelationsPerCandidate = nFeatures;
selection.chosenCandidate = minCorrIdx;
selection.chosenExponent = expToTry(minCorrIdx);
selection.changedSetEqual = changedEqual;
selection.maintainedSetEqual = maintainedEqual;
selection.reducedSetEqual = reducedEqual;

fprintf('changed set: %d greedy + %d drawn concepts; equal to the set in data/sampled_image_sets.mat: %d\n', ...
    selection.numberOfDissimilarConcepts, selection.numberOfFurtherChangedConcepts, changedEqual);
fprintf('maintained set: %d draws; equal to the set in data/sampled_image_sets.mat: %d\n', selection.numberOfMaintainedDraws, maintainedEqual);
fprintf('reduced set: %d candidates (exponents %.2f to %.2f in steps of %.2f, %d repetitions); chosen candidate %d (exponent %.2f); equal to the set in data/sampled_image_sets.mat: %d\n', ...
    selection.numberOfExponents, selection.smallestExponent, selection.largestExponent, selection.exponentStep, ...
    selection.repetitionsPerExponent, minCorrIdx, selection.chosenExponent, reducedEqual);
if ~(changedEqual && maintainedEqual && reducedEqual)
    warning('reproduce_set_selection:Differs', 'a drawn set differs from the set in data/sampled_image_sets.mat (see outputs/set_selection_check.csv)');
end

quantity = {'top-scoring concepts considered for the sparse part of the changed set'; ...
    'concepts chosen greedily (lowest mean similarity) for the changed set'; ...
    'further concepts drawn for the changed set'; ...
    'concepts drawn for the maintained set'; ...
    'concepts drawn for each reduced-set candidate'; ...
    'reduced-set candidates (exponents tried)'; ...
    'smallest exponent'; 'largest exponent'; 'exponent step'; 'repetitions of each exponent'; ...
    'normalized concept score, lowest'; 'normalized concept score, highest'; ...
    'Spearman correlations in each candidate''s composite score'; ...
    'chosen candidate (lowest composite score)'; 'exponent of the chosen candidate'; ...
    'drawn changed-set concepts equal the set in data/sampled_image_sets.mat (1 = yes)'; ...
    'drawn maintained-set concepts equal the set in data/sampled_image_sets.mat (1 = yes)'; ...
    'drawn reduced-set concepts equal the set in data/sampled_image_sets.mat (1 = yes)'};
value = [selection.numberOfCandidateConcepts; selection.numberOfDissimilarConcepts; ...
    selection.numberOfFurtherChangedConcepts; selection.numberOfMaintainedDraws; selection.numberOfReducedDraws; ...
    selection.numberOfExponents; selection.smallestExponent; selection.largestExponent; selection.exponentStep; ...
    selection.repetitionsPerExponent; selection.normalizedScoreRange(1); selection.normalizedScoreRange(2); ...
    selection.numberOfCorrelationsPerCandidate; minCorrIdx; selection.chosenExponent; ...
    changedEqual; maintainedEqual; reducedEqual];
checkTable = table(quantity, value);
writetable(checkTable, fullfile(paths.outputs, 'set_selection_check.csv'));
end
