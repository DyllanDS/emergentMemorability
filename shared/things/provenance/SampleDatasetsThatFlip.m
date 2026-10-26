 clear all;
load('<ImageInfo.mat>'); % imgInfo
load('ConceptInfo.mat'); % concInfo

% Set random number generator seed for reproducibility
rng(6, 'twister'); % Using seed to ensure consistent results across runs

nConcepts = length(concInfo.concepts);


%% Setup and Initialization
% Normalize concept similarity matrix
for i = 1:length(concInfo.concepts)
  concInfo.simMatrix(i,:) = normpdf(concInfo.thingsSimRSM(i,:), 1, 2);
end

%% Feature Analysis for Memorability
% Analyze which feature dimensions predict memorability
nFeatures = size(concInfo.features, 2);
featureCorrs = zeros(nFeatures, 1);
for i = 1:nFeatures
  featureCorrs(i) = corr(concInfo.features(:,i), concInfo.memorabilityCat, 'Type', 'Spearman', 'rows', 'complete');
end
% Calculate correlations between weighted similarity and features
[featureCorrsSim, featurePvalsSim] = corr(concInfo.weightedSimByConc, concInfo.features, 'type', 'Spearman');

%% Concept Selection for Flip Memorability Dataset
% Find items high in the "bad" features (aka metal, etc) [the ones negatively correlated with memorability]
% and put in only a very sparse sampling of them into the dataset, 
% all maximally dissimilar from each other as much as possible  -- 160, 10% of them
% Then find items high in the "good" features (aka animate, etc) [the ones positively correlated with memorability]
% and put in many of them, including several from each concept for the most "good" concepts (per the positive features)
% and put in 1440 of them

% Identify positive and negative features
posFeatures = find(featureCorrs > 0);
negFeatures = find(featureCorrs < 0);
conceptScores = zeros(nConcepts, 1);
for i = 1:nConcepts
    % Subtract positive feature values (lower is better)
    conceptScores(i) = -sum(concInfo.features(i, posFeatures) .* (featureCorrs(posFeatures).^2)');
    % Add negative feature values (higher is better)
    conceptScores(i) = conceptScores(i) + sum(concInfo.features(i, negFeatures) .* (featureCorrs(negFeatures).^2)');
end

% Sort concepts by their scores
[~, sortedConceptIdx] = sort(conceptScores, 'descend');

% Take the top 500 concepts
candidateConceptIdx = sortedConceptIdx(1:500);

%% Select Dissimilar Concepts for Negative Features
% Initialize with the first concept
selectedConceptIndices = candidateConceptIdx(1);
remainingCandidates = candidateConceptIdx(2:end);

% Greedy algorithm to select maximally dissimilar concepts
while length(selectedConceptIndices) < 160
    % Calculate average similarity of each remaining candidate to already selected concepts
    avgSimilarity = zeros(length(remainingCandidates), 1);
    for i = 1:length(remainingCandidates)
        candidateIdx = remainingCandidates(i);
        similarities = zeros(length(selectedConceptIndices), 1);
        for j = 1:length(selectedConceptIndices)
            selectedIdx = selectedConceptIndices(j);
            similarities(j) = concInfo.simMatrix(candidateIdx, selectedIdx);
        end
        avgSimilarity(i) = mean(similarities);
    end
    
    % Find the candidate with minimum average similarity to already selected concepts
    [~, minSimIdx] = min(avgSimilarity);
    
    % Add this candidate to selected concepts
    selectedConceptIndices(end+1,1) = remainingCandidates(minSimIdx);
    
    % Remove this candidate from remaining candidates
    remainingCandidates(minSimIdx) = [];
end

% Print the names of the selected concepts
fprintf('Selected concepts for the dataset:\n');
selectedConceptNames = concInfo.concepts(selectedConceptIndices);
for i = 1:length(selectedConceptNames)
    fprintf('%d. %s\n', i, selectedConceptNames{i});
end

%% Select Concepts for Positive Features
% Now find concepts that are high on positive features and low on negative features
conceptScoresPositive = -conceptScores;
[~, sortedConceptIdxPositive] = sort(conceptScoresPositive, 'descend');
conceptScoresPositive(conceptScoresPositive<0) = 0;
conceptScoresPositive = conceptScoresPositive.^2;
conceptScoresPositive = conceptScoresPositive ./ sum(conceptScoresPositive);

%% Create Final Flip Memorability Dataset
numToSample = 1440;
sampledIndices = randsample(length(conceptScoresPositive), numToSample, true, conceptScoresPositive);
finalDatasetConceptIndices = [selectedConceptIndices; sampledIndices];
flipMemorabilityDatasetConceptIndices = sort(finalDatasetConceptIndices);

%% Calculate New Weighted Similarity for Flip Dataset
stdForWeightedSim=2;
weights = zeros(nConcepts,1);
for i = 1:nConcepts
  weights(i) = sum(finalDatasetConceptIndices == i);
end
weights = weights ./ sum(weights);
for i = 1:nConcepts
  otherConcepts = true(nConcepts,1);
  otherConcepts(i) = false;
  weightedSim = normpdf(concInfo.thingsSimRSM(i,otherConcepts),1,stdForWeightedSim) * weights(otherConcepts);
  concInfo.newDatasetWeightedSimByConc(i,1) = weightedSim;
end

% Calculate new similarity correlations
newFeatureCorrsSim = zeros(nFeatures, 1);
for i = 1:nFeatures
    newFeatureCorrsSim(i) = corr(concInfo.newDatasetWeightedSimByConc(finalDatasetConceptIndices), ...
                                concInfo.features(finalDatasetConceptIndices, i), ...
                                'type', 'Spearman');
end

%% Create Representative Dataset of 1600 concepts
%% ---------------------------------------
fprintf('\nCreating a representative dataset that preserves original similarity-feature relationships...\n');

% Calculate a score for each concept based on how well it represents the original dataset
representativeScores = zeros(nConcepts, 1);
for i = 1:nConcepts
    % Calculate how similar this concept's feature-similarity relationship is to the original
    featureSimilarityMatch = 0;
    for j = 1:nFeatures
        % Higher score for concepts that maintain the original correlation pattern
        featureSimilarityMatch = featureSimilarityMatch + ...
            (concInfo.features(i, j) * sign(featureCorrsSim(j)) * abs(featureCorrsSim(j))^2);
    end
    representativeScores(i) = featureSimilarityMatch;
end

% Normalize scores to create a probability distribution
representativeScores = representativeScores - min(representativeScores);
representativeScores = representativeScores ./ sum(representativeScores);

% Sample 1600 concepts (same size as the first dataset) based on these scores
numToSampleRep = 1600;
representativeDatasetIndices = randsample(nConcepts, numToSampleRep, true, representativeScores);
representativeDatasetConceptIndices = sort(representativeDatasetIndices);

%% Create Equated Dataset of 1600 concepts
%% ---------------------------------------
fprintf('\nCreating an equated dataset that aims for zero correlation between features and similarity...\n');
expToTry = repmat(0.01:0.01:0.5, [1, 10]);
numToSampleEquated = 1600;
equatedDatasetIndices = zeros(length(expToTry), numToSampleEquated);
equatedDatasetConceptIndices = zeros(length(expToTry), numToSampleEquated);
equatedDatasetWeightedSimByConc = zeros(length(expToTry), nConcepts);
equatedFeatureCorrsSimL = zeros(length(expToTry), nFeatures);
scoreValue = zeros(length(expToTry), 1);

conceptScoresPositive = -conceptScores;
normalizedConceptScores = (conceptScoresPositive - min(conceptScoresPositive)) / (max(conceptScoresPositive) - min(conceptScoresPositive));
for n=1:length(expToTry)
  % Calculate scores that will lead to a balanced representation
  equatedScores = normalizedConceptScores.^(expToTry(n));
  equatedScores = equatedScores ./ sum(equatedScores);

  % Sample 1600 concepts based on these scores
  equatedDatasetIndices(n,:) = randsample(nConcepts, numToSampleEquated, true, equatedScores);
  equatedDatasetConceptIndices(n,:) = sort(equatedDatasetIndices(n,:));

  % Calculate weighted similarity for equated dataset
  equatedWeights = zeros(nConcepts,1);
  for i = 1:nConcepts
    equatedWeights(i) = sum(equatedDatasetIndices(n,:) == i);
  end
  equatedWeights = equatedWeights ./ sum(equatedWeights);
  equatedDatasetWeightedSimByConc(n,:) = zeros(nConcepts, 1);
  equatedFeatureCorrsSimL(n,:) = zeros(nFeatures, 1);
  for i = 1:nConcepts
    otherConcepts = true(nConcepts,1);
    otherConcepts(i) = false;
    weightedSim = normpdf(concInfo.thingsSimRSM(i,otherConcepts),1,stdForWeightedSim) * equatedWeights(otherConcepts);
    equatedDatasetWeightedSimByConc(n,i) = weightedSim;
  end
  for i = 1:nFeatures
    equatedFeatureCorrsSimL(n,i) = corr(equatedDatasetWeightedSimByConc(n,equatedDatasetIndices(n,:))', ...
                                concInfo.features(equatedDatasetIndices(n,:), i), ...
                                'type', 'Spearman');
  end
  % Add a penalty for the maximum correlation and the average correlation
  scoreValue(n) = mean((equatedFeatureCorrsSimL(n,:))) + max(equatedFeatureCorrsSimL(n,:));
  fprintf('Mean correlation for equated dataset: %.4f\n', scoreValue(n));
end
figure(1); clf;
plot(expToTry, scoreValue);

% Find the index of the equated dataset with the lowest mean absolute correlation
[~, minCorrIdx] = min(scoreValue);
equatedFeatureCorrsSim = equatedFeatureCorrsSimL(minCorrIdx,:)';

%% Create Dataset Structs with Concept and Image Information
fprintf('\nCreating dataset structs with concept and image information...\n');

% Process the flip memorability dataset
flipMemorabilityDataset=struct();
flipMemorabilityDataset.conceptIndex = flipMemorabilityDatasetConceptIndices;
flipMemorabilityDataset.datasetName = 'FlipMemorability';
flipMemorabilityDataset.conceptName = concInfo.concepts(flipMemorabilityDatasetConceptIndices);
flipMemorabilityDataset.imageIndex = nan(length(flipMemorabilityDatasetConceptIndices),1);
flipMemorabilityDataset.imageFilename = cell(length(flipMemorabilityDatasetConceptIndices),1);
for i = 1:nConcepts
  conceptIdx = flipMemorabilityDatasetConceptIndices == i;
  conceptImgIndices = find(imgInfo.conceptNumber == i);
  numImagesForConcept = length(conceptImgIndices);
  imageWithinConceptIndex = mod(0:(sum(conceptIdx)-1), numImagesForConcept) + 1;
  flipMemorabilityDataset.numImagesForConcept(i,1) = sum(conceptIdx);
  flipMemorabilityDataset.imageIndex(conceptIdx) = conceptImgIndices(imageWithinConceptIndex);
  flipMemorabilityDataset.imageFilename(conceptIdx) = imgInfo.fileNames(flipMemorabilityDataset.imageIndex(conceptIdx));
end

% Process the representative dataset
representativeDataset=struct();
representativeDataset.conceptIndex = representativeDatasetConceptIndices;
representativeDataset.datasetName = 'Representative';
representativeDataset.conceptName = concInfo.concepts(representativeDatasetConceptIndices);
representativeDataset.imageIndex = nan(length(representativeDatasetConceptIndices),1);
representativeDataset.imageFilename = cell(length(representativeDatasetConceptIndices),1);
for i = 1:nConcepts
  conceptIdx = representativeDatasetConceptIndices == i;
  conceptImgIndices = find(imgInfo.conceptNumber == i);
  numImagesForConcept = length(conceptImgIndices);
  imageWithinConceptIndex = mod(0:(sum(conceptIdx)-1), numImagesForConcept) + 1;
  representativeDataset.numImagesForConcept(i,1) = sum(conceptIdx);
  representativeDataset.imageIndex(conceptIdx) = conceptImgIndices(imageWithinConceptIndex);
  representativeDataset.imageFilename(conceptIdx) = imgInfo.fileNames(representativeDataset.imageIndex(conceptIdx));
end 

% Process the equated dataset
equatedDataset=struct();
equatedDataset.conceptIndex = equatedDatasetConceptIndices(minCorrIdx,:);
equatedDataset.datasetName = 'Equated';
equatedDataset.conceptName = concInfo.concepts(equatedDatasetConceptIndices(minCorrIdx,:));
equatedDataset.imageIndex = nan(length(equatedDatasetConceptIndices(minCorrIdx,:)),1);
equatedDataset.imageFilename = cell(length(equatedDatasetConceptIndices(minCorrIdx,:)),1);
for i = 1:nConcepts
  conceptIdx = equatedDatasetConceptIndices(minCorrIdx,:) == i;
  conceptImgIndices = find(imgInfo.conceptNumber == i);
  numImagesForConcept = length(conceptImgIndices);
  imageWithinConceptIndex = mod(0:(sum(conceptIdx)-1), numImagesForConcept) + 1;
  equatedDataset.numImagesForConcept(i,1) = sum(conceptIdx);
  equatedDataset.imageIndex(conceptIdx) = conceptImgIndices(imageWithinConceptIndex);
  equatedDataset.imageFilename(conceptIdx) = imgInfo.fileNames(equatedDataset.imageIndex(conceptIdx));
end

save('SampledDatasets.mat', 'flipMemorabilityDataset', 'representativeDataset', 'equatedDataset');

%% Calculate Weighted Similarity for Representative Dataset
repWeights = representativeDataset.numImagesForConcept ./ sum(representativeDataset.numImagesForConcept);
concInfo.repDatasetWeightedSimByConc = zeros(nConcepts, 1);
for i = 1:nConcepts
  otherConcepts = true(nConcepts,1);
  otherConcepts(i) = false;
  weightedSim = normpdf(concInfo.thingsSimRSM(i,otherConcepts),1,stdForWeightedSim) * repWeights(otherConcepts);
  concInfo.repDatasetWeightedSimByConc(i,1) = weightedSim;
end
% Calculate similarity correlations for the representative dataset
repFeatureCorrsSim = zeros(nFeatures, 1);
for i = 1:nFeatures
    repFeatureCorrsSim(i) = corr(concInfo.repDatasetWeightedSimByConc(representativeDatasetIndices), ...
                                concInfo.features(representativeDatasetIndices, i), ...
                                'type', 'Spearman');
end

% Equated Dataset
equatedWeights = equatedDataset.numImagesForConcept ./ sum(equatedDataset.numImagesForConcept);
concInfo.equatedDatasetWeightedSimByConc = zeros(nConcepts, 1);
for i = 1:nConcepts
  otherConcepts = true(nConcepts,1);
  otherConcepts(i) = false;
  weightedSim = normpdf(concInfo.thingsSimRSM(i,otherConcepts),1,stdForWeightedSim) * equatedWeights(otherConcepts);
  concInfo.equatedDatasetWeightedSimByConc(i,1) = weightedSim;
end
% Calculate similarity correlations for the equated dataset
repFeatureCorrsSim = zeros(nFeatures, 1);
for i = 1:nFeatures
    equatedFeatureCorrsSim(i) = corr(concInfo.equatedDatasetWeightedSimByConc(equatedDatasetIndices(minCorrIdx,:)), ...
                                concInfo.features(equatedDatasetIndices(minCorrIdx,:), i), ...
                                'type', 'Spearman');
end


%% Visualization and Comparison of Datasets
% Plot comparison of original vs new correlations vs representative dataset vs equated dataset
figure(9); clf;

% Sort features by original similarity correlations
[sortedFeatureCorrs, sortIdx] = sort(featureCorrs, 'descend');
sortedFeatureCorrsSim = featureCorrsSim(sortIdx);
sortedNewFeatureCorrsSim = newFeatureCorrsSim(sortIdx);
sortedRepFeatureCorrsSim = repFeatureCorrsSim(sortIdx);
sortedEquatedFeatureCorrsSim = equatedFeatureCorrsSim(sortIdx);
sortedFeatureLabels = concInfo.featureLabels(sortIdx);

% Create bar chart with sorted data
b = bar([sortedFeatureCorrs, sortedNewFeatureCorrsSim, sortedFeatureCorrsSim', sortedRepFeatureCorrsSim, sortedEquatedFeatureCorrsSim]);
title('Comparison of Feature Correlations Across Datasets');
ylabel('Correlation');
xticks(1:length(sortedFeatureLabels));
xticklabels(sortedFeatureLabels);
xtickangle(45);
legend('Memorability in orig', 'Summed sim in new dataset', 'Original sim', 'Representative dataset sim', 'Equated dataset sim');

% Plot scatter of original vs flip memorability dataset correlations
figure(10); clf;
scatter(featureCorrsSim, newFeatureCorrsSim);
text(featureCorrsSim, newFeatureCorrsSim, concInfo.featureLabels);
xlabel('Original summed sim');
ylabel('Flipped memorability dataset summed sim');
title('Original vs Flipped Memorability Dataset Similarity Correlations');
AddCorrToScatter();

% Plot scatter of original vs representative dataset correlations
figure(11); clf;
scatter(featureCorrsSim, repFeatureCorrsSim);
text(featureCorrsSim, repFeatureCorrsSim, concInfo.featureLabels);
xlabel('Original summed sim');
ylabel('Representative dataset summed sim');
title('Original vs Representative Dataset Similarity Correlations');
AddCorrToScatter();

% Plot scatter of original vs equated dataset correlations
figure(14); clf;
scatter(featureCorrsSim, equatedFeatureCorrsSim);
text(featureCorrsSim, equatedFeatureCorrsSim, concInfo.featureLabels);
xlabel('Original summed sim');
ylabel('Equated dataset summed sim');
title('Original vs Equated Dataset Similarity Correlations');
AddCorrToScatter();

% PCA visualization including all datasets
figure(12); clf;

nPC = 3;
Xpca = concInfo.features';
[pcaFeatures, explained] = pcaCoeffsExplained(Xpca, nPC);
% Project features into PCA space

% Subplot 1: Original dataset colored by weighted similarity
subplot(2, 2, 1);
concCnt = concInfo.conceptNumber./sum(concInfo.conceptNumber);
markerSizes = 10 + concCnt/max(concCnt) * 100;
s1 = scatter3(pcaFeatures(:,1), pcaFeatures(:,2), pcaFeatures(:,3), markerSizes, ...
              concInfo.weightedSimByConc, 'filled', 'MarkerFaceAlpha', 0.3);
title('Original Dataset - PCA of Feature Space');
xlabel(['PC1 (' num2str(explained(1), '%.1f') '%)']);
ylabel(['PC2 (' num2str(explained(2), '%.1f') '%)']);
zlabel(['PC3 (' num2str(explained(3), '%.1f') '%)']);
colorbar;
grid on;
s1.DataTipTemplate.DataTipRows(1).Label = 'PC1';
s1.DataTipTemplate.DataTipRows(2).Label = 'PC2';
s1.DataTipTemplate.DataTipRows(3).Label = 'PC3';
row = dataTipTextRow('Concept', concInfo.concepts);
row2 = dataTipTextRow('Weighted Sim', concInfo.weightedSimByConc);
s1.DataTipTemplate.DataTipRows(end+1) = row;
s1.DataTipTemplate.DataTipRows(end+1) = row2;

% Subplot 2: Representative dataset colored by weighted similarity
subplot(2, 2, 2);
repWeightMarkerSizes = 10 + (representativeDataset.numImagesForConcept/max(representativeDataset.numImagesForConcept)) * 100;
s3 = scatter3(pcaFeatures(:,1), pcaFeatures(:,2), pcaFeatures(:,3), repWeightMarkerSizes, ...
              concInfo.repDatasetWeightedSimByConc, 'filled', 'MarkerFaceAlpha', 0.3);
title('Representative Dataset - PCA of Feature Space');
xlabel(['PC1 (' num2str(explained(1), '%.1f') '%)']);
ylabel(['PC2 (' num2str(explained(2), '%.1f') '%)']);
zlabel(['PC3 (' num2str(explained(3), '%.1f') '%)']);
colorbar;
grid on;
s3.DataTipTemplate.DataTipRows(1).Label = 'PC1';
s3.DataTipTemplate.DataTipRows(2).Label = 'PC2';
s3.DataTipTemplate.DataTipRows(3).Label = 'PC3';
row1 = dataTipTextRow('Concept', concInfo.concepts);
row2 = dataTipTextRow('Weight', representativeDataset.numImagesForConcept);
row3 = dataTipTextRow('Weighted Sim', concInfo.repDatasetWeightedSimByConc);
s3.DataTipTemplate.DataTipRows(end+1) = row1;
s3.DataTipTemplate.DataTipRows(end+1) = row2;
s3.DataTipTemplate.DataTipRows(end+1) = row3;

% Subplot 3: New dataset colored by weighted similarity
subplot(2, 2, 3);
weightMarkerSizes = 10 + (flipMemorabilityDataset.numImagesForConcept/max(flipMemorabilityDataset.numImagesForConcept)) * 100;
s2 = scatter3(pcaFeatures(:,1), pcaFeatures(:,2), pcaFeatures(:,3), weightMarkerSizes, ...
              concInfo.newDatasetWeightedSimByConc, 'filled', 'MarkerFaceAlpha', 0.3);
title('Flip Memorability Dataset - PCA of Feature Space');
xlabel(['PC1 (' num2str(explained(1), '%.1f') '%)']);
ylabel(['PC2 (' num2str(explained(2), '%.1f') '%)']);
zlabel(['PC3 (' num2str(explained(3), '%.1f') '%)']);
colorbar;
grid on;
s2.DataTipTemplate.DataTipRows(1).Label = 'PC1';
s2.DataTipTemplate.DataTipRows(2).Label = 'PC2';
s2.DataTipTemplate.DataTipRows(3).Label = 'PC3';
row1 = dataTipTextRow('Concept', concInfo.concepts);
row2 = dataTipTextRow('Weight', flipMemorabilityDataset.numImagesForConcept);
row3 = dataTipTextRow('Weighted Sim', concInfo.newDatasetWeightedSimByConc);
s2.DataTipTemplate.DataTipRows(end+1) = row1;
s2.DataTipTemplate.DataTipRows(end+1) = row2;
s2.DataTipTemplate.DataTipRows(end+1) = row3;

% Subplot 4: Equated dataset colored by weighted similarity
subplot(2, 2, 4);
equatedWeightMarkerSizes = 10 + (equatedDataset.numImagesForConcept/max(equatedDataset.numImagesForConcept)) * 100;
s4 = scatter3(pcaFeatures(:,1), pcaFeatures(:,2), pcaFeatures(:,3), equatedWeightMarkerSizes, ...
              concInfo.equatedDatasetWeightedSimByConc, 'filled', 'MarkerFaceAlpha', 0.3);
title('Equated Dataset - PCA of Feature Space');
xlabel(['PC1 (' num2str(explained(1), '%.1f') '%)']);
ylabel(['PC2 (' num2str(explained(2), '%.1f') '%)']);
zlabel(['PC3 (' num2str(explained(3), '%.1f') '%)']);
colorbar;
grid on;
s4.DataTipTemplate.DataTipRows(1).Label = 'PC1';
s4.DataTipTemplate.DataTipRows(2).Label = 'PC2';
s4.DataTipTemplate.DataTipRows(3).Label = 'PC3';
row1 = dataTipTextRow('Concept', concInfo.concepts);
row2 = dataTipTextRow('Weight', equatedDataset.numImagesForConcept);
row3 = dataTipTextRow('Weighted Sim', concInfo.equatedDatasetWeightedSimByConc);
s4.DataTipTemplate.DataTipRows(end+1) = row1;
s4.DataTipTemplate.DataTipRows(end+1) = row2;
s4.DataTipTemplate.DataTipRows(end+1) = row3;


% Create 2D PCA visualization
figure(13); clf;

% Define common colorbar limits
colorLimits = [0.17 0.19];

% Subplot 1: All concepts colored by weighted similarity
subplot(2, 2, 1);
s1 = scatter(pcaFeatures(:,1), pcaFeatures(:,2), 50, concInfo.weightedSimByConc, 'filled', 'MarkerFaceAlpha', 0.3);
title('All Concepts - PCA of Feature Space');
xlabel(['PC1 (' num2str(explained(1), '%.1f') '%)']);
ylabel(['PC2 (' num2str(explained(2), '%.1f') '%)']);
colorbar('Limits', colorLimits);
caxis(colorLimits);
grid on;
s1.DataTipTemplate.DataTipRows(1).Label = 'PC1';
s1.DataTipTemplate.DataTipRows(2).Label = 'PC2';
row = dataTipTextRow('Concept', concInfo.concepts);
row2 = dataTipTextRow('Weighted Sim', concInfo.weightedSimByConc);
s1.DataTipTemplate.DataTipRows(end+1) = row;
s1.DataTipTemplate.DataTipRows(end+1) = row2;

% Subplot 2: Representative dataset colored by weighted similarity
subplot(2, 2, 2);
repWeightMarkerSizes = 10 + (repWeights/max(repWeights)) * 100;
s3 = scatter(pcaFeatures(:,1), pcaFeatures(:,2), repWeightMarkerSizes, ...
             concInfo.repDatasetWeightedSimByConc, 'filled', 'MarkerFaceAlpha', 0.3);
title('Representative Dataset - PCA of Feature Space');
xlabel(['PC1 (' num2str(explained(1), '%.1f') '%)']);
ylabel(['PC2 (' num2str(explained(2), '%.1f') '%)']);
colorbar('Limits', colorLimits);
caxis(colorLimits);
grid on;
s3.DataTipTemplate.DataTipRows(1).Label = 'PC1';
s3.DataTipTemplate.DataTipRows(2).Label = 'PC2';
row1 = dataTipTextRow('Concept', concInfo.concepts);
row2 = dataTipTextRow('Weight', repWeights);
row3 = dataTipTextRow('Weighted Sim', concInfo.repDatasetWeightedSimByConc);
s3.DataTipTemplate.DataTipRows(end+1) = row1;
s3.DataTipTemplate.DataTipRows(end+1) = row2;
s3.DataTipTemplate.DataTipRows(end+1) = row3;

% Subplot 3: Flip dataset colored by weighted similarity
subplot(2, 2, 3);
weightMarkerSizes = 10 + (weights/max(weights)) * 100;
s2 = scatter(pcaFeatures(:,1), pcaFeatures(:,2), weightMarkerSizes, ...
             concInfo.newDatasetWeightedSimByConc, 'filled', 'MarkerFaceAlpha', 0.3);
title('Flip Memorability Dataset - PCA of Feature Space');
xlabel(['PC1 (' num2str(explained(1), '%.1f') '%)']);
ylabel(['PC2 (' num2str(explained(2), '%.1f') '%)']);
colorbar('Limits', colorLimits);
caxis(colorLimits);
grid on;
s2.DataTipTemplate.DataTipRows(1).Label = 'PC1';
s2.DataTipTemplate.DataTipRows(2).Label = 'PC2';
row1 = dataTipTextRow('Concept', concInfo.concepts);
row2 = dataTipTextRow('Weight', weights);
row3 = dataTipTextRow('Weighted Sim', concInfo.newDatasetWeightedSimByConc);
s2.DataTipTemplate.DataTipRows(end+1) = row1;
s2.DataTipTemplate.DataTipRows(end+1) = row2;
s2.DataTipTemplate.DataTipRows(end+1) = row3;

% Subplot 4: Equated dataset colored by weighted similarity
subplot(2, 2, 4);
equatedWeightMarkerSizes = 10 + (equatedWeights/max(equatedWeights)) * 100;
s4 = scatter(pcaFeatures(:,1), pcaFeatures(:,2), equatedWeightMarkerSizes, ...
             concInfo.equatedDatasetWeightedSimByConc, 'filled', 'MarkerFaceAlpha', 0.3);
title('Equated Dataset - PCA of Feature Space');
xlabel(['PC1 (' num2str(explained(1), '%.1f') '%)']);
ylabel(['PC2 (' num2str(explained(2), '%.1f') '%)']);
colorbar('Limits', colorLimits);
caxis(colorLimits);
grid on;
s4.DataTipTemplate.DataTipRows(1).Label = 'PC1';
s4.DataTipTemplate.DataTipRows(2).Label = 'PC2';
row1 = dataTipTextRow('Concept', concInfo.concepts);
row2 = dataTipTextRow('Weight', equatedWeights);
row3 = dataTipTextRow('Weighted Sim', concInfo.equatedDatasetWeightedSimByConc);
s4.DataTipTemplate.DataTipRows(end+1) = row1;
s4.DataTipTemplate.DataTipRows(end+1) = row2;
s4.DataTipTemplate.DataTipRows(end+1) = row3;


%% Create image montages for each dataset
fprintf('\nCreating image montages for each dataset...\n');

% 

% Create a vector with each equatedDataset.numImagesForConcept replicated that many times
% (excluding concepts with 0 images)
equatedDatasetReplicatedCounts = [];
for i = 1:length(equatedDataset.numImagesForConcept)
    if equatedDataset.numImagesForConcept(i) > 0
        replicatedValue = repmat(concInfo.equatedDatasetWeightedSimByConc(i), 1, equatedDataset.numImagesForConcept(i));
        equatedDatasetReplicatedCounts = [equatedDatasetReplicatedCounts, replicatedValue];
    end
end

% For flip memorability dataset
flipMemorabilityReplicatedCounts = [];
for i = 1:length(flipMemorabilityDataset.numImagesForConcept)
    if flipMemorabilityDataset.numImagesForConcept(i) > 0
        replicatedValue = repmat(concInfo.newDatasetWeightedSimByConc(i), 1, flipMemorabilityDataset.numImagesForConcept(i));
        flipMemorabilityReplicatedCounts = [flipMemorabilityReplicatedCounts, replicatedValue];
    end
end

% For representative dataset
representativeReplicatedCounts = [];
for i = 1:length(representativeDataset.numImagesForConcept)
    if representativeDataset.numImagesForConcept(i) > 0
        replicatedValue = repmat(concInfo.repDatasetWeightedSimByConc(i), 1, representativeDataset.numImagesForConcept(i));
        representativeReplicatedCounts = [representativeReplicatedCounts, replicatedValue];
    end
end

% Display summary statistics for each dataset
fprintf('\nDataset summary statistics:\n');
fprintf('Flip Memorability Dataset: Mean = %.4f, Std = %.4f, Min = %.4f, Max = %.4f, N = %d\n', ...
    mean(flipMemorabilityReplicatedCounts), std(flipMemorabilityReplicatedCounts), ...
    min(flipMemorabilityReplicatedCounts), max(flipMemorabilityReplicatedCounts), ...
    length(flipMemorabilityReplicatedCounts));

fprintf('Representative Dataset: Mean = %.4f, Std = %.4f, Min = %.4f, Max = %.4f, N = %d\n', ...
    mean(representativeReplicatedCounts), std(representativeReplicatedCounts), ...
    min(representativeReplicatedCounts), max(representativeReplicatedCounts), ...
    length(representativeReplicatedCounts));

fprintf('Equated Dataset: Mean = %.4f, Std = %.4f, Min = %.4f, Max = %.4f, N = %d\n', ...
    mean(equatedDatasetReplicatedCounts), std(equatedDatasetReplicatedCounts), ...
    min(equatedDatasetReplicatedCounts), max(equatedDatasetReplicatedCounts), ...
    length(equatedDatasetReplicatedCounts));



% Create montages for each dataset
%createDatasetMontage(flipMemorabilityDataset, 'FlipMemorabilityDatasetMontage.png', true);
%createDatasetMontage(representativeDataset, 'RepresentativeDatasetMontage.png', true);
%createDatasetMontage(equatedDataset, 'EquatedDatasetMontage.png', true);

fprintf('Montage creation complete.\n');

keyboard;

% Function to create a montage from a dataset
function createDatasetMontage(dataset, outputFilename, shuffledOrder)
  % Parameters
  thumbnailSize = [50, 50];
  gridSize = [40, 40];
  
  % Initialize the large image
  montageImg = zeros(gridSize(1) * thumbnailSize(1), gridSize(2) * thumbnailSize(2), 3, 'uint8');
  
  % Determine how many images to include (up to 1600)
  numImages = min(length(dataset.imageFilename), gridSize(1) * gridSize(2));
  
  order = 1:numImages;
  if shuffledOrder
    order = randperm(numImages);
  end

  % Load and place each image
  for i = 1:numImages
      % Construct the path to the image
      conceptName = dataset.conceptName{order(i)};
      filename = dataset.imageFilename{order(i)};
      imagePath = fullfile('external', 'things_images', conceptName, filename);
      
      % Load and resize the image
      img = imread(imagePath);
      img = imresize(img, thumbnailSize);
      
      % Calculate position in the grid
      row = ceil(i / gridSize(2));
      col = mod(i - 1, gridSize(2)) + 1;
      
      % Calculate pixel positions
      rowStart = (row - 1) * thumbnailSize(1) + 1;
      rowEnd = row * thumbnailSize(1);
      colStart = (col - 1) * thumbnailSize(2) + 1;
      colEnd = col * thumbnailSize(2);
      
      % Place the image in the montage
      montageImg(rowStart:rowEnd, colStart:colEnd, :) = img;
  end
  
  % Save the montage
  imwrite(montageImg, outputFilename);
  fprintf('Saved montage to %s\n', outputFilename);
end
