function measures = fig1_measures(figureData)
%FIG1_MEASURES  Every number of Fig. 1 from one run of the REM simulation, with each similarity
%   measure and each correlation named.
%
%   Input
%     figureData  the struct returned by fig1_remSim.m (or the struct of the saved run,
%                 data/paper_run/fig1_paper_run.mat)
%
%   Output
%     measures    struct with one field per number (listed below) and two per-item vectors
%                 (summedInverseDistance, the x axis of Fig. 1b, and memorability)
%
%   The four measures of item similarity, each correlated with memorability (each item's distances
%   are Euclidean distances to the other items in the 49-dimensional feature space):
%     similarity = 1 / mean distance    Pearson r with memorability: the r = -0.56 of Fig. 1b
%                                       (stored by the simulation as r_sim)
%     sum over the other items of       Pearson r with memorability: the quantity plotted on the
%       1 / distance                    x axis of Fig. 1b ("Summed Similarity")
%     mean distance                     Pearson r with memorability (positive: items far from the
%                                       others are better remembered; stored as "correlation")
%     negated mean distance             Spearman rho with memorability (the rho = -0.55 of
%                                       Methods, Basic REM Simulation of Memorability, and the
%                                       measure of SI Appendix, Fig. S2; it has the same ranks as
%                                       1 / mean distance, so it is also the Spearman rho of
%                                       similarity)
%   The mean distance is taken over all 1,600 items including the item itself (distance 0), as the
%   simulation computes it; dividing by 1,599 instead would change no correlation.
%
%   The feature-level numbers (Figs. 1c and 1d) use Spearman correlations: each feature's
%   correlation with memorability and with similarity (1 / mean distance) across the items, and
%   the Spearman correlation of those two sets of 49 correlations across the features (the
%   rho = -0.44 of Fig. 1d, stored as r_meta).

memorability = figureData.hitFARate(:);
hitRate = figureData.hitRate(:);
falseAlarmRate = figureData.faRate(:);
itemFeatures = double(figureData.imagesOfInterest);
pairwiseDistances = double(figureData.pairwiseDistances);
meanDistance = figureData.avgDistances(:);
similarity = figureData.similarity(:);
testedOldMask = double(figureData.simHitMask);
testedNewMask = double(figureData.simFaMask);

measures = struct();

%% Design of the run, counted from what the simulation stored
measures.seed = figureData.seedValue;
measures.numberOfSimulations = size(testedOldMask, 2);
measures.numberOfItems = size(itemFeatures, 1);
measures.numberOfFeatures = size(itemFeatures, 2);
measures.smallestFeatureValue = min(itemFeatures(:));
measures.largestFeatureValue = max(itemFeatures(:));

% The study-list length and the size of the generated pool cannot be counted from the stored
% results (studied items that are not tested leave no trace); fig1_remSim.m returns them. The
% saved run in data/paper_run/ does not store them.
if isfield(figureData, 'studySize')
    measures.studySize = figureData.studySize;
    measures.itemsGeneratedBeforeDrawingTheSet = figureData.thingsDatasetSize;
else
    measures.studySize = NaN;
    measures.itemsGeneratedBeforeDrawingTheSet = NaN;
end

% Old and new test items in every simulated experiment (each simulation tests the same numbers)
oldItemsPerSimulation = sum(testedOldMask, 1);
newItemsPerSimulation = sum(testedNewMask, 1);
measures.oldTestItemsPerSimulation = unique(oldItemsPerSimulation);
measures.newTestItemsPerSimulation = unique(newItemsPerSimulation);
if numel(measures.oldTestItemsPerSimulation) > 1 || numel(measures.newTestItemsPerSimulation) > 1
    error('fig1_measures:UnequalTests', 'the simulations do not all test the same number of items');
end
measures.testItemsPerSimulation = measures.oldTestItemsPerSimulation + measures.newTestItemsPerSimulation;

% No item is both an old test item and a foil in the same simulation
measures.itemOldAndNewInSameSimulation = any(testedOldMask(:) > 0 & testedNewMask(:) > 0);

% Items tested both as an old item and as a foil, and how often each item is tested in each role
timesTestedOld = sum(testedOldMask, 2);
timesTestedAsFoil = sum(testedNewMask, 2);
measures.itemsWithHitAndFalseAlarmRate = sum(timesTestedOld > 0 & timesTestedAsFoil > 0);
measures.fewestTimesTestedOld = min(timesTestedOld);
measures.mostTimesTestedOld = max(timesTestedOld);
measures.fewestTimesTestedAsFoil = min(timesTestedAsFoil);
measures.mostTimesTestedAsFoil = max(timesTestedAsFoil);

%% Model parameters the run used
measures.baseRateOfFeatures = figureData.baseRateOfFeatures;      % g
measures.probEncoding = figureData.probEncoding;                  % u
measures.correctEncodeProb = figureData.correctEncodeProb;        % c
measures.criterion = figureData.criterion;

%% Item memorability (hit rate minus false alarm rate)
measures.meanHitRate = mean(hitRate);
measures.meanFalseAlarmRate = mean(falseAlarmRate);
measures.meanMemorability = mean(memorability);
measures.highestMemorability = max(memorability);
measures.lowestMemorability = min(memorability);

%% Item memorability against item similarity (Fig. 1b), every measure named
% Similarity = 1 / mean distance, Pearson (the r = -0.56 of Fig. 1b). The p value corr returns comes
% from whichever tcdf.m is first on the MATLAB path, and some toolboxes' own tcdf.m return 0 for a p
% this small. The two-sided t-test p is therefore also computed directly with the incomplete beta
% function, which gives the same value on every MATLAB: p = I(df / (df + t^2); df / 2, 1 / 2),
% t = r sqrt(df / (1 - r^2)).
[measures.pearsonMemorabilityVsInverseMeanDistance, measures.pValueMemorabilityVsInverseMeanDistance] = ...
    corr(similarity, memorability, 'rows', 'complete');
degreesOfFreedom = sum(~isnan(memorability)) - 2;
tStatistic = measures.pearsonMemorabilityVsInverseMeanDistance * ...
    sqrt(degreesOfFreedom / (1 - measures.pearsonMemorabilityVsInverseMeanDistance ^ 2));
measures.tTestPValueMemorabilityVsInverseMeanDistance = betainc(degreesOfFreedom / (degreesOfFreedom + tStatistic ^ 2), ...
    degreesOfFreedom / 2, 0.5);

% Sum over the other items of 1 / distance, Pearson (the x axis of Fig. 1b). Computed as
% 1 / (distance + eps) for every pair, the item's similarity to itself (the diagonal) set to 0,
% then summed over each item's row.
numberOfItems = measures.numberOfItems;
pairwiseInverseDistance = 1 ./ (pairwiseDistances + eps);
pairwiseInverseDistance(1:numberOfItems + 1:end) = 0;
summedInverseDistance = sum(pairwiseInverseDistance, 2);
measures.pearsonMemorabilityVsSummedInverseDistance = corr(summedInverseDistance, memorability, 'rows', 'complete');
measures.smallestSummedInverseDistance = min(summedInverseDistance);
measures.largestSummedInverseDistance = max(summedInverseDistance);

% Mean distance, Pearson
measures.pearsonMemorabilityVsMeanDistance = corr(meanDistance, memorability, 'rows', 'complete');

% Negated mean distance, Spearman
measures.spearmanMemorabilityVsNegatedMeanDistance = corr(-meanDistance, memorability, ...
    'Type', 'Spearman', 'rows', 'complete');

%% Feature level (Figs. 1c and 1d)
featureCorrelationsWithMemorability = figureData.featureCorrsMemory(:);
featureCorrelationsWithSimilarity = figureData.featureCorrsSimil(:);
featurePValuesWithMemorability = figureData.featurePvalsMemory(:);

measures.featuresCorrelatedWithMemorabilityP05 = sum(featurePValuesWithMemorability < 0.05);
measures.featuresPositivelyCorrelatedWithMemorability = sum(featureCorrelationsWithMemorability > 0);
measures.featuresSignificantForMemorabilityAndSimilarity = sum(figureData.bothSig(:) > 0);
measures.weakestFeatureMemorabilityCorrelation = min(featureCorrelationsWithMemorability);
measures.strongestFeatureMemorabilityCorrelation = max(featureCorrelationsWithMemorability);
measures.lowestFeatureSimilarityCorrelation = min(featureCorrelationsWithSimilarity);
measures.highestFeatureSimilarityCorrelation = max(featureCorrelationsWithSimilarity);

% Across the 49 features: Spearman (the rho = -0.44 of Fig. 1d) and Pearson
[measures.spearmanAcrossFeatures, measures.pValueSpearmanAcrossFeatures] = corr( ...
    featureCorrelationsWithSimilarity, featureCorrelationsWithMemorability, 'Type', 'Spearman');
measures.pearsonAcrossFeatures = corr(featureCorrelationsWithSimilarity, featureCorrelationsWithMemorability);

%% Per-item vectors the figure needs
measures.summedInverseDistance = summedInverseDistance;
measures.memorability = memorability;
end
