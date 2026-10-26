function fig5 = things_reanalysis()
%THINGS_REANALYSIS  Fig. 5: the THINGS feature reanalysis.
%
% PAPER
%   Fig. 5; Representational Analyses: Memorability Follows the Structure of the Image Set (subsection
%   The Similarity Structure of an Image Set Determines What Will Be Memorable); Methods,
%   Representational Similarity Analysis.
%
% WHAT IT DOES
%   Kramer et al. (2023) showed that some of the 49 features of the THINGS similarity embedding
%   (Hebart et al., 2020) predict which concepts are well remembered. This script asks whether those
%   same features are the ones that make a concept distinctive inside the THINGS database. For each
%   of the 49 features of a representational space it computes
%     (1) the feature's Spearman correlation with concept memorability (Kramer et al.'s corrected
%         recognition, averaged over the concept's images; 1,853 of the 1,854 concepts have a score),
%     (2) the feature's Spearman correlation with the concept's summed similarity to all other
%         THINGS concepts (all 1,854 concepts; step 3 below rebuilds this measure),
%   and then the Pearson correlation of (1) with (2) across the 49 features.
%   Two representational spaces:
%     human  = the 49 SPoSE features of Hebart et al. (2020)                    -> Fig. 5c
%     monkey = the first 49 principal components of the concept-averaged
%              multi-unit responses of the IT arrays of the THINGS Ventral
%              Stream Spiking Dataset (Papale et al., 2025)                      -> Fig. 5d
%   It also draws Fig. 5a (the 49 human features ranked by their memorability correlation) and
%   Fig. 5b (all 1,854 concepts on the first three principal components of the 49-feature space,
%   colored by summed similarity).
%
% INPUTS (all in the repository)
%   shared/things/concept_info_human.mat and concept_info_monkey.mat, read by
%       shared/matlab/load_concept_info.m (features, featureLabels, concepts, conceptNumber,
%       memorabilityCat, weightedSimByConc, conceptCounts; the concept-by-concept similarity
%       matrix thingsSimRSM = corrcoef(features') is computed there)
%   shared/things/things_images.csv              the public per-image memorability scores (check of
%                                                the concept memorability, step 2)
%   shared/matlab/pca3d_feature_space_view.csv   camera angles of Fig. 5b
%   05_things_reanalysis_fig5/data/paper_run/paperStats_kramer.mat   r of the saved run the paper
%                                                reports (checked in step 12)
%
% OUTPUTS (in 05_things_reanalysis_fig5/outputs/)
%   figures/fig5a_ranked_feature_memorability.png                   Fig. 5a (300 dpi)
%   figures/fig5b_pca_feature_space.png                              Fig. 5b
%   figures/fig5c_human_features_similarity_vs_memorability.png     Fig. 5c
%   figures/fig5d_monkey_features_similarity_vs_memorability.png    Fig. 5d
%   figure_data/<the same four names>.csv                            the numbers each panel plots
%   fig5_statistics.csv     one row per space: Pearson r, number of features and concepts, the
%                           least-squares line, the r of the saved run in data/paper_run/ and the
%                           text printed in the panel
%   Return value fig5: a struct with every number run_module.m writes into outputs/numbers.csv.
%
% SEEDS: none (the script draws no random numbers).
%
% HOW TO RUN (from the repository root; run_module.m calls this function and then writes numbers.csv)
%   matlab -batch "restoredefaultpath; cd('05_things_reanalysis_fig5/code'); run_module"

close all;

%% 1. Folders and the shared helpers
codeFolder = fileparts(mfilename('fullpath'));
addpath(fullfile(codeFolder, '..', '..', 'shared', 'matlab'));
paths = release_paths(mfilename('fullpath'));

%% 2. Human similarity space: the concept table
% concInfo.features            1,854 concepts x 49 SPoSE features (THINGS concept order)
% concInfo.memorabilityCat     concept memorability (NaN for the one concept without a score)
% concInfo.weightedSimByConc   summed similarity of each concept to all other THINGS concepts
% concInfo.conceptCounts       number of THINGS images of each concept (as stored)
% concInfo.thingsSimRSM        1,854 x 1,854 Pearson correlations between the concepts' feature rows
%                              (the Methods' dissimilarity is 1 minus this correlation)
concInfo = load_concept_info('human');
nFeatures = size(concInfo.features, 2);
numberOfConcepts = size(concInfo.features, 1);
numberOfConceptsWithScore = sum(~isnan(concInfo.memorabilityCat));
fprintf('\nHuman space: %d concepts x %d features; similarity matrix %d x %d (correlations from %.3f to %.3f)\n', ...
    numberOfConcepts, nFeatures, size(concInfo.thingsSimRSM, 1), size(concInfo.thingsSimRSM, 2), ...
    min(concInfo.thingsSimRSM(:)), max(concInfo.thingsSimRSM(:)));

% Check of the concept memorability (Methods, Representational Similarity Analysis): the stored score
% of each concept is the mean of the public per-image scores of Kramer et al. (2023) over the
% concept's images.
imageTable = readtable(fullfile(paths.sharedThings, 'things_images.csv'), 'TextType', 'string');
recomputedMemorability = zeros(numberOfConcepts, 1);
for conceptIndex = 1:numberOfConcepts
    imagesOfThisConcept = imageTable.concept == string(concInfo.concepts{conceptIndex});
    recomputedMemorability(conceptIndex) = mean(imageTable.kramer_cr(imagesOfThisConcept));
end
conceptWithoutScore = concInfo.concepts(isnan(concInfo.memorabilityCat));
largestDifferenceMemorability = max(abs(recomputedMemorability - concInfo.memorabilityCat), [], 'omitnan');
sameMissingConcepts = isequal(isnan(recomputedMemorability), isnan(concInfo.memorabilityCat));
fprintf('Concept memorability recomputed from %d image scores: largest difference from the stored values %.2g; concepts without a score: %s\n', ...
    height(imageTable), largestDifferenceMemorability, strjoin(conceptWithoutScore, ', '));
if largestDifferenceMemorability > 1e-12 || ~sameMissingConcepts
    error('things_reanalysis:Memorability', 'The recomputed concept memorability does not equal the stored values.');
end

%% 3. Check of the summed similarity (Methods, Representational Similarity Analysis)
% The stored summed similarity of concept i is the sum over all OTHER concepts j of
%     kernel(similarity(i, j)) x (conceptCounts(j) / sum(conceptCounts)),
% with a Gaussian kernel (normal density) centered on a correlation of 1, which is a dissimilarity
% of 0, with SD 2. Rebuilding it from the similarity matrix with these settings must give the
% stored values (the correlations below use the stored values).
kernelCentreCorrelation = 1;                                % kernel center in correlation units (r = 1)
kernelCentreDissimilarity = 1 - kernelCentreCorrelation;    % the same center as a dissimilarity: 0
kernelSD = 2;                                               % kernel standard deviation
totalImageCount = sum(concInfo.conceptCounts);
rebuiltSummedSimilarityHuman = summed_similarity_full_things(concInfo.thingsSimRSM, ...
    concInfo.conceptCounts, kernelCentreCorrelation, kernelSD);
largestDifferenceHuman = max(abs(rebuiltSummedSimilarityHuman - concInfo.weightedSimByConc));
sameRanksHuman = isequal(tiedrank(rebuiltSummedSimilarityHuman), tiedrank(concInfo.weightedSimByConc));
fprintf('Summed similarity rebuilt (kernel center r = %g, i.e. dissimilarity %g; SD %g; weights = image count / %d): largest difference from the stored values %.2g; same ranks: %d\n', ...
    kernelCentreCorrelation, kernelCentreDissimilarity, kernelSD, totalImageCount, largestDifferenceHuman, sameRanksHuman);
if largestDifferenceHuman > 1e-12 || ~sameRanksHuman
    error('things_reanalysis:SummedSimilarity', 'The rebuilt summed similarity does not equal the stored values.');
end

% The dissimilarities between two different concepts and the kernel weights they receive, relative
% to the weight at a dissimilarity of 0 (the dissimilarity 1 - r can run from 0 to 2)
largestPossibleDissimilarity = 2;
kernelWeightAtLargestPossibleDissimilarity = normpdf(largestPossibleDissimilarity, kernelCentreDissimilarity, kernelSD) / ...
    normpdf(kernelCentreDissimilarity, kernelCentreDissimilarity, kernelSD);
[dissimilarityRangeHuman, kernelWeightRangeHuman] = dissimilarity_and_kernel_weight_range(concInfo.thingsSimRSM, ...
    kernelCentreDissimilarity, kernelSD);
fprintf(['Dissimilarity between two concepts: %.3f to %.3f; kernel weight relative to a dissimilarity of 0: ' ...
    '%.2f to %.2f (%.2f at a dissimilarity of %d)\n'], dissimilarityRangeHuman(1), dissimilarityRangeHuman(2), ...
    kernelWeightRangeHuman(1), kernelWeightRangeHuman(2), kernelWeightAtLargestPossibleDissimilarity, ...
    largestPossibleDissimilarity);

%% 4. Per-feature Spearman correlations with memorability (Fig. 5a; the y axis of Fig. 5c)
% 'rows', 'complete' leaves out the concept without a memorability score.
featureOriginalMemorabilityCorrs = zeros(nFeatures, 1);
for i = 1:nFeatures
    rho = corr(concInfo.features(:,i), concInfo.memorabilityCat, ...
        'Type', 'Spearman', 'rows', 'complete');
    featureOriginalMemorabilityCorrs(i) = rho;
end

%% 5. Per-feature Spearman correlations with summed similarity (the x axis of Fig. 5c)
% One call returns a 1 x 49 row: the Spearman correlation of the summed similarity with each
% feature column (a loop over the 49 columns changes the last binary digit of the final r).
[featureCorrsSimOriginal, ~] = corr(concInfo.weightedSimByConc, concInfo.features, ...
    'type', 'Spearman', 'rows', 'complete');

%% 6. Colors: PiYG, ordered by the memorability correlation (pink = lowest, green = highest)
[~, sortIdxOrig] = sort(featureOriginalMemorabilityCorrs, 'ascend');
cmap = piyg_colormap(nFeatures);
featureColors = zeros(nFeatures, 3);
featureColors(sortIdxOrig,:) = cmap;

%% 7. Fig. 5c: each feature's correlation with summed similarity (x) against its correlation with memorability (y)
fprintf('\n=== Fig. 5c, human space: summed similarity vs memorability ===\n');
figureHandle5c = figure('Visible', 'off');
x_orig = featureCorrsSimOriginal(:)';
y_orig = featureOriginalMemorabilityCorrs(:)';
scatter(x_orig, y_orig, 50, featureColors, 'filled');
xlabel({'Spearman \rho with summed similarity', '(human)'});
ylabel('Spearman \rho with memorability');
xlim([-0.6 0.6]);
ylim([-0.4 0.4]);
xticks([-0.6 0 0.6]);
yticks([-0.4 0 0.4]);
set(gca, 'FontSize', 20);

% Pearson correlation across the 49 features, and the least-squares line
validIdx = ~isnan(x_orig) & ~isnan(y_orig);
x_vals = x_orig(validIdx);
y_vals = y_orig(validIdx);
r = corr(x_vals', y_vals');
lineHandle = lsline;
set(lineHandle, 'LineWidth', 3, 'Color', [0.75 0.75 0.75]);
lineCoefficientsHuman = polyfit(x_vals, y_vals, 1);   % slope and intercept of the line lsline draws
panelTextHuman = format_r(r);
put_text_top_left(panelTextHuman);
fprintf('Human space: r = %.5f (Pearson across %d features)\n', r, numel(x_vals));

humanFeatureTable = table((1:nFeatures)', string(concInfo.featureLabels(:)), ...
    featureCorrsSimOriginal(:), featureOriginalMemorabilityCorrs(:), ...
    rank_from_highest(featureOriginalMemorabilityCorrs), rank_from_lowest(featureCorrsSimOriginal(:)), ...
    featureColors(:, 1), featureColors(:, 2), featureColors(:, 3), ...
    'VariableNames', {'feature_number', 'feature_label', 'spearman_rho_summed_similarity', ...
    'spearman_rho_memorability', 'memorability_rank_from_highest', 'summed_similarity_rank_from_most_negative', ...
    'color_red', 'color_green', 'color_blue'});
save_panel_quietly(figureHandle5c, 'fig5c_human_features_similarity_vs_memorability', paths, humanFeatureTable);

%% 8. Fig. 5b: all concepts on the first three principal components of the 49-feature space
% pcaCoeffsExplained(features', 3) centers each concept's 49 feature values on their mean and
% returns the first three right singular vectors: three coordinates per concept. Visualization only.
fprintf('\n=== Fig. 5b: 3-D PCA of the feature space ===\n');
nPC = 3;
[pcaFeatures, ~] = pcaCoeffsExplained(concInfo.features', nPC);
conceptNumber = concInfo.conceptNumber(:);       % position of the concept in the THINGS concept list (1 to 1,854)
weightedSimByConc = concInfo.weightedSimByConc(:);

figureHandle5b = figure('Visible', 'off');
colormap(flipud(parula(256)));                   % yellow = low summed similarity (distinctive), dark blue = high (crowded)
% Marker size: 20 + 200 x conceptNumber / 1,854 points squared (20.1 to 220), growing linearly with
% conceptNumber (the concept's position in the alphabetical THINGS concept list)
concCnt = conceptNumber ./ sum(conceptNumber);
markerSizes = 20 + concCnt / max(concCnt) * 200;
fixedColorLimits = [0.174, 0.185];
fontSizePca = 15;
scatter3(pcaFeatures(:,1), pcaFeatures(:,2), pcaFeatures(:,3), markerSizes, ...
    weightedSimByConc, 'filled', 'MarkerFaceAlpha', 0.3);
title({'Full THINGS image set', '(human similarity data)'}, 'FontSize', fontSizePca);
xlabel('PC1', 'FontSize', fontSizePca);
ylabel('PC2', 'FontSize', fontSizePca);
zlabel('PC3', 'FontSize', fontSizePca);
cb = colorbar;
cb.Limits = fixedColorLimits;
caxis(fixedColorLimits);
stylePca3dAxes(gca, cb, fontSizePca);
% Color bar labels: no numbers, 'More crowded in 49D' above the bar, 'More distinctive in 49D'
% below it. The bar is placed just right of the 3-D axes and made a little shorter than them, so
% that both labels fit inside the figure.
cb.Ticks = [];
cb.Label.String = 'Summed similarity';
axesPosition = get(gca, 'Position');
cb.Position = [axesPosition(1) + axesPosition(3) + 0.08, axesPosition(2) + 0.08, 0.035, axesPosition(4) - 0.16];
labelWidth = 0.34;
annotation(figureHandle5b, 'textbox', [cb.Position(1) + cb.Position(3) / 2 - labelWidth / 2, ...
    cb.Position(2) + cb.Position(4), labelWidth, 0.07], 'String', 'More crowded in 49D', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', fontSizePca);
annotation(figureHandle5b, 'textbox', [cb.Position(1) + cb.Position(3) / 2 - labelWidth / 2, ...
    cb.Position(2) - 0.07, labelWidth, 0.07], 'String', 'More distinctive in 49D', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', fontSizePca);

% The two example concepts of the caption, face and chainsaw: a black-edged marker and the concept
% name (THINGS photographs are not included in the repository).
exampleConcepts = {'face', 'chainsaw'};
exampleSummedSimilarity = zeros(1, numel(exampleConcepts));
examplePercentLower = zeros(1, numel(exampleConcepts));    % percent of the concepts with a lower summed similarity
examplePercentHigher = zeros(1, numel(exampleConcepts));   % percent of the concepts with a higher summed similarity
hold on;
for exampleIndex = 1:numel(exampleConcepts)
    conceptRow = find(strcmp(concInfo.concepts, exampleConcepts{exampleIndex}), 1);
    scatter3(pcaFeatures(conceptRow, 1), pcaFeatures(conceptRow, 2), pcaFeatures(conceptRow, 3), 120, ...
        weightedSimByConc(conceptRow), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    text(pcaFeatures(conceptRow, 1), pcaFeatures(conceptRow, 2), pcaFeatures(conceptRow, 3), ...
        ['  ' exampleConcepts{exampleIndex}], 'FontSize', fontSizePca, 'FontWeight', 'bold');
    exampleSummedSimilarity(exampleIndex) = weightedSimByConc(conceptRow);
    examplePercentLower(exampleIndex) = 100 * mean(weightedSimByConc < weightedSimByConc(conceptRow));
    examplePercentHigher(exampleIndex) = 100 * mean(weightedSimByConc > weightedSimByConc(conceptRow));
    fprintf('Fig. 5b example concept %s: summed similarity %.4f (%.1f percent of the concepts lower, %.1f percent higher)\n', ...
        exampleConcepts{exampleIndex}, exampleSummedSimilarity(exampleIndex), examplePercentLower(exampleIndex), ...
        examplePercentHigher(exampleIndex));
end
hold off;
viewSpec = readPca3dFeatureSpaceView('', 'original');
applyAxesPca3dView(gca, viewSpec);
fprintf('Fig. 5b: view(%.5g, %.5g); camva(%.5g); camroll(%.5g)\n', ...
    viewSpec.az, viewSpec.el, viewSpec.camva, viewSpec.camroll);

isExampleConcept = ismember(concInfo.concepts(:), exampleConcepts);
pcaTable = table(conceptNumber, string(concInfo.concepts(:)), pcaFeatures(:, 1), pcaFeatures(:, 2), ...
    pcaFeatures(:, 3), weightedSimByConc, markerSizes, isExampleConcept, ...
    'VariableNames', {'concept_number', 'concept', 'pc1', 'pc2', 'pc3', 'summed_similarity', 'marker_size', ...
    'labeled_example'});
save_panel_quietly(figureHandle5b, 'fig5b_pca_feature_space', paths, pcaTable);

%% 9. Fig. 5a: the 49 features ranked by their correlation with memorability
fprintf('\n=== Fig. 5a: ranked feature-memorability correlations ===\n');
figureHandle5a = figure('Visible', 'off');
set(gcf, 'Position', [100 100 800 500]);
[sortedMemCorrs, sortIdxMem] = sort(featureOriginalMemorabilityCorrs, 'descend');
sortedLabels = concInfo.featureLabels(sortIdxMem);
scatter(1:nFeatures, sortedMemCorrs, 150, featureColors(sortIdxMem, :), 'filled');
hold on;
plot([0, nFeatures + 1], [0, 0], 'k--', 'LineWidth', 2);
xlim([0, nFeatures + 1]);
ylim([-0.4, 0.4]);
xlabel('Feature (ranked by Spearman \rho with memorability)');
ylabel('Spearman \rho with memorability');
xticks(1:5:nFeatures);
xticklabels(sortedLabels(1:5:end));
xtickangle(45);
set(gca, 'FontSize', 20, 'LineWidth', 2);
fprintf('Highest: %s (rho = %.3f); lowest: %s (rho = %.3f)\n', sortedLabels{1}, sortedMemCorrs(1), ...
    sortedLabels{end}, sortedMemCorrs(end));
fprintf('Tick labels (every fifth feature): %s\n', strjoin(sortedLabels(1:5:end), ' | '));

rankedTable = table((1:nFeatures)', sortIdxMem(:), string(sortedLabels(:)), sortedMemCorrs(:), ...
    ismember((1:nFeatures)', 1:5:nFeatures), ...
    featureColors(sortIdxMem, 1), featureColors(sortIdxMem, 2), featureColors(sortIdxMem, 3), ...
    'VariableNames', {'rank', 'feature_number', 'feature_label', 'spearman_rho_memorability', ...
    'has_tick_label', 'color_red', 'color_green', 'color_blue'});
save_panel_quietly(figureHandle5a, 'fig5a_ranked_feature_memorability', paths, rankedTable);

%% 10. Monkey IT space: the same analysis with the 49 IT principal components as the features
concInfoMonkey = load_concept_info('monkey');
nFeaturesMonkey = size(concInfoMonkey.features, 2);
fprintf('\nMonkey space: %d concepts x %d components; similarity matrix %d x %d (correlations from %.3f to %.3f)\n', ...
    size(concInfoMonkey.features, 1), nFeaturesMonkey, size(concInfoMonkey.thingsSimRSM, 1), ...
    size(concInfoMonkey.thingsSimRSM, 2), min(concInfoMonkey.thingsSimRSM(:)), max(concInfoMonkey.thingsSimRSM(:)));

% The same check of the summed similarity, in the monkey space (same image-count weights, same kernel)
rebuiltSummedSimilarityMonkey = summed_similarity_full_things(concInfoMonkey.thingsSimRSM, ...
    concInfoMonkey.conceptCounts, kernelCentreCorrelation, kernelSD);
largestDifferenceMonkey = max(abs(rebuiltSummedSimilarityMonkey - concInfoMonkey.weightedSimByConc));
sameRanksMonkey = isequal(tiedrank(rebuiltSummedSimilarityMonkey), tiedrank(concInfoMonkey.weightedSimByConc));
fprintf('Monkey space: summed similarity rebuilt: largest difference from the stored values %.2g; same ranks: %d\n', ...
    largestDifferenceMonkey, sameRanksMonkey);
if largestDifferenceMonkey > 1e-12 || ~sameRanksMonkey
    error('things_reanalysis:SummedSimilarity', 'The rebuilt monkey summed similarity does not equal the stored values.');
end
[dissimilarityRangeMonkey, kernelWeightRangeMonkey] = dissimilarity_and_kernel_weight_range( ...
    concInfoMonkey.thingsSimRSM, kernelCentreDissimilarity, kernelSD);
fprintf('Monkey space: dissimilarity between two concepts %.3f to %.3f; kernel weight relative to a dissimilarity of 0: %.2f to %.2f\n', ...
    dissimilarityRangeMonkey(1), dissimilarityRangeMonkey(2), kernelWeightRangeMonkey(1), kernelWeightRangeMonkey(2));
if ~isequaln(concInfoMonkey.memorabilityCat, concInfo.memorabilityCat)
    error('things_reanalysis:Memorability', 'The two concept tables carry different memorability scores.');
end

% Per-feature Spearman correlations with memorability
featureOrigMemCorrsMonkey = zeros(nFeaturesMonkey, 1);
for i = 1:nFeaturesMonkey
    rho = corr(concInfoMonkey.features(:,i), concInfoMonkey.memorabilityCat, ...
        'Type', 'Spearman', 'rows', 'complete');
    featureOrigMemCorrsMonkey(i) = rho;
end

% Per-feature Spearman correlations with summed similarity (one call, 1 x 49, as in step 5)
[featureCorrsSimMonkey, ~] = corr(concInfoMonkey.weightedSimByConc, concInfoMonkey.features, ...
    'type', 'Spearman', 'rows', 'complete');

% Colors ordered by the monkey memorability correlation
[~, sortIdxMonkey] = sort(featureOrigMemCorrsMonkey, 'ascend');
cmapMonkey = piyg_colormap(nFeaturesMonkey);
featureColorsMonkey = zeros(nFeaturesMonkey, 3);
featureColorsMonkey(sortIdxMonkey,:) = cmapMonkey;

%% 11. Fig. 5d: summed similarity vs memorability, monkey IT space
fprintf('\n=== Fig. 5d, monkey IT space: summed similarity vs memorability ===\n');
figureHandle5d = figure('Visible', 'off');
x_monkey = featureCorrsSimMonkey(:)';
y_monkey = featureOrigMemCorrsMonkey(:)';
scatter(x_monkey, y_monkey, 50, featureColorsMonkey, 'filled');
xlabel({'Spearman \rho with summed similarity', '(monkey IT)'});
ylabel('Spearman \rho with memorability');
xlim([-0.6 0.6]);
ylim([-0.4 0.4]);
xticks([-0.6 0 0.6]);
yticks([-0.4 0 0.4]);
set(gca, 'FontSize', 20);

validIdxM = ~isnan(x_monkey) & ~isnan(y_monkey);
x_valsM = x_monkey(validIdxM);
y_valsM = y_monkey(validIdxM);
rM = corr(x_valsM', y_valsM');
lineHandleMonkey = lsline;
set(lineHandleMonkey, 'LineWidth', 3, 'Color', [0.75 0.75 0.75]);
lineCoefficientsMonkey = polyfit(x_valsM, y_valsM, 1);
panelTextMonkey = format_r(rM);
put_text_top_left(panelTextMonkey);
fprintf('Monkey space: r = %.5f (Pearson across %d components)\n', rM, numel(x_valsM));

componentLabels = arrayfun(@(k) sprintf('IT component %d', k), (1:nFeaturesMonkey)', 'UniformOutput', false);
monkeyFeatureTable = table((1:nFeaturesMonkey)', string(componentLabels), ...
    featureCorrsSimMonkey(:), featureOrigMemCorrsMonkey(:), ...
    rank_from_highest(featureOrigMemCorrsMonkey), rank_from_lowest(featureCorrsSimMonkey(:)), ...
    featureColorsMonkey(:, 1), featureColorsMonkey(:, 2), featureColorsMonkey(:, 3), ...
    'VariableNames', {'feature_number', 'feature_label', 'spearman_rho_summed_similarity', ...
    'spearman_rho_memorability', 'memorability_rank_from_highest', 'summed_similarity_rank_from_most_negative', ...
    'color_red', 'color_green', 'color_blue'});
save_panel_quietly(figureHandle5d, 'fig5d_monkey_features_similarity_vs_memorability', paths, monkeyFeatureTable);

%% 12. Check against the saved run in data/paper_run/ (the run the paper reports)
paperRun = load(fullfile(paths.paperRun, 'paperStats_kramer.mat'));
paperStats = paperRun.paperStats;
differences = [r - paperStats.r_human, rM - paperStats.r_monkey];
fprintf('\nAgainst the saved run (data/paper_run/paperStats_kramer.mat):\n');
fprintf('  human  r = %.10f (saved run %.10f)\n', r, paperStats.r_human);
fprintf('  monkey r = %.10f (saved run %.10f)\n', rM, paperStats.r_monkey);
if any(abs(differences) > 1e-10)
    error('things_reanalysis:PaperRun', 'The correlations are not equal to the saved run in data/paper_run/.');
end
fprintf('  equal to the saved run (largest difference in r %.2g)\n', max(abs(differences)));

%% 13. Save the statistics table and return every number numbers.csv needs
statisticsTable = table(["human"; "monkey"], [numel(x_vals); numel(x_valsM)], ...
    [numberOfConceptsWithScore; sum(~isnan(concInfoMonkey.memorabilityCat))], ...
    [numberOfConcepts; size(concInfoMonkey.features, 1)], ...
    [r; rM], ...
    [lineCoefficientsHuman(1); lineCoefficientsMonkey(1)], [lineCoefficientsHuman(2); lineCoefficientsMonkey(2)], ...
    [paperStats.r_human; paperStats.r_monkey], ...
    [string(panelTextHuman); string(panelTextMonkey)], ...
    'VariableNames', {'space', 'number_of_features', 'concepts_with_memorability', 'concepts_with_summed_similarity', ...
    'pearson_r', 'line_slope', 'line_intercept', 'paper_run_r', 'panel_text'});
writetable(statisticsTable, fullfile(paths.outputs, 'fig5_statistics.csv'));
fprintf('Saved outputs/fig5_statistics.csv\n');

fig5 = struct();
fig5.numberOfFeatures = nFeatures;
fig5.numberOfConcepts = numberOfConcepts;
fig5.numberOfConceptsWithScore = numberOfConceptsWithScore;
fig5.conceptWithoutScore = strjoin(conceptWithoutScore, ', ');
fig5.largestDifferenceMemorability = largestDifferenceMemorability;
fig5.similarityMatrixSizeHuman = size(concInfo.thingsSimRSM);
fig5.similarityRangeHuman = [min(concInfo.thingsSimRSM(:)), max(concInfo.thingsSimRSM(:))];
fig5.similarityMatrixSizeMonkey = size(concInfoMonkey.thingsSimRSM);
fig5.similarityRangeMonkey = [min(concInfoMonkey.thingsSimRSM(:)), max(concInfoMonkey.thingsSimRSM(:))];
fig5.featuresSizeMonkey = size(concInfoMonkey.features);
fig5.kernelCentreDissimilarity = kernelCentreDissimilarity;
fig5.kernelSD = kernelSD;
fig5.dissimilarityRangeHuman = dissimilarityRangeHuman;
fig5.dissimilarityRangeMonkey = dissimilarityRangeMonkey;
fig5.kernelWeightRangeHuman = kernelWeightRangeHuman;
fig5.kernelWeightRangeMonkey = kernelWeightRangeMonkey;
fig5.largestPossibleDissimilarity = largestPossibleDissimilarity;
fig5.kernelWeightAtLargestPossibleDissimilarity = kernelWeightAtLargestPossibleDissimilarity;
fig5.totalImageCount = totalImageCount;
fig5.largestDifferenceHuman = largestDifferenceHuman;
fig5.largestDifferenceMonkey = largestDifferenceMonkey;
fig5.summedSimilarityRangeHuman = [min(concInfo.weightedSimByConc), max(concInfo.weightedSimByConc)];
fig5.summedSimilarityMeanHuman = mean(concInfo.weightedSimByConc);
fig5.summedSimilarityRangeMonkey = [min(concInfoMonkey.weightedSimByConc), max(concInfoMonkey.weightedSimByConc)];
fig5.colorLimits = fixedColorLimits;
fig5.markerSizeRange = [min(markerSizes), max(markerSizes)];
fig5.exampleConcepts = exampleConcepts;
fig5.exampleSummedSimilarity = exampleSummedSimilarity;
fig5.examplePercentLower = examplePercentLower;
fig5.examplePercentHigher = examplePercentHigher;
fig5.numberOfPCs = nPC;
fig5.featureLabels = concInfo.featureLabels(:);
fig5.memorabilityCorrelationsHuman = featureOriginalMemorabilityCorrs;
fig5.summedSimilarityCorrelationsHuman = featureCorrsSimOriginal(:);
fig5.memorabilityCorrelationsMonkey = featureOrigMemCorrsMonkey;
fig5.summedSimilarityCorrelationsMonkey = featureCorrsSimMonkey(:);
fig5.rankedLabels = sortedLabels(:);
fig5.rankedMemorabilityCorrelations = sortedMemCorrs(:);
fig5.tickLabels = sortedLabels(1:5:end);
fig5.r_human = r;
fig5.r_monkey = rM;
fig5.panelTextHuman = panelTextHuman;
fig5.panelTextMonkey = panelTextMonkey;
end


%% Local functions

function save_panel_quietly(figureHandle, panelName, paths, panelData)
% Saves one panel with shared/matlab/save_panel.m (a 300 dpi PNG in outputs/figures/ and the plotted
% numbers as a CSV in outputs/figure_data/) and prints the two file names relative to the module
% folder.
evalc('save_panel(figureHandle, panelName, paths, panelData)');
fprintf('Saved outputs/figures/%s.png and outputs/figure_data/%s.csv\n', panelName, panelName);
end

function summedSimilarity = summed_similarity_full_things(similarityMatrix, imageCounts, kernelCentre, kernelSD)
% Summed similarity of every concept to all other THINGS concepts (Methods, Representational
% Similarity Analysis):
%   for concept i: the sum over every other concept j of
%       normpdf(similarity(i, j), kernelCentre, kernelSD) x imageCount(j) / (sum of all image counts)
numberOfConcepts = size(similarityMatrix, 1);
weights = imageCounts / sum(imageCounts);
summedSimilarity = zeros(numberOfConcepts, 1);
for conceptIndex = 1:numberOfConcepts
    otherConcepts = true(numberOfConcepts, 1);
    otherConcepts(conceptIndex) = false;
    kernelSimilarity = normpdf(similarityMatrix(conceptIndex, otherConcepts), kernelCentre, kernelSD);
    summedSimilarity(conceptIndex) = kernelSimilarity * weights(otherConcepts);
end
end

function put_text_top_left(panelText)
% The r text in the top-left corner of the current axes, 5 percent in from the left and the
% top edge (the point 0.05, 0.95 of the axes in normalized units, given in data units because MATLAB
% R2023b run with -batch misplaces normalized-unit text when it exports a figure)
xLimits = xlim;
yLimits = ylim;
text(xLimits(1) + 0.05 * (xLimits(2) - xLimits(1)), yLimits(1) + 0.95 * (yLimits(2) - yLimits(1)), panelText, ...
    'VerticalAlignment', 'top', 'FontSize', 20, 'BackgroundColor', 'white');
end

function panelText = format_r(r)
% The text printed in Figs. 5c and 5d: r with three decimals
panelText = sprintf('r = %.3f', r);
end

function [dissimilarityRange, kernelWeightRange] = dissimilarity_and_kernel_weight_range(similarityMatrix, kernelCentre, kernelSD)
% The smallest and largest dissimilarity (1 - r) between two different concepts, and the kernel
% weights of the summed similarity at those dissimilarities relative to the weight at a dissimilarity
% of 0 (kernelWeightRange: the weight at the largest dissimilarity, then at the smallest)
differentConcepts = ~eye(size(similarityMatrix, 1));
dissimilarities = 1 - similarityMatrix(differentConcepts);
dissimilarityRange = [min(dissimilarities), max(dissimilarities)];
kernelWeightRange = normpdf(fliplr(dissimilarityRange), kernelCentre, kernelSD) / ...
    normpdf(kernelCentre, kernelCentre, kernelSD);
end

function ranks = rank_from_highest(values)
% 1 = the largest value
[~, order] = sort(values(:), 'descend');
ranks = zeros(numel(values), 1);
ranks(order) = (1:numel(values))';
end

function ranks = rank_from_lowest(values)
% 1 = the smallest (most negative) value
[~, order] = sort(values(:), 'ascend');
ranks = zeros(numel(values), 1);
ranks(order) = (1:numel(values))';
end
