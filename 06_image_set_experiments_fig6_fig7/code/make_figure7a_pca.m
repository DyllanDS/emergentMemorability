function fig7a = make_figure7a_pca()
%MAKE_FIGURE7A_PCA  Fig. 7a: the reduced image set in the first three principal components of the
%   49-dimensional SPoSE feature space.
%
%   What it does
%     Projects the 1,854 THINGS concepts onto the first three principal components of the feature
%     space (the same projection and camera angles as Fig. 6a, 6b) and draws every concept, sized
%     by its share of the reduced set's slots (concepts not in the set at the smallest size) and
%     colored by its summed similarity to the reduced set (summed_similarity_to_set.m: kernel
%     normpdf(similarity, 1, 2) to every other concept, weighted by that concept's share of the
%     set's slots, the concept itself left out). The color scale is the one of Fig. 6a, 6b.
%
%   Inputs:  data/sampled_image_sets.mat; shared/things/concept_info_human.mat (load_concept_info.m)
%   Outputs: outputs/figures/fig7a_pca_reduced_set.png (300 dpi) and
%            outputs/figure_data/fig7a_pca_reduced_set.csv (the plotted numbers, one row per concept)
%   Returns a struct with the number of principal components, of features and of concepts.
%   No random numbers are used.

paths = release_paths(mfilename('fullpath'));

concInfo = load_concept_info('human');
sampledSets = load(fullfile(paths.data, 'sampled_image_sets.mat'));
equatedDataset = sampledSets.equatedDataset;   % the reduced set

stdForWeightedSim = 2;
similarityKernel = normpdf(concInfo.thingsSimRSM, 1, stdForWeightedSim);
eqW = equatedDataset.numImagesForConcept(:);
equatedSimToSet = summed_similarity_to_set(similarityKernel, eqW);

nPC = 3;
[pcaFeatures, ~] = pcaCoeffsExplained(concInfo.features', nPC);
equatedWeights = eqW / max(sum(eqW), eps);
equatedWeightMarkerSizes = 20 + (equatedWeights / max(equatedWeights)) * 200;

figureHandle = figure('Visible', 'off', 'Color', 'w');
colormap(flipud(parula(256)));
scatter3(pcaFeatures(:, 1), pcaFeatures(:, 2), pcaFeatures(:, 3), equatedWeightMarkerSizes, ...
    equatedSimToSet, 'filled', 'MarkerFaceAlpha', 0.3);
fontSizePca = 15;
title('a   Reduced similarity variance', 'FontSize', fontSizePca);
xlabel('PC1', 'FontSize', fontSizePca);
ylabel('PC2', 'FontSize', fontSizePca);
zlabel('PC3', 'FontSize', fontSizePca);
fixedColorLimitsPCA = [0.174, 0.185];
colorBar = colorbar;
colorBar.Limits = fixedColorLimitsPCA;
caxis(fixedColorLimitsPCA);
ylabel(colorBar, 'Summed similarity');
stylePca3dAxes(gca, colorBar, fontSizePca);
colorBar.Position(1) = colorBar.Position(1) - 0.08;   % room for the color bar label inside the figure
viewSpecEq = readPca3dFeatureSpaceView('', 'equated');
applyAxesPca3dView(gca, viewSpecEq);

numberOfConcepts = numel(eqW);
panelData = table((1:numberOfConcepts)', string(concInfo.concepts(:)), pcaFeatures(:, 1), pcaFeatures(:, 2), ...
    pcaFeatures(:, 3), eqW, equatedWeightMarkerSizes, equatedSimToSet, ...
    'VariableNames', {'concept_number', 'concept', 'pc1', 'pc2', 'pc3', 'slots_in_set', 'marker_size', ...
    'summed_similarity_to_set'});
save_panel(figureHandle, 'fig7a_pca_reduced_set', paths, panelData);
close(figureHandle);

fig7a = struct();
fig7a.nPrincipalComponents = size(pcaFeatures, 2);
fig7a.nFeatures = size(concInfo.features, 2);
fig7a.conceptsDrawn = numberOfConcepts;
fig7a.conceptsInSet = sum(eqW > 0);
fprintf('Fig. 7a: all %d concepts (%d in the reduced set) in the first %d principal components of the %d features\n', ...
    fig7a.conceptsDrawn, fig7a.conceptsInSet, fig7a.nPrincipalComponents, fig7a.nFeatures);
end
