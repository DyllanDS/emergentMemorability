function fig6 = maintained_vs_changed()
%MAINTAINED_VS_CHANGED  Fig. 6: memorability in the maintained and the changed image set.
%
%   What it does
%     Two 1,600-slot image sets were sampled from THINGS (Methods, Image Set Structure
%     Manipulations): the maintained set keeps the similarity structure of the full database, the
%     changed set partly inverts it. 150 participants per set studied and recognized images from
%     their set (after the exclusions in the Methods). For each set this function computes
%       - each concept's corrected recognition, CR = hit rate minus false alarm rate, pooled over
%         all kept test trials of the concept ("old" = responses 1 to 3 on the 6-point scale);
%       - the set's feature-memorability profile: for each of the 49 SPoSE features, the
%         Spearman correlation across concepts between the feature and CR;
%       - each concept's summed similarity within the set: the kernel similarity
%         normpdf(similarity, 1, 2) to every concept, weighted by that concept's share of the set's
%         slots (one slot of the concept itself removed, weights renormalized), and for each
%         feature the Spearman correlation across concepts between the feature and summed
%         similarity; once in the human SPoSE space, once in the monkey IT space (where the 49
%         "features" are the first 49 principal components of the IT responses).
%     Fig. 6c to 6h are Pearson correlations between two such 49-feature profiles:
%       6c  full-THINGS memorability profile (Kramer et al. concept scores) vs maintained profile
%       6d  human summed-similarity profile vs maintained profile
%       6e  monkey summed-similarity profile vs maintained profile (monkey features)
%       6f  full-THINGS memorability profile vs changed profile
%       6g  human summed-similarity profile vs changed profile
%       6h  monkey summed-similarity profile vs changed profile (monkey features)
%     Fig. 6a and 6b show all 1,854 concepts in the first three principal components of the
%     49-feature space, sized by their share of the set's slots and colored by their summed
%     similarity to the set (summed_similarity_to_set.m).
%
%   Inputs
%     data/maintained_trials.csv, data/changed_trials.csv (through shared/matlab/load_image_set_trials.m)
%     data/sampled_image_sets.mat (the slot counts of each concept in each set)
%     shared/things/concept_info_human.mat and concept_info_monkey.mat (through load_concept_info.m)
%     data/paper_run/paperStats_maintained_vs_changed.mat (the saved run, for the comparison)
%
%   Outputs (outputs/)
%     paperStats_maintained_vs_changed.mat      the six correlations and the participant counts, with
%                                           the field names of the saved run in data/paper_run/
%     fig6_correlations.csv                 the six correlations, next to the saved run in
%                                           data/paper_run/
%     concept_scores_maintained_changed.csv per concept and set: slots, hit rate, false alarm rate,
%                                           CR, summed similarity (human and monkey space) and the
%                                           Kramer et al. concept memorability
%     figures/fig6a ... fig6h.png (300 dpi) and figure_data/fig6a ... fig6h.csv (the plotted numbers)
%   Returns a struct with the six correlations and the participant counts. The interval and the p
%   value of each correlation come from the participant bootstrap of stats_fig6_fig7.py.
%   No random numbers are used.

paths = release_paths(mfilename('fullpath'));
close all;

%% Part 1: concept information (SPoSE features, Kramer et al. concept memorability, similarity)
concInfo = load_concept_info('human');
stdForWeightedSim = 2;   % SD of the similarity kernel normpdf(similarity, 1, 2)

%% Part 2: trials of the maintained and changed sets
% maintained = representativeImages (dataset_index 1), changed = flipMemorabilityImages (dataset_index 0)
datasetNames = {'representativeImages', 'flipMemorabilityImages'};
datasetIndices = [1, 0];
% (the per-trial summed similarity is not used here, so the loader skips it: third argument false)
T = load_image_set_trials({'maintained', 'changed'}, concInfo, false);

%% Part 3: participant exclusions, trial filter and the 150 cap (the exclusion criteria of the Methods)
[T, exclusionReport] = apply_participant_exclusions(T);
badSubjects = exclusionReport.badSubjects;
for d = 1:length(datasetNames)
    nSub = length(unique(T.subject(T.dataset_name == string(datasetNames{d}))));
    fprintf('Final %s: %d subjects\n', datasetNames{d}, nSub);
end

%% Part 4: per-concept memorability (CR) in each set
nDatsets = length(datasetNames);
datasets = datasetIndices;

tByConcept = cell(nDatsets, 1);
for dataIdx = 1:nDatsets
    datasetTrials = T(T.dataset_index == datasets(dataIdx), :);
    for i = 1:length(concInfo.concepts)
        conceptTrials = datasetTrials(strcmp(datasetTrials.concept, concInfo.concepts{i}), :);
        oldTrials = conceptTrials(conceptTrials.is_new == 0, :);
        newTrials = conceptTrials(conceptTrials.is_new == 1, :);

        tByConcept{dataIdx}(i).concept = concInfo.concepts{i};
        if ~isempty(oldTrials) && ~isempty(newTrials)
            tByConcept{dataIdx}(i).hit_rate = mean(oldTrials.response <= 3);
            tByConcept{dataIdx}(i).fa_rate = mean(newTrials.response <= 3);
            tByConcept{dataIdx}(i).cr = tByConcept{dataIdx}(i).hit_rate - tByConcept{dataIdx}(i).fa_rate;
        else
            tByConcept{dataIdx}(i).hit_rate = NaN;
            tByConcept{dataIdx}(i).fa_rate = NaN;
            tByConcept{dataIdx}(i).cr = NaN;
        end
    end
end

%% Part 5: feature-memorability profiles (Spearman across concepts, one value per feature)
nFeatures = size(concInfo.features, 2);
crThisDataset = zeros(nDatsets, length(concInfo.concepts));
featureMemorabilityCorrs = zeros(nDatsets, nFeatures);

for d = 1:nDatsets
    crThisDataset(d, :) = [tByConcept{d}.cr]';
    rho = corr(concInfo.features, crThisDataset(d, :)', 'Type', 'Spearman', 'rows', 'complete');
    featureMemorabilityCorrs(d, :) = rho;
end

% The full THINGS profile (Kramer et al. concept memorability; goalpost has no score)
featureOriginalMemorabilityCorrs = zeros(nFeatures, 1);
for i = 1:nFeatures
    rho = corr(concInfo.features(:, i), concInfo.memorabilityCat, 'Type', 'Spearman', 'rows', 'complete');
    featureOriginalMemorabilityCorrs(i) = rho;
end

% Feature colors: the PiYG map in the order of the full THINGS profile (pink = poor memory)
[~, sortIdxOrig] = sort(featureOriginalMemorabilityCorrs, 'ascend');
cmap = piyg_colormap(nFeatures);
featureColors = zeros(nFeatures, 3);
featureColors(sortIdxOrig, :) = cmap;

%% Part 6: summed similarity within each set (human SPoSE space) and its feature profile
sampledSets = load(fullfile(paths.data, 'sampled_image_sets.mat'));
datasetInfo = {sampledSets.representativeDataset, sampledSets.flipMemorabilityDataset};

featureCorrsSim = zeros(nDatsets, nFeatures);
for d = 1:nDatsets
    for i = 1:length(concInfo.concepts)
        datasetInfo{d}.weightedSim(i, 1) = nan;
        if datasetInfo{d}.numImagesForConcept(i) > 0
            counts = datasetInfo{d}.numImagesForConcept;
            counts(i) = counts(i) - 1;
            weights = counts / sum(counts);
            weightedSim = sum(normpdf(concInfo.thingsSimRSM(i, :), 1, stdForWeightedSim) * weights);
            datasetInfo{d}.weightedSim(i, 1) = weightedSim;
        end
    end
    featureCorrsSim(d, :) = corr(datasetInfo{d}.weightedSim, concInfo.features, 'type', 'Spearman', 'rows', 'complete');
end

%% Part 6b: the same in the monkey IT space (similarity of the 49 IT components)
concInfoMonkey = load_concept_info('monkey');

featureCorrsSimMonkey = zeros(nDatsets, nFeatures);
monkeyWeightedSimBySet = nan(length(concInfoMonkey.concepts), nDatsets);
for d = 1:nDatsets
    monkeyWeightedSim = nan(length(concInfoMonkey.concepts), 1);
    for i = 1:length(concInfoMonkey.concepts)
        if datasetInfo{d}.numImagesForConcept(i) > 0
            counts = datasetInfo{d}.numImagesForConcept;
            counts(i) = counts(i) - 1;
            weights = counts / sum(counts);
            monkeyWeightedSim(i) = sum(normpdf(concInfoMonkey.thingsSimRSM(i, :), 1, stdForWeightedSim) * weights);
        end
    end
    monkeyWeightedSimBySet(:, d) = monkeyWeightedSim;
    featureCorrsSimMonkey(d, :) = corr(monkeyWeightedSim, concInfoMonkey.features, 'type', 'Spearman', 'rows', 'complete');
end

% Memorability profile over the monkey components (the same CR)
featureMemorabilityCorrsMonkey = zeros(nDatsets, nFeatures);
for d = 1:nDatsets
    rho = corr(concInfoMonkey.features, crThisDataset(d, :)', 'Type', 'Spearman', 'rows', 'complete');
    featureMemorabilityCorrsMonkey(d, :) = rho;
end

% Colors of the 49 IT components: the PiYG map in the order of each component's correlation with
% full THINGS memorability, as in Fig. 5d (pink = poor memory)
monkeyOriginalMemorabilityCorrs = zeros(nFeatures, 1);
for i = 1:nFeatures
    monkeyOriginalMemorabilityCorrs(i) = corr(concInfoMonkey.features(:, i), concInfoMonkey.memorabilityCat, ...
        'Type', 'Spearman', 'rows', 'complete');
end
[~, sortIdxMonkey] = sort(monkeyOriginalMemorabilityCorrs, 'ascend');
monkeyColors = zeros(nFeatures, 3);
monkeyColors(sortIdxMonkey, :) = piyg_colormap(nFeatures);

%% Part 7: Fig. 6c to 6h, Pearson correlations between two 49-feature profiles
repIdx = 1;   % maintained set
flipIdx = 2;  % changed set
humanLabels = cellstr(string(concInfo.featureLabels(:)));
monkeyLabels = cellstr(compose("IT component %d", (1:nFeatures)'));
profileLabel = 'Spearman correlation with memorability';

% Fig. 6c: full THINGS profile (x) vs maintained profile (y)
x_data = featureOriginalMemorabilityCorrs(:)';
y_data = featureMemorabilityCorrs(repIdx, :);
validIdx = ~isnan(x_data) & ~isnan(y_data);
x_vals = x_data(validIdx);
y_vals = y_data(validIdx);
r = corr(x_vals', y_vals');
fprintf('Original (Kramer) vs Maintain feature-mem corrs: r = %.5f\n', r);
r_fig6c = r;
drawFeatureScatter(paths, 'fig6c_things_vs_maintained_profile', 'c', x_vals, y_vals, featureColors(validIdx, :), ...
    humanLabels(validIdx), [-0.4 0.4], [-0.5 0.5], 'Correlation with memorability, full THINGS', ...
    {profileLabel, '(maintained set)'}, r);

% Fig. 6d: human summed-similarity profile (x) vs maintained profile (y)
x_data = featureCorrsSim(repIdx, :);
y_data = featureMemorabilityCorrs(repIdx, :);
validIdx = ~isnan(x_data) & ~isnan(y_data);
x_vals = x_data(validIdx);
y_vals = y_data(validIdx);
r = corr(x_vals', y_vals');
fprintf('Summed sim vs Maintain feature-mem corrs: r = %.5f\n', r);
r_fig6d = r;
drawFeatureScatter(paths, 'fig6d_human_summed_similarity_vs_maintained_profile', 'd', x_vals, y_vals, ...
    featureColors(validIdx, :), humanLabels(validIdx), [-1 1], [-0.5 0.5], ...
    'Correlation with summed similarity (human)', {profileLabel, '(maintained set)'}, r);

% Fig. 6f: full THINGS profile (x) vs changed profile (y)
x_data = featureOriginalMemorabilityCorrs(:)';
y_data = featureMemorabilityCorrs(flipIdx, :);
validIdx = ~isnan(x_data) & ~isnan(y_data);
x_vals = x_data(validIdx);
y_vals = y_data(validIdx);
r = corr(x_vals', y_vals');
fprintf('Original (Kramer) vs Changed feature-mem corrs: r = %.5f\n', r);
r_fig6f = r;
drawFeatureScatter(paths, 'fig6f_things_vs_changed_profile', 'f', x_vals, y_vals, featureColors(validIdx, :), ...
    humanLabels(validIdx), [-0.4 0.4], [-0.5 0.5], 'Correlation with memorability, full THINGS', ...
    {profileLabel, '(changed set)'}, r);

% Fig. 6g: human summed-similarity profile of the changed set (x) vs changed profile (y)
x_data = featureCorrsSim(flipIdx, :);
y_data = featureMemorabilityCorrs(flipIdx, :);
validIdx = ~isnan(x_data) & ~isnan(y_data);
x_vals = x_data(validIdx);
y_vals = y_data(validIdx);
r = corr(x_vals', y_vals');
fprintf('Summed sim vs Changed feature-mem corrs: r = %.5f\n', r);
r_fig6g = r;
drawFeatureScatter(paths, 'fig6g_human_summed_similarity_vs_changed_profile', 'g', x_vals, y_vals, ...
    featureColors(validIdx, :), humanLabels(validIdx), [-1 1], [-0.5 0.5], ...
    'Correlation with summed similarity (human)', {profileLabel, '(changed set)'}, r);

% Fig. 6e: monkey summed-similarity profile (x) vs maintained profile over the monkey components (y)
% (the points are the 49 IT components, colored as in Fig. 5d)
x_data = featureCorrsSimMonkey(repIdx, :);
y_data = featureMemorabilityCorrsMonkey(repIdx, :);
validIdx = ~isnan(x_data) & ~isnan(y_data);
x_vals = x_data(validIdx);
y_vals = y_data(validIdx);
r = corr(x_vals', y_vals');
fprintf('Monkey summed sim vs Maintain feature-mem corrs: r = %.5f\n', r);
r_fig6e = r;
drawFeatureScatter(paths, 'fig6e_monkey_summed_similarity_vs_maintained_profile', 'e', x_vals, y_vals, ...
    monkeyColors(validIdx, :), monkeyLabels(validIdx), [-1 1], [-0.5 0.5], ...
    'Correlation with summed similarity (monkey)', {profileLabel, '(maintained set)'}, r);

% Fig. 6h: monkey summed-similarity profile of the changed set (x) vs changed profile (y)
x_data = featureCorrsSimMonkey(flipIdx, :);
y_data = featureMemorabilityCorrsMonkey(flipIdx, :);
validIdx = ~isnan(x_data) & ~isnan(y_data);
x_vals = x_data(validIdx);
y_vals = y_data(validIdx);
r = corr(x_vals', y_vals');
fprintf('Monkey summed sim vs Changed feature-mem corrs: r = %.5f\n', r);
r_fig6h = r;
drawFeatureScatter(paths, 'fig6h_monkey_summed_similarity_vs_changed_profile', 'h', x_vals, y_vals, ...
    monkeyColors(validIdx, :), monkeyLabels(validIdx), [-1 1], [-0.5 0.5], ...
    'Correlation with summed similarity (monkey)', {profileLabel, '(changed set)'}, r);

%% Save the Fig. 6 statistics and compare them with the saved run in data/paper_run/
paperStats = struct();
paperStats.r_origVsMaintain = r_fig6c;
paperStats.r_sumSimVsMaintain = r_fig6d;
paperStats.r_origVsChanged = r_fig6f;
paperStats.r_sumSimVsChanged = r_fig6g;
paperStats.r_monkeyVsMaintain = r_fig6e;
paperStats.r_monkeyVsChanged = r_fig6h;

nMaintainFinal = length(unique(T.subject(T.dataset_name == string(datasetNames{1}))));
nChangedFinal = length(unique(T.subject(T.dataset_name == string(datasetNames{2}))));
paperStats.nParticipantsMaintain = nMaintainFinal;
paperStats.nParticipantsChanged = nChangedFinal;
paperStats.nExcluded = length(badSubjects);
save(fullfile(paths.outputs, 'paperStats_maintained_vs_changed.mat'), 'paperStats');

paperRun = load(fullfile(paths.paperRun, 'paperStats_maintained_vs_changed.mat'));
panelLetters = {'c'; 'd'; 'e'; 'f'; 'g'; 'h'};
statNames = {'origVsMaintain'; 'sumSimVsMaintain'; 'monkeyVsMaintain'; 'origVsChanged'; 'sumSimVsChanged'; 'monkeyVsChanged'};
xProfiles = {'full THINGS memorability'; 'summed similarity, human, maintained set'; ...
    'summed similarity, monkey, maintained set'; 'full THINGS memorability'; ...
    'summed similarity, human, changed set'; 'summed similarity, monkey, changed set'};
yProfiles = {'memorability, maintained set'; 'memorability, maintained set'; ...
    'memorability over the monkey components, maintained set'; 'memorability, changed set'; ...
    'memorability, changed set'; 'memorability over the monkey components, changed set'};
numberOfPanels = numel(panelLetters);
pearsonR = zeros(numberOfPanels, 1);
paperRunR = zeros(numberOfPanels, 1);
for panelIndex = 1:numberOfPanels
    pearsonR(panelIndex) = paperStats.(['r_' statNames{panelIndex}]);
    paperRunR(panelIndex) = paperRun.paperStats.(['r_' statNames{panelIndex}]);
    fprintf('Fig. 6%s: r = %.10f (saved run %.10f, difference %.2g)\n', panelLetters{panelIndex}, ...
        pearsonR(panelIndex), paperRunR(panelIndex), pearsonR(panelIndex) - paperRunR(panelIndex));
end
numberOfFeatures = repmat(nFeatures, numberOfPanels, 1);
correlationTable = table(panelLetters, statNames, xProfiles, yProfiles, numberOfFeatures, pearsonR, ...
    paperRunR, pearsonR - paperRunR, ...
    'VariableNames', {'panel', 'paper_stats_name', 'x_profile', 'y_profile', 'n_features', 'pearson_r', ...
    'paper_run_r', 'difference_to_paper_run'});
writetable(correlationTable, fullfile(paths.outputs, 'fig6_correlations.csv'));
fprintf('participants kept: maintained %d, changed %d; excluded in the two sets: %d (saved run %d)\n', ...
    nMaintainFinal, nChangedFinal, paperStats.nExcluded, paperRun.paperStats.nExcluded);

% per-concept scores of both sets (the numbers behind the profiles)
conceptRows = cell(0, 10);
setLabels = {'maintained', 'changed'};
for d = 1:nDatsets
    for i = 1:length(concInfo.concepts)
        if datasetInfo{d}.numImagesForConcept(i) > 0
            conceptRows(end + 1, :) = {setLabels{d}, i, concInfo.concepts{i}, ...
                datasetInfo{d}.numImagesForConcept(i), tByConcept{d}(i).hit_rate, tByConcept{d}(i).fa_rate, ...
                tByConcept{d}(i).cr, datasetInfo{d}.weightedSim(i), monkeyWeightedSimBySet(i, d), ...
                concInfo.memorabilityCat(i)}; %#ok<AGROW>
        end
    end
end
conceptTable = cell2table(conceptRows, 'VariableNames', {'image_set', 'concept_number', 'concept', 'slots_in_set', ...
    'hit_rate', 'fa_rate', 'cr', 'summed_similarity_human', 'summed_similarity_monkey', 'kramer_concept_cr'});
writetable(conceptTable, fullfile(paths.outputs, 'concept_scores_maintained_changed.csv'));

%% Part 8: Fig. 6a and 6b, the concepts in the first three principal components of the 49 features
% All 1,854 concepts, sized by the concept's share of the set's slots (concepts not in the set at
% the smallest size) and colored by the concept's summed similarity to the set
% (summed_similarity_to_set.m: every other concept weighted by its share of the set's slots, the
% concept itself left out).
nPC = 3;
[pcaFeatures, ~] = pcaCoeffsExplained(concInfo.features', nPC);
fixedColorLimitsPCA = [0.174, 0.185];
fontSizePca = 15;

repW = datasetInfo{repIdx}.numImagesForConcept(:);
repWeights = repW / max(sum(repW), eps);
flipW = datasetInfo{flipIdx}.numImagesForConcept(:);
flipWeights = flipW / max(sum(flipW), eps);
repWeightMarkerSizes = 20 + (repWeights / max(repWeights)) * 200;
flipWeightMarkerSizes = 20 + (flipWeights / max(flipWeights)) * 200;

similarityKernel = normpdf(concInfo.thingsSimRSM, 1, stdForWeightedSim);
repSimToSet = summed_similarity_to_set(similarityKernel, repW);
flipSimToSet = summed_similarity_to_set(similarityKernel, flipW);

drawPcaPanel(paths, 'fig6a_pca_maintained_set', 'a   Maintained similarity structure', pcaFeatures, ...
    repWeightMarkerSizes, repSimToSet, repW, concInfo.concepts, fixedColorLimitsPCA, fontSizePca, 'representative');
drawPcaPanel(paths, 'fig6b_pca_changed_set', 'b   Changed similarity structure', pcaFeatures, ...
    flipWeightMarkerSizes, flipSimToSet, flipW, concInfo.concepts, fixedColorLimitsPCA, fontSizePca, 'flipped');

fig6 = struct();
fig6.panelLetters = panelLetters;
fig6.r = pearsonR;
fig6.paperRunR = paperRunR;
fig6.nFeatures = nFeatures;
fig6.nPrincipalComponents = nPC;
fig6.numberOfSetsCompared = nDatsets;
fig6.nParticipants = [nMaintainFinal, nChangedFinal];
fig6.nExcluded = paperStats.nExcluded;
fig6.conceptsPresent = [sum(repW > 0), sum(flipW > 0)];
fig6.conceptsDrawn = numel(repW);
end


function drawFeatureScatter(paths, panelName, panelLetter, x_vals, y_vals, colors, labels, xLimits, yLimits, ...
    xLabelText, yLabelText, r)
% One panel of Fig. 6c to 6h: one point per feature, the least-squares line and r in the title.
figureHandle = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 520 520]);
scatter(x_vals, y_vals, 100, colors, 'filled');
hold on;
lineHandle = lsline;
set(lineHandle, 'LineWidth', 4, 'Color', [0.7 0.7 0.7]);
xlim(xLimits);
ylim(yLimits);
xticks([xLimits(1) 0 xLimits(2)]);
yticks([yLimits(1) 0 yLimits(2)]);
xlabel(xLabelText, 'FontSize', 13);
ylabel(yLabelText, 'FontSize', 13);
title(sprintf('%s   r = %.2f', panelLetter, r), 'FontSize', 15);
set(gca, 'FontSize', 13, 'LineWidth', 1.5, 'Box', 'on');
panelData = table((1:numel(x_vals))', string(labels(:)), x_vals(:), y_vals(:), colors(:, 1), colors(:, 2), colors(:, 3), ...
    'VariableNames', {'point', 'feature', 'x', 'y', 'color_red', 'color_green', 'color_blue'});
save_panel(figureHandle, panelName, paths, panelData);
close(figureHandle);
end


function drawPcaPanel(paths, panelName, titleText, pcaFeatures, markerSizes, summedSimilarity, slotCounts, ...
    concepts, colorLimits, fontSizePca, viewName)
% One 3-D panel (Fig. 6a, 6b): all 1,854 concepts in the first three principal components, sized
% by their share of the set's slots and colored by their summed similarity to the set.
figureHandle = figure('Visible', 'off', 'Color', 'w');
colormap(flipud(parula(256)));
scatter3(pcaFeatures(:, 1), pcaFeatures(:, 2), pcaFeatures(:, 3), markerSizes, summedSimilarity, 'filled', ...
    'MarkerFaceAlpha', 0.3);
title(titleText, 'FontSize', fontSizePca);
xlabel('PC1', 'FontSize', fontSizePca);
ylabel('PC2', 'FontSize', fontSizePca);
zlabel('PC3', 'FontSize', fontSizePca);
colorBar = colorbar;
colorBar.Limits = colorLimits;
caxis(colorLimits);
ylabel(colorBar, 'Summed similarity');
stylePca3dAxes(gca, colorBar, fontSizePca);
colorBar.Position(1) = colorBar.Position(1) - 0.08;   % room for the color bar label inside the figure
viewSpec = readPca3dFeatureSpaceView('', viewName);
applyAxesPca3dView(gca, viewSpec);
numberOfConcepts = numel(slotCounts);
panelData = table((1:numberOfConcepts)', string(concepts(:)), pcaFeatures(:, 1), pcaFeatures(:, 2), ...
    pcaFeatures(:, 3), slotCounts(:), markerSizes(:), summedSimilarity(:), ...
    'VariableNames', {'concept_number', 'concept', 'pc1', 'pc2', 'pc3', 'slots_in_set', 'marker_size', ...
    'summed_similarity_to_set'});
save_panel(figureHandle, panelName, paths, panelData);
close(figureHandle);
end
