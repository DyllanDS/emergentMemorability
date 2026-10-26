function fig1_makeFigure1(figureDataSource, runLabel)
%FIG1_MAKEFIGURE1  Draws Fig. 1 of the paper (panels a to d in one 2 x 2 figure) from one run of
%   the REM simulation, and writes the numbers each panel plots.
%
%   Usage (from 01_rem_simulation_fig1/code/)
%     fig1_makeFigure1                       the saved run (data/paper_run/)
%     fig1_makeFigure1(figureData, label)    a run held in memory (run_module.m passes its rerun)
%     fig1_makeFigure1('run.mat', label)     a run saved by fig1_remSim.m
%
%   Panels (as in the published figure)
%     a  memorability (hit rate minus false alarm rate) of the 1,600 items, ranked from highest to
%        lowest
%     b  item memorability against summed similarity (for each item, the sum over the other items
%        of 1 / Euclidean distance), with the least-squares line
%     c  each feature's Spearman correlation with memorability (across the 1,600 items), ranked
%        from highest to lowest
%     d  each feature's Spearman correlation with memorability against its Spearman correlation
%        with item similarity (1 / mean Euclidean distance), with the least-squares line
%
%   Outputs (in 01_rem_simulation_fig1/outputs/)
%     figures/fig1_rem_simulation.png                              the figure, 300 dpi
%     figure_data/fig1a_item_memorability_rank_order.csv           rank, item, memorability
%     figure_data/fig1b_memorability_vs_summed_similarity.csv      item, summed similarity,
%                                                                  memorability, least-squares line
%     figure_data/fig1c_feature_correlation_rank_order.csv         rank, feature, Spearman rho with
%                                                                  memorability
%     figure_data/fig1d_feature_memorability_vs_similarity.csv     feature, Spearman rho with
%                                                                  similarity and with memorability,
%                                                                  least-squares line
%
%   The summed similarity of panel b is computed in fig1_measures.m. Axis limits: panels a and b,
%   y from the lowest memorability to 1; panel b, x rounded outward to a multiple of 50; panels c
%   and d, y rounded outward to a multiple of 0.02; panel d, x from the lowest correlation to 0.
%   The x axis of panel d (labeled "Correlation With Summed Similarity") shows each feature's
%   Spearman correlation with similarity (1 / mean distance).

%% Folders and the run to draw
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared', 'matlab'));
paths = release_paths(mfilename('fullpath'));

if nargin < 1 || isempty(figureDataSource)
    figureDataSource = fullfile(paths.paperRun, 'fig1_paper_run.mat');
    runLabel = 'saved run (data/paper_run)';
elseif nargin < 2 || isempty(runLabel)
    runLabel = 'run given as input';
end
if ischar(figureDataSource) || isstring(figureDataSource)
    loaded = load(figureDataSource, 'figureData');
    figureData = loaded.figureData;
else
    figureData = figureDataSource;
end

measures = fig1_measures(figureData);

%% Style of the published figure
pointColor = [0 0.4470 0.7410];   % blue dots (MATLAB's first default color)
trendColor = [0.5 0.5 0.5];       % gray least-squares lines
labelFontSize = 12;
letterFontSize = 16;

%% Panel a: items ranked by memorability
% Every item has a memorability score in a run of 10,000 simulated experiments; a short test run
% can leave a few items untested in one of the two roles, and those are left out of panels a and b.
itemHasScore = ~isnan(measures.memorability);
if ~all(itemHasScore)
    fprintf('%d items have no memorability score in this run and are not drawn\n', sum(~itemHasScore));
end
memorability = measures.memorability(itemHasScore);
allItemNumbers = (1:numel(measures.memorability))';
itemNumber = allItemNumbers(itemHasScore);
numberOfItems = numel(memorability);
[sortedMemorability, rankOrder] = sort(memorability, 'descend');
itemInRankOrder = itemNumber(rankOrder);
itemRank = (1:numberOfItems)';

%% Panel b: memorability against summed similarity (sum of 1 / distance)
summedSimilarity = measures.summedInverseDistance(itemHasScore);
lineCoefficientsB = polyfit(summedSimilarity, memorability, 1);
lineValuesB = polyval(lineCoefficientsB, summedSimilarity);
pearsonPlottedB = corr(summedSimilarity, memorability);

%% Panel c: features ranked by their correlation with memorability
featureCorrelationWithMemorability = figureData.featureCorrsMemory(:);
featureCorrelationWithSimilarity = figureData.featureCorrsSimil(:);
numberOfFeatures = numel(featureCorrelationWithMemorability);
[sortedFeatureCorrelation, featureInRankOrder] = sort(featureCorrelationWithMemorability, 'descend');
featureRank = (1:numberOfFeatures)';

%% Panel d: feature correlation with memorability against feature correlation with similarity
featureNumber = (1:numberOfFeatures)';
lineCoefficientsD = polyfit(featureCorrelationWithSimilarity, featureCorrelationWithMemorability, 1);
lineValuesD = polyval(lineCoefficientsD, featureCorrelationWithSimilarity);
spearmanAcrossFeatures = corr(featureCorrelationWithSimilarity, featureCorrelationWithMemorability, ...
    'Type', 'Spearman');

%% Draw the four panels
figureHandle = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [1 1 8 6.6]);
layout = tiledlayout(figureHandle, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% a
axesA = nexttile(layout, 1);
plot(axesA, itemRank, sortedMemorability, '.', 'Color', pointColor, 'MarkerSize', 5);
xlim(axesA, [0 numberOfItems]);
xticks(axesA, 0:400:numberOfItems);
ylim(axesA, [min(memorability) 1]);
xlabel(axesA, 'Item Rank Order');
ylabel(axesA, 'Item Memorability');
grid(axesA, 'on');
set(axesA, 'FontSize', labelFontSize);
addPanelLetter(axesA, 'a', letterFontSize);

% b
axesB = nexttile(layout, 2);
scatter(axesB, summedSimilarity, memorability, 10, 'filled', 'MarkerFaceAlpha', 0.7, ...
    'MarkerFaceColor', pointColor, 'MarkerEdgeColor', pointColor);
hold(axesB, 'on');
lineXB = linspace(min(summedSimilarity), max(summedSimilarity), 100);
plot(axesB, lineXB, polyval(lineCoefficientsB, lineXB), '-', 'Color', trendColor, 'LineWidth', 1.5);
hold(axesB, 'off');
ylim(axesB, [min(memorability) 1]);
xlim(axesB, roundedLimits(summedSimilarity, 50));
xlabel(axesB, 'Summed Similarity');
ylabel(axesB, 'Item Memorability');
grid(axesB, 'on');
set(axesB, 'FontSize', labelFontSize);
addPanelLetter(axesB, 'b', letterFontSize);

% c
axesC = nexttile(layout, 3);
plot(axesC, featureRank, sortedFeatureCorrelation, '.', 'Color', pointColor, 'MarkerSize', 9);
xlim(axesC, [0 numberOfFeatures]);
xticks(axesC, 0:5:numberOfFeatures);
ylim(axesC, roundedLimits(sortedFeatureCorrelation, 0.02));
xlabel(axesC, 'Feature Rank Order');
ylabel(axesC, 'Feature Corr With Memorability');
grid(axesC, 'on');
set(axesC, 'FontSize', labelFontSize);
addPanelLetter(axesC, 'c', letterFontSize);

% d
axesD = nexttile(layout, 4);
scatter(axesD, featureCorrelationWithSimilarity, featureCorrelationWithMemorability, 18, 'filled', ...
    'MarkerFaceAlpha', 0.7, 'MarkerFaceColor', pointColor, 'MarkerEdgeColor', pointColor);
hold(axesD, 'on');
lineXD = linspace(min(featureCorrelationWithSimilarity), max(featureCorrelationWithSimilarity), 200);
plot(axesD, lineXD, polyval(lineCoefficientsD, lineXD), '-', 'Color', trendColor, 'LineWidth', 1.5);
hold(axesD, 'off');
xlim(axesD, [min(featureCorrelationWithSimilarity) max(0, max(featureCorrelationWithSimilarity))]);
ylim(axesD, roundedLimits(featureCorrelationWithMemorability, 0.02));
xlabel(axesD, 'Correlation With Summed Similarity');
ylabel(axesD, 'Feature Corr With Memorability');
grid(axesD, 'on');
set(axesD, 'FontSize', labelFontSize);
addPanelLetter(axesD, 'd', letterFontSize);

%% Save the figure and the plotted numbers
figureFile = fullfile(paths.figures, 'fig1_rem_simulation.png');
exportgraphics(figureHandle, figureFile, 'Resolution', 300);
close(figureHandle);

panelA = table(itemRank, itemInRankOrder, sortedMemorability, ...
    'VariableNames', {'rank', 'item', 'memorability'});
writetable(panelA, fullfile(paths.figureData, 'fig1a_item_memorability_rank_order.csv'));

panelB = table(itemNumber, summedSimilarity, memorability, lineValuesB, ...
    'VariableNames', {'item', 'summed_similarity_sum_of_inverse_distances', 'memorability', 'least_squares_line'});
writetable(panelB, fullfile(paths.figureData, 'fig1b_memorability_vs_summed_similarity.csv'));

panelC = table(featureRank, featureInRankOrder, sortedFeatureCorrelation, ...
    'VariableNames', {'rank', 'feature', 'spearman_rho_with_memorability'});
writetable(panelC, fullfile(paths.figureData, 'fig1c_feature_correlation_rank_order.csv'));

panelD = table(featureNumber, featureCorrelationWithSimilarity, featureCorrelationWithMemorability, lineValuesD, ...
    'VariableNames', {'feature', 'spearman_rho_with_similarity', 'spearman_rho_with_memorability', 'least_squares_line'});
writetable(panelD, fullfile(paths.figureData, 'fig1d_feature_memorability_vs_similarity.csv'));

fprintf('Fig. 1 drawn from the %s:\n', runLabel);
fprintf('  outputs/figures/fig1_rem_simulation.png\n');
fprintf('  outputs/figure_data/fig1a_item_memorability_rank_order.csv (%d items)\n', numberOfItems);
fprintf('  outputs/figure_data/fig1b_memorability_vs_summed_similarity.csv (Pearson r of the plotted points %.4f)\n', pearsonPlottedB);
fprintf('  outputs/figure_data/fig1c_feature_correlation_rank_order.csv (%d features)\n', numberOfFeatures);
fprintf('  outputs/figure_data/fig1d_feature_memorability_vs_similarity.csv (Spearman rho across features %.4f)\n', spearmanAcrossFeatures);
end

function addPanelLetter(axesHandle, letter, fontSize)
% The panel letter, written as the panel's title and aligned with the left edge of the panel
title(axesHandle, letter, 'FontSize', fontSize, 'FontWeight', 'normal');
axesHandle.TitleHorizontalAlignment = 'left';
end

function limits = roundedLimits(values, step)
% Axis limits from the smallest and largest value, rounded outward to a multiple of step (as the
% axes of the published figure: 100 to 250 for panel b, 0.04 to 0.16 for panels c and d)
lowest = min(values);
highest = max(values);
limits = [floor(lowest / step) * step, ceil(highest / step) * step];
end
