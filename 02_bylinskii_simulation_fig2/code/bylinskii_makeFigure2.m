function bylinskii_makeFigure2(figureDataFile)
%BYLINSKII_MAKEFIGURE2  Draws the two panels of Fig. 2 from one saved run of bylinskiiRemSim.m and
%   writes the numbers each panel plots to a CSV file.
%
%   Input
%     figureDataFile  a saved run (default: the saved run the paper reports,
%                     data/paper_run/fig2_paper_run.mat)
%
%   Outputs (in <module>/outputs/)
%     figures/fig2a_memorability_within_vs_across.png       Fig. 2a, 300 dpi
%     figures/fig2b_summed_similarity_within_vs_across.png  Fig. 2b, 300 dpi
%     figure_data/fig2a_memorability_within_vs_across.csv   one row per target
%     figure_data/fig2b_summed_similarity_within_vs_across.csv
%
%   What the panels show (one point per target tested in both contexts, 1,575 in the full run)
%     (a) memorability (hit rate minus false alarm rate) in Simulation 1, within category (x), against
%         Simulation 2, across categories (y); gray line: least-squares fit. Color: the mean of the
%         two contexts' z-scored summed similarities, multiplied by 1.4 above zero and divided by 1.4
%         below zero, on a color scale from -2 (orange, more distinctive) through green to 4 (dark
%         blue, more similar to the other items, that is, less distinctive).
%     (b) summed similarity (minus the mean Euclidean distance to the other items of the context)
%         within category (x) against across categories (y), both shifted by the smallest of all
%         plotted values so the axes start at 0; gray line: least-squares fit; dashed line: y = x.
%   The function stops with an error unless the summed similarities of the run are negative (minus
%   the mean distance). bylinskii_numbers.m writes the parameters to numbers.csv and stats_fig2.py
%   the correlations.

if nargin < 1 || isempty(figureDataFile)
    codeFolder = fileparts(mfilename('fullpath'));
    figureDataFile = fullfile(codeFolder, '..', 'data', 'paper_run', 'fig2_paper_run.mat');
end
codeFolder = fileparts(mfilename('fullpath'));
addpath(fullfile(codeFolder, '..', '..', 'shared', 'matlab'));
paths = release_paths(mfilename('fullpath'));

globalFontSize = 22;
pointColor = [0 0.4470 0.7410];
trendColor = [0.5 0.5 0.5];
panelSizeInPixels = [760 620];   % width and height of each panel's figure window

%% Load the run
loaded = load(figureDataFile, 'figureData');
figureData = loaded.figureData;
[~, runName, runExtension] = fileparts(figureDataFile);
fprintf('Fig. 2 from %s\n', [runName runExtension]);

amt1_valid = figureData.amt1_valid;
amt2_valid = figureData.amt2_valid;
validBoth = logical(figureData.validBoth);
summedSim_amt1 = figureData.summedSim_amt1;
summedSim_amt2 = figureData.summedSim_amt2;
targetsPerCategory = figureData.parameters.targetsPerCategory;

% Target numbers and categories of the plotted points. The simulation generates the targets
% category by category (targets 1 to 75 belong to category 1, 76 to 150 to category 2, ...).
targetNumber = find(validBoth);
category = ceil(targetNumber / targetsPerCategory);

%% Panel a: memorability within category against across categories
summedSim1 = summedSim_amt1(validBoth);
summedSim2 = summedSim_amt2(validBoth);

% bylinskiiRemSim.m saves the summed similarity as minus the mean distance (higher = more similar);
% stop unless the summed similarities are negative.
if any(summedSim1 >= 0) || any(summedSim2 >= 0)
    error('bylinskii_makeFigure2:SignConvention', ...
        'Summed similarities must be negative (minus the mean distance); this run uses another convention.');
end
summedSim1_sim = summedSim1;
summedSim2_sim = summedSim2;

poolMin = min([summedSim1_sim; summedSim2_sim], [], 'omitnan');
summedSim1_plot = summedSim1_sim - poolMin;
summedSim2_plot = summedSim2_sim - poolMin;

% Z-score each context's summed similarity before averaging for color
summedSim1_z = (summedSim1_sim - mean(summedSim1_sim, 'omitnan')) ./ std(summedSim1_sim, 'omitnan');
summedSim2_z = (summedSim2_sim - mean(summedSim2_sim, 'omitnan')) ./ std(summedSim2_sim, 'omitnan');
colorData = mean([summedSim1_z, summedSim2_z], 2, 'omitnan');
stretchFactor = 1.4;
colorData = (colorData >= 0) .* (colorData * stretchFactor) + (colorData < 0) .* (colorData / stretchFactor);

figureA = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 panelSizeInPixels]);
scatter(amt1_valid, amt2_valid, 30, colorData, 'filled', ...
    'MarkerFaceAlpha', 0.8, 'MarkerEdgeAlpha', 0.8);
hold on;
remSim_lsline(amt1_valid, amt2_valid, trendColor, 2);

basePalette = jet(256);
basePalette = basePalette(1:180, :) * 0.9;
pcaPalette = max(min(basePalette, 1), 0);
pcaPalette = interp1(linspace(0, 1, size(pcaPalette, 1)), pcaPalette, linspace(0, 1, 256));
pcaPalette = flipud(pcaPalette);
colormap(gca, pcaPalette);
c = colorbar;
c.Label.String = 'Z-scored Sum Sim';
c.FontSize = globalFontSize - 4;
caxis([-2, 4]);
c.Ticks = -2:4;
c.TickLabels = arrayfun(@(v) sprintf('%.0f', v), c.Ticks, 'UniformOutput', false);

xlabel({'Memorability', 'Within Category Simulations'}, 'FontSize', globalFontSize);
ylabel({'Memorability', 'Across Category Simulations'}, 'FontSize', globalFontSize);
grid on;
axis equal;
axis([-0.1 1 -0.1 1]);   % every target shows (the lowest memorability across categories is -0.067)
set(gca, 'FontSize', globalFontSize, 'XTick', 0:0.2:1, 'YTick', 0:0.2:1);
addPanelLetter(figureA, '(a)', globalFontSize + 4);

panelA = table(targetNumber, category, amt1_valid, amt2_valid, summedSim1, summedSim2, colorData, ...
    'VariableNames', {'target_number', 'category', 'memorability_within_category', ...
    'memorability_across_categories', 'summed_similarity_within_category', ...
    'summed_similarity_across_categories', 'color_value_stretched_mean_z'});
savePanel(figureA, panelA, 'fig2a_memorability_within_vs_across', paths);

%% Panel b: summed similarity within category against across categories
figureB = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 panelSizeInPixels]);

simX = summedSim1_plot;
simY = summedSim2_plot;
scatter(simX, simY, 30, pointColor, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
remSim_lsline(simX, simY, trendColor, 2);
sharedMin = min([simX; simY]);
sharedMax = max([simX; simY]);
plot([sharedMin sharedMax], [sharedMin sharedMax], 'k--', 'LineWidth', 1.5);

simMin = min([simX; simY], [], 'omitnan');
simMax = max([simX; simY], [], 'omitnan');
if isfinite(simMin) && isfinite(simMax) && simMin < simMax
    xlim([simMin simMax]);
    ylim([simMin simMax]);
    tickVals = linspace(simMin, simMax, 5);
    tickLabels = arrayfun(@(v) sprintf('%.2f', v), tickVals, 'UniformOutput', false);
    set(gca, 'XTick', tickVals, 'XTickLabel', tickLabels, ...
        'YTick', tickVals, 'YTickLabel', tickLabels);
end

xlabel({'Summed Similarity', 'Within Category Simulations'}, 'FontSize', globalFontSize);
ylabel({'Summed Similarity', 'Across Category Simulations'}, 'FontSize', globalFontSize);
grid on;
axis equal;
set(gca, 'FontSize', globalFontSize);
addPanelLetter(figureB, '(b)', globalFontSize + 4);

pooledMinimum = repmat(poolMin, numel(simX), 1);
panelB = table(targetNumber, category, summedSim1, summedSim2, pooledMinimum, simX, simY, ...
    'VariableNames', {'target_number', 'category', 'summed_similarity_within_category', ...
    'summed_similarity_across_categories', 'pooled_minimum_subtracted', ...
    'within_category_as_plotted', 'across_categories_as_plotted'});
savePanel(figureB, panelB, 'fig2b_summed_similarity_within_vs_across', paths);

close(figureA);
close(figureB);
end

function savePanel(figureHandle, panelTable, panelName, paths)
% the panel as a 300 dpi PNG and its plotted numbers as a CSV (paths printed relative to the
% module folder). The CSV holds every number with 17 significant digits, so it keeps the exact
% values of the run.
exportgraphics(figureHandle, fullfile(paths.figures, [panelName '.png']), 'Resolution', 300);
variableNames = panelTable.Properties.VariableNames;
for variableIndex = 1:numel(variableNames)
    name = variableNames{variableIndex};
    panelTable.(name) = compose('%.17g', panelTable.(name));
end
writetable(panelTable, fullfile(paths.figureData, [panelName '.csv']));
fprintf('saved outputs/figures/%s.png and outputs/figure_data/%s.csv\n', panelName, panelName);
end

function addPanelLetter(figureHandle, letter, fontSize)
% panel letter in the top left corner of the figure
annotation(figureHandle, 'textbox', [0.01 0.90 0.10 0.09], 'String', letter, 'FontSize', fontSize, ...
    'FontWeight', 'bold', 'EdgeColor', 'none', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
end

function hLine = remSim_lsline(x, y, lineColor, lineWidth)
% Least-squares line matching lsline styling; uses centered/scaled polyfit
% to avoid badly conditioned polynomial warnings for stiff or large-magnitude data.
x = x(:);
y = y(:);
ok = isfinite(x) & isfinite(y);
x = x(ok);
y = y(ok);
if nargin < 4 || isempty(lineWidth), lineWidth = 2; end
if numel(x) < 2 || std(x, 'omitnan') < eps(max(abs(x)))
    hLine = plot(nan, nan, 'Color', lineColor, 'LineWidth', lineWidth);
    return
end
[p, ~, mu] = polyfit(x, y, 1);
xfit = [min(x), max(x)];
if xfit(1) == xfit(2)
    xfit(2) = xfit(1) + eps;
end
yfit = polyval(p, xfit, [], mu);
hLine = plot(xfit, yfit, 'Color', lineColor, 'LineWidth', lineWidth);
end
