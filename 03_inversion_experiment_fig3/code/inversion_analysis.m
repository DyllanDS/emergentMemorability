function results = inversion_analysis(thingsImageFolder)
%INVERSION_ANALYSIS  Fig. 3d and 3e: how well the 40 critical items were remembered in this task,
%   against their published memorability.
%
%   results = inversion_analysis()
%   results = inversion_analysis(thingsImageFolder)
%       thingsImageFolder: a folder of THINGS images (<concept>/<file>.jpg) to draw thumbnails in
%       Fig. 3e; default external/things_images (not included in the repository: without it,
%       Fig. 3e shows markers and concept names). No number depends on it.
%
%   What it does
%     Part 1  The test trials of the 54 sessions, S001 to S054 (shared/matlab/load_inversion_trials.m).
%     Part 2  The exclusion rules of the Methods (shared/matlab/apply_participant_exclusions.m): a
%             participant is excluded for one key on more than 80 percent of test trials, more
%             than 20 percent of test trials under 200 ms or over 7,500 ms, or hit rate minus
%             false alarm rate below 0.05; then test trials under 200 ms or over 10,000 ms are
%             dropped; then the first 50 participants are kept (nobody is removed by this cap).
%     Part 3  For each critical item, over the kept participants: the hit rate (the item was
%             studied and called old), the false alarm rate (the item was not studied and called
%             old), "old" = responses 1 to 3 on the 6-point scale, and its memorability in this
%             task, corrected recognition CR = hit rate minus false alarm rate. Its published
%             memorability is the Kramer et al. (2023) score of the image: the mean of the image's
%             score over the item's test trials (every trial of an item shows the same image), at
%             the full precision of the saved run (data/paper_run/published_scores.csv, checked
%             against the mean computed here to 1e-12), which fixes the order of the ranks of
%             items that share a score.
%     Part 4  Fig. 3d: the 40 items ranked by published score (left) and by CR in this task
%             (right); one line per item, rank 1 at the top, colored on the PiYG scale by
%             published rank (shared/matlab/piyg_colormap.m, 40 steps), green for the highest
%             published scores and pink for the lowest; each end is labeled with the item's
%             concept name (with the image code where two items share a concept).
%     Part 5  Fig. 3e: CR in this task against the published score, and the Spearman rank
%             correlation between them over the 40 items (two-sided p). One marker and concept
%             name per item, or a thumbnail when THINGS images are given; x axis 0.50 to 1.00;
%             y axis from -0.2 to 1.00 with ticks 0.00, 0.50 and 1.00, so that every item is
%             inside the axes; rho and p in the panel; no fitted line.
%     Part 6  Compares rho, p and the three participant counts with the saved run in
%             data/paper_run/ (the output the paper reports): an error when a printed value would
%             change, a warning for a difference in the last digits.
%
%   Inputs:  data/inversion_trials.csv, data/inversion_design.csv (through the shared loader);
%            shared/things/things_images.csv (the published per-image scores and image paths);
%            data/paper_run/paperStats_inversion.mat and published_scores.csv (the saved run of
%            this analysis)
%   Outputs: outputs/inversion_items_50.csv  the 40 items: image_name, block, memorability_quintile,
%                                             n_similar_items, kramer_cr (the published score as
%                                             the analysis uses it, full precision), n_old, hits,
%                                             n_new, false_alarms, hit_rate, fa_rate, cr; rows in
%                                             image-name order (modules A3a and A3b check their
%                                             item tables against it)
%            outputs/fig3_statistics.csv     rho, p and the counts, beside the saved run
%            outputs/figures/fig3d_rank_order_lines.png, fig3e_memorability_vs_cr.png (300 dpi)
%            outputs/figure_data/fig3d_rank_order_lines.csv, fig3e_memorability_vs_cr.csv
%   Returns a struct with the per-item table, rho, p, the counts and the axis ticks of Fig. 3e.
%   No random numbers.

paths = release_paths(mfilename('fullpath'));
if nargin < 1 || isempty(thingsImageFolder)
    thingsImageFolder = fullfile(paths.releaseRoot, 'external', 'things_images');
end

%% Part 1: the test trials of the 54 sessions
[T, design] = load_inversion_trials();
participantsBeforeExclusion = numel(unique(T.subjectNumber));

%% Part 2: exclude participants (exclusion rules of the Methods), then filter trials
targetSubjectsAfterExclusion = 50;
[T, exclusionReport] = apply_participant_exclusions(T, 'subjectColumn', 'subjectNumber', 'isNewColumn', 'isNew', ...
    'datasetColumn', '', 'cap', targetSubjectsAfterExclusion, 'slowRtMs', 7500);
badSubjects = exclusionReport.badSubjects;
participantsAfterExclusion = numel(unique(T.subjectNumber));
fprintf('Final participants: %d\n', participantsAfterExclusion);

%% Part 3: hit rate, false alarm rate and corrected recognition of each critical item
T.saidOld = T.response <= 3;
criticalTrials = T(T.isCritical == true, :);
uniqueImageNames = unique(criticalTrials.imageName);        % sorted by image name
numberOfItems = numel(uniqueImageNames);
savedScores = readSavedScores(fullfile(paths.paperRun, 'published_scores.csv'));

imageName = strings(numberOfItems, 1);
concept = strings(numberOfItems, 1);
block = zeros(numberOfItems, 1);
memorabilityQuintile = zeros(numberOfItems, 1);
numberOfSimilarItems = zeros(numberOfItems, 1);
publishedScore = zeros(numberOfItems, 1);
numberOfOldTrials = zeros(numberOfItems, 1);
numberOfHits = zeros(numberOfItems, 1);
numberOfNewTrials = zeros(numberOfItems, 1);
numberOfFalseAlarms = zeros(numberOfItems, 1);
hitRate = zeros(numberOfItems, 1);
falseAlarmRate = zeros(numberOfItems, 1);
for itemIndex = 1:numberOfItems
    currentImageName = uniqueImageNames(itemIndex);
    imageTrials = criticalTrials(criticalTrials.imageName == currentImageName, :);

    oldTrials = imageTrials(imageTrials.isNew == 0, :);           % the item had been studied
    hitRate(itemIndex) = sum(oldTrials.saidOld) / height(oldTrials);

    newTrials = imageTrials(imageTrials.isNew == 1, :);           % the item had not been studied
    falseAlarmRate(itemIndex) = sum(newTrials.saidOld) / height(newTrials);

    % the published score: the mean over the item's test trials of the score of the tested image
    % (every trial of an item shows the same image), at the full precision of the saved run
    savedRow = find(savedScores.image_name == currentImageName);
    if numel(savedRow) ~= 1 || abs(savedScores.kramer_cr(savedRow) - mean(imageTrials.memorabilityScore)) > 1e-12
        error('inversion_analysis:Score', '%s: data/paper_run/published_scores.csv does not hold the published score of this item', currentImageName);
    end
    publishedScore(itemIndex) = savedScores.kramer_cr(savedRow);

    imageName(itemIndex) = currentImageName;
    concept(itemIndex) = imageTrials.concept(1);
    block(itemIndex) = imageTrials.block(1);
    numberOfOldTrials(itemIndex) = height(oldTrials);
    numberOfHits(itemIndex) = sum(oldTrials.saidOld);
    numberOfNewTrials(itemIndex) = height(newTrials);
    numberOfFalseAlarms(itemIndex) = sum(newTrials.saidOld);

    designRow = find(design.image_name == currentImageName);
    if numel(designRow) ~= 1 || design.block(designRow) ~= block(itemIndex)
        error('inversion_analysis:Design', '%s is not a critical item of block %d in inversion_design.csv', currentImageName, block(itemIndex));
    end
    memorabilityQuintile(itemIndex) = design.memorability_quintile(designRow);
    numberOfSimilarItems(itemIndex) = design.n_similar_items(designRow);
end
correctedRecognition = hitRate - falseAlarmRate;
itemLabel = labelsOfItems(concept, imageName);      % concept names for the figures

items = table(imageName, block, memorabilityQuintile, numberOfSimilarItems, publishedScore, numberOfOldTrials, ...
    numberOfHits, numberOfNewTrials, numberOfFalseAlarms, hitRate, falseAlarmRate, correctedRecognition, ...
    'VariableNames', {'image_name', 'block', 'memorability_quintile', 'n_similar_items', 'kramer_cr', 'n_old', ...
    'hits', 'n_new', 'false_alarms', 'hit_rate', 'fa_rate', 'cr'});
writeExactCsv(items, fullfile(paths.outputs, 'inversion_items_50.csv'));
fprintf('wrote outputs/inversion_items_50.csv (%d items, %d participants)\n', numberOfItems, participantsAfterExclusion);

%% Part 4: Fig. 3d, rank by published score (left) against rank by CR in this task (right)
[~, kramerOrder] = sort(publishedScore, 'descend');           % kramerOrder(1): highest published score
[~, taskOrder] = sort(correctedRecognition, 'descend');       % taskOrder(1): highest CR in this task
lineColours = flipud(piyg_colormap(numberOfItems));            % row 1 dark green ... last row dark pink

kramerRank = zeros(numberOfItems, 1);
taskRank = zeros(numberOfItems, 1);
leftHeight = zeros(numberOfItems, 1);
rightHeight = zeros(numberOfItems, 1);
colourOfItem = zeros(numberOfItems, 3);
for rankPosition = 1:numberOfItems
    kramerRank(kramerOrder(rankPosition)) = rankPosition;
    taskRank(taskOrder(rankPosition)) = rankPosition;
end
figureD = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [1 1 6 9]);
axesD = axes(figureD);
hold(axesD, 'on');
for rankPosition = 1:numberOfItems
    item = kramerOrder(rankPosition);
    % rank 1 at the top: height 1 for rank 1, 1/40 for rank 40
    leftHeight(item) = (numberOfItems - kramerRank(item) + 1) / numberOfItems;
    rightHeight(item) = (numberOfItems - taskRank(item) + 1) / numberOfItems;
    colourOfItem(item, :) = lineColours(rankPosition, :);
    plot(axesD, [1, 2], [leftHeight(item), rightHeight(item)], '-', 'Color', colourOfItem(item, :), 'LineWidth', 5);
    text(axesD, 0.97, leftHeight(item), itemLabel(item), 'FontSize', 8, 'HorizontalAlignment', 'right', 'Interpreter', 'none');
    text(axesD, 2.03, rightHeight(item), itemLabel(item), 'FontSize', 8, 'HorizontalAlignment', 'left', 'Interpreter', 'none');
end
titleHeight = 1 + 1.5 / numberOfItems;
text(axesD, 1, titleHeight, 'Kramer et al.', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(axesD, 2, titleHeight, 'This task', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(axesD, 0.5, 0.5, 'Memorability rank (highest at the top)', 'FontSize', 11, 'Rotation', 90, ...
    'HorizontalAlignment', 'center');
set(axesD, 'XLim', [0.45, 2.55], 'YLim', [0, titleHeight + 0.5 / numberOfItems]);
axis(axesD, 'off');
hold(axesD, 'off');
panelD = table(imageName, itemLabel, publishedScore, correctedRecognition, kramerRank, taskRank, leftHeight, rightHeight, ...
    colourOfItem(:, 1), colourOfItem(:, 2), colourOfItem(:, 3), ...
    'VariableNames', {'image_name', 'label', 'kramer_cr', 'cr', 'rank_kramer', 'rank_this_task', 'height_left', ...
    'height_right', 'line_red', 'line_green', 'line_blue'});
panelD = sortrows(panelD, 'rank_kramer');
savePanelExact(figureD, 'fig3d_rank_order_lines', paths, panelD);

%% Part 5: Fig. 3e, CR in this task against the published score
[r, p] = corr(publishedScore, correctedRecognition, 'Type', 'Spearman');
fprintf('Memorability vs. CR: Spearman rho = %.3f, p = %.3e\n', r, p);

figureE = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', 'Position', [1 1 7 7]);
axesE = axes(figureE);
hold(axesE, 'on');
xLimitsE = [0.5, 1.0];
yLimitsE = [-0.2, 1.0];
imageSize = 0.06;              % thumbnail width in x data units
imageHeight = imageSize * diff(yLimitsE) / diff(xLimitsE);   % the same height on screen (axis square)
images = readtable(fullfile(paths.sharedThings, 'things_images.csv'), ...
    setvartype(detectImportOptions(fullfile(paths.sharedThings, 'things_images.csv')), {'filename', 'rel_path'}, 'string'));
thumbnailsDrawn = 0;
[labelPosition, labelAlignment] = placeLabels(publishedScore, correctedRecognition, itemLabel);
for item = 1:numberOfItems
    imageRow = find(images.filename == imageName(item), 1);
    imagePath = fullfile(thingsImageFolder, char(images.rel_path(imageRow)));
    if exist(imagePath, 'file')
        picture = imread(imagePath);
        if size(picture, 3) == 1                    % a gray-scale image: the same value in red, green and blue
            picture = repmat(picture, 1, 1, 3);
        end
        picture = flipud(picture);                  % image() puts the first row at the bottom of these axes
        image(axesE, 'XData', [publishedScore(item) - imageSize / 2, publishedScore(item) + imageSize / 2], ...
            'YData', [correctedRecognition(item) - imageHeight / 2, correctedRecognition(item) + imageHeight / 2], 'CData', picture);
        thumbnailsDrawn = thumbnailsDrawn + 1;
    else
        scatter(axesE, publishedScore(item), correctedRecognition(item), 70, colourOfItem(item, :), 'filled', ...
            'MarkerEdgeColor', [0.2 0.2 0.2], 'LineWidth', 0.75);
        text(axesE, labelPosition(item, 1), labelPosition(item, 2), itemLabel(item), 'FontSize', 8, ...
            'Interpreter', 'none', 'VerticalAlignment', 'middle', 'HorizontalAlignment', labelAlignment(item));
    end
end
if thumbnailsDrawn > 0
    fprintf('Fig. 3e: %d thumbnails drawn from %s\n', thumbnailsDrawn, thingsImageFolder);
else
    fprintf('Fig. 3e: no THINGS images found (external/things_images); markers and concept names drawn\n');
end
xTickValues = [0.5, 0.75, 1.0];
yTickValues = [0, 0.5, 1.0];
set(axesE, 'XLim', xLimitsE, 'YLim', yLimitsE, 'XTick', xTickValues, 'YTick', yTickValues, ...
    'XTickLabel', cellstr(compose('%.2f', xTickValues)), 'YTickLabel', cellstr(compose('%.2f', yTickValues)), ...
    'FontSize', 16, 'LineWidth', 2, 'Box', 'off', 'TickDir', 'out');
axis(axesE, 'square');
xlabel(axesE, 'Memorability in Kramer et al.');
ylabel(axesE, 'Memorability in this task');
if p < 0.001
    pText = 'p < 0.001';
else
    pText = sprintf('p = %.3f', p);
end
% rho and p in the lower left corner of the panel (data units: axis square shrinks the box inside
% the axes position, so normalized units would put the text outside it)
xLimits = axesE.XLim;
yLimits = axesE.YLim;
text(axesE, xLimits(1) + 0.03 * diff(xLimits), yLimits(1) + 0.03 * diff(yLimits), {sprintf('\\rho = %.2f', r), pText}, ...
    'FontSize', 16, 'VerticalAlignment', 'bottom');
hold(axesE, 'off');
if any(publishedScore < axesE.XLim(1) | publishedScore > axesE.XLim(2)) || ...
        any(correctedRecognition < axesE.YLim(1) | correctedRecognition > axesE.YLim(2))
    error('inversion_analysis:Axes', 'an item of Fig. 3e lies outside the axes');
end
panelE = table(imageName, itemLabel, publishedScore, correctedRecognition, hitRate, falseAlarmRate, ...
    colourOfItem(:, 1), colourOfItem(:, 2), colourOfItem(:, 3), ...
    'VariableNames', {'image_name', 'label', 'kramer_cr', 'cr', 'hit_rate', 'fa_rate', 'marker_red', 'marker_green', ...
    'marker_blue'});
savePanelExact(figureE, 'fig3e_memorability_vs_cr', paths, panelE);

%% Part 6: compare with the saved run in data/paper_run/ (the output the paper reports)
paperRun = load(fullfile(paths.paperRun, 'paperStats_inversion.mat'));
paperStats = paperRun.paperStats;
thisRun = [r, p, participantsAfterExclusion, participantsBeforeExclusion, numel(badSubjects)];
paperValues = [paperStats.r_memVsCR, paperStats.p_memVsCR, paperStats.nParticipantsFinal, ...
    paperStats.nParticipantsBeforeExclusion, paperStats.nExcluded];
quantity = {'Spearman rho, published score vs CR in this task (40 items)'; 'p (two-sided)'; ...
    'participants kept'; 'participants before exclusion'; 'participants excluded'};
statisticsTable = table(quantity, thisRun', paperValues', thisRun' - paperValues', ...
    'VariableNames', {'quantity', 'this_run', 'paper_run', 'difference'});
writeExactCsv(statisticsTable, fullfile(paths.outputs, 'fig3_statistics.csv'));
largestDifference = max(abs(thisRun - paperValues));
fprintf('saved run (data/paper_run/paperStats_inversion.mat): rho = %.10f, p = %.10g; this run: rho = %.10f, p = %.10g; largest difference %.3g\n', ...
    paperStats.r_memVsCR, paperStats.p_memVsCR, r, p, largestDifference);
% The run stops only when a printed value would change: rho to two decimals (-0.50), p at three
% decimals and against the printed bound 0.01, and the three participant counts exactly. A
% difference in the last digits is reported as a warning, with the exact difference in
% fig3_statistics.csv.
printedValuesDiffer = round(r, 2) ~= round(paperStats.r_memVsCR, 2) || ...
    round(p, 3) ~= round(paperStats.p_memVsCR, 3) || (p < 0.01) ~= (paperStats.p_memVsCR < 0.01) || ...
    any(thisRun(3:5) ~= paperValues(3:5));
if printedValuesDiffer
    error('inversion_analysis:PaperRun', 'this run differs from the saved run in data/paper_run/ at the printed precision');
elseif largestDifference > 1e-12
    fprintf(['WARNING: rho or p differs from the saved run in data/paper_run/ in the last digits (largest difference %.3g); ' ...
        'every printed value is the same (outputs/fig3_statistics.csv)\n'], largestDifference);
end

%% Return
results = struct();
results.items = items;
results.rho = r;
results.p = p;
results.participantsBeforeExclusion = participantsBeforeExclusion;
results.participantsExcluded = numel(badSubjects);
results.participantsKept = participantsAfterExclusion;
results.numberOfItems = numberOfItems;
results.xTicks = axesE.XTick;
results.yTicks = axesE.YTick;
results.thumbnailsDrawn = thumbnailsDrawn;
results.largestDifferenceFromPaperRun = largestDifference;
end


function savedScores = readSavedScores(fileName)
% image_name and kramer_cr of data/paper_run/published_scores.csv; kramer_cr is read as text and
% converted with str2double, so that each value is the double of the saved run
options = detectImportOptions(fileName, 'Delimiter', ',');
options = setvartype(options, {'image_name', 'kramer_cr'}, 'string');
savedScores = readtable(fileName, options);
savedScores.kramer_cr = str2double(savedScores.kramer_cr);
end


function label = labelsOfItems(concept, imageName)
% the concept name of each item (underscores as spaces); where two items share a concept, the image
% code is added, e.g. "filing cabinet (02s)"
numberOfItems = numel(concept);
label = strings(numberOfItems, 1);
for item = 1:numberOfItems
    label(item) = strrep(concept(item), "_", " ");
    if sum(concept == concept(item)) > 1
        nameParts = split(erase(imageName(item), ".jpg"), "_");
        label(item) = label(item) + " (" + nameParts(end) + ")";
    end
end
end


function [position, alignment] = placeLabels(x, y, label)
% where to write each item's name in Fig. 3e: right of its marker if that space is free, else left
% of it, else the same one line lower, and so on (names are placed from the top of the panel down;
% a space is free when it overlaps no name placed before and no other marker)
characterWidth = 0.0024;        % width of one character in x units (font size 8 on these axes)
lineHeight = 0.025;             % height of one line in y units
gap = 0.008;                    % space between a marker and its name
markerHalfWidth = 0.005;        % half the size of a marker in x units
markerHalfHeight = 0.012;       % and in y units
numberOfItems = numel(x);
position = [x + gap, y];
alignment = repmat("left", numberOfItems, 1);
placedBoxes = zeros(0, 4);      % left, right, bottom, top of every name placed so far
[~, order] = sort(y, 'descend');
for orderIndex = 1:numberOfItems
    item = order(orderIndex);
    width = characterWidth * strlength(label(item));
    found = false;
    for linesDown = 0:numberOfItems
        textHeight = y(item) - linesDown * lineHeight;
        for side = ["right", "left"]
            if side == "right"
                box = [x(item) + gap, x(item) + gap + width, textHeight - lineHeight / 2, textHeight + lineHeight / 2];
            else
                box = [x(item) - gap - width, x(item) - gap, textHeight - lineHeight / 2, textHeight + lineHeight / 2];
            end
            isFree = true;
            for placedIndex = 1:size(placedBoxes, 1)
                other = placedBoxes(placedIndex, :);
                if box(1) < other(2) && other(1) < box(2) && box(3) < other(4) && other(3) < box(4)
                    isFree = false;
                end
            end
            for otherItem = 1:numberOfItems
                if otherItem ~= item && box(1) < x(otherItem) + markerHalfWidth && x(otherItem) - markerHalfWidth < box(2) && ...
                        box(3) < y(otherItem) + markerHalfHeight && y(otherItem) - markerHalfHeight < box(4)
                    isFree = false;
                end
            end
            if isFree
                if side == "right"
                    position(item, :) = [box(1), textHeight];
                    alignment(item) = "left";          % the text starts at the left edge of its box
                else
                    position(item, :) = [box(2), textHeight];
                    alignment(item) = "right";         % the text ends at the right edge of its box
                end
                placedBoxes(end + 1, :) = box; %#ok<AGROW>
                found = true;
                break;
            end
        end
        if found
            break;
        end
    end
end
end


function savePanelExact(figureHandle, panelName, paths, panelData)
% a figure panel as a 300 dpi PNG in outputs/figures and its plotted numbers as a CSV in
% outputs/figure_data, the numbers at full precision
exportgraphics(figureHandle, fullfile(paths.figures, [panelName '.png']), 'Resolution', 300);
writeExactCsv(panelData, fullfile(paths.figureData, [panelName '.csv']));
fprintf('saved outputs/figures/%s.png and outputs/figure_data/%s.csv\n', panelName, panelName);
end


function writeExactCsv(data, fileName)
% Write a table as CSV with every number as the shortest decimal (15, 16 or 17 significant digits)
% that reads back as the same double, so that other scripts (MATLAB or Python) read exactly the
% values computed here. (writetable keeps 15 significant digits.)
handle = fopen(fileName, 'w', 'n', 'UTF-8');
names = data.Properties.VariableNames;
fprintf(handle, '%s\n', strjoin(names, ','));
for rowIndex = 1:height(data)
    fields = cell(1, numel(names));
    for columnIndex = 1:numel(names)
        value = data.(names{columnIndex})(rowIndex);
        if iscell(value)
            value = value{1};
        end
        if isnumeric(value) || islogical(value)
            fields{columnIndex} = exactNumberText(double(value));
        else
            text = char(string(value));
            if any(text == ',') || any(text == '"')
                text = ['"' strrep(text, '"', '""') '"'];
            end
            fields{columnIndex} = text;
        end
    end
    fprintf(handle, '%s\n', strjoin(fields, ','));
end
fclose(handle);
end


function text = exactNumberText(value)
if isnan(value)
    text = 'NaN';
    return;
end
for digits = 15:17
    text = sprintf('%.*g', digits, value);
    if str2double(text) == value
        return;
    end
end
end
