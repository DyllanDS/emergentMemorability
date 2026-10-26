function allIdentical = bylinskii_numbers(reportedRunFile, numbersMode, sourceNote, codeParameters)
%BYLINSKII_NUMBERS  Writes outputs/numbers.csv of module 02: the numbers the paper reports about the
%   Bylinskii et al. (2015) simulation (Fig. 2, the section "Stable Memorability Across Contexts
%   Does Not Require Intrinsic Memorability", the Methods section "Bylinskii et al. (2015)
%   Context Manipulation Simulation" and the sentences of the Introduction and of "Direct
%   Manipulations: Changing What an Item Is Studied or Tested With" that restate Fig. 2a; 22
%   ledger items), computed from one saved run of bylinskiiRemSim.m, and compares that run with
%   the saved run in data/paper_run/. stats_fig2.py, which runs after this function, writes the
%   numbers.csv rows of the four Fig. 2 correlations with their 95% intervals (L1-034 to L1-041).
%
%   Inputs
%     reportedRunFile  the run whose numbers are reported: the full run of run_module('full')
%                      (outputs/simulation_runs/seed_47826_runs_1000_figureData.mat) or the saved
%                      run (data/paper_run/fig2_paper_run.mat)
%     numbersMode      the mode column of numbers.csv: 'rerun' (numbers from the full run of this
%                      module) or 'saved_run' (numbers from the saved run; the full simulation does
%                      not run)
%     sourceNote       one sentence for the note column: where the numbers come from
%     codeParameters   the parameters of the full run (modes 'full' and 'report') or of the short
%                      check (mode 'skip_long'). They supply the study-list sizes numTargetsPerSim
%                      and numFillersPerSim (30 targets and 90 fillers), constants of
%                      bylinskiiRemSim.m that the saved run in data/paper_run/ does not store.
%                      Default: the reported run's parameters.
%
%   Outputs
%     outputs/numbers.csv     one row per ledger item (columns: ledger_id, location, quote,
%                             paper_value, computed_value, match, match_type, script, mode, note);
%                             written by shared/matlab/write_numbers_csv.m, which takes the printed
%                             value, location and quote from shared/ledger/paper_numbers.csv
%     outputs/logs/<run name>_vs_paper_run.csv   (when the reported run is not the saved run file)
%                             every field of the reported run against the saved run
%     allIdentical            true when the reported run equals the saved run in every field (the
%                             p values of the correlations count as equal when both are below
%                             1e-15; see compareRuns at the bottom)
%
%   Statistic names: every correlation of Fig. 2 is a Spearman rank correlation
%   (corr(..., 'Type', 'Spearman')).

codeFolder = fileparts(mfilename('fullpath'));
addpath(fullfile(codeFolder, '..', '..', 'shared', 'matlab'));
paths = release_paths(mfilename('fullpath'));
paperRunFile = fullfile(paths.paperRun, 'fig2_paper_run.mat');

loaded = load(reportedRunFile, 'figureData');
figureData = loaded.figureData;
parameters = figureData.parameters;
if nargin < 4 || isempty(codeParameters)
    codeParameters = parameters;
end
[~, reportedName, reportedExtension] = fileparts(reportedRunFile);
fprintf('numbers.csv from %s (%s)\n', [reportedName reportedExtension], numbersMode);

%% The per-target values (targets tested in both contexts)
validBoth = logical(figureData.validBoth);
memorabilityWithin = figureData.amt1_valid;          % hit rate minus false alarm rate, Simulation 1
memorabilityAcross = figureData.amt2_valid;          % the same in Simulation 2
summedSimilarityWithin = figureData.summedSim_amt1(validBoth);   % minus mean distance to the 299 category-mates
summedSimilarityAcross = figureData.summedSim_amt2(validBoth);   % minus mean distance to all 6,299 other items
numberOfTargetsInBoth = sum(validBoth);
numberOfTargets = numel(validBoth);

%% The correlations, recomputed from the per-target values (Spearman, as in the simulation code)
[rhoMemorability, pMemorability] = corr(memorabilityWithin, memorabilityAcross, 'Type', 'Spearman');
[rhoWithin, pWithin] = corr(memorabilityWithin, summedSimilarityWithin, 'Type', 'Spearman');
[rhoAcross, pAcross] = corr(memorabilityAcross, summedSimilarityAcross, 'Type', 'Spearman');
[rhoSummedSimilarity, pSummedSimilarity] = corr(summedSimilarityWithin, summedSimilarityAcross, 'Type', 'Spearman');
% Pearson correlation of memorability within vs across (checked against the saved rho_pearson below)
pearsonMemorability = corr(memorabilityWithin, memorabilityAcross, 'Type', 'Pearson');

% The recomputed values must equal the ones the simulation saved (tolerance 1e-12, for last-digit
% differences between processors or MATLAB versions)
recomputed = [rhoMemorability, rhoWithin, rhoAcross, rhoSummedSimilarity, pearsonMemorability];
saved = [figureData.rho_spearman, figureData.rho_withinMem_sim, figureData.rho_acrossMem_sim, ...
    figureData.rho_sim, figureData.rho_pearson];
largestDifference = max(abs(recomputed - saved));
fprintf('recomputed correlations vs saved: largest difference %g\n', largestDifference);
if largestDifference > 1e-12
    error('bylinskii_numbers:Recompute', ...
        'Recomputed correlations differ from the saved ones (largest difference %g)', largestDifference);
end

%% Compare the reported run with the saved run in data/paper_run/, field by field
% (nothing to compare when the reported run is the saved run itself)
[~, paperName, paperExtension] = fileparts(paperRunFile);
reportedRunIsPaperRun = strcmp([reportedName reportedExtension], [paperName paperExtension]);
allIdentical = true;
runNote = sourceNote;
if ~reportedRunIsPaperRun
    paperLoaded = load(paperRunFile, 'figureData');
    paperRun = paperLoaded.figureData;
    comparisonName = [reportedName '_vs_paper_run.csv'];
    allIdentical = compareRuns(figureData, paperRun, fullfile(paths.logs, comparisonName));
    if allIdentical
        comparisonText = sprintf(['This run equals the saved run in data/paper_run/fig2_paper_run.mat ' ...
            'in every field (p values below 1e-15 count as equal).']);
    else
        comparisonText = sprintf(['This run DIFFERS from the saved run in data/paper_run/fig2_paper_run.mat ' ...
            'in at least one field; the paper reports the numbers of the saved run.']);
    end
    fprintf('%s\n', comparisonText);
    runNote = [sourceNote ' ' comparisonText];
end

fprintf('targets tested in both contexts: %d of %d\n', numberOfTargetsInBoth, numberOfTargets);
fprintf('Fig. 2a memorability within vs across: Spearman rho = %.4f (p = %g)\n', rhoMemorability, pMemorability);
fprintf('Fig. 2a memorability vs summed similarity, within: Spearman rho = %.4f (p = %g)\n', rhoWithin, pWithin);
fprintf('Fig. 2a memorability vs summed similarity, across: Spearman rho = %.4f (p = %g)\n', rhoAcross, pAcross);
fprintf('Fig. 2b summed similarity within vs across: Spearman rho = %.4f (p = %g)\n', rhoSummedSimilarity, ...
    pSummedSimilarity);

%% Design values, read from the run's saved parameters
itemsPerCategory = parameters.targetsPerCategory + parameters.fillersPerCategory;
otherItemsInCategory = itemsPerCategory - 1;
otherItemsInAllCategories = parameters.numCategories * itemsPerCategory - 1;
itemSpecificFeatures = parameters.numFeaturesPerItem - parameters.categoryFeatures;
runsWithinInTotal = parameters.numCategories * parameters.simLen;

%% Plotted ranges (for the two figure-panel rows)
plottedMinimum = min([summedSimilarityWithin; summedSimilarityAcross]);
plottedMaximum = max([summedSimilarityWithin; summedSimilarityAcross]);

%% Summed similarity within category minus across categories, per target (a positive difference puts
% the target below the dashed line y = x of Fig. 2b)
withinMinusAcross = summedSimilarityWithin - summedSimilarityAcross;
numberOfTargetsHigherWithin = sum(withinMinusAcross > 0);

%% The gray least-squares line of Fig. 2a (memorability across against within, as drawn)
lineOfPanelA = polyfit(memorabilityWithin, memorabilityAcross, 1);

%% The rows of numbers.csv
rows = struct('ledger_id', {}, 'computed_value', {}, 'mode', {}, 'note', {});

designValues = [parameters.numCategories, itemsPerCategory, parameters.targetsPerCategory, ...
    parameters.fillersPerCategory];
rows = addRow(rows, 'L1-027', designValues, numbersMode, ...
    ['Number of categories, items per category (targets plus fillers), targets per category and fillers per category of the run (numCategories, targetsPerCategory + fillersPerCategory, targetsPerCategory, fillersPerCategory). ' runNote]);
rows = addRow(rows, 'L1-028', parameters.numFeaturesPerItem, numbersMode, ...
    ['Number of features of every simulated item (numFeaturesPerItem). ' runNote]);
rows = addRow(rows, 'L1-029', parameters.categoryFeatures, numbersMode, ...
    ['Number of category-defining features (categoryFeatures): features 1 to 10 of every item come from one prototype draw per category. ' runNote]);

parameterText = sprintf(['seed %d; g = %g; storage probability %g; correct copy probability %g; ' ...
    'criterion %g within category and %g across categories; %d categories of %d targets and %d fillers; ' ...
    'study lists of %d targets and %d fillers; %d runs per category within and %d runs across; ' ...
    'false alarms from each studied target probed against %d new lures generated and stored in every run; serial'], ...
    parameters.seedValue, parameters.baseRateOfFeatures, parameters.probEncoding, parameters.correctEncodeProb, ...
    parameters.criterion_amt1, parameters.criterion_amt2, parameters.numCategories, parameters.targetsPerCategory, ...
    parameters.fillersPerCategory, codeParameters.numTargetsPerSim, codeParameters.numFillersPerSim, ...
    parameters.simLen, parameters.simLen, codeParameters.numTargetsPerSim);
rows = addRow(rows, 'L1-030', parameterText, numbersMode, ...
    ['Settings of the simulation run: seed, REM parameters, decision criteria, category structure, study-list sizes, numbers of runs and the false alarm procedure (Methods, "Bylinskii et al. (2015) Context Manipulation Simulation"). ' runNote]);

panelAText = sprintf(['outputs/figures/fig2a_memorability_within_vs_across.png: %d targets; memorability ' ...
    'within category %.3f to %.3f, across categories %.3f to %.3f'], numberOfTargetsInBoth, ...
    min(memorabilityWithin), max(memorabilityWithin), min(memorabilityAcross), max(memorabilityAcross));
rows = addRow(rows, 'L1-031', panelAText, numbersMode, ...
    ['Fig. 2a as drawn by code/bylinskii_makeFigure2.m: the number of targets plotted and the range of memorability (hit rate minus false alarm rate) in each context; the plotted values are in outputs/figure_data/. ' runNote]);

panelBText = sprintf(['outputs/figures/fig2b_summed_similarity_within_vs_across.png: summed similarity ' ...
    'within category %.3f to %.3f, across categories %.3f to %.3f; %d of %d targets below the dashed line ' ...
    'y = x (within minus across %.3f to %.3f); plotted after subtracting the pooled minimum, so the axes ' ...
    'run from 0 to %.2f'], min(summedSimilarityWithin), max(summedSimilarityWithin), ...
    min(summedSimilarityAcross), max(summedSimilarityAcross), numberOfTargetsHigherWithin, ...
    numberOfTargetsInBoth, min(withinMinusAcross), max(withinMinusAcross), plottedMaximum - plottedMinimum);
rows = addRow(rows, 'L1-032', panelBText, numbersMode, ...
    ['Fig. 2b as drawn by code/bylinskii_makeFigure2.m: the range of summed similarity (minus the mean Euclidean distance) ' ...
    'in each context and the number of targets below the dashed line y = x (higher summed similarity within ' ...
    'category than across categories); both axes are shifted by the pooled minimum, so they start at 0. ' runNote]);

rows = addRow(rows, 'L1-033', numberOfTargetsInBoth, numbersMode, ...
    sprintf('Number of targets tested at least once in both contexts: %d of %d. %s', numberOfTargetsInBoth, ...
    numberOfTargets, runNote));

lineText = sprintf(['outputs/figures/fig2a_memorability_within_vs_across.png: gray least-squares line of ' ...
    'memorability across categories against memorability within category over the %d targets, ' ...
    'slope %.3f, intercept %.3f'], numberOfTargetsInBoth, lineOfPanelA(1), lineOfPanelA(2));
rows = addRow(rows, 'L1-124', lineText, numbersMode, ...
    ['Fig. 2a regression line as drawn by code/bylinskii_makeFigure2.m (polyfit of degree 1). ' runNote]);

increaseText = sprintf(['%d of %d targets have higher summed similarity within category than across ' ...
    'categories (within minus across %.3f to %.3f); Spearman rho between the two = %.4f'], ...
    numberOfTargetsHigherWithin, numberOfTargetsInBoth, min(withinMinusAcross), max(withinMinusAcross), ...
    rhoSummedSimilarity);
rows = addRow(rows, 'L1-125', increaseText, numbersMode, ...
    ['Summed similarity (minus the mean Euclidean distance) of each target within category against across ' ...
    'categories, the two axes of Fig. 2b, and their Spearman rho (test T05 in outputs/tests.csv). ' runNote]);

% The sentences of the Introduction and of "Direct Manipulations: Changing What an Item Is Studied or
% Tested With" that restate Fig. 2a
restatementNote = ['Restates Fig. 2a; the value is the Spearman rho of memorability within category against ' ...
    'across categories over the targets tested in both contexts (test T03 in outputs/tests.csv). ' runNote];
rows = addRow(rows, 'L1-126', rhoMemorability, 'restatement', restatementNote);
rows = addRow(rows, 'L1-127', rhoMemorability, 'restatement', restatementNote);

rows = addRow(rows, 'L5-017', designValues, numbersMode, ...
    ['Number of categories, items per category (targets plus fillers), targets per category and fillers per category of the run (numCategories, targetsPerCategory + fillersPerCategory, targetsPerCategory, fillersPerCategory). ' runNote]);
rows = addRow(rows, 'L5-018', [parameters.numFeaturesPerItem, parameters.baseRateOfFeatures, ...
    parameters.probEncoding, parameters.correctEncodeProb], numbersMode, ...
    ['Features per item, geometric parameter g, storage probability and probability of a correct copy (numFeaturesPerItem, baseRateOfFeatures, probEncoding, correctEncodeProb). ' runNote]);
rows = addRow(rows, 'L5-019', [parameters.categoryFeatures, parameters.categoryFeatures, itemSpecificFeatures], ...
    numbersMode, ['Category-defining features (categoryFeatures), length of the prototype draw of each category ' ...
    '(geornd(g, [1, categoryFeatures]) + 1) and item-specific features (numFeaturesPerItem minus categoryFeatures). ' runNote]);
rows = addRow(rows, 'L5-020', [parameters.criterion_amt1, parameters.criterion_amt2], numbersMode, ...
    ['Decision criterion within category (criterion_amt1) and across categories (criterion_amt2). ' runNote]);
rows = addRow(rows, 'L1-121', parameters.criterion_amt1, numbersMode, ...
    ['Decision criterion of the within-category runs (criterion_amt1). ' runNote]);
rows = addRow(rows, 'L1-122', [codeParameters.numTargetsPerSim, codeParameters.numFillersPerSim, ...
    parameters.numCategories, parameters.criterion_amt2], numbersMode, ...
    ['Targets and fillers per across-category study list (numTargetsPerSim, numFillersPerSim), drawn from ' ...
    'all categories (numCategories), and the decision criterion of these runs (criterion_amt2). ' runNote]);
rows = addRow(rows, 'L1-123', parameters.numCategories, numbersMode, ...
    ['Number of within-category experiments: one per category (numCategories), each with simLen runs. ' runNote]);
rows = addRow(rows, 'L5-021', [codeParameters.numTargetsPerSim, codeParameters.numFillersPerSim], numbersMode, ...
    ['Targets and fillers per study list (numTargetsPerSim and numFillersPerSim, constants of bylinskiiRemSim.m, ' ...
    'from the parameters of the full run or, in skip_long mode, of the short check). ' runNote]);
rows = addRow(rows, 'L5-022', [parameters.simLen, parameters.simLen, runsWithinInTotal], numbersMode, ...
    ['Runs across categories, runs per category within category, and all within-category runs (simLen, simLen, numCategories x simLen). ' runNote]);

lureText = sprintf(['In every run %d new lure items are generated (geometric features, g = %g) and stored like ' ...
    'a study list; each studied target is probed against these lure traces and counts as a false ' ...
    'alarm when the mean likelihood ratio exceeds the criterion (the target is the probe, the lures ' ...
    'are the memory)'], codeParameters.numTargetsPerSim, parameters.baseRateOfFeatures);
rows = addRow(rows, 'L5-023', lureText, numbersMode, ['False alarm procedure of bylinskiiRemSim.m (the "Test false alarms" blocks). ' runNote]);

summedSimilarityText = sprintf(['within category: mean Euclidean distance to the other %d items of the ' ...
    'target''s category; across categories: mean Euclidean distance to the other %d items; both negated ' ...
    '(higher = more similar); Spearman rho between the two = %.4f'], ...
    otherItemsInCategory, otherItemsInAllCategories, rhoSummedSimilarity);
rows = addRow(rows, 'L5-024', summedSimilarityText, numbersMode, ['Summed similarity of bylinskiiRemSim.m (the "Summed Similarity" block); higher values mean more similar. ' runNote]);

write_numbers_csv(rows, paths.outputs, 'code/bylinskii_numbers.m');
end

function rows = addRow(rows, ledgerId, computedValue, mode, note)
% one numbers.csv row: the ledger id, the value the code computed, the mode and a note
newRow = struct('ledger_id', ledgerId, 'computed_value', computedValue, 'mode', mode, 'note', note);
rows(end + 1) = newRow;
end

function allIdentical = compareRuns(reportedRun, paperRun, comparisonFile)
% Every field of the saved run (and of its parameters) against the reported run: the largest
% absolute difference and whether the values are identical. The per-target values, the
% correlations and the parameters must be identical. The p values of the correlations (the fields
% p_spearman, p_pearson, p_sim, p_withinMem_sim, p_acrossMem_sim) also count as equal when both are
% below 1e-15 ("both below 1e-15" in the comparison file); the saved run stores 0 for all five.
pValueLimit = 1e-15;
fieldNames = fieldnames(paperRun);
fieldNames = fieldNames(~strcmp(fieldNames, 'parameters'));
parameterNames = fieldnames(paperRun.parameters);

names = {};
largestDifferences = [];
identical = {};
for fieldIndex = 1:numel(fieldNames)
    name = fieldNames{fieldIndex};
    [difference, isSame] = compareValues(reportedRun.(name), paperRun.(name));
    sameText = yesNo(isSame);
    isPValue = startsWith(name, 'p_');
    if isPValue && ~isSame
        reportedBelowLimit = all(double(reportedRun.(name)) < pValueLimit);
        paperBelowLimit = all(double(paperRun.(name)) < pValueLimit);
        if reportedBelowLimit && paperBelowLimit
            sameText = 'both below 1e-15';
        end
    end
    names{end + 1, 1} = name; %#ok<AGROW>
    largestDifferences(end + 1, 1) = difference; %#ok<AGROW>
    identical{end + 1, 1} = sameText; %#ok<AGROW>
end
for parameterIndex = 1:numel(parameterNames)
    name = parameterNames{parameterIndex};
    if isfield(reportedRun.parameters, name)
        [difference, isSame] = compareValues(reportedRun.parameters.(name), paperRun.parameters.(name));
    else
        difference = NaN;
        isSame = false;
    end
    names{end + 1, 1} = ['parameters.' name]; %#ok<AGROW>
    largestDifferences(end + 1, 1) = difference; %#ok<AGROW>
    identical{end + 1, 1} = yesNo(isSame); %#ok<AGROW>
end
comparison = table(names, largestDifferences, identical, ...
    'VariableNames', {'field', 'largest_absolute_difference', 'identical'});
writetable(comparison, comparisonFile);
countsAsEqual = strcmp(identical, 'yes') | strcmp(identical, 'both below 1e-15');
allIdentical = all(countsAsEqual);
end

function [difference, isSame] = compareValues(reportedValue, paperValue)
reportedValue = double(reportedValue);
paperValue = double(paperValue);
if ~isequal(size(reportedValue), size(paperValue))
    difference = NaN;
    isSame = false;
    return;
end
difference = max(abs(reportedValue(:) - paperValue(:)));
if isempty(difference)
    difference = 0;
end
isSame = isequal(reportedValue, paperValue);
end

function text = yesNo(flag)
if flag
    text = 'yes';
else
    text = 'no';
end
end
