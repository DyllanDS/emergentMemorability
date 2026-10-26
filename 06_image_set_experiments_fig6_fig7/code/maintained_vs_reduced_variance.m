function fig7 = maintained_vs_reduced_variance()
%MAINTAINED_VS_REDUCED_VARIANCE  Fig. 7b to 7d: summed similarity and feature-memorability strength in
%   the maintained and the reduced image set, with the participant bootstrap of Fig. 7d.
%
%   What it does
%     1. Summed similarity of every concept within the maintained set and within the reduced
%        set (kernel normpdf(similarity, 1, 2), weights = the concepts' shares of the set's slots
%        with one slot of the concept itself removed, renormalized). Its mean and SD over the
%        concepts of each set are the numbers printed with Fig. 7b, 7c (maintained SD 0.00322,
%        reduced 0.00156, both means 0.18). The histograms of Fig. 7b, 7c show the summed
%        similarity of all 1,854 concepts to the set (summed_similarity_to_set.m).
%     2. For the 150 kept participants of each set (after the exclusions in the Methods), each
%        participant's hit rate and false alarm rate for each concept.
%     3. Bootstrap (1,000 resamples, seed rng(42)): resample the participants of each set with
%        replacement; average their per-concept hit rates and false alarm rates (ignoring
%        participants who did not see the concept as old, or as new); CR = hit rate minus false
%        alarm rate; Spearman correlation of each of the 49 features with CR across concepts; the
%        sum of the 49 squared correlations ("summed variance explained").
%     4. Fig. 7d: the mean of that sum over resamples per set with its 2.5 and 97.5 percentiles;
%        the percent reduction from the maintained to the reduced set (mean over resamples of
%        (maintained - reduced) / maintained x 100, 27.3 percent) and the one-sided bootstrap p
%        for the predicted direction (share of resamples in which the reduced-set sum is at least
%        the maintained-set sum, 0.036).
%
%   Inputs
%     data/maintained_trials.csv, data/reduced_trials.csv (through shared/matlab/load_image_set_trials.m)
%     data/sampled_image_sets.mat; shared/things/concept_info_human.mat (through load_concept_info.m)
%     data/paper_run/fig7_variance_paper_run.mat (the saved run, for the comparison)
%
%   Outputs (outputs/)
%     varianceData.mat               the Fig. 7 results, in the format of the saved run in
%                                    data/paper_run/
%     fig7_variance_summary.csv      every Fig. 7 number, next to the saved run in data/paper_run/
%     summed_similarity_by_concept.csv  per concept: slots and summed similarity in the maintained,
%                                    changed and reduced set
%     figures/fig7b, fig7c, fig7d (PNG, 300 dpi) and figure_data/fig7b, fig7c, fig7d (CSV)
%   Returns a struct with the Fig. 7 numbers.
%
%   Random numbers: rng(42) is set right before the bootstrap loop (Part 6), after all data are
%   loaded, and no random number is drawn before it. In each resample the maintained set's
%   participants are drawn first (randi), then the reduced set's; participants are indexed in the
%   order of their sorted codes. In this order the 1,000 resamples equal those of the saved run in
%   data/paper_run/.

paths = release_paths(mfilename('fullpath'));
close all;

%% Part 1: supporting data
concInfo = load_concept_info('human');
sampledSets = load(fullfile(paths.data, 'sampled_image_sets.mat'));
representativeDataset = sampledSets.representativeDataset;   % the maintained set
equatedDataset = sampledSets.equatedDataset;                 % the reduced set
flipMemorabilityDataset = sampledSets.flipMemorabilityDataset;   % the changed set (only for the table of summed similarities)

stdForWeightedSim = 2;
nConcepts = length(concInfo.concepts);
nFeatures = size(concInfo.features, 2);

%% Part 2: summed similarity within each set
% Representative (maintained) dataset
for i = 1:nConcepts
    representativeDataset.weightedSim(i, 1) = nan;
    if representativeDataset.numImagesForConcept(i) > 0
        counts = representativeDataset.numImagesForConcept;
        counts(i) = counts(i) - 1;
        weights = counts / sum(counts);
        representativeDataset.weightedSim(i, 1) = sum(normpdf(concInfo.thingsSimRSM(i, :), 1, stdForWeightedSim) * weights);
    end
end

% Equated (reduced) dataset
for i = 1:nConcepts
    equatedDataset.weightedSim(i, 1) = nan;
    if equatedDataset.numImagesForConcept(i) > 0
        counts = equatedDataset.numImagesForConcept;
        counts(i) = counts(i) - 1;
        weights = counts / sum(counts);
        equatedDataset.weightedSim(i, 1) = sum(normpdf(concInfo.thingsSimRSM(i, :), 1, stdForWeightedSim) * weights);
    end
end

repWeightedSim = representativeDataset.weightedSim;
equatedWeightedSim = equatedDataset.weightedSim;

%% Part 3: trials of the maintained and reduced sets
datasetLabels = {'representativeImages', 'equatedImages'};
T = load_image_set_trials({'maintained', 'reduced'}, concInfo, false);

%% Part 4: participant exclusions, trial filter and the 150 cap (the exclusion criteria of the Methods)
[T, exclusionReport] = apply_participant_exclusions(T);
badSubjects = exclusionReport.badSubjects;
for d = 1:length(datasetLabels)
    nSub = length(unique(T.subject(T.dataset_name == string(datasetLabels{d}))));
    fprintf('Final %s: %d subjects\n', datasetLabels{d}, nSub);
end

%% Part 5: per-participant, per-concept hit and false alarm rates
T_rep = T(T.dataset_index == 1, :);
T_equated = T(T.dataset_index == 2, :);

subjects_rep = unique(T_rep.subject);
subjects_equated = unique(T_equated.subject);
n_rep = length(subjects_rep);
n_equated = length(subjects_equated);

fprintf('\nBuilding per-subject per-concept hit/FA matrices...\n');
fprintf('  Maintained: %d subjects, Reduced: %d subjects\n', n_rep, n_equated);

perConceptSubj_rep_hit = NaN(nConcepts, n_rep);
perConceptSubj_rep_fa = NaN(nConcepts, n_rep);
perConceptSubj_equated_hit = NaN(nConcepts, n_equated);
perConceptSubj_equated_fa = NaN(nConcepts, n_equated);

for i = 1:n_rep
    subjTrials = T_rep(T_rep.subject == subjects_rep(i), :);
    for c = 1:nConcepts
        conceptOld = subjTrials(strcmp(subjTrials.concept, concInfo.concepts{c}) & subjTrials.is_new == 0, :);
        conceptNew = subjTrials(strcmp(subjTrials.concept, concInfo.concepts{c}) & subjTrials.is_new == 1, :);
        if ~isempty(conceptOld)
            perConceptSubj_rep_hit(c, i) = mean(conceptOld.response <= 3);
        end
        if ~isempty(conceptNew)
            perConceptSubj_rep_fa(c, i) = mean(conceptNew.response <= 3);
        end
    end
    if mod(i, 50) == 0
        fprintf('  Maintained: %d/%d subjects processed\n', i, n_rep);
    end
end

for i = 1:n_equated
    subjTrials = T_equated(T_equated.subject == subjects_equated(i), :);
    for c = 1:nConcepts
        conceptOld = subjTrials(strcmp(subjTrials.concept, concInfo.concepts{c}) & subjTrials.is_new == 0, :);
        conceptNew = subjTrials(strcmp(subjTrials.concept, concInfo.concepts{c}) & subjTrials.is_new == 1, :);
        if ~isempty(conceptOld)
            perConceptSubj_equated_hit(c, i) = mean(conceptOld.response <= 3);
        end
        if ~isempty(conceptNew)
            perConceptSubj_equated_fa(c, i) = mean(conceptNew.response <= 3);
        end
    end
    if mod(i, 50) == 0
        fprintf('  Reduced: %d/%d subjects processed\n', i, n_equated);
    end
end

%% Part 6: bootstrap over participants (1,000 resamples, rng(42))
nBootstraps = 1000;
fprintf('\nRunning %d bootstrap iterations...\n', nBootstraps);
tic;

bootstrapFeatureMemCorrs_rep = zeros(nBootstraps, nFeatures);
bootstrapFeatureMemCorrs_equated = zeros(nBootstraps, nFeatures);

rng(42, 'twister'); % reproducibility
for b = 1:nBootstraps
    % Resample subjects with replacement (maintained first, then reduced, in every resample)
    bootstrap_subjects_rep = randi(n_rep, n_rep, 1);
    bootstrap_subjects_equated = randi(n_equated, n_equated, 1);

    % Per-concept CR from the resampled subjects
    curHit_rep = mean(perConceptSubj_rep_hit(:, bootstrap_subjects_rep), 2, 'omitnan');
    curFa_rep = mean(perConceptSubj_rep_fa(:, bootstrap_subjects_rep), 2, 'omitnan');
    curHit_equated = mean(perConceptSubj_equated_hit(:, bootstrap_subjects_equated), 2, 'omitnan');
    curFa_equated = mean(perConceptSubj_equated_fa(:, bootstrap_subjects_equated), 2, 'omitnan');
    bootstrap_cr_rep = curHit_rep - curFa_rep;
    bootstrap_cr_equated = curHit_equated - curFa_equated;

    % Spearman correlation of each feature with the bootstrap CR
    rho_rep = corr(concInfo.features, bootstrap_cr_rep, 'Type', 'Spearman', 'rows', 'complete');
    rho_equated = corr(concInfo.features, bootstrap_cr_equated, 'Type', 'Spearman', 'rows', 'complete');
    bootstrapFeatureMemCorrs_rep(b, :) = rho_rep;
    bootstrapFeatureMemCorrs_equated(b, :) = rho_equated;

    if mod(b, 200) == 0
        fprintf('  %d/%d bootstraps complete\n', b, nBootstraps);
    end
end
fprintf('Bootstrap finished in %.1f seconds.\n', toc);

%% Part 7: summary statistics
% Summed rho^2 across all features per bootstrap
summedVarExplained_rep = sum(bootstrapFeatureMemCorrs_rep .^ 2, 2);
summedVarExplained_equated = sum(bootstrapFeatureMemCorrs_equated .^ 2, 2);

mean_summed_rep = mean(summedVarExplained_rep);
mean_summed_equated = mean(summedVarExplained_equated);
ci_summed_rep = prctile(summedVarExplained_rep, [2.5, 97.5]);
ci_summed_equated = prctile(summedVarExplained_equated, [2.5, 97.5]);

summed_diff = summedVarExplained_rep - summedVarExplained_equated;

percentReduction_summed = summed_diff ./ summedVarExplained_rep * 100;
mean_percentReduction_summed = mean(percentReduction_summed);

% One-sided p for the predicted direction (reduced >= maintained)
p_value_summed = mean(summedVarExplained_equated >= summedVarExplained_rep);

fprintf('\n=== Summed Variance Explained (sum of rho^2 across all features) ===\n');
fprintf('Maintained Dataset:\n');
fprintf('  Mean: %.6f, 95%% CI: [%.6f, %.6f]\n', mean_summed_rep, ci_summed_rep(1), ci_summed_rep(2));
fprintf('Reduced Dataset:\n');
fprintf('  Mean: %.6f, 95%% CI: [%.6f, %.6f]\n', mean_summed_equated, ci_summed_equated(1), ci_summed_equated(2));
fprintf('  Percent reduction: %.2f%%\n', mean_percentReduction_summed);
fprintf('  P-value (one-sided, reduced >= maintained): %.4f\n', p_value_summed);
fprintf('===================================================================\n\n');

% Summed similarity: mean and SD over the concepts of each set (Fig. 7b, 7c)
repMean = mean(repWeightedSim, 'omitnan');
repStd = std(repWeightedSim, 'omitnan');
repVar = var(repWeightedSim, 'omitnan');
equatedMean = mean(equatedWeightedSim, 'omitnan');
equatedStd = std(equatedWeightedSim, 'omitnan');
equatedVar = var(equatedWeightedSim, 'omitnan');
fprintf('Maintained Dataset: Mean Weighted Sim = %.6f, Std = %.6f, Var = %.6f\n', repMean, repStd, repVar);
fprintf('Reduced Dataset:    Mean Weighted Sim = %.6f, Std = %.6f, Var = %.6f\n', equatedMean, equatedStd, equatedVar);

% The summed variance explained on the full sample, without resampling (stats_fig6_fig7.py
% checks its own code against these two values)
fullSampleCr_rep = mean(perConceptSubj_rep_hit, 2, 'omitnan') - mean(perConceptSubj_rep_fa, 2, 'omitnan');
fullSampleCr_equated = mean(perConceptSubj_equated_hit, 2, 'omitnan') - mean(perConceptSubj_equated_fa, 2, 'omitnan');
fullSampleSummed_rep = sum(corr(concInfo.features, fullSampleCr_rep, 'Type', 'Spearman', 'rows', 'complete') .^ 2);
fullSampleSummed_equated = sum(corr(concInfo.features, fullSampleCr_equated, 'Type', 'Spearman', 'rows', 'complete') .^ 2);

% The histograms of Fig. 7b, 7c show the summed similarity of all 1,854 concepts to the set
% (summed_similarity_to_set.m; every other concept weighted by its share of the set's slots, the
% concept itself left out). Their mean and SD are listed in the summary table below; the mean and
% SD printed with Fig. 7b, 7c (repMean, repStd, equatedMean, equatedStd above) are over the
% concepts in each set, with the within-set summed similarity of Part 2.
similarityKernel = normpdf(concInfo.thingsSimRSM, 1, stdForWeightedSim);
maintainedSimToSet = summed_similarity_to_set(similarityKernel, representativeDataset.numImagesForConcept);
reducedSimToSet = summed_similarity_to_set(similarityKernel, equatedDataset.numImagesForConcept);

%% Part 8: save the results and compare them with the saved run in data/paper_run/
varianceData = struct();
varianceData.repWeightedSim = repWeightedSim;
varianceData.equatedWeightedSim = equatedWeightedSim;
varianceData.bootstrapFeatureMemCorrs_rep = bootstrapFeatureMemCorrs_rep;
varianceData.bootstrapFeatureMemCorrs_equated = bootstrapFeatureMemCorrs_equated;
varianceData.n_rep = n_rep;
varianceData.n_equated = n_equated;
varianceData.nBootstraps = nBootstraps;
varianceData.nFeatures = nFeatures;
varianceData.repMean = repMean;
varianceData.repStd = repStd;
varianceData.equatedMean = equatedMean;
varianceData.equatedStd = equatedStd;
varianceData.mean_percentReduction_summed = mean_percentReduction_summed;
varianceData.p_value_summed = p_value_summed;
varianceData.mean_summed_rep = mean_summed_rep;
varianceData.ci_summed_rep = ci_summed_rep;
varianceData.mean_summed_equated = mean_summed_equated;
varianceData.ci_summed_equated = ci_summed_equated;
varianceData.nExcluded = length(badSubjects);
save(fullfile(paths.outputs, 'varianceData.mat'), 'varianceData', '-v7');

paperRun = load(fullfile(paths.paperRun, 'fig7_variance_paper_run.mat'));
paperRun = paperRun.varianceData;
largestDifferenceMaintained = largest_difference(bootstrapFeatureMemCorrs_rep, paperRun.bootstrapFeatureMemCorrs_rep);
largestDifferenceReduced = largest_difference(bootstrapFeatureMemCorrs_equated, paperRun.bootstrapFeatureMemCorrs_equated);
fprintf('largest difference of the bootstrap correlations from the saved run: maintained %.2g, reduced %.2g\n', ...
    largestDifferenceMaintained, largestDifferenceReduced);

quantity = {'percent reduction in summed rho^2 (mean over resamples)'; ...
    'one-sided bootstrap p (reduced >= maintained)'; 'summed rho^2, maintained set (mean over resamples)'; ...
    'summed rho^2, maintained set, 2.5th percentile'; 'summed rho^2, maintained set, 97.5th percentile'; ...
    'summed rho^2, reduced set (mean over resamples)'; 'summed rho^2, reduced set, 2.5th percentile'; ...
    'summed rho^2, reduced set, 97.5th percentile'; ...
    'summed similarity, maintained set, mean over concepts'; 'summed similarity, maintained set, SD over concepts'; ...
    'summed similarity, reduced set, mean over concepts'; 'summed similarity, reduced set, SD over concepts'; ...
    'concepts in the maintained set'; 'concepts in the reduced set'; ...
    'participants kept, maintained set'; 'participants kept, reduced set'; ...
    'participants excluded (maintained and reduced sets)'; 'bootstrap resamples'; 'features'; ...
    'summed rho^2, maintained set, full sample (no resampling)'; ...
    'summed rho^2, reduced set, full sample (no resampling)'; ...
    'largest difference of the bootstrap correlations from the saved run, maintained set'; ...
    'largest difference of the bootstrap correlations from the saved run, reduced set'; ...
    'Fig. 7c histogram: summed similarity of all 1,854 concepts to the maintained set, mean'; ...
    'Fig. 7c histogram: summed similarity of all 1,854 concepts to the maintained set, SD'; ...
    'Fig. 7b histogram: summed similarity of all 1,854 concepts to the reduced set, mean'; ...
    'Fig. 7b histogram: summed similarity of all 1,854 concepts to the reduced set, SD'};
value = [mean_percentReduction_summed; p_value_summed; mean_summed_rep; ci_summed_rep(1); ci_summed_rep(2); ...
    mean_summed_equated; ci_summed_equated(1); ci_summed_equated(2); repMean; repStd; equatedMean; equatedStd; ...
    sum(~isnan(repWeightedSim)); sum(~isnan(equatedWeightedSim)); n_rep; n_equated; length(badSubjects); ...
    nBootstraps; nFeatures; fullSampleSummed_rep; fullSampleSummed_equated; largestDifferenceMaintained; ...
    largestDifferenceReduced; mean(maintainedSimToSet); std(maintainedSimToSet); mean(reducedSimToSet); ...
    std(reducedSimToSet)];
paper_run_value = [paperRun.mean_percentReduction_summed; paperRun.p_value_summed; ...
    paperRun.mean_summed_rep; paperRun.ci_summed_rep(1); paperRun.ci_summed_rep(2); paperRun.mean_summed_equated; ...
    paperRun.ci_summed_equated(1); paperRun.ci_summed_equated(2); paperRun.repMean; paperRun.repStd; ...
    paperRun.equatedMean; paperRun.equatedStd; sum(~isnan(paperRun.repWeightedSim)); ...
    sum(~isnan(paperRun.equatedWeightedSim)); paperRun.n_rep; paperRun.n_equated; paperRun.nExcluded; ...
    paperRun.nBootstraps; paperRun.nFeatures; NaN; NaN; NaN; NaN; NaN; NaN; NaN; NaN];
summaryTable = table(quantity, value, paper_run_value, value - paper_run_value, ...
    'VariableNames', {'quantity', 'value', 'paper_run_value', 'difference'});
writetable(summaryTable, fullfile(paths.outputs, 'fig7_variance_summary.csv'));
fprintf('Fig. 7d: %.4f percent (saved run %.4f), p one-sided %.3f\n', mean_percentReduction_summed, ...
    paperRun.mean_percentReduction_summed, p_value_summed);

% within-set summed similarity of every concept in each of the three sets (the changed set too)
changedWeightedSim = nan(nConcepts, 1);
for i = 1:nConcepts
    if flipMemorabilityDataset.numImagesForConcept(i) > 0
        counts = flipMemorabilityDataset.numImagesForConcept;
        counts(i) = counts(i) - 1;
        weights = counts / sum(counts);
        changedWeightedSim(i) = sum(normpdf(concInfo.thingsSimRSM(i, :), 1, stdForWeightedSim) * weights);
    end
end
similarityTable = table((1:nConcepts)', string(concInfo.concepts(:)), ...
    representativeDataset.numImagesForConcept(:), repWeightedSim, ...
    flipMemorabilityDataset.numImagesForConcept(:), changedWeightedSim, ...
    equatedDataset.numImagesForConcept(:), equatedWeightedSim, ...
    'VariableNames', {'concept_number', 'concept', 'slots_maintained', 'summed_similarity_maintained', ...
    'slots_changed', 'summed_similarity_changed', 'slots_reduced', 'summed_similarity_reduced'});
writetable(similarityTable, fullfile(paths.outputs, 'summed_similarity_by_concept.csv'));

%% Part 9: Fig. 7b, 7c (histograms of summed similarity) and 7d (summed variance explained)
fontSizePt = 18;
colorMaintained = [0.9 0.3 0.3];
colorReduced = [0.3 0.6 0.9];

drawHistogram(paths, 'fig7b_reduced_set_histogram', 'b   Reduced image set', reducedSimToSet, colorReduced, fontSizePt);
drawHistogram(paths, 'fig7c_maintained_set_histogram', 'c   Maintained image set', maintainedSimToSet, colorMaintained, ...
    fontSizePt);

figureHandle = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 560 560]);
barHandle = bar([mean_summed_rep, mean_summed_equated], 'BarWidth', 0.65);
barHandle.FaceColor = 'flat';
barHandle.CData(1, :) = colorMaintained;
barHandle.CData(2, :) = colorReduced;
barHandle.EdgeColor = 'none';
hold on;
errLo = [mean_summed_rep - ci_summed_rep(1); mean_summed_equated - ci_summed_equated(1)];
errHi = [ci_summed_rep(2) - mean_summed_rep; ci_summed_equated(2) - mean_summed_equated];
% two calls so MATLAB does not draw a line connecting the bars
errorbar(1, mean_summed_rep, errLo(1), errHi(1), 'Color', 'k', 'LineWidth', 2, 'CapSize', 10);
errorbar(2, mean_summed_equated, errLo(2), errHi(2), 'Color', 'k', 'LineWidth', 2, 'CapSize', 10);
xticks([1, 2]);
xticklabels({'Maintained image set', 'Reduced image set'});
xlim([0.4 2.6]);
ylabel({'Summed variance explained', 'by image features'}, 'FontSize', fontSizePt);
title(sprintf('d   %.1f%% lower, one-sided p = %.3f', mean_percentReduction_summed, p_value_summed), 'FontSize', fontSizePt);
grid on;
box on;
set(gca, 'LineWidth', 2, 'FontSize', fontSizePt - 2, 'TickDir', 'out');
barData = table({'maintained'; 'reduced'}, [mean_summed_rep; mean_summed_equated], ...
    [ci_summed_rep(1); ci_summed_equated(1)], [ci_summed_rep(2); ci_summed_equated(2)], [n_rep; n_equated], ...
    [nBootstraps; nBootstraps], 'VariableNames', {'image_set', 'summed_rho_squared_mean', 'ci_2_5', 'ci_97_5', ...
    'participants', 'bootstrap_resamples'});
save_panel(figureHandle, 'fig7d_summed_variance_explained', paths, barData);
close(figureHandle);

fig7 = struct();
fig7.repMean = repMean;
fig7.repStd = repStd;
fig7.equatedMean = equatedMean;
fig7.equatedStd = equatedStd;
fig7.percentReduction = mean_percentReduction_summed;
fig7.pOneSided = p_value_summed;
fig7.summedRep = mean_summed_rep;
fig7.summedRepCi = ci_summed_rep;
fig7.summedEquated = mean_summed_equated;
fig7.summedEquatedCi = ci_summed_equated;
fig7.nBootstraps = nBootstraps;
fig7.nFeatures = nFeatures;
fig7.nParticipants = [n_rep, n_equated];
fig7.nExcluded = length(badSubjects);
fig7.numberOfSetsCompared = length(datasetLabels);
fig7.largestDifferenceFromPaperRun = max(largestDifferenceMaintained, largestDifferenceReduced);
fig7.conceptsInMaintained = sum(~isnan(repWeightedSim));
fig7.conceptsInReduced = sum(~isnan(equatedWeightedSim));
fig7.histogramConcepts = numel(maintainedSimToSet);
fig7.histogramMeanMaintained = mean(maintainedSimToSet);
fig7.histogramSdMaintained = std(maintainedSimToSet);
fig7.histogramMeanReduced = mean(reducedSimToSet);
fig7.histogramSdReduced = std(reducedSimToSet);
end


function drawHistogram(paths, panelName, titleText, summedSimilarity, faceColor, fontSizePt)
% One histogram of Fig. 7b, 7c: the summed similarity of the 1,854 concepts to one set (30 bins,
% proportion of the concepts per bin).
figureHandle = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 560 460]);
histogramHandle = histogram(summedSimilarity, 30, 'Normalization', 'probability', 'FaceColor', faceColor, ...
    'EdgeColor', 'none');
xlabel('Summed similarity', 'FontSize', fontSizePt);
ylabel('Proportion', 'FontSize', fontSizePt);
title(titleText, 'FontSize', fontSizePt - 2);
xlim([0.17 0.19]);
ylim([0 0.11]);
xticks([0.17 0.18 0.19]);
grid on;
box on;
set(gca, 'LineWidth', 2, 'FontSize', fontSizePt - 2, 'TickDir', 'out');
binEdges = histogramHandle.BinEdges(:);
binData = table(binEdges(1:end - 1), binEdges(2:end), histogramHandle.Values(:), ...
    'VariableNames', {'bin_low', 'bin_high', 'proportion_of_concepts'});
save_panel(figureHandle, panelName, paths, binData);
close(figureHandle);
end

function difference = largest_difference(rerun, saved)
% Largest absolute difference between two arrays of the same size (NaN when their sizes differ)
if isequal(size(rerun), size(saved))
    difference = max(abs(rerun(:) - saved(:)), [], 'includenan');
else
    difference = NaN;
end
end
