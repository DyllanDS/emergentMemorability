%% criterion_dprime_da.m: criterion c, d' and d_a in the image-set experiments, and the REM check
% SI Appendix, Section 1 (Measuring Memory, Fig. S1), Methods (Experimental Procedures and Analysis)
% and the main-text statement that the analyses hold with d' or d_a (Simulations of Memorability as
% an Emergent Property).
%
% WHAT IT DOES
% PART 1, the three image-set experiments of Figs. 6 and 7 (602 sessions; the exclusions in Methods
% keep 150 per set, 450 in all). Every concept tested in each image set is scored in three ways from
% that set's kept test trials (a response of 1 to 3 on the six-point scale = "old"):
%   corrected recognition  hit rate minus false alarm rate (the paper's memorability score)
%   d'                     z(hit rate) - z(false alarm rate), each rate kept inside [1/(2n), 1 - 1/(2n)]
%                          with n the concept's number of old (or new) test trials
%   d_a                    sqrt(2 / (1 + s^2)) * (z(H) - s * z(F)) with the z-ROC slope fixed at
%                          s = 0.80 (Didi-Barnea et al. 2016); H and F are the cumulative "old" rates
%                          at the middle confidence boundary (responses 1 to 3)
% A concept score is used when d' and d_a are both finite (2,453 of 2,554). The script then computes
%   the response criterion c = -(z(H) + z(F)) / 2 from the MEAN of the per-concept hit rates and the
%   mean of the per-concept false alarm rates (c = 0.29), and
%   the Pearson correlations of corrected recognition with d' (r = 0.96) and with d_a (r = 0.98),
%   pooled over the three sets and within each set.
% PART 2, the REM simulation of Fig. 1 (the simulation side of the main-text statement that the
% analyses hold with d' or d_a). From the saved per-trial familiarity odds of the Fig. 1 run
% (module 01), for each of the 1,600 simulated items: hit and false alarm rates at the criterion of 1
% (the criterion of the simulation), d', and d_a with the item's own z-ROC slope, fitted over the
% criteria 0.5, 0.75, 1, 1.5 and 2; then the correlations of corrected recognition with d' and with
% d_a across the items.
%
% INPUTS (paths relative to the repository root)
%   06_image_set_experiments_fig6_fig7/data/maintained_trials.csv, changed_trials.csv,
%       reduced_trials.csv (read by shared/matlab/load_image_set_trials.m)
%   shared/things/concept_info_human.mat (the 1,854 THINGS concept names; load_concept_info.m)
%   01_rem_simulation_fig1/data/paper_run/fig1_paper_run.mat
%       (the Fig. 1 run: simHitOdds, simFaOdds, simHitMask, simFaMask)
%   A1_measuring_memory_figS1/data/paper_run/paperStats_criterionDpDa.mat (the saved run that
%       step 4 below compares with)
% OUTPUTS
%   A1_measuring_memory_figS1/outputs/paperStats_criterionDpDa.mat (struct paperStats, MATLAB v7)
%   and a printed summary
% RUN (from the repository root)
%   matlab -batch "restoredefaultpath; cd('A1_measuring_memory_figS1/code'); criterion_dprime_da"
%   or: python A1_measuring_memory_figS1/code/run_module.py (this script, then the Python steps)
% Needs the Statistics and Machine Learning Toolbox (norminv, corr). No random numbers. Runtime about
% half a minute; memory about 2 GB at peak (the REM run holds four 1,600 x 10,000 arrays).

clear; close all;

%% 0. Folders and the shared functions
codeFolder = fileparts(mfilename('fullpath'));
addpath(fullfile(codeFolder, '..', '..', 'shared', 'matlab'));
paths = release_paths(mfilename('fullpath'));
sdt = sdt_rates();                       % the signal-detection formulas (shared/matlab/sdt_rates.m)
fixedDaSlope = 0.8;                      % the z-ROC slope of d_a (sdt.daFromRatingCounts uses 0.8)

%% ========================================================================
%  PART 1: THE THREE IMAGE-SET EXPERIMENTS
%  ========================================================================

%% 1.1 Load the concept names and the test trials of the three image sets
concInfo = load_concept_info('human');
% The loader's dataset names of the three sets (column dataset_name), in the order maintained,
% changed, reduced.
datasetNames = {'representativeImages', 'flipMemorabilityImages', 'equatedImages'};
setNames = {'maintained', 'changed', 'reduced'};
nDatasets = length(datasetNames);
% One row per test trial: subject, is_new (1 = new at test), rt (ms), response (1 = sure old to
% 6 = sure new), concept, dataset_name. The summed-similarity columns are not needed here.
T = load_image_set_trials(setNames, concInfo, false);
fprintf('Loaded %d test trials of %d sessions.\n', height(T), numel(unique(T.subject)));

%% 1.2 Exclusions (Methods), trial filter and the cap of 150 per set
% Participants are excluded for one key on more than 80 percent of trials, more than 20 percent of
% trials under 200 ms or over 7,500 ms, or hit rate minus false alarm rate below 0.05; then trials
% under 200 ms or over 10,000 ms are dropped; then the first 150 participants of each set are kept
% (sorted id order). With the sessions in the repository the cap removes nobody.
fprintf('\n=== Exclusions ===\n');
[T, exclusionReport] = apply_participant_exclusions(T);
for d = 1:nDatasets
    nSub = length(unique(T.subject(T.dataset_name == string(datasetNames{d}))));
    fprintf('  Final %s (%s set): %d subjects\n', datasetNames{d}, setNames{d}, nSub);
end

%% 1.3 Per-concept scores in each set: hit rate, false alarm rate, corrected recognition, d_a
fprintf('\n=== Computing per-concept metrics ===\n');
concepts = concInfo.concepts;
nConcepts = length(concepts);

tByConcept = cell(nDatasets, 1);

for d = 1:nDatasets
    datasetTrials = T(T.dataset_name == string(datasetNames{d}), :);

    for i = 1:nConcepts
        conceptTrials = datasetTrials(strcmp(datasetTrials.concept, concepts{i}), :);
        oldTrials = conceptTrials(conceptTrials.is_new == 0, :);
        newTrials = conceptTrials(conceptTrials.is_new == 1, :);

        tByConcept{d}(i).concept = concepts{i};
        tByConcept{d}(i).dataset = datasetNames{d};

        if ~isempty(oldTrials) && ~isempty(newTrials)
            tByConcept{d}(i).hit_rate = mean(oldTrials.response <= 3);
            tByConcept{d}(i).fa_rate = mean(newTrials.response <= 3);
            tByConcept{d}(i).cr = tByConcept{d}(i).hit_rate - tByConcept{d}(i).fa_rate;

            % counts of the six responses, from "sure old" (1) to "sure new" (6)
            targf = zeros(1, 6);
            luref = zeros(1, 6);
            for r = 1:6
                targf(r) = sum(oldTrials.response == r);
                luref(r) = sum(newTrials.response == r);
            end
            tByConcept{d}(i).targf = targf;
            tByConcept{d}(i).luref = luref;
            % d_a at the fixed slope 0.8 (sdt_rates.m, daFromRatingCounts):
            % NaN unless at least two of the five cumulative boundaries give finite z values
            tByConcept{d}(i).da = sdt.daFromRatingCounts(targf, luref);
        else
            tByConcept{d}(i).hit_rate = NaN;
            tByConcept{d}(i).fa_rate = NaN;
            tByConcept{d}(i).cr = NaN;
            tByConcept{d}(i).targf = nan(1, 6);
            tByConcept{d}(i).luref = nan(1, 6);
            tByConcept{d}(i).da = NaN;
        end
    end
    fprintf('  %s: %d concepts processed\n', datasetNames{d}, nConcepts);
end

%% 1.4 Collapse across sets and compute each concept's d'
allHR  = [];
allFAR = [];
allCR  = [];
allDA  = [];
allDp  = [];
allDataset = {};

for d = 1:nDatasets
    for i = 1:nConcepts
        hr  = tByConcept{d}(i).hit_rate;
        far = tByConcept{d}(i).fa_rate;

        allHR(end+1)  = hr; %#ok<SAGROW>
        allFAR(end+1) = far; %#ok<SAGROW>
        allCR(end+1)  = tByConcept{d}(i).cr; %#ok<SAGROW>
        allDA(end+1)  = tByConcept{d}(i).da; %#ok<SAGROW>
        allDataset{end+1} = tByConcept{d}(i).dataset; %#ok<SAGROW>

        % d' with each rate kept inside [1/(2n), 1 - 1/(2n)]; NaN with fewer than 2 old or new trials
        nTarg = sum(tByConcept{d}(i).targf);
        nLure = sum(tByConcept{d}(i).luref);
        if isnan(hr) || isnan(far) || nTarg < 2 || nLure < 2
            allDp(end+1) = NaN; %#ok<SAGROW>
        else
            allDp(end+1) = sdt.dprimeClamped(hr, far, nTarg, nLure); %#ok<SAGROW>
        end
    end
end

allHR  = allHR(:);
allFAR = allFAR(:);
allCR  = allCR(:);
allDA  = allDA(:);
allDp  = allDp(:);

% concept scores that exist (the concept had old and new test trials in that set)
nComputed = sum(~isnan(allCR));

% keep the scores with a finite d' and a finite d_a (a hit rate of 1 or a false alarm rate of 0 at
% the middle boundary gives an infinite d_a)
validIdx = ~isnan(allCR) & ~isnan(allDA) & ~isnan(allDp) & ~isnan(allHR) & ~isnan(allFAR) ...
         & isfinite(allDA) & isfinite(allDp);
allHR  = allHR(validIdx);
allFAR = allFAR(validIdx);
allCR  = allCR(validIdx);
allDA  = allDA(validIdx);
allDp  = allDp(validIdx);
allDataset = allDataset(validIdx);

itemHitFA = allHR - allFAR;
nUsable = sum(validIdx);

fprintf('\nUsing %d concept-dataset observations (%d of the %d computed dropped for a non-finite d'' or d_a).\n', ...
    nUsable, nComputed - nUsable, nComputed);

%% 1.5 Criterion c from the mean per-concept hit and false alarm rates (overall and per set)
% The mean rates are kept inside [0.5/40, 1 - 0.5/40] before the z transform (no effect here: the
% means are far from 0 and 1).
correctExtreme = @(rate, n) max(0.5/n, min(1 - 0.5/n, rate));
nTrialsEstimate = 40;

overallHR  = mean(allHR);
overallFAR = mean(allFAR);
overallHR_c  = correctExtreme(overallHR, nTrialsEstimate);
overallFAR_c = correctExtreme(overallFAR, nTrialsEstimate);
zHR_all  = norminv(overallHR_c);
zFAR_all = norminv(overallFAR_c);
dprime_all     = zHR_all - zFAR_all;
criterion_c_all = -0.5 * (zHR_all + zFAR_all);

fprintf('\n=== SDT Criterion Analysis ===\n');
fprintf('  Overall: HR=%.4f, FAR=%.4f, d''=%.4f, c=%.4f', ...
    overallHR, overallFAR, dprime_all, criterion_c_all);
if criterion_c_all > 0
    fprintf(' (conservative)\n');
elseif criterion_c_all < 0
    fprintf(' (liberal)\n');
else
    fprintf(' (unbiased)\n');
end

criterionByDataset = nan(nDatasets, 1);
hitRateByDataset = nan(nDatasets, 1);
faRateByDataset = nan(nDatasets, 1);
dprimeByDataset = nan(nDatasets, 1);
for d = 1:nDatasets
    mask = strcmp(allDataset, datasetNames{d});
    dsHR  = mean(allHR(mask));
    dsFAR = mean(allFAR(mask));
    dsHR_c  = correctExtreme(dsHR, nTrialsEstimate);
    dsFAR_c = correctExtreme(dsFAR, nTrialsEstimate);
    zHR  = norminv(dsHR_c);
    zFAR = norminv(dsFAR_c);
    dp = zHR - zFAR;
    cc = -0.5 * (zHR + zFAR);
    criterionByDataset(d) = cc;
    hitRateByDataset(d) = dsHR;
    faRateByDataset(d) = dsFAR;
    dprimeByDataset(d) = dp;
    fprintf('  %s: HR=%.4f, FAR=%.4f, d''=%.4f, c=%.4f', datasetNames{d}, dsHR, dsFAR, dp, cc);
    if cc > 0
        fprintf(' (conservative)\n');
    elseif cc < 0
        fprintf(' (liberal)\n');
    else
        fprintf(' (unbiased)\n');
    end
end

%% 1.6 Correlations of corrected recognition with d' and d_a (pooled and per set)
[r_dp, p_dp] = corr(itemHitFA, allDp, 'rows', 'complete');
[r_da, p_da] = corr(itemHitFA, allDA, 'rows', 'complete');

nDs = nDatasets;
rRow_dp = nan(nDs, 1);  pRow_dp = nan(nDs, 1);
rRow_da = nan(nDs, 1);  pRow_da = nan(nDs, 1);
nByDs = zeros(nDs, 1);
for d = 1:nDs
    mask = strcmp(allDataset, datasetNames{d});
    nByDs(d) = sum(mask);
    x  = itemHitFA(mask);
    yp = allDp(mask);
    ya = allDA(mask);
    if sum(isfinite(x) & isfinite(yp)) >= 2
        [rRow_dp(d), pRow_dp(d)] = corr(x, yp, 'rows', 'complete');
    end
    if sum(isfinite(x) & isfinite(ya)) >= 2
        [rRow_da(d), pRow_da(d)] = corr(x, ya, 'rows', 'complete');
    end
end

fprintf('\n=== Correlation Results (Experimental Data; Pearson r) ===\n');
fprintf("  Hit-FA vs d'  (pooled): r = %.4f, p = %.2e (n = %d)\n", r_dp, p_dp, nUsable);
fprintf('  Hit-FA vs d_a (pooled): r = %.4f, p = %.2e (n = %d)\n', r_da, p_da, nUsable);
for d = 1:nDs
    fprintf("  %s, d':  r = %.4f, p = %.2e (n = %d)\n", datasetNames{d}, rRow_dp(d), pRow_dp(d), nByDs(d));
    fprintf('  %s, d_a: r = %.4f, p = %.2e (n = %d)\n', datasetNames{d}, rRow_da(d), pRow_da(d), nByDs(d));
end

%% ========================================================================
%  PART 2: REM SIMULATION (the Fig. 1 run)
%  ========================================================================

%% 2.1 Load the per-trial familiarity odds of the Fig. 1 run
% The Fig. 1 run of module 01 (data/paper_run/), read by name.
remRunName = 'fig1_paper_run.mat';
remRunFile = fullfile(paths.releaseRoot, '01_rem_simulation_fig1', 'data', 'paper_run', remRunName);
if ~exist(remRunFile, 'file')
    error('criterion_dprime_da:MissingRemRun', ['The Fig. 1 REM run is missing: %s\n' ...
        '(it is part of module 01; see 01_rem_simulation_fig1/README.md).'], remRunFile);
end
fprintf('\n=== REM Simulation Analysis ===\n');
fprintf('Loading: 01_rem_simulation_fig1/data/paper_run/%s\n', remRunName);
loaded = load(remRunFile, 'figureData');
% simHitOdds(i, k): mean likelihood ratio (familiarity odds) of item i when it was tested as a
% studied item in simulation k (simHitMask(i, k) = 1); simFaOdds and simFaMask the same for item i
% tested as a new item (foil).
simHitOdds = loaded.figureData.simHitOdds;
simFaOdds  = loaded.figureData.simFaOdds;
simHitMask = loaded.figureData.simHitMask == 1;
simFaMask  = loaded.figureData.simFaMask == 1;
clear loaded

%% 2.2 The criteria of each item's z-ROC
criteriaValues = [0.5, 0.75, 1, 1.5, 2];
nCriteria = length(criteriaValues);

%% 2.3 Per-item d', d_a and hit rate minus false alarm rate
nItems = size(simHitOdds, 1);
defaultCrit = 1;                       % the criterion of the Fig. 1 simulation

remItemHR    = nan(nItems, 1);
remItemFAR   = nan(nItems, 1);
remItemDp    = nan(nItems, 1);
remItemDa    = nan(nItems, 1);
remItemSlope = nan(nItems, 1);

itemHR_roc  = nan(nItems, nCriteria);
itemFAR_roc = nan(nItems, nCriteria);

for i = 1:nItems
    hitOddsOfItem = simHitOdds(i, simHitMask(i, :));    % the simulations that tested item i as studied
    faOddsOfItem  = simFaOdds(i, simFaMask(i, :));      % the simulations that tested item i as new
    nHit = numel(hitOddsOfItem);
    nFa  = numel(faOddsOfItem);
    if nHit < 2 || nFa < 2
        continue;                      % item left out (NaN); no item of the Fig. 1 run is left out
    end

    hr  = mean(hitOddsOfItem > defaultCrit);
    far = mean(faOddsOfItem > defaultCrit);

    % rates kept inside [1/(2n), 1 - 1/(2n)]
    hr  = min(max(hr,  1/(2*nHit)), 1 - 1/(2*nHit));
    far = min(max(far, 1/(2*nFa)),  1 - 1/(2*nFa));

    remItemHR(i)  = hr;
    remItemFAR(i) = far;
    remItemDp(i)  = norminv(hr) - norminv(far);

    % the item's own ROC over the five criteria, and its z-ROC slope
    for ci = 1:nCriteria
        h = mean(hitOddsOfItem > criteriaValues(ci));
        f = mean(faOddsOfItem  > criteriaValues(ci));
        h = min(max(h, 1/(2*nHit)), 1 - 1/(2*nHit));
        f = min(max(f, 1/(2*nFa)),  1 - 1/(2*nFa));
        itemHR_roc(i, ci) = h;
        itemFAR_roc(i, ci) = f;
    end

    zHR_roc  = norminv(itemHR_roc(i, :));
    zFAR_roc = norminv(itemFAR_roc(i, :));
    p_fit = polyfit(zFAR_roc, zHR_roc, 1);
    s = p_fit(1);
    remItemSlope(i) = s;
    remItemDa(i) = sqrt(2 / (1 + s^2)) * (norminv(hr) - s * norminv(far));
end

remItemHitFA = remItemHR - remItemFAR;
remItemsUsed = sum(~isnan(remItemHitFA));
clear simHitOdds simFaOdds simHitMask simFaMask

%% 2.4 REM correlations (Pearson; Spearman beside them)
[r_dp_rem, p_dp_rem] = corr(remItemHitFA, remItemDp, 'rows', 'complete');
[r_da_rem, p_da_rem] = corr(remItemHitFA, remItemDa, 'rows', 'complete');
rho_dp_rem = corr(remItemHitFA, remItemDp, 'rows', 'complete', 'Type', 'Spearman');
rho_da_rem = corr(remItemHitFA, remItemDa, 'rows', 'complete', 'Type', 'Spearman');

fprintf('\n=== Correlation Results (REM Simulation, %d items) ===\n', remItemsUsed);
fprintf("  Hit-FA vs d':  Pearson r = %.4f, p = %.2e; Spearman rho = %.4f\n", r_dp_rem, p_dp_rem, rho_dp_rem);
fprintf('  Hit-FA vs d_a: Pearson r = %.4f, p = %.2e; Spearman rho = %.4f\n', r_da_rem, p_da_rem, rho_da_rem);
fprintf('  mean hit rate %.4f, mean false alarm rate %.4f, median per-item z-ROC slope %.3f\n', ...
    mean(remItemHR, 'omitnan'), mean(remItemFAR, 'omitnan'), median(remItemSlope, 'omitnan'));

%% 3. Save the statistics
% The statistics, the counts, the per-set values and the REM values.
paperStats = struct();
paperStats.criterion_c = criterion_c_all;
paperStats.dprime_overall = dprime_all;
paperStats.overallHR = overallHR;
paperStats.overallFAR = overallFAR;
paperStats.r_hitFA_dp = r_dp;
paperStats.p_hitFA_dp = p_dp;
paperStats.r_hitFA_da = r_da;
paperStats.p_hitFA_da = p_da;
paperStats.r_hitFA_dp_rem = r_dp_rem;
paperStats.p_hitFA_dp_rem = p_dp_rem;
paperStats.r_hitFA_da_rem = r_da_rem;
paperStats.p_hitFA_da_rem = p_da_rem;
paperStats.daSlope = fixedDaSlope;
paperStats.nSessionsBeforeExclusion = exclusionReport.nBefore;
paperStats.nExcluded = exclusionReport.nExcluded;
paperStats.nKept = exclusionReport.nAfter;
paperStats.nKeptTrials = height(T);
paperStats.nComputed = nComputed;
paperStats.nUsable = nUsable;
paperStats.setNames = setNames;
paperStats.datasetNames = datasetNames;
paperStats.nUsableBySet = nByDs';
paperStats.criterionBySet = criterionByDataset';
paperStats.hitRateBySet = hitRateByDataset';
paperStats.faRateBySet = faRateByDataset';
paperStats.dprimeBySet = dprimeByDataset';
paperStats.r_hitFA_dp_bySet = rRow_dp';
paperStats.r_hitFA_da_bySet = rRow_da';
paperStats.remRunFile = remRunName;
paperStats.remItemsUsed = remItemsUsed;
paperStats.rho_hitFA_dp_rem = rho_dp_rem;
paperStats.rho_hitFA_da_rem = rho_da_rem;
paperStats.remMeanHitRate = mean(remItemHR, 'omitnan');
paperStats.remMeanFaRate = mean(remItemFAR, 'omitnan');
paperStats.remMedianZrocSlope = median(remItemSlope, 'omitnan');

outputFile = fullfile(paths.outputs, 'paperStats_criterionDpDa.mat');
save(outputFile, 'paperStats', '-v7');
fprintf('\nSaved paperStats to A1_measuring_memory_figS1/outputs/paperStats_criterionDpDa.mat\n');

%% 4. Compare with the saved run (data/paper_run/paperStats_criterionDpDa.mat)
% The six image-set values of this run against the saved run; a difference above 1e-9 gives a warning.
saved = load(fullfile(paths.paperRun, 'paperStats_criterionDpDa.mat'));
saved = saved.paperStats;
fprintf('\n=== CHECK against data/paper_run/paperStats_criterionDpDa.mat ===\n');
fprintf('%-16s %16s %16s %12s\n', 'field', 'saved run', 'this run', 'difference');
checkFields = {'criterion_c', 'dprime_overall', 'overallHR', 'overallFAR', 'r_hitFA_dp', 'r_hitFA_da'};
largestDifference = 0;
for k = 1:numel(checkFields)
    name = checkFields{k};
    difference = abs(saved.(name) - paperStats.(name));
    largestDifference = max(largestDifference, difference);
    fprintf('%-16s %16.12f %16.12f %12.2g\n', name, saved.(name), paperStats.(name), difference);
end
if largestDifference < 1e-9
    fprintf('PART 1 reproduces the saved run (largest difference %.2g).\n', largestDifference);
else
    warning('criterion_dprime_da:SavedRunDifference', 'PART 1 differs from the saved run by up to %.3g', largestDifference);
end
fprintf('REM correlations of the Fig. 1 run (%s): r(CR, d'') = %.4f, r(CR, d_a) = %.4f\n', ...
    remRunName, paperStats.r_hitFA_dp_rem, paperStats.r_hitFA_da_rem);

fprintf('\n=== Analysis Complete ===\n');
