function outputFile = bylinskiiRemSim(numberOfRuns, outputFile, seedValue)
%BYLINSKIIREMSIM  REM simulation of Bylinskii et al.'s (2015) within- and across-category
%   memorability experiments (Fig. 2 of the paper; Methods "Bylinskii et al. (2015) Context
%   Manipulation Simulation").
%
%   What it does
%     Builds 21 categories of 300 simulated items each (75 targets and 225 fillers). Every item is a
%     vector of 49 REM features drawn from a geometric distribution (g = 0.65, values 1, 2, 3, ...).
%     The first 10 features are a category prototype: one draw per category that all 300 members
%     share. Features 11 to 49 are drawn separately for every item.
%     Simulation 1, within category ("AMT 1" in the code, criterion 20): for each category,
%       numberOfRuns study lists of 30 targets and 90 fillers drawn from that category alone.
%     Simulation 2, across categories ("AMT 2" in the code, criterion 5): numberOfRuns study lists of
%       30 targets and 90 fillers drawn from all 21 categories.
%     In every run the 120 studied items are stored as REM traces (each feature stored with
%     probability 0.3; a stored feature is copied correctly with probability 0.6, otherwise it gets
%     a new geometric draw, see EncodeImage). Each studied target is then probed against the stored
%     list: a hit when the mean likelihood ratio over the 120 traces exceeds the criterion. False
%     alarms: 30 new lure items (no category prototype) are generated and stored in the same way,
%     each studied target is probed against those 30 lure traces, and the response counts as a false
%     alarm when the mean likelihood ratio exceeds the criterion.
%     Memorability of a target in a context = its hit rate minus its false alarm rate there.
%     Summed similarity of a target = minus its mean Euclidean distance to the other 299 items of its
%     own category (Simulation 1) or to all other 6,299 items (Simulation 2); higher values mean more
%     similar to the other items, that is, less distinctive.
%     Correlations over the targets tested in both contexts (all 1,575 in the full run), all
%     Spearman; the run also saves the Pearson r of memorability within vs across (rho_pearson):
%       rho_spearman       memorability within vs across                    (Fig. 2a, rho = 0.60)
%       rho_withinMem_sim  memorability within vs summed similarity within  (Fig. 2a, rho = -0.25)
%       rho_acrossMem_sim  memorability across vs summed similarity across  (Fig. 2a, rho = -0.54)
%       rho_sim            summed similarity within vs across               (Fig. 2b, rho = 0.93)
%
%   Inputs (all optional)
%     numberOfRuns  runs per category in Simulation 1 and runs in Simulation 2. Default 1000, the
%                   paper's value. A smaller number runs the same code faster and gives noisier
%                   estimates (run_module('skip_long') uses 10 to check that the code runs).
%     outputFile    where the results are saved. Default:
%                   <module>/outputs/simulation_runs/seed_<seed>_runs_<numberOfRuns>_figureData.mat
%     seedValue     default 47826, the seed of the run the paper reports
%
%   Output
%     outputFile, holding the struct figureData: amt1_valid and amt2_valid (memorability of the
%     targets tested in both contexts), validBoth, summedSim_amt1 and summedSim_amt2 (all 1,575
%     targets, in the order the targets were generated: category 1 first), the correlations and
%     their p values, and the parameters of the run (seed, REM parameters, criteria, category
%     structure, number of runs and the study-list sizes numTargetsPerSim and numFillersPerSim).
%
%   Seed and random draws: rng(seedValue, 'twister') once at the top names the generator, so the
%   stream does not depend on the generator a MATLAB session selected before. Every random draw
%   (geornd, rand, randperm) comes from that one stream in a fixed order; the code runs serially
%   (no parfor). With the defaults (seed 47826, 1,000 runs) the output equals the saved run in
%   data/paper_run/fig2_paper_run.mat in every saved value (per-target memorability and summed
%   similarity, every correlation and every parameter; bylinskii_numbers.m counts two p values as
%   equal when both are below 1e-15).
%   Runtime on an Apple M2 Pro: see the module README.

%% Inputs (defaults: the paper's values)
if nargin < 1 || isempty(numberOfRuns)
    numberOfRuns = 1000;
end
if nargin < 3 || isempty(seedValue)
    seedValue = 47826;
end
codeFolder = fileparts(mfilename('fullpath'));
addpath(fullfile(codeFolder, '..', '..', 'shared', 'matlab'));
paths = release_paths(mfilename('fullpath'));
if nargin < 2 || isempty(outputFile)
    outputFile = fullfile(paths.outputs, 'simulation_runs', ...
        sprintf('seed_%d_runs_%d_figureData.mat', seedValue, numberOfRuns));
end

% one Mersenne Twister stream for every random draw below
rng(seedValue, 'twister');
simulationTimer = tic;    % elapsed time for the progress lines
fprintf('=== Bylinskii et al. (2015) Simulation ===\n');
fprintf('Random seed: %d\n\n', seedValue);

%% REM Model Parameters
baseRateOfFeatures = 0.65;
numFeaturesPerItem = 49;
probEncoding = 0.3;
correctEncodeProb = 0.6;
criterion_amt1 = 20;   % within-category
criterion_amt2 = 5;    % across-category

%% Dataset Structure
numCategories = 21;
targetsPerCategory = 75;
fillersPerCategory = 225;
simLen = numberOfRuns;    % runs per category (within) and in total (across)

totalTargets = numCategories * targetsPerCategory;
totalFillers = numCategories * fillersPerCategory;
categoryFeatures = 10;

fprintf('Dataset: %d categories, %d targets, %d fillers, %d sims\n\n', ...
    numCategories, totalTargets, totalFillers, simLen);

%% Generate Dataset with Category Structure
fprintf('Generating dataset with category structure...\n');

fullDataset = zeros(totalTargets + totalFillers, numFeaturesPerItem);
categoryLabels = zeros(totalTargets + totalFillers, 1);
itemType = zeros(totalTargets + totalFillers, 1);

currentIdx = 1;
for catIdx = 1:numCategories
    categoryPrototype = geornd(baseRateOfFeatures, [1, categoryFeatures]) + 1;
    numItemsInCategory = targetsPerCategory + fillersPerCategory;
    categoryItems = GenImages(numItemsInCategory, baseRateOfFeatures, numFeaturesPerItem);
    categoryItems(:, 1:categoryFeatures) = repmat(categoryPrototype, numItemsInCategory, 1);

    endIdx = currentIdx + numItemsInCategory - 1;
    fullDataset(currentIdx:endIdx, :) = categoryItems;
    categoryLabels(currentIdx:endIdx) = catIdx;
    itemType(currentIdx:currentIdx+targetsPerCategory-1) = 1;
    currentIdx = endIdx + 1;
end

targetMask = itemType == 1;
targets = fullDataset(targetMask, :);
fillers = fullDataset(~targetMask, :);
targetCategories = categoryLabels(targetMask);
fillerCategories = categoryLabels(~targetMask);

fprintf('Dataset generated: %d targets, %d fillers\n\n', size(targets,1), size(fillers,1));

%% AMT 1: Within-Category Memorability
fprintf('=== AMT 1: Within-Category Memorability ===\n');

amt1_hitRates = zeros(totalTargets, 1);
amt1_faRates = zeros(totalTargets, 1);
amt1_hitCounts = zeros(totalTargets, 1);
amt1_faCounts = zeros(totalTargets, 1);

numTargetsPerSim = 30;
numFillersPerSim = 90;

for catIdx = 1:numCategories
    fprintf('Testing category %d/%d... (%.0f s elapsed)\n', catIdx, numCategories, toc(simulationTimer));

    catTargetIdxs = find(targetCategories == catIdx);
    catFillerIdxs = find(fillerCategories == catIdx);
    catTargets = targets(catTargetIdxs, :);
    catFillers = fillers(catFillerIdxs, :);

    for nSim = 1:simLen
        targetSample = randperm(length(catTargetIdxs), min(numTargetsPerSim, length(catTargetIdxs)));
        fillerSample = randperm(length(catFillerIdxs), min(numFillersPerSim, length(catFillerIdxs)));

        selectedTargets = catTargets(targetSample, :);
        selectedFillers = catFillers(fillerSample, :);
        memorySet = [selectedTargets; selectedFillers];
        encodedSet = EncodeImage(memorySet, baseRateOfFeatures, probEncoding, correctEncodeProb);

        % Test hits
        for tIdx = 1:size(selectedTargets, 1)
            globalTargetIdx = catTargetIdxs(targetSample(tIdx));
            lambdas = zeros(size(memorySet, 1), 1);
            for mIdx = 1:size(memorySet, 1)
                lambdas(mIdx) = ProbeImage(selectedTargets(tIdx,:), encodedSet(mIdx,:), ...
                    baseRateOfFeatures, numFeaturesPerItem, correctEncodeProb);
            end
            isHit = (mean(lambdas) > criterion_amt1);
            amt1_hitRates(globalTargetIdx) = amt1_hitRates(globalTargetIdx) + isHit;
            amt1_hitCounts(globalTargetIdx) = amt1_hitCounts(globalTargetIdx) + 1;
        end

        % Test false alarms
        newLures = GenImages(numTargetsPerSim, baseRateOfFeatures, numFeaturesPerItem);
        encodedLures = EncodeImage(newLures, baseRateOfFeatures, probEncoding, correctEncodeProb);

        for tIdx = 1:size(selectedTargets, 1)
            globalTargetIdx = catTargetIdxs(targetSample(tIdx));
            lambdas = zeros(size(encodedLures, 1), 1);
            for mIdx = 1:size(encodedLures, 1)
                lambdas(mIdx) = ProbeImage(selectedTargets(tIdx,:), encodedLures(mIdx,:), ...
                    baseRateOfFeatures, numFeaturesPerItem, correctEncodeProb);
            end
            isFA = (mean(lambdas) > criterion_amt1);
            amt1_faRates(globalTargetIdx) = amt1_faRates(globalTargetIdx) + isFA;
            amt1_faCounts(globalTargetIdx) = amt1_faCounts(globalTargetIdx) + 1;
        end
    end
end

validHits = amt1_hitCounts > 0;
validFAs = amt1_faCounts > 0;
amt1_hitRates(validHits) = amt1_hitRates(validHits) ./ amt1_hitCounts(validHits);
amt1_faRates(validFAs) = amt1_faRates(validFAs) ./ amt1_faCounts(validFAs);
amt1_memorability = amt1_hitRates - amt1_faRates;

fprintf('AMT 1 complete. Mean HR: %.2f%%, Mean FAR: %.2f%%\n\n', ...
    100*mean(amt1_hitRates(validHits)), 100*mean(amt1_faRates(validFAs)));

%% AMT 2: Across-Category Memorability
fprintf('=== AMT 2: Across-Category Memorability ===\n');

amt2_hitRates = zeros(totalTargets, 1);
amt2_faRates = zeros(totalTargets, 1);
amt2_hitCounts = zeros(totalTargets, 1);
amt2_faCounts = zeros(totalTargets, 1);

for nSim = 1:simLen
    if mod(nSim, simLen/10) == 0
        fprintf('Progress: %d%% (%.0f s elapsed)\n', round(100 * nSim / simLen), toc(simulationTimer));
    end

    targetSample = randperm(totalTargets, numTargetsPerSim);
    fillerSample = randperm(totalFillers, numFillersPerSim);

    selectedTargets = targets(targetSample, :);
    selectedFillers = fillers(fillerSample, :);
    memorySet = [selectedTargets; selectedFillers];
    encodedSet = EncodeImage(memorySet, baseRateOfFeatures, probEncoding, correctEncodeProb);

    % Test hits
    for tIdx = 1:size(selectedTargets, 1)
        globalTargetIdx = targetSample(tIdx);
        lambdas = zeros(size(memorySet, 1), 1);
        for mIdx = 1:size(memorySet, 1)
            lambdas(mIdx) = ProbeImage(selectedTargets(tIdx,:), encodedSet(mIdx,:), ...
                baseRateOfFeatures, numFeaturesPerItem, correctEncodeProb);
        end
        isHit = (mean(lambdas) > criterion_amt2);
        amt2_hitRates(globalTargetIdx) = amt2_hitRates(globalTargetIdx) + isHit;
        amt2_hitCounts(globalTargetIdx) = amt2_hitCounts(globalTargetIdx) + 1;
    end

    % Test false alarms
    newLures = GenImages(numTargetsPerSim, baseRateOfFeatures, numFeaturesPerItem);
    encodedLures = EncodeImage(newLures, baseRateOfFeatures, probEncoding, correctEncodeProb);

    for tIdx = 1:size(selectedTargets, 1)
        globalTargetIdx = targetSample(tIdx);
        lambdas = zeros(size(encodedLures, 1), 1);
        for mIdx = 1:size(encodedLures, 1)
            lambdas(mIdx) = ProbeImage(selectedTargets(tIdx,:), encodedLures(mIdx,:), ...
                baseRateOfFeatures, numFeaturesPerItem, correctEncodeProb);
        end
        isFA = (mean(lambdas) > criterion_amt2);
        amt2_faRates(globalTargetIdx) = amt2_faRates(globalTargetIdx) + isFA;
        amt2_faCounts(globalTargetIdx) = amt2_faCounts(globalTargetIdx) + 1;
    end
end

validHits = amt2_hitCounts > 0;
validFAs = amt2_faCounts > 0;
amt2_hitRates(validHits) = amt2_hitRates(validHits) ./ amt2_hitCounts(validHits);
amt2_faRates(validFAs) = amt2_faRates(validFAs) ./ amt2_faCounts(validFAs);
amt2_memorability = amt2_hitRates - amt2_faRates;

fprintf('AMT 2 complete. Mean HR: %.2f%%, Mean FAR: %.2f%%\n\n', ...
    100*mean(amt2_hitRates(validHits)), 100*mean(amt2_faRates(validFAs)));

%% Analysis
validBoth = (amt1_hitCounts > 0) & (amt1_faCounts > 0) & ...
            (amt2_hitCounts > 0) & (amt2_faCounts > 0);

amt1_valid = amt1_memorability(validBoth);
amt2_valid = amt2_memorability(validBoth);

[rho_spearman, p_spearman] = corr(amt1_valid, amt2_valid, 'Type', 'Spearman');
[rho_pearson, p_pearson] = corr(amt1_valid, amt2_valid, 'Type', 'Pearson');

fprintf('Items tested in both: %d\n', sum(validBoth));
fprintf('Memorability consistency: Spearman rho = %.3f (p = %.3e), Pearson r = %.3f (p = %.3e)\n\n', ...
    rho_spearman, p_spearman, rho_pearson, p_pearson);

%% Summed Similarity
fprintf('Computing summed similarity... (%.0f s elapsed)\n', toc(simulationTimer));

summedSim_amt1 = zeros(totalTargets, 1);
summedSim_amt2 = zeros(totalTargets, 1);

fprintf('AMT1 (within-category)...\n');
for tIdx = 1:totalTargets
    targetCat = targetCategories(tIdx);
    targetVec = targets(tIdx, :);

    sameCatTargets = (targetCategories == targetCat) & (1:totalTargets)' ~= tIdx;
    sameCatFillers = fillerCategories == targetCat;

    distances = [];
    for otherIdx = find(sameCatTargets)'
        distances = [distances; norm(targetVec - targets(otherIdx,:))];
    end
    catFillers = fillers(sameCatFillers, :);
    for fIdx = 1:size(catFillers, 1)
        distances = [distances; norm(targetVec - catFillers(fIdx,:))];
    end
    summedSim_amt1(tIdx) = mean(distances);
end

fprintf('AMT2 (across-category)... (%.0f s elapsed)\n', toc(simulationTimer));
for tIdx = 1:totalTargets
    % progress line every 315 targets
    if mod(tIdx, 315) == 0
        fprintf('  %d of %d targets (%.0f s elapsed)\n', tIdx, totalTargets, toc(simulationTimer));
    end
    targetVec = targets(tIdx, :);
    distances = [];
    for otherIdx = 1:totalTargets
        if otherIdx ~= tIdx
            distances = [distances; norm(targetVec - targets(otherIdx,:))];
        end
    end
    for fIdx = 1:size(fillers, 1)
        distances = [distances; norm(targetVec - fillers(fIdx,:))];
    end
    summedSim_amt2(tIdx) = mean(distances);
end

% Convert distance to similarity (negate so higher = more similar)
summedSim_amt1 = -summedSim_amt1;
summedSim_amt2 = -summedSim_amt2;

[rho_sim, p_sim] = corr(summedSim_amt1(validBoth), summedSim_amt2(validBoth), 'Type', 'Spearman');
fprintf('Summed similarity consistency: Spearman rho = %.3f (p = %.3e)\n\n', rho_sim, p_sim);

%% Within/across memorability vs summed similarity correlations
[rho_withinMem_sim, p_withinMem_sim] = corr(amt1_memorability(validBoth), summedSim_amt1(validBoth), 'Type', 'Spearman');
[rho_acrossMem_sim, p_acrossMem_sim] = corr(amt2_memorability(validBoth), summedSim_amt2(validBoth), 'Type', 'Spearman');
fprintf('Within-category memorability vs summed similarity: Spearman rho = %.3f (p = %.3e)\n', rho_withinMem_sim, p_withinMem_sim);
fprintf('Across-category memorability vs summed similarity: Spearman rho = %.3f (p = %.3e)\n\n', rho_acrossMem_sim, p_acrossMem_sim);

%% Save figureData to outputFile
outDir = fileparts(outputFile);
if ~exist(outDir, 'dir'), mkdir(outDir); end

figureData = struct();
figureData.amt1_valid = amt1_valid;
figureData.amt2_valid = amt2_valid;
figureData.validBoth = validBoth;
figureData.summedSim_amt1 = summedSim_amt1;
figureData.summedSim_amt2 = summedSim_amt2;
figureData.rho_spearman = rho_spearman;
figureData.p_spearman = p_spearman;
figureData.rho_pearson = rho_pearson;
figureData.p_pearson = p_pearson;
figureData.rho_sim = rho_sim;
figureData.p_sim = p_sim;
figureData.rho_withinMem_sim = rho_withinMem_sim;
figureData.p_withinMem_sim = p_withinMem_sim;
figureData.rho_acrossMem_sim = rho_acrossMem_sim;
figureData.p_acrossMem_sim = p_acrossMem_sim;

figureData.parameters.seedValue = seedValue;
figureData.parameters.baseRateOfFeatures = baseRateOfFeatures;
figureData.parameters.numFeaturesPerItem = numFeaturesPerItem;
figureData.parameters.probEncoding = probEncoding;
figureData.parameters.correctEncodeProb = correctEncodeProb;
figureData.parameters.criterion_amt1 = criterion_amt1;
figureData.parameters.criterion_amt2 = criterion_amt2;
figureData.parameters.numCategories = numCategories;
figureData.parameters.targetsPerCategory = targetsPerCategory;
figureData.parameters.fillersPerCategory = fillersPerCategory;
figureData.parameters.simLen = simLen;
figureData.parameters.categoryFeatures = categoryFeatures;
% the study-list sizes
figureData.parameters.numTargetsPerSim = numTargetsPerSim;
figureData.parameters.numFillersPerSim = numFillersPerSim;

save(outputFile, 'figureData');
fprintf('Saved figureData to %s (%.0f s elapsed)\n', relativeToModule(outputFile, paths.module), ...
    toc(simulationTimer));
end

%% ========== Local Functions ==========

function image_shown = GenImages(nImages, g, w)
    % nImages items with w features each; every value is a geometric draw (parameter g) plus 1, so
    % values are 1, 2, 3, ...
    image_shown = geornd(g, [nImages, w]) + 1;
end

function image_encoded = EncodeImage(image_shown, g, u, c)
    % REM storage of a list. Each feature is stored with probability u; a stored feature is copied
    % correctly with probability c, otherwise it gets a new geometric draw (parameter g). The new
    % draw has no "+ 1", so it can be 0, which then counts as "not stored". Features that are not
    % stored are 0.
    featureEncodedAtAll = rand(size(image_shown)) < u;
    featureEncodedCorrectly = rand(size(image_shown)) < c;
    numWrongEncoded = sum(featureEncodedCorrectly(:) == 0);

    image_encoded = image_shown;
    image_encoded(featureEncodedCorrectly == 0) = geornd(g, [1 numWrongEncoded]);
    image_encoded(featureEncodedAtAll == 0) = 0;
end

function lambda = ProbeImage(imgShown, encodedimage, g, w, c)
    % Likelihood ratio of one memory trace for one probe item (REM). Every stored feature that does
    % not match the probe multiplies the ratio by (1 - c); every matching feature multiplies it by
    % the diagnosticity of its value (rare values count more).
    M = find(imgShown == encodedimage);
    MM = sum(imgShown(encodedimage > 0) ~= encodedimage(encodedimage > 0));
    lambda = (1 - c) ^ MM;

    for idx = 1:length(M)
        if M(idx) <= w
            lambda = lambda * oddsOfThisParticularMatch(encodedimage(M(idx)), c, g);
        end
    end
end

function om = oddsOfThisParticularMatch(v_kj, c, g)
    % How much more likely a match on the value v_kj is when the trace was stored from the probe
    % item itself (c + (1 - c) P(v_kj)) than when it was stored from another item (P(v_kj)); P is
    % the geometric probability of the value, g (1 - g)^(v_kj - 1).
    r = v_kj - 1;
    probThisMatchIfWeReallySawIt = c + (1 - c) * geopdf(r, g);
    probThisMatchIfWeDidntSeeIt = geopdf(r, g);
    om = probThisMatchIfWeReallySawIt / probThisMatchIfWeDidntSeeIt;
end

function text = relativeToModule(fullPath, moduleFolder)
% a path relative to the module folder, for the progress lines
if startsWith(fullPath, moduleFolder)
    text = fullPath(numel(moduleFolder) + 2:end);
else
    [~, name, extension] = fileparts(fullPath);
    text = [name extension];
end
end
