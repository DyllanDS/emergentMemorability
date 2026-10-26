function run_module(numberOfSimulations)
%RUN_MODULE  Module 01 (Fig. 1): the REM simulation of memorability, from the simulation to the
%   figure and the numbers.
%
%   From the repository root:
%       matlab -batch "restoredefaultpath; cd('01_rem_simulation_fig1/code'); run_module"
%   then the uncertainty statistics (Python), which also add the rows of the two correlations of
%   Fig. 1 to outputs/numbers.csv:
%       python 01_rem_simulation_fig1/code/stats_fig1.py
%
%   What it does
%     1. Reruns the simulation (fig1_remSim.m: seed 12345, 10,000 simulated experiments) and
%        compares every array of the saved run in data/paper_run/ with the same array of the rerun
%        (outputs/rerun_vs_paper_run.csv). The item set is drawn with seed 12345 and every
%        simulated experiment draws from its own random stream. Five of the stored arrays are p
%        values (or built from p values) that MATLAB's corr computes with tcdf.m; their last
%        digits can differ between MATLAB installations.
%     2. Draws Fig. 1 from the rerun (fig1_makeFigure1.m).
%     3. Writes outputs/numbers.csv and outputs/summed_similarity_measures.csv (fig1_numbers.m).
%
%   Input (optional)
%     numberOfSimulations  simulated experiments (default 10000, as in the paper). A smaller number
%                          only tests the code quickly, for example run_module(500): the numbers
%                          then do not match the paper, but the first simulations must still equal
%                          the first simulations of the saved run, and step 1 checks that. A test
%                          run writes over the files in outputs/; run both steps again afterwards
%                          (or test in a copy of the module).
%
%   Outputs (in 01_rem_simulation_fig1/outputs/)
%     rerun_vs_paper_run.csv           each array of the saved run: what it depends on, identical
%                                      in the rerun or not, largest difference
%     figures/fig1_rem_simulation.png and figure_data/fig1*.csv
%     numbers.csv, summed_similarity_measures.csv
%     logs/run_module.log              the messages of this run
%
%   Runtime on an Apple M2 Pro laptop (serial, no parallel pool): about 2.5 minutes. Memory: about
%   2.5 GB at the peak, when the rerun and the saved run are compared.

if nargin < 1 || isempty(numberOfSimulations)
    numberOfSimulations = 10000;
end

addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared', 'matlab'));
paths = release_paths(mfilename('fullpath'));

logFile = fullfile(paths.logs, 'run_module.log');
if exist(logFile, 'file')
    delete(logFile);
end
diary(logFile);

paperSeed = 12345;
paperRunFile = fullfile(paths.paperRun, 'fig1_paper_run.mat');
isTestRun = numberOfSimulations ~= 10000;

fprintf('Module 01 (Fig. 1, REM simulation of memorability), run of %s\n', datestr(now, 'yyyy-mm-dd'));
fprintf('MATLAB %s on %s\n', version, computer);
if isTestRun
    fprintf(['TEST RUN with %d simulated experiments (the paper uses 10,000): the numbers will not match the paper, ' ...
        'and the files of this run replace those in outputs/ (run run_module and stats_fig1.py again afterwards).\n'], numberOfSimulations);
end
moduleTimer = tic;

%% 1. Rerun with seed 12345 and compare it with the saved run in data/paper_run/
fprintf('\n1. Rerun of the simulation (seed %d)\n', paperSeed);
rerunTimer = tic;
rerun = fig1_remSim(paperSeed, '', numberOfSimulations);
fprintf('rerun took %.0f s\n', toc(rerunTimer));
rerunMeasures = fig1_measures(rerun);

loaded = load(paperRunFile, 'figureData');
paperRun = loaded.figureData;
clear loaded
paperRunMeasures = fig1_measures(paperRun);
comparison = compareRuns(paperRun, rerun);
clear paperRun
writetable(comparison, fullfile(paths.outputs, 'rerun_vs_paper_run.csv'));

% Which tcdf.m MATLAB uses (described in words, so the log carries no folder of this computer)
tcdfInUse = which('tcdf');
if startsWith(tcdfInUse, matlabroot)
    tcdfDescription = 'the Statistics and Machine Learning Toolbox''s tcdf.m';
else
    tcdfDescription = 'a tcdf.m outside the MATLAB installation, ahead of the toolbox''s on the MATLAB path';
end

dependsOnSimulationOnly = strcmp(comparison.depends_on, 'the simulation');
numberOfSimulationArrays = sum(dependsOnSimulationOnly);
numberOfPValueArrays = sum(~dependsOnSimulationOnly);
simulationArraysIdentical = comparison.identical(dependsOnSimulationOnly);
pValueArraysIdentical = comparison.identical(~dependsOnSimulationOnly);
rerunEqualsPaperRun = ~isTestRun && all(simulationArraysIdentical);

if isTestRun
    % Only the item set and the simulations both runs have can be compared (see the note column)
    comparableRows = (contains(comparison.note, 'first') | contains(comparison.note, 'does not depend')) & ...
        dependsOnSimulationOnly;
    firstSimulationsIdentical = all(comparison.identical(comparableRows));
    fprintf(['TEST RUN: the item set and the %d simulated experiments of this run (list draws, ' ...
        'likelihood ratios, test masks), compared with the first %d of the saved run: ' ...
        'identical = %s\n'], numberOfSimulations, numberOfSimulations, mat2str(firstSimulationsIdentical));
elseif rerunEqualsPaperRun
    fprintf('The rerun equals the saved run: all %d arrays of the simulation are identical (outputs/rerun_vs_paper_run.csv).\n', ...
        numberOfSimulationArrays);
else
    fprintf('WARNING: the rerun differs from the saved run in %d of the %d arrays of the simulation (outputs/rerun_vs_paper_run.csv).\n', ...
        sum(~simulationArraysIdentical), numberOfSimulationArrays);
end
if ~isTestRun
    if all(pValueArraysIdentical)
        fprintf('The %d arrays of p values (or built from p values) are identical too (tcdf in use: %s).\n', ...
            numberOfPValueArrays, tcdfDescription);
    else
        largestRelativeDifference = max(comparison.largest_relative_difference(~dependsOnSimulationOnly));
        fprintf(['%d of the %d arrays of p values differ from the saved run in their last digits, largest ' ...
            'relative difference %.2g over the nonzero saved values ' ...
            '(tcdf in use: %s).\n'], ...
            sum(~pValueArraysIdentical), numberOfPValueArrays, largestRelativeDifference, tcdfDescription);
    end
end

%% 2. Fig. 1 from the rerun
fprintf('\n2. Fig. 1\n');
fig1_makeFigure1(rerun, sprintf('rerun with seed %d', paperSeed));
clear rerun

%% 3. The numbers of the paper
fprintf('\n3. Numbers\n');
fig1_numbers(rerunMeasures, paperRunMeasures, rerunEqualsPaperRun);

fprintf('\nSummary (rerun with seed %d)\n', paperSeed);
fprintf('  Fig. 1b: Pearson r, memorability vs similarity (1 / mean distance) = %.4f (paper: r = -0.56)\n', ...
    rerunMeasures.pearsonMemorabilityVsInverseMeanDistance);
fprintf('  Fig. 1d: Spearman rho across the 49 features = %.4f (paper: rho = -0.44)\n', rerunMeasures.spearmanAcrossFeatures);
fprintf('  identical to the saved run in data/paper_run/: %s\n', mat2str(rerunEqualsPaperRun));
fprintf('Module 01 finished in %.0f s. Next: python 01_rem_simulation_fig1/code/stats_fig1.py (from the repository root)\n', ...
    toc(moduleTimer));
diary off
end

function comparison = compareRuns(paperRun, rerun)
% Each array of the saved run in data/paper_run/, against the same array of the rerun.
%   depends_on: 'the simulation' (set by the seeded simulation alone), or 'the simulation
%   and tcdf.m' for the p values that MATLAB's corr computes with tcdf.m (the Pearson p of Fig. 1b
%   and the Spearman p of each feature, whose values have ties) and the list of features
%   significant for both memorability and similarity, which is built from those p values.
%   In a test run with fewer simulations, the four per-simulation arrays (one column per simulated
%   experiment) and the item set are compared over the simulations both runs have; every other
%   array depends on all 10,000 simulations and is not comparable.
arraysUsingTcdf = {'pval_sim', 'featurePvalsMemory', 'featurePvalsDist', 'featurePvalsSimil', 'bothSig'};
perSimulationArrays = {'simHitOdds', 'simFaOdds', 'simHitMask', 'simFaMask'};
itemSetArrays = {'imagesOfInterest', 'pairwiseDistances', 'avgDistances', 'similarity', ...
    'mostIsolatedIdx', 'mostTypicalIdx', 'seedValue', 'baseRateOfFeatures', 'numFeaturesPerItem', ...
    'probEncoding', 'correctEncodeProb', 'criterion', 'featureLabels', 'featureCorrsDist', 'featurePvalsDist', ...
    'featureCorrsSimil', 'featurePvalsSimil'};
numberOfSimulationsInRerun = size(rerun.simHitMask, 2);
numberOfSimulationsInPaperRun = size(paperRun.simHitMask, 2);
isTestRun = numberOfSimulationsInRerun < numberOfSimulationsInPaperRun;

fieldNames = fieldnames(paperRun);
numberOfFields = numel(fieldNames);
arraySize = cell(numberOfFields, 1);
dependsOn = cell(numberOfFields, 1);
identical = false(numberOfFields, 1);
largestDifference = nan(numberOfFields, 1);
largestRelativeDifference = nan(numberOfFields, 1);
note = repmat({''}, numberOfFields, 1);
for fieldIndex = 1:numberOfFields
    name = fieldNames{fieldIndex};
    savedValue = paperRun.(name);
    rerunValue = rerun.(name);
    arraySize{fieldIndex} = sprintf('%d x %d', size(savedValue, 1), size(savedValue, 2));
    if any(strcmp(name, arraysUsingTcdf))
        dependsOn{fieldIndex} = 'the simulation and tcdf.m';
    else
        dependsOn{fieldIndex} = 'the simulation';
    end

    % In a test run, keep only the simulations both runs have
    if isTestRun && any(strcmp(name, perSimulationArrays))
        savedValue = savedValue(:, 1:numberOfSimulationsInRerun);
        note{fieldIndex} = sprintf('first %d of %d simulations compared (test run)', ...
            numberOfSimulationsInRerun, numberOfSimulationsInPaperRun);
    elseif isTestRun && any(strcmp(name, itemSetArrays))
        note{fieldIndex} = 'does not depend on the number of simulations (test run)';
    elseif isTestRun
        note{fieldIndex} = 'depends on all simulations; not comparable in a test run';
    end

    if iscell(savedValue)
        % the feature labels: text, compared as text
        identical(fieldIndex) = isequal(savedValue, rerunValue);
        continue
    end
    savedNumbers = double(savedValue);
    rerunNumbers = double(rerunValue);
    if ~isequal(size(savedNumbers), size(rerunNumbers))
        continue
    end
    identical(fieldIndex) = isequaln(savedNumbers, rerunNumbers);
    absoluteDifference = abs(savedNumbers(:) - rerunNumbers(:));
    largestDifference(fieldIndex) = max(absoluteDifference);
    % relative difference over the nonzero saved values (left empty when every saved value is 0
    % and the rerun differs, as for the Fig. 1b p value, which the saved run stores as 0)
    nonzeroSaved = savedNumbers(:) ~= 0;
    if identical(fieldIndex)
        largestRelativeDifference(fieldIndex) = 0;
    elseif any(nonzeroSaved)
        largestRelativeDifference(fieldIndex) = max(absoluteDifference(nonzeroSaved) ./ abs(savedNumbers(nonzeroSaved)));
    end
end
comparison = table(fieldNames, arraySize, dependsOn, identical, largestDifference, largestRelativeDifference, note, ...
    'VariableNames', {'stored_array', 'size_in_paper_run', 'depends_on', 'identical', ...
    'largest_absolute_difference', 'largest_relative_difference', 'note'});
end
