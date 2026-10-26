function monkeyVariance = monkey_variance_explained()
%MONKEY_VARIANCE_EXPLAINED  How much of the variance of the monkey responses the 49 components keep.
%
% PAPER
%   Methods, Representational Similarity Analysis (test T53; ledger item L5-048).
%
% WHAT IT DOES
%   The monkey space of Fig. 5d is the first 49 principal components of a PCA of the multi-unit
%   responses of the THINGS Ventral Stream Spiking Dataset (Papale et al., 2025), averaged within
%   each of the 1,854 THINGS concepts. The PCA input is the 576 channels of the IT arrays of the two
%   monkeys (monkey F channels 513 to 832, monkey N channels 769 to 1,024; the PCA script is
%   shared/things/provenance/monkeyFeatGen.m). This script reads the 576 eigenvalues of that PCA
%   (the variance of each component) and computes the percent of the total variance that the first
%   49 components keep:
%       100 x (sum of the first 49 eigenvalues) / (sum of all 576 eigenvalues)
%
% INPUT   shared/things/monkey_pca_explained.csv  (component, eigenvalue, percent_of_variance,
%         cumulative_percent; the eigenvalues of the PCA made by shared/things/provenance/monkeyFeatGen.m)
% OUTPUT  outputs/monkey_variance_explained.csv   (one row: components kept, components in total,
%         percent of variance kept)
%         Return value monkeyVariance: the same numbers, for run_module.m.
% SEEDS: none.
%
% HOW TO RUN (from the repository root; run_module.m calls it)
%   matlab -batch "restoredefaultpath; cd('05_things_reanalysis_fig5/code'); run_module"

codeFolder = fileparts(mfilename('fullpath'));
addpath(fullfile(codeFolder, '..', '..', 'shared', 'matlab'));
paths = release_paths(mfilename('fullpath'));

componentsKept = 49;      % the number of components that form the monkey feature space

pcaTable = readtable(fullfile(paths.sharedThings, 'monkey_pca_explained.csv'));
eigenvalues = pcaTable.eigenvalue;
numberOfComponents = numel(eigenvalues);

% Total variance, and the variance of the first 49 components, added up one component at a time
totalVariance = 0;
for componentIndex = 1:numberOfComponents
    totalVariance = totalVariance + eigenvalues(componentIndex);
end
keptVariance = 0;
for componentIndex = 1:componentsKept
    keptVariance = keptVariance + eigenvalues(componentIndex);
end
percentKept = 100 * keptVariance / totalVariance;

% Check against the cumulative percent column of the same table
if abs(percentKept - pcaTable.cumulative_percent(componentsKept)) > 1e-9
    error('monkey_variance_explained:Mismatch', 'The percent of variance does not equal the table''s cumulative percent.');
end
fprintf('\nMonkey IT PCA: the first %d of %d components keep %.2f percent of the variance (the first component alone %.2f percent)\n', ...
    componentsKept, numberOfComponents, percentKept, 100 * eigenvalues(1) / totalVariance);

resultTable = table(componentsKept, numberOfComponents, percentKept, ...
    'VariableNames', {'components_kept', 'components_in_total', 'percent_of_variance_kept'});
writetable(resultTable, fullfile(paths.outputs, 'monkey_variance_explained.csv'));
fprintf('Saved outputs/monkey_variance_explained.csv\n');

monkeyVariance = struct('componentsKept', componentsKept, 'numberOfComponents', numberOfComponents, ...
    'percentKept', percentKept);
end
