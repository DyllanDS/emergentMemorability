function continuous_rem_spose(seed)
% continuous_rem_spose.m: REM written for continuous feature values and run in the measured SPoSE
% space of 1,600 THINGS concepts (SI Appendix, Section 2, Measured Representations, Fig. S6).
%
% continuous_rem_spose or continuous_rem_spose(1)
%   runs the simulation with rng(1, 'twister'), the run the paper reports, and writes its results to
%   this module's outputs/ folder. The optional argument is the seed of rng (default 1).
%
% WHAT IT DOES
%   Each simulated item is one THINGS concept, and its 49 features are that concept's 49 SPoSE values
%   (Hebart et al., 2020). The model keeps REM's likelihood ratio and changes two things, because the
%   SPoSE values are continuous numbers:
%     - the rarity of a feature value is how common that value is among the 1,600 concepts (the
%       distribution of that SPoSE dimension, estimated with ksdensity), in place of REM's geometric
%       base rate;
%     - a correctly stored feature keeps the studied value plus Gaussian retention noise (standard
%       deviation = a fraction of that dimension's standard deviation across the 1,600 concepts), and a
%       probe value counts as evidence for a trace according to how close it falls to the stored value
%       under that noise.
%   For a probe value x and a stored trace value t on feature j, the likelihood ratio is
%       lambda_j = [ c * N(t - x; sigma_j) + (1 - c) * f_j(t) ] / f_j(t)
%                = 1 + c * ( N(t - x; sigma_j) / f_j(t) - 1 )
%   where c is the copy probability, N the normal density of the retention noise and f_j(t) how
%   common the value t is among all concepts. A trace's likelihood ratio is the product of lambda_j
%   over the features that trace stored, familiarity is the mean of that product over the traces of
%   the study list, and the model responds "old" when familiarity is above 1.0, as in REM.
%
%   Two analyses, each at four retention-noise levels (0.02, 0.03, 0.05 and 0.08 of each dimension's
%   standard deviation; Fig. S6 shows the first three):
%     1. Between items. 2,500 simulated study-test experiments. Each studies 40 concepts and
%        tests 20 of them plus 20 unstudied concepts, all drawn at random from the 1,600. A concept's
%        memorability is its hit rate minus its false alarm rate over the experiments. Its crowding is
%        minus its mean correlation distance (1 minus the Pearson correlation of two SPoSE vectors) to
%        its 20 nearest concepts. The script reports the Spearman correlation of the two.
%     2. Designed lists (Fig. S6). 120 critical concepts, each studied on one list with 0 and on one
%        list with 16 close neighbors (its 2nd to 17th nearest concepts). Its nearest concept is
%        never studied and is tested as the lure. The rest of each 40-item list is drawn from the
%        farther half of the concepts. 100 simulated participants study each list. The d′ change is
%        d′ with 16 neighbors minus d′ with none, each computed from the mean hit and false alarm
%        rates across the 120 critical concepts.
%   check_against_paper_run.py compares the report with the saved report in data/paper_run/.
%
% READS
%   shared/things/spose_embedding_49d_sorted.txt   1,854 THINGS concepts x 49 SPoSE dimensions
%                                                  (Hebart et al., 2020; CC BY 4.0)
%   shared/things/unique_id.txt                    the concept names, in the same row order
% WRITES (this module's outputs/ folder)
%   continuous_rem_spose.txt                    the report: the settings, the results table at each
%                                               noise level and closing notes
%   concept_subset.csv                          the 1,600 concepts drawn after rng(1): item_number (row
%                                               of the item files below), concept_row_in_embedding,
%                                               concept
%   continuous_rem_settings.csv                 every setting as run
%   continuous_rem_items_sigma<noise>.csv       between items, one row per concept: concept row, hit
%                                               count, number of old tests, false alarm count, number
%                                               of new tests, local crowding (20 nearest), global
%                                               crowding (all other concepts)
%   continuous_rem_designed_sigma<noise>.csv    designed lists, one row per critical concept: concept
%                                               row, hit and false alarm rates with 0 and with 16
%                                               close neighbors on the list
% SEED
%   rng(1, 'twister'), set once (the generator MATLAB starts with); the choice of the 1,600 concepts
%   is the first random draw. Then, for each noise level in turn, the between-item experiments and
%   then the designed lists draw their numbers in the order of the code below. Run serially.
% RUN (from the repository root; MATLAB R2023b with the Statistics and Machine Learning Toolbox; about
%      1 minute; no parallel pool)
%   matlab -batch "restoredefaultpath; cd('A2e_measured_representations_figS6/code'); continuous_rem_spose"
%   or: python A2e_measured_representations_figS6/code/run_module.py
%   (restoredefaultpath, for that MATLAB session only, removes folders added to the MATLAB path, so
%   that no other toolbox can replace a MATLAB function used here)

if nargin < 1
    seed = 1;                   % the run the paper reports
end

% ---- warn when a folder on the MATLAB path replaces a function this script uses ----
% Looking a function up draws no random numbers.
functionsUsed = {'randperm', 'rand', 'randn', 'randi', 'ksdensity', 'pdist', 'squareform', ...
                 'fillmissing', 'interp1', 'corr', 'norminv', 'std', 'writetable', 'readlines'};
for functionIndex = 1:numel(functionsUsed)
    functionFile = which(functionsUsed{functionIndex});
    if isempty(functionFile)
        error('continuous_rem_spose:missingFunction', ...
              '%s was not found; this script needs the Statistics and Machine Learning Toolbox', ...
              functionsUsed{functionIndex});
    end
    if ~startsWith(functionFile, matlabroot) && ~startsWith(functionFile, 'built-in')
        warning('continuous_rem_spose:shadowedFunction', ...
                ['%s is taken from %s, not from MATLAB''s own toolboxes; ' ...
                 'run restoredefaultpath first to reproduce the run the paper reports'], ...
                functionsUsed{functionIndex}, functionFile);
    end
end

% ---- the repository's folders (shared/matlab/release_paths.m) ----
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared', 'matlab'));
releasePaths = release_paths(mfilename('fullpath'));
outputFolder = releasePaths.outputs;
fprintf('MATLAB %s; seed %d\n', version, seed);

rng(seed, 'twister');           % seed 1: the run the paper reports (MATLAB's default Mersenne twister)

% ---- every setting of the simulation ----
settings = struct();
settings.nConceptsUsed = 1600;              % concepts drawn at random from the 1,854 THINGS concepts
settings.storage = 0.30;                    % u: probability that a feature is stored at all
settings.copy = 0.60;                       % c: probability that a stored feature is copied correctly
settings.nStudied = 40;                     % concepts on each study list
settings.nTestedOld = 20;                   % studied concepts tested in each experiment
settings.nTestedNew = 20;                   % unstudied concepts tested in each experiment
settings.criterion = 1.0;                   % respond "old" when familiarity is above 1.0
settings.nNearestNeighbours = 20;           % crowding = closeness to the 20 nearest concepts
settings.nDensityGridPoints = 512;          % grid on which each dimension's density is estimated
settings.lowestDensity = 1e-9;              % floor on a density value
settings.lowestLikelihoodRatio = 1e-12;     % floor on a feature's likelihood ratio
settings.noiseFractions = [0.02 0.03 0.05 0.08];   % retention noise, fraction of a dimension's SD
settings.nExperiments = 2500;               % between items: simulated study-test experiments
settings.nCriticalItems = 120;              % designed lists: critical concepts
settings.nParticipantsPerList = 100;        % designed lists: simulated participants per list
settings.nCloseNeighbours = [0 16];         % designed lists: close neighbors on the two lists

% ---- the SPoSE embedding and the concept names ----
embedding = load(fullfile(releasePaths.sharedThings, 'spose_embedding_49d_sorted.txt'));
conceptNames = readlines(fullfile(releasePaths.sharedThings, 'unique_id.txt'));
conceptNames = conceptNames(strlength(conceptNames) > 0);      % drop the empty line after the last name
if size(embedding, 1) ~= numel(conceptNames)
    error('continuous_rem_spose:names', 'unique_id.txt has %d names for %d embedding rows', ...
          numel(conceptNames), size(embedding, 1));
end
nFeatures = size(embedding, 2);             % 49 SPoSE dimensions
settings.nFeatures = nFeatures;

% ---- the 1,600 concepts (first random draw) ----
chosenConcepts = randperm(size(embedding, 1), settings.nConceptsUsed);
X = embedding(chosenConcepts, :);           % one row per simulated item, 49 SPoSE values
nConcepts = size(X, 1);

conceptSubset = table((1:nConcepts)', chosenConcepts(:), conceptNames(chosenConcepts), ...
    'VariableNames', {'item_number', 'concept_row_in_embedding', 'concept'});
writetable(conceptSubset, fullfile(outputFolder, 'concept_subset.csv'));

% ---- the report ----
reportFile = fopen(fullfile(outputFolder, 'continuous_rem_spose.txt'), 'w');
report(reportFile, 'CONTINUOUS REM IN THE SPoSE SPACE\n');
report(reportFile, [repmat('=', 1, 92) '\n']);
report(reportFile, ['Familiarity is a likelihood ratio computed feature by feature, so rarity\n' ...
    'weights every comparison between a probe and a stored trace.\n' ...
    'Criterion is 1.0, which is meaningful because the quantity is a ratio.\n\n']);
report(reportFile, '%d THINGS concepts, %d SPoSE dimensions, %d studied, %d old and %d new tested.\n', ...
       nConcepts, nFeatures, settings.nStudied, settings.nTestedOld, settings.nTestedNew);
report(reportFile, 'storage %.2f, stored-correctly %.2f\n\n', settings.storage, settings.copy);

% ---- how common each value is: the density of each SPoSE dimension across the 1,600 concepts ----
% (no random numbers are used here)
densityGrid = zeros(settings.nDensityGridPoints, nFeatures);
density = zeros(settings.nDensityGridPoints, nFeatures);
for feature = 1:nFeatures
    [densityValues, gridValues] = ksdensity(X(:, feature), 'NumPoints', settings.nDensityGridPoints);
    densityGrid(:, feature) = gridValues(:);
    density(:, feature) = max(densityValues(:), settings.lowestDensity);
end
settings.densityGrid = densityGrid;
settings.density = density;
settings.featureSD = std(X, 0, 1);         % each dimension's SD across the 1,600 concepts

% ---- crowding of each concept, from the correlation distance between SPoSE vectors ----
distance = squareform(pdist(X, 'correlation'));     % 1 minus the Pearson r of two concepts' vectors
distance(1:nConcepts+1:end) = NaN;                  % a concept is not its own neighbor
crowdingGlobal = -mean(distance, 2, 'omitnan');     % minus the mean distance to all other concepts
sortedDistance = sort(fillmissing(distance, 'constant', Inf), 2);
crowdingLocal = -mean(sortedDistance(:, 1:settings.nNearestNeighbours), 2);   % to the 20 nearest

% Surprisal of each concept (minus the summed log density of its 49 values), for the report line
% that compares the rarity the model uses with the two crowding measures.
surprisal = zeros(nConcepts, 1);
for feature = 1:nFeatures
    valueDensity = max(interp1(densityGrid(:, feature), density(:, feature), X(:, feature), ...
                               'linear', 'extrap'), settings.lowestDensity);
    surprisal = surprisal - log(valueDensity);
end
report(reportFile, 'surprisal vs global crowding rho = %+.3f;  vs local crowding rho = %+.3f\n\n', ...
       corr(surprisal, crowdingGlobal, 'type', 'Spearman'), ...
       corr(surprisal, crowdingLocal, 'type', 'Spearman'));

% ---- the two analyses at each noise level ----
report(reportFile, '  sigma      hit      FA     d''    rho vs global   rho vs local   designed d''  hits-FA\n');
for noiseIndex = 1:numel(settings.noiseFractions)
    noiseFraction = settings.noiseFractions(noiseIndex);
    runSettings = settings;
    runSettings.noiseSD = noiseFraction * settings.featureSD;     % sigma_j of each dimension

    % 1. between items (SI Appendix, Section 2)
    [rhoGlobal, rhoLocal, meanHitRate, meanFalseAlarmRate, hitCount, oldTests, falseAlarmCount, ...
        newTests] = between_items(X, runSettings, crowdingGlobal, crowdingLocal);
    % d′ of the mean rates, for the report only (rates kept between 0.001 and 0.999)
    clampedZ = @(rate) norminv(min(max(rate, 1e-3), 1 - 1e-3));

    % 2. designed lists (Fig. S6)
    [listChanges, criticalItems, hitRates, falseAlarmRates] = designed_lists(X, runSettings, distance);

    report(reportFile, '  %5.2f %8.3f %8.3f %6.2f %15.3f %14.3f %13.3f %9.3f\n', ...
           noiseFraction, meanHitRate, meanFalseAlarmRate, ...
           clampedZ(meanHitRate) - clampedZ(meanFalseAlarmRate), rhoGlobal, rhoLocal, ...
           listChanges(1), listChanges(2));

    % the item-level results, for the bootstrap intervals (bootstrap_ci.py, stats_figS6.py)
    itemTable = table(chosenConcepts(:), hitCount, oldTests, falseAlarmCount, newTests, ...
                      crowdingLocal, crowdingGlobal, ...
                      'VariableNames', {'concept_row_in_embedding', 'hit_count', 'old_tests', ...
                                        'false_alarm_count', 'new_tests', 'crowding_local20', ...
                                        'crowding_global'});
    writetable(itemTable, fullfile(outputFolder, sprintf('continuous_rem_items_sigma%.2f.csv', noiseFraction)));

    designedTable = table(chosenConcepts(criticalItems(:))', hitRates(:, 1), falseAlarmRates(:, 1), ...
                          hitRates(:, 2), falseAlarmRates(:, 2), ...
                          'VariableNames', {'concept_row_in_embedding', 'hit_rate_0_neighbors', ...
                                            'false_alarm_rate_0_neighbors', 'hit_rate_16_neighbors', ...
                                            'false_alarm_rate_16_neighbors'});
    writetable(designedTable, fullfile(outputFolder, sprintf('continuous_rem_designed_sigma%.2f.csv', noiseFraction)));
end

% Closing notes of the report. The two reference values in them are typed in, not computed here:
% REM with generated features in module 01 (rho -0.551, the run of Fig. 1) and in module A2d (d′
% change -0.132, the designed lists at the Fig. 1 values).
report(reportFile, '\n');
report(reportFile, [repmat('-', 1, 92) '\n']);
report(reportFile, 'READING THIS\n');
report(reportFile, [repmat('-', 1, 92) '\n']);
report(reportFile, ['  sigma is the only quantity the continuous case adds, given as a fraction of\n' ...
    '  each dimension''s standard deviation across the concepts. Fig. S6 shows\n' ...
    '  sigma 0.02, 0.03 and 0.05.\n' ...
    '  The comparable numbers are: REM in the run of Fig. 1 gives rho =\n' ...
    '  -0.551 between memorability and crowding, and the designed-list contrast\n' ...
    '  gives d'' -0.132.\n' ...
    '  rho vs local and rho vs global correlate memorability with crowding over\n' ...
    '  the 20 nearest concepts and over all other concepts.\n' ...
    '  designed d'' and hits-FA are the changes in d'' and in hit rate minus false\n' ...
    '  alarm rate from 0 to 16 close neighbors on the designed lists.\n']);
fclose(reportFile);

write_settings(fullfile(outputFolder, 'continuous_rem_settings.csv'), settings, size(embedding, 1), seed);
fprintf('\nwrote the report, concept_subset.csv, continuous_rem_settings.csv and the item files to %s\n', ...
        outputFolder);
end


% ======================================================================= the model
function [traces, isStored] = store_study_list(X, studiedRows, settings, nParticipants)
% Memory traces of one study list, for nParticipants simulated participants at once.
% traces(participant, listPosition, feature) holds the stored value; isStored says which features
% were stored at all (probability u). A stored feature is copied correctly with probability c and
% then keeps the studied value plus Gaussian retention noise (SD sigma_j); otherwise it takes that
% dimension's value from a concept drawn at random. (A value is filled in for every feature; the
% unstored ones are ignored when familiarity is computed.)
% Random draws, in this order: rand (stored?), rand (copied correctly?), then for each feature in
% turn randn (the noise of the correct copies) and randi (the concepts that give the incorrect
% copies).
nStudied = numel(studiedRows);
traces = repmat(reshape(X(studiedRows, :), [1, nStudied, settings.nFeatures]), [nParticipants, 1, 1]);
isStored = rand(size(traces)) < settings.storage;
isCopiedCorrectly = rand(size(traces)) < settings.copy;
for feature = 1:settings.nFeatures
    values = traces(:, :, feature);
    % linear indices keep the shape right when there is one participant (the slice is then a row)
    correctPositions = find(isCopiedCorrectly(:, :, feature));
    wrongPositions = find(~isCopiedCorrectly(:, :, feature));
    values(correctPositions) = values(correctPositions) + settings.noiseSD(feature) * ...
        reshape(randn(numel(correctPositions), 1), size(values(correctPositions)));
    values(wrongPositions) = reshape(X(randi(size(X, 1), numel(wrongPositions), 1), feature), ...
                                     size(values(wrongPositions)));
    traces(:, :, feature) = values;
end
end


function familiarity = rem_familiarity(traces, isStored, probes, settings)
% REM's familiarity for continuous features: for each probe (row of probes) and each simulated
% participant, the mean over the study list's traces of the trace's likelihood ratio, which is the
% product over the trace's stored features of
%     lambda_j = 1 + c * ( N(t - x; sigma_j) / f_j(t) - 1 ).
% The product is computed as the sum of logs. Every probe, participant and trace is computed at once
% (arrays probe x participant x trace); MATLAB expands a probe column across the participants and
% traces, and a trace row across the probes.
[nParticipants, nTraces, ~] = size(traces);
nProbes = size(probes, 1);
logLikelihoodRatio = zeros(nProbes, nParticipants, nTraces);
for feature = 1:settings.nFeatures
    traceValues = reshape(traces(:, :, feature), [1, nParticipants, nTraces]);
    stored = reshape(isStored(:, :, feature), [1, nParticipants, nTraces]);
    % f_j(t): how common the trace value is among all concepts (the dimension's density)
    valueDensity = max(interp1(settings.densityGrid(:, feature), settings.density(:, feature), ...
                               traceValues, 'linear', 'extrap'), settings.lowestDensity);
    % N(t - x; sigma_j): how likely the trace value is if it was stored from this probe, with noise
    noiseDensity = exp(-0.5 * ((traceValues - probes(:, feature)) / settings.noiseSD(feature)) .^ 2) ...
                   / (settings.noiseSD(feature) * sqrt(2 * pi));
    likelihoodRatio = 1 + settings.copy * (noiseDensity ./ valueDensity - 1);
    likelihoodRatio = max(likelihoodRatio, settings.lowestLikelihoodRatio);
    % unstored features add log(lambda) x 0 = 0
    logLikelihoodRatio = logLikelihoodRatio + log(likelihoodRatio) .* stored;
end
familiarity = squeeze(mean(exp(logLikelihoodRatio), 3));   % mean over the traces
if nProbes == 1
    familiarity = reshape(familiarity, 1, nParticipants);
end
end


% ======================================================================= the two analyses
function [rhoGlobal, rhoLocal, meanHitRate, meanFalseAlarmRate, hitCount, oldTests, ...
          falseAlarmCount, newTests] = between_items(X, settings, crowdingGlobal, crowdingLocal)
% Simulated study-test experiments (one simulated participant each). Returns the Spearman
% correlation of each concept's memorability (hit rate minus false alarm rate) with its global and
% its local crowding, the mean hit and false alarm rates, and each concept's counts.
nConcepts = size(X, 1);
hitCount = zeros(nConcepts, 1);
oldTests = zeros(nConcepts, 1);
falseAlarmCount = zeros(nConcepts, 1);
newTests = zeros(nConcepts, 1);
for experiment = 1:settings.nExperiments
    % 60 different concepts: the first 40 are studied, the other 20 are the new test items
    drawn = randperm(nConcepts, settings.nStudied + settings.nTestedNew);
    studied = drawn(1:settings.nStudied);
    newItems = drawn(settings.nStudied+1:end);
    % 20 of the 40 studied concepts are tested
    testedOld = studied(randperm(settings.nStudied, settings.nTestedOld));
    [traces, isStored] = store_study_list(X, studied, settings, 1);
    familiarityOld = rem_familiarity(traces, isStored, X(testedOld, :), settings);
    familiarityNew = rem_familiarity(traces, isStored, X(newItems, :), settings);
    for testIndex = 1:settings.nTestedOld
        concept = testedOld(testIndex);
        oldTests(concept) = oldTests(concept) + 1;
        if familiarityOld(testIndex) > settings.criterion
            hitCount(concept) = hitCount(concept) + 1;
        end
    end
    for testIndex = 1:settings.nTestedNew
        concept = newItems(testIndex);
        newTests(concept) = newTests(concept) + 1;
        if familiarityNew(testIndex) > settings.criterion
            falseAlarmCount(concept) = falseAlarmCount(concept) + 1;
        end
    end
end
% concepts tested at least once as old and once as new (all 1,600 with seed 1)
testedBothWays = oldTests > 0 & newTests > 0;
hitRate = hitCount(testedBothWays) ./ oldTests(testedBothWays);
falseAlarmRate = falseAlarmCount(testedBothWays) ./ newTests(testedBothWays);
memorability = hitRate - falseAlarmRate;
meanHitRate = mean(hitRate);
meanFalseAlarmRate = mean(falseAlarmRate);
rhoGlobal = corr(memorability, crowdingGlobal(testedBothWays), 'type', 'Spearman');
rhoLocal = corr(memorability, crowdingLocal(testedBothWays), 'type', 'Spearman');
end


function [listChanges, criticalItems, hitRates, falseAlarmRates] = designed_lists(X, settings, distance)
% The same critical concept on a list with 0 and on a list with 16 close neighbors.
% Returns listChanges = [d′ change, change in hit rate minus false alarm rate] (16 minus 0
% neighbors, from the mean rates across the critical concepts), the critical concepts (positions
% in X) and their hit and false alarm rates (columns: 0 and 16 neighbors).
nConcepts = size(X, 1);
distanceWithoutSelf = distance;
distanceWithoutSelf(isnan(distanceWithoutSelf)) = Inf;     % a concept sorts last in its own row
[~, neighbourOrder] = sort(distanceWithoutSelf, 2);        % row i: concepts from nearest to farthest
criticalItems = randperm(nConcepts, settings.nCriticalItems);
hitRates = zeros(settings.nCriticalItems, 2);
falseAlarmRates = zeros(settings.nCriticalItems, 2);
for itemIndex = 1:settings.nCriticalItems
    critical = criticalItems(itemIndex);
    neighbours = neighbourOrder(critical, :);
    neighbours(neighbours == critical) = [];
    lure = neighbours(1);                                  % the nearest concept, never studied
    pool = neighbours(2:end);                              % all other concepts, nearest first
    farHalf = pool(round(numel(pool) * 0.5):end);          % the farther half of them
    for listType = 1:2
        nClose = settings.nCloseNeighbours(listType);
        nFar = settings.nStudied - 1 - nClose;
        studied = [critical, pool(1:nClose), farHalf(randperm(numel(farHalf), nFar))];
        [traces, isStored] = store_study_list(X, studied, settings, settings.nParticipantsPerList);
        hitRates(itemIndex, listType) = ...
            mean(rem_familiarity(traces, isStored, X(critical, :), settings) > settings.criterion);
        falseAlarmRates(itemIndex, listType) = ...
            mean(rem_familiarity(traces, isStored, X(lure, :), settings) > settings.criterion);
    end
end
% d′ from the mean rates, each rate kept between 1/(2N) and 1 - 1/(2N), N = participants per list
lowestRate = 1 / (2 * settings.nParticipantsPerList);
clampedZ = @(rate) norminv(min(max(rate, lowestRate), 1 - lowestRate));
meanHitRates = mean(hitRates, 1);
meanFalseAlarmRates = mean(falseAlarmRates, 1);
dPrimeChange = (clampedZ(meanHitRates(2)) - clampedZ(meanFalseAlarmRates(2))) ...
             - (clampedZ(meanHitRates(1)) - clampedZ(meanFalseAlarmRates(1)));
correctedRecognitionChange = (meanHitRates(2) - meanFalseAlarmRates(2)) ...
                           - (meanHitRates(1) - meanFalseAlarmRates(1));
listChanges = [dPrimeChange, correctedRecognitionChange];
end


% ======================================================================= output helpers
function report(fileId, varargin)
% Print a line to the screen and to the report file.
fprintf(varargin{:});
fprintf(fileId, varargin{:});
end


function write_settings(filePath, settings, nConceptsInEmbedding, seed)
% Every setting as run, one per row (setting, value, description).
rows = {
    'n_concepts_in_embedding', nConceptsInEmbedding, 'THINGS concepts in the SPoSE embedding'
    'n_features', settings.nFeatures, 'SPoSE dimensions = features of each simulated item'
    'n_concepts_used', settings.nConceptsUsed, 'concepts drawn with randperm after the seed is set'
    'seed', seed, 'MATLAB rng(seed) with the Mersenne twister (1 = the run the paper reports)'
    'storage_u', settings.storage, 'probability that a feature is stored'
    'copy_c', settings.copy, 'probability that a stored feature is copied correctly'
    'n_studied', settings.nStudied, 'concepts on each study list'
    'n_tested_old', settings.nTestedOld, 'studied concepts tested per experiment'
    'n_tested_new', settings.nTestedNew, 'unstudied concepts tested per experiment'
    'criterion', settings.criterion, 'old when familiarity is above this value'
    'n_nearest_neighbors_crowding', settings.nNearestNeighbours, ...
        'crowding = minus the mean correlation distance to this many nearest concepts'
    'density_grid_points', settings.nDensityGridPoints, 'ksdensity grid of each dimension (default bandwidth)'
    'lowest_density', settings.lowestDensity, 'floor on a density value'
    'lowest_likelihood_ratio', settings.lowestLikelihoodRatio, 'floor on a feature likelihood ratio'
    'noise_fractions', settings.noiseFractions, ...
        'retention noise SD as a fraction of each dimension SD across the concepts'
    'n_experiments_between_items', settings.nExperiments, ...
        'simulated study-test experiments (one simulated participant each)'
    'n_critical_items_designed', settings.nCriticalItems, 'critical concepts on the designed lists'
    'n_participants_per_list_designed', settings.nParticipantsPerList, 'simulated participants per designed list'
    'n_close_neighbors_designed', settings.nCloseNeighbours, ...
        'close neighbors on the two designed lists (the 2nd to 17th nearest; the nearest is the lure)'
    };
fileId = fopen(filePath, 'w');
fprintf(fileId, 'setting,value,description\n');
for rowIndex = 1:size(rows, 1)
    values = rows{rowIndex, 2};
    valueText = strjoin(arrayfun(@(value) sprintf('%.15g', value), values, 'UniformOutput', false), ' ');
    fprintf(fileId, '%s,%s,"%s"\n', rows{rowIndex, 1}, valueText, rows{rowIndex, 3});
end
fclose(fileId);
end
