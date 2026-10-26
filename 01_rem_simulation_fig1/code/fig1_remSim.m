function [figureData, outFile] = fig1_remSim(seedValue, outFile, simLen)
%FIG1_REMSIM  The REM simulation behind Fig. 1: 10,000 simulated recognition memory experiments
%   over a set of 1,600 generated items, each item a vector of 49 random feature values.
%
%   What it does
%     1. Generates 26,107 items (the number of THINGS images) with 49 features each, every feature
%        value drawn from a geometric distribution (g = 0.65) plus 1, and keeps a random subset of
%        1,600 of them as the simulated image set.
%     2. Runs 10,000 simulated experiments. In each one a simulated participant studies 40 items
%        drawn at random from the 1,600, then is tested on 20 of the studied items (old) and 20
%        other items of the 1,600 that were not studied (new foils). Encoding and the old/new
%        decision follow REM (Shiffrin and Steyvers, 1997): each feature is stored with
%        probability 0.3, stored correctly with probability 0.6, and the item is called "old" when
%        the mean likelihood ratio over the 40 memory traces is above 1.
%     3. For every item: hit rate (over the simulations in which it is a tested old item), false
%        alarm rate (over the simulations in which it is a foil) and memorability = hit rate minus
%        false alarm rate.
%     4. The analyses of Fig. 1: each item's mean Euclidean distance to the items of the set (the
%        mean over all 1,600 distances, its own distance of 0 included), its similarity = 1 / mean
%        distance, the Pearson correlation of similarity with memorability (r_sim, the r = -0.56 of
%        Fig. 1b), and the feature-level analysis (each feature's Spearman correlation with
%        memorability and with similarity, and the Spearman correlation of those two sets of 49
%        correlations, r_meta, the rho = -0.44 of Fig. 1d).
%
%   Inputs (all optional)
%     seedValue  random seed (default 12345, the seed of Fig. 1)
%     outFile    where to save the figureData struct (.mat, saved with -v7); empty or missing = do
%                not save (run_module.m does not save it)
%     simLen     number of simulated experiments (default 10000, as in the paper; a smaller
%                number is only for quick tests: simulation k always uses the same random draws,
%                so the first k simulations of a short run equal the first k of a full run)
%
%   Outputs
%     figureData  struct with every per-item and per-feature result (the fields of the saved run in
%                 data/paper_run/, plus the design constants)
%     outFile     the file it was saved to ('' when not saved)
%
%   Random numbers
%     rng(seedValue, 'twister') (MATLAB's default Mersenne Twister) generates the 26,107 items and
%     picks the 1,600. Simulation k then uses its own stream, Threefry with seed seedValue and
%     substream k, for its list draw and its encoding. Because every simulation sets its own
%     stream, the draws do not depend on the order in which the simulations run; they run one
%     after another here, with no parallel pool. The last digits of the p values that corr returns
%     depend on the tcdf.m that MATLAB uses.
%
%   Runtime: about 130 seconds for 10,000 simulations on an Apple M2 Pro laptop (one core).
%   Memory: about 1 GB (six 1,600 x 10,000 arrays of per-simulation results; four are returned).

if nargin < 1 || isempty(seedValue)
  seedValue = 12345;
end
if nargin < 2
  outFile = '';
end
if nargin < 3 || isempty(simLen)
  simLen = 10000;
end

%% Parameters (Methods, Basic REM Simulation of Memorability)
fprintf('Using random seed: %d for reproducibility\n', seedValue);

baseRateOfFeatures = 0.65;   % g, the geometric distribution's parameter
numFeaturesPerItem = 49;     % w, as in the 49 SPoSE dimensions of THINGS
probEncoding = .3;           % u, probability that a feature is stored at all
correctEncodeProb = .6;      % c, probability that a stored feature is copied correctly
criterion = 1;               % "old" when the mean likelihood ratio is above 1 (odds of 1:1)

thingsDatasetSize = 26107;   % items generated before the 1,600 are drawn (the number of THINGS images)
totalImages = 1600;          % the simulated image set
studySize    = 40;   % items studied per simulation
testOldSize  = 20;   % old items tested (the first 20 of the 40 studied)
testNewSize  = 20;   % new items (foils) tested, drawn from the 1,600 but not studied in that simulation

%% Generate the item pool (seeded once, before the simulations)
rng(seedValue, 'twister');
thingDatasetImgs = GenImages(thingsDatasetSize, baseRateOfFeatures, numFeaturesPerItem);
representativeSubset = thingDatasetImgs(randperm(thingsDatasetSize, totalImages), :);

fprintf('Starting memory performance simulations (%d simulations, run one after another)...\n', simLen);
ticSim = tic;

%% Main simulation loop
% Each simulation: pick 40 studied items and 20 foils (all different) from the 1,600. Encode the
% studied items. Probe the first 20 studied items and the 20 foils against the memory of the 40
% studied items. Studied probes give hits, foil probes give false alarms. Over the simulations every
% item is tested both as an old item and as a foil, so each item gets its own hit and false alarm rate.
simHits     = zeros(totalImages, simLen);
simFAs      = zeros(totalImages, simLen);
simHitMask  = zeros(totalImages, simLen);  % which items were tested as old items in each simulation
simFaMask   = zeros(totalImages, simLen);  % which items were foils in each simulation
simHitOdds  = zeros(totalImages, simLen);
simFaOdds   = zeros(totalImages, simLen);

% The loop below replaces the global random stream in every simulation; keep the current one so it
% can be put back afterwards.
streamBeforeSimulations = RandStream.getGlobalStream;

for nSim = 1:simLen
  % Random stream of this simulation: Threefry, seed seedValue, substream nSim.
  stream = RandStream('Threefry', 'Seed', seedValue);
  stream.Substream = nSim;
  RandStream.setGlobalStream(stream);

  % Draw studySize + testNewSize distinct items from the 1,600.
  % The first studySize are studied; of those, the first testOldSize are
  % the "old" items tested. The next testNewSize are the foils ("new" items).
  perm = randperm(totalImages, studySize + testNewSize);
  studiedIdx   = perm(1:studySize);
  testedOldIdx = studiedIdx(1:testOldSize);   % the studied items that are tested
  foilIdx      = perm(studySize + 1:studySize + testNewSize);

  studiedItems   = representativeSubset(studiedIdx, :);
  testedOldItems = representativeSubset(testedOldIdx, :);
  foilItems      = representativeSubset(foilIdx, :);

  % Encode all studied items into memory traces
  studiedEncoded = EncodeImage_par(studiedItems, baseRateOfFeatures, ...
    probEncoding, correctEncodeProb);

  % Hits: probe each tested old item against the memory of the studied items
  hitVec = zeros(1, testOldSize);
  hitOddsVec = zeros(1, testOldSize);
  for idx = 1:testOldSize
    lambdas = zeros(1, studySize);
    for midx = 1:studySize
      lambdas(midx) = ProbeImage(testedOldItems(idx,:), studiedEncoded(midx,:), ...
        baseRateOfFeatures, numFeaturesPerItem, correctEncodeProb);
    end
    hitOddsVec(idx) = mean(lambdas);
    hitVec(idx) = hitOddsVec(idx) > criterion;
  end

  % False alarms: probe each foil against the memory of the studied items
  faVec = zeros(1, testNewSize);
  faOddsVec = zeros(1, testNewSize);
  for idx = 1:testNewSize
    lambdas = zeros(1, studySize);
    for midx = 1:studySize
      lambdas(midx) = ProbeImage(foilItems(idx,:), studiedEncoded(midx,:), ...
        baseRateOfFeatures, numFeaturesPerItem, correctEncodeProb);
    end
    faOddsVec(idx) = mean(lambdas);
    faVec(idx) = faOddsVec(idx) > criterion;
  end

  % Store the results per item. Only items that appeared at test (old or new)
  % are marked in the masks; studied items that were not tested do not
  % count toward any rate in this simulation.
  localHits     = zeros(totalImages, 1);
  localFAs      = zeros(totalImages, 1);
  localHitMask  = zeros(totalImages, 1);
  localFaMask   = zeros(totalImages, 1);
  localHitOdds  = zeros(totalImages, 1);
  localFaOdds   = zeros(totalImages, 1);
  for idx = 1:testOldSize
    sID = testedOldIdx(idx);
    localHits(sID)    = hitVec(idx);
    localHitMask(sID) = 1;
    localHitOdds(sID) = hitOddsVec(idx);
  end
  for idx = 1:testNewSize
    fID = foilIdx(idx);
    localFAs(fID)     = faVec(idx);
    localFaMask(fID)  = 1;
    localFaOdds(fID)  = faOddsVec(idx);
  end
  simHits(:, nSim)    = localHits;
  simFAs(:, nSim)     = localFAs;
  simHitMask(:, nSim) = localHitMask;
  simFaMask(:, nSim)  = localFaMask;
  simHitOdds(:, nSim) = localHitOdds;
  simFaOdds(:, nSim)  = localFaOdds;

  if mod(nSim, 1000) == 0
    fprintf('  %d of %d simulations done (%.0f s)\n', nSim, simLen, toc(ticSim));
  end
end

RandStream.setGlobalStream(streamBeforeSimulations);
fprintf('Simulations finished in %.1f seconds.\n', toc(ticSim));

%% Per-item hit rate, false alarm rate and memorability over the simulations
hitCounts = sum(simHitMask, 2);
faCounts  = sum(simFaMask, 2);

allHitRates = zeros(totalImages, 1);
allFARates  = zeros(totalImages, 1);
seHitFA     = NaN(totalImages, 1);

for i = 1:totalImages
  hMask = simHitMask(i, :) == 1;
  fMask = simFaMask(i, :) == 1;
  if any(hMask)
    allHitRates(i) = mean(simHits(i, hMask));
  end
  if any(fMask)
    allFARates(i)  = mean(simFAs(i, fMask));
  end
  % Standard error of (hit rate - false alarm rate), the two rates treated as independent samples
  nH = sum(hMask); nF = sum(fMask);
  if nH > 1 && nF > 1
    varH = var(simHits(i, hMask)) / nH;
    varF = var(simFAs(i, fMask))  / nF;
    seHitFA(i) = sqrt(varH + varF);
  end
end

hitFARate = allHitRates - allFARates;

% Items that were never tested in one of the two roles get no memorability score
validIndices = (hitCounts > 0) & (faCounts > 0);
imagesWithData = sum(validIndices);
if imagesWithData < totalImages
  fprintf('WARNING: %d/%d images lack data. Increase simLen if needed.\n', ...
    totalImages - imagesWithData, totalImages);
  hitFARate(~validIndices) = NaN;
end
fprintf('Simulations complete: %d/%d items with valid data.\n', imagesWithData, totalImages);

%% Analyses of Fig. 1
imagesOfInterest = representativeSubset;
hitFARate = hitFARate(:);
numItems = totalImages;

% Euclidean distance between every pair of items in the 49-dimensional feature space
pairwiseDistances = squareform(pdist(imagesOfInterest, 'euclidean'));

% Mean distance of each item to all items (the zero distance to itself is included in the mean),
% and similarity = 1 / mean distance
avgDistances = mean(pairwiseDistances, 2);
similarity   = 1 ./ avgDistances;

[~, mostIsolatedIdx] = max(avgDistances);
[~, mostTypicalIdx]  = min(avgDistances);

% Pearson correlation: mean distance (distinctiveness) with memorability
[r, p] = corrcoef(avgDistances, hitFARate, 'rows', 'complete');
correlation = r(1,2);
pValue      = p(1,2);

% Pearson correlation: similarity (1 / mean distance) with memorability (the r = -0.56 of Fig. 1b)
[r_sim, pval_sim] = corr(similarity, hitFARate, 'rows', 'complete');

%% Feature-level analyses (Figs. 1c and 1d)
featureData = imagesOfInterest;
nFeatures   = size(featureData, 2);

featureLabels = cell(nFeatures, 1);
for i = 1:nFeatures
  featureLabels{i} = sprintf('Feature %d', i);
end

% Each feature with memorability (Spearman)
featureCorrsMemory = zeros(nFeatures, 1);
featurePvalsMemory = zeros(nFeatures, 1);
for i = 1:nFeatures
  [rho, pval] = corr(featureData(:,i), hitFARate, 'Type', 'Spearman', 'rows', 'complete');
  featureCorrsMemory(i) = rho;
  featurePvalsMemory(i) = pval;
end

% Each feature with mean distance (Spearman)
featureCorrsDist = zeros(nFeatures, 1);
featurePvalsDist = zeros(nFeatures, 1);
for i = 1:nFeatures
  [rho, pval] = corr(featureData(:,i), avgDistances, 'Type', 'Spearman', 'rows', 'complete');
  featureCorrsDist(i) = rho;
  featurePvalsDist(i) = pval;
end

% Each feature with similarity (Spearman)
featureCorrsSimil = zeros(nFeatures, 1);
featurePvalsSimil = zeros(nFeatures, 1);
for i = 1:nFeatures
  [rho, pval] = corr(featureData(:,i), similarity, 'Type', 'Spearman', 'rows', 'complete');
  featureCorrsSimil(i) = rho;
  featurePvalsSimil(i) = pval;
end

% Across the 49 features: Spearman correlation of the feature-with-similarity correlations and the
% feature-with-memorability correlations (the rho = -0.44 of Fig. 1d)
[r_meta, p_meta] = corr(featureCorrsSimil, featureCorrsMemory, 'Type', 'Spearman');
bothSig = (featurePvalsSimil < 0.05) & (featurePvalsMemory < 0.05);

% Items ranked by memorability (Fig. 1a)
[sortedHitFA, sortIdx] = sort(hitFARate, 'descend');
sortedSE = seHitFA(sortIdx);

%% Collect everything in the figureData struct (the fields of the saved run in data/paper_run/)
figureData = struct();

figureData.hitRate              = allHitRates;
figureData.faRate               = allFARates;
figureData.hitFARate            = hitFARate;
figureData.imagesOfInterest     = imagesOfInterest;
figureData.seHitFA              = seHitFA;

figureData.pairwiseDistances    = pairwiseDistances;
figureData.avgDistances         = avgDistances;
figureData.similarity           = similarity;
figureData.mostIsolatedIdx      = mostIsolatedIdx;
figureData.mostTypicalIdx       = mostTypicalIdx;

figureData.correlation          = correlation;
figureData.pValue               = pValue;
figureData.r_sim                = r_sim;
figureData.pval_sim             = pval_sim;

figureData.featureLabels        = featureLabels;
figureData.featureCorrsMemory   = featureCorrsMemory;
figureData.featurePvalsMemory   = featurePvalsMemory;
figureData.featureCorrsDist     = featureCorrsDist;
figureData.featurePvalsDist     = featurePvalsDist;
figureData.featureCorrsSimil    = featureCorrsSimil;
figureData.featurePvalsSimil    = featurePvalsSimil;
figureData.r_meta               = r_meta;
figureData.p_meta               = p_meta;
figureData.bothSig              = bothSig;

figureData.sortedHitFA          = sortedHitFA;
figureData.sortedSE             = sortedSE;
figureData.sortIdx              = sortIdx;

figureData.simHitOdds           = simHitOdds;
figureData.simFaOdds            = simFaOdds;
figureData.simHitMask           = simHitMask;
figureData.simFaMask            = simFaMask;

figureData.seedValue            = seedValue;
figureData.baseRateOfFeatures   = baseRateOfFeatures;
figureData.numFeaturesPerItem   = numFeaturesPerItem;
figureData.probEncoding         = probEncoding;
figureData.correctEncodeProb    = correctEncodeProb;
figureData.criterion            = criterion;

% Design constants of the run
figureData.simLen               = simLen;
figureData.totalImages          = totalImages;
figureData.thingsDatasetSize    = thingsDatasetSize;
figureData.studySize            = studySize;
figureData.testOldSize          = testOldSize;
figureData.testNewSize          = testNewSize;

fprintf('r_sim (Pearson, 1 / mean distance vs memorability) = %.4f\n', r_sim);
fprintf('r_meta (Spearman across the %d features)           = %.4f\n', nFeatures, r_meta);

%% Save
if ~isempty(outFile)
  outFolder = fileparts(outFile);
  if ~isempty(outFolder) && ~exist(outFolder, 'dir')
    mkdir(outFolder);
  end
  save(outFile, 'figureData', '-v7');
  [~, outName, outExtension] = fileparts(outFile);
  fprintf('Saved figureData as %s%s\n', outName, outExtension);
end
end

%% Local functions

function image_shown = GenImages(nImages, g, w)
  % nImages items with w features each; every value is a geometric draw (parameter g) plus 1
  image_shown = geornd(g, [nImages, w]) + 1;
end

function image_encoded = EncodeImage_par(image_shown, g, u, c)
  % REM encoding of the studied items: each feature is stored with probability u; a stored feature
  % is copied correctly with probability c, otherwise it gets a new geometric draw (parameter g).
  % The new draw has no "+ 1", so it can be 0, which then counts as "not stored".
  featureEncodedAtAll     = rand(size(image_shown)) < u;
  featureEncodedCorrectly = rand(size(image_shown)) < c;
  numWrongEncoded = sum(featureEncodedCorrectly(:) == 0);

  image_encoded = image_shown;
  image_encoded(featureEncodedCorrectly == 0) = geornd(g, [1 numWrongEncoded]);
  image_encoded(featureEncodedAtAll == 0) = 0;
end

function lambda = ProbeImage(imgShown, encodedimage, g, w, c)
  % Likelihood ratio of one memory trace for one test item (REM): every stored feature that does
  % not match multiplies the ratio by (1 - c); every matching feature multiplies it by the
  % diagnosticity of its value (rare values count more).
  M  = find(imgShown == encodedimage);

  MM = sum(imgShown(encodedimage > 0) ~= encodedimage(encodedimage > 0));
  lambda = (1-c).^(MM);

  for idx = 1:length(M)
    if M(idx) <= w
      lambda = lambda * oddsOfThisParticularMatch(encodedimage(M(idx)), c, g);
    end
  end
end

function om = oddsOfThisParticularMatch(v_kj, c, g)
  % Odds that a match on value v_kj came from the studied item rather than from chance:
  % (c + (1 - c) * P(value)) / P(value), with P the geometric probability of the value
  r = v_kj - 1;

  probThisMatchIfWeReallySawIt = c + (1-c) * geopdf(r, g);
  probThisMatchIfWeDidntSeeIt  = geopdf(r, g);

  om = probThisMatchIfWeReallySawIt / probThisMatchIfWeDidntSeeIt;
end
