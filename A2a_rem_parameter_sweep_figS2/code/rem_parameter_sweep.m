function rem_parameter_sweep(saveItems)
% rem_parameter_sweep: the 32-cell REM parameter sweep of SI Appendix, Section 2
% (REM Parameters) and SI Appendix, Fig. S2. Its cell at the standard values of
% Shiffrin & Steyvers (1997) gives the rho = -0.52 of the Methods (Basic REM
% Simulation of Memorability).
%
% The simulation of Fig. 1 uses w = 49 features and g = 0.65, which depart from
% the standard values, and accuracy rises with w and falls with g. The sweep
% varies the parameters to test whether the relationship between memorability
% and distinctiveness depends on the values chosen.
%
% Two quantities are computed per cell, matching the two results of the
% simulation of Fig. 1:
%   item level     Spearman rho between per-item memorability (hit - false alarm)
%                  and per-item summed similarity (minus the mean pairwise L2
%                  distance to the rest of the database; see Sign below)
%   feature level  across the w features, Pearson r between each feature's
%                  Spearman correlation with memorability and its Spearman
%                  correlation with summed similarity (the feature-level
%                  relationship of Fig. 1d)
%
% Design is a g x w factorial at a reference cell, plus one-at-a-time sweeps of
% storage, copy fidelity, list length and criterion away from the Shiffrin &
% Steyvers values. A full factorial over six parameters is not needed to answer
% whether any one of them carries the effect.
%
% Random numbers: rng(1, 'twister') once, then the 32 cells run serially in file
% order (never in a parallel pool). A cell's random numbers depend on every cell
% before it, so a cell reproduces only within the whole sweep. The check of the
% functions used and the per-item output draw no random numbers.
% Sign: the variable dist is MINUS the mean Euclidean distance from an item to
% the other items (the summed similarity of the Fig. S2 caption). A higher value
% means an item closer to the others on average (more similar, less
% distinctive), so a negative rhoItem means that distinctive items are
% remembered better.
% check_against_paper_run.py compares every column of this run with the saved
% run in data/paper_run/rem_parameter_sweep.csv.
%
% How to run (from the repository root, MATLAB R2023b with the Statistics and
% Machine Learning Toolbox; about 2 minutes; never a parallel pool):
%   matlab -batch "restoredefaultpath; cd('A2a_rem_parameter_sweep_figS2/code'); rem_parameter_sweep(true)"
% or through python A2a_rem_parameter_sweep_figS2/code/run_module.py
% (restoredefaultpath, for this MATLAB session only, removes folders added to
% the MATLAB path, so that they cannot replace MATLAB's own functions).
%
% Inputs: none (the simulated items are generated here).
% Outputs (in A2a_rem_parameter_sweep_figS2/outputs/):
%   rem_parameter_sweep.csv        one row per cell: label, g, w, u, c, nStudy,
%                                  crit, rhoItem, pItem, rFeat, pFeat, meanHit,
%                                  meanFA, dprime
%   rem_parameter_sweep.txt        the printed report of the same table
%   rem_parameter_sweep_items.csv  (saveItems, default true) one row per cell
%                                  and item: cell_number (row of the csv), item
%                                  (1 to 1,600), old_trials, hits, new_trials,
%                                  false_alarms, neg_mean_distance (minus the
%                                  mean Euclidean distance to the other items;
%                                  NaN for an item never tested both as old
%                                  and as new). code/stats_figS2.py recomputes
%                                  the item-level correlations from it and
%                                  bootstraps them over items.
%   logs/rem_parameter_sweep_progress.log   one line per finished cell

% per-item output switch (default true)
if nargin < 1
  saveItems = true;
end

% Warn when a folder on the MATLAB path replaces a function of MATLAB or of the
% Statistics and Machine Learning Toolbox that this script relies on. Example:
% FieldTrip's external/stats/tcdf.m returns 0 for p values below about 1e-16, so
% every pItem would be written as 0. run_module.py starts MATLAB with
% restoredefaultpath, so only MATLAB's own functions are used. Looking up a
% function draws no random numbers.
functionsUsed = {'geornd', 'randperm', 'rand', 'corr', 'tcdf', 'norminv', 'writetable'};
for functionIndex = 1:numel(functionsUsed)
  functionFile = which(functionsUsed{functionIndex});
  if isempty(functionFile)
    error('rem_parameter_sweep:missingFunction', ...
          '%s was not found; the sweep needs the Statistics and Machine Learning Toolbox', ...
          functionsUsed{functionIndex});
  end
  if ~startsWith(functionFile, matlabroot) && ~startsWith(functionFile, 'built-in')
    warning('rem_parameter_sweep:shadowedFunction', ...
            ['%s is taken from %s, not from MATLAB''s own toolboxes; ' ...
             'run restoredefaultpath first to reproduce the saved run in data/paper_run/'], ...
            functionsUsed{functionIndex}, functionFile);
  end
end

rng(1, 'twister');

% the module's outputs/ folder, found from this file's location
% (shared/matlab/release_paths.m)
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared', 'matlab'));
releasePaths = release_paths(mfilename('fullpath'));
OUT  = releasePaths.outputs;
if ~exist(OUT, 'dir'), mkdir(OUT); end

% ---- reference cell and the sweeps taken from it ---------------------------
% nDatabase: the 1,600 generated items that every study list and every set of
% new test items is drawn from; nSim: simulated experiments per cell.
ref = struct('g', 0.40, 'w', 20, 'u', 0.30, 'c', 0.70, ...
             'nStudy', 40, 'crit', 1.0, 'nDatabase', 1600, 'nSim', 10000);

cells = {};
cells = addCell(cells, ref, 'Shiffrin & Steyvers reference');

% g x w factorial: the two parameters named at the top
gs = [0.30 0.35 0.40 0.50 0.65];
ws = [20 30 49];
for gi = 1:numel(gs)
  for wi = 1:numel(ws)
    p = ref; p.g = gs(gi); p.w = ws(wi);
    cells = addCell(cells, p, sprintf('grid g=%.2f w=%d', p.g, p.w));
  end
end

% one-at-a-time from the reference
for v = [0.15 0.30 0.45]
  p = ref; p.u = v; cells = addCell(cells, p, sprintf('storage u=%.2f', v));
end
for v = [0.50 0.60 0.70 0.85]
  p = ref; p.c = v; cells = addCell(cells, p, sprintf('copy c=%.2f', v));
end
for v = [40 100 187 400]
  p = ref; p.nStudy = v; cells = addCell(cells, p, sprintf('list length=%d', v));
end
for v = [0.5 1.0 2.0 5.0]
  p = ref; p.crit = v; cells = addCell(cells, p, sprintf('criterion=%.1f', v));
end
% the parameter values of Fig. 1 (g 0.65, 49 features, copy fidelity 0.60)
p = ref; p.g = 0.65; p.w = 49; p.c = 0.60;
cells = addCell(cells, p, 'paper values (g=0.65 w=49 c=0.60)');

% ---- run ------------------------------------------------------------------
n = numel(cells);
fprintf('%d cells\n', n);
rows = cell(n, 1);
% Labels and numbers are kept apart and assembled at the end, so no
% concatenation of variable-length char fields is ever needed.
labels = cell(n, 1);
M = nan(n, 13);
csvPath = fullfile(OUT, 'rem_parameter_sweep.csv');
% the progress log goes to outputs/logs/
logPath = fullfile(releasePaths.logs, 'rem_parameter_sweep_progress.log');
lg = fopen(logPath, 'w'); fprintf(lg, '%d cells\n', n); fclose(lg);
% one block of per-item rows per cell (filled when saveItems)
itemRowsPerCell = cell(n, 1);

for i = 1:n
  p = cells{i};
  t0 = tic;
  r = runCell(p);
  msg = sprintf(['%3d/%3d  %-34s  rho_item=%+.3f  r_feat=%+.3f  ' ...
                 'hit=%.3f fa=%.4f  d=%.2f  (%.0fs)'], ...
                i, n, p.label, r.rhoItem, r.rFeat, r.meanHit, r.meanFA, ...
                r.dprime, toc(t0));
  fprintf('%s\n', msg);
  lg = fopen(logPath, 'a'); fprintf(lg, '%s\n', msg); fclose(lg);

  labels{i} = p.label;
  M(i, :) = [p.g, p.w, p.u, p.c, p.nStudy, p.crit, ...
             r.rhoItem, r.pItem, r.rFeat, r.pFeat, ...
             r.meanHit, r.meanFA, r.dprime];

  % keep this cell's per-item counts and distinctiveness
  if saveItems
    cellNumber = repmat(i, p.nDatabase, 1);
    itemNumber = (1:p.nDatabase)';
    itemRowsPerCell{i} = [cellNumber, itemNumber, ...
                          r.hitTrials, r.hitCount, r.faTrials, r.faCount, ...
                          r.negMeanDistanceUsed];
  end

  % rewrite after every cell, so a killed run keeps everything up to here
  writetable(buildTable(labels(1:i), M(1:i, :)), csvPath);
end

T = buildTable(labels, M);
writetable(T, csvPath);
writeReport(T, ref, fullfile(OUT, 'rem_parameter_sweep.txt'));
fprintf('\nwrote %s\n', csvPath);

% write the per-item file (exact doubles, %.17g)
if saveItems
  writeItemFile(itemRowsPerCell, fullfile(OUT, 'rem_parameter_sweep_items.csv'));
end
end


function T = buildTable(labels, M)
T = table(labels, M(:,1), M(:,2), M(:,3), M(:,4), M(:,5), M(:,6), ...
          M(:,7), M(:,8), M(:,9), M(:,10), M(:,11), M(:,12), M(:,13), ...
  'VariableNames', {'label','g','w','u','c','nStudy','crit', ...
                    'rhoItem','pItem','rFeat','pFeat','meanHit','meanFA','dprime'});
end


% =========================================================================
function cells = addCell(cells, p, label)
p.label = label;
cells{end+1} = p; %#ok<AGROW>
end


function r = runCell(p)
% One parameter cell: simulate, then relate memorability to distinctiveness.
db = geornd(p.g, [p.nDatabase, p.w]) + 1;

nTestOld = min(20, p.nStudy);
nTestNew = 20;
hitCount = zeros(p.nDatabase, 1); hitTrials = zeros(p.nDatabase, 1);
faCount  = zeros(p.nDatabase, 1); faTrials  = zeros(p.nDatabase, 1);

for s = 1:p.nSim
  perm   = randperm(p.nDatabase, p.nStudy + nTestNew);
  studyG = perm(1:p.nStudy);
  oldG   = studyG(randperm(p.nStudy, nTestOld));
  newG   = perm(p.nStudy+1:end);

  traces = encodeImage(db(studyG, :), p.g, p.u, p.c);
  famOld = meanLambda(db(oldG, :), traces, p.g, p.c);
  famNew = meanLambda(db(newG, :), traces, p.g, p.c);

  hitCount(oldG)  = hitCount(oldG)  + (famOld > p.crit);
  hitTrials(oldG) = hitTrials(oldG) + 1;
  faCount(newG)   = faCount(newG)   + (famNew > p.crit);
  faTrials(newG)  = faTrials(newG)  + 1;
end

ok = hitTrials > 0 & faTrials > 0;
hit = hitCount(ok) ./ hitTrials(ok);
fa  = faCount(ok)  ./ faTrials(ok);
mem = hit - fa;

% distinctiveness: negative mean pairwise L2 (the summed similarity of the
% Fig. S2 caption). A HIGHER value is an item CLOSER to the others on average,
% i.e. LESS distinctive, so a negative rhoItem means distinctive items are
% remembered better (see header).
dist = -meanPairwiseDistance(db(ok, :));

[r.rhoItem, r.pItem] = corr(mem, dist, 'type', 'Spearman');

% feature level: does a feature that predicts memorability also predict
% distinctiveness? This is the relationship of Fig. 1d, computed for each cell.
X = double(db(ok, :));
cm = zeros(p.w, 1); cd = zeros(p.w, 1);
for k = 1:p.w
  cm(k) = corr(X(:, k), mem, 'type', 'Spearman');
  cd(k) = corr(X(:, k), dist, 'type', 'Spearman');
end
[r.rFeat, r.pFeat] = corr(cm, cd);

r.meanHit = mean(hit); r.meanFA = mean(fa);
r.sdCR = std(mem);
za = @(q) norminv(min(max(q, 0.001), 0.999));
r.dprime = za(mean(hit)) - za(mean(fa));

% Hand back the per-item numbers the correlation used (no random numbers are
% drawn here). An item enters the correlation when it was tested at least once
% as old and once as new (ok); its distinctiveness is the value correlated
% above, NaN for an item left out.
r.hitCount = hitCount;
r.hitTrials = hitTrials;
r.faCount = faCount;
r.faTrials = faTrials;
r.negMeanDistanceUsed = nan(p.nDatabase, 1);
r.negMeanDistanceUsed(ok) = dist;
end


function image_encoded = encodeImage(image_shown, g, u, c)
featureEncodedAtAll     = rand(size(image_shown)) < u;
featureEncodedCorrectly = rand(size(image_shown)) < c;
numWrong = nnz(~featureEncodedCorrectly);
image_encoded = image_shown;
image_encoded(featureEncodedCorrectly == 0) = geornd(g, [1 numWrong]) + 1;
image_encoded(featureEncodedAtAll == 0)     = 0;
end


function fam = meanLambda(testItems, traces, g, c)
% Shiffrin & Steyvers (1997) Eq. 4A in log space, vectorized over test items.
T = size(testItems, 1); M = size(traces, 1);
log1mc = log(1 - c);
lambdas = zeros(T, M);
for j = 1:M
  tr = traces(j, :);
  storedMask = tr > 0;
  nStored = nnz(storedMask);
  if nStored == 0, lambdas(:, j) = 1; continue; end
  v      = double(tr(storedMask));
  pBase  = g .* (1 - g).^(v - 1);
  logf   = log((c + (1 - c) .* pBase) ./ pBase);
  matchMask = double(testItems(:, storedMask)) == v;
  lambdas(:, j) = exp(matchMask * logf(:) + ...
                      (nStored - sum(matchMask, 2)) * log1mc);
end
fam = mean(lambdas, 2);
end


function d = meanPairwiseDistance(X)
X = double(X); n = size(X, 1);
sq = sum(X.^2, 2);
D2 = sq + sq' - 2 * (X * X');
D2(D2 < 0) = 0;
D = sqrt(D2);
D(1:n+1:end) = NaN;
d = mean(D, 2, 'omitnan');
end


function writeReport(T, ref, path)
fid = fopen(path, 'w');
fp = @(varargin) fprintf(fid, varargin{:});
fp('REM PARAMETER SWEEP\n');
fp('%s\n', repmat('=', 1, 96));
fp(['Does the memorability-distinctiveness relationship depend on the ' ...
    'parameters chosen?\n\n']);
fp(['Reference cell: g=%.2f w=%d storage=%.2f copy=%.2f list=%d ' ...
    'criterion=%.1f, database=%d, %d runs per cell.\n'], ...
   ref.g, ref.w, ref.u, ref.c, ref.nStudy, ref.crit, ref.nDatabase, ref.nSim);
fp(['rho_item is the Spearman correlation between per-item memorability and\n' ...
    'distinctiveness. r_feat is the Figure 1d analogue: across features, the\n' ...
    'correlation between a feature''s relation to memorability and its relation\n' ...
    'to distinctiveness.\n\n']);
fp('%-34s %6s %4s %5s %5s %5s %5s | %8s %8s | %6s %6s %6s\n', ...
   'cell', 'g', 'w', 'u', 'c', 'list', 'crit', 'rho_item', 'r_feat', ...
   'hit', 'fa', 'd''');
fp('%s\n', repmat('-', 1, 118));
for i = 1:height(T)
  fp('%-34s %6.2f %4d %5.2f %5.2f %5d %5.1f | %+8.3f %+8.3f | %6.3f %6.4f %6.2f\n', ...
     T.label{i}, T.g(i), T.w(i), T.u(i), T.c(i), T.nStudy(i), T.crit(i), ...
     T.rhoItem(i), T.rFeat(i), T.meanHit(i), T.meanFA(i), T.dprime(i));
end
fp('\n%s\n', repmat('-', 1, 96));
fp('SUMMARY\n');
fp('%s\n', repmat('-', 1, 96));
fp('  rho_item ranges %+.3f to %+.3f across all %d cells; sign is %s.\n', ...
   min(T.rhoItem), max(T.rhoItem), height(T), ...
   ternary(all(sign(T.rhoItem) == sign(T.rhoItem(1))), 'constant', 'NOT constant'));
fp('  r_feat  ranges %+.3f to %+.3f; sign is %s.\n', ...
   min(T.rFeat), max(T.rFeat), ...
   ternary(all(sign(T.rFeat) == sign(T.rFeat(1))), 'constant', 'NOT constant'));
fp('  Overall accuracy moves with the parameters:\n');
fp('    d'' ranges %.2f to %.2f, hit rate %.3f to %.3f.\n', ...
   min(T.dprime), max(T.dprime), min(T.meanHit), max(T.meanHit));
sub = T(startsWith(T.label, 'grid '), :);
fp('  Within the g x w grid alone, rho_item ranges %+.3f to %+.3f while d''\n', ...
   min(sub.rhoItem), max(sub.rhoItem));
fp('    ranges %.2f to %.2f.\n', ...
   min(sub.dprime), max(sub.dprime));
iS = find(strcmp(T.label, 'Shiffrin & Steyvers reference'), 1);
iA = find(startsWith(T.label, 'paper values'), 1);
fp('  Standard S&S cell: rho_item %+.3f, r_feat %+.3f.\n', ...
   T.rhoItem(iS), T.rFeat(iS));
fp('  Paper values:      rho_item %+.3f, r_feat %+.3f.\n', ...
   T.rhoItem(iA), T.rFeat(iA));
fclose(fid);
end


function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end


% the per-item file of all cells
function writeItemFile(itemRowsPerCell, path)
% One line per cell and item. The counts are whole numbers; the distance is
% written with 17 significant digits so it reads back as the same double.
fid = fopen(path, 'w');
fprintf(fid, 'cell_number,item,old_trials,hits,new_trials,false_alarms,neg_mean_distance\n');
for cellIndex = 1:numel(itemRowsPerCell)
  rowsOfCell = itemRowsPerCell{cellIndex};
  for rowIndex = 1:size(rowsOfCell, 1)
    fprintf(fid, '%d,%d,%d,%d,%d,%d,%.17g\n', rowsOfCell(rowIndex, :));
  end
end
fclose(fid);
fprintf('wrote %s\n', path);
end
