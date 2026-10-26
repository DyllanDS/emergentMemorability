function list_binned
% Random lists, binned by how many close neighbors a test item happened to get.
%
% list_binned         runs the simulation from rng(1, 'twister') and writes its
%                     outputs to A2d_list_context_figS5/outputs/
%
% Sampling 40 items from 1,600 does vary an item's competition, just not by much
% on average: an item's expected number of near neighbors on a list is a
% little over one. Averaging that variation away, as a continuous crowding
% measure does, hides the fact that some lists deliver three or four close
% competitors and others none. Counting them instead puts random lists on the
% same footing as the designed ones, where the count is the manipulation.
%
% "Similar" means inside an item's 50 nearest neighbors in the database, which
% is the closest 3% of the space. For a studied item the count is over its 39
% companions; for a lure, over all 40 studied items.
%
% Reported per bin: hit rate, false alarm rate, corrected recognition (hits minus
% false alarms, the measure this literature uses) and d-prime. Corrected
% recognition matters here because it is what the published memorability scores
% are, and because it is the measure that turns out to move least: false alarms
% and hits shift together, and subtracting one from the other removes much of
% what the model actually does.
%
% Both a raw and a within-item version are given. The within-item version
% subtracts each item's own mean across the lists it appeared on, so the item is
% held fixed and only the count varies.
%
% The script runs two parameter cells in order. CELL 2 (g 0.65, 49 features,
% storage 0.30, copy fidelity 0.60, criterion 1.0, the parameter values of
% Fig. 1) is the simulation of SI Appendix, Fig. S5. CELL 1 (the standard values
% of Shiffrin and Steyvers, 1997: g 0.40, 20 features, copy fidelity 0.70) runs
% first, and CELL 2 draws its random numbers after it, so CELL 2 reproduces only
% when CELL 1 has run before it in the same MATLAB session.
% code/check_against_paper_run.py compares the printed table
% (outputs/list_binned.txt) with the saved run (data/paper_run/list_binned.txt)
% line by line.
%
% HOW THE SIMULATION WORKS, IN PLAIN WORDS (per cell)
%   - A database of 1,600 simulated items, each a list of w features; every
%     feature value is a geometric draw (parameter g) plus 1 (variables db, X).
%   - Closeness = Euclidean distance between two items' feature lists. An
%     item's close neighbors are its 50 nearest items in the database
%     (isNear), fixed by the database and independent of any list.
%   - 600 lists. For each list, 60 different items are drawn at random from the
%     1,600: 40 are studied and 20 are lures (tested, never studied). The list
%     is studied by 120 simulated participants (REM encoding: each feature
%     stored with probability u; a stored feature copied correctly with
%     probability c, otherwise replaced by a new geometric value counted from
%     1). At test, each of the 60 test items is compared with every stored
%     trace; the mean likelihood ratio over the 40 traces is its familiarity,
%     and the simulated participant says "old" when it is above the criterion
%     (crit = 1.0).
%   - For each studied item: its hit rate over the list's 120 simulated
%     participants (oHit) and how many of its 39 list companions are among its
%     close neighbors (oCnt). For each lure: its false alarm rate (lFA) and how
%     many of the 40 studied items are among its close neighbors (lCnt).
%   - Test items are binned by that count (0, 1, 2, 3 or more). In each bin,
%     hit rate = the mean of oHit over the studied items in the bin and
%     false alarm rate = the mean of lFA over the lures in the bin; d-prime =
%     z(hit rate) minus z(false alarm rate), rates kept within 0.0001 and
%     0.9999. Fig. S5 compares the 3+ bin with the 0 bin.
%   - The printed table also lists the mean number of close neighbors per
%     studied item and per lure, and the within-item version (bands of an
%     item's count relative to its own mean count, and the Spearman
%     correlations of the centered count with the centered hit and false alarm
%     rates).
%
% How to run (from the repository root; MATLAB R2023b with the Statistics and
% Machine Learning Toolbox; about 35 seconds; no parallel pool):
%   matlab -batch "restoredefaultpath; cd('A2d_list_context_figS5/code'); list_binned"
% or through python A2d_list_context_figS5/code/run_module.py
% (restoredefaultpath, for this MATLAB session only, removes folders added to
% the MATLAB path, so that they cannot replace MATLAB's own functions).
%
% Inputs: none (the simulated items are generated here).
% Outputs (A2d_list_context_figS5/outputs/):
%   list_binned.txt          the printed table, both cells
%   list_binned_cell2.mat    the per-item results of CELL 2 (MATLAB v7,
%                            readable by scipy.io.loadmat; CELL 1 saves none):
%     oItem, oHit, oCnt, oList   one row per studied item of every list
%                                (600 x 40 = 24,000 rows): its database number
%                                (1 to 1,600), hit rate, number of close
%                                neighbors among its 39 list companions, and
%                                the list number (1 to 600)
%     lItem, lFA, lCnt, lList    one row per lure of every list (600 x 20 =
%                                12,000 rows): database number, false alarm
%                                rate, number of close neighbors among the 40
%                                studied items, list number
%     parameters                 the cell's settings: name, g, w, u, c, crit
%                                (the criterion), nStudy, nLure, nDatabase,
%                                nLists, nSubjPerList, nNeighbour, seed

% The seed of the run: rng(1, 'twister'), the Mersenne Twister (MATLAB's
% default generator). It is also saved with the settings of CELL 2.
seed = 1;
fprintf('seed %d\n', seed);

% Warn when a folder on the MATLAB path replaces a function of MATLAB or of the
% Statistics and Machine Learning Toolbox that this script relies on.
% run_module.py starts MATLAB with restoredefaultpath, so only MATLAB's own
% functions are used. Looking up a function draws no random numbers.
functionsUsed = {'geornd', 'randperm', 'rand', 'pdist', 'squareform', 'accumarray', ...
                 'corr', 'tcdf', 'norminv'};
for functionIndex = 1:numel(functionsUsed)
  functionFile = which(functionsUsed{functionIndex});
  if isempty(functionFile)
    error('list_binned:missingFunction', ...
          '%s was not found; the simulation needs the Statistics and Machine Learning Toolbox', ...
          functionsUsed{functionIndex});
  end
  if ~startsWith(functionFile, matlabroot) && ~startsWith(functionFile, 'built-in')
    warning('list_binned:shadowedFunction', ...
            ['%s is taken from %s, not from MATLAB''s own toolboxes; ' ...
             'run restoredefaultpath first to reproduce the saved table in data/paper_run/'], ...
            functionsUsed{functionIndex}, functionFile);
  end
end

rng(seed, 'twister');          % one stream for both cells: CELL 1 draws first, then CELL 2

cells = { ...
  struct('name', 'Shiffrin & Steyvers standard parameters', ...
         'g', 0.40, 'w', 20, 'u', 0.30, 'c', 0.70, 'crit', 1.0), ...
  struct('name', 'paper values: g 0.65, w 49, c 0.60', ...
         'g', 0.65, 'w', 49, 'u', 0.30, 'c', 0.60, 'crit', 1.0)};

% outputs go to this module's outputs/ folder, found from this file's location
% (shared/matlab/release_paths.m)
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared', 'matlab'));
releasePaths = release_paths(mfilename('fullpath'));
out = releasePaths.outputs;
if ~exist(out, 'dir'); mkdir(out); end
fid = fopen(fullfile(out, 'list_binned.txt'), 'w');
fp = @(varargin) both(fid, varargin{:});

fp('RANDOM LISTS, BINNED BY NUMBER OF CLOSE NEIGHBORS ON THE LIST\n');
fp([repmat('=', 1, 92) '\n']);
fp(['Lists are drawn at random. Each test item is then binned by how many of the\n' ...
    'studied items fall inside its 50 nearest neighbors in the database.\n' ...
    'This is the same quantity the designed lists manipulate, occurring here by\n' ...
    'chance.\n\n']);

for ci = 1:numel(cells)
  p = cells{ci};
  p.nStudy = 40; p.nLure = 20; p.nDatabase = 1600;
  p.nLists = 600; p.nSubjPerList = 120; p.nNeighbour = 50;
  fp([repmat('-', 1, 92) '\n']);
  fp('CELL %d: %s\n', ci, p.name);
  fp('g=%.2f w=%d u=%.2f c=%.2f crit=%.2f\n', p.g, p.w, p.u, p.c, p.crit);
  fp([repmat('-', 1, 92) '\n']);
  % where this cell's per-item rates are saved: CELL 2 only (an empty name
  % saves nothing); the seed is saved with the settings (it is not used
  % inside the simulation)
  p.saveFile = '';
  if ci == 2
    p.saveFile = fullfile(out, 'list_binned_cell2.mat');
  end
  p.seed = seed;
  runBinned(p, fp);
end

fp([repmat('-', 1, 92) '\n']);
fp('READING THIS\n');
fp([repmat('-', 1, 92) '\n']);
fp(['  The count of close neighbors is what the designed lists set deliberately\n' ...
    '  and what random lists deliver by chance, so the two analyses can be read\n' ...
    '  against each other directly.\n' ...
    '  Watch corrected recognition against d-prime. Hits and false alarms both\n' ...
    '  rise with the count, so their difference moves far less than d-prime does.\n']);
fclose(fid);
fprintf('\nwrote %s\n', fullfile(out, 'list_binned.txt'));
end


function runBinned(p, fp)
fprintf('  %s\n', p.name);
db = geornd(p.g, [p.nDatabase, p.w]) + 1;
X  = double(db);
pj       = p.g .* (1 - p.g) .^ (X - 1);
logRatio = log((p.c + (1 - p.c) .* pj) ./ pj);
logMiss  = log(1 - p.c);

% neighborhood membership, fixed by the database and independent of any list
D = squareform(pdist(X));
D(1:p.nDatabase+1:end) = Inf;
[~, ord] = sort(D, 2);
isNear = false(p.nDatabase);
rows = repmat((1:p.nDatabase)', 1, p.nNeighbour);
isNear(sub2ind(size(isNear), rows(:), reshape(ord(:, 1:p.nNeighbour), [], 1))) = true;

nO = p.nLists * p.nStudy;  nL = p.nLists * p.nLure;
oItem = zeros(nO,1); oHit = zeros(nO,1); oCnt = zeros(nO,1);
lItem = zeros(nL,1); lFA  = zeros(nL,1); lCnt = zeros(nL,1);

t0 = tic;
for L = 1:p.nLists
  perm   = randperm(p.nDatabase, p.nStudy + p.nLure);
  studyG = perm(1:p.nStudy);
  lureG  = perm(p.nStudy+1:end);
  testG  = [studyG(:); lureG(:)];
  nT = numel(testG); S = p.nSubjPerList;
  studyX = X(studyG, :);

  tr = repmat(reshape(studyX, [1, p.nStudy, p.w]), [S, 1, 1]);
  stored  = rand(size(tr)) < p.u;
  correct = rand(size(tr)) < p.c;
  tr(~correct) = geornd(p.g, [nnz(~correct), 1]) + 1;
  tr(~stored)  = 0;

  nPresent = sum(tr > 0, 3);
  W = zeros(nT, S, p.nStudy); N = zeros(nT, S, p.nStudy);
  Tv = X(testG, :);
  for j = 1:p.w
    Rj = reshape(tr(:, :, j), [1, S, p.nStudy]);
    m  = bsxfun(@eq, Tv(:, j), Rj) & (Rj > 0);
    W  = W + bsxfun(@times, m, logRatio(testG, j));
    N  = N + m;
  end
  logL = W + logMiss .* bsxfun(@minus, reshape(nPresent, [1, S, p.nStudy]), N);
  said = mean(exp(logL), 3) > p.crit;

  % how many studied items sit inside each test item's neighborhood
  cnt = sum(isNear(testG, studyG), 2);

  io = (L-1)*p.nStudy + (1:p.nStudy);
  oItem(io) = studyG(:); oHit(io) = mean(said(1:p.nStudy, :), 2);
  oCnt(io)  = cnt(1:p.nStudy);

  il = (L-1)*p.nLure + (1:p.nLure);
  lItem(il) = lureG(:); lFA(il) = mean(said(p.nStudy+1:end, :), 2);
  lCnt(il)  = cnt(p.nStudy+1:end);

  if mod(L, 200) == 0, fprintf('    list %d/%d (%.0f s)\n', L, p.nLists, toc(t0)); end
end

% save every test item's rate, count and list number (for the bootstrap
% intervals) and the cell's settings, without the file name; MATLAB v7 so that
% scipy.io.loadmat reads it. Only CELL 2 has a file name.
if ~isempty(p.saveFile)
  oList = repelem((1:p.nLists)', p.nStudy); lList = repelem((1:p.nLists)', p.nLure);
  parameters = rmfield(p, 'saveFile');
  save(p.saveFile, 'oItem', 'oHit', 'oCnt', 'oList', 'lItem', 'lFA', 'lCnt', 'lList', 'parameters', '-v7');
end
fp('%d lists x %d participants; "similar" = inside an item''s %d nearest neighbors.\n', ...
   p.nLists, p.nSubjPerList, p.nNeighbour);
fp('mean close neighbors per studied item: %.2f (lures %.2f)\n\n', ...
   mean(oCnt), mean(lCnt));

edges = [0 1 2 3 Inf];
lab = {'0', '1', '2', '3+'};
za = @(q) norminv(min(max(q, 1e-4), 1 - 1e-4));

fp('BY COUNT OF CLOSE NEIGHBORS ON THE LIST\n');
fp('  similar     n(old)    n(new)      hit       FA    hits-FA   d-prime\n');
H = zeros(numel(lab),1); F = H;
for k = 1:numel(lab)
  mo = oCnt >= edges(k) & oCnt < edges(k+1);
  ml = lCnt >= edges(k) & lCnt < edges(k+1);
  H(k) = mean(oHit(mo)); F(k) = mean(lFA(ml));
  fp('  %7s %10d %9d %8.3f %8.3f %10.3f %9.3f\n', ...
     lab{k}, nnz(mo), nnz(ml), H(k), F(k), H(k)-F(k), za(H(k))-za(F(k)));
end
fp('  0 -> 3+:                        %+8.3f %+8.3f %+10.3f %+9.3f\n\n', ...
   H(end)-H(1), F(end)-F(1), (H(end)-F(end))-(H(1)-F(1)), ...
   (za(H(end))-za(F(end))) - (za(H(1))-za(F(1))));

% ---- within item: each item's own mean removed, so only the list varies ----
[cOd, cOh] = centreBy(oItem, oCnt, oHit, p.nDatabase);
[cLd, cLf] = centreBy(lItem, lCnt, lFA,  p.nDatabase);
mH = mean(oHit); mF = mean(lFA);
edgesW = [-Inf -1 -0.001 0.999 Inf];
labW = {'below own mean', 'slightly below', 'at/above', 'well above'};
fp('WITHIN ITEM (each item''s own mean removed; only which list it landed on varies)\n');
fp('  band                  n(old)    n(new)      hit       FA    hits-FA   d-prime\n');
for k = 1:numel(labW)
  mo = cOd >= edgesW(k) & cOd < edgesW(k+1);
  ml = cLd >= edgesW(k) & cLd < edgesW(k+1);
  if nnz(mo) < 50 || nnz(ml) < 50, continue; end
  h = mH + mean(cOh(mo)); f = mF + mean(cLf(ml));
  fp('  %-20s %8d %9d %8.3f %8.3f %10.3f %9.3f\n', ...
     labW{k}, nnz(mo), nnz(ml), h, f, h-f, za(h)-za(f));
end
[rh, ph] = corr(cOd, cOh, 'type', 'Spearman');
[rf, pf] = corr(cLd, cLf, 'type', 'Spearman');
fp('\n  within item: count vs hit rho = %+.3f (p = %.3g); count vs FA rho = %+.3f (p = %.3g)\n\n', ...
   rh, ph, rf, pf);
end


function [cD, cY] = centreBy(item, d, y, nDb)
mD = accumarray(item, d, [nDb 1], @mean, NaN);
mY = accumarray(item, y, [nDb 1], @mean, NaN);
n  = accumarray(item, 1, [nDb 1]);
ok = n(item) >= 2;
cD = d(ok) - mD(item(ok));
cY = y(ok) - mY(item(ok));
end


function both(fid, varargin)
fprintf(varargin{:});
fprintf(fid, varargin{:});
end
