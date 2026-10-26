function list_designed
% The same item on lists built to crowd it, and on lists built to leave it alone.
%
% list_designed       runs the simulation from rng(1, 'twister') and writes its
%                     outputs to A2d_list_context_figS5/outputs/
%
% Sampling lists at random varies an item's crowding only by chance, and over 40
% items drawn from 1,600 that variation is small. The memorability inversion
% experiment (Fig. 3) does not sample lists; it places a chosen number of
% similar items alongside each critical item. This script does the same in the
% model.
%
% For each critical item the neighbors in the database are ranked by distance.
% The nearest is held out as a lure and never studied. The next few are the
% "similar" items, and the number of them placed on the list is the manipulation:
% 0, 2, 4, 8 or 16 of the 39 companions, the rest drawn from the far half of the
% database. So across conditions the critical item is identical, the list length
% is identical, and only the company changes.
%
% Two quantities follow directly:
%
%   hit rate on the critical item: does crowding cost target detection?
%   false alarm rate on its lure: does crowding make a near neighbor of the
%                                 studied set look old?
%
% Together they give d-prime (from the mean rates over the 300 critical items)
% as a function of the list the critical items were put on, with each item held
% fixed by the design rather than by statistical control. This is the model
% counterpart of the inversion experiment, and its manipulation is much larger
% than the variation random lists produce (list_binned.m).
%
% The script runs two parameter cells in order. CELL 2 (g 0.65, 49 features,
% storage 0.30, copy fidelity 0.60, criterion 1.0, the parameter values of
% Fig. 1) is the simulation of SI Appendix, Fig. S5. CELL 1 (the standard values
% of Shiffrin and Steyvers, 1997: g 0.40, 20 features, copy fidelity 0.70) runs
% first, and CELL 2 draws its random numbers after it, so CELL 2 reproduces only
% when CELL 1 has run before it in the same MATLAB session.
% code/check_against_paper_run.py compares the printed table
% (outputs/list_designed.txt) with the saved run (data/paper_run/list_designed.txt)
% line by line.
%
% HOW THE SIMULATION WORKS, IN PLAIN WORDS (per cell)
%   - A database of 1,600 simulated items, each a list of w features; every
%     feature value is a geometric draw (parameter g) plus 1 (variables db, X).
%   - Closeness = Euclidean distance between two items' feature lists. For
%     every item the other items are sorted from nearest to farthest (nnIdx).
%   - 300 critical items are drawn at random (crit). For each critical item:
%     its nearest neighbor is its lure (tested, never studied); its next
%     nearest neighbors are its "similar" items. Each critical item is put on
%     five 40-item study lists that hold 0, 2, 4, 8 or 16 of its similar items;
%     the rest of each list is filled with items drawn from the far half of its
%     neighbor ranking.
%   - Each list is studied by 200 simulated participants (REM encoding: each
%     feature stored with probability u; a stored feature copied correctly with
%     probability c, otherwise replaced by a new geometric value counted from
%     1). At test, the critical item and its lure are each compared with every
%     stored trace; the mean likelihood ratio over the 40 traces is the item's
%     familiarity, and the simulated participant says "old" when it is above
%     the criterion (crit = 1.0).
%   - hitRate(i, k) = share of the 200 simulated participants who said "old" to
%     critical item i on its list with nSimilar(k) similar items;
%     faRate(i, k) = the same for its lure.
%   - d-prime of a condition: z(mean hit rate over the 300 items) minus z(mean
%     false alarm rate), rates kept within 1/(2 x 200) and 1 - 1/(2 x 200).
%   - The printed table also lists, for each condition, the mean crowding
%     achieved (crowd) and the response criterion -0.5 x (z(hit rate) +
%     z(false alarm rate)), and the within-item Spearman correlations of
%     crowding with the hit and false alarm rates.
%
% How to run (from the repository root; MATLAB R2023b with the Statistics and
% Machine Learning Toolbox; about 1 minute; no parallel pool):
%   matlab -batch "restoredefaultpath; cd('A2d_list_context_figS5/code'); list_designed"
% or through python A2d_list_context_figS5/code/run_module.py
% (restoredefaultpath, for this MATLAB session only, removes folders added to
% the MATLAB path, so that they cannot replace MATLAB's own functions).
%
% Inputs: none (the simulated items are generated here).
% Outputs (A2d_list_context_figS5/outputs/):
%   list_designed.txt         the printed table, both cells
%   list_designed_cell2.mat   the per-item results of CELL 2 (MATLAB v7,
%                             readable by scipy.io.loadmat; CELL 1 saves none):
%     hitRate      300 critical items x 5 conditions (0, 2, 4, 8, 16 similar
%                  items on the list): hit rate of the critical item
%     faRate       300 x 5: false alarm rate of its lure
%     crowd        300 x 5: minus the mean Euclidean distance from the critical
%                  item to the other items of its list (crowding achieved)
%     crit         1 x 300: the database numbers (1 to 1,600) of the critical
%                  items (not the criterion)
%     parameters   the cell's settings: name, g, w, u, c, crit (the
%                  criterion), nStudy, nDatabase, nCritical, nSubj, nSimilar,
%                  seed

% The seed of the run: rng(1, 'twister'), the Mersenne Twister (MATLAB's
% default generator). It is also saved with the settings of CELL 2.
seed = 1;
fprintf('seed %d\n', seed);

% Warn when a folder on the MATLAB path replaces a function of MATLAB or of the
% Statistics and Machine Learning Toolbox that this script relies on.
% run_module.py starts MATLAB with restoredefaultpath, so only MATLAB's own
% functions are used. Looking up a function draws no random numbers.
functionsUsed = {'geornd', 'randperm', 'rand', 'pdist', 'squareform', 'pdist2', ...
                 'corr', 'tcdf', 'norminv'};
for functionIndex = 1:numel(functionsUsed)
  functionFile = which(functionsUsed{functionIndex});
  if isempty(functionFile)
    error('list_designed:missingFunction', ...
          '%s was not found; the simulation needs the Statistics and Machine Learning Toolbox', ...
          functionsUsed{functionIndex});
  end
  if ~startsWith(functionFile, matlabroot) && ~startsWith(functionFile, 'built-in')
    warning('list_designed:shadowedFunction', ...
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

nSimilar = [0 2 4 8 16];

% outputs go to this module's outputs/ folder, found from this file's location
% (shared/matlab/release_paths.m)
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared', 'matlab'));
releasePaths = release_paths(mfilename('fullpath'));
out = releasePaths.outputs;
if ~exist(out, 'dir'); mkdir(out); end
fid = fopen(fullfile(out, 'list_designed.txt'), 'w');
fp = @(varargin) both(fid, varargin{:});

fp('THE SAME ITEM ON DESIGNED CROWDED AND UNCROWDED LISTS\n');
fp([repmat('=', 1, 88) '\n']);
fp(['Each critical item appears on every list; only the number of close\n' ...
    'neighbors placed alongside it changes. The item is therefore held fixed by\n' ...
    'construction, not by statistical control, so nothing about the item can\n' ...
    'produce a difference across conditions.\n\n']);

for ci = 1:numel(cells)
  p = cells{ci};
  p.nStudy = 40; p.nDatabase = 1600;
  p.nCritical = 300; p.nSubj = 200; p.nSimilar = nSimilar;
  fp([repmat('-', 1, 88) '\n']);
  fp('CELL %d: %s\n', ci, p.name);
  fp('g=%.2f w=%d u=%.2f c=%.2f crit=%.2f\n', p.g, p.w, p.u, p.c, p.crit);
  fp([repmat('-', 1, 88) '\n']);
  % where this cell's per-item rates are saved: CELL 2 only (an empty name
  % saves nothing); the seed is saved with the settings (it is not used
  % inside the simulation)
  p.saveFile = '';
  if ci == 2
    p.saveFile = fullfile(out, 'list_designed_cell2.mat');
  end
  p.seed = seed;
  runDesigned(p, fp);
end

fp([repmat('-', 1, 88) '\n']);
fp('READING THIS\n');
fp([repmat('-', 1, 88) '\n']);
fp(['  Every row is the same set of critical items. If d-prime falls as similar\n' ...
    '  items are added, then in the model an item''s memorability is a fact about\n' ...
    '  the list as much as about the item, which is what the experiments report.\n' ...
    '  Compare the size against the random-list version in list_binned.txt: if\n' ...
    '  designing the lists produces a far larger effect, then the small effect\n' ...
    '  under random sampling is a fact about sampling and not about the model.\n']);
fclose(fid);
fprintf('\nwrote %s\n', fullfile(out, 'list_designed.txt'));
end


function runDesigned(p, fp)
fprintf('  %s\n', p.name);
db = geornd(p.g, [p.nDatabase, p.w]) + 1;
X  = double(db);
pj       = p.g .* (1 - p.g) .^ (X - 1);
logRatio = log((p.c + (1 - p.c) .* pj) ./ pj);
logMiss  = log(1 - p.c);

D = squareform(pdist(X));
D(1:p.nDatabase+1:end) = Inf;
[~, nnIdx] = sort(D, 2);              % nearest first, for every item

crit = randperm(p.nDatabase, p.nCritical);
nC = numel(p.nSimilar);
hitRate = zeros(p.nCritical, nC);
faRate  = zeros(p.nCritical, nC);
crowd   = zeros(p.nCritical, nC);

t0 = tic;
for ii = 1:p.nCritical
  c = crit(ii);
  order = nnIdx(c, :);
  lure  = order(1);                   % nearest neighbor, never studied
  pool  = order(2:end);
  far   = pool(round(numel(pool)*0.5):end);   % far half, for filler draws

  for k = 1:nC
    ns  = p.nSimilar(k);
    sim = pool(1:ns);
    nFill = p.nStudy - 1 - ns;
    fill = far(randperm(numel(far), nFill));
    studyG = [c, sim, fill];
    studyX = X(studyG, :);

    S = p.nSubj;
    tr = repmat(reshape(studyX, [1, p.nStudy, p.w]), [S, 1, 1]);
    stored  = rand(size(tr)) < p.u;
    correct = rand(size(tr)) < p.c;
    tr(~correct) = geornd(p.g, [nnz(~correct), 1]) + 1;
    tr(~stored)  = 0;

    testG = [c, lure];
    Tv = X(testG, :);
    nPresent = sum(tr > 0, 3);
    W = zeros(2, S, p.nStudy); N = zeros(2, S, p.nStudy);
    for j = 1:p.w
      Rj = reshape(tr(:, :, j), [1, S, p.nStudy]);
      m  = bsxfun(@eq, Tv(:, j), Rj) & (Rj > 0);
      W  = W + bsxfun(@times, m, logRatio(testG, j));
      N  = N + m;
    end
    logL = W + logMiss .* bsxfun(@minus, reshape(nPresent, [1, S, p.nStudy]), N);
    said = mean(exp(logL), 3) > p.crit;

    hitRate(ii, k) = mean(said(1, :));
    faRate(ii, k)  = mean(said(2, :));
    % crowding achieved on this list: minus the mean Euclidean distance from
    % the critical item to the other items of the list
    dd = pdist2(X(c, :), studyX);
    crowd(ii, k) = -mean(dd(dd > 0));
  end
  if mod(ii, 100) == 0
    fprintf('    item %d/%d (%.0f s)\n', ii, p.nCritical, toc(t0));
  end
end

% save every critical item's rates (for the bootstrap intervals) and the cell's
% settings, without the file name; MATLAB v7 so that scipy.io.loadmat reads it.
% Only CELL 2 has a file name.
if ~isempty(p.saveFile)
  parameters = rmfield(p, 'saveFile');
  save(p.saveFile, 'hitRate', 'faRate', 'crowd', 'crit', 'parameters', '-v7');
end
H = mean(hitRate, 1);  F = mean(faRate, 1);
za = @(q) norminv(min(max(q, 1/(2*p.nSubj)), 1 - 1/(2*p.nSubj)));
dp = za(H) - za(F);
cr = -0.5 * (za(H) + za(F));

fp('%d critical items x %d conditions x %d participants.\n\n', ...
   p.nCritical, nC, p.nSubj);
fp('  similar items      crowding      hit      FA    d-prime   criterion\n');
for k = 1:nC
  fp('  %10d %14.3f %8.3f %7.3f %10.3f %11.3f\n', ...
     p.nSimilar(k), mean(crowd(:, k)), H(k), F(k), dp(k), cr(k));
end
fp('\n  0 -> %d similar items:  hit %+.3f   FA %+.3f   d-prime %+.3f   criterion %+.3f\n', ...
   p.nSimilar(end), H(end)-H(1), F(end)-F(1), dp(end)-dp(1), cr(end)-cr(1));

% within-item Spearman correlations across the five conditions: crowding, hit
% rate and false alarm rate, each centered on the critical item's own mean
ii = repmat((1:p.nCritical)', 1, nC);
cD = crowd - mean(crowd, 2);
cH = hitRate - mean(hitRate, 2);
cF = faRate  - mean(faRate,  2);
[rH, pH] = corr(cD(:), cH(:), 'type', 'Spearman');
[rF, pF] = corr(cD(:), cF(:), 'type', 'Spearman');
fp('\n  within item, across designed lists:  crowding vs hit rho = %+.3f (p = %.3g)\n', rH, pH);
fp('                                       crowding vs FA  rho = %+.3f (p = %.3g)\n\n', rF, pF);
end


function both(fid, varargin)
fprintf(varargin{:});
fprintf(fid, varargin{:});
end
