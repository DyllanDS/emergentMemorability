function bylinskii_robustness(saveItems)
% BYLINSKII_ROBUSTNESS  The 18 REM simulations of SI Appendix, Fig. S3.
%
% Repeats the REM simulation of the Bylinskii et al. (2015) context
% manipulation (Fig. 2a) 18 times, varying one factor at a time (SI Appendix,
% Section 2, Category Structure and Decision Criteria). The quantity of
% interest is the Spearman correlation between the same target items'
% memorability in a within-category context and in an across-category
% context: the correspondence that Bylinskii et al. report as evidence that
% memorability is independent of context, and that Fig. 2a reproduces in a
% model with no parameter for intrinsic memorability.
%
% Fig. 2a uses a criterion of 20 for the within-category lists and 5 for
% the across-category lists. Four cells use the same criterion in both
% contexts. If the correspondence survives in these cells, the difference
% between the two criteria is not what produces it. The category structure
% is varied too: how many features define a category, how many categories
% there are, and how many items each category contains.
%
% Reference cell (the first cell; every other cell sets one factor and keeps
% the rest):
% REM with g = 0.40 and copy fidelity c = 0.70 (the standard values of
% Shiffrin and Steyvers, 1997), storage probability u = 0.30 and 49
% features; 21 categories of 300 items, of which the first 75 are the
% targets; 10 category-defining features; criteria 20 (within) and 5
% (across). Within-category context: for each category, 250 study lists of
% 30 of its targets and 90 of its fillers. Across-category context: 5,000
% study lists of 30 targets and 90 fillers drawn from all categories. In both
% contexts, up to 30 targets that were not studied in the list are tested as
% foils.
%
% Random numbers: the generator is seeded once (seed 11, Mersenne Twister),
% then the 18 cells run serially in the order below, so the draws of a cell
% depend on every cell before it (never run the cells in a parallel pool).
% The check of the MATLAB path and the per-target file (saveItems) draw no
% random numbers.
%
% How to run (from the repository root, MATLAB R2023b with the Statistics and
% Machine Learning Toolbox; about 3.5 minutes):
%   matlab -batch "restoredefaultpath; cd('A2b_category_structure_criteria_figS3/code'); bylinskii_robustness(true)"
% or through python A2b_category_structure_criteria_figS3/code/run_module.py
% (restoredefaultpath, for this MATLAB session only, removes folders added to
% the MATLAB path, so that they cannot replace MATLAB's own functions).
%
% Input: saveItems (default true) also writes the per-target file. The
% simulated items are generated here.
% Outputs (in A2b_category_structure_criteria_figS3/outputs/):
%   bylinskii_robustness.csv        one row per cell: label,
%                                   critWithin, critAcross, catFeatures, nCat,
%                                   nPerCat, rho, pRho, simCorr, hitWithin,
%                                   faWithin, hitAcross, faAcross, dWithin,
%                                   dAcross
%   bylinskii_robustness.txt        a printed report of the same table
%   bylinskii_robustness_items.csv  (saveItems) one row per cell and target
%                                   item: cell_number (row of the csv), target
%                                   (1 to the number of targets), category,
%                                   within_old_trials, within_hits,
%                                   within_new_trials, within_false_alarms,
%                                   across_old_trials, across_hits,
%                                   across_new_trials, across_false_alarms,
%                                   summed_similarity_within (minus the mean
%                                   Euclidean distance to the other items of
%                                   the target's category),
%                                   summed_similarity_across (minus the mean
%                                   Euclidean distance to all items of the set,
%                                   the target itself included); both NaN for
%                                   a target that was not tested as old and as
%                                   new in both contexts
%   logs/bylinskii_progress.log     one line per finished cell

% saveItems (default true): also write the per-target file
if nargin < 1
    saveItems = true;
end

% Warn when a folder on the MATLAB path replaces a function of MATLAB or of
% the Statistics and Machine Learning Toolbox that this script relies on.
% FieldTrip's external/stats/tcdf.m, for example, returns 0 for p values
% below about 1e-16, which would change the pRho column. run_module.py starts
% MATLAB with restoredefaultpath, so only MATLAB's own functions are used.
% Looking up a function draws no random numbers.
functionsUsed = {'geornd', 'randperm', 'rand', 'corr', 'tcdf', 'norminv', 'writetable'};
for functionIndex = 1:numel(functionsUsed)
    functionFile = which(functionsUsed{functionIndex});
    if isempty(functionFile)
        error('bylinskii_robustness:missingFunction', ...
              '%s was not found; the sweep needs the Statistics and Machine Learning Toolbox', ...
              functionsUsed{functionIndex});
    end
    if ~startsWith(functionFile, matlabroot) && ~startsWith(functionFile, 'built-in')
        warning('bylinskii_robustness:shadowedFunction', ...
                ['%s is taken from %s, not from MATLAB''s own toolboxes; ' ...
                 'run restoredefaultpath first to reproduce the saved run in data/paper_run/'], ...
                functionsUsed{functionIndex}, functionFile);
    end
end

rng(11, 'twister');

% the module's outputs/ folder, found from this file's location
% (shared/matlab/release_paths.m)
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared', 'matlab'));
releasePaths = release_paths(mfilename('fullpath'));
OUT  = releasePaths.outputs;
if ~exist(OUT, 'dir'), mkdir(OUT); end

% ---- reference cell: the category structure and criteria of Fig. 2a ------
ref = struct('g', 0.40, 'w', 49, 'u', 0.30, 'c', 0.70, ...
             'nCat', 21, 'nPerCat', 300, 'nTargetsPerCat', 75, ...
             'nCatFeatures', 10, 'nStudyTargets', 30, 'nStudyFillers', 90, ...
             'critWithin', 20, 'critAcross', 5, ...
             'nRunsWithin', 250, 'nRunsAcross', 5000);

cells = {};
cells = add(cells, ref, 'reference (crit 20 / 5)');

% criterion pairs (within / across); the last four use the same criterion in
% both contexts
for pr = {[20 5], [10 2.5], [40 10], [5 5], [20 20], [1 1], [50 50]}
    p = ref; p.critWithin = pr{1}(1); p.critAcross = pr{1}(2);
    cells = add(cells, p, sprintf('crit %g / %g', p.critWithin, p.critAcross));
end

% structural assumptions, one at a time
for v = [3 5 10 15]
    p = ref; p.nCatFeatures = v;
    cells = add(cells, p, sprintf('category-defining features = %d', v));
end
for v = [10 21 40]
    p = ref; p.nCat = v;
    cells = add(cells, p, sprintf('categories = %d', v));
end
for v = [150 300 600]
    p = ref; p.nPerCat = v; p.nTargetsPerCat = round(v / 4);
    cells = add(cells, p, sprintf('items per category = %d', v));
end

n = numel(cells);
fprintf('%d cells\n', n);
csvPath = fullfile(OUT, 'bylinskii_robustness.csv');
% the progress log goes to outputs/logs/
logPath = fullfile(releasePaths.logs, 'bylinskii_progress.log');
lg = fopen(logPath, 'w'); fprintf(lg, '%d cells\n', n); fclose(lg);

% one block of per-target rows per cell (filled when saveItems)
itemRowsPerCell = cell(n, 1);

labels = cell(n, 1);
M = nan(n, 14);
for i = 1:n
    p = cells{i};
    t0 = tic;
    r = runCell(p);
    msg = sprintf(['%2d/%2d  %-34s  rho=%+.3f  simCorr=%+.3f  ' ...
                   'W hit/fa %.3f/%.3f  A hit/fa %.3f/%.3f  (%.0fs)'], ...
                  i, n, p.label, r.rho, r.simCorr, r.hitW, r.faW, ...
                  r.hitA, r.faA, toc(t0));
    fprintf('%s\n', msg);
    lg = fopen(logPath, 'a'); fprintf(lg, '%s\n', msg); fclose(lg);

    labels{i} = p.label;
    M(i, :) = [p.critWithin, p.critAcross, p.nCatFeatures, p.nCat, p.nPerCat, ...
               r.rho, r.pRho, r.simCorr, r.hitW, r.faW, r.hitA, r.faA, r.dW, r.dA];
    writetable(buildTable(labels(1:i), M(1:i, :)), csvPath);

    % keep this cell's per-target counts and summed similarities
    if saveItems
        numberOfTargets = numel(r.targetCategory);
        cellNumber = repmat(i, numberOfTargets, 1);
        targetNumber = (1:numberOfTargets)';
        itemRowsPerCell{i} = [cellNumber, targetNumber, r.targetCategory, ...
                              r.withinOldTrials, r.withinHits, ...
                              r.withinNewTrials, r.withinFalseAlarms, ...
                              r.acrossOldTrials, r.acrossHits, ...
                              r.acrossNewTrials, r.acrossFalseAlarms, ...
                              r.summedSimilarityWithin, r.summedSimilarityAcross];
    end
end

T = buildTable(labels, M);
writetable(T, csvPath);
writeReport(T, ref, fullfile(OUT, 'bylinskii_robustness.txt'));
fprintf('\nwrote %s\n', csvPath);

% write the per-target file of all cells (exact doubles, %.17g)
if saveItems
    writeItemFile(itemRowsPerCell, fullfile(OUT, 'bylinskii_robustness_items.csv'));
end
end


% =========================================================================
function cells = add(cells, p, label)
p.label = label;
cells{end+1} = p; %#ok<AGROW>
end


function T = buildTable(labels, M)
T = table(labels, M(:,1), M(:,2), M(:,3), M(:,4), M(:,5), M(:,6), M(:,7), ...
          M(:,8), M(:,9), M(:,10), M(:,11), M(:,12), M(:,13), M(:,14), ...
  'VariableNames', {'label','critWithin','critAcross','catFeatures','nCat', ...
                    'nPerCat','rho','pRho','simCorr','hitWithin','faWithin', ...
                    'hitAcross','faAcross','dWithin','dAcross'});
end


function r = runCell(p)
% Build the categorized item set, then run the two contexts over the same targets.
nItems = p.nCat * p.nPerCat;
items = zeros(nItems, p.w);
catLabel = zeros(nItems, 1);
isTarget = false(nItems, 1);
for c = 1:p.nCat
    proto = geornd(p.g, [1 p.nCatFeatures]) + 1;
    rest  = geornd(p.g, [p.nPerCat, p.w - p.nCatFeatures]) + 1;
    rows = (c-1)*p.nPerCat + (1:p.nPerCat);
    items(rows, :) = [repmat(proto, p.nPerCat, 1), rest];
    catLabel(rows) = c;
    isTarget(rows(1:p.nTargetsPerCat)) = true;
end
targetIdx = find(isTarget);
nT = numel(targetIdx);
slot = zeros(nItems, 1); slot(targetIdx) = 1:nT;

acc = @() struct('hit', zeros(nT,1), 'nH', zeros(nT,1), ...
                 'fa', zeros(nT,1), 'nF', zeros(nT,1));
W = acc(); A = acc();

% ---- within-category context ----
for c = 1:p.nCat
    rows = find(catLabel == c);
    tRows = rows(1:p.nTargetsPerCat);
    fRows = rows(p.nTargetsPerCat+1:end);
    for s = 1:p.nRunsWithin
        W = oneRun(W, items, tRows, fRows, tRows, slot, p, p.critWithin);
    end
end

% ---- across-category context: same targets, mixed-category lists ----
allT = targetIdx;
allF = find(~isTarget);
for s = 1:p.nRunsAcross
    A = oneRun(A, items, allT, allF, allT, slot, p, p.critAcross);
end

ok = W.nH > 0 & W.nF > 0 & A.nH > 0 & A.nF > 0;
memW = W.hit(ok)./W.nH(ok) - W.fa(ok)./W.nF(ok);
memA = A.hit(ok)./A.nH(ok) - A.fa(ok)./A.nF(ok);
[r.rho, r.pRho] = corr(memW, memA, 'type', 'Spearman');

% summed similarity in each context: within the category for the within
% context, across all items for the across context
tIdxOk = targetIdx(ok);
simW = zeros(numel(tIdxOk), 1);
for k = 1:numel(tIdxOk)
    same = catLabel == catLabel(tIdxOk(k));
    same(tIdxOk(k)) = false;
    simW(k) = -meanDistTo(items(tIdxOk(k), :), items(same, :));
end
simA = -meanDistTo2(items(tIdxOk, :), items);
r.simCorr = corr(simW, simA);

r.hitW = mean(W.hit(ok)./W.nH(ok)); r.faW = mean(W.fa(ok)./W.nF(ok));
r.hitA = mean(A.hit(ok)./A.nH(ok)); r.faA = mean(A.fa(ok)./A.nF(ok));
za = @(q) norminv(min(max(q,0.001),0.999));
r.dW = za(r.hitW) - za(r.faW);
r.dA = za(r.hitA) - za(r.faA);

% Hand back the per-target numbers the correlations used (no random
% numbers are drawn here). Row k belongs to target targetIdx(k). A
% target enters the correlations when it was tested at least once as old and
% once as new in both contexts (ok); its two summed similarities are the
% values correlated above, NaN for a target left out.
r.targetCategory = catLabel(targetIdx);
r.withinOldTrials = W.nH;
r.withinHits = W.hit;
r.withinNewTrials = W.nF;
r.withinFalseAlarms = W.fa;
r.acrossOldTrials = A.nH;
r.acrossHits = A.hit;
r.acrossNewTrials = A.nF;
r.acrossFalseAlarms = A.fa;
r.summedSimilarityWithin = nan(nT, 1);
r.summedSimilarityWithin(ok) = simW;
r.summedSimilarityAcross = nan(nT, 1);
r.summedSimilarityAcross(ok) = simA;
end


function S = oneRun(S, items, tPool, fPool, foilPool, slot, p, crit)
studyT = tPool(randperm(numel(tPool), p.nStudyTargets));
studyF = fPool(randperm(numel(fPool), p.nStudyFillers));
traces = encode(items([studyT; studyF], :), p.g, p.u, p.c);

foilT = setdiff(foilPool, studyT);
nFoil = min(p.nStudyTargets, numel(foilT));
foils = foilT(randperm(numel(foilT), nFoil));

famT = meanLambda(items(studyT, :), traces, p.g, p.c);
famF = meanLambda(items(foils, :),  traces, p.g, p.c);

sT = slot(studyT); sF = slot(foils);
S.hit(sT) = S.hit(sT) + (famT(:) > crit);
S.nH(sT)  = S.nH(sT) + 1;
S.fa(sF)  = S.fa(sF) + (famF(:) > crit);
S.nF(sF)  = S.nF(sF) + 1;
end


function tr = encode(x, g, u, c)
wrong = rand(size(x)) < (1 - c);
tr = x;
tr(wrong) = geornd(g, [1 nnz(wrong)]) + 1;
tr(rand(size(x)) >= u) = 0;
end


function fam = meanLambda(testItems, traces, g, c)
T = size(testItems, 1); M = size(traces, 1);
log1mc = log(1 - c);
lam = zeros(T, M);
for j = 1:M
    tr = traces(j, :);
    st = tr > 0; nSt = nnz(st);
    if nSt == 0, lam(:, j) = 1; continue; end
    v = double(tr(st));
    pB = g .* (1 - g).^(v - 1);
    lf = log((c + (1 - c) .* pB) ./ pB);
    mm = double(testItems(:, st)) == v;
    lam(:, j) = exp(mm * lf(:) + (nSt - sum(mm, 2)) * log1mc);
end
fam = mean(lam, 2);
end


function d = meanDistTo(x, Y)
d = mean(sqrt(sum((double(Y) - double(x)).^2, 2)));
end


function d = meanDistTo2(X, Y)
X = double(X); Y = double(Y);
sx = sum(X.^2, 2); sy = sum(Y.^2, 2);
D2 = sx + sy' - 2 * (X * Y');
D2(D2 < 0) = 0;
d = mean(sqrt(D2), 2);
end


function writeReport(T, ref, path)
fid = fopen(path, 'w');
fp = @(varargin) fprintf(fid, varargin{:});
fp('BYLINSKII CONTEXT MANIPULATION: %d REM SIMULATIONS\n', height(T));
fp('%s\n', repmat('=', 1, 104));
fp(['rho is the Spearman correlation, across the same target items, between\n' ...
    'memorability (hit rate minus false alarm rate) in a within-category\n' ...
    'context and in an across-category context. Bylinskii et al. (2015) report\n' ...
    '0.60 for hit rates.\n\n']);
fp(['The cells with equal criteria use the same criterion in both contexts. If\n' ...
    'the correspondence survives there, the difference between the two criteria\n' ...
    'is not what produces it.\n\n']);
fp('%-34s %6s %6s %5s %5s %6s | %7s %8s | %11s %11s | %5s %5s\n', ...
   'cell', 'critW', 'critA', 'feat', 'nCat', 'perCat', 'rho', 'simCorr', ...
   'W hit/fa', 'A hit/fa', 'dW', 'dA');
fp('%s\n', repmat('-', 1, 124));
for i = 1:height(T)
    fp(['%-34s %6g %6g %5d %5d %6d | %+7.3f %+8.3f | ' ...
        '%.3f/%.3f %.3f/%.3f | %5.2f %5.2f\n'], ...
       T.label{i}, T.critWithin(i), T.critAcross(i), T.catFeatures(i), ...
       T.nCat(i), T.nPerCat(i), T.rho(i), T.simCorr(i), ...
       T.hitWithin(i), T.faWithin(i), T.hitAcross(i), T.faAcross(i), ...
       T.dWithin(i), T.dAcross(i));
end
fp('\n%s\nSUMMARY\n%s\n', repmat('-', 1, 104), repmat('-', 1, 104));
fp('  rho ranges %+.3f to %+.3f across all %d cells; sign is %s.\n', ...
   min(T.rho), max(T.rho), height(T), ...
   tern(all(T.rho > 0), 'positive throughout', 'NOT constant'));
eq = T.critWithin == T.critAcross;
if any(eq)
    fp('  Equal-criterion cells (n = %d): rho from %+.3f to %+.3f.\n', ...
       sum(eq), min(T.rho(eq)), max(T.rho(eq)));
end
fp('  Hit rates vary from %.3f to %.3f and false alarm rates from %.3f to %.3f\n', ...
   min([T.hitWithin; T.hitAcross]), max([T.hitWithin; T.hitAcross]), ...
   min([T.faWithin; T.faAcross]), max([T.faWithin; T.faAcross]));
fp('    over the cells and both contexts (the mean rates over targets in the\n');
fp('    hitWithin, faWithin, hitAcross and faAcross columns).\n');
i0 = find(strcmp(T.label, 'reference (crit 20 / 5)'), 1);
fp('  Reference cell: rho %+.3f, summed-similarity correlation %+.3f.\n', ...
   T.rho(i0), T.simCorr(i0));
fclose(fid);
end


function o = tern(c, a, b)
if c, o = a; else, o = b; end
end


% the per-target file of all cells
function writeItemFile(itemRowsPerCell, path)
% One line per cell and target item. The counts are whole numbers; the two
% summed similarities are written with 17 significant digits so they read back
% as the same doubles (NaN for a target left out of the correlations).
fid = fopen(path, 'w');
fprintf(fid, ['cell_number,target,category,' ...
              'within_old_trials,within_hits,within_new_trials,within_false_alarms,' ...
              'across_old_trials,across_hits,across_new_trials,across_false_alarms,' ...
              'summed_similarity_within,summed_similarity_across\n']);
for cellIndex = 1:numel(itemRowsPerCell)
    rowsOfCell = itemRowsPerCell{cellIndex};
    for rowIndex = 1:size(rowsOfCell, 1)
        fprintf(fid, '%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.17g,%.17g\n', rowsOfCell(rowIndex, :));
    end
end
fclose(fid);
fprintf('wrote %s\n', path);
end
