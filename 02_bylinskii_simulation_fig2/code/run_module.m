function run_module(runMode)
%RUN_MODULE  Module 02 (Fig. 2): the REM simulation of the within-category and across-category
%   memorability experiments of Bylinskii et al. (2015), both panels of Fig. 2, and
%   outputs/numbers.csv.
%
%   Run from the repository root (MATLAB R2023b with the Statistics and Machine Learning Toolbox;
%   everything runs serially, no parallel pool):
%     matlab -batch "restoredefaultpath; cd('02_bylinskii_simulation_fig2/code'); run_module"                 (full, about 15 min)
%     matlab -batch "restoredefaultpath; cd('02_bylinskii_simulation_fig2/code'); run_module('skip_long')"    (short, under 1 min)
%     matlab -batch "restoredefaultpath; cd('02_bylinskii_simulation_fig2/code'); run_module('report')"       (redraw only, under 1 min)
%   and then the statistics of tests T03 to T05 and the numbers.csv rows of the Fig. 2 correlations (Python):
%     python 02_bylinskii_simulation_fig2/code/stats_fig2.py
%
%   Modes
%     'full' (default)  runs the simulation with seed 47826 and 1,000 runs per context, saves it to
%                       outputs/simulation_runs/seed_47826_runs_1000_figureData.mat, compares it field
%                       by field with the saved run in data/paper_run/fig2_paper_run.mat, and draws
%                       Fig. 2 and writes numbers.csv from the new run (mode 'rerun').
%     'skip_long'       runs the same simulation code with 10 runs per context, only to check that
%                       the code runs (its results are not kept), then draws Fig. 2 and writes
%                       numbers.csv from the saved run in data/paper_run/ (mode 'saved_run').
%     'report'          no new simulation: draws Fig. 2, writes numbers.csv and repeats the
%                       field-by-field comparison from the run that run_module('full') saved in
%                       outputs/simulation_runs/ (mode 'rerun'); stops with an error when that file
%                       does not exist.
%
%   Outputs
%     outputs/figures/fig2a_memorability_within_vs_across.png, fig2b_summed_similarity_within_vs_across.png
%     outputs/figure_data/ (the plotted numbers, one CSV per panel)
%     outputs/numbers.csv (22 of the 30 rows of this module; stats_fig2.py adds the 8 rows of the Fig. 2 correlations)
%     outputs/simulation_runs/seed_47826_runs_1000_figureData.mat (full mode)
%     outputs/logs/run_summary_<mode>.txt and, in full and report modes, the field-by-field
%     comparison outputs/logs/seed_47826_runs_1000_figureData_vs_paper_run.csv

if nargin < 1 || isempty(runMode)
    runMode = 'full';
end
codeFolder = fileparts(mfilename('fullpath'));
addpath(codeFolder);
addpath(fullfile(codeFolder, '..', '..', 'shared', 'matlab'));
paths = release_paths(mfilename('fullpath'));

paperRunFile = fullfile(paths.paperRun, 'fig2_paper_run.mat');
paperSeed = 47826;               % the seed of the run the paper reports
paperNumberOfRuns = 1000;        % runs per category (within) and in total (across) in the paper
reducedNumberOfRuns = 10;        % the short check of 'skip_long'

moduleTimer = tic;
summaryLines = {};
summaryLines{end + 1} = sprintf('Module 02 run summary, mode ''%s''', runMode);
summaryLines{end + 1} = sprintf('date: %s', datestr(now, 'yyyy-mm-dd HH:MM'));
summaryLines{end + 1} = sprintf('MATLAB %s', version);

switch runMode
    case 'full'
        fprintf('Module 02, full run: seed %d, %d runs per context (about 15 minutes)\n\n', ...
            paperSeed, paperNumberOfRuns);
        simulationTimer = tic;
        reportedRunFile = bylinskiiRemSim(paperNumberOfRuns, [], paperSeed);
        simulationSeconds = toc(simulationTimer);
        loaded = load(reportedRunFile, 'figureData');
        codeParameters = loaded.figureData.parameters;
        [~, runName, runExtension] = fileparts(reportedRunFile);
        numbersMode = 'rerun';
        sourceNote = sprintf('Computed from the full simulation run (seed %d, %d runs per context, outputs/simulation_runs/%s).', ...
            paperSeed, paperNumberOfRuns, [runName runExtension]);
        summaryLines{end + 1} = sprintf('simulation: seed %d, %d runs per context, %.0f s (%.1f min)', ...
            paperSeed, paperNumberOfRuns, simulationSeconds, simulationSeconds / 60);
        summaryLines{end + 1} = sprintf('reported run: outputs/simulation_runs/%s', [runName runExtension]);

    case 'skip_long'
        fprintf(['Module 02, short run: the full simulation (about 15 minutes) is skipped; a run with %d ' ...
            'runs per context checks the code, and Fig. 2 and numbers.csv come from the saved run in data/paper_run/\n\n'], ...
            reducedNumberOfRuns);
        checkFile = fullfile(tempdir, 'bylinskii_reduced_check_figureData.mat');
        simulationTimer = tic;
        bylinskiiRemSim(reducedNumberOfRuns, checkFile, paperSeed);
        simulationSeconds = toc(simulationTimer);
        loaded = load(checkFile, 'figureData');
        reducedCheck = loaded.figureData;
        delete(checkFile);
        codeParameters = reducedCheck.parameters;
        reportedRunFile = paperRunFile;
        numbersMode = 'saved_run';
        sourceNote = ['Computed from the saved run in data/paper_run/fig2_paper_run.mat; ' ...
            'the full simulation does not run in skip_long mode.'];
        summaryLines{end + 1} = sprintf(['reduced check: seed %d, %d runs per context, %.0f s, %d targets ' ...
            'tested in both contexts, Spearman rho (memorability within vs across) = %.3f. This short run ' ...
            'only checks the code; the reported numbers come from the saved run.'], paperSeed, ...
            reducedNumberOfRuns, simulationSeconds, sum(reducedCheck.validBoth), reducedCheck.rho_spearman);
        summaryLines{end + 1} = 'reported run: data/paper_run/fig2_paper_run.mat (the saved run the paper reports)';

    case 'report'
        reportedRunFile = fullfile(paths.outputs, 'simulation_runs', ...
            sprintf('seed_%d_runs_%d_figureData.mat', paperSeed, paperNumberOfRuns));
        [~, runName, runExtension] = fileparts(reportedRunFile);
        if ~exist(reportedRunFile, 'file')
            error('run_module:NoFreshRun', ['There is no full run in outputs/simulation_runs/%s; ' ...
                'run run_module(''full'') first.'], [runName runExtension]);
        end
        fprintf(['Module 02, report: no new simulation; Fig. 2 and numbers.csv from the full run ' ...
            'saved in outputs/simulation_runs/%s\n\n'], [runName runExtension]);
        loaded = load(reportedRunFile, 'figureData');
        codeParameters = loaded.figureData.parameters;
        numbersMode = 'rerun';
        sourceNote = sprintf('Computed from the full simulation run (seed %d, %d runs per context, outputs/simulation_runs/%s).', ...
            paperSeed, paperNumberOfRuns, [runName runExtension]);
        fileInformation = dir(reportedRunFile);
        summaryLines{end + 1} = sprintf(['no new simulation: reports the full run saved in ' ...
            'outputs/simulation_runs/%s (written %s)'], [runName runExtension], ...
            datestr(fileInformation.datenum, 'yyyy-mm-dd HH:MM'));
        summaryLines{end + 1} = sprintf('reported run: outputs/simulation_runs/%s', [runName runExtension]);

    otherwise
        error('run_module:UnknownMode', ...
            'Unknown mode ''%s''; use ''full'', ''skip_long'' or ''report''.', runMode);
end

%% Fig. 2 and numbers.csv from the reported run
bylinskii_makeFigure2(reportedRunFile);
allIdentical = bylinskii_numbers(reportedRunFile, numbersMode, sourceNote, codeParameters);
if strcmp(runMode, 'full') || strcmp(runMode, 'report')
    if allIdentical
        summaryLines{end + 1} = sprintf(['equal to the saved run in data/paper_run/ in every field, p values ' ...
            'below 1e-15 counted as equal (outputs/logs/%s_vs_paper_run.csv): yes'], runName);
    else
        summaryLines{end + 1} = sprintf(['equal to the saved run in data/paper_run/: NO, see ' ...
            'outputs/logs/%s_vs_paper_run.csv; the paper reports the numbers of the saved run'], runName);
        warning('run_module:NotIdentical', 'This run differs from the saved run in data/paper_run/fig2_paper_run.mat.');
    end
end

%% How the numbers compare with the paper (counts of the match column of numbers.csv)
numbersOptions = detectImportOptions(fullfile(paths.outputs, 'numbers.csv'), 'Delimiter', ',', 'TextType', 'char');
numbersOptions = setvartype(numbersOptions, numbersOptions.VariableNames, 'char');
numbersTable = readtable(fullfile(paths.outputs, 'numbers.csv'), numbersOptions);
matchWords = unique(numbersTable.match);
countTexts = {};
for wordIndex = 1:numel(matchWords)
    countTexts{end + 1} = sprintf('%s %d', matchWords{wordIndex}, sum(strcmp(numbersTable.match, matchWords{wordIndex}))); %#ok<AGROW>
end
summaryLines{end + 1} = sprintf('numbers.csv: %d rows (%s), mode %s', height(numbersTable), strjoin(countTexts, ', '), numbersMode);
summaryLines{end + 1} = 'figures: outputs/figures/fig2a_memorability_within_vs_across.png, outputs/figures/fig2b_summed_similarity_within_vs_across.png';
summaryLines{end + 1} = sprintf('total time: %.0f s', toc(moduleTimer));
writeSummary(summaryLines, fullfile(paths.logs, sprintf('run_summary_%s.txt', runMode)));
end

function writeSummary(summaryLines, summaryFile)
% the run summary: printed and written to outputs/logs/ (paths relative to the module folder)
handle = fopen(summaryFile, 'w');
for lineIndex = 1:numel(summaryLines)
    fprintf(handle, '%s\n', summaryLines{lineIndex});
    fprintf('%s\n', summaryLines{lineIndex});
end
fclose(handle);
[~, summaryName, summaryExtension] = fileparts(summaryFile);
fprintf('wrote outputs/logs/%s\n', [summaryName summaryExtension]);
end
