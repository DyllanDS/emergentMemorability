%RUN_MODULE  Module 03 (paper Fig. 3, the memorability inversion experiment): every MATLAB step.
%
%   Run from the repository root:
%       matlab -batch "restoredefaultpath; cd('03_inversion_experiment_fig3/code'); run_module"
%   Then the uncertainty statistics (Python, under a minute):
%       python 03_inversion_experiment_fig3/code/stats_fig3.py
%
%   The steps (each is a function in this folder and writes its own files to outputs/):
%     1. inversion_design_check   the design of Fig. 3a to 3c and of the Methods, counted from the
%                                 deployed lists and the trial records
%     2. participant_flow         54 sessions, the exclusions by rule, 50 kept (Methods)
%     3. inversion_analysis       Fig. 3d and 3e: the per-item table and the Spearman rank
%                                 correlation between the published memorability (Kramer et al.
%                                 2023) and corrected recognition in this task over the 40 critical
%                                 items, compared with the saved run in data/paper_run/
%     4. write_module_numbers     outputs/numbers.csv, one row per number of the paper this module
%                                 reproduces (stats_fig3.py adds the two rows of the Fig. 3e interval)
%
%   No random numbers. Runtime: under a minute on a laptop (Apple M2 Pro).
%   A log of the whole run is written to outputs/logs/run_module_log.txt.

codeFolder = fileparts(mfilename('fullpath'));
addpath(codeFolder);
addpath(fullfile(codeFolder, '..', '..', 'shared', 'matlab'));
paths = release_paths(mfilename('fullpath'));

logFile = fullfile(paths.logs, 'run_module_log.txt');
if exist(logFile, 'file')
    delete(logFile);
end
diary(logFile);
runTimer = tic;
fprintf('Module 03 (Fig. 3), MATLAB %s, started %s\n\n', version, char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm')));

results = struct();

fprintf('\n===== Step 1: the design, counted from the deployed lists and the trial records =====\n');
results.design = inversion_design_check();

fprintf('\n===== Step 2: participant flow =====\n');
results.flow = participant_flow();

fprintf('\n===== Step 3: Fig. 3d and 3e =====\n');
results.fig3 = inversion_analysis();

fprintf('\n===== Step 4: numbers.csv =====\n');
write_module_numbers(results);

close all;
fprintf('\nModule 03 finished in %.1f minutes.\n', toc(runTimer) / 60);
diary off;

% The log names files by their path in the repository, not by this computer's folders
releaseFolder = fileparts(paths.module);
logText = fileread(logFile);
logText = strrep(logText, [releaseFolder filesep], '');
logText = strrep(logText, releaseFolder, '.');
logHandle = fopen(logFile, 'w');
fprintf(logHandle, '%s', logText);
fclose(logHandle);
