%RUN_MODULE  Module 06 (Figs. 6 and 7, the three image-set experiments): every MATLAB step.
%
%   Run from the repository root:
%       matlab -batch "restoredefaultpath; cd('06_image_set_experiments_fig6_fig7/code'); run_module"
%   Then the uncertainty statistics and the Fig. 6 rows of outputs/numbers.csv (Python, under 1 minute):
%       python 06_image_set_experiments_fig6_fig7/code/stats_fig6_fig7.py
%   and test T19b, the Fig. 7d reduction over the 559 concepts in both sets (Python, about 10 seconds):
%       python 06_image_set_experiments_fig6_fig7/code/shared_concepts_test.py
%
%   The steps (each is a function in this folder and writes its own files to outputs/):
%     1. export_sampled_sets             the three 1,600-slot image sets as CSV tables
%     2. reproduce_set_selection         redraws the concepts of the three image sets (Methods, Image
%                                        Set Structure Manipulations) and checks them against
%                                        data/sampled_image_sets.mat
%     3. check_task_design               blocks, list lengths and the rating scale (Methods,
%                                        Recognition Memory Task)
%     4. participant_flow                602 sessions, exclusions by rule, 450 kept (Methods,
%                                        Statistical Analysis and Exclusion Criteria; Participants)
%     5. maintained_vs_changed           Fig. 6a to 6h
%     6. maintained_vs_reduced_variance  Fig. 7b to 7d, with the participant bootstrap (rng(42), 1,000 resamples)
%     7. make_figure7a_pca               Fig. 7a
%     8. category_shares                 the reduced set's higher-level categories and the size of
%                                        the THINGS categories
%     9. write_module_numbers            outputs/numbers.csv, one row per number of the paper this
%                                        module reproduces (stats_fig6_fig7.py then adds the rows of
%                                        Fig. 6c to 6h)
%
%   Random numbers: only rng(6) in step 2 (the seed of the set selection) and rng(42) in step 6
%   (the seed of the Fig. 7d bootstrap). Each step sets its own seed.
%   Runtime: about 2 minutes on a laptop (Apple M2 Pro), most of it in steps 4 to 6.
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
fprintf('Module 06 (Figs. 6 and 7), MATLAB %s, started %s\n\n', version, char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm')));

results = struct();

fprintf('\n===== Step 1: export the three image sets =====\n');
results.sets = export_sampled_sets();

fprintf('\n===== Step 2: redraw the concepts of the three image sets =====\n');
results.selection = reproduce_set_selection();

fprintf('\n===== Step 3: task design read from the trial files =====\n');
results.design = check_task_design();

fprintf('\n===== Step 4: participant flow =====\n');
results.flow = participant_flow();

fprintf('\n===== Step 5: Fig. 6 (maintained vs changed image set) =====\n');
results.fig6 = maintained_vs_changed();

fprintf('\n===== Step 6: Fig. 7b to 7d (maintained vs reduced image set) =====\n');
results.fig7 = maintained_vs_reduced_variance();

fprintf('\n===== Step 7: Fig. 7a =====\n');
results.fig7a = make_figure7a_pca();

fprintf('\n===== Step 8: category make-up =====\n');
results.categories = category_shares();

fprintf('\n===== Step 9: numbers.csv =====\n');
write_module_numbers(results);

close all;
fprintf('\nModule 06 finished in %.1f minutes.\n', toc(runTimer) / 60);
diary off;

% The log names files by their place in the repository, not by this computer's folders
releaseFolder = fileparts(paths.module);
logText = fileread(logFile);
logText = strrep(logText, [releaseFolder filesep], '');
logText = strrep(logText, releaseFolder, '.');
logHandle = fopen(logFile, 'w');
fprintf(logHandle, '%s', logText);
fclose(logHandle);
