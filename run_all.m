function run_all(skipLong)
%RUN_ALL  The MATLAB part of the release, for readers who use MATLAB only.
%
%   What it produces, and what it does not
%     It reruns every MATLAB analysis of the paper: the main-text Figs. 1, 2, 3, 5, 6 and 7 with the rows of
%     outputs/numbers.csv that MATLAB computes (modules 01, 02, 03, 05, 06; the rows with bootstrap intervals
%     and p values come from the modules' Python steps), and the MATLAB simulations and analyses of the
%     SI Appendix (A1 criterion_dprime_da.m, A2a rem_parameter_sweep.m, A2b bylinskii_robustness.m, A2d
%     list_designed.m and list_binned.m, A2e continuous_rem_spose.m), which write their result tables to
%     the modules' outputs/ folders.
%     It produces NO SI Appendix figures, does not update the SI Appendix modules' numbers.csv, and
%     computes NONE of the statistics layer (outputs/tests.csv, statistics/all_tests.csv): those are
%     Python. Modules 04, 04b, 06b and the other SI Appendix modules are Python only. For the whole release run
%         python run_all.py
%     from the repository root (README.md, "Quick start").
%
%   Run from the repository root:
%       matlab -batch "run_all"          everything, about 35 minutes on an Apple M2 Pro
%       matlab -batch "run_all(true)"    without module 02's simulation (Fig. 2 then comes from the saved
%                                        run in data/paper_run/), about 15 minutes
%
%   Each step starts with restoredefaultpath (for this MATLAB session only), so that no folder a user
%   added to the MATLAB path can replace one of MATLAB's own functions, then changes to the module's
%   code folder and makes the call its README gives. MATLAB R2023b with the Statistics and Machine
%   Learning Toolbox; no parallel pool. A failed step is reported and the next one runs; at the end
%   run_all stops with an error if any step failed.

if nargin < 1 || isempty(skipLong)
    skipLong = false;
end
releaseRoot = fileparts(mfilename('fullpath'));

% The steps: module code folder (from the repository root) and the call.
steps = {};
steps(end + 1, :) = {'01_rem_simulation_fig1/code', 'run_module'};
if skipLong
    steps(end + 1, :) = {'02_bylinskii_simulation_fig2/code', 'run_module(''skip_long'')'};
else
    steps(end + 1, :) = {'02_bylinskii_simulation_fig2/code', 'run_module(''full'')'};
end
steps(end + 1, :) = {'03_inversion_experiment_fig3/code', 'run_module'};
steps(end + 1, :) = {'05_things_reanalysis_fig5/code', 'run_module'};
steps(end + 1, :) = {'06_image_set_experiments_fig6_fig7/code', 'run_module'};
steps(end + 1, :) = {'A1_measuring_memory_figS1/code', 'criterion_dprime_da'};
steps(end + 1, :) = {'A2a_rem_parameter_sweep_figS2/code', 'rem_parameter_sweep(true)'};
steps(end + 1, :) = {'A2b_category_structure_criteria_figS3/code', 'bylinskii_robustness(true)'};
steps(end + 1, :) = {'A2d_list_context_figS5/code', 'list_designed'};
steps(end + 1, :) = {'A2d_list_context_figS5/code', 'list_binned'};
steps(end + 1, :) = {'A2e_measured_representations_figS6/code', 'continuous_rem_spose(1)'};

numberOfSteps = size(steps, 1);
status = cell(numberOfSteps, 1);
minutes = zeros(numberOfSteps, 1);
fprintf('run_all: %d MATLAB steps (skipLong = %d)\n', numberOfSteps, skipLong);

for stepIndex = 1:numberOfSteps
    codeFolder = steps{stepIndex, 1};
    call = steps{stepIndex, 2};
    fprintf('\n==== %d of %d: cd(''%s''); %s\n', stepIndex, numberOfSteps, codeFolder, call);
    cd(releaseRoot);
    restoredefaultpath;
    stepTimer = tic;
    try
        runOneStep(fullfile(releaseRoot, codeFolder), call);
        status{stepIndex} = 'ok';
    catch stepError
        status{stepIndex} = ['FAILED: ' stepError.message];
        fprintf('%s\n', getReport(stepError, 'basic'));
    end
    minutes(stepIndex) = toc(stepTimer) / 60;
    close all;
    cd(releaseRoot);
end

fprintf('\n==== Summary\n');
numberFailed = 0;
for stepIndex = 1:numberOfSteps
    fprintf('%-45s %-28s %6.1f min  %s\n', steps{stepIndex, 1}, steps{stepIndex, 2}, minutes(stepIndex), ...
        status{stepIndex});
    if ~strcmp(status{stepIndex}, 'ok')
        numberFailed = numberFailed + 1;
    end
end
if numberFailed > 0
    error('run_all: %d of %d steps failed (see above)', numberFailed, numberOfSteps);
end
fprintf('run_all: every step ran. Next, for the statistics layer and the appendix figures: python run_all.py\n');
end


function runOneStep(codeFolder, call)
% Runs one call in a module's code folder, in a workspace of its own (some module scripts start with
% "clear", which then clears only this function's variables).
cd(codeFolder);
eval(call);
end
