function paths = release_paths(scriptFile)
%RELEASE_PATHS  The folders of the release, found from a script's own location (MATLAB twin of
%   shared/python/release_paths.py). No script of the release contains an absolute path.
%
%   A module script first puts shared/matlab on the path with one relative line, then asks for
%   its folders:
%       addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared', 'matlab'));
%       paths = release_paths(mfilename('fullpath'));
%
%   Input:  scriptFile, the full path of the calling script (mfilename('fullpath')), which must sit
%           in <module>/code/ (or deeper inside code/)
%   Output: struct with fields
%       releaseRoot, sharedMatlab, sharedThings, sharedLedger    the release and shared folders
%       module, data, provided (data/provided), paperRun (data/paper_run), code
%       outputs, figures, figureData, logs                       created if missing

folder = fileparts(scriptFile);
moduleFolder = '';
while true
    [parentFolder, folderName] = fileparts(folder);
    if strcmp(folderName, 'code')
        moduleFolder = parentFolder;
        break;
    end
    if strcmp(parentFolder, folder) || isempty(folderName)
        error('release_paths:NotInModule', '%s is not inside a module code/ folder', scriptFile);
    end
    folder = parentFolder;
end

paths = struct();
paths.sharedMatlab = fileparts(mfilename('fullpath'));
paths.releaseRoot = fileparts(fileparts(paths.sharedMatlab));
paths.sharedThings = fullfile(paths.releaseRoot, 'shared', 'things');
paths.sharedLedger = fullfile(paths.releaseRoot, 'shared', 'ledger');
paths.module = moduleFolder;
paths.data = fullfile(moduleFolder, 'data');
paths.provided = fullfile(moduleFolder, 'data', 'provided');
paths.paperRun = fullfile(moduleFolder, 'data', 'paper_run');
paths.code = fullfile(moduleFolder, 'code');
paths.outputs = fullfile(moduleFolder, 'outputs');
paths.figures = fullfile(moduleFolder, 'outputs', 'figures');
paths.figureData = fullfile(moduleFolder, 'outputs', 'figure_data');
paths.logs = fullfile(moduleFolder, 'outputs', 'logs');

outputFolders = {paths.outputs, paths.figures, paths.figureData, paths.logs};
for folderIndex = 1:numel(outputFolders)
    if ~exist(outputFolders{folderIndex}, 'dir')
        mkdir(outputFolders{folderIndex});
    end
end
end
