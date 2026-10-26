function save_panel(figureHandle, panelName, paths, panelData)
%SAVE_PANEL  Save one figure panel as a 300 dpi PNG and its plotted numbers as a CSV.
%
%   save_panel(figureHandle, 'fig3e_memorability_vs_cr', paths, panelData)
%
%   figureHandle  the figure to save
%   panelName     file name without extension (e.g. 'fig6c_things_vs_maintained_profile')
%   paths         the struct from release_paths(mfilename('fullpath')) (uses paths.figures and
%                 paths.figureData), or a module folder (then <module>/outputs/figures and
%                 <module>/outputs/figure_data are used)
%   panelData     a table, or a struct whose fields are equal-length vectors (one column each),
%                 holding exactly the numbers the panel plots (no literal numbers in the figure)
%
%   Writes <outputs>/figures/<panelName>.png (300 dpi) and <outputs>/figure_data/<panelName>.csv.

if ischar(paths) || isstring(paths)
    moduleFolder = char(paths);
    paths = struct('figures', fullfile(moduleFolder, 'outputs', 'figures'), ...
        'figureData', fullfile(moduleFolder, 'outputs', 'figure_data'));
end
if ~exist(paths.figures, 'dir')
    mkdir(paths.figures);
end
if ~exist(paths.figureData, 'dir')
    mkdir(paths.figureData);
end

imageFile = fullfile(paths.figures, [panelName '.png']);
exportgraphics(figureHandle, imageFile, 'Resolution', 300);

if isstruct(panelData)
    panelData = struct2table(structOfColumns(panelData));
end
dataFile = fullfile(paths.figureData, [panelName '.csv']);
writetable(panelData, dataFile);
fprintf('saved %s and %s\n', imageFile, dataFile);
end

function columns = structOfColumns(data)
% every field as a column vector
columns = struct();
names = fieldnames(data);
for nameIndex = 1:numel(names)
    value = data.(names{nameIndex});
    if isrow(value)
        value = value';
    end
    columns.(names{nameIndex}) = value;
end
end
