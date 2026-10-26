function v = readPca3dFeatureSpaceView(codeRoot, figureId) %#ok<INUSL>
%READPCA3DFEATURESPACEVIEW Load 3D PCA camera settings from pca3d_feature_space_view.csv
%   codeRoot  - not used (the camera table is read from shared/matlab/)
%   figureId  - 'original' | 'representative' | 'flipped' | 'equated' | 'default' (case-insensitive)
%
%   Returns struct v with fields .az .el .camva .camroll (degrees).
%
%   The camera table pca3d_feature_space_view.csv sits next to this function in
%   shared/matlab/; a missing table is an error.

if nargin < 2 || isempty(figureId)
    figureId = 'default';
end
figureId = lower(strtrim(char(string(figureId))));

v = struct('az', -58.768, 'el', 40.947, 'camva', 10.92, 'camroll', 0);

csvPath = fullfile(fileparts(mfilename('fullpath')), 'pca3d_feature_space_view.csv');
if exist(csvPath, 'file') ~= 2
    error('readPca3dFeatureSpaceView:MissingCsv', 'Camera table not found (it ships in shared/matlab):\n%s', csvPath);
end

T = readtable(csvPath);
if isempty(T)
    return;
end
ids0 = string(T.figure_id);
keep = ~ismissing(ids0) & strlength(strtrim(ids0)) > 0;
T = T(keep, :);
if isempty(T)
    return;
end

if ~ismember('figure_id', T.Properties.VariableNames)
    error('readPca3dFeatureSpaceView:BadCsv', ...
        'Column figure_id missing in %s', csvPath);
end

ids = lower(strtrim(string(T.figure_id)));
idx = find(ids == string(figureId), 1);
if isempty(idx)
    idx = find(ids == "default", 1);
end
if isempty(idx)
    idx = 1;
end

v.az = toDouble(T, idx, {'azimuth_deg', 'azimuth', 'az'});
v.el = toDouble(T, idx, {'elevation_deg', 'elevation', 'el'});
v.camva = toDouble(T, idx, {'camva_deg', 'camva'});
try
    v.camroll = toDouble(T, idx, {'camroll_deg', 'camroll'});
catch
    v.camroll = 0;
end
if ~isfinite(v.camroll)
    v.camroll = 0;
end
end

function x = toDouble(T, idx, names)
for k = 1:numel(names)
    nm = names{k};
    if ismember(nm, T.Properties.VariableNames)
        x = double(T{idx, nm});
        if ~isscalar(x)
            x = x(1);
        end
        return;
    end
end
error('readPca3dFeatureSpaceView:MissingCol', 'None of %s found in CSV', strjoin(names, ', '));
end
