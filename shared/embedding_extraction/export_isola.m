function export_isola(dataFolder, outputFolder)
% Unpack the Isola et al. (2011) scene memorability package into images on disk
% plus one score table, for extract_embeddings.py and module A6a.
%
%   export_isola(dataFolder, outputFolder), e.g. from the repository root:
%       export_isola('external/isola/Data', 'external/isola')
%   writes <outputFolder>/images/ (10,442 jpg) and <outputFolder>/isola_scores.csv.
%   Pass both folders.
%
% The package stores the images as MATLAB uint8 arrays and the behavioral results
% as cell arrays of structs. Targets and fillers are both scored: fillers are the
% context a target sits in, and they carry their own hit and false alarm rates
% from the vigilance repeats, so they are outcomes as well as context.
%
% The package contains 2,400 targets; Isola et al. (2011) analyzed the first
% 2,222 (the other 178 are textural images). All are exported, with the flag
% in_cvpr_2222.

HERE = fileparts(mfilename('fullpath'));
ROOT = fileparts(HERE);
if nargin < 1
  DATA = fullfile(ROOT, 'data', 'Data');
  OUT = fullfile(ROOT, 'data');
else
  DATA = dataFolder;
  OUT = outputFolder;
end
IMG_OUT = fullfile(OUT, 'images');
if ~exist(IMG_OUT, 'dir'), mkdir(IMG_OUT); end

rows = {};

for which = {'target', 'filler'}
  kind = which{1};
  fprintf('\n=== %s ===\n', kind);
  S = load(fullfile(DATA, 'Image data', sprintf('%s_images.mat', kind)));
  fn = fieldnames(S);
  img = S.(fn{1});
  fprintf('images: %s %s\n', mat2str(size(img)), class(img));

  D = load(fullfile(DATA, 'Experiment data', sprintf('sorted_%s_data.mat', kind)));
  fn = fieldnames(D);
  dat = D.(fn{1});
  n = numel(dat);
  fprintf('score entries: %d\n', n);
  if size(img, 4) ~= n
    warning('image count %d does not match score count %d', size(img,4), n);
  end

  for i = 1:n
    e = dat{i};
    % file names in the package follow the SUN database; make them safe on disk
    % and unique across the two sets
    raw = e.filepath;   % the package calls it filepath, not filename
    safe = regexprep(raw, '[\\/:*?"<>|]', '_');
    outname = sprintf('%s__%s', kind, safe);
    if numel(outname) < 4 || ~strcmpi(outname(end-3:end), '.jpg')
      outname = [outname '.jpg']; %#ok<AGROW>
    end
    imwrite(img(:, :, :, i), fullfile(IMG_OUT, outname), 'jpg', 'Quality', 98);

    hits = double(e.hits); miss = double(e.misses);
    fa = double(e.false_alarms); cr = double(e.correct_rejections);
    hr = hits / max(hits + miss, 1);
    far = fa / max(fa + cr, 1);
    rows(end+1, :) = {outname, raw, kind, i, hits, miss, fa, cr, hr, far, ...
                      hr - far, double(strcmp(kind,'target') && i <= 2222)}; %#ok<AGROW>
    if mod(i, 1000) == 0, fprintf('  %d / %d\n', i, n); end
  end
end

T = cell2table(rows, 'VariableNames', {'filename','sun_filename','kind', ...
    'index_in_set','hits','misses','false_alarms','correct_rejections', ...
    'hit_rate','false_alarm_rate','corrected_recognition','in_cvpr_2222'});
writetable(T, fullfile(OUT, 'isola_scores.csv'));

fprintf('\nwrote %s\n', fullfile(OUT, 'isola_scores.csv'));
fprintf('images written: %d\n', height(T));
fprintf('  targets: %d (of which in the CVPR 2,222: %d)\n', ...
        sum(strcmp(T.kind,'target')), sum(T.in_cvpr_2222));
fprintf('  fillers: %d\n', sum(strcmp(T.kind,'filler')));
fprintf('mean HR  targets %.3f, fillers %.3f\n', ...
        mean(T.hit_rate(strcmp(T.kind,'target'))), ...
        mean(T.hit_rate(strcmp(T.kind,'filler'))));
fprintf('mean FAR targets %.3f, fillers %.3f\n', ...
        mean(T.false_alarm_rate(strcmp(T.kind,'target'))), ...
        mean(T.false_alarm_rate(strcmp(T.kind,'filler'))));
end
