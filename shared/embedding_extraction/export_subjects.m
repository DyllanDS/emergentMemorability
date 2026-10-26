function export_subjects(dataFolder, outputFile)
% Per-subject response vectors, for a split-half noise ceiling computed the way
% this literature computes it: over participants, not over trials.
%
%   export_subjects(dataFolder, outputFile), e.g. from the repository root:
%       export_subjects('external/isola/Data', 'external/isola/subject_matrices.mat')
%   Pass both arguments. The file is written in MATLAB v7.3 format (readable with h5py); module
%   A6a reads the same matrices from A6a_faces_and_scenes_figS15/data/scenes/subject_matrices.mat.
HERE = fileparts(mfilename('fullpath'));
ROOT = fileparts(HERE);
if nargin < 1
  dataFolder = fullfile(ROOT, 'data', 'Data');
  outputFile = fullfile(ROOT, 'data', 'subject_matrices.mat');
end
S = load(fullfile(dataFolder,'Experiment data','subj_data.mat'));
f = fieldnames(S); d = S.(f{1});
nS = numel(d);
first = d{find(cellfun(@isstruct,d),1)};
nT = numel(first.target_detection_vector);
nF = numel(first.filler_detection_vector);
fprintf('%d subjects, %d targets, %d fillers\n', nS, nT, nF);
% 108 of the 951 cells are empty in the package; only the structs are usable
ok = find(cellfun(@isstruct, d));
nU = numel(ok);
fprintf('usable subject records: %d of %d\n', nU, nS);
tdet = nan(nU,nT); tfa = nan(nU,nT); fdet = nan(nU,nF); ffa = nan(nU,nF);
for k = 1:nU
  e = d{ok(k)};
  tdet(k,:) = e.target_detection_vector(:)';
  tfa(k,:)  = e.target_false_alarm_vector(:)';
  fdet(k,:) = e.filler_detection_vector(:)';
  ffa(k,:)  = e.filler_false_alarm_vector(:)';
end
u = unique(tdet(~isnan(tdet)));
fprintf('target_detection values: %s\n', mat2str(u(1:min(6,numel(u)))'));
fprintf('non-NaN per subject (targets): %.1f\n', mean(sum(~isnan(tdet),2)));
save(outputFile, ...
     'tdet','tfa','fdet','ffa','-v7.3');
fprintf('wrote subject_matrices.mat\n');
end
