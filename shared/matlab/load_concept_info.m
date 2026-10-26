function concInfo = load_concept_info(space)
%LOAD_CONCEPT_INFO  The THINGS concept table the main-text MATLAB analyses read (struct concInfo).
%
%   concInfo = load_concept_info('human')    SPoSE space (Hebart et al. 2020):
%                                            shared/things/concept_info_human.mat
%   concInfo = load_concept_info('monkey')   monkey IT space (Papale et al. 2025):
%                                            shared/things/concept_info_monkey.mat
%
%   Reads shared/things/concept_info_<space>.mat (written by
%   shared/things/provenance/strip_concept_info.m) and adds
%       concInfo.thingsSimRSM = corrcoef(concInfo.features')
%   the 1,854 x 1,854 concept similarity matrix (the correlation of the concepts' feature rows).
%
%   Fields: features (1,854 x 49), featureLabels, concepts (THINGS order), conceptNumber,
%   categoryNumber and category, memorabilityCat (Kramer et al. concept memorability; NaN for
%   goalpost), weightedSimByConc (summed similarity over the full THINGS set), conceptCounts (the
%   number of THINGS images of each concept, 26,107 in all), and thingsSimRSM (computed here).

if nargin < 1
    space = 'human';
end
if ~any(strcmp(space, {'human', 'monkey'}))
    error('load_concept_info:Space', 'space must be ''human'' or ''monkey''');
end
sharedMatlabFolder = fileparts(mfilename('fullpath'));
conceptFile = fullfile(sharedMatlabFolder, '..', 'things', sprintf('concept_info_%s.mat', space));
loaded = load(conceptFile, 'concInfo');
concInfo = loaded.concInfo;
concInfo.thingsSimRSM = corrcoef(concInfo.features');
end
