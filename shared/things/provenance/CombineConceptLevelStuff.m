clear all;
spose_sim = load(fullfile('shared', 'things', 'spose_embedding_49d_sorted.txt'));
im = load('<images1854.mat>');
labels = load(fullfile('shared', 'things', 'labels_short49.mat'));
concToCat = readtable(fullfile('external', 'kramer', 'Concept_to_category_linking.csv'));

concInfo.images = im.images1854';

concInfo.features = spose_sim;
concInfo.featureLabels = labels.labels_short;
concInfo.thingsSimRSM = corrcoef(concInfo.features');

concInfo.concepts = {concInfo.images.name};
concInfo.concepts = concInfo.concepts';
concInfo.conceptNumber = (1:length(concInfo.concepts))';

concInfo.categoryNumber = concToCat.category;
concInfo.category = concToCat.category_label;

mem_data = readtable('external/kramer/THINGS_Memorability_Scores.csv');
mem_concepts = cellfun(@(x) strsplit(x,'/'), mem_data.file_path, 'UniformOutput', false);
mem_concepts = cellfun(@(x) x{2}, mem_concepts, 'UniformOutput', false);
memorability_scores = zeros(length(concInfo.images),1);
memorability_items = zeros(length(concInfo.images),1);
for i = 1:length(concInfo.images)
    concept_matches = strcmp(mem_concepts, concInfo.images(i).name);
    if any(concept_matches)
        memorability_scores(i) = mean(mem_data.cr(concept_matches));
    end
    vv = mem_data.cr(concept_matches);
    memorability_items(i) = vv(1);
end
concInfo.memorabilityCat = memorability_scores;
concInfo.memorabilityItem = memorability_items;
concInfo.memorabilityWilma = concToCat.CR;

% Calculate weighted average similarity for each concept
load('<ImageInfo.mat>'); % imgInfo
stdForWeightedSim = 2; % SD of the normal kernel applied to the similarities
nConcepts = length(concInfo.concepts);
concInfo.weightedSimByConc = zeros(nConcepts, 1);
[uniqueConcepts, ~, conceptIdx] = unique(imgInfo.concepts);
conceptCounts = accumarray(conceptIdx, 1);
concInfo.conceptCounts = conceptCounts;
weights = conceptCounts / sum(conceptCounts);
for i = 1:nConcepts
    otherConcepts = true(nConcepts,1);
    otherConcepts(i) = false;
    weightedSim = normpdf(concInfo.thingsSimRSM(i,otherConcepts),1,stdForWeightedSim) * weights(otherConcepts);
    concInfo.weightedSimByConc(i) = weightedSim;
end

% Initialize array to store concept-level averages for ResMem
nConcepts = length(concInfo.concepts);
concInfo.resMemPredictions = nan(nConcepts, 1);
for i = 1:nConcepts
    curConcept = concInfo.concepts{i};
    conceptMask = strcmp(imgInfo.concepts, curConcept);
    if any(conceptMask)
        concInfo.resMemPredictions(i,1) = mean(imgInfo.resMemPredictions(conceptMask));
    end
end

% images
save ConceptInfo.mat concInfo;
