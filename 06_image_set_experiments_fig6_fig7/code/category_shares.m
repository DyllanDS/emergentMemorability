function categories = category_shares()
%CATEGORY_SHARES  The higher-level categories in the reduced image set and in THINGS.
%
%   What it does
%     1. For the reduced image set: how many of its 1,600 slots (and of its concepts) fall into
%        each higher-level THINGS category, as a percentage. The category of a concept is its
%        label in shared/things/concept_table.csv (column category_label). Concepts without a
%        higher-level category carry the label "NA"; "NA" is not counted as a category when the
%        largest categories are named.
%     2. For the whole THINGS database: the number of concepts and of images in each category
%        (the General Discussion compares categories such as tools and faces). The higher-level
%        categories contain no face category; the concept "face" is counted on its own.
%
%   Inputs:  data/sampled_image_sets.mat; shared/things/concept_table.csv, things_images.csv
%   Outputs: outputs/category_shares.csv (reduced set: category, slots, percent of slots,
%            concepts, percent of concepts; largest first)
%            outputs/things_category_sizes.csv (whole THINGS: category, concepts, images)
%   Returns a struct with the largest shares and the tool and face counts.
%   No random numbers are used.

paths = release_paths(mfilename('fullpath'));
sampledSets = load(fullfile(paths.data, 'sampled_image_sets.mat'));
reducedSet = sampledSets.equatedDataset;

conceptTable = readConceptTable(fullfile(paths.releaseRoot, 'shared', 'things', 'concept_table.csv'));
imageTable = readtable(fullfile(paths.releaseRoot, 'shared', 'things', 'things_images.csv'), ...
    'TextType', 'string', 'Delimiter', ',');
labelOfConcept = conceptTable.category_label;
categoryNames = unique(labelOfConcept);

%% 1. The reduced set: slots and concepts per category
slotsPerConcept = reducedSet.numImagesForConcept(:);
totalSlots = sum(slotsPerConcept);
totalConceptsInSet = sum(slotsPerConcept > 0);
slotsPerCategory = zeros(numel(categoryNames), 1);
conceptsPerCategory = zeros(numel(categoryNames), 1);
for categoryIndex = 1:numel(categoryNames)
    inCategory = labelOfConcept == categoryNames(categoryIndex);
    slotsPerCategory(categoryIndex) = sum(slotsPerConcept(inCategory));
    conceptsPerCategory(categoryIndex) = sum(slotsPerConcept(inCategory) > 0);
end
percentOfSlots = 100 * slotsPerCategory / totalSlots;
percentOfConcepts = 100 * conceptsPerCategory / totalConceptsInSet;

% one row per category, largest first
[~, order] = sort(slotsPerCategory, 'descend');
shareRows = cell(numel(order), 5);
for k = 1:numel(order)
    categoryIndex = order(k);
    shareRows(k, :) = {categoryNames(categoryIndex), slotsPerCategory(categoryIndex), ...
        percentOfSlots(categoryIndex), conceptsPerCategory(categoryIndex), percentOfConcepts(categoryIndex)};
end
shareTable = cell2table(shareRows, 'VariableNames', {'category', 'slots', 'percent_of_slots', 'concepts', ...
    'percent_of_concepts'});
writetable(shareTable, fullfile(paths.outputs, 'category_shares.csv'));

% the two largest named categories (the label "NA" is not a category)
isNamed = categoryNames ~= "NA";
namedNames = categoryNames(isNamed);
namedPercentOfSlots = percentOfSlots(isNamed);
namedPercentOfConcepts = percentOfConcepts(isNamed);
[sortedPercent, namedOrder] = sort(namedPercentOfSlots, 'descend');
categories = struct();
categories.totalSlots = totalSlots;
categories.conceptsInSet = totalConceptsInSet;
categories.largestCategory = namedNames(namedOrder(1));
categories.largestPercentOfSlots = sortedPercent(1);
categories.secondCategory = namedNames(namedOrder(2));
categories.secondPercentOfSlots = sortedPercent(2);
categories.noCategoryPercentOfSlots = percentOfSlots(categoryNames == "NA");
categories.largestPercentOfConcepts = max(namedPercentOfConcepts);
categories.numberOfNamedCategories = sum(isNamed);
fprintf('reduced set: largest named categories %s %.1f%% and %s %.1f%% of the %d slots (no category: %.1f%%); largest share of its %d concepts %.1f%%\n', ...
    categories.largestCategory, categories.largestPercentOfSlots, categories.secondCategory, ...
    categories.secondPercentOfSlots, totalSlots, categories.noCategoryPercentOfSlots, totalConceptsInSet, ...
    categories.largestPercentOfConcepts);

%% 2. The whole THINGS database: concepts and images per category
[isKnown, conceptRowOfImage] = ismember(imageTable.concept, conceptTable.concept);
if ~all(isKnown)
    error('category_shares:Images', 'an image belongs to a concept missing from the concept table');
end
labelOfImage = labelOfConcept(conceptRowOfImage);
sizeRows = cell(numel(categoryNames), 3);
for categoryIndex = 1:numel(categoryNames)
    sizeRows(categoryIndex, :) = {categoryNames(categoryIndex), sum(labelOfConcept == categoryNames(categoryIndex)), ...
        sum(labelOfImage == categoryNames(categoryIndex))};
end
sizeTable = cell2table(sizeRows, 'VariableNames', {'category', 'concepts_in_things', 'images_in_things'});
writetable(sizeTable, fullfile(paths.outputs, 'things_category_sizes.csv'));

categories.toolConcepts = sum(labelOfConcept == "tool");
categories.toolImages = sum(labelOfImage == "tool");
categories.hasFaceCategory = any(contains(categoryNames, "face"));
categories.faceConceptImages = sum(imageTable.concept == "face");
categories.faceConceptCategory = labelOfConcept(conceptTable.concept == "face");
categories.numberOfConcepts = height(conceptTable);
categories.numberOfImages = height(imageTable);
fprintf('THINGS (%d concepts, %d images): tool %d concepts with %d images; %d named categories, a face category among them: %d; the concept "face" (category %s) has %d images\n', ...
    categories.numberOfConcepts, categories.numberOfImages, categories.toolConcepts, categories.toolImages, ...
    categories.numberOfNamedCategories, categories.hasFaceCategory, categories.faceConceptCategory, ...
    categories.faceConceptImages);
end


function conceptTable = readConceptTable(fileName)
% concept and category_label as text (the label "NA" stays a label, it is not a missing value)
options = detectImportOptions(fileName, 'Delimiter', ',');
options = setvartype(options, {'concept', 'category_label'}, 'string');
options = setvaropts(options, {'concept', 'category_label'}, 'TreatAsMissing', {});
conceptTable = readtable(fileName, options);
end
