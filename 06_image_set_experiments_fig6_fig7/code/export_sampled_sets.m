function sets = export_sampled_sets()
%EXPORT_SAMPLED_SETS  The three 1,600-slot image sets of the image-set experiments as CSV tables.
%
%   What it does
%     Reads the three image sets the experiments sampled their study lists and foils from and
%     writes each as a plain table, one row per slot of the set.
%
%   Input (data/sampled_image_sets.mat)
%     representativeDataset     the maintained image set ("maintained similarity structure")
%     flipMemorabilityDataset   the changed image set ("changed similarity structure")
%     equatedDataset            the reduced image set ("reduced variability")
%     each with conceptIndex (THINGS concept number of each slot), conceptName, imageIndex
%     (public THINGS image number), imageFilename and numImagesForConcept (1,854 x 1: how many
%     slots each concept has in the set)
%
%   Output (outputs/)
%     maintained_set_images.csv, changed_set_images.csv, reduced_set_images.csv with the columns
%     slot, concept_number (THINGS order, as in shared/things/concept_table.csv), concept,
%     things_image_index (the public THINGS image number), image_filename
%
%   Returns a struct with, for each set, the number of slots, of different images, of concepts
%   present and of concepts with more than one slot.

paths = release_paths(mfilename('fullpath'));
sampledSets = load(fullfile(paths.data, 'sampled_image_sets.mat'));

setNames = {'maintained', 'changed', 'reduced'};
setStructs = {sampledSets.representativeDataset, sampledSets.flipMemorabilityDataset, ...
    sampledSets.equatedDataset};

sets = struct();
for setIndex = 1:numel(setNames)
    setName = setNames{setIndex};
    imageSet = setStructs{setIndex};

    numberOfSlots = numel(imageSet.conceptIndex);
    slotTable = table();
    slotTable.slot = (1:numberOfSlots)';
    slotTable.concept_number = imageSet.conceptIndex(:);
    slotTable.concept = string(imageSet.conceptName(:));
    slotTable.things_image_index = imageSet.imageIndex(:);
    slotTable.image_filename = string(imageSet.imageFilename(:));

    % the slot counts per concept must add up to the slots listed
    if sum(imageSet.numImagesForConcept) ~= numberOfSlots
        error('export_sampled_sets:Counts', 'the %s set: concept counts do not add up to its slots', setName);
    end

    outputFile = fullfile(paths.outputs, sprintf('%s_set_images.csv', setName));
    writetable(slotTable, outputFile);

    sets.(setName).numberOfSlots = numberOfSlots;
    sets.(setName).numberOfDistinctImages = numel(unique(imageSet.imageIndex));
    sets.(setName).numberOfConcepts = sum(imageSet.numImagesForConcept > 0);
    sets.(setName).conceptsWithSeveralSlots = sum(imageSet.numImagesForConcept > 1);
    sets.(setName).largestCountPerConcept = max(imageSet.numImagesForConcept);
    fprintf('%-10s set: %d slots, %d different images, %d concepts present (at most %d slots per concept) -> %s\n', ...
        setName, numberOfSlots, sets.(setName).numberOfDistinctImages, sets.(setName).numberOfConcepts, ...
        sets.(setName).largestCountPerConcept, outputFile);
end
end
