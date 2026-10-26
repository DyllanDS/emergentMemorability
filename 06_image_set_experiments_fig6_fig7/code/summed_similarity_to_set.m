function summedSimilarity = summed_similarity_to_set(similarityKernel, slotsPerConcept)
%SUMMED_SIMILARITY_TO_SET  Every THINGS concept's summed similarity to one image set, as plotted
%   in Figs. 6a, 6b and 7a to 7c.
%
%   summedSimilarity = summed_similarity_to_set(similarityKernel, slotsPerConcept)
%
%   similarityKernel  1,854 x 1,854: normpdf(similarity, 1, 2) for every pair of concepts
%                     (similarity = Pearson correlation of the two concepts' 49 feature values)
%   slotsPerConcept   1,854 x 1: how many of the set's 1,600 slots each concept has (0 when the
%                     concept is not in the set)
%   summedSimilarity  1,854 x 1: for every concept c, in the set or not, the sum over all OTHER
%                     concepts j of kernel(c, j) x (slots of j / all slots of the set); the concept
%                     itself is left out.
%
%   reproduce_set_selection.m chooses the reduced set with this summed similarity, and Figs. 6a,
%   6b and 7a to 7c plot it. The analysis numbers of Figs. 6 and 7 (the profiles, and the mean and
%   SD printed with Fig. 7b, 7c) use the within-set summed similarity of the concepts in the set
%   instead (maintained_vs_changed.m Part 6, maintained_vs_reduced_variance.m Part 2).

numberOfConcepts = numel(slotsPerConcept);
slotShare = slotsPerConcept(:) / sum(slotsPerConcept);
summedSimilarity = zeros(numberOfConcepts, 1);
for conceptIndex = 1:numberOfConcepts
    otherConcepts = true(numberOfConcepts, 1);
    otherConcepts(conceptIndex) = false;
    summedSimilarity(conceptIndex) = similarityKernel(conceptIndex, otherConcepts) * slotShare(otherConcepts);
end
end
