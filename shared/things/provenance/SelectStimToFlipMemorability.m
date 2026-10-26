%clear all;

% Does ResMem predict memorability?
figure(1); clf;
scatter(imgInfo.resMemPredictions, imgInfo.memorabilityScores);
AddCorrToScatter();

%% Make a potential stim set to flip memorability
%---

% Find images in each quintile bin for both memorability and ResMem
nBins = 5; % 5 bins
nImages = length(imgInfo.resMemPredictions);
predBins = prctile(imgInfo.resMemPredictions, linspace(0,100,nBins+1));
memBins = prctile(imgInfo.memorabilityScores, linspace(0,100,nBins+1));
binCandidates = cell(nBins,1);
for i = 1:nBins
    inPredBin = imgInfo.resMemPredictions >= predBins(i) & imgInfo.resMemPredictions < predBins(i+1);
    inMemBin = imgInfo.memorabilityScores >= memBins(i) & imgInfo.memorabilityScores < memBins(i+1);
    binCandidates{i} = find(inPredBin & inMemBin);
end

% Initialize array for selected images
selectedImages = zeros(nBins,1);
% Select first image randomly from first bin
selectedImages(1) = binCandidates{1}(randi(length(binCandidates{1})));
% For remaining bins, find image that's most dissimilar to already selected ones
for i = 2:nBins
    candidates = binCandidates{i};
    maxMinDissimilarity = -inf;
    bestCandidate = [];
    for c = 1:length(candidates)
        candidate = candidates(c);
        % Calculate minimum similarity to already selected images
        conceptSims = imgInfo.conceptRSM(candidate, selectedImages(1:i-1));
        deepNetDissimilarity = imgInfo.deepNetRDM(candidate, selectedImages(1:i-1));
        
        % Combine similarities (flip deepNetSims since higher means more dissimilar)
        combinedMinSim = max(conceptSims) + (1 - min(deepNetDissimilarity));
        
        % Keep track of most dissimilar candidate
        if combinedMinSim < maxMinDissimilarity || isempty(bestCandidate)
            maxMinDissimilarity = combinedMinSim;
            bestCandidate = candidate;
        end
    end
    selectedImages(i) = bestCandidate;
end

% Display selected images and their properties
figure(2); clf;
for i = 1:nBins
    idx = selectedImages(i);
    subplot(1,5,i);
    img_path = fullfile('external', 'things_images', imgInfo.concepts{idx}, imgInfo.fileNames{idx});
    imshow(img_path);
    title(sprintf('Bin %d\nPred: %.2f\nMem: %.2f', i, ...
        imgInfo.resMemPredictions(idx), ...
        imgInfo.memorabilityScores(idx)), ...
        'Interpreter', 'none');
end

% Print dissimilarity statistics
fprintf('\nDissimilarity Statistics for Selected Images:\n');
fprintf('Average Concept Similarity: %.3f\n', mean(imgInfo.conceptRSM(selectedImages,selectedImages),'all'));
fprintf('Average DeepNet Dissimilarity: %.3f\n', mean(imgInfo.deepNetRDM(selectedImages,selectedImages),'all'));

%% Make a potential stim set to flip memorability for these 5 images
reqSimilar = [0 1 2 4 8]; % Number of similar images to use for each selected image

% Initialize array to store the 40 images
displayImages = zeros(40,1);
displayCount = 0;
simScores = cell(5,1);
for i = 1:5
  % Combine the concept and deep-network similarities, weighting the deep-network one twice as much because it differs between images of the same concept
    simScores{i} = imgInfo.conceptRSM(selectedImages(i),:)./2 + (1 - imgInfo.deepNetRDM(selectedImages(i),:));
end

% Find similar images for selectedImages(2-5) according to requirements
whichSimTo = zeros(40,1);
for i = 2:5
    % Sort by similarity to current selected image
    [~, sortedIdx] = sort(simScores{i}, 'descend');
    % Skip the selected image itself
    sortedIdx = sortedIdx(2:end);
    % Take required number of similar images
    displayImages(displayCount+1:displayCount+reqSimilar(i)) = sortedIdx(1:reqSimilar(i));
    whichSimTo(displayCount+1:displayCount+reqSimilar(i)) = i;
    displayCount = displayCount + reqSimilar(i);
end

% Fill remaining slots with maximally dissimilar images
remainingSlots = 40 - displayCount;

% Calculate average similarity to all selected images
totalSim = zeros(size(imgInfo.conceptRSM,1), 1);
for i = 1:5
    totalSim = totalSim + simScores{i};
end

% Sort by dissimilarity (ascending similarity)
[sortedSims, sortedIdx] = sort(totalSim, 'ascend');

% Remove already used images
usedImages = [selectedImages; displayImages(1:displayCount)];
validMask = ~ismember(sortedIdx, usedImages);
sortedIdx = sortedIdx(validMask);
sortedSims = sortedSims(validMask);

% For each remaining slot, pick the most dissimilar image that's not from 
% a concept we've already used
usedConcepts = imgInfo.concepts(usedImages);
remainingIdxs = zeros(remainingSlots, 1);
for i = 1:remainingSlots
    % Find first image from a new concept
    for j = 1:length(sortedIdx)
        candidateIdx = sortedIdx(j);
        if ~ismember(imgInfo.concepts{candidateIdx}, usedConcepts)
            remainingIdxs(i) = candidateIdx;
            usedConcepts = [usedConcepts; imgInfo.concepts{candidateIdx}];
            sortedIdx(j) = []; % Remove used index
            break;
        end
    end
end

displayImages(displayCount+1:end) = remainingIdxs;
whichSimTo(displayCount+1:end) = 0;

% Display all 40 images in an 8x5 grid
figure(3); clf;
for i = 1:40
    idx = displayImages(i);
    subplot(8,5,i);
    img_path = fullfile('external', 'things_images', imgInfo.concepts{idx}, imgInfo.fileNames{idx});
    imshow(img_path);
    
    if whichSimTo(i) == 0
        title('Maximally dissimilar');
    else
        title(sprintf('Similar to %d', whichSimTo(i)));
    end
end
sgtitle('40 Images with Controlled Similarity Distribution');

% Save the images


