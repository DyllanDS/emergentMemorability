%clear all; close all;

MUAF = load('<TVSD download>/monkeyF.mat');
MUAN = load('<TVSD download>/monkeyN.mat');
conceptTable = readtable('<Filtered_Final_THINGS_Memorability_Concepts.csv>');
% Set the number of PCA components to use

%% Calculate RDMs for IT regions of both monkeys and perform PCA
% For Monkey F (IT: 513:832)
monkeyF_Feat = MUAF.train_MUA(513:832,:);
monkeyN_Feat = MUAN.train_MUA(769:1024,:);

% Combine both monkeys' electrodes into a single feature matrix
% Transpose the matrices so each row is an image and each column is an electrode
monkeyAll_Feat = [monkeyF_Feat', monkeyN_Feat']; % Images × All electrodes

% Initialize matrix for concept-averaged features
numConcepts = max(conceptTable.concept_number);
monkeyAll_Feat_byConcept = zeros(numConcepts, size(monkeyAll_Feat, 2));

% Average features for each concept
for i = 1:numConcepts
    % Find indices of images belonging to this concept
    conceptIdx = conceptTable.concept_number == i;
    % Average features across all images of this concept
    monkeyAll_Feat_byConcept(i, :) = mean(monkeyAll_Feat(conceptIdx, :), 1);
end

% Now monkeyAll_Feat_byConcept is 1854×576 (concepts × electrodes)

% Perform PCA on the concept-averaged feature matrix
[coeff, score, latent] = pca(monkeyAll_Feat_byConcept);

% Save PCA results
save('monkeyPCA.mat', 'coeff', 'score', 'latent');

% Calculate RDMs for the PCA components
monkeyF_RDM = pdist(monkeyF_Feat);
monkeyN_RDM = pdist(monkeyN_Feat);

% Plot variance explained by PCs
figure(1); 
varExplained = cumsum(latent)./sum(latent)*100;
plot(varExplained, 'b-', 'LineWidth', 2);
xlabel('Number of Principal Components');
ylabel('Cumulative Variance Explained (%)');
title('PCA Variance Explained');
grid on;

% Plot the 1,854 concepts on the first three components
figure(2);
subplot(1,3,1);
scatter(score(:,1), score(:,2), 10, 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('PC1');
ylabel('PC2');
title('PC1 vs PC2');

subplot(1,3,2);
scatter(score(:,1), score(:,3), 10, 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('PC1');
ylabel('PC3');
title('PC1 vs PC3');

subplot(1,3,3);
scatter(score(:,2), score(:,3), 10, 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('PC2');
ylabel('PC3');
title('PC2 vs PC3');

% 3D visualization
figure(3);
scatter3(score(:,1), score(:,2), score(:,3), 10, 'filled', 'MarkerFaceAlpha', 0.3);
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
title('First 3 Principal Components');
grid on;
view(45, 30); % Set viewing angle
