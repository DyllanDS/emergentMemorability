function [pcaFeatures, explained] = pcaCoeffsExplained(X, nPC)
%PCACOEFFSEXPLAINED First k PC directions and percent variance (SVD, no toolbox pca).
if nargin < 2 || isempty(nPC)
    nPC = 3;
end
mu = mean(X, 1);
Xc = bsxfun(@minus, X, mu);
[U, S, V] = svd(Xc, 'econ');
s = diag(S);
n = size(X, 1);
denom = max(n - 1, 1);
latent = (s .^ 2) ./ denom;
latent = latent(:);
sLat = sum(latent);
if sLat <= 0
    explainedAll = zeros(size(latent));
else
    explainedAll = 100 * latent ./ sLat;
end
coeff = V;
nComp = min(nPC, size(coeff, 2));
pcaFeatures = coeff(:, 1:nComp);
explained = explainedAll(1:nComp);
end
