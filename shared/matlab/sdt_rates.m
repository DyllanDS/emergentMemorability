function sdt = sdt_rates()
%SDT_RATES  The signal-detection definitions the analyses use, in one place (MATLAB twin of
%   shared/python/sdt.py). Returns a struct of functions:
%
%       sdt = sdt_rates();
%       cr = sdt.correctedRecognition(hitRate, faRate)        hit rate minus false alarm rate
%       [d, c] = sdt.dprimeLoglinear(hits, nOld, falseAlarms, nNew)
%                       log-linear corrected rates (k + 0.5) / (n + 1) (Hautus 1995)
%       [d, c] = sdt.dprimeClamped(hitRate, faRate, nOld, nNew)
%                       rates kept inside [1/(2n), 1 - 1/(2n)] (module A1, per concept)
%       c = sdt.criterion(zHit, zFa)                            -(z(H) + z(F)) / 2
%       da = sdt.daFixedSlope(hitRate, faRate, slope)           sqrt(2 / (1 + s^2)) (z(H) - s z(F))
%       da = sdt.daFromRatingCounts(targetCounts, lureCounts)   d_a from rating counts (module A1)
%                       (six response counts from 'sure old' to 'sure new'; slope 0.8; h and f at the
%                       middle criterion; NaN unless two finite z pairs)
%
%   Old = response 1 to 3 on the six-point scale (1 = sure old). The fixed z-ROC slope of the paper
%   is 0.8 (SI Appendix, Section 1).

sdt = struct();
sdt.hitRate = @(hits, nOld) hits ./ nOld;
sdt.falseAlarmRate = @(falseAlarms, nNew) falseAlarms ./ nNew;
sdt.correctedRecognition = @(hitRate, faRate) hitRate - faRate;
sdt.loglinearRate = @(count, n) (count + 0.5) ./ (n + 1);
sdt.criterion = @(zHit, zFa) -0.5 .* (zHit + zFa);
sdt.dprimeLoglinear = @dprimeLoglinear;
sdt.dprimeClamped = @dprimeClamped;
sdt.daFixedSlope = @daFixedSlope;
sdt.daFromRatingCounts = @daFromRatingCounts;
end

function [dprime, criterion] = dprimeLoglinear(hits, nOld, falseAlarms, nNew)
zHit = norminv((hits + 0.5) ./ (nOld + 1));
zFa = norminv((falseAlarms + 0.5) ./ (nNew + 1));
dprime = zHit - zFa;
criterion = -0.5 .* (zHit + zFa);
end

function [dprime, criterion] = dprimeClamped(hitRate, faRate, nOld, nNew)
hitClamped = min(max(hitRate, 1 ./ (2 .* nOld)), 1 - 1 ./ (2 .* nOld));
faClamped = min(max(faRate, 1 ./ (2 .* nNew)), 1 - 1 ./ (2 .* nNew));
zHit = norminv(hitClamped);
zFa = norminv(faClamped);
dprime = zHit - zFa;
criterion = -0.5 .* (zHit + zFa);
end

function da = daFixedSlope(hitRate, faRate, slope)
if nargin < 3
    slope = 0.8;
end
da = sqrt(2 ./ (1 + slope .^ 2)) .* (norminv(hitRate) - slope .* norminv(faRate));
end

function da = daFromRatingCounts(targetCounts, lureCounts)
% d_a at a fixed slope of 0.8 from the six rating counts
numberOfLevels = length(targetCounts);
numberOfTargets = sum(targetCounts);
numberOfLures = sum(lureCounts);
if numberOfTargets == 0 || numberOfLures == 0
    da = NaN;
    return;
end
targetCumulative = cumsum(targetCounts) / numberOfTargets;
lureCumulative = cumsum(lureCounts) / numberOfLures;
targetZ = norminv(targetCumulative(1:numberOfLevels - 1));
lureZ = norminv(lureCumulative(1:numberOfLevels - 1));
validPairs = isfinite(targetZ) & isfinite(lureZ);
if sum(validPairs) < 2
    da = NaN;
    return;
end
middleLevel = floor(numberOfLevels / 2);
hitRate = targetCumulative(middleLevel);
faRate = lureCumulative(middleLevel);
slope = 0.8;
da = sqrt(2 / (1 + slope ^ 2)) * (norminv(hitRate) - slope * norminv(faRate));
end
