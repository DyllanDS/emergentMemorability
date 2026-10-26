function stylePca3dAxes(ax, cb, fontSizePt)
%STYLEPCA3DAXES  Axes and colorbar layout of the 3-D PCA panels (Figs. 5b, 6a, 6b and 7a).
%   ax          axes with scatter3 already drawn
%   cb          colorbar handle (optional)
%   fontSizePt  font size in points (optional, default 15)

if nargin < 3 || isempty(fontSizePt)
    fontSizePt = 15;
end
if ~isgraphics(ax)
    return;
end
fig = ancestor(ax, 'figure');
set(fig, 'Position', get(groot, 'DefaultFigurePosition'), 'Color', 'w');

axesInset = 0.18;
axisLineWidth = fontSizePt / 8;
tickLength = [0.02 0.02] * (fontSizePt / 12);

if nargin >= 2 && ~isempty(cb) && isgraphics(cb)
    cb.FontSize = fontSizePt;
    ax.Position = [axesInset, axesInset, 1 - axesInset - 0.18, 1 - axesInset * 2];
    ax.Position = ax.Position .* [1.15, 1.15, 0.7, 0.7] + [0.08, 0.08, 0, 0];
    cp = cb.Position;
    cb.Position = [cp(1) + 0.04, cp(2), cp(3), cp(4)];
end

grid(ax, 'on');
set(ax, 'LineWidth', axisLineWidth, 'FontSize', fontSizePt, 'TickLength', tickLength);
set(ax, 'XTickLabel', [], 'YTickLabel', [], 'ZTickLabel', []);
drawnow;
end
