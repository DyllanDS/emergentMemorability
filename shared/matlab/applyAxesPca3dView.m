function applyAxesPca3dView(ax, v)
%APPLYAXESPCA3DVIEW Apply view / camva / camroll from readPca3dFeatureSpaceView.
if ~isgraphics(ax)
    return;
end
view(ax, v.az, v.el);
camva(ax, v.camva);
try
    camroll(ax, v.camroll);
catch
end
end
