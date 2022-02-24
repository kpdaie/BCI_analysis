function [centerXY_pix,cn] = manual_conditioned_neuron_coords(dat,roi_idx)

[~,~,hig] = scanimage.util.readTiffRoiData([dat.folder,dat.siFiles{dat.currentPlane}{5}]);
XY = hig.rois(roi_idx).scanfields.centerXY;

rect = dat.siHeader.SI.hRoiManager.imagingFovDeg;
scaling = dat.dim./range(rect);
centerXY_pix = ((XY - rect(1,:)).*scaling)';

if isfield(dat,'roi');
    cl = centerXY_pix';
    for i = 1:length(dat.roi);
        dist(i) = sqrt(sum((dat.roi(i).centroid - cl).^2));
    end
    [a,b] = sort(dist);
    cn = b(1);
    a(1)
    if a(1)>10
        'error'
    end
end