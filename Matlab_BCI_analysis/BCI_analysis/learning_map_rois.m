function learning_map_rois(dat,del,clim,show_cn);

n = length(dat.roi);
clear XY
for ii = 1:length(dat.roi);
    try
        XY(1,ii) = dat.roi(ii).centroid(1);
        XY(2,ii) = dat.roi(ii).centroid(2);
    catch
        XY(1:2,ii) = 0;
    end
end

mask = 0*dat.IM;
for i = 1:n;
    pixels = dat.roi(i).pixelList;
    mask(pixels) = del(i);
end
if nargin == 2;
    imagesc(mask);
else
    imagesc(mask,clim);
end
colormap(parula)
cb = colorbar;
% set(cb,'FontColor','w');

a = get(cb,'position');
a(4) = a(4)/5;
set(cb,'position',a);
try
    cn = dat.conditioned_neuron;
catch
    cn = dat.cn;
end
hold on;
if nargin == 4;
    if show_cn == 1
        plot(XY(1,cn),XY(2,cn),'mo','markersize',20,'linewidth',2);
        set(gca,'xtick',[],'ytick',[]);
    end
end


