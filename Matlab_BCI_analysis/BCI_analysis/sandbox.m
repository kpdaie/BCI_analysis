cl = dat.conditioned_coordinates';
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
for i = 1:length(dat.roi);
    dist(i) = sqrt(sum((dat.roi(i).centroid - cl).^2));
end
zoom = dat.siHeader.SI.hRoiManager.scanZoomFactor;
dist = dist*1.5*(1/zoom);
n = length(dat.roi);
nt = length(dat.roi(1).intensity);
clear F
F = nan(1000,n,nt);
fluor_fun = @(x) cell2mat(cellfun(@(x) x',x,'uni',0));
for i = 1:n;
    f = fluor_fun(dat.roi(i).intensity);
    bl(i) = prctile(f,20);
end
for j = 1:n;
    for i = 1:nt;
        a = dat.roi(j).intensity{i};
        a = (a-bl(j))/bl(j);
        b = 1000 - length(a);
        if b > 0;
            a = [a;nan(b,1)];
        else
            a = a(1:1000);
        end
        F(:,j,i) = a;
    end;
end

for i = 1:n;
    a = fluor_fun(dat.roi(i).intensity(3:end));
    bl = std(a);
    df(:,i) = (a-bl)/bl;
end