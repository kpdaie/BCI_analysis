try
    cl = dat.conditioned_coordinates';
catch
    [dat.siHeader,~] = scanimage.util.opentif([dat.folder,char(dat.siFiles{dat.currentPlane}(1))]);[dat.conditioned_coordinates,dat.conditioned_neuron] = manual_conditioned_neuron_coords(dat,1);
    cl = dat.conditioned_coordinates';
end
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
clear dist
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
clear bl
for i = 1:n;
    f = fluor_fun(dat.roi(i).intensity);
    bll(i) = prctile(f,20);
end
clear F
for j = 1:n;
    for i = 1:nt;
        a = dat.roi(j).intensity{i};
        a = (a-bll(j))/bll(j);
        b = 1000 - length(a);
        if b > 0;
            a = [a;nan(b,1)];
        else
            a = a(1:1000);
        end
        F(:,j,i) = a;
    end;
end
%%
clear df raw bl
base = dat.bases{dat.currentPlane};
try
    strt = cell2mat(cellfun(@(x) ~isempty(strfind(x,base)),...
        dat.intensityFile,'uni',0));
    strt = min(find(strt==1));
catch
    strt = 1;
end
strt = 1;
for i = 1:n;
    a = fluor_fun(dat.roi(i).intensity(strt:end));
                bl = std(a);
    bl = prctile(a,50);
    df(:,i) = (a-bl)/bl;
    
    raw(:,i) = a - min(a);
end

del = nanmean(df(15000:21000,:)) - nanmean(df(1:3000,:));
cn = dat.conditioned_neuron;cnn = df(:,cn);
clf
scatter(dist,del);hold on;
scatter(dist(cn),del(cn),'k','markerfacecolor','r');