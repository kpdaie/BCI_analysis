function [df,dist,F,epoch,tsta,raw] = BCI_dat_extract(dat)
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
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
dist = dist*1.5*(1/zoom);
n = length(dat.roi);
nt = length(dat.roi(1).intensity);
clear F
F = nan(1000,n,nt);
fluor_fun = @(x) cell2mat(cellfun(@(x) x',x,'uni',0));
clear bl
for i = 1:n;
    f = fluor_fun(dat.roi(i).intensity);
%     bll(i) = prctile(f,20);
    bll(i) = std(f);
end
clear F
pre = 40;
for j = 1:n;
    for i = 1:nt;
        try
            g = dat.roi(j).intensity{i-1}(end-pre-1:end);
        catch
            g = nan(pre,1);
        end
        a = [g;dat.roi(j).intensity{i}];
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
tsta = 0:dt_si:dt_si*(size(F,1)-1);
tsta = tsta - tsta(pre);
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
strt
% strt = 1;
for i = 1:n;
    a = fluor_fun(dat.roi(i).intensity(strt:end));
    bl = std(a);
    bl = prctile(a,50);
    df(:,i) = (a-bl)/bl;
    
    raw(:,i) = a - min(a);
end
%%
L = cell2mat(cellfun(@(x) length(x),dat.roi(1).intensity,'uni',0));
try
    files = dat.intensityFile';
    for i = 1:length(files)
        s = char(files{i});
        s = s(max(find(s=='\'))+1:max(find(s=='_'))-1);
        epoch(i) = find(cell2mat(cellfun(@(x) strcmp(x,s),dat.bases,'uni',0))==1);
    end
catch 
    epoch = [];
end














