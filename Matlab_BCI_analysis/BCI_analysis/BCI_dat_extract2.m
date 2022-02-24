function [df,dist,F,epoch,tsta,raw,df_all,epochName,Fraw] = BCI_dat_extract4(dat,base,len)
if nargin == 2
    len = 240;
end
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
F = nan(len,n,nt);
Fraw = nan(len,n,nt);
fluor_fun = @(x) cell2mat(cellfun(@(x) x',x,'uni',0));
clear bl
for i = 1:n;
    f = fluor_fun(dat.roi(i).intensity);
    %     bll(i) = prctile(f,20);
    bll(i) = std(f);
end
clear F
pre = 40;
strt = cell2mat(cellfun(@(x) ~isempty(strfind(x,base)),...
    dat.intensityFile,'uni',0));
strt = find(strt==1);
nt = length(strt);
for j = 1:n;
    for i = 1:nt;
        try
            g = dat.roi(j).intensity{strt(i-1)}(end-pre-1:end);
        catch
            g = nan(pre,1);
        end
        a = [g;dat.roi(j).intensity{strt(i)}];
        araw = a;
        a = (a-bll(j))/bll(j);
        b = len - length(a);
        if b > 0;
            a = [a;nan(b,1)];
            araw = [araw;nan(b,1)];
        else
            a = a(1:len);
            araw = araw(1:len);
        end
        F(:,j,i) = a;
        Fraw(:,j,i) = araw;
    end;
end
tsta = 0:dt_si:dt_si*(size(F,1)-1);
tsta = tsta - tsta(pre);
%%
clear df raw bl
% base = dat.bases{dat.currentPlane};
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


base = cellfun(@(x) x(max(find(x=='\'))+1:max(find(x=='_'))-1),dat.intensityFile','uni',0);
bases = unique(base,'stable');
for bi = 1:length(bases);
    strt = cell2mat(cellfun(@(x) (strcmp(x,bases{bi})),...
        base,'uni',0));
    inds = (find(strt==1));
    for i = 1:n;
        a = fluor_fun(dat.roi(i).intensity(inds));
        bl = std(a);
        bl = prctile(a,50);
        df_all{bi}(:,i) = (a-bl)/bl;
    end
end

%%
L = cell2mat(cellfun(@(x) length(x),dat.roi(1).intensity,'uni',0));
try
    files = dat.intensityFile';
    for i = 1:length(files)
        s = char(files{i});
        s = s(max(find(s=='\'))+1:max(find(s=='_'))-1);
        epoch(i) = find(cell2mat(cellfun(@(x) strcmp(x,s),dat.bases,'uni',0))==1);
        epoch_Name = dat.bases(epoch);
    end
catch
    epoch = [];
end














