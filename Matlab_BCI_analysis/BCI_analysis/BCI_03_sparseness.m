keep dat
w = ws.loadDataFile([dat.folder,char(dat.wsFiles{dat.currentPlane})]);
%%
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
clear bl
for i = 1:n;
    f = fluor_fun(dat.roi(i).intensity);
    bl(i) = prctile(f,20);
end
clear F
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
clear df
base = dat.bases{dat.currentPlane};
try
    strt = cell2mat(cellfun(@(x) ~isempty(strfind(x,base)),...
        dat.intensityFile,'uni',0));
catch
    strt = 1;
end
strt = min(find(strt==1));
for i = 1:n;
    a = fluor_fun(dat.roi(i).intensity(strt:end));
    bl = std(a);
    bl = prctile(a,50);
    df(:,i) = (a-bl)/bl;
end
dt_si = 1/dat.siHeader.SI.hRoiManager.scanFrameRate;
ts = 0:dt_si:dt_si*(length(df)-1);
scans = w.sweep_0001.analogScans;
dt_ws = 1/w.header.AcquisitionSampleRate;
tw = 0:dt_ws:dt_ws*(length(scans)-1);
scan = interp1(tw,scans,ts,'linear');
rew = scan(:,5);
rind = find(rew>1);
rind(diff(rind)<2) = [];
%%
clf
subplot(121);
cn = dat.conditioned_neuron;cnn = df(:,cn);
plot(conv(cnn,ones(6000,1))/6000,'k');xlim([6000 length(cnn)]);hold on
yl = ylim;
% title(dat.folder)
box off
ind1 = 1:1000;
ind2 = 12000:22000;
del = mean(df(ind2,:)) - mean(df(ind1,:));
subplot(122);
scatter(dist,del,'k');
subplot(121);
plot(ind1([1 end])+0,[1 1]*yl(1)+.1,'color',[.5 .5 .5]);hold on;
plot(ind2([1 end]),[1 1]*yl(1)+.1,'k');hold on;
%%
clf
cn = dat.conditioned_neuron;
clear rta
for i = 1:length(rind)
    ind = rind(i)-200:rind(i)+100;
    ind(ind<1) = 1;ind(ind>length(df))=length(df);
    rta(:,:,i) = df(ind,:);
end
k = squeeze(mean(dr(51:160,:,:)))-squeeze(mean(dr(201:end,:,:)));
d = mean(k(:,51:end-10)')-mean(k(:,1:21)');
scatter(dist,d,'k')