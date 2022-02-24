

w = ws.loadDataFile([dat.folder,char(dat.wsFiles{dat.currentPlane})]);
rois = readtable('D:\KD\BCI_03\050521\neuron8_IntegrationRois_00001.csv');
rois = table2array(rois);

%%
scans = w.sweep_0001.analogScans;
str = dat.bases(dat.currentPlane);
inds = cell2mat(cellfun(@(x) ~isempty(strfind(x,str)),dat.intensityFile,'uni',0));
% f = cell2mat(cellfun(@(x) x',dat.roi(dat.conditioned_neuron).intensity(inds),'uni',0));
% dt_si = 1/dat.siHeader.SI.hRoiManager.scanFrameRate;
dt_ws = 1/w.header.AcquisitionSampleRate;
f = interp1(rois(:,1),rois(:,10),ts,'linear');
ts = 0:dt_si:dt_si*(length(f)-1);
tw = 0:dt_ws:dt_ws*(length(scans)-1);
scan = interp1(tw,scans,ts,'linear');
motor = scans(:,4);
motor = conv(motor,ones(1000,1));motor = motor(1:length(scans));
motor = interp1(tw,motor,ts,'linear');
trl = cell2mat(cellfun(@(x) ones(1,length(x))*rand*1000,dat.roi(1).intensity(inds),'uni',0));
trl = (diff(trl)~=0);
newTrl = find(trl==1);
%%
for i = 1:length(newTrl)-1;
    ind = newTrl(i):newTrl(i+1);
    rew = min(find(scan(ind,5)>1));
%     clf
%     subplot(311);
% %     plot(f(ind));hold on;
%     plot(scan(ind,2));hold on;
% %     plot(xlim,[50 50]);
% %     ylim([0 200])
%     subplot(312);
%     plot(scan(ind,5));
%     ylim([-1  5])
%     subplot(313);
%     plot(motor(ind));
%     pause;
    if ~isempty(rew);
        ind = ind(1:rew);
        tot(i) = mean(f(ind));
        tot(i) = mean(scan(ind,2));
    else
        try
        ind = ind(1:floor(10/dt_si));
        tot(i) = -mean(f(ind));
        tot(i) = -mean(scan(ind,2));
        catch 
            tot(i) = nan;
        end
%         tot(i) = -sum(scan(ind,2));
    end
end
%%
t = ol(:,1);
df = ol(:,10);
ten = floor(10/mean(diff(t)));
for i = 1:1000;
    strt = round(rand*(length(t)-ten))+1;
    ind = strt:strt+ten;
    amp(i) = mean(df(ind));
end
