photostim_group_old('F:\BCI\BCI11\072021\')
global dat
ws10 = ws.loadDataFile([dat.folder,char(dat.wsFiles{3})]);
ws2 = ws.loadDataFile([dat.folder,char(dat.wsFiles{4})]);
%%
t_si2 = neuron2(:,1);
t_si10 = neuron10(:,1);

dt_ws = 1/ws2.header.AcquisitionSampleRate;

strt10 = [(1-double(ws10.sweep_0001.digitalScans(:,1)))];
% strt10 = [(double(ws10.sweep_0001.analogScans(:,1)))];
t_ws10 = 0:dt_ws:dt_ws*(length(strt10)-1);
ind = find(t_ws10 < t_si10(end));
t_ws10 = t_ws10(ind);
strt10 = strt10(ind);
strt10 = interp1(t_ws10,strt10,t_si10,'linear');

strt2 = [(1-double(ws2.sweep_0001.digitalScans(:,1)))];
% strt2 = [(double(ws2.sweep_0001.analogScans(:,1)))];
t_ws2 = 0:dt_ws:dt_ws*(length(strt2)-1);
ind = find(t_ws2 < t_si2(end));
t_ws2 = t_ws2(ind);
strt2 = strt2(ind);
strt2 = interp1(t_ws2,strt2,t_si2,'linear');

strt = [strt10;strt2];
rois = [neuron10;neuron2];

ind10 = find(diff(strt10)>.1);
ind10(diff(ind10)<10) = [];
ind = find(diff(strt)>.1);
ind(diff(ind)<10) = [];
clear sta
for i = 1:length(ind);
    win = ind(i)-100:ind(i)+200;
    win(win<1) = 1;
    win(win>length(rois)) = length(rois);
    sta(:,i) = rois(win,12);
end
len = length(ind10);
clf
subplot(121)
dt_si = mean(diff(rois(:,1)));
t = 0:dt_si:dt_si*(length(sta)-1);
t = t - t(100);
plot(t,nanmean(sta(:,1:len),2),'b');hold on;
plot(t,nanmean(sta(:,len+1:80),2),'r');hold on;
p = patch(t([130 200 200 130]),[min(ylim) min(ylim) max(ylim) max(ylim)],'r');
p.FaceColor = 'k';
p.FaceAlpha = .2;
p.EdgeColor = 'none';
p = patch(t([100 120 120 100]),[min(ylim) min(ylim) max(ylim) max(ylim)],'r');
p.FaceColor = 'k';
p.FaceAlpha = .2;
p.EdgeColor = 'none';
legend('Before switch','After switch');
xlabel('Time from trial start');
ylabel('Fluorescence');
box off
subplot(122);
quitt = 80;
k = nanmean(sta(130:200,:)) - nanmean(sta(100:120,:));
plot(conv(k,ones(1,1)),'k');hold on;
plot([1 1]*quitt,ylim,'m','linewidth',2)
plot([1 1]*len,ylim,'r','linewidth',2)
xlim([1 90])
xlabel('Trial #')
ylabel('\Delta F')
legend('\Delta activity trial start','Switch CN','Quit licking');
box off
