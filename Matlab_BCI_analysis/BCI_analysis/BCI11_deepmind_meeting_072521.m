dat720 = load('F:\BCI\BCI11\072021\session_072021_analyzed_dat_small_neuron10_22072521.mat');
dat721 = load('F:\BCI\BCI11\072121\session_072121_analyzed_dat_small_neuron1072221.mat')
dat712 = load('F:\BCI\BCI11\071221\session_071221_analyzed_dat_small_neuron16071221.mat')
dat722 = load('F:\BCI\BCI11\072221\session_072221_analyzed_dat_small_neuron1072321.mat');
dat723 = load('F:\BCI\BCI11\072321\session_072321_analyzed_dat_small_neuron1072421.mat');
dat726 = load('F:\BCI\BCI11\072621\session_072621_analyzed_dat_small_neuron37072621.mat')
%%
figure(1);
subplot(121);
imagesc(dat720.IM)
showROIsPatch(gcf,'r',dat720.roi,1,0)
xlim([450 650]);ylim([1 200])
title('July 20')
set(gca,'visible','off')

subplot(122);
imagesc(dat721.IM)
showROIsPatch(gcf,'r',dat721.roi,1,0)
colormap(gray)
xlim([450 650]);ylim([1 200])
title('July 21')
set(gca,'visible','off')
%%
figure(1);clf;
dat = dat720;
fluor_fun = @(x) cell2mat(cellfun(@(x) x',x,'uni',0));strt = 1;
cn = dat.conditioned_neuron;
a = fluor_fun(dat.roi(cn).intensity(strt:end));
bl = std(a);
bl = prctile(a,50);
df = (a-bl)/bl;
subplot(121);
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
t = 0:dt_si:dt_si*(length(df)-1);
plot(t/60,df,'r')
box off;xlabel('Time (min.)');ylabel('\DeltaF/F');

clear F
nt = length(dat.roi(1).intensity);
for j = cn;
    for i = 1:nt;
        a = dat.roi(j).intensity{i};
        bl = prctile(a,20);
        a = (a-bl)/bl;
        b = 1000 - length(a);
        if b > 0;
            a = [a;nan(b,1)];
        else
            a = a(1:1000);
        end
        F(:,j,i) = a;
    end;
end
subplot(222);
tsta = 0:dt_si:dt_si*(size(F,1)-1);
plot(tsta,squeeze(nanmean(F(:,cn,:),3)),'r');
xlim([0 10]);
box off;ylabel('\DeltaF/F');
subplot(224);
ind = find(tsta<10);
imagesc(squeeze(F(ind,cn,:))',[1 5]);colormap(parula);colorbar
set(gca,'xtick',xlim,'xticklabel',{'0','10'});
xlabel('Time from trial start (s)');
figure(5);
subplot(122);
k = squeeze(nanmean(F(50:150,cn,:)));
plot(conv(k,ones(20,1))/20,'k');box off
xlim([20,size(F,3)]);
xlabel('Trial #');ylabel('Response size (\DeltaF/F)');
figure(7);clf
plot(tsta(ind),squeeze(nanmean(F(ind,cn,1:end),3)),'k');hold on;
%%
figure(2);clf;
dat = dat712;
imagesc(dat.IM,[0 3000]);colormap(gray)
showROIsPatch(gcf,'r',dat.roi,dat.conditioned_neuron,0)
xlim([500 700]);ylim([300 500]);
set(gca,'visible','off');
figure(3)
fluor_fun = @(x) cell2mat(cellfun(@(x) x',x,'uni',0));strt = 1;
cn = dat.conditioned_neuron;
a = fluor_fun(dat.roi(cn).intensity(strt:end));
bl = std(a);
bl = prctile(a,50);
df = (a-bl)/bl;
subplot(121);
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
t = 0:dt_si:dt_si*(length(df)-1);
plot(t/60,df,'r')
box off;xlabel('Time (min.)');ylabel('\DeltaF/F');

clear F
nt = length(dat.roi(1).intensity);
n = length(dat.roi);
for j = 1:n;
    for i = 1:nt;
        a = dat.roi(j).intensity{i};
        bl = prctile(a,20);
        a = (a-bl)/bl;
        b = 1000 - length(a);
        if b > 0;
            a = [a;nan(b,1)];
        else
            a = a(1:1000);
        end
        F(:,j,i) = a;
    end;
end
subplot(222);
tsta = 0:dt_si:dt_si*(size(F,1)-1);
plot(tsta,squeeze(nanmean(F(:,cn,:),3)),'r');
xlim([0 10]);
box off;ylabel('\DeltaF/F');
subplot(224);
ind = find(tsta<10);
imagesc(squeeze(F(ind,cn,:))',[1 5]);colormap(parula);colorbar
set(gca,'xtick',xlim,'xticklabel',{'0','10'});
xlabel('Time from trial start (s)');
figure(5);
subplot(121);
k = squeeze(nanmean(F(50:150,cn,:)));
plot(conv(k,ones(20,1))/20,'k');box off
xlim([20,size(F,3)]);
xlabel('Trial #');ylabel('Response size (\DeltaF/F)');
%%
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
clear dist
cl = dat.conditioned_coordinates';
for i = 1:length(dat.roi);
    dist(i) = sqrt(sum((dat.roi(i).centroid - cl).^2));
end
zoom = dat.siHeader.SI.hRoiManager.scanZoomFactor;
dist = dist*1.5*(1/zoom);
k = squeeze(nanmean(F(50:150,:,:)));
del = nanmean(k(:,60:120)') - nanmean(k(:,1:40)');
figure(6);clf
scatter(dist,del,'k');hold on;scatter(dist(cn),del(cn),'k','markerfacecolor','r');
xlabel('Distance from Neuron 1 (\mum)');
ylabel(['Change in response amp.',char(10),' (trials 60:120 vs trials 1:40)']);
figure(7);
plot(tsta(ind),squeeze(nanmean(F(ind,cn,1:40),3)),'color',[.6 .6 .6]);hold on;
plot(tsta(ind),squeeze(nanmean(F(ind,cn,60:120),3)),'color',[.3 .3 .3])

xlabel('Time from trial start (s)')
ylabel('\DeltaF/F')
legend('7/20','7/12 trl 1:40','7/12 trl 60:120')
%%
dat = dat722;
[df,dist,F] = BCI_dat_extract(dat);

clear bind
for i = 1:length(dat.intensityFile);
    name = char(dat.intensityFile{i});
    name = name(max(find(name=='\'))+1:end)
    ind = find(name == '_');
    name = name(1:ind(end)-1);
    a = cell2mat(cellfun(@(x) (strcmp(name,x)),...
    dat.bases,'uni',0));
    bind(i) = find(a == 1);
end
L = cell2mat(cellfun(@(x) length(x),dat.roi(1).intensity,'uni',0));
strt = min(find(bind==1));
quiet = min(find(bind==4));
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
ts = 0:dt_si:dt_si*(size(df,1)-1);
ts = ts/60;
cn = dat.conditioned_neuron;
figure(8);clf
subplot(211);
quietT = ts(sum(L(1:quiet-1)));
plot(ts,df(:,cn),'r');hold on;
plot([1 1]*quietT,ylim,'k','linewidth',2);
ylabel('\DeltaF/F');
box off
subplot(212);
win = conv(df(:,cn),ones(2000,1)/2000);
win = win(1:size(df,1));
plot(ts(2001:end),win(2001:end),'r');hold on;
plot([1 1]*quietT,ylim,'k','linewidth',2);
box off
xlabel('Time (min.)');
ylabel('Smoothed fluorescence')
figure(9)
tsta = 0:dt_si:dt_si*(size(F,1)-1);
ind = find(tsta<10);
imagesc(squeeze(F(ind,cn,:))',[0 5]);hold on;
set(gca,'xtick',xlim,'xticklabel',{'0','10'});
xlabel('Time from trial start (s)');
ylabel('Trial number');
hold on;
plot(xlim,[1 1]*quiet,'k:','linewidth',2)
figure(10);
subplot(211);
imagesc(dat.IM,[0 1000]);colormap(gray);
showROIsPatch(gcf,'r',dat.roi,1,0)
showROIsPatch(gcf,'b',dat.roi,3,0)
subplot(223);
plot(tsta(ind),squeeze(nanmean(F(ind,cn,1:25),3)),'r');hold on;
plot(tsta(ind),squeeze(nanmean(F(ind,3,1:25),3)),'b');hold on;box off
title('Trials 1:25');
subplot(224);
plot(tsta(ind),squeeze(nanmean(F(ind,cn,50:90),3)),'r');hold on;
plot(tsta(ind),squeeze(nanmean(F(ind,3,50:90),3)),'b');hold on;box off
title('Trials 50:90');

figure(11);
k = squeeze(nanmean(F(50:150,:,:)));
del = nanmean(k(:,50:90)') - nanmean(k(:,1:25)');
figure(6);clf
scatter(dist,del,'k');hold on;
scatter(dist(cn),del(cn),'k','markerfacecolor','r');
scatter(dist(3),del(3),'k','markerfacecolor','b');
xlabel('Distance from Neuron 1 (\mum)');
ylabel(['Change in response amp.',char(10),' (trials 50:90vs trials 1:25)']);
ylim([-2 2]);xlim([-30 max(xlim)])
%%
dat = dat723;
[df,dist,F] = BCI_dat_extract(dat);
%%
figure(12);
cn = dat.conditioned_neuron;
win = conv(df(:,cn),ones(2000,1)/2000);
win = win(1:length(df));
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
ts = 0:dt_si:dt_si*(length(df)-1);
subplot(121);
plot(ts/60,df(:,cn),'m');
xlabel('Time (min.)');
ylabel('\Delta F/F');
box off
subplot(122);
plot(ts/60,win,'m');
ylabel('Smoothed fluorescence');
box off
figure(13);clf
imagesc(dat.IM,[0 1000]);colormap(gray)
showROIsPatch(gcf,'r',dat.roi,1,0)
showROIsPatch(gcf,'b',dat.roi,3,0)
showROIsPatch(gcf,'m',dat.roi,cn,0)
figure(14);clf
cls = [1 3 cn];
clear win;
colors = [1 0 0;0 0 1;1 0 1];
for i = 1:length(cls);
    a = conv(df(:,cls(i)),ones(3000,1)/3000);
    a = a(1:length(df));
    KDsubplot(3,2,[i 1],.5);
    plot(ts(3000:end)/60,a(3000:end),'color',colors(i,:));
    ylim([-.2 1]);
    box off
    if i == 3;
        xlabel('Time (min.)');
    end
    if i == 2;
        ylabel('Smoothed fluorescence');
    end
    KDsubplot(3,2,[i 2],[.7 .5]);
    imagesc(squeeze(F(ind,cls(i),:))',[0 5])
    set(gca,'xtick',xlim,'xticklabel',{'0','10'})
    set(gca,'ytick',[1 100]);
    if i == 2;
        ylabel('Trial #');
    end
    if i == 3;
        xlabel('Time (s)');
    end
end
figure(20);clf
k = squeeze(nanmean(F(50:150,:,:)));
del = nanmean(k(:,50:90)') - nanmean(k(:,1:25)');
figure(6);clf
scatter(dist,del,'k');hold on;
scatter(dist(cn),del(cn),'k','markerfacecolor','m');
scatter(dist(3),del(3),'k','markerfacecolor','b');
scatter(dist(1),del(1),'k','markerfacecolor','r');
xlabel('Distance from Neuron 1 (\mum)');
ylabel(['Change in response amp.',char(10),' (trials 50:90vs trials 1:25)']);
ylim([-2 2]);xlim([-30 max(xlim)])
%%
dat = dat720;
[df,dist,F20] = BCI_dat_extract(dat);
dat = dat722;
[df,dist,F22] = BCI_dat_extract(dat);
dat = dat723;
[df,dist,F23] = BCI_dat_extract(dat); 
%%
p20 = nanmean(F20(1:200,:,:),3);p20 = p20 - repmat(nanmean(p20(1:10,:)),size(p20,1),1);
p22 = nanmean(F22(1:200,:,2:24),3);p22 = p22 - repmat(nanmean(p22(1:10,:)),size(p22,1),1);
p22_2 = nanmean(F22(1:200,:,50:90),3);p22_2 = p22_2 - repmat(nanmean(p22_2(1:10,:)),size(p22_2,1),1);
p23 = nanmean(F23(1:200,:,90:end),3);p23 = p23 - repmat(nanmean(p23(1:10,:)),size(p23,1),1);
% ind = find(nanvar(p20)<1 & nanvar(p22)<1);
ind = find(isnan(p22(1,:))==0 & isnan(p23(1,:))==0 & nanvar(p20)<10 & nanvar(p22)<10);
p20 = p20(:,ind);p22=p22(:,ind);
p22_2 = p22_2(:,ind);
p23 = p23(:,ind);

n = size(F20,2);
clear f20 f23 f22
for i = 1:n
    bl = squeeze(nanmean(nanmean(F20(1:10,i,:),3)));
    f20(:,i,:) = F20(1:200,i,:) - bl;
    bl = squeeze(nanmean(nanmean(F23(1:10,i,:),3)));
    f23(:,i,:) = F23(1:200,i,:) - bl;
    bl = squeeze(nanmean(nanmean(F22(1:10,i,:),3)));
    f22(:,i,:) = F22(1:200,i,:) - bl;
end
    
%%

[df26,dist,F26] = BCI_dat_extract(dat726);
[df23,dist,F23] = BCI_dat_extract(dat723);
%%
p26 = nanmean(F26(1:200,:,50:end),3);p26 = p26 - repmat(mean(p26(1:2,:)),size(p26,1),1);
p23 = nanmean(F23(1:200,:,50:end),3);p23 = p23 - repmat(mean(p23(1:2,:)),size(p23,1),1);
figure(32);gcf
[a,b] = sort(sum(p23+p26));
subplot(223)
imagesc(p23(:,b)',[-1 1])
set(gca,'xtick',[1 200],'xticklabel',{'0','10'});
xlabel('Time (s)');
ylabel('Neuron #');
subplot(224)
imagesc(p26(:,b)',[-1 1])
subplot(221)
imagesc(dat723.IM,[0 4000])
colormap(gca,'gray')
imagesc(dat723.IM,[0 1000])
colormap(gca,'gray')
subplot(222)
imagesc(dat726.IM,[0 500])
colormap(gca,'gray')
title('July 26')
subplot(221)
imagesc(dat726.IM,[0 500])
colormap(gca,'gray')
showROIsPatch(gcf,[1 .5 0],dat726.roi,dat726.conditioned_neuron,0)
title('July 23')

dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
tsta = 0:dt_si:dt_si*(size(p23,1)-1);
figure(33);clf
first = [df23(:,dat723.conditioned_neuron);df26(:,dat723.conditioned_neuron)];
second = [df23(:,dat726.conditioned_neuron);df26(:,dat726.conditioned_neuron)];
t = 0:dt_si:dt_si*(length(first)-1);
t = t/60;
newDay = t(length(df23)+1);
subplot(221);
plot(t,first,'c');hold on;
plot([1 1]*newDay,ylim,'k:','linewidth',2);
subplot(222);
plot(t,second,'color',[1 .5 0]);hold on;
plot([1 1]*newDay,ylim,'k:','linewidth',2);
subplot(223);
win1 = conv(first,ones(3000,1))/3000;
win1 = win1(1:length(second));
win1(1:3000) = nan;
plot(t,win1,'c');box off
subplot(224);
win2 = conv(second,ones(3000,1))/3000;
win2 = win2(1:length(second));
win2(1:3000) = nan;
plot(t,win2,'color',[1 .5 0]);box off

