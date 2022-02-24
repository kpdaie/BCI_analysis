[df,dist,F,epoch,tsta,raw] = BCI_dat_extract(dat);
[dfo,disto,Fo,epoch,tsta,raw0] = BCI_dat_extract(old);
%%
figure(400);
cn = dat.conditioned_neuron;
f = nanmean(F(1:240,:,:),3);f = f-repmat(mean(f(1:20,:)),size(f,1),1);
fo = nanmean(Fo(1:240,:,:),3);fo = fo-repmat(mean(fo(1:20,:)),size(fo,1),1);
del = mean(f(40:150,:))-mean(fo(40:150,:));
clf
subplot(3,1,3);
scatter(dist,del,'k');hold on;
scatter(dist(cn),del(cn),'ko','markerfacecolor','r');hold on;
xlim([-20 max(xlim)]);
xlabel('Distance from CN (\mum)')
ylabel('Day 2 - Day 1')

subplot(321);
plot(tsta(1:240),fo(:,cn),'k')
ylim([-.4 .7])
xlabel('Time from trial start (s)')
ylabel('\DeltaF/F')
title(old.folder)

subplot(322);
plot(tsta(1:240),f(:,cn),'k')
ylim([-.4 .7])
[a,b] = sort(sum([f(50:100,:);fo(50:100,:)]));
xlabel('Time from trial start (s)')
title(dat.folder)

subplot(323);
ticks = [min(find(tsta>0)) min(find(tsta>5))];
imagesc(fo(:,b)',[0 .4]);
hold on;
plot([1 30],[1 1]*find(b==cn),'m','linewidth',3)
set(gca,'xtick',ticks,'xticklabel',{'0','5'})
xlabel('Time from trial start (s)')
ylabel('Neuron #')
xlabel('Time from trial start (s)')


subplot(324);
imagesc(f(:,b)',[0 .4]);
set(gca,'xtick',ticks,'xticklabel',{'0','5'})
a = get(gca,'position');
colorbar;
set(gca,'position',a)
xlabel('Time from trial start (s)')
colormap(jet)

hold on;
plot([1 30],[1 1]*find(b==cn),'m','linewidth',3)
figure_finalize

%%
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
    a = fluor_fun(dat.roi(i).intensity(1:strt-1));
    bl = std(a);
    bl = prctile(a,50);
    spnt(:,i) = (a-bl)/bl;
end