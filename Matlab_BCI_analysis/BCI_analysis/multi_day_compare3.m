ind =cell2mat(cellfun(@(x) ~isempty(strfind(x,'neuron')),old.bases,'uni',0))
len=cell2mat(cellfun(@(x) length(x),old.siFiles,'uni',0))
ind = find(ind.*len==max(ind.*len))
[dfo,disto,Fo,epoch,tsta,raw0,df_all0] = BCI_dat_extract3(old,old.bases{ind});

ind =cell2mat(cellfun(@(x) ~isempty(strfind(x,'neuron')),new.bases,'uni',0))
len=cell2mat(cellfun(@(x) length(x),new.siFiles,'uni',0))
ind = find(ind.*len==max(ind.*len))
[df,dist,F,epoch,tsta,rawn,df_all] = BCI_dat_extract3(new,new.bases{ind});
% base = dat.bases{newind};
% [df,dist,F,epoch,tsta,raw,df_all] = BCI_dat_extract2(dat,base);
% base = old.bases{oldind};
% [dfo,disto,Fo,epoch,tsta,raw0,df_all0] = BCI_dat_extract2(old,base);
%%
figure(400);
cn = new.conditioned_neuron;
f = nanmean(F(1:240,:,:),3);f = f-repmat(mean(f(1:20,:)),size(f,1),1);
fo = nanmean(Fo(1:240,:,:),3);fo = fo-repmat(mean(fo(1:20,:)),size(fo,1),1);
del = mean(f(40:150,:))-mean(fo(40:150,:));
clf
subplot(3,2,5);
scatter(dist,del,'k');hold on;
scatter(dist(cn),del(cn),'ko','markerfacecolor','r');hold on;
xlim([-20 max(xlim)]);
xlabel('Distance from CN (\mum)')
ylabel('Day 2 - Day 1')
%%
subplot(3,2,6);cla
lag = 0;
cc = corr(df_all{1}(1+lag:end,:),df_all{1}(1:end-lag,cn));
% cc = corr(df_all{1}(1+lag:end,cn),df_all{1}(1:end-lag,:));
ind = 1:length(new.roi);ind(cn) = [];
% pp = plot(cc(ind),del(ind),'o','MarkerFaceColor',[.7 .7 .7],'MarkerEdgeColor','w');hold on;
[a,b,c,d,e] =mean_bin_plot((cc(ind)),del(ind));e
% xlim([-20 max(xlim)]);
xlabel('Spont. Corr. with CN')
ylabel('Day 2 - Day 1')
%%

subplot(321);
plot(tsta(1:240),fo(:,cn),'k')
confidence_bounds(tsta(1:240),fo(:,cn),...
    nanstd(Fo(1:240,cn,:),0,3)/sqrt(size(Fo,3)),'k','k',.2);hold on;
ylim([-.4 1])
xlabel('Time from trial start (s)')
ylabel('\DeltaF/F')
title(old.folder)

% subplot(322);
confidence_bounds(tsta(1:240),f(:,cn),...
    nanstd(F(1:240,cn,:),0,3)/sqrt(size(F,3)),'m','m',.2);
ylim([-.4 1])
[a,b] = sort(sum([f(50:100,:);fo(50:100,:)]));
xlabel('Time from trial start (s)')
title(new.folder)

subplot(322);
learning_map_rois(new,del,[0 max(del)],1)

subplot(323);
ticks = [min(find(tsta>0)) min(find(tsta>5))];
imagesc(fo(:,b)',[0 .4]);
hold on;
set(gca,'colormap',jet)

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
set(gca,'position',a,'colormap',jet)
xlabel('Time from trial start (s)')

hold on;
plot([1 30],[1 1]*find(b==cn),'m','linewidth',3)
figure_finalize

%%
% base = new.bases{new.currentPlane};
% try
%     strt = cell2mat(cellfun(@(x) ~isempty(strfind(x,base)),...
%         new.intensityFile,'uni',0));
%     strt = min(find(strt==1));
% catch
%     strt = 1;
% end
% strt
% % strt = 1;
% for i = 1:n;
%     a = fluor_fun(new.roi(i).intensity(1:strt-1));
%     bl = std(a);
%     bl = prctile(a,50);
%     spnt(:,i) = (a-bl)/bl;
% end