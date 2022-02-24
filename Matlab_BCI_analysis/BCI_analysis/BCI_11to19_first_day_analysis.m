folder = 'F:\BCI\firstDay\';
files = dir(folder);
files = {files.name};files = files(4:end);

% fi = 1;
for fi = 3
    keep folder files fi learning DEL DEC DIST task preTask fileInfo
    global dat
    dat = load([folder,char(files(fi))]);
    [df,dist,F,epoch,tsta,raw] = BCI_dat_extract(dat);
    
    figure(fi);
    figure_initialize
    set(gcf,'position',[3 3 3 4.5])
    dt = median(diff(tsta));
    t = 0:dt:dt*(size(df,1)-1);
    marg = [.4 .3];
    cn = dat.conditioned_neuron;
    KDsubplot(3,1,[1 1],marg)
    plot(t/60,df(:,cn),'k')
    xlabel('Time (min.');ylabel('\DeltaF/F')
    title('Conditioned neuron (CN)')
    str = dat.folder;ind = strfind(str,'BCI');
    str = str(ind(2):ind(2)+4);
    title(str)
    
    KDsubplot(3,1,[2 1],marg)
    bins = round(round(60*5/dt));
    clear smooth
    for i = 1:length(t);
        ind = i-bins:i;
        ind(ind<1)= [];
        smooth(i) = mean(df(ind,cn));
    end
    learning{fi} = smooth;
    plot(t/60,smooth,'k');hold on;
    ylim([-.2 1])
    len = size(df,1);
    %     post = len-10000:len-2000;
    strt = 8;
    post = [floor(strt*60/dt):floor(strt*60/dt)+floor(5*60/dt)];
    pre = 1:bins;
    plot(t(pre)/60,0*pre-.1,'k');
    plot(t(post)/60,0*post-.1,'k');
    xlabel('Time (min.');ylabel('\DeltaF/F')
    
    KDsubplot(3,1,[3 1],marg)
    del = nanmean(df(post,:))-nanmean(df(pre,:));
    DEL{fi} = del;
    DIST{fi} = dist;
    scatter(dist,del,'ko','markerfacecolor','w');hold on;
    scatter(dist(cn),del(cn),'ko','markerfacecolor','r');hold on;
    xlabel('Distance from CN (\mum)');
    ylabel('\Delta activity (late - early)')
    figure_finalize
    print(figure(fi),'-dpng',[folder,'\figures\_',str,'.png'])
    drawnow
    
    
    figure(fi+50);
%     figure_initialize
%     set(gcf,'position',[3 3 2 2])
    
%     KDsubplot(3,1,[1 1],marg);
%     plot(tsta(1:240),squeeze(nanmean(F(1:240,cn,:),3)),'k')
%     xlabel('Time after trial start (s)')
%     ylabel('\DeltaF/F');
%     title(str);
%     KDsubplot(3,1,[2,1],marg);
%     imagesc(squeeze(F(1:240,cn,:))')
%     times = [min(find(tsta>0)) min(find(tsta>5))];
%     set(gca,'xtick',times,'xticklabel',{'0','5'});
%     xlabel('Time after trial start (s)')
%     ylabel('Trial #');
%     colorbar
%     
%     KDsubplot(3,1,[2,1],marg);
    plot(conv(nanmean(squeeze(F(101:end,cn,:))),ones(20,1)),'k');hold on;
    plot(conv(nanmean(squeeze(F(41:100,cn,:))),ones(20,1)),'r')
    plot(conv(nanmean(squeeze(F(1:20,cn,:))),ones(20,1)),'b')
    a = squeeze(F(101:end,cn,:));
    a(isnan(a))=[];
    a = conv(a,ones(3000,1))/3000;
    task{fi} = a(3000:end);
    a = squeeze(F(1:20,cn,:));
    a(isnan(a))=[];
    a = conv(a,ones(80,1))/80;
    Pretask{fi} = a(80:end);
    xlim([20 size(F,3)])
    %     KDsubplot(1,2,[2 1],marg);
    % [a,b] = sort(dist);
    % imagesc(nanmean(F(1:240,b,:),3)',[0 5])
    %
%         KDsubplot(3,1,[3 1],marg);
%         plot(tsta(1:240),squeeze(nanmean(F(1:240,cn,1:20),3)),'c');hold on;
%         plot(tsta(1:240),squeeze(nanmean(F(1:240,cn,40:80),3)),'r')
    figure_finalize
    fileInfo{fi} = dat.folder;
%     print(figure(fi+50),'-dpng',[folder,'\figures\_',str,'_2.png'])
%     
%     figure(90+fi);clf
%     del = squeeze(nanmean(nanmean(F(1:20,:,40:100),3)))-squeeze(nanmean(nanmean(F(1:20,:,1:10),3)));
%     DEC{fi} = del
%     scatter(dist,del,'ko','markerfacecolor','w');hold on;
%     scatter(dist(cn),del(cn),'ko','markerfacecolor','r');hold on;
%     xlabel('Distance from CN (\mum)');
%     ylabel('\Delta activity (late - early)')
%     ylim([-1 1])
%     print(figure(fi+90),'-dpng',[folder,'\figures\_',str,'_3.png'])
end
%%
clf
clear g
for i = 1:5;
    g(:,i) = learning{i}(1:28000);
%     g(:,i) = task{i}(1:20000);
end
% g = g - repmat(mean(g(1:2000,:)),size(g,1),1);
confidence_bounds(t(1:28000)/60,g-.15,[],'k','k',.2);hold on;
plot(xlim,xlim*0,'k:');
% ylim([-.2 .6])
xlabel('Time (min.)')
ylabel('\DeltaF/F')
%%
clf
dist = [DIST{:}];
cn = cellfun(@(x) (x==min(x)),DIST,'uni',0);
del = [DEL{:}];
% scatter(dist,del,'ko','markerfacecolor','w');hold on;
cn = cell2mat(cn);
% scatter(dist(cn),del(cn),'ko','markerfacecolor','r');hold on;
fixed_bin_plots(dist,del,[0 20 50 100:50:1000],1,'k');
xlabel('Distance from CN (\mum)');
ylabel('\Delta activity (late - early)')
xlim([-20 800])
hold on;
plot(xlim,xlim*0,'k:')
figure_finalize




