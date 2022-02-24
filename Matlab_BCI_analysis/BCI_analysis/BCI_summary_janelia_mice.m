keys = fieldnames(data);
mi = 1;
close all
eval(['dat=data.',keys{mi}]);
clear IM cc
for si = 2:length(dat);
    IM(:,:,si) = dat(si).IM;
end
for i = 2:size(IM,3);
    for j = 1:size(IM,3);
        a = IM(:,:,i);
        b = IM(:,:,j);
        cc(i,j) = corr(a(:),b(:));
    end
end
clf
imagesc(cc,[0 1]);colormap(jet);
colorbar
%%
for si = 3:length(dat);
    
    struct2ws(dat(si));
    Fo = dat(si-1).F;dfo = dat(si-1).df;
    n = size(F,2);
    
    figure(400);set(gcf,'position',[ 680   422   593   676]);
    f = nanmean(F(1:240,:,:),3);f = f-repmat(mean(f(1:40,:)),size(f,1),1);
    fo = nanmean(Fo(1:240,:,:),3);fo = fo-repmat(mean(fo(1:40,:)),size(fo,1),1);
    del = mean(f(40:150,:))-mean(fo(40:150,:));
    clf
    subplot(3,2,5);
    scatter(dist,del,'k');hold on;
    scatter(dist(cn),del(cn),'ko','markerfacecolor','r');hold on;
    xlim([-20 max(xlim)]);
    xlabel('Distance from CN (\mum)')
    ylabel('Day 2 - Day 1')
    title(['\Delta CN =',num2str(round(mean(del<del(cn))*100)),' %tile'])
    
    subplot(3,2,6);cla
    lag = 0;
    cc = corr(df_all{1}(1+lag:end,:),df_all{1}(1:end-lag,cn));
    % cc = corr(df_all{1}(1+lag:end,cn),df_all{1}(1:end-lag,:));
    ind = 1:n;ind(cn) = [];
    % pp = plot(cc(ind),del(ind),'o','MarkerFaceColor',[.7 .7 .7],'MarkerEdgeColor','w');hold on;
    [a,b,c,d,e] =mean_bin_plot((cc(ind)),del(ind));e
    title(['p = ',num2str(e)]);
    % xlim([-20 max(xlim)]);
    xlabel('Spont. Corr. with CN')
    ylabel('Day 2 - Day 1')
    
    
    subplot(321);
    plot(tsta(1:240),fo(:,cn),'k')
    confidence_bounds(tsta(1:240),fo(:,cn),...
        nanstd(Fo(1:240,cn,:),0,3)/sqrt(size(Fo,3)),'k','k',.2);hold on;
    ylim([-.4 1])
    xlabel('Time from trial start (s)')
    ylabel('\DeltaF/F')
    
    
    % subplot(322);
    confidence_bounds(tsta(1:240),f(:,cn),...
        nanstd(F(1:240,cn,:),0,3)/sqrt(size(F,3)),'m','m',.2);
    ylim([-.4 1])
    [a,b] = sort(sum([f(50:100,:);fo(50:100,:)]));
    xlabel('Time from trial start (s)')
    
    
    subplot(322);
    learning_map_rois(dat(si),del,[-max(del) max(del)],1);colormap((parula));
    
    subplot(323);
    ticks = [min(find(tsta>0)) min(find(tsta>5))];
    imagesc(fo(:,b)',[0 .4]);
    hold on;
    set(gca,'colormap',jet)
    str = dat(si-1).file;str = str(max(find(str=='\'))+1:end);ind = find(str=='_');
    str = [keys{mi},' ',str(ind(1)+1:ind(2)-1)]
    title(str)
    
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
    str = dat(si).file;str = str(max(find(str=='\'))+1:end);ind = find(str=='_');
    str = [keys{mi},' ',str(ind(1)+1:ind(2)-1)]
    title(str)
    
    hold on;
    plot([1 30],[1 1]*find(b==cn),'m','linewidth',3)
    figure_finalize
    folder = 'G:\My Drive\Learning rules\BCI_backups\summary_plots\';
    str = dat(si).file;str = str(max(find(str=='\'))+1:end);ind = find(str=='_');
    str = [keys{mi},'_',str(ind(1)+1:ind(2)-1)]
    print(figure(400),'-dpng',[folder,str,'2.png'])
end