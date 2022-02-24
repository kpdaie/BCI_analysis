
parent_folder = 'D:\KD\BCI_data\Janelia_multiday\';
files = dir(parent_folder);files = {files.name};
files = files(cell2mat(cellfun(@(x) ~isempty(strfind(x,'BCI')),files,'uni',0)))';
for mi = 1:length(files);
    file = [parent_folder,char(files{mi})];
    
    dat = load(file);
    dat = dat.data;
    
    %     figure(1)
    %     for si = 2:length(dat);
    %         IM(:,:,si) = dat(si).IM;
    %     end
    %     for i = 2:size(IM,3);
    %         for j = 1:size(IM,3);
    %             a = IM(:,:,i);
    %             b = IM(:,:,j);
    %             cc(i,j) = corr(a(:),b(:));
    %         end
    %     end
    %     clf
    %     imagesc(cc,[0 1]);colormap(jet);
    %     colorbar
    
    for si = 2:length(dat)
        try
            struct2ws(dat(si));
            Fo = dat(si-1).Fraw+10;dfo = dat(si-1).df;
            
            if mi == 10;
                Fo = dat(si-1).Fraw(:,1:181,:)+10;dfo = dat(si-1).df;
            end
            
            n = size(F,2);
            F = Fraw+10;
            for ci = 1:n
                a = F(:,ci,:);
                bl = nanmean(nanmean(F(1:40,ci,:)));
                %     bl = prctile(a(:),10);
                bl = nanmedian(a(:));
                bl = nanstd(a(:));
                F(:,ci,:) = (a-bl)/bl;
                
                a = Fo(:,ci,:);
                bl = nanmean(nanmean(F(1:40,ci,:)));
                %     bl = prctile(a(:),10);
                bl = nanmedian(a(:));
                bl = nanstd(a(:));
                Fo(:,ci,:) = (a-bl)/bl;
            end

            
            figure(400);set(gcf,'position',[ 300 100   1000 1000]);
            f = nanmean(F(1:240,:,:),3);f = f-repmat(mean(f(1:40,:)),size(f,1),1);
            fo = nanmean(Fo(1:240,:,:),3);fo = fo-repmat(mean(fo(1:40,:)),size(fo,1),1);
            del = mean(f(40:150,:))-mean(fo(40:150,:));
            clf
            subplot(3,3,7);
            scatter(dist,del,'k');hold on;
            scatter(dist(cn),del(cn),'ko','markerfacecolor','m');hold on;
            xlim([-20 max(xlim)]);
            xlabel('Distance from CN (\mum)')
            ylabel('Day 2 - Day 1')
            title(['\Delta CN =',num2str(round(mean(del<del(cn))*100)),' %tile'])
            sparse(mi,si) = mean(del<del(cn));
            
            subplot(3,3,8);cla
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
            pval(mi,si) = e;
            cval(mi,si) = d;
            
            
            subplot(321);
            plot(tsta(1:240),fo(:,cn),'k');hold on;plot(tsta,f(:,cn),'m');
            confidence_bounds(tsta(1:240),fo(:,cn),...
                nanstd(Fo(1:240,cn,:),0,3)/sqrt(size(Fo,3)),'k','k',.2);hold on;
            ylim([-.4 1])
            confidence_bounds(tsta(1:240),f(:,cn),...
                nanstd(F(1:240,cn,:),0,3)/sqrt(size(F,3)),'m','m',.2);
            legend('Day 0','Day 1')
            
            [a,b] = sort(sum([f(50:100,:);fo(50:100,:)]));
            subplot(322);
            learning_map_rois(dat(si),del,[-max(del) max(del)],1);colormap(bluewhitered);
            axis square
           
            
            subplot(6,3,10);
            imagesc(f(:,b)',[0 .4]);
            set(gca,'xtick',ticks,'xticklabel',{'0','5'})
            a = get(gca,'position');
            hold on;
            plot([1 30],[1 1]*find(b==cn),'m','linewidth',3)            
            set(gca,'position',a,'colormap',jet)
%             cb=colorbar;a=get(cb,'position');a(4)=a(4)/4;set(cb,'position',a)
            xlabel('Time from trial start (s)')
            str = dat(si).file;str = str(max(find(str=='\'))+1:end);ind = find(str=='_');
            str = [dat(1).mouse,' ',str(ind(1)+1:ind(2)-1)]
            title(str)
            
            subplot(6,3,7);
            ticks = [min(find(tsta>0)) min(find(tsta>5))];
            imagesc(fo(:,b)',[0 .4]);
            hold on;
            plot([1 30],[1 1]*find(b==cn),'m','linewidth',3)
            set(gca,'colormap',jet)
            set(gca,'xtick',ticks,'xticklabel',{'0','5'})
            str = dat(si-1).file;str = str(max(find(str=='\'))+1:end);ind = find(str=='_');
            str = [dat(1).mouse,' ',str(ind(1)+1:ind(2)-1)]
            title(str)
            
            hold on;
            plot([1 30],[1 1]*find(b==cn),'m','linewidth',3)
            figure_finalize
            folder = 'G:\My Drive\Learning rules\BCI_backups\summary_plots\';
            str = dat(si).file;str = str(max(find(str=='\'))+1:end);ind = find(str=='_');
            str = [dat(1).mouse,'_',str(ind(1)+1:ind(2)-1)]
            
%             subplot(3,5,8);
%             scatter(mean(fo(40:150,:)),mean(f(40:150,:)),'ko');
%             xlabel('Amp. Day 0');
%             ylabel('Amp. Day 1');
%             
            
            subplot(3,3,6);
            k = squeeze(F(:,cn,:));dk = k - nanmean(nanmean(k(1:40,:)));
            len = 10;
            plot(conv(nanmean(k(40:150,:)),ones(len,1))/len,'k')
            box off
            xlim([len size(k,2)]);
            xlabel('Trial #');
            ylabel('Amp.');
            
            subplot(3,3,5);
            stack_plot_patch(tsta,dk,[-.0 -.7])
            xlabel('Time from trial start (s)');
            title('CN (single trials)');
            set(gca,'ytick',[min(ylim) max(ylim)]);
            set(gca,'yticklabel',{num2str(size(dk,2)),'1'});
            ylabel('Trial #');
            
            fo2 = nanmean(Fo(1:240,:,2:2:end),3);fo2 = fo2-repmat(mean(fo2(1:40,:)),size(fo2,1),1);
            fo1 = nanmean(Fo(1:240,:,1:2:end),3);fo1 = fo1-repmat(mean(fo1(1:40,:)),size(fo1,1),1);
            f1 = nanmean(F(1:240,:,:),3);f1 = f1-repmat(mean(f1(1:40,:)),size(f1,1),1);
            f2 = nanmean(F(1:240,:,:),3);f2 = f2-repmat(mean(f2(1:40,:)),size(f2,1),1);
            [u,s,v] = svd(fo1);lv = mean(f1(40:150,:))-mean(fo1(40:150,:)); 
            for i = 1:size(v,2);
                v(:,i) = v(:,i)*sign(sum(fo1*v(:,i)));
            end
            v = Gram_Schmidt_Process([v(:,1:3) lv'/norm(lv)]);   
%             subplot(3,3,9);
            d2 = (f2-fo2)*v(:,1:end-1)*v(:,1:end-1)';
            off_manifold_frac(mi,si) = (sum(f2(40:150,cn)-fo2(40:150,cn))-sum(d2(40:150,cn)))/sum(f2(40:150,cn)-fo2(40:150,cn));            
            off_manifold(mi,si) = (sum(d2(40:150,cn)));            
            delta_cn(mi,si)    = sum(f2(40:150,cn)-fo2(40:150,cn));
            plot(tsta,d2(:,cn),'color',[.7 .7 .7]);hold on;
            d2 = (f2-fo2)*v(:,end)*v(:,end)';
%             on_manifold = sum(d2(40:150,cn))/sum(f2(40:150,cn)-fo2(40:150,cn));
            plot(tsta,d2(:,cn),'color',[.7 .7 .7]*0);hold on;
%             ll= legend('\Delta On manifold','\Delta Off manifold');set(ll,'box','off');
%             box off
%             v = v(:,[1 2 end]);
%             for ii = 1:3;
%                 subplot(1,3,ii);
%                 plot(tsta,fo2*v(:,ii),'k');hold on;
%                 plot(tsta,f*v(:,ii),'m');hold on;
%             end
%            
            
            print(figure(400),'-dpng','-r300',['D:\KD\BCI_data\Janelia_multiday\summary_plots\',str,'_test','.png'])
        end
    end
end
%%
clf
subplot(131);
ind = find(delta_cn~=0);
[a,b]=sort(delta_cn(ind));
ind = ind(b);
plot((delta_cn(ind))/110,'ko-');
hold on;
plot(xlim,[0 0],'k:');
xlabel('Session rank');ylabel('\Delta CN');
subplot(132);
plot(sparse(ind),'ko-');
xlabel('Session #');ylabel('% tile rank of CN');
subplot(133);
plot(off_manifold_frac(ind),'ko-');
xlabel('Session rank');ylabel('Fraction \Delta CN on manifold');
