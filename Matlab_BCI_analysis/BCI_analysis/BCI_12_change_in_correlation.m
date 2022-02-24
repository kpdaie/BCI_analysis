[df,dist,F,epoch,tsta,raw,df_all] = BCI_dat_extract(dat);
%%
clf
[a,b] = sort(dist);
cpre = corr(df_all{1}(:,b));
cpost = corr(df_all{3}(:,b));
dcorr = nansum(cpost) - nansum(cpre);
[~,~,~,L] = fixed_bin_plots(dist(b),dcorr,[0 10 50 100 120 200 300 400 600],1,'k');hold on;
xlim([-20 500]);
ylim([-5 1.5]);
plot(xlim,xlim*0,'k')
box off
xlabel('Distance from CN (\mum)')
ylabel('Summed change in correlation')

%%
figure(1);
[cpre,p] = corr(df_all{3});
colors = jet;
cn = dat.conditioned_neuron;
crs = cpre(cn,:);
p   = p(cn,:);
rng = prctile(crs,[5 95]);
crs(crs<rng(1)) = rng(1);
crs(crs>rng(2)) = rng(2);
[a,b] = sort(crs);
clf
% imagesc(dat.IM,[0 50]);
% colormap(gray);hold on;
IM = 0*dat.IM;
x = [dat.roi.centroid];
y = x(2:2:end);
x = x(1:2:end);
for i = 1:length(x);
%     if p(i) < 0.01;
%         plot(x(i),y(i),'o','color',colors(i,:),'markerfacecolor',colors(i,:));hold on;
        pix = dat.roi(i).pixelList;
        IM(pix) = crs(i);
%     end
end
imagesc(IM);hold on;
cm = bluewhitered;
colormap(cm)
colorbar
plot(x(cn),y(cn),'mo','markersize',20,'linewidth',2);
box off
set(gca,'visible','off');
%%
[cpre,p] = corr(df_all{1});
[cpost,p] = corr(df_all{3});
crs = cpost(cn,:) - cpre(cn,:);
rng = prctile(crs,[5 95]);
crs(crs<rng(1)) = rng(1);
crs(crs>rng(2)) = rng(2);
for i = 1:length(x);
%     if p(i) < 0.01;
%         plot(x(i),y(i),'o','color',colors(i,:),'markerfacecolor',colors(i,:));hold on;
        pix = dat.roi(i).pixelList;
        IM(pix) = crs(i);
%     end
end
imagesc(IM);hold on;
cm = bluewhitered;
colormap(cm)
colorbar
plot(x(cn),y(cn),'mo','markersize',20,'linewidth',2);
box off
set(gca,'visible','off');

%%
clf
crs = cpost(cn,:) - cpre(cn,:);
p = nanmean(F(1:240,:,7:end-10),3);
p = p - repmat(mean(p(1:20,:)),size(p,1),1);
amp = mean(p(40:150,:));
subplot(121);
[a,b] = sort(amp);
imagesc(p(:,b)',[-1 1]/3);
colormap(parula)
ylabel('Neuron #');
set(gca,'xtick',[40 240],'xticklabel',{'0','10'})
xlabel('Time from trial start (s)')
colorbar


subplot(122);
mean_bin_plot(amp,crs,4)
xlabel(['Responses at trial start'])
ylabel('\Delta CN correlation')
figure_finalize
%%
clf
IM = 0*dat.IM;
thr = .1;
for i = 1:length(x);
%     if p(i) < 0.01;
%         plot(x(i),y(i),'o','color',colors(i,:),'markerfacecolor',colors(i,:));hold on;
        pix = dat.roi(i).pixelList;
        a = amp(i);
        a(a <thr & a>-thr) = 0;       
        IM(pix) = a;        
%     end
end
imagesc(IM);hold on;
cm = bluewhitered;
colormap(cm)
colorbar
hold on;
plot(x(cn),y(cn),'mo','markersize',20,'linewidth',2);
box off
set(gca,'visible','off');