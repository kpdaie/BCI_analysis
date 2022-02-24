clf
gi = 15
KDsubplot(4,4,[2 3],.6);
ind=find(g1==gi);
scatter(x1(ind),y1(ind)./e1(ind),'k');hold on;

ind=find(g1==gi & id1==cn);
scatter(x1(ind),y1(ind)./e1(ind),'k','MarkerFaceColor','r');
xlim([-40 700])

KDsubplot(4,4,[4 3],.6);
ind=find(g2==gi);
scatter(x2(ind),y2(ind)./e2(ind),'k');hold on;

ind=find(g2==gi & id2==cn);
scatter(x2(ind),y2(ind)./e2(ind),'k','MarkerFaceColor','r');
xlim([-40 700])

ind=find(g1==gi);

for si = 1:2;
    KDsubplot(2,2,[si 1],.3);
%     XY = [dat.roi.centroid];
    clear XY
    for ii = 1:length(dat.roi);
        try
            XY(1,ii) = dat.roi(ii).centroid(1);
            XY(2,ii) = dat.roi(ii).centroid(2);
        catch
            XY(1:2,ii) = 0;
        end
    end
    y = y1;err = e1;x = x1;
    if si == 2;
        y = y2;
        err = e2;
        x = x2;
    end
%     XY = [XY(1:2:end);XY(2:2:end)];
    mask = 0*dat.IM;
    for i = 1:n;
        pixels = dat.roi(i).pixelList;
        mask(pixels) = y(ind(i))/err(ind(i));
    end
    imagesc(mask,[-8 8]);
    colormap(bluewhitered)
    cb = colorbar;
    a = get(cb,'position');
    a(4) = a(4)/5;
    set(cb,'position',a);
    ylabel('zscore')
    slm = hStimRoiGroups(si).rois(2).scanfields.slmPattern;
    sg = units_to_pixels(hStimRoiGroups(si).rois(2).scanfields,dat.siHeader,dat.dim);
    pix = sg.SLM_pix;
    hold on;
    scl = 2;
    for i = 1:size(pix,2);
        plot(pix(1,i),pix(2,i),'o','markersize',15,'color',[0 0 0]+0);
    end
    plot(XY(1,cn),XY(2,cn),'ro','markersize',20,'linewidth',2);
    set(gca,'xtick',[],'ytick',[]);
end

for si = 1:2
    KDsubplot(4,8,[1+(si-1)*2 5],.6);
    n = length(dat.roi);
    dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate
    t = 0:dt_si:dt_si*(size(delta_activity,1)-1);
    tt = t;
    stm = 8:13;
    t(stm) = [];
    ind = find(SEQ{si}==gi);
    a = squeeze(Fs{si}(:,cn,ind));
    stm = 8:13;
    a(stm,:) = [];
    a = interp1(t,a,tt,'linear');
    a = a-repmat(mean(a(1:7,:)),size(a,1),1);
    confidence_bounds(tt,a,[],'k','k',.2)
    ylim([-30 30]);
    hold on;
    plot(xlim,xlim*0,'k:')
    
    KDsubplot(4,8,[1+(si-1)*2 6],.6);
    cls = find(DDD{si}(:,gi)<40);
    confidence_bounds(tt,da(:,cls,gi,si),[],'k','k',.2);
    
   c
end
