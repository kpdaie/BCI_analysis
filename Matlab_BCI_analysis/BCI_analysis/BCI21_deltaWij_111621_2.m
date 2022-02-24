

dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate
t = 0:dt_si:dt_si*(size(delta_activity,1)-1);
tt = t;
stm = 8:13;
t(stm) = [];

nt = size(delta_activity,3);
nc = size(delta_activity,2);
for si = 1:2
    for ci = 1:nc;
        for ti = 1:nt;
            a = delta_activity(:,ci,ti,si);
            a = a - mean(a(1:8));
            a(stm) = [];
            a = interp1(t,a,tt,'linear');
            da(:,ci,ti,si) = a;
        end
    end
end
for ci = 1:nc;
    ind = find(DDD{1}(ci,:)>30);
    a = squeeze(da(:,ci,ind,:));
    del_abs(:,ci) = nanmean(abs(a(:,:,2)-a(:,:,1)),2);
end

del_abs = del_abs - repmat(mean(del_abs(1:8,:)),size(del_abs,1),1);
%%
for i = 1:length(dat.roi);
    ind = find(DDD{1}(i,:)>30);
    a = diff(squeeze(da(:,i,ind,:)),[],3);
    [u,s,v] = svd(a);
    amp(i) = abs(mean(a*v(:,1)));
end
%%
for i = 1:100;
    subplot(211)
    ind=find(g1==i);
    scatter(x1(ind),y1(ind),'k');hold on;
    
    ind=find(g1==i & id1==cn);
    scatter(x1(ind),y1(ind),'k','MarkerFaceColor','r');
    
    subplot(212)
    ind=find(g2==i);
    scatter(x2(ind),y2(ind),'k');hold on;
    
    ind=find(g2==i & id2==cn);
    scatter(x2(ind),y2(ind),'k','MarkerFaceColor','r');
    
    pause;clf;
end
%%



