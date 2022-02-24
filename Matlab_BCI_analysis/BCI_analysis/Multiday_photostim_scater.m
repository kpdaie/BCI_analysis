dat = new;
n = length(dat.roi);
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
t = 0:dt_si:dt_si*(size(S{1},1)-1);
tt = t;
stm = 8:13;
t(stm) = [];
ddd = DDD{1};
shuffle = 0;
try
    cl = new.conditioned_coordinates';
    clear dist
    for i = 1:length(dat.roi);
        dist(i) = sqrt(sum((dat.roi(i).centroid - cl).^2));
    end
    cn = dat.conditioned_neuron;
catch
    dist = 1:n;
    cn = 1;
end
for i = 1:size(S,1);
    for j = 1:size(S,2);
        a = S{i,j};
        a(stm,:,:) = [];
        a = interp1(t,a,tt,'linear');
        S{i,j} = a;
    end
end
for ci = 1:n;
    for j = 1:size(S,1);
        
        day1 = squeeze(S{j,1}(:,ci,:));
        day2 = squeeze(S{j,2}(:,ci,:));
        if shuffle == 1
            in = 1:size(day1,2);
            in1 = randsample(in,floor(length(in)/2));
            in2 = setdiff(in,in1);
            day2 = day1(:,in2);
            day1 = day1(:,in1);
        end
        
        
        
        bl = nanmean(nanmean(day2(1:7,:)));
        day2=(day2-bl)/bl;
        bl = nanmean(nanmean(day1(1:7,:)));
        day1=(day1-bl)/bl;
        mn1 = mean(mean(day2(15:20,:),2));
        mn2 = mean(mean(day1(15:20,:),2));
        mn = mn1-mn2;
        vr1 = var(mean(day2(15:20,:)));
        vr2 = var(mean(day1(15:20,:)));
        vr = sqrt(vr1/2+vr2/2);
        MN1(ci,j)=mn1;
        MN2(ci,j)=mn2;
        VR1(ci,j)=sqrt(vr1)/sqrt(size(day1,2));
        VR2(ci,j)=sqrt(vr2)/sqrt(size(day2,2));
        if ddd(ci,j) > 30 & size(day2,2)>5 & size(day1,2)>5;
            z(ci,j) = mn;
        else
            z(ci,j) = nan;
        end
    end
end
clf
% z=abs(z);
plot(dist,nanmean((z')),'o','color',[0 0 0]+.5);hold on;
plot(dist(cn),nanmean((z(cn,:)')),'o','color',[.7 .7 .7],...
    'Markerfacecolor','r');hold on;
fixed_bin_plots(dist,nanmean((z')),[0:50:500 1000],1,'k');hold on;
xlim([-50 800])
plot(xlim,xlim*0,'k:')
figure_finalize
%%
clf
bins = [0 30 50 80 100 150 200 300 500 1000];
for i = 1:length(bins)-1;
    ind = find(ddd>bins(i) & ddd<bins(i+1));
    % ind = find(ddd<30);
    if i < 5
        subplot(3,2,i);
        errorbar(MN1(ind),MN2(ind),VR1(ind),'k.','horizontal');hold on;
        errorbar(MN1(ind),MN2(ind),VR2(ind),'k.');hold on;
        plot(MN1(ind),MN2(ind),'ko','MarkerFaceColor','w','markersize',3);hold on;
        
        plot(xlim,xlim,'k:')
        xlabel('Day 1 (\DeltaF/F)');
        ylabel('Day 2 (\DeltaF/F)');
        title(['Dist = ',num2str(bins(i)),'--',num2str(bins(i+1))])
    end
    for iter = 1:100;
        a = MN1(ind);
        b = MN2(ind);k=find(isnan(b+a)==0);
        a=a(k);b=b(k);
        rind = randsample(1:length(a),length(a),1);
        a=a(rind);
        b=b(rind);
        [cc(i,iter),p(i)] = corr(a,b);
        [h,pp(i)] = ttest(a-b);
    end
end
subplot(3,1,3);
confidence_bounds_percentile(bins(1:end-1),cc,95,'k','k',.2)
hold on;
plot(bins(1:end-1),mean(cc'),'ko');
plot(xlim,xlim*0,'k:')
title('Corr(day1,day2)');
ylabel('Corr coeff');
xlabel(['Distance from nearest target (\mum)',char(10),dat.folder]);
figure_finalize

%%
clear far

bins = [0 30 80 120 150 200 300   1000];
for j = 1:size(S,1);
    for bi = 1:length(bins)-1
        for iter = 1:20
            iter
            % j = 7;
            day1 = squeeze(S{j,1});
            day2 = squeeze(S{j,2});
            
            in = 1:size(day1,3);
            in1 = randsample(in,floor(length(in)/2));
            in2 = setdiff(in,in1);
            day1_1 = day1(:,:,in2);
            day1_2 =day1(:,:,in1);
            day1_1 = day2(:,:,randsample(1:size(day2,3),size(day2,3),1));
            day1_2 = day1(:,:,randsample(1:size(day1,3),size(day1,3),1));
            near = find(ddd(:,j)<30);
            far = find(ddd(:,j)>bins(bi) & ddd(:,j)<bins(bi+1));
            a = nanmean(day1_1(:,far,:),3);a=a-repmat(mean(a(1:7,:)),size(a,1),1);
            b = nanmean(day1_2(:,far,:),3);b=b-repmat(mean(b(1:7,:)),size(b,1),1);
            [u,s,v]=svd(a(15:20,:));
            v(:,1) = v(:,1)*sign(sum(a*v(:,1)));
            % clf
            % plot(a*v(:,1))
            % hold on;
            % plot(b*v(:,1))
            f_far2(:,j,bi,iter) = b*v(:,1);
        end
    end
end
plot(mean(mean(f_far2,2),3))
