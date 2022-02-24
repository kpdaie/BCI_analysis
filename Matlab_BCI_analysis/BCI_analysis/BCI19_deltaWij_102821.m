new = load('F:\BCI\BCI19\102521\session_102521_analyzed_dat_small_neuron1102521.mat');
old = load('F:\BCI\BCI19\102221\session_102221_analyzed_dat_small_slm22102521.mat');
%%
for session = 1:2;
    if session == 1;
        dat = old;
    else
        dat = new;
    end
    dat.currentPlane = 4;
    if session == 2;
        dat.currentPlane = 2;
    end
    file = [dat.folder,char(dat.siFiles{dat.currentPlane}{1})];
    [hMroiRoiGroup hStimRoiGroups] = scanimage.util.readTiffRoiData(file);
    header = scanimage.util.opentif(file);
    seq = header.SI.hPhotostim.sequenceSelectedStimuli;
    seq = repmat(seq,1,10);
    if session == 1
        seq = seq(header.SI.hPhotostim.sequencePosition:end);
    else
        seq = seq(header.SI.hPhotostim.sequencePosition+1:end);
    end
    seq = seq(1:length(dat.siFiles{dat.currentPlane})-1);
    dat.stimGroup = hStimRoiGroups;
    
    base = dat.bases{dat.currentPlane};
    inds = cell2mat(cellfun(@(x) ~isempty(strfind(x,base)),...
        dat.intensityFile,'uni',0));
    inds = (find(inds==1));
    clear Fstim
    for i = 1:length(dat.roi);
        for j = 1:length(inds)-1;
            if j > 1
                a = dat.roi(i).intensity{inds(j-1)}(end-10:end);
            else
                a = nan(11,1);
            end
            Fstim(:,i,j) = [a; dat.roi(i).intensity{inds(j)}(1:20)];
        end
    end
    Fs{session} = Fstim;
    SEQ{session} = seq;
    Gx = [];Gy = [];resp = [];x=[];y=[];
    clear ff ddd
    for si = 1:length(dat.stimGroup)
        ind = find(seq==si);
        slm = hStimRoiGroups(si).rois(2).scanfields.slmPattern;
        sg = units_to_pixels(hStimRoiGroups(si).rois(2).scanfields,dat.siHeader,dat.dim);
        pix = sg.SLM_pix;
        galvo = sg.centerXY_pix;
        clear XY distance
        for i = 1:length(dat.roi);
            XY(i,:) = dat.roi(i).centroid;
        end
        for cl = 1:length(dat.roi);
            minDist(cl) = min(sqrt(sum((bsxfun(@minus,pix,XY(cl,:)')).^2,1)));
            gDist(cl) = min(sqrt(sum((bsxfun(@minus,galvo,XY(cl,:)')).^2,1)));
        end
        clf
        f = nanmean(Fstim(:,:,ind),3);
        del = nanmean(f(16:21,:)) - nanmean(f(1:8,:));
        a = Fstim(:,:,ind);
        aft = squeeze(nanmean(a(15:21,:,:))-nanmean(a(1:8,:,:)));
        [h,p] = ttest(aft');
        P(:,si) = p;
        
        ind = find(minDist<25);
        Gx = [Gx gDist(ind)];
        Gy = [Gy del(ind)];
        x  = [x gDist];
        y  = [y del];
        resp = [resp f(:,ind)];
        %     f(10:15,:) = nan;
        delta_activity(:,:,si,session) = f;
        ddd(:,si) = minDist;
        dddd(:,si,session) = gDist;
        DDD{session} =ddd;
        %     pause;
    end
end
%%
clf
cl = new.conditioned_coordinates';
clear dist
for i = 1:length(dat.roi);
    dist(i) = sqrt(sum((dat.roi(i).centroid - cl).^2));
end
cn = dat.conditioned_neuron;
coupling=squeeze(mean(delta_activity(15:20,:,:,:))-mean(delta_activity(1:8,:,:,:)));
del = diff(coupling.*(cat(3,ddd,ddd)>30),[],3);
plot(dist,mean(del,2),'o','color',[.6 .6 .6]);hold on;
fixed_bin_plots(dist*500/333,mean(del,2),[0 50:100:800],1,'k');hold on;
plot(dist(cn)*500/333,mean(del(cn,:),2),'ko','markerfacecolor','r');
xlim([-40 800]);
plot(xlim,xlim*0,'k:');
figure_finalize
xlabel('Distance from CN (\mum)')
ylabel('\Delta Effective connection (\DeltaF/F)')
%%
ind =cell2mat(cellfun(@(x) ~isempty(strfind(x,'neuron')),old.bases,'uni',0))
len=cell2mat(cellfun(@(x) length(x),old.siFiles,'uni',0))
ind = find(ind.*len==max(ind.*len))
[dfo,disto,Fo,epoch,tsta,raw0] = BCI_dat_extract2(old,old.bases{ind});

ind =cell2mat(cellfun(@(x) ~isempty(strfind(x,'neuron')),new.bases,'uni',0))
len=cell2mat(cellfun(@(x) length(x),new.siFiles,'uni',0))
ind = find(ind.*len==max(ind.*len))
[dfn,distn,Fn,epoch,tsta,rawn] = BCI_dat_extract2(new,new.bases{ind});
%%
clf
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate
t = 0:dt_si:(size(Fn,1)-1)*dt_si;
t = t - t(40);
cn = new.conditioned_neuron;
fnew = nanmean(Fn,3);fnew = fnew-repmat(mean(fnew(1:30,:)),size(fnew,1),1);
fold = nanmean(Fo,3);fold = fold - repmat(mean(fold(1:30,:)),size(fold,1),1);
[a,b] = sort(sum(fold(40:200,:)));
subplot(231);
imagesc(fold(:,b)',[-.2 1]/2);
set(gca,'xtick',[40 240],'xticklabel',{'0','10'});
xlabel('Time from trial start')
ylabel('Neuron #')
subplot(232);
imagesc(fnew(:,b)',[-.2 1]/2);hold on;
set(gca,'xtick',[40 240],'xticklabel',{'0','10'});
xlabel('Time from trial start')
a=find(b==new.conditioned_neuron);
plot([220 240],[a a],'m');
subplot(233);
plot(t,fnew(:,cn));hold on;
plot(t,fold(:,cn))
xlabel('Time from trial start')
ylabel('CN activity (\DeltaF/F)');
ll=legend('Day 0','Day 1');
ll.Position = [.8 .69 .12 .1];


subplot(234);
dSel = mean(fnew)-mean(fold);
scatter(distn*.8,dSel,'ko');
hold on;
scatter(distn(cn)*.8,dSel(cn),'ko','markerfacecolor','r');
xlim([-40 900])
xlabel('Distance from CN (\mum)')
ylabel('Delta selectivity (\DeltaF/F)')

subplot(235);
dSel = mean(fnew)-mean(fold);
scatter(mean(fold),dSel,'ko');
hold on;
scatter(mean(fold(:,cn)),dSel(cn),'ko','markerfacecolor','r');
xlabel('Selectivity Day 0')
ylabel('Delta selectivity (\DeltaF/F)')


figure_finalize
%%
clf
cl = new.conditioned_coordinates';
clear dist
for i = 1:length(dat.roi);
    dist(i) = sqrt(sum((dat.roi(i).centroid - cl).^2));
end
cn = dat.conditioned_neuron;
coupling=squeeze(mean(delta_activity(15:20,:,:,:))-mean(delta_activity(1:8,:,:,:)));
del = diff(coupling.*(cat(3,ddd,ddd)>80),[],3);
plot(dSel,mean(del,2),'ko','color',[.6 .6 .6]*0,'markerfacecolor','w');hold on;

% mean_bin_plot(dSel,mean(del,2),6,1,1,'k');hold on;
plot(dSel(cn),mean(del(cn,:),2),'ko','markerfacecolor','r');
plot(xlim,xlim*0,'k:');
figure_finalize
xlabel('Delta selectivity (\DeltaF/F)')
ylabel('\Delta Effective connection (\DeltaF/F)')
%%
figure(12)
clf
[u,s,v] = svd(fold(100:end,:));
vcd = v(:,1);
vcd = vcd*(sign(sum(fold*vcd)));
% vcd = mean(fold(100:end,:))';
% vcd = mean(fold(end-20:end,:))';
for i = 1:size(ddd,2);
    vstim = (ddd(:,i)<20);
    stmCD(i) = vstim'*vcd;
    
    vstim = coupling(:,i,1).*(ddd(:,i)>30 & ddd(:,i)<7000);
    coupleCD(i) = vstim'*vcd;
end
% subplot(131);
% plot(fold*vcd);hold on;
% plot(fnew*vcd);hold on;
subplot(121)
scatter(stmCD,coupleCD,'k');hold on;
[cc,p]=corr(stmCD',coupleCD')
mean_bin_plot(stmCD,coupleCD,3);
xlabel('Stim vector * BCI vector');
ylabel('\Delta activity BCI vector (non. stim neurons)')
subplot(122)
ind = find(ddd(cn,:)>30);
scatter(stmCD(ind),del(cn,ind),'k');hold on;
mean_bin_plot(stmCD(ind),del(cn,ind),3);
xlabel('Stim vector * BCI vector');
ylabel('\Delta activity CN')
[cc,p]=corr(stmCD(ind)',del(cn,ind)')
figure_finalize
%%
figure(22);
clf
n = length(dat.roi);
t = 0:dt_si:dt_si*(size(delta_activity,1)-1);
tt = t;
stm = 8:13;
t(stm) = [];

for i = 1:n
    ind = find(DDD{1}(i,:)>20);
    day0 = nanmean(delta_activity(:,i,ind,1),3);
    ind = find(DDD{2}(i,:)>20);
    day1 = nanmean(delta_activity(:,i,ind,2),3);
    day0 = day0 - nanmean(day0(1:5,:));
    day1 = day1 - nanmean(day1(1:5,:));
    day0(stm,:) = [];
    day1(stm,:) = [];
    Day0(:,i) = interp1(t,day0,tt,'linear');
    Day1(:,i) = interp1(t,day1,tt,'linear');
end
clf
tt = tt - tt(stm(end));
plot(tt,medfilt1(Day0(:,cn),3),'k');hold on;
plot(tt,medfilt1(Day1(:,cn),3),'m');
plot(tt(stm),ones(size(stm))*-1,'r:')
xlabel('Time from photostim.')
ylabel('\DeltaF/F');
title('Connections onto CN')
box off
%%
figure(21);clf
del = (mean(Day1(16:end,:))-mean(Day0(16:end,:)))';
plot(dist,mean(del,2),'o','color',[.6 .6 .6]);hold on;
fixed_bin_plots(dist*500/333,mean(del,2),[0 50:100:800],1,'k');hold on;
plot(dist(cn)*500/333,mean(del(cn,:),2),'ko','markerfacecolor','r');
xlim([-40 800]);
plot(xlim,xlim*0,'k:');
figure_finalize
xlabel('Distance from CN (\mum)')
ylabel('\Delta Effective connection (\DeltaF/F)')


%
% for i = 1:N
%     [ccc(i),p]=corr(stmCD(ind)',del(i,ind)')
% end

%%
clf
figure(11)
cpl= (coupling(:,:,1).*(ddd>30));
cpl(:,:,2) = (coupling(:,:,2).*(ddd>30));
subplot(121);
imagesc(cpl(:,:,1),[-4 4]);
subplot(122);
imagesc(cpl(:,:,2),[-4 4]);
figure(21);
subplot(121);
a = coupling(:,:,1);b = coupling(:,:,2);
plot(a(:).*(ddd(:)<10),b(:).*(ddd(:)<10),'ko','MarkerSize',4,'markerfacecolor','w');
xlabel('Stim. amplitude day 0')
ylabel('Stim. amplitude day 1')
title('Stim. neurons (<10 \mum)')
hold on;
plot(xlim,xlim,'k:')
subplot(122);
a = coupling(:,:,1);b = coupling(:,:,2);
plot(a(:).*(ddd(:)>30),b(:).*(ddd(:)>30),'ko','markersize',4,'markerfacecolor','w');
xlabel('Effective connection (day 0)')
title('Non stim. neurons (>30 \mum)')
ylabel('Effective connection (day 1)')
figure_finalize
%%
n = length(dat.roi);
t = 0:dt_si:dt_si*(size(delta_activity,1)-1);
tt = t;
stm = 8:13;
t(stm) = [];

for i = 1:n
    ind = find(ddd(i,:)>80);
    day0 = mean(delta_activity(:,i,ind,1),3);
    day1 = mean(delta_activity(:,i,ind,2),3);
    day0 = day0 - mean(day0(1:5,:));
    day1 = day1 - mean(day1(1:5,:));
    day0(stm,:) = [];
    day1(stm,:) = [];
    Day0(:,i) = interp1(t,day0,tt,'linear');
    Day1(:,i) = interp1(t,day1,tt,'linear');
end
clf
tt = tt - tt(stm(end));
plot(tt,medfilt1(Day0(:,cn),3),'k');hold on;
plot(tt,medfilt1(Day1(:,cn),3),'m');
plot(tt(stm),ones(size(stm))*-1,'r:')
xlabel('Time from photostim.')
ylabel('\DeltaF/F');
title('Connections onto CN')
%%
n = length(dat.roi);
t = 0:dt_si:dt_si*(size(delta_activity,1)-1);
tt = t;
stm = 8:13;
t(stm) = [];
clf
q = ddd;
q = q*500/333;
lab{1} = 'dist < 10\mum';
lab{2} = '30\mum < dist < 100\mum';
lab{3} = '100\mum < dist ';

for j=1:3;
    for i = 1:n
        if j == 1;
            ind = find(q(i,:)<20);
        elseif j == 2;
            ind = find(q(i,:)>20 & q(i,:)<100);
        elseif j == 3;
            ind = find(q(i,:)>100 & q(i,:)<99900);
        end
        day0 = mean(delta_activity(:,i,ind,1),3);
        day1 = mean(delta_activity(:,i,ind,2),3);
        day0 = day0 - mean(day0(1:5,:));
        day1 = day1 - mean(day1(1:5,:));
        day0(stm,:) = [];
        day1(stm,:) = [];
        Day0(:,i) = interp1(t,day0,tt,'linear');
        Day1(:,i) = interp1(t,day1,tt,'linear');
    end
    subplot(1,3,j)
    confidence_bounds(tt-tt(stm(end)),Day0,[],'k','k',.2);hold on;
    plot(xlim,xlim*0,'k:')
%     plot(tt,nanmean(Day0,2),'k');hold on;
    plot(tt(stm),ones(size(stm))*-1/2,'r:');
    xlabel('Time from photostim end (s)');
    title(lab{j});
end
%%
amp = squeeze(mean(Fstim(14:20,:,:))-mean(Fstim(2:8,:,:)));
clear E L se sl
for si = 1:max(seq);
    ind = find(seq==si);
    m = floor(length(ind)/2);
    E(:,si) = nanmean(amp(:,ind(1:2:end))');
    L(:,si) = nanmean(amp(:,ind(2:2:end))');    
    
    cls = find(ddd(:,si)<20 & ddd(:,si)<20);
    se{si} = nanmean(amp(cls,ind(1:2:end))');
    sl{si} = nanmean(amp(cls,ind(2:2:end))');    
end
clf
plot([se{:}],[sl{:}],'k.')
hold on;
plot(xlim,xlim,'k:');
corr([se{:}]',[sl{:}]')



