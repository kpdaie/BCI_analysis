dat.currentPlane = 3;
file = [dat.folder,char(dat.siFiles{dat.currentPlane}{1})];
[hMroiRoiGroup hStimRoiGroups] = scanimage.util.readTiffRoiData(file);
header = scanimage.util.opentif(file);
seq = header.SI.hPhotostim.sequenceSelectedStimuli;
seq = repmat(seq,1,10);
seq = seq(header.SI.hPhotostim.sequencePosition:end);
seq = seq(1:length(dat.siFiles{dat.currentPlane}));
%%
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
    f = nanmean(F(31:81,:,ind),3);
    del = mean(f(16:21,:)) - mean(f(4:9,:));
    a = F(31:81,:,ind);
    aft = squeeze(nanmean(a(20:50,:,:))-nanmean(a(1:9,:,:)));
    [h,p] = ttest(aft');
    P(:,si) = p;
    %     subplot(211);
    %     fixed_bin_plots(minDist,del,[0 10 50 100 200 500 10000],1,'k');
    %     subplot(212);
    %     fixed_bin_plots(gDist,del,[0 10 50 100 200 500 10000],1,'k');
    ind = find(minDist<25);
    Gx = [Gx gDist(ind)];
    Gy = [Gy del(ind)];
    x  = [x gDist];
    y  = [y del];
    resp = [resp f(:,ind)];
    f(10:15,:) = nan;
    ff(:,:,si) = f;
    ddd(:,si) = minDist;
    dddd(:,si) = gDist;
    %     pause;
end
subplot(121)
mean_bin_plot(Gx/.7,Gy,8,1,1,'k');
plot(xlim,xlim*0,'k:')
xlabel('Distance from SLM center (\mum)');
ylabel('\Delta activity')
title('Target neurons')
subplot(122);
mean_bin_plot(x/.7,y,8,1,1,'k');
xlabel('Distance from nearest target (\mum)');
plot(xlim,xlim*0,'k:')
title('All neurons')
figure_finalize

%%
clf
for i = 1:length(dat.roi);
    ind = find(ddd(i,:)>30);
    a = mean(ff(:,i,ind),3);
    del(i) = mean(a(20:40))-mean(a(1:9));
    numIn(i) = sum(P(i,ind)<.1);
end
subplot(121);

subplot(122);
scatter(dist,del,'k')
hold on;
scatter(dist(cn),del(cn),'k','markerfacecolor','r');
% clf
% scatter(dist,numIn)
xlabel('Distance from CN');
ylabel('Summed effective connection inputs (\DeltaF/F)');
xlim([-20 1000])
%%
    
    
    
    
    
    
