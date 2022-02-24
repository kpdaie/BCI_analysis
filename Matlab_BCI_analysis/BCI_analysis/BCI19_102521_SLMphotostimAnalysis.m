dat.currentPlane = 4;
file = [dat.folder,char(dat.siFiles{dat.currentPlane}{1})];
[hMroiRoiGroup hStimRoiGroups] = scanimage.util.readTiffRoiData(file);
header = scanimage.util.opentif(file);
seq = header.SI.hPhotostim.sequenceSelectedStimuli;
seq = repmat(seq,1,10);
seq = seq(header.SI.hPhotostim.sequencePosition:end);
seq = seq(1:length(dat.siFiles{dat.currentPlane})-1);
dat.stimGroup = hStimRoiGroups;
%%
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
%%
figure(826);
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
    del = nanmean(f(14:16,:)) - nanmean(f(5:8,:));
    a = Fstim(:,:,ind);
    aft = squeeze(nanmean(a(15:21,:,:))-nanmean(a(1:2,:,:)));
    [h,p] = ttest(aft');
    P(:,si) = p;
    ind = find(minDist<25);
    Gx = [Gx gDist(ind)];
    Gy = [Gy del(ind)];
    x  = [x gDist];
    y  = [y del];
    resp = [resp f(:,ind)];
    ff(:,:,si) = f;
    ddd(:,si) = minDist;
    dddd(:,si) = gDist;
    DEL(:,si) = del;
end
subplot(122)
mean_bin_plot(Gx*1200/800,Gy,8,1,1,'k');
plot(xlim,xlim*0,'k:')
xlabel('Distance from SLM center (\mum)');
ylabel('\Delta activity')
title('Target neurons')
ylim([-1 8])
subplot(121);
fixed_bin_plots(x*1200/800,y,[0  20 30 50:50:800],1,'k');hold on;
xlabel('Distance from nearest target (\mum)');
plot(xlim,xlim*0,'k:')
title('All neurons')
plot([30 30],ylim,'k:')
figure_finalize

%%
clf
cn = dat.conditioned_neuron;
for i = 1:length(dat.roi);
    ind = find(ddd(i,:)>30);
    a = mean(ff(:,i,ind),3);
    del(i) = mean(a(16:20))-mean(a(1:7));
    numIn(i) = sum(P(i,ind)<.1);
end
clf
% subplot(121);

% subplot(122);
scatter(dist,del,'k')
hold on;
scatter(dist(cn),del(cn),'k','markerfacecolor','r');
% clf
% scatter(dist,numIn)
xlabel('Distance from CN');
ylabel('Summed effective connection inputs (\DeltaF/F)');
xlim([-20 1000])
%%
    
    
    
    
    
    
