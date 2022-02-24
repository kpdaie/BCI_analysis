function [x,y,id,GRP,e] = effective_connections_vs_space(dat,slmPln,shift);
%%
if nargin == 1;
    isslm = cell2mat(cellfun(@(x) ~isempty(strfind(x,'slm')),dat.bases,'uni',0));
    len = cell2mat(cellfun(@(x) length(x),dat.siFiles,'uni',0));
    [~,slmPln] = max(len.*isslm);
end
file = [dat.folder,char(dat.siFiles{slmPln}{1})];
[hMroiRoiGroup hStimRoiGroups] = scanimage.util.readTiffRoiData(file);
header = scanimage.util.opentif(file);
seq = header.SI.hPhotostim.sequenceSelectedStimuli;
seq = repmat(seq,1,10);
seq = seq(header.SI.hPhotostim.sequencePosition+shift:end);
seq = seq(1:length(dat.siFiles{slmPln})-1);
dat.stimGroup = hStimRoiGroups;
%%
base = dat.bases{slmPln};
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
    a = squeeze(Fstim(:,i,:));
    bl = prctile(a(:),20);
%     Fstim(:,i,:) = (Fstim(:,i,:)-bl)/bl;
end
seq = seq(1:size(Fstim,3));
%%
figure(826+slmPln);
Gx = [];Gy = [];resp = [];x=[];y=[];id = [];GRP = [];e = [];
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
    del = (nanmean(f(15:20,:)) - nanmean(f(1:7,:)))./nanmean(f(1:7,:));
    err = nanstd(f(1:7,:));
    a = Fstim(:,:,ind);
    stm{si} = a;
    aft = squeeze(nanmean(a(15:21,:,:))-nanmean(a(1:2,:,:)));
    [h,p] = ttest(aft');
    P(:,si) = p;
    ind = find(minDist<25);
    Gx = [Gx gDist(ind)];
    Gy = [Gy del(ind)];
    x  = [x gDist];
    y  = [y del];
    e  = [e err];
    id = [id 1:length(dat.roi)];
    GRP = [GRP ones(1,length(del))*si];
    resp = [resp f(:,ind)];
    ff(:,:,si) = f;
    ddd(:,si) = minDist;
    dddd(:,si) = gDist;
    DEL(:,si) = del;
end
good = find(isnan(Gy)==0);
subplot(122)
mean_bin_plot(Gx(good)*1200/800,Gy(good),8,1,1,'k');
plot(xlim,xlim*0,'k:')
xlabel('Distance from SLM center (\mum)');
ylabel('\Delta activity')
title('Target neurons')
% ylim([-1 8])
subplot(121);
good = find(isnan(y)==0);
fixed_bin_plots(x(good)*1200/800,y(good),[0  20 30 50:50:800],1,'k');hold on;
xlabel('Distance from nearest target (\mum)');
plot(xlim,xlim*0,'k:')
title('All neurons')
plot([30 30],ylim,'k:')
figure_finalize
