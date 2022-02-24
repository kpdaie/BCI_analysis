dat = load('F:\BCI2\AF1\110321\session_110321_analyzed_dat_small__slm22110421.mat')
%%
dat.currentPlane = 1;
file = [dat.folder,char(dat.siFiles{dat.currentPlane}{1})];
[hMroiRoiGroup hStimRoiGroups] = scanimage.util.readTiffRoiData(file);
header = scanimage.util.opentif(file);
seq = header.SI.hPhotostim.sequenceSelectedStimuli;
seq = repmat(seq,1,10);
seq = seq(header.SI.hPhotostim.sequencePosition:end);
seq = seq(1:length(dat.siFiles{dat.currentPlane})-1);
dat.stimGroup = hStimRoiGroups;
%%
clear D
figure(826);
figure_initialize
for si = 1:max(seq);
    clf
    set(gcf,'position',[3 3 3 3])
    ind = find(seq==si);
    files = dat.siFiles{dat.currentPlane}(1:end);
    clear pre post
    for i = 2:length(ind);
        file = [dat.folder,char(files{ind(i)})];
        [~,a] = scanimage.util.opentif(file);
        a = squeeze(a);
        post(:,:,:,i-1) = a(:,:,3:10);
        file = [dat.folder,char(files{ind(i)-1})];
        [~,a] = scanimage.util.opentif(file);
        a = squeeze(a);
        pre(:,:,i-1) = mean(a(:,:,4:end-2),3);
    end
    d = medfilt2(mean(mean(post(:,:,:,:),3),4)-mean(pre,3),[7 7]);
    d = sign(median(d(:)))*d/median(d(:));
    imagesc(d,[-1 1]*prctile(d(:),100));
    if si ~=9 & si ~=19
        imagesc(d,[-150 150]);
    else
        imagesc(d,[-1050 1050]);
    end
    cm = bluewhitered;
    colormap(cm);
    %     colorbar
    slm = hStimRoiGroups(si).rois(2).scanfields.slmPattern;
    sg = units_to_pixels(hStimRoiGroups(si).rois(2).scanfields,dat.siHeader,dat.dim);
    pix = sg.SLM_pix;
    hold on;
    scl = 2;
    for i = 1:size(pix,2);
        plot(pix(1,i),pix(2,i),'o','markersize',10,'color',[0 0 0]+0);
%         p1 = [pix(1,i)-65;pix(2,i)];
%         p2 = [pix(1,i)-90;pix(2,i)];
%         dp = p1 - p2;                         % Difference
%         qq = quiver(p1(1),p1(2),dp(1)*scl,dp(2)*1,0,'MaxHeadSize',10)
%         qq.LineWidth = 1;
%         qq.Color = 'k';
        %         set(qq,'AutoScale','on', 'AutoScaleFactor', 2)
    end
    
    set(gca,'xtick',[],'ytick',[])
    set(gca,'units','normalized','visible','off','position',[0 0 1 1]);
    D(:,:,si) = d;
    str = ['map',num2str(si)];
    axis square
    title([dat.folder,'\figures\_',str])
    drawnow
    print(gcf,'-dpng',[dat.folder,'\figures\_',str,'_1.png'])
end
% close all
%%
dat.currentPlane = 1;
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