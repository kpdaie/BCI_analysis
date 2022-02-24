addpath(genpath('X:\MATLAB\photostim\photostim_newRig\'));addpath(genpath('X:\MATLAB\BCI\'));
addpath(('X:\MATLAB\photostim\photostim_groups'))
if ~exist('dat')
    try
        photostim_group_old(hSI.hScan2D.logFilePath)
        % ok
    catch
        folder = input('folder?')
        photostim_group_old(folder)
    end
end
global dat
earlyTrials = 10:11;
lateTrials = 80:100;
frames = 20:30;
L = cell2mat(cellfun(@(x) length(x),dat.siFiles,'uni',0));
if ~isfield(dat,'currentPlane');
    dat.currentPlane = find(L==max(L));
end
str = dat.bases{dat.currentPlane};
register = 1;
if ~isfield(dat,'siHeader')
    [dat.siHeader,~] = ...
        scanimage.util.opentif([dat.folder,dat.siFiles{dat.currentPlane}{11}]);
end
pre = template_image_maker2(earlyTrials,dat.currentPlane,register,0,frames,0,0);
post = template_image_maker2(lateTrials,dat.currentPlane,register,0,frames,0,0);
BL = mean(pre,3);
DF = mean(post,3);
DF=shiftIm(DF,BL,1);
%%
figure(479);
clf
dMap = ((DF-BL)./BL).*(BL>prctile(BL(:),80));
imagesc(dMap)
hold on;
subplot(221);
imagesc(BL);hold on;
cn_name = dat.siHeader.SI.hIntegrationRoiManager.outputChannelsRoiNames{2};
[~,~,hig] = scanimage.util.readTiffRoiData([dat.folder,dat.siFiles{dat.currentPlane}{11}]);
names = {hig.rois.name}';
cn = cell2mat(cellfun(@(x) ~isempty(strfind(x,char(cn_name))),names,'uni',0));
cn = find(cn==1); %this gave me an error!!
cn = cn(1);
sg = units_to_pixels(hig.rois(cn).scanfields,dat.siHeader,dat.dim);
sg.centerXY_pix
dat.conditioned_coordinates = sg.centerXY_pix;
% dat.conditioned_coordinates = manual_conditioned_neuron_coords(dat,cn)
plot(dat.conditioned_coordinates(1),dat.conditioned_coordinates(2),'ro',...
    'markersize',20,'linewidth',2);
% set(gca,'colormap',gray)
subplot(222);
imagesc(medfilt2(dMap,[5 5]));colorbar
title('Late - Early');
hold on;
plot(dat.conditioned_coordinates(1),dat.conditioned_coordinates(2),'bo',...
    'markersize',20,'linewidth',2);
% set(gca,'colormap',jet)

roi_dat = dir(dat.folder);roi_dat = {roi_dat.name};
ind = cellfun(@(x) ~isempty(strfind(x,'.csv')),roi_dat,'uni',0);
roi_dat = roi_dat(cell2mat(ind));
ind = cellfun(@(x) ~isempty(strfind(x,dat.bases{dat.currentPlane}...
    )),roi_dat,'uni',0);
roi_dat = roi_dat(cell2mat(ind));
roi_dat = roi_dat(3);
rois = readtable([dat.folder,'\',char(roi_dat)]);

subplot(212);
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
ts = 0:dt_si:dt_si*(length(rois{:,1})-1);
plot(ts/60,rois{:,cn+2},'k');
title([dat.folder,dat.bases{dat.currentPlane}])
xlabel('Time (min.)');
summary = [dat.folder,dat.siFiles{dat.currentPlane}{1}(1:end-4),'summary','.jpg']
print(figure(479),'-djpeg',summary);

% plot(dat.conditioned_coordinates(1),dat.conditioned_coordinates(2),'ro',...
%     'markersize',20);
% IM(:,:,1) = dMap*1;
% IM(:,:,2) = dMap*1;
% IM(:,:,3) = 0;
% imshow(BL/300);hold on;
% h = imshow(IM+repmat(BL/1500,1,1,3));
% set(h,'AlphaData',.5);