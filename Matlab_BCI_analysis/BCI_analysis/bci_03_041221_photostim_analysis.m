
folder = 'D:\KD\BCI_03\04\';
files = dir(folder);
files = {files.name};
stmFiles = files(cell2mat(cellfun(@(x) ~isempty(strfind(x,'.stim')),files,'uni',0)))';
intFiles = files(cell2mat(cellfun(@(x) ~isempty(strfind(x,'.csv')),files,'uni',0)))';
img = 'cell4_1mhz_2per_00001.tif';
[header,~]=scanimage.util.opentif([folder,img]);
clf
I = 1;
for i = [2 7 8];
    
    out = scanimage.util.readPhotostimMonitorFile([folder,stmFiles{i}]);
    rois = readtable([folder,'\',intFiles{i}]);
    f = [rois{:,3}];
    if length(out.Beam) > 100 && length(f) > 200
        KDsubplot(3,1,I,[.4 .6]);
        stm = find(diff(medfilt1(out.Beam(:,1),21)) > .01);
        stm(diff(stm)<100) = [];
        
        stm = ceil(stm*length(f)/length(out.Beam));
        
        offset = [-10:50];
        stm = repmat(stm',length(offset),1) + repmat(offset',1,length(stm));
        f = f(stm);
        dt_si = 1/header.SI.hRoiManager.scanVolumeRate;
        t = 0:dt_si:dt_si*(length(f)-1);
%         plot(t,nanmean(f'));
        try
        confidence_bounds(t,f,[],'k','k',.2);
        end
        title(stmFiles{i})
        I = I + 1;
        axis tight
    end
end
%%
img = 'cell4_1mhz_2per_00001.tif';
[header,im]=scanimage.util.opentif([folder,img]);
[m,n,hig] = scanimage.util.readTiffRoiData([folder,img]);
im = squeeze(im);
out = scanimage.util.readPhotostimMonitorFile([folder,stmFiles{7}]);
stm = find(diff(medfilt1(out.Beam(:,1),21)) > .01);
stm(diff(stm)<100) = [];
stm = ceil(stm*size(im,3)/length(out.Beam));
offset = [-10:50];
stm = repmat(stm',length(offset),1) + repmat(offset',1,length(stm));
stm(stm<1) = 1;
stm(stm>size(im,3)) = size(im,3);
for i = 1:size(stm,2)
    A(:,:,:,i) = im(:,:,stm(:,i));
end
dMap = mean(A,4);dMap = mean(dMap(:,:,14:20),3) - mean(dMap(:,:,1:6),3);
subplot(211);cla
imagesc(medfilt2(dMap,[5 5]),[0 25]);
title(img);
hold on;
sg = units_to_pixels(n.rois(2).scanfields,header,[size(im,1),size(im,2)])
plot(sg.centerXY_pix(1),sg.centerXY_pix(2),'ms','markersize',20)
%%
img = 'cell5_1mhz_2per_00001.tif';
[header,im]=scanimage.util.opentif([folder,img]);
[m,n,hig] = scanimage.util.readTiffRoiData([folder,img]);
im = squeeze(im);
out = scanimage.util.readPhotostimMonitorFile([folder,stmFiles{8}]);
stm = find(diff(medfilt1(out.Beam(:,1),21)) > .01);
stm(diff(stm)<100) = [];
stm = ceil(stm*size(im,3)/length(out.Beam));
offset = [-10:50];
stm = repmat(stm',length(offset),1) + repmat(offset',1,length(stm));
stm(stm<1) = 1;
stm(stm>size(im,3)) = size(im,3);
for i = 1:size(stm,2)
    A(:,:,:,i) = im(:,:,stm(:,i));
end
dMap = mean(A,4);dMap = mean(dMap(:,:,14:20),3) - mean(dMap(:,:,1:6),3);
subplot(212);cla
imagesc(medfilt2(dMap,[5 5]),[0 25]);
title(img);
hold on;
sg = units_to_pixels(n.rois(2).scanfields,header,[size(im,1),size(im,2)])
plot(sg.centerXY_pix(1),sg.centerXY_pix(2),'ms','markersize',20)
%%
img = 'cell120percent_00001.tif';
[header,im]=scanimage.util.opentif([folder,img]);
[m,n,hig] = scanimage.util.readTiffRoiData([folder,img]);
im = squeeze(im);
out = scanimage.util.readPhotostimMonitorFile([folder,stmFiles{2}]);
stm = find(diff(medfilt1(out.Beam(:,1),21)) > .01);
stm(diff(stm)<100) = [];
stm = ceil(stm*size(im,3)/length(out.Beam));
offset = [-10:50];
stm = repmat(stm',length(offset),1) + repmat(offset',1,length(stm));
stm(stm<1) = 1;
stm(stm>size(im,3)) = size(im,3);
for i = 1:size(stm,2)
    A(:,:,:,i) = im(:,:,stm(:,i));
end
dMap = mean(A,4);dMap = mean(dMap(:,:,14:20),3) - mean(dMap(:,:,1:6),3);
clf
imagesc(medfilt2(dMap,[5 5]),[0 25]);
title(img);
hold on;
sg = units_to_pixels(n.rois(2).scanfields,header,[size(im,1),size(im,2)])
plot(sg.centerXY_pix(1),sg.centerXY_pix(2),'ms','markersize',20)