global dat
[~,dat.currentPlane] = max(cell2mat(cellfun(@(x) length(x),dat.siFiles,'uni',0)));
out = scanimage.util.readPhotostimMonitorFile('D:\KD\test\test\file23_00001.stim');

base = dat.bases{dat.currentPlane};
files = dir(dat.folder);files = {files.name};
gotBase = cell2mat(cellfun(@(x) ~isempty(strfind(x,base)),files,'uni',0));
gotStim = cell2mat(cellfun(@(x) ~isempty(strfind(x,'.stim')),files,'uni',0));
gotRoi = cell2mat(cellfun(@(x) ~isempty(strfind(x,'.csv')),files,'uni',0));
stmInd = find(gotBase.*gotStim==1);
stmInd = stmInd(end);
roiInd = find(gotBase.*gotRoi==1);
roiInd = roiInd(end);

bitcode = scanimage.util.readPhotostimMonitorFile([dat.folder,'\',files{stmInd}]);
rois    = readtable([dat.folder,'\',files{roiInd}]);
if ~isfield(dat,'siHeader')
    [dat.siHeader,~] = ...
        scanimage.util.opentif([dat.folder,dat.siFiles{dat.currentPlane}{11}]);
end

cn_name = dat.siHeader.SI.hIntegrationRoiManager.outputChannelsRoiNames{2};
[~,~,hig] = scanimage.util.readTiffRoiData([dat.folder,dat.siFiles{dat.currentPlane}{1}]);
names = {hig.rois.name}';
cn = cell2mat(cellfun(@(x) ~isempty(strfind(x,char(cn_name))),names,'uni',0));
cn = find(cn==1);
%%
subplot(211);