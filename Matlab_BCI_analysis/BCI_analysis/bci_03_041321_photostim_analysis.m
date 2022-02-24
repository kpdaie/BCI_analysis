folder = dat.folder;
files = dir(folder);
files = {files.name};
stmFiles = files(cell2mat(cellfun(@(x) ~isempty(strfind(x,'.stim')),files,'uni',0)))';
intFiles = files(cell2mat(cellfun(@(x) ~isempty(strfind(x,'.csv')),files,'uni',0)))';
stmFile = stmFiles(cell2mat(cellfun(@(x) ~isempty(strfind(x,dat.bases(dat.currentPlane))),stmFiles,'uni',0)));
beam = [];
for i = 1:length(stmFile)
    out = scanimage.util.readPhotostimMonitorFile([dat.folder,stmFile{i}]);
    beam = [beam;out.Beam];
end
dt_ps = 1/dat.siHeader.SI.hPhotostim.monitoringSampleRate;
t_ps = 0:dt_ps:dt_ps*(length(beam)-1);

dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
fluor_fun = @(x) cell2mat(cellfun(@(x) x',x,'uni',0));
f = fluor_fun(dat.roi(1).intensity);
t_si = 0:dt_si:dt_si*(length(f));
%%
stm = find(diff(beam)>1);
stm(diff(stm)<20) = [];

stm = ceil(stm * dt_ps/dt_si);
seq = dat.siHeader.SI.hPhotostim.sequenceSelectedStimuli(dat.siHeader.SI.hPhotostim.sequencePosition+0:end);
i = 0;
while i == 0;
    if length(seq) < length(stm);
        seq = [seq seq];
    else
        i = 1;
    end
end
clear aa
offset = -45:125;
for i = 1:length(dat.roi);
    f = fluor_fun(dat.roi(i).intensity);
    ind = find(seq == i);
    ind(ind>length(stm)) = [];
    s = stm(ind);
    s(s>length(f)) = [];
    s = repmat(s',length(offset),1) + repmat(offset',1,length(s));
    s(s<1) = 1;
    s(s>length(f)) = length(f);
    ff{i} = f(s);
    F(:,i) = f;
end
for i = 1:30;a = mean(ff{i}');aa(:,i)=(a-mean(a(1:10)))/mean(a(1:10));end
plot(mean(aa'))