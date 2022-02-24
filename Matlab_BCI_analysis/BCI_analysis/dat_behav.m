dat = load('D:\bScope\BCI_03\121620\121620\session_121620neuron2_analyzed_dat_small121820.mat');
behav = load('C:\Users\labadmin\Downloads\121620-bpod_zaber.mat');

% behav=load('C:\Users\labadmin\Downloads\121720-bpod_zaber.mat');
% dat = load('D:\bScope\BCI_03\121720\121720\session_121720neuron1_1__analyzed_dat_small121820.mat')
%%
cn = dat.conditioned_neuron;
for i = 1:length(dat.roi);
    a = cell2mat(cellfun(@(x) x',dat.roi(i).intensity,'uni',0));
    bl = prctile(a,20);
    a = (a-bl)/bl;
    F(:,i) = a;
end
f = F(:,cn);
tn = [];
for i = 1:length(dat.roi(1).intensity);
    tn = [tn ones(1,length(dat.roi(1).intensity{i}))*i];
end
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
t = 0:dt_si:dt_si*(length(f)-1);
try
    siNum = cell2mat(cellfun(@(x) str2num(x(max(find(x=='_'))+1:end-4)),b.scanimage_file_names,'uni',0));
catch
    siNum = cell2mat(cellfun(@(x) str2num(x(max(find(x=='_'))+1:end-4)),cellstr(squeeze(behav.scanimage_file_names)),'uni',0));
end
strt = cell2mat(cellfun(@(x) datenum(x)*(24*60*60),cellstr(behav.trial_start_times),'uni',0));
strt = strt-strt(1);
go   = behav.go_cue_times;
go   = strt + go*0;
rew  = cell2mat(behav.time_to_hit)';
rew = rew+strt;
rew(isnan(rew)) = [];

rewInd = (arrayfun(@(x) min(find(x<t)),rew,'uni',0));
a = cell2mat(cellfun(@(x) ~isempty(x),rewInd,'uni',0));
rewInd = cell2mat(rewInd(a));
goInd = (arrayfun(@(x) min(find(x<t)),go,'uni',0));
ind = cell2mat(cellfun(@(x) isempty(x),goInd,'uni',0));
goInd(ind) = [];goInd = cell2mat(goInd);
offset = -100:100;
rewInd = repmat(rewInd',length(offset),1)+repmat(offset',1,length(rewInd));
rta = f(rewInd);
clf
subplot(211)
plot(mean(rta(:,1:50),2));hold on;
plot(mean(rta(:,51:end),2))
offset = -10:150;
goInd = repmat(goInd',length(offset),1)+repmat(offset',1,length(goInd));
goInd(goInd<1) = 1;goInd(goInd>length(f))=length(f);
sta = f(goInd);
% clf
subplot(212)
plot(mean(sta(:,1:50),2));hold on;
plot(mean(sta(:,51:100),2))
%%
cnInd = find(diff(medfilt1(f,21))>.3);
cnInd(diff(cnInd)<20) = [];
pre   = find(tn(cnInd)<50);
offset = -100:250;
cnInd = repmat(cnInd',length(offset),1)+repmat(offset',1,length(cnInd));
cnInd(cnInd<1) = 1;cnInd(cnInd>length(f))=length(f);
clear cnta;
for i = 1:length(dat.roi);
    a = F(:,i);
    cnta(:,:,i) = a(cnInd);
end
%%
strt = cell2mat(cellfun(@(x) datenum(x)*(24*60*60),cellstr(behav.trial_start_times),'uni',0));
strt = strt-strt(1);

sInd = (arrayfun(@(x) min(find(x<t)),strt,'uni',0));
ind = cell2mat(cellfun(@(x) isempty(x),sInd,'uni',0));
sInd(ind) = [];sInd = cell2mat(sInd);
rewInd = cell2mat(arrayfun(@(x) min(find(x<t)),rew,'uni',0));

for i = 1:length(sInd)-1;
    ind = sInd(i):sInd(i+1);
    r   = find(rewInd);
    if ~isempty(r);
        ind(r:end) = [];
    end
    L(i) = length(ind);
    a = mean(F(ind(10:end),:));
    bl = mean(F(ind(1:9),:));
    D(i,:) = (a-bl);
end




