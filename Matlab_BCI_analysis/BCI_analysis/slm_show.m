function slm_show(dat,slmPln,grps,save)
%%
% dat = load('F:\BCI\BCI19\102521\session_102521_analyzed_dat_small_neuron1102521.mat');
% dat = load('F:\BCI\BCI19\102221\session_102221_analyzed_dat_small_slm22102521.mat');

% dat.currentPlane = 4;
isslm = cell2mat(cellfun(@(x) ~isempty(strfind(x,'slm')),dat.bases,'uni',0));
len = cell2mat(cellfun(@(x) length(x),dat.siFiles,'uni',0));
[~,slmPln] = max(len.*isslm);
file = [dat.folder,char(dat.siFiles{slmPln}{1})];
[hMroiRoiGroup hStimRoiGroups] = scanimage.util.readTiffRoiData(file);
header = scanimage.util.opentif(file);
seq = header.SI.hPhotostim.sequenceSelectedStimuli;
seq = repmat(seq,1,10);
seq = seq(header.SI.hPhotostim.sequencePosition:end);
seq = seq(1:length(dat.siFiles{slmPln})-1);
dat.stimGroup = hStimRoiGroups;
if nargin == 2;
    grps = 1:max(seq);
end
si = grps;
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
if nargin == 4
    if save == 1
        folder = [dat.folder,'\figures\_'];
        if ~isdir(folder);
            mkdir(folder)
        end
        str = num2str(si);
        print(gcf,'-dpng',[folder,str,'pvalues.png'])
    end
end
    % close all