

out = scanimage.util.readPhotostimMonitorFile('D:\KD\BCI_03\031121\photostim_post22_00001.stim');
%%
clf
beam = diff(medfilt1(out.Beam,201));
beam = sgolayfilt(beam,3,201);
% plot(beam(1:2100000));hold on;
% plot(xlim,[1 1]*prctile(beam,99.55));
stm = find(beam>prctile(beam,99.55));
stm(diff(stm)<3000) = [];
length(stm)
stm = floor(stm*length(f)/length(beam));
seq = dat.siHeader.SI.hPhotostim.sequenceSelectedStimuli;
seq = seq(dat.siHeader.SI.hPhotostim.sequencePosition+0:length(stm));

clear A
for i = 1:length(distance);
    [a,b] = min(distance{i});
    cl(i)= b;
    f = cell2mat(cellfun(@(x) x',dat.roi(22).intensity,'uni',0));
    ff(:,i) = f;
    bl = prctile(f,20);
    f = (f-bl)/bl;
    ind = find(seq == i);
    stmind = stm(ind);
    subplot(10,3,i);
    offset = -30:100;
    stmind = repmat(stmind',length(offset),1) + repmat(offset',1,length(stmind));
    stmind(stmind<1) = 1;
    stmind(stmind>length(f)) = length(f);
    plot(mean(f(stmind),2))
    A(:,i) = mean(f(stmind),2);
    axis tight
    set(gca,'visible','off')
end
% clf
% plot(medfilt1(nanmean(A'),7))
% plot(mean(A,2))
    
    
    