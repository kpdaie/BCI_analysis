folder = 'D:\KD\BCI_03\learners\';
files = dir(folder);
files = files(3:end);
dt    = {files.date};
[a,b] = sort(dt);
files = files(b);dt = dt(b);
files = {files.name};
ind = cell2mat(cellfun(@(x) ~isempty(strfind(x,'.csv')),files,'uni',0));
files = files(ind);
dt = dt(ind);
[a,b] = sort(dt);
files
figure(85);
for i = 1:length(files)
    str = [folder,char(files{i})];
    rois = table2array(readtable(str));
    a = strfind(str,'neuron');
    b=strfind(str,'_');
    ind = min(find(b>a));
    b = b(ind);
    num = str2num(str(a+6:b-1));
    subplot(2,7,i);
    cn = rois(:,num+2);
    bl = prctile(cn,50);
    cn = (cn-bl)/bl;
    len = 6000;
    cn = conv(cn,ones(len,1)/len);
    dtt = mean(diff(rois(:,1)));
    t = 0:dtt:(length(cn)-1)*dtt;
    t = t/60;
    plot(t,cn,'k');
    axis tight
    xlim(t([1 length(cn)-len]));
    title(dt{i}(1:6))
    if i == 8;
        xlabel('Time (min.)');
        ylabel('Smoothed \DeltaF/F (CN)')
    end
    box off
end
    