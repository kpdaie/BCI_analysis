clear
folder = 'G:\My Drive\Learning rules\BCI_backups\';

files = dir(folder);
files = files(3:end);
fileSizes = [files.bytes]';
dates = {files.date};
names = {files.name};

clear mouse
mouse = zeros(1,length(names));
for i = 1:length(names);
    try
        file = [folder,char(names{i})];
        session = load(file,'folder');
        ind = find(session.folder == '\');
        str = session.folder(ind(2)+1:ind(3)-1);
        AcqDate{i} = session.folder(ind(3)+1:ind(4)-1);
        ind = find(str=='I');
        sess{i} = [file];
        mouse(i) = str2num(str(ind+1:end));
    end
end
%%
mice = unique(mouse);
mice(mice==0)=[];
for mi = 1:length(mice);
    ind = find(mouse==mice(mi));
    AcqDates = cell2mat(cellfun(@(x) (str2num(x)),AcqDate(ind)','uni',0));
    uAcqDates = unique(AcqDates);
    clear temp
    for di = 1:length(uAcqDates);
        ind1 = find(AcqDates == uAcqDates(di));
        sz = fileSizes(ind(ind1));
        ind1 = ind1(max(find(sz==max(sz))));
        file = [folder,names{ind(ind1)}];
        dat = load(file);
        ['BCI',num2str(mice(mi)),char(10),dat.folder,char(10),num2str(di/length(uAcqDates)*100),'%']
        
        try
            indd =cell2mat(cellfun(@(x) ~isempty(strfind(x,'neuron')),dat.bases,'uni',0));
            len=cell2mat(cellfun(@(x) length(x),dat.siFiles,'uni',0));
            indd = find(indd.*len==max(indd.*len));
        catch
            dat.bases
            indd = input('which base?')
        end
        try
            [temp(di).df,...
                temp(di).dist,...
                temp(di).F,...
                temp(di).epoch,...
                temp(di).tsta,...
                temp(di).raw,...
                temp(di).df_all] = BCI_dat_extract2(dat,dat.bases{indd});
            temp(di).file = file;
            try
                temp(di).cn = dat.conditioned_neuron;
            catch
                dat.folder
                temp(di).cn = input(1)
            end
            temp(di).IM = dat.IM;
            temp(di).roi = dat.roi;
        catch
        end
    end
    eval(['data.BCI',num2str(mice(mi)),'=temp;'])
end