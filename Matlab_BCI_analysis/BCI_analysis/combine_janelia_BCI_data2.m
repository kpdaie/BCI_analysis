clear all;close all;
% folder = 'G:\My Drive\Learning rules\BCI_backups\';
folder = 'D:\KD\BCI_data\Janelia_multiday\backups\';

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

mice = unique(mouse);
mice(mice==0)=[];
for mi = 1:length(mice)
    ind = find(mouse==mice(mi));
    AcqDates = cell2mat(cellfun(@(x) (str2num(x)),AcqDate(ind)','uni',0));
    uAcqDates = unique(AcqDates);
    clear temp
    I = 0;
    for di = 1:length(uAcqDates);
        I = I + 1;
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
%             indd = input('which base?')
            indd = 1;
        end
        try
           temp(I).cn = dat.conditioned_neuron;
        catch
           temp(I).cn = [];
        end
        try
            % function [df,dist,F,epoch,tsta,raw,df_all,epoch_Name,Fraw,BCIfile] = BCI_dat_extract4(dat,base,len)
               [temp(I).dff_sessionwise_closed_loop,...
                temp(I).dist,...
                temp(I).dff_trialwise_closded_loop,...
                temp(I).epoch_number,...
                temp(I).time_from_trial_start,...
                temp(I).f_sessionwise_closed_loop,...
                temp(I).dff_sessionwise_all_epochs,...
                temp(I).epoch_name,...
                temp(I).f_trialwise,...
                temp(I).closed_loop_filenames] = BCI_dat_extract4(dat,dat.bases{indd});
            temp(I).file = file;
            temp(I).mouse_name = ['BCI',num2str(mice(mi))];
            temp(I).sessionDate = uAcqDates(di);
            
            temp(I).mean_image = dat.IM;
%             temp(I).roi = dat.roi;
            temp(I).all_si_filenames = dat.intensityFile;
        catch
            temp(I).segmented_data_filename = file;
        end
    end
    data = temp;
    
    data(1).mouse = (['BCI',num2str(mice(mi))])
%     save([folder,data(1).mouse,'_',datestr(now,'mmddyy'),'v4'],'data','-v7.3')
%     save(['D:\KD\BCI_data\Janelia_multiday\',data(1).mouse,'_',datestr(now,'mmddyy'),'v5'],'data','-v7.3');
    save(['D:\KD\BCI_data\Janelia_multiday\',data(1).mouse,'_',datestr(now,'mmddyy'),'v7'],'data','-v7.3','-nocompression');
%     save(['D:\KD\BCI_data\Janelia_multiday\',data(1).mouse,'_',datestr(now,'mmddyy'),'v6'],'data');
%     h5write(['D:\KD\BCI_data\Janelia_multiday\',data(1).mouse,'_',datestr(now,'mmddyy'),'h5'],'BCI_data',data)
end