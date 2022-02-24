function cn = conditioned_neuron_list(parent);
% parent = 'F:\BCI2\BCI22\';
dirs = dir(parent);
dirs = {dirs(3:end).name};
for i = 1:length(dirs);
    % i = 1;
    try
    folder = [parent,'\',char(dirs{i})];
    files = dir(folder);ff = files(3:end);
    files = {files(3:end).name};
    datfiles = cell2mat(cellfun(@(x) ~isempty(strfind(x,'session')),files,'uni',0));
    a = [ff.bytes].*datfiles;
    [~,ind] = max(a);
    str = [folder,'\',char(files(ind))];
    load(str,'conditioned_neuron')
    cn(i) = conditioned_neuron;
    end
    
end
