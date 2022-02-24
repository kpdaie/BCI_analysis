csvFiles = {...
    'D:\bScope\BCI2\062320\neuron1_4_IntegrationRois_00001.csv',...
    'D:\bScope\BCI2\062420\neuron1_1__IntegrationRois_00001.csv',...
    'D:\bScope\BCI2\062520\neuron2__IntegrationRois_00001.csv',...
    'D:\bScope\BCI2\062720\neuron1_pole_IntegrationRois_00001.csv',...
    'D:\bScope\BCI2\062820\neuron3_IntegrationRois_00001.csv',...
    'D:\bScope\BCI2\062920\neurons2_IntegrationRois_00001.csv',...
    'D:\bScope\BCI2\070220\neuron1_2_IntegrationRois_00001.csv',...
    'D:\bScope\BCI2\070220\neuron5_IntegrationRois_00001.csv',...
    'D:\bScope\BCI1\063020\neurons1_IntegrationRois_00001.csv',...
    'D:\bScope\BCI1\070320\neuron1_IntegrationRois_00062.csv'};
conditioned_neuron = [9,3,4,8,13,3,9,3,2,9];
strt = [20000 1 1 1 1 1 1 1 1 1];
%%
for i = 1:length(csvFiles);
    % i = 2;
    i
    T = readtable(csvFiles{i});
    T = T{:,conditioned_neuron(i)+2};
    try
        T = cell2mat(cellfun(@(x) str2num(x),T(2:end),'uni',0));
    end
    data(i).file = csvFiles{i};
    data(i).cn = T;
end
%%
clf
len = 7000;
colors = ['kkkkkmmkmk'];
clear K
subplot(211);
num = 30000;
t = (1:num)/(20*60);
clear gg
p_i = [1:4 9:13];
for i = 1:length(data);
    f = data(i).cn(strt(i):end);
    bl = prctile(f(1:1000),10);
    f = (f-bl)/bl;    
    med = median(f(1:end));
%     med = prctile(f(1:1000),90);
    y = conv(f>med,ones(len,1))/len*(20*60);
    y = conv(f>med,ones(len,1))/len*(20*60);
%     y = conv(f,ones(len,1))/len*(20*60);
    y = y(len:end-len);    
    d = num - length(y);
    if d>0
        yy = [y;nan(d,1)];        
    else
        yy = y(1:num);
    end    
    K(:,i)=yy-mean(yy(1:100));
    KDsubplot(2,8,p_i(i),.3);
    plot(t,yy-mean(yy(1:100)),'k');hold on;
    plot(xlim,xlim*0,'k:');
    
    
    a=diff(medfilt1(f,21));
    ind = find(a>.14);
%     ind = find(f>prctile(f,90));
    ind(diff(ind)<20) = [];
    ind(ind<101) = 101;
    ind(ind>length(f))=length(f);
%     inds = repmat(ind',200,1) + repmat((-49:150)',1,length(ind));
%     inds(inds>length(f))=length(f);inds(inds<1) = 1;    
    for ti = 1:length(ind);
        try
        gg{i}(:,ti) = f(ind(ti)-100:ind(ti)+100);
        end
    end
    a = (ylim);
    if a(1)>-80;
        a(1) = -80;
    end    
    if a(2)<80
        a(2) = 80;
    end
    ylim(a);
end
subplot(122);
inds = [1:5 8 10];
% inds = [7:9];
inds = 1:8;
% inds = setdiff(1:10,inds);
K = K(:,inds);
notnan=size(K,2)-sum(isnan(K)');
% plot(nanmedian(K'))
confidence_bounds(t,nanmedian(K'),nanstd(K')./(sqrt(notnan)),'k','k',.2);hold on;
plot(xlim,xlim*0,'k:');
%%
inds = [1:4 8 10];
for i = inds;
    try
    e(:,i) = mean(gg{i}(:,1:10)');
    l(:,i) = mean(gg{i}(:,70:120)');
    end
end
clf
plot(nanmean(l'));hold on;
plot(nanmean(e'));
%%
