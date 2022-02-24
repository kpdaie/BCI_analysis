parent_folder = 'F:\BCI\BCI13\';
sessions = dir(parent_folder);
sessions = sessions(3:end);sessions = {sessions.name};
I = 0;
for i = 1:length(sessions);
    folder = [parent_folder,char(sessions{i})];
    files = dir(folder);names = {files.name};
    dat = cell2mat(cellfun(@(x) ~isempty(strfind(x,'session')),names,'uni',0));
    if sum(dat)>0
        I = I + 1;
        files = files(find(dat==1));
        names = {files.name};
        sz = cell2mat({files.bytes});
        sz = find(sz == max(sz));
        file = [folder,'\',char(names{sz})];
        dat = load(file);
        [data(I).df,...
            data(I).dist,...
            data(I).F,...
            data(I).epoch,...
            data(I).tsta,...
            data(I).raw] = BCI_dat_extract(dat);
        data(I).file = file;
        data(I).cn = dat.conditioned_neuron;
        data(I).IM = dat.IM;
        data(I).roi = dat.roi;
    end
end
%%
figure(1);
num = length(data);
tsta = data(1).tsta;
cns = [data.cn];
marg = .3;
tsta = tsta - tsta(40);
clf
colors = 'bmrc';
for i = 1:num;
    for j = 1:num;
        f = nanmean(data(i).F(:,cns(j),7:end-10),3);
        f = f - mean(f(1:20));
        KDsubplot(num,num+.3,[j,i+.3],marg);
        color = 'k';
        if i == j;color = colors(j);end
        plot(tsta(1:240),f(1:240),color);
        ylim([-.5 1]);
        if i == 1;
            yy=ylabel(['Neuron ',num2str(j),char(10),'DF/F'],'color',colors(j));
            %             yy.color = colors(j);
        end
        if j == 1;
            title(['Day ',num2str(i)],'color',colors(i));
        end
        set(gca,'ytick',[0 .8]);
        if i > 1;
            set(gca,'ytick',[]);
        end
        set(gca,'xtick',[0 8]);
        if j < num;
            set(gca,'xtick',[]);
        end
        if j == num & i == 1;
            xlabel('Time from trial start (s)');
        end
    end
end
figure_finalize
dat = load(data(1).file);
%%
figure(2);clf
imagesc(dat.IM,[0 prctile(dat.IM(:),99.9)]);
for i = 1:num
    showROIsPatch(gcf,colors(i),dat.roi,cns(i),0)
end
set(gca,'position',[0 0 1 1]);
%%
figure(3);clf
for i = 1:num;
    a = nanmean(data(i).F(1:240,:,7:end-10),3);
    a = a - repmat(mean(a(1:20,:)),size(a,1),1);
    p(:,:,i) = a;
end
for i = 2:num;
    dist = data(i).dist;
    del = mean(p(40:150,:,i)) - mean(p(40:150,:,i-1));
    DIST(:,i) = dist;
    DEL(:,i) = del;
    subplot(4,1,i);
    scatter(dist,del,'ko');hold on;
    scatter(dist(cns(i)),del(cns(i)),'ko','markerfacecolor',colors(i));hold on;
    xlim([-20 410]);
end
xlabel('Distance (\mum)')

subplot(411);
fixed_bin_plots(DIST(:),DEL,[0 10 30 60 100 150 200 300 400],1,'k');
ylabel('Day_{0} - Day_{-1}')

figure_finalize
%%
figure(4);clf
clear DIST DEL
for i = 1:num;
    a = nanmean(data(i).F(1:240,:,7:end-10),3);
    a = a - repmat(mean(a(1:20,:)),size(a,1),1);
    p(:,:,i) = a;
end
for i = 2:num-1;
    dist = data(i).dist-1;
    del = mean(p(40:150,:,i+1)) - mean(p(40:150,:,i-1));
    DIST(:,i) = dist;
    DEL(:,i) = del;
    subplot(4,1,i);
    scatter(dist,del,'ko');hold on;
    scatter(dist(cns(i)),del(cns(i)),'ko','markerfacecolor',colors(i));hold on;
    xlim([-20 410]);
end
xlabel('Distance (\mum)')
subplot(411);
fixed_bin_plots(DIST(:),DEL,[0 10 30 60 100 150 200 300 400],1,'k');
ylabel('Day_{+1} - Day_{-1}')

figure_finalize
%%
i = 3;
dat = load(data(i).file);
old = load(data(i-1).file);
multi_day_compare