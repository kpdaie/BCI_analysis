clear
folder = 'D:\KD\BCI_03\learners\';
files = dir(folder);
files = {files(3:end).name};
files = files(cell2mat(cellfun(@(x) ~isempty(strfind(x,'session')),files,'uni',0)));
returnToBaseline = [6 7 10 11 12];
fls{2} = files(returnToBaseline);
fls{1} = files(setdiff(1:length(files),returnToBaseline));
figure(1);clf;figure(2);clf;
figure(4);clf;figure(5);clf;
for tt = 1:2
    files = fls{tt};
    nf = length(files);
    for fi = 1:nf;
        figure(tt);
        dat = load([folder,files{fi}]);
        keep dat fi files folder fls tt prc DEL stats
        try
            cl = dat.conditioned_coordinates';
        catch
            [dat.siHeader,~] = scanimage.util.opentif([dat.folder,char(dat.siFiles{dat.currentPlane}(1))]);[dat.conditioned_coordinates,dat.conditioned_neuron] = manual_conditioned_neuron_coords(dat,1);
            cl = dat.conditioned_coordinates';
        end
        dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
        for i = 1:length(dat.roi);
            dist(i) = sqrt(sum((dat.roi(i).centroid - cl).^2));
        end
        zoom = dat.siHeader.SI.hRoiManager.scanZoomFactor;
        dist = dist*1.5*(1/zoom);
        n = length(dat.roi);
        nt = length(dat.roi(1).intensity);
        clear F
        F = nan(1000,n,nt);
        fluor_fun = @(x) cell2mat(cellfun(@(x) x',x,'uni',0));
        clear bl
        for i = 1:n;
            f = fluor_fun(dat.roi(i).intensity);
            bl(i) = prctile(f,20);
        end
        % clear F
        % for j = 1:n;
        %     for i = 1:nt;
        %         a = dat.roi(j).intensity{i};
        %         a = (a-bl(j))/bl(j);
        %         b = 1000 - length(a);
        %         if b > 0;
        %             a = [a;nan(b,1)];
        %         else
        %             a = a(1:1000);
        %         end
        %         F(:,j,i) = a;
        %     end;
        % end
        clear df
        base = dat.bases{dat.currentPlane};
        try
            strt = cell2mat(cellfun(@(x) ~isempty(strfind(x,base)),...
                dat.intensityFile,'uni',0));
        catch
            strt = 1;
        end
        strt = min(find(strt==1));
        for i = 1:n;
            a = fluor_fun(dat.roi(i).intensity(strt:end));
%             bl = std(a);
                        bl = prctile(a,20);
            df(:,i) = (a-bl)/bl;
            
            raw(:,i) = a - min(a);
        end
        
        cn = dat.conditioned_neuron;cnn = df(:,cn);
        len = 5000;
        smoothed = conv(cnn,ones(len,1))/len;
        smoothed = smoothed(len:end);
        [~,b] = max(smoothed);
        [~,a] = min(smoothed(1:10000));
        ind1 = 1:3600;
        ind2 = round(b-len/4):round(b+len/1.5);
        ind2(ind2>length(df))=[];ind2(ind2 < ind1(end)+1000) = [];
        dt_si = 1/dat.siHeader.SI.hRoiManager.scanFrameRate;
        ts = 0:dt_si:dt_si*(length(df)-1);
        ts = ts/60;
        
        KDsubplot(2,7,[1 fi],.55);
        plot(ts,cnn,'k');hold on
        yl = ylim;ylim([-3 yl(2)]);yl = ylim;
        plot(ts(ind1([1 end]))+0,[1 1]*yl(1)+1,'color',[.5 .5 .5]);hold on;
        plot(ts(ind2([1 end])),[1 1]*yl(1)+1,'k');hold on;
        if fi == 1
            xlabel('Time (min)');
            ylabel('Conditioned neuron (\DeltaF/F)')
        end
        title(dat.folder)
        box off
        
        
        del = mean(df(ind2,:)) - mean(df(ind1,:));
        KDsubplot(2,7,[2 fi],.55);
        scatter(dist,del,'k');
        if fi == 1
            xlabel('Distance (\mum)');
            ylabel('\Delta activity')
        end
        xl = xlim;
        xlim([-10 xl(2)]);
        hold on;
        scatter(dist(cn),del(cn),'k','markerfacecolor','r');
        drawnow
        
        stats{tt}(:,fi) = [mean(del) var(del)];
        prc{tt}(fi) = mean(del<del(cn));
        DEL{tt}(fi) = del(cn);
        
%         cn = dat.conditioned_neuron;cnn = raw(:,cn);
%         len = 5000;
%         smoothed = conv(cnn,ones(len,1))/len;
%         smoothed = smoothed(len:end);
%         [~,b] = max(smoothed);
%         [~,a] = min(smoothed(1:10000));
%         ind1 = 1:3600;
%         ind2 = round(b-len/4):round(b+len/1.5);
%         ind2(ind2>length(df))=[];ind2(ind2 < ind1(end)+1000) = [];
%        
%         for ci = 1:size(df,2);
%             a = diff(medfilt1(df(:,ci),21));
%             evt = a.*(a > prctile(a,90));
%             in1 = evt(ind1);in1(in1==0) = [];
%             pre = mean(in1);
%             in2 = evt(ind2);in2(in2==0) = [];
%             post = mean(in2);
%             d_amp(ci) = post/pre;
%             evt = double(evt~=0);
%             rate1 = sum(evt(ind1))/(diff(ts(ind1([1 end]))));
%             rate2 = sum(evt(ind2))/(diff(ts(ind1([1 end]))));
%             d_rate(ci) = rate2/rate1;
%         end
%         figure(3+tt);
%         subplot(1,7,fi);
%         scatter(dist,mean(raw(ind2,:))./mean(raw(ind1,:)),'k');
% %         loglog(d_amp,d_rate,'ko');hold on;
% %         loglog(d_amp(cn),d_rate(cn),'ko','markerfacecolor','r')
%         
    end
end
%%
figure(3);clf
subplot(121);
off = [1 9];
for tt = 1:2
    y = prc{tt}*100;nn = length(y);
    scatter(off(tt)+randn(1,nn),y,'k','markerfacecolor','w');hold on;
end
ylabel('Percentile of CN');
set(gca,'xtick',off,'xticklabel',{'No','Yes'});
xlabel('Return to baseline?')
subplot(122);
off = [1 5];
for tt = 1:2
    y = DEL{tt};nn = length(y);
    scatter(off(tt)+randn(1,nn)/2,y,'k','markerfacecolor','w');hold on;
end
ylabel('\Delta activity of CN');
set(gca,'xtick',off,'xticklabel',{'No','Yes'});
xlabel('Return to baseline?')
figure(55);clf
off = [1 5];
for k = 1:2
    subplot(1,2,k);
    for tt = 1:2
        y = stats{tt}(k,:);nn = length(y);
        scatter(off(tt)+randn(1,nn)/2,y,'k','markerfacecolor','w');hold on;
        if k == 1;
            plot(xlim,xlim*0,'k:')
        end
    end
    if k == 1
        ylabel('Avg. \Delta activity of population');
    else
        ylabel('Var. \Delta activity of population');
    end
    set(gca,'xtick',off,'xticklabel',{'No','Yes'});
    xlabel('Return to baseline?')
end

%%
figure(22);
load('X:\bci_behavior\BCI_03_may2021.mat')
len = 20;
clf
for si = 1:length(dat);
    hit = dat(si).hit;   
    
    hit = hit(dat(si).FirstTrial:end);
    clear perf
    for i = 3:length(hit);
        if i <= len
            perf(i) = mean(hit(1:i)==2);
        else
            perf(i) = mean(hit(i-len:i)==2);
        end
    end
    subplot(2,5,si)
    plot(perf(3:end),'k');
    box off
    perf2(:,si) = interp1(1:length(perf),perf,1:100,'linear','extrap');
    title(dat(si).date(1:6));
    if si == 6;
        xlabel('Trial #');
        ylabel('Hit rate');
    end
end
    

%%
% w = ws.loadDataFile([dat.folder,char(dat.wsFiles{dat.currentPlane})]);
% dt_si = 1/dat.siHeader.SI.hRoiManager.scanFrameRate;
% ts = 0:dt_si:dt_si*(length(df)-1);
% scans = w.sweep_0001.analogScans;
% dt_ws = 1/w.header.AcquisitionSampleRate;
% tw = 0:dt_ws:dt_ws*(length(scans)-1);
% scan = interp1(tw,scans,ts,'linear');
% rew = scan(:,5);
% rind = find(rew>1);
% rind(diff(rind)<2) = [];
% clf
% cn = dat.conditioned_neuron;
% clear rta
% for i = 1:length(rind)
%     ind = rind(i)-200:rind(i)+100;
%     ind(ind<1) = 1;ind(ind>length(df))=length(df);
%     rta(:,:,i) = df(ind,:);
% end
% k = squeeze(mean(dr(51:160,:,:)))-squeeze(mean(dr(201:end,:,:)));
% d = mean(k(:,51:end-10)')-mean(k(:,1:21)');
% scatter(dist,d,'k')