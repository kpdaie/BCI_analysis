baseOrder = [4 2 6];
clear epoch
for i = 1:length(baseOrder)
    len(i) = length(dat.siFiles{baseOrder(i)});;
end
len = cumsum(len)
L = [0 len];
n = length(dat.roi);
for i = 1:n;
%     bl(i) = prctile(cell2mat(cellfun(@(x) x',dat.roi(i).intensity,'uni',0)),50);
    bl(i) = std(cell2mat(cellfun(@(x) x',dat.roi(i).intensity,'uni',0)));
end
for ei = 1:length(len);
    w = ws.loadDataFile([dat.folder,char(dat.wsFiles{baseOrder(ei)})]);
    clear F
    scans = w.sweep_0001.analogScans;
    
    clear F
    for i = 1:n;
        F(:,i) = cell2mat(cellfun(@(x) x',dat.roi(i).intensity(L(ei)+1:L(ei+1)),'uni',0));
        if ei == 1;
            F(:,i) = F(:,i) + 40;
        end
    end
    for i = 1:n;
        F(:,i) = (F(:,i)-bl(i))/bl(i);
    end
    dt_si = 1/dat.siHeader.SI.hRoiManager.scanFrameRate;
    dt_ws = 1/w.header.AcquisitionSampleRate;
    ts = 0:dt_si:dt_si*(length(F)-1);
    tw = 0:dt_ws:dt_ws*(length(scans)-1);
    if tw(end) > ts(end);
        ind = 1:min(find(tw>ts(end)));
        tw = tw(ind);
        scans = scans(ind,:);
    elseif ts(end) > tw(end)
        ind = 1:min(find(ts>tw(end)));
        ts = ts(ind);
        F  = F(ind,:);
    end
    scan = interp1(tw,scans,ts,'linear');
    ind = find(~isnan(F(:,1)).*~isnan(scan(:,4))==1);
    cc = corr(scan(ind,:),F(ind,:));
    figure(ei);clf
    for k = 1:2
        [a,b] = sort(cc(3+k,:),'descend');
        for i = 1:10;
            KDsubplot(11,2,[i k],[.3 .2]);
            col = 'k';
            if b(i) == dat.conditioned_neuron
                col = 'r';
            end
            plot(F(:,b(i)),col);
            axis tight;
            set(gca,'xtick',[],'ytick',[]);box off
            if i == 1;
                title(w.header.AIChannelNames{k+3});
            end
        end
        KDsubplot(11,2,[11 k],[.3 0]);
        plot(ts,scan(:,3+k))
        axis tight;
        set(gca,'visible','off')
    end
    epoch.cc{ei} = cc;
    epoch.F{ei} = F;
    epoch.scan{ei} = scan;
    
    stm = find(scan(:,4)>1);
    stm(diff(stm)<2) = [];
    offset = -80:80;
    ind = repmat(offset,length(stm),1)+...
        repmat(stm,1,length(offset));
    ind(ind<1) = 1;ind(ind>length(F)) = length(F);
    for i = 1:n
        a = F(:,i);
        epoch.motor{ei}(:,i) = mean(a(ind),1);
    end
    
    stm = find(scan(:,5)>1);
    stm(diff(stm)<2) = [];
    offset = -80:80;
    ind = repmat(offset,length(stm),1)+...
        repmat(stm,1,length(offset));
    ind(ind<1) = 1;ind(ind>length(F)) = length(F);
    for i = 1:n
        a = F(:,i);
        epoch.rew{ei}(:,i) = mean(a(ind),1);
    end
end



