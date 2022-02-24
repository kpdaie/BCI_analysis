fluor_fun = @(x) cell2mat(cellfun(@(x) x',x,'uni',0));
spont = 5;
open  = 4;
bad   = 2;
bci   = 3;
spont = fluor_fun(dat.roi(1).intensity(1:length(dat.siFiles{spont})));
open = fluor_fun(dat.roi(1).intensity(1:length(dat.siFiles{open})));
bad = fluor_fun(dat.roi(1).intensity(1:length(dat.siFiles{bad})));
L1 = length(spont);
L2 = L1 + length(open);
L3 = L2 + length(bad);
f = fluor_fun(dat.roi(1).intensity);bl = prctile(f,20);
f = (f - bl)/bl;
clf
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
t = 0:dt_si:dt_si*(length(f)-1);
subplot(211);
plot(t/60,f,'r');
axis tight
hold on;
plot(t([1 1]*L1)/60,ylim,'k:');
plot(t([1 1]*L2)/60,ylim,'k:');
plot(t([1 1]*L3)/60,ylim,'k:');
subplot(212);
len = 3000;
a = conv(f,ones(len,1))/len;a=a(len:end);
plot(t/60,a,'r');hold on;
axis tight;
plot(t([1 1]*L1)/60,ylim,'k:');
plot(t([1 1]*L2)/60,ylim,'k:');
plot(t([1 1]*L3)/60,ylim,'k:');

%%
n = length(dat.roi);
nt = length(dat.roi(1).intensity);
clear F
F = nan(1000,n,nt);
fluor_fun = @(x) cell2mat(cellfun(@(x) x',x,'uni',0));
for i = 1:n;
    f = fluor_fun(dat.roi(i).intensity);
    bl(i) = prctile(f,20);
end
for j = 1:n;
    for i = 1:nt;
        a = dat.roi(j).intensity{i};
        a = (a-bl(j))/bl(j);
        b = 1000 - length(a);
        if b > 0;
            
            a = [a;nan(b,1)];
        else
            a = a(1:1000);
        end
        F(:,j,i) = a;
    end;
end
%%
w = ws.loadDataFile([dat.folder,char(dat.wsFiles{3})]);
w2 = ws.loadDataFile([dat.folder,char(dat.wsFiles{4})]);
%%
f = fluor_fun(dat.roi(1).intensity);
bl = prctile(f,20);
dt_ws = 1/w.header.AcquisitionSampleRate;
scans = w.sweep_0001.analogScans;
tw = 0:dt_ws:dt_ws*(length(scans)-1);

fluor_fun = @(x) cell2mat(cellfun(@(x) x',x,'uni',0));
% f = fluor_fun(dat.roi(1).intensity);
f = fluor_fun(dat.roi(1).intensity(53:end));
f = (f - bl)/bl;
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;;
ts = 0:dt_si:dt_si*(length(f)-1);

[~,b] = min([ts(end) tw(end)]);
if b == 1
    tw = tw(1:max(find(tw<ts(end))));
else
    f  = f(1:max(find(ts<tw(end))));
    ts = ts(1:max(find(ts<tw(end))));
end
L = cellfun(@(x) length(x),dat.roi(1).intensity,'uni',0);
L = L(53:end);
L = cell2mat(L);
strt = [0 cumsum(L)]+1;
clear rta sta
for i = 1:length(strt);
    ind = strt(i)-20:strt(i)+100;
    ind(ind<1) = 1;
    ind(ind>length(f)) = length(f);
    sta(:,i) = f(ind);
end
rew = find(scans(:,5)>1);rew(diff(rew)<1000) = [];
rew = round(rew*length(ts)/length(tw));
for i = 1:length(rew);
    ind = rew(i)-50:rew(i)+100;
    ind(ind<1) = 1;
    ind(ind>length(f)) = length(f);
    rta(:,i) = f(ind);
end
clear int
I = 0;
for i = 1:length(strt)-1
    rt = rew(find(rew>strt(i) & rew<strt(i+1)));
    if ~isempty(rt);
        I = I+1;
        ind = strt(i):rt;
        ind(ind>length(f)) = length(f);        
        int(I) = mean(f(ind));
    end
end
%%
f = fluor_fun(dat.roi(1).intensity(9:31));
f = (f - bl)/bl;
L = cellfun(@(x) length(x),dat.roi(1).intensity,'uni',0);
L = L(9:31);
L = cell2mat(L);
strt = [0 cumsum(L)]+1;
clear staOL rtaOL
for i = 1:length(strt);
    ind = strt(i)-20:strt(i)+100;
    ind(ind<1) = 1;
    ind(ind>length(f)) = length(f);
    staOL(:,i) = f(ind);
end
scans = w2.sweep_0001.analogScans;
tw2 = 0:dt_ws:dt_ws*(length(scans)-1);
rew = find(scans(:,5)>1);rew(diff(rew)<1000) = [];
rew = round(rew*length(ts)/length(tw));
for i = 1:length(rew);
    ind = rew(i)-50:rew(i)+100;
    ind(ind<1) = 1;
    ind(ind>length(f)) = length(f);
    rtaOL(:,i) = f(ind);
end
clear intOL
I = 0;
for i = 1:length(strt)-1
    rt = rew(find(rew>strt(i) & rew<strt(i+1)));
    if ~isempty(rt);
        I = I+1;
        ind = strt(i):rt;
        ind(ind>length(f)) = length(f);        
        intOL(I) = mean(f(ind));
    end
end
%%
f = fluor_fun(dat.roi(1).intensity(32:52));
f = (f - bl)/bl;
L = cellfun(@(x) length(x),dat.roi(1).intensity,'uni',0);
L = L(32:52);
L = cell2mat(L);
strt = [0 cumsum(L)]+1;
clear staB rtaB
for i = 1:length(strt);
    ind = strt(i)-20:strt(i)+100;
    ind(ind<1) = 1;
    ind(ind>length(f)) = length(f);
    staB(:,i) = f(ind);
end
scans = w2.sweep_0001.analogScans;
tw2 = 0:dt_ws:dt_ws*(length(scans)-1);
rew = find(scans(:,5)>1);rew(diff(rew)<1000) = [];
rew = round(rew*length(ts)/length(tw));
for i = 1:length(rew);
    ind = rew(i)-50:rew(i)+100;
    ind(ind<1) = 1;
    ind(ind>length(f)) = length(f);
    rtaB(:,i) = f(ind);
end
clear intB
I = 0;
for i = 1:length(strt)-1
    rt = rew(find(rew>strt(i) & rew<strt(i+1)));
    if ~isempty(rt);
        I = I+1;
        ind = strt(i):rt;
        ind(ind>length(f)) = length(f);        
        intB(I) = mean(f(ind));
    end
end
%%
clf
tt = 0:dt_si:dt_si*(size(rtaOL,1)-1);
subplot(131);
plot(tt-tt(50),mean(rtaOL'));hold on;
plot(tt-tt(50),mean(rta'));
xlabel('Time from reward (s)');
plot([0 0],ylim,'k');
xlim([-3 6]);
subplot(132);
tt = 0:dt_si:dt_si*(size(sta,1)-1);
plot(tt-tt(20),mean(staOL'));hold on;
plot(tt-tt(20),mean(sta'));
plot([0 0],ylim,'k');
xlim([-1 6]);
legend('open','closed');
xlabel('Time from trial start (s)');
subplot(133);
eb = errorbar([1 2],[mean(intOL) mean(int)],[std(intOL)/sqrt(length(intOL)) std(int)/sqrt(length(int))])
set(eb,'color','k','marker','s','markerfacecolor','w','markersize',20);
set(gca,'xtick',[1 2],'xticklabel',{'open','closed'});
title('pre-reward');