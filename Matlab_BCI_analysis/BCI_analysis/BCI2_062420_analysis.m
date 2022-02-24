% So far, I am using my notes and manual clicking to ID the conditioned
% neuron, need to check if SI has this information saved somewhere.
cl = dat.conditioned_coordinates;
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;

for i = 1:length(dat.roi);
    dist(i) = sqrt(sum((dat.roi(i).centroid - cl).^2));
end
[a,b] = sort(dist);
dat.conditioned_neuron = b(1);
conditioned_neuron = b(1);
N = length(dat.roi);
mins = 3;
len = 20*60*mins;
f = @(x) reshape(x,length(x(:)),[]);
F = f(dat.roi(1).intensity(1:end,1:end));
t = 0:dt_si:dt_si*(length(F)-1);
t = t/60;
pre = find(t<2);
post = find(t>2 & t<35);
for i = 1:N;
    F = f(dat.roi(i).intensity(1:end,1:end));
    if i == 1
    ind = find(F==0);
    end
    F(ind) = [];
    bl = prctile(F,10);
    F = (F - bl)/bl;
    df(:,i) = F;
    
    thr = std(F(pre))/1;
    thr = prctile(F(pre),50);
    ddf(:,i) = diff(medfilt1(df(:,i),15))>thr;
    X = conv(F,ones(1,len))/len;
    XX(:,i) = X(1:end-len+1)/X(len);
    
    X = conv(ddf(:,i),ones(1,len))/mins;
    dXX(:,i) = X(1:end-len+1)-X(len);
    
    aft = mean(F(post));
    bef = mean(F(pre));
    D(i) = (aft-bef)/bef;    
    
    aft = mean(dXX(post,i));
    bef = mean(dXX(pre,i));
    dD(i) = aft-bef;    
end
figure(4)
figure_initialize
set(gcf,'position',[4 5 1.3*3 3*2]);
marg = [.1 .2];
[a,b] = sort(dist);
ind1 = setdiff(1:length(b),b(1));
colors = repmat([.7 .7 .7],N,1)+randn(N,3)/5;colors(colors<0) = 0;colors(colors>1) = 1;
% plot(dist,dD,'o');
clf
lim = 20;
KDsubplot(2,1.2,[1 1.2],marg)
% scatter(dD,D,'ko');hold on;
% scatter(dD(b(1)),D(b(1)),'kx')
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
t = 0:dt_si:dt_si*(length(dXX)-1);
t = t/60;
% plot(t-t(len),dXX(:,ind1),'linewidth',.2)
plot(t-t(len),XX(2:end,ind1),'linewidth',.2)
set(gca,'ColorOrder',colors)
colors = repmat([.7 .7 .7],N,1)+randn(N,3)/5;colors(colors<0) = 0;colors(colors>1) = 1;
xlim([0 lim])
hold on;
% plot(t-t(len),dXX(:,b(1)),'k','linewidth',2)
plot(t-t(len),XX(2:end,b(1)),'k','linewidth',2)
% plot(t-t(len),dXX(:,b(1)),'g','linewidth',1)
set(gca,'xtick',[])
KDsubplot(2,1.2,[2 1.2],marg)
plot(t,df(2:end,b(1)),'k')
xlim([0 lim]);
figure_finalize
set(gca,'xtick',[0 lim])
%%
figure(1)
cr = corr(df(t<2,:));
% mean_bin_plot(cr(b(1),ind1),dD(ind1)/dD(b(1)));
cla
plot(cr(b(1),ind1),dD(ind1),'ko','markerfacecolor','w','markersize',8);hold on;
plot(xlim,[dD(b(1)) dD(b(1))],'r:');
xlabel(['Early correlation (t<2 min.)' char(10),' with cond. neuron'])
ylabel('\Delta event rate')
box off
%%
figure(1)
cla
plot(dist,dD,'ko','markerfacecolor','w','markersize',8);hold on;
plot(xlim,[dD(b(1)) dD(b(1))],'r:');
xlabel(['Distance from' char(10),' cond. neuron (\mum)'])
ylabel('\Delta event rate')
box off
%%
figure(1);
cla
plot(dD(ind1),D(ind1),'ko','markersize',8,'markerfacecolor','w');hold on;
plot(dD(b(1)),D(b(1)),'ko','markersize',8,'markerfacecolor','g');hold on;
xlabel('\Delta event rate')
ylabel('\Delta fluorscence')
%%
[a,b] = sort(dist);
num = -10:30;
tt = 0:dt_si:dt_si*(length(num)-1);
[~,rrr] = min(abs(num));
tt = tt - tt(rrr);
cls = b(1);
evt = find(diff(medfilt1(df(:,cls),15))>.5);
devt = diff(evt);
evt(devt<15) = [];
inds = repmat(evt',length(num),1) + repmat(num',1,length(evt));
T = size(df,1);
inds(inds>T) = T;inds(inds<1)=1;
clear G
for i = 1:size(df,2);
    g = df(:,i);
    g = (g(inds));
    G(:,:,i) = g;
end
f = @(x) squeeze(mean(x,2));
bl = f(G);bl = repmat(mean(bl(1:5,:)),size(bl,1),1);
X = (f(G)-bl);
clf
subplot(211)
% (imagesc(cr));
% colormap(colors);
% colorbar
% a = get(gca,'position');
% set(gca,'visible','off');
% axes
% colors = [0 0 1;1 0 0];
% colors = jet;
% [aa,bb] = sort(cr(:,b(1)),'descend');
% x = linspace(0,1,size(colors,1));
% xx = linspace(0,1,size(df,2));
% colors = interp1(x,colors,xx,'linear');
plot(tt,X(:,fliplr(b)));
% set(gca,'ColorOrder',colors,'position',a)
xlabel('Time from "spike" (s)')
ylabel('\DeltaF/F')
axis tight
box off
subplot(212);
g = mean(X(:,b(1)));
plot(cr(ind1,b(1)),mean(X(:,ind1)),'ko','markersize',8,'markerfacecolor','w');hold on;
% plot(xlim,[g g],'r:');
xlabel(['Early correlation (t<2 min.)' char(10),' with cond. neuron'])
ylabel('Event triggered avg. amp');
box off
%%
clf
idx = [1 3;2 4];
for gg = 1:2
for k = 1:2
    if k == 2
        cls = find(mean(X)<-.1);
    else
        cls = find(mean(X)>.1);
    end
    evt = find(diff(medfilt1(df(:,cls),15))>.5);
    devt = diff(evt);
    evt(devt<15) = [];evt(evt>length(t))=[];
    if gg == 2
        evt(t(evt)<3) = [];
    else
        evt(t(evt)>3) = [];
    end
inds = repmat(evt',length(num),1) + repmat(num',1,length(evt));
T = size(df,1);
inds(inds>T) = T;inds(inds<1)=1;
clear G
for i = 1:size(df,2);
    g = df(:,i);
    g = (g(inds));
    G(:,:,i) = g;
end
f = @(x) squeeze(mean(x,2));
bl = f(G);bl = repmat(mean(bl(1:5,:)),size(bl,1),1);
Y = (f(G)-bl);
subplot(2,2,idx(k,gg))
plot(tt,mean(Y(:,b(1)),2));hold on;
plot(tt,mean(Y(:,cls),2));hold on;
xlabel('Time from "spike" (s)')
ylabel('\DeltaF/F')
box off
set(gca,'Fontsize',12)
ylim([0 2])
% subplot(2,2,k+2);
% plot(tt,mean(X(:,b(1)),2));hold on;
% plot(tt,mean(X(:,cls),2));hold on;
% xlabel('Time from "spike" (s)')
% ylabel('\DeltaF/F')
% box off
end
end
% legend('Conditioned neuron')
%%
f = @(x) reshape(x,length(x(:)),[]);
ff = @(x) sgolayfilt(x,3,15);
F = f(dat.roi(b(1)).intensity(1:end,1:end));
ind = find(F==0);
F(ind) = [];
bl = prctile(F,10);
F = (F - bl)/bl;
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
t = 0:dt_si:dt_si*(length(F)-1);
t = t/60;
clf
ax1 = axes;
plot(t,ff(F),'k')
xlim([0 20])
ylim([0 15])

ax2 = axes;
X = conv(F,ones(1,len))/len;X = X(1:end-len+1)/X(len);
plot(t-t(len),X,'c','linewidth',2);
set(gca,'YAxisLocation','right')
set(gca,'Color','none')
xlim([0 20])
%%
% So far, I am using my notes and manual clicking to ID the conditioned
% neuron, need to check if SI has this information saved somewhere.
cls = find(mean(X)<-.1);
cl = dat.conditioned_coordinates;
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;

for i = 1:length(dat.roi);
    dist(i) = sqrt(sum((dat.roi(i).centroid - cl).^2));
end
[a,b] = sort(dist);
N = length(dat.roi);
mins = 3;
len = 20*60*mins;
f = @(x) reshape(x,length(x(:)),[]);
F = f(dat.roi(1).intensity(1:end,1:end));
t = 0:dt_si:dt_si*(length(F)-1);
t = t/60;
pre = find(t<2);
post = find(t>2 & t<35);
ind = find(F==0);
clear trl
for j = 1:length(cls);
    i = cls(j);
    
    F = f(dat.roi(i).intensity(1:end,1:end));
    ff = dat.roi(i).intensity(1:end,1:end)*0;
    ff(1,:) = 1;
    ff = f(ff);
    if j ==1
        ind = find(F==0);
    end
    F(ind) = [];
    ff(ind) = [];
    bl = prctile(F,10);
    F = (F - bl)/bl;
    num = size(dat.roi(1),2);
    inds = find(ff==1);
    inds(diff(inds)<200) = [];
    a = 0:200;
    inds = repmat(inds,1,length(a)) + repmat(a,length(inds),1);
    inds(inds>length(F)) = length(F);
    inds(inds<1) = 1;
    trl(:,:,j)=F(inds);
end
%%
clf
t = 0:dt_si:dt_si*(size(trl,2)-1);
plot(t,mean(mean(trl(1:end,:,end),1),3)+.7,'k');hold on;
plot(t,mean(mean(trl(1:end,:,1:end-1),1),3),'color',[.5 .5 .5]);hold on;
box off
xlabel('Time from trial start (s)');
ylabel('DF/F')