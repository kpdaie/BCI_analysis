for i = 1:length(dat.roi);
    a=[];
    for k = 1:length(dat.roi(1).intensity);
        a=[a;dat.roi(i).intensity{k}];
    end;
    F(:,i) = a;
end
bl = prctile(F(1:1000,:),20);bl = repmat(bl,size(F,1),1);
df = (F-bl)./bl;
num = (2*60*20);
del = (mean(df(30000:end,:)) - mean(df(1:num,:)))./mean(df(1:num,:));
cl = dat.conditioned_coordinates';
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
for i = 1:length(dat.roi);
    dist(i) = sqrt(sum((dat.roi(i).centroid - cl).^2));
end
[a,b] = sort(dist);
dat.conditioned_neuron = b(1);
zoom = dat.siHeader.SI.hRoiManager.scanZoomFactor;
pix = dat.dim(1);
distCorrection = 1500/zoom/pix;
scatter(dist*distCorrection,del,'k')
%%
clf
cc = corr(df(1:1000,:),...
    df(1:1000,dat.conditioned_neuron));
[a,b] = sort(cc,'descend');
[a,b] = sort(dist);
rng = [-.5 1.5];
ratio = rng(2)/abs(rng(1));
num = 100;
len = floor(num/ratio);
inds = 1:10000;
pos = interp1([0 1],[1 1 1;1 0 0],linspace(0,1,num));
neg = interp1([0 1],[0 0 1;1 1 1],linspace(0,1,len));
gap = ones(2,3);
cm = [neg;gap;pos];
KDsubplot(10,1,[1 1],[.7 .3]);
imagesc(medfilt1(df(inds,b(1)),21)',rng)
set(gca,'xtick',[]);
set(gca,'ytick',[1]);
KDsubplot(1.111,1,[1.075 1],[.7 .3]);
imagesc(medfilt1(df(inds,b(2:end)),21)',rng)
colormap(cm);
set(gca,'ytick',[1 length(b)-1],'yticklabel',{'2',num2str(length(b))});
figure_finalize
mins = 5;
minss = floor((mins*60)/dt_si);
set(gca,'xtick',[1 minss],'xticklabel',{'0',num2str(mins)});
%%
bl = prctile(F(1:end,:),10);
bl = repmat(bl,size(F,1),1);
df = (F-bl)./bl;
df = medfilt1(df,21);
t = 0:dt_si:dt_si*(length(df)-1);
t = t/60;

ind = [1:8000];
n = size(df,2);
clf
dy = 0;dx = 0;
[~,inds] = sort(dist,'descend');
dys = linspace(15,0,n);
dxs = linspace(200,0,n)/60;
colors = interp1([0 1],[[1 1 1]*0;0 0 0],linspace(0,1,n),'linear');
cn = dat.conditioned_neuron;
for j = 1:n;
    in = inds(j);
    dx = dxs(j);
    dy = dys(j);
    x = [t(ind) fliplr(t(ind))]+dx;
    y = [df(ind,in)' zeros(1,length(ind))]+dy;
    pp = patch(x,y,'r');hold on;
    pp.FaceColor = 'w';
    pp.EdgeColor = 'none';
    if in == cn;
        lw = 3;
    else
        lw = .1;
    end
    plot(t(ind)+dx,df(ind,in)+dy,'color',colors(j,:));    
end
plot(t(ind)+dx,df(ind,in)+dy,'color',colors(j,:),'linewidth',2);    
%%
subplot(212);
imagesc(dat.IM);
ppp = showROIsPatchFace(gcf,{'none',[1 .5 0]},dat.roi,1:length(dat.roi),0,0);
num = (2*60*20);
del = mean(df(num+1:8000,:)) - mean(df(1:num,:));
del = del/max(del);
del(del<0) = 0;
for i = 1:n;ppp(i).FaceAlpha = del(i);end