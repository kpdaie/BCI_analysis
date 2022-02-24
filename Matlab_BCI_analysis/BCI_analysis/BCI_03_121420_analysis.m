cl = dat.conditioned_coordinates';
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
for i = 1:length(dat.roi);
    dist(i) = sqrt(sum((dat.roi(i).centroid - cl).^2));
end
zoom = dat.siHeader.SI.hRoiManager.scanZoomFactor;
dist = dist*1.5*(1/zoom);
%%
len = length(dat.siFiles{dat.currentPlane});
pre = 1:40;
post = 80:150;
for fi = 1:len;
    tic
    file = [dat.folder,'\registered','\',dat.siFiles{dat.currentPlane}{fi}];
    file = [file(1:end-4),'shift','.tif'];
    im1 = KDimread(file,dat.dim,length(pre),pre(1),1);
    im2 = KDimread(file,dat.dim,length(post),post(1),1);
    IM1(:,:,fi) = mean(im1,3);
    IM2(:,:,fi) = mean(im2,3);
    [fi/len toc]
end
%%
D = (IM2 - IM1);
bl = mean(IM1(:,:,:),3);
mask = (bl>prctile(bl(:),50));
dd = mean(D(:,:,51:end),3)-mean(D(:,:,1:50),3);
for i = 31:203;
    i
    A(:,:,i) =  medfilt2(mean(D(:,:,i-30:i),3)./bl.*mask,[3 3]);
end
D = mean(A(:,:,51:100),3) - mean(A(:,:,1:50),3);
%%
colormap(jet);
for i = 31:203;
    imagesc(A(:,:,i),[0 .7]);
    hold on;
    plot(dat.conditioned_coordinates(1),dat.conditioned_coordinates(2),'ro','markersize',20);
    title(num2str(i));
    drawnow;pause(.1);
    clf;
end
%%
clf
imagesc(D,[0 .1]);
hold on;
plot(dat.conditioned_coordinates(1),dat.conditioned_coordinates(2),'ro','markersize',20);
title(dat.folder)
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
for i = 1:n;
    a = squeeze(mean(F(:,i,:),3));
    bl = mean(a(1:10,:));
    F(:,i,:) = (F(:,i,:)-bl)/bl;
    BL(i) = bl;
end
%%
cn = dat.conditioned_neuron;
bins = 6;
num  = floor(nt/bins);
clear binned
bl = mean(mean(F(1:10,cn,:),3));
for bi = 1:bins
    ind = (bi-1)*num + 1;
    ind = ind:ind+num;
    if bi == num;
        ind = ind(1):nt;
    end
    a = sgolayfilt(mean(F(:,cn,ind),3),3,5);
    binned(:,bi) = (a - mean(a(1:10)))/bl;;
%     binned(:,bi) = a;
end

clf
plot(reshape([binned;nan(20,bins)],1,[]),'k')
% dys = linspace(0,0,bins);
% dxs = linspace(bins*55,0,bins);
% cn = dat.conditioned_neuron;
% t = 1:size(binned,1);
% colors = interp1([0 1],[0 0 0;1 0 0],linspace(0,1,bins),'linear');
% for j = 1:bins;    
%     dx = dxs(j);
%     dy = dys(j);
%     yy = binned(:,j);    
%     x = [t fliplr(t)]+dx;
%     y = [binned(:,j)' zeros(1,length(t))]+dy;
%     pp = patch(x,y,'r');hold on;
%     pp.FaceColor = 'w';
%     pp.EdgeColor = 'none';    
%     plot(t+dx,binned(:,j)+dy,'color',colors(j,:));  
% end
%%
clf
subplot(212)
d = squeeze(nanmean(F(20:end,:,:)))-squeeze(nanmean(F(1:10,:,:)));
scatter(dist,mean(d(:,51:end)')-mean(d(:,1:50)'),'k');hold on;
plot(dist(cn),mean(d(cn,51:end)')-mean(d(cn,1:50)'),'rs','markersize',20)
xlim([-30 max(dist)])
ylim([-2 3])
xlabel('Distance (\mum)');

subplot(211);
t = 0:dt_si:dt_si*(size(F,1)-1);
% plot(t,nanmean(F(:,cn,51:100),3),'b');hold on;
% plot(t,nanmean(F(:,cn,1:50),3),'r')
confidence_bounds(t,squeeze(F(:,cn,21:end)),[],'b','b',.2)
confidence_bounds(t,squeeze(F(:,cn,1:20)),[],'r','r',.2)
xlabel('Time (s)');
ylabel('\DeltaF/F');
legend('trls 51:159','trls 1:50')
title(dat.folder)
figure_finalize
%%
pre = pre- repmat(mean(pre(1:10,:)),size(pre,1),1);
post = post - repmat(mean(post(1:10,:)),size(post,1),1);
[u,s,v1] = svd(pre);
[u,s,v2] = svd(post);
clf
scatter(v1(:,1),v2(:,1));hold on;
plot(v1(cn,1),v2(cn,1),'rs','markersize',20)