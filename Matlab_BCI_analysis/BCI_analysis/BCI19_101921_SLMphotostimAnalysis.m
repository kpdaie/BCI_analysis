
r = rois(:,3:end);
a = mean(r');
ind = find(diff(a)>50);
for i = 1:length(ind);
    ff(:,:,i) = r(ind-50:ind+100,:);
end
stackedplot(mean(ff,3))
%%
dat.currentPlane = 6;
fi = 2;
folder = dat.folder;
files = dat.siFiles{dat.currentPlane};
file = [folder,char(files(fi))];
[~,IM] = scanimage.util.opentif(file);
[hMroiRoiGroup hStimRoiGroups] = scanimage.util.readTiffRoiData(file);
%%
a = squeeze(mean(mean(IM,1),2));
ind = find(diff(a)>prctile(diff(a),99));
bef = zeros(dat.dim);
aft = bef;
for i = 1:length(ind);
    post = ind(i)+4:ind(i)+40;
    pre = ind(i)-50:ind(i)-2;
    pre(pre<1) = 1;
    %     post(post>size(IM,3)) = size(IM,3);
    aft = aft + mean(IM(:,:,post),3);
    bef = bef + mean(IM(:,:,pre),3);
end
clf
D = aft - bef;
rng = prctile(D(:),99.96);
imagesc(medfilt2(aft-bef,[5 5]),[-1 1]*rng)
slm = hStimRoiGroups.rois(2).scanfields.slmPattern;
sg = units_to_pixels(hStimRoiGroups.rois(2).scanfields,dat.siHeader,dat.dim);
pix = sg.SLM_pix;
hold on;
for i = 1:size(pix,2);
    plot(pix(1,i),pix(2,i),'ko','markersize',15);
end
title([file,char(10),'power ~ ',num2str(sg.powers*10/size(pix,2)),' mW per cell'])
set(gca,'xtick',[],'ytick',[]);box off;
cm = bluewhitered;
colormap(cm)
plot(dat.conditioned_coordinates(1),dat.conditioned_coordinates(2),'ro','markersize',20)
%%
[df,dist,F,epoch,tsta,raw,df_all] = BCI_dat_extract(dat);
%%
f = df_all{6};
a = mean(f');
ind = find(diff(a)>prctile(diff(a),99));
ind(diff(ind)<10) = [];
length(ind);
clear dff
for i = 1:length(ind);
    win = [ind(i)-15:ind(i)-2 ind(i)+4:ind(i)+40];
    win(win<1)=1;win(win>length(f))=length(f);
    dff(:,:,i)=f(win,:);
end
aff = mean(dff,3);
aff = aff - repmat(mean(aff(1:13,:)),size(aff,1),1);
eff = mean(aff(19:33,:));

clf
a = [dat.roi.centroid];
XY(:,1) = a(1:2:end);XY(:,2) = a(2:2:end);
for cl = 1:length(dat.roi);
    distance(cl) = min(sqrt(sum((bsxfun(@minus,pix,XY(cl,:)')).^2,1)));
end
scatter(distance,eff,'ko');hold on
scatter(distance(cn),eff(cn),'ko','markerfacecolor','r')
xlim([-50 600])
ind = find(distance>30);
% scatter(del(ind),eff(ind));