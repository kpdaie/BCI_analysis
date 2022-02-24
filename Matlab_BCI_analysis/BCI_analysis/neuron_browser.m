figure(899);
n = length(dat.roi);
subplot(311);
p = nanmean(F(1:200,:,:),3);
p = p - repmat(mean(p(1:20,:)),size(p,1),1);
[a,b] = sort(sum(p),'ascend');
for i = 1:n
    subplot(121);
    imagesc(dat.IM,[0 prctile(dat.IM(:),99.4)]);hold on;set(gca,'colormap',gray);
    plot(dat.roi(b(i)).centroid(1),dat.roi(b(i)).centroid(2),'ro','markersize',20);
    title(['Neuron #',num2str(b(i))]);
    subplot(143);
    plot(df(:,b(i)),'k');box off
    subplot(144);
    plot(tsta(1:size(p,1)),p(:,b(i)),'k');box off
    title(dat.folder)
    pause;
    clf
end
    
    