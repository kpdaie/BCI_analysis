clf

for ii = [2 1 3 4];
    F = data(ii).F;
    n = size(p,2);
    sx = .06;
    sy = .02;
    p = nanmean(F(1:240,:,7:end-10),3);;
    p = p - repmat(mean(p(1:20,:)),size(p,1),1);
    pp(:,:,ii) = p;
end
%%
today = 2;
[u,s,v] = svd(pp(:,:,today-1));
[u,s,rot] = svd(p(50:100,:,today-1)*v(:,1:2));
v = v(:,1:2)*rot;
del = mean(pp(40:150,:,today))-mean(pp(40:150,:,today-1));del = del/norm(del);
v = [v(:,1:2) del'];
v(:,1) = -v(:,1);
v = Gram_Schmidt_Process(v);

figure(24);clf
subplot(121);
[a,b] = sort(sum(pp(50:100,:,today-1)));
imagesc(pp(:,b,today-1)',[0 .3])
set(gca,'xtick',[40 240],'xticklabel',{'0','10'})
xlabel('Time from trial start (s)');
ylabel('Neuron #')

subplot(122)
plot(tsta,pp(:,:,today-1)*v(:,1:2))
xlabel('Time from trial start (s)');
ylabel('\DeltaF/F')
title('First 2 PCA modes')
figure_finalize
cc = colororder;

figure(11);
clf

marg = .5;
for i = 1:2;
    for j = 1:3;
        KDsubplot(2,3,[i,j],marg);
        plot(tsta,pp(:,:,i+(today-2))*v(:,j),'color',cc(j,:));
        ylim([-3 3])
        xlabel('Time from trial start (s)');
        ylabel('\DeltaF/F')
    end
end
figure_finalize
