folders = {'D:\bScope\bci08\012221\',...
    'D:\bScope\bci08\012121\012121\'};
fi = 2
photostim_group_old(folders{fi});
global dat
pre = template_image_maker2(1:4,dat.currentPlane,1,0,[],2,0);
post = template_image_maker2(15:20,dat.currentPlane,0,0,[],2,0);
%%
ref = mean(pre,3);
for i = 1:size(post,3);
    i/size(post,3)
    post(:,:,i) = shiftIm(post(:,:,i),ref,3);
end
f1 = mean(pre,3);
f2 = mean(post,3);
%%
figure(fi+20);
D = (f2 - f1)./f1;
D = D.*(f1 > prctile(f1(:),80));
imagesc(mean(pre,3),[0 300]);
colormap((gray));
set(gca,'position',[0 0 1 1],'visible','off');
X = getframe(gcf);
X = X.cdata;
folder = 'D:\bScope\bci08\';
imwrite(X,[folder,'img012221_',num2str(fi),'.png']);
figure(fi+21);
imagesc(D,[-1 1]*2);
colormap(jet)
set(gca,'position',[0 0 1 1],'visible','off');
X = getframe(gcf);
X = X.cdata;
imagesc(medfilt2(abs(D)>.5,[9 9]));
set(gca,'position',[0 0 1 1],'visible','off');
A = getframe(gcf);
A = A.cdata;
A = double(A(:,:,1)>0);
imwrite(X,[folder,'map012221_',num2str(fi),'.png'],'Alpha',A);
imwrite(X,[folder,'map012221_2_',num2str(fi),'.png']);

%%
cm = jet;s = length(jet)/2;
rng = prctile(D(:),[.1 99.99]);
rng = [-1 1]*2;
ratio = abs(rng(2)/rng(1));
imagesc(D,rng);
red = interp1([0 1],[1 1 1;1 0 0],linspace(0,1,100),'linear');
blue = interp1([0 1],[0 0 1;1 1 1],linspace(0,1,floor(100/ratio)),'linear');
gap  = ones(4,3);
% cm = [blue;gap;red];
% cm = [cm(1:s,:);ones(44,3);cm(33:end,:)];
% cm = interp1(linspace(0,1,3),[0 0 1;1 1 1;1 0 0],linspace(0,1,100),'linear');
colormap(cm);
