keep dat
files = dat.siFiles{1};
clear IM cam

tone = zeros(1,14);
fileNums = 46:53;% awesome performance, quick response to tone
% fileNums = 18:25;% threshold was high, good for demo of bmi control
TONE = [];F = [];
L = cell2mat(cellfun(@(x) length(x),dat.roi(1).intensity,'uni',0));
frames = sum(L(fileNums));

for fileNum = fileNums;
    
end

for fileNum = fileNums;
    fileNum
    newSiFile = char(dat.siFiles{dat.currentPlane}{fileNum});
    a = max(find(newSiFile=='_'));b = regexp(newSiFile,'.tif');
    file_shift = [dat.folder,'registered\',newSiFile(1:end-4),'shift.tif'];
    len = length(imfinfo(file_shift));
    im = KDimread(file_shift,[800 800],len,1,1);
    im = im(1:500,1:500,:);
    
    
    camRate = 300;
    twoPrate= 20;
    dat.camFolder = 'D:\videos\kd\BCI2\102420\';
    camFiles = dir(dat.camFolder );camFiles = {camFiles.name};
    dat.camFiles = camFiles(3:end);
    file = [dat.camFolder,dat.camFiles{fileNum}];
    a=VideoReader(file);
    b=read(a);
    b = squeeze(b(:,:,1,1:floor(camRate/twoPrate):end));
%     if size(b,3) < size(im,3);
%         b(:,:,size(b,3)+1:size(im,3)) = .5;
%     end
    clear tone
    if fileNum == fileNums(1)
        cam = b;
    else
        cam = cat(3,cam,b);
    end
    
    if size(b,3) < size(im,3);
       im = im(:,:,1:size(b,3));
    end
     if fileNum == fileNums(1)
        IM = im(:,:,:);
    else
        IM = cat(3,IM,im(:,:,:));
     end
     
     f = dat.roi(17).intensity{fileNum}';
     f = f(1:size(b,3));
     F = [F f];
     f = f*0;f(14:20) = 1;
     TONE = [TONE f];
end
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
t = 0:dt_si:dt_si*(length(F)-1);
F = (F-prctile(F,10))/prctile(F,10);
%%
vidObj = VideoWriter(['bci',num2str(round(rand*10000)),'.avi']);
set(vidObj,'FrameRate',20);
open(vidObj);
clf
set(gcf,'units','inches');
colormap(gray)
set(gcf,'position',[16.8125    2.4479  10 5]);
set(gcf,'color','w')
marg = [.35 .05];
ax1 = KDsubplot(1.4,2,[1 1],marg)
ax2 = KDsubplot(4.8,1,[4.6 1],marg)
ax3 = KDsubplot(1.4,2,[1 2],marg)
for i = 15:size(IM,3);
    axes(ax1);cla
    imagesc(mean(IM(:,:,i-14:i),3),[30 700]);hold on;
    plot(dat.roi(17).centroid(1),dat.roi(17).centroid(2),'sr','markersize',33,...
        'LineWidth',2);
    if TONE(i) == 1;
        tt = text(10,20,'Tone On');
        tt.Color = 'r';
        tt.FontSize = 12;
        tt.FontWeight = 'bold';
    end
    tm = round(t(i)*10)/10;
    tt = text(20,440,[num2str(tm),' s'],'color','w');
    set(gca,'visible','off');
    axes(ax3)
    imagesc(cam(:,:,i));
    set(gca,'visible','off');
    axes(ax2)
   
    cla
    plot(t(1:i),TONE(1:i)*3,'r');hold on;
    plot(t(1:i),F(1:i),'k');hold on;
    set(gca,'xtick',[0 30 60],'ytick',[0 3]);
    ylabel('\DeltaF/F');
    xlabel('Time (s)');
    box off
    xlim([0 t(end)]);ylim([-.3 4]);
    currFrame = getframe(gcf);
    writeVideo(vidObj,currFrame);
    drawnow;
end

close(vidObj);

%%
siFiles = dat.siFiles{1};
folder = dat.folder;
IM = zeros(800,800,120,2);
inds = {1:10,80:120};
for k = 1:length(inds);
    for fileNum = 1:length(inds{k});
        fileNum
        newSiFile = char(siFiles{inds{k}(fileNum)});
        a = max(find(newSiFile=='_'));b = regexp(newSiFile,'.tif');
        file_shift = [folder,'registered\',newSiFile(1:end-4),'shift.tif'];
        %     file_shift = [folder,newSiFile];
        im = KDimread(file_shift,[800 800],120,1,1);
        IM(:,:,:,k) = IM(:,:,:,k) + im(:,:,1:120);
    end
    IM(:,:,:,k) = IM(:,:,:,k) / length(inds{k});
end
%%
clf
int = dat.roi(1).intensity;
trl = [];
num = [];
for i = 90:100;
    a = 0*int{i}';
    a(40:60) = 1;
    b = conv(a,exp(-(1:150)/13));
    b = b(1:length(a));
    trl = [trl a];
    num = [num ones(1,length(b))*i];
end
ind = find(num>40);
cr = corr(trl(ind)',X(ind,:),'type','pearson');
plot(dist,cr,'ko','markersize',4,'markerfacecolor','w');hold on;
xlim([-10 700]);ylim([-.2 .33]);
plot(xlim,xlim*0,'k:');hold on;
box off
set(gca,'TickDir','out','ytick',[-.2 0 .2],'xtick',[0 500])
%%
files = 120:127;
%%
files = dir(dat.folder);
files = {files.name};
files = files(3:end);
ind = cell2mat(cellfun(@(x) ~isempty(strfind(x,'threshold')),files,'uni',0));
files = files(ind);
for i = 1:length(files);
    str = char(files{i});
    load([dat.folder,str]);
    a = max(find(str=='_'));
    b = max(find(str=='.'));
    trlNum=str2num(str(a+1:b-1));
    thr(trlNum,:) = BCI_threshold;
end
%%
siFiles = dat.siFiles{dat.currentPlane};
folder = dat.folder;
for fileNum = 1:length(siFiles);
        fileNum
        newSiFile = char(siFiles{(fileNum)});
        a = max(find(newSiFile=='_'));b = regexp(newSiFile,'.tif');
        file_shift = [folder,'registered\',newSiFile(1:end-4),'shift.tif'];
        %     file_shift = [folder,newSiFile];
        im = KDimread(file_shift,[800 800],60,1,1);
        IM2(:,:,fileNum) = mean(im(:,:,40:60),3);
        IM1(:,:,fileNum) = mean(im(:,:,1:15),3);
 end
 %%
 clear F
 F = nan(600,length(dat.roi),length(dat.roi(1).intensity));
 for i = 1:length(dat.roi);
     for j = 1:length(dat.roi(1).intensity);
         a = dat.roi(i).intensity{j}(1:end);
         if length(a) > 600
             a = a(1:600);
         end
         
         F(1:length(a),i,j) = a;
     end
 end
 for i = 1:size(F,2);
     a = F(1:10,i,:);
     bl = mean(a(:));
     F(:,i,:) = (F(:,i,:)-bl)/bl;
 end
 %%
 N = 10;
 vidObj = VideoWriter(['bciMap',num2str(round(rand*10000)),'.avi']);
set(vidObj,'FrameRate',5);
open(vidObj);
 for i = N+1:size(IM2,3)-N;
     a = mean(IM1(:,:,i-N:i+N),3);
     b = mean(IM2(:,:,i-N:i+N),3);
     p = zeros(size(IM1,1),size(IM1,2),3);
     g = ((b - a)./a).*(a>prctile(a(:),90));
     p(:,:,1) = g;
     p(:,:,2) = g;
     cla
     imagesc(dat.IM);
     set(gca,'xtick',[],'ytick',[])
     hold on;
     h = imagesc(p,[.1 .8]);
%      cm = [1 1 1;1 .5 0];
%      cm = interp1([0 1],cm,linspace(0,1,64),'linear');
%      colormap(cm);
     
%      cm = [1 1 1;0 0 0];
%      cm = interp1([0 1],cm,linspace(0,1,64),'linear');
%      colormap(cm);
     set(h,'AlphaData',.7);
%      colorbar
     set(gca,'xtick',[],'ytick',[])
     tt = text(500,100,['Trial #',num2str(i-N)]);     
     tt.FontSize = 18;
     tt.Color = 'w';
     drawnow;
     currFrame = getframe(gcf);
     writeVideo(vidObj,currFrame);
 end
 close(vidObj);
 %%
%  figure
subplot(121);
t = 0:.05:.05*(size(F,1)-1);

plot(t-t(20),mean(squeeze(F(:,17,:)),2),'k');hold on;
plot(t-t(20),(t>t(14) & t<t(20))*.6,'r')
box off
xlabel('Time from tone end (s)')
ylabel('\DeltaF/F')
subplot(122);

imagesc(squeeze(F(:,17,:))',[0 .8])
set(gca,'xtick',[20 80],'xticklabel',{'0','4'})
xlabel('Time from tone end (s)')
ylabel('Trial #')
size(F)
imagesc(squeeze(F(:,17,:))',[0 .8])
set(gca,'xtick',[20 80],'xticklabel',{'0','4'})
xlabel('Time from tone end (s)')
ylabel('Trial #')
%  close(vidObj);
%%
clf
clear F
for i = 1:length(dat.roi)
    for j = 1:length(dat.roi(1).intensity);
        F(:,i,j) = dat.roi(i).intensity{j}(1:140);
    end
end
for i = 1:length(dat.roi)
    a = squeeze(F(:,i,:));
    bl = mean(mean(a(1:10,:)));
    F(:,i,:) = (F(:,i,:)-bl)/bl;
end
del = squeeze(mean(F(20:60,:,:)))-squeeze(mean(F([1:10 120:140],:,:)));
for i = 1:length(dat.roi)
    dist(i) = sqrt(sum((dat.roi(i).centroid-dat.roi(17).centroid).^2));
end
plot(dist,mean(del'),'o')
% plot(conv(del(17,:),ones(10,1)))
% xlim([10 160])
plot(dist,mean(del(:,120:end)'),'ko','markerfacecolor','w')
xlim([-10 700]);
xlabel('Distance from conditioned neuron (\mum)')
ylabel('\DeltaF/F after tone')

%%
keep dat pixels
F = [];TONE = [];LP=[];
for fileNum = 106:116;
    camRate = 300;
    twoPrate= 20;
    dat.camFolder = 'D:\videos\kd\BCI2\102420\';
    camFiles = dir(dat.camFolder );camFiles = {camFiles.name};
    dat.camFiles = camFiles(3:end);
    file = [dat.camFolder,dat.camFiles{fileNum}];
    a=VideoReader(file);
    b=read(a);
    b = squeeze(b(:,:,1,1:floor(camRate/twoPrate):end));
    if ~exist('pixels')
        imagesc(b(:,:,1));
        pixels = roipoly;
    end
    f = dat.roi(17).intensity{fileNum}(1:size(b,3));
    lp = f*0;
    tone = 0*f;
    tone(14:20) = 1;
    for fi = 1:size(b,3);
        a = b(:,:,fi);        
        lp(fi) = mean(a(pixels==1));
    end
    F = [F; f];
    TONE= [TONE; tone];
    LP = [LP;lp];
end
F = (F-prctile(F,10))/prctile(F,10);
LP = (LP-min(LP));LP = LP/max(LP);
%%
figure(12)
clf
t = 0:.05:.05*(size(F,1)-1);
t = t-65;
subplot(211);
plot(t,F,'k');hold on;
% plot(t,TONE*6,'r');
xlim([0 25])
ylabel(' \DeltaF/F')
box off
set(gca,'xtick',[0 10 20])
subplot(212);
y = medfilt1(abs(LP),27);
y = medfilt1((abs(diff(LP))*100+100),13);
plot(t(2:end),y*30,'k');hold on;
% plot(t,TONE,'r');
xlim([0 25])
xlabel('Time (s)');
ylabel('Lickport velocity (AU)')
yl = ylim;
set(gca,'xtick',[0 10 20])
set(gca,'ytick',[yl(1) yl(end)],'yticklabel',{'0','1'});
box off
figure(8)
ind = find(F>median(F));
y = medfilt1((abs(diff(LP))),15);
mean_bin_plot(F(ind),y(ind),11)

box off
ylabel('Lickport velocity (AU)')
ylabel('Lickport velocity (AU)');
xlabel('Conditioned neuron activity \DeltaF/F')