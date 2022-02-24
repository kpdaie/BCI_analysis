clear XYs XYi
clf
[hMroiRoiGroup,hStimRoiGroups,hIntegrationRoiGroup] = scanimage.util.readTiffRoiData([dat.folder,dat.siFiles{5}{1}]);
% for i = 1:length(hStimRoiGroups);
    XYs = hStimRoiGroups(1).rois(2).scanfields.slmPattern;
    offset = hStimRoiGroups(2).rois(2).scanfields.centerXY;
%     XYs = XYs(:,1:2) + repmat(offset,size(XYs,1),1);
% end
for i = 1:length(hIntegrationRoiGroup.rois);
    XYi(i,:) = hIntegrationRoiGroup.rois(i).scanfields.centerXY;
end
rect = dat.siHeader.SI.hRoiManager.imagingFovDeg;
scaling = dat.dim./range(rect);
for i = 1:size(XYi,1)
    XYi(i,:) = ((XYi(i,:) - rect(1,:)).*scaling)';    
end
for i = 1:size(XYs,1)    
    XYs(i,1:2) = ((XYs(i,1:2) - rect(1,:)).*scaling)';
end
try
    imagesc(dat.IM);colormap(gray);hold on;
end
plot(XYs(:,1),XYs(:,2),'ro','markersize',10);hold on;
scatter(XYi(:,1),XYi(:,2),'co')
plot(XYi(3,1),XYi(3,2),'mo','linewidth',3,'markersize',15)
%%
clf
pre=2181:5821;
post = 15590:21390;
x = [neuron32IntegrationRois00001{:,10}];
yy = [neuron32IntegrationRois00001{:,5}];
bl = prctile(yy,10);
    yy = (yy-bl)/bl;
T= [neuron32IntegrationRois00001{:,1}]/60;
stm = find(x>130);
stm(diff(stm)<4) = [];
stimOn = 0*x;stimOn(stm) = 1;
clear y
for i = 1:length(stm);
    y(:,i) = yy(stm(i)-90:stm(i)+150);
end
% plot(y)
d = mean(y(120:160,:)) - mean(y(1:60,:));
plot(d,'o')
subplot(311);
dt_si = 1/dat.siHeader.SI.hRoiManager.scanVolumeRate;
t = 0:dt_si:dt_si*(size(y,1)-1);
v = mean(y');v=v-v(1);
plot(t,v);hold on;
ylim([-.2 .4])
plot(xlim,xlim*0,'k:');
subplot(312);
plot(T,stimOn*6,'m');hold on;
plot(T,yy,'k');
plot(T(pre),yy(pre),'g');hold on;
plot(T(post),yy(post),'b');
xlabel('Time (min.)')
subplot(3,1,3)
amp = conv(d,ones(10,1));
plot(T(stm),amp(10:end),'o')
ylabel('Photostim. amp');
xlabel('Time (min.)')
%%
clf
for i = 3:size(neuron32IntegrationRois00001,2);
    yy = [neuron32IntegrationRois00001{:,i}];
    bl = prctile(yy,10);
    yy = (yy-bl)/bl;
    dff(:,i) = [mean(yy(pre)) mean(yy(post))];
end
plot(dff,'co-');hold on;
plot(dff(:,5),'mo-','linewidth',3);
xlim([.6 2.4]);
box off