function roi_old_vs_new(neuron_number,old);
global dat
figure(822);
clf
colormap(gray);

for i = 1:2
    a = dat;
    if i == 1;
        a = old;
    end
    subplot(1,2,i);
    win = 90;
    x = a.roi(neuron_number).centroid;
    imagesc(a.IM,[0 prctile(a.IM(:),99.9)]);
    showROIsPatch(gcf,'r',a.roi,neuron_number,0)
    xlim([x(1)-win x(1)+win]);
    ylim([x(2)-win,x(2)+win]);
    title(['neuron #',num2str(neuron_number),'    ',a.folder]);
end
