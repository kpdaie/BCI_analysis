for i = 1:20;
    subplot(211);
    learning_map_rois(dat,-log(p1(:,i)).*sign(MN1(:,i)),[log(.001) -log(.001)],0)
    slm_show(old,old.currentPlane,i);
    title('-log(p)')
    
    subplot(212);
    learning_map_rois(dat,-log(p2(:,i)).*sign(MN2(:,i)),[log(.001) -log(.001)],0)
    slm_show(old,old.currentPlane,i,1);
    colormap(bluewhitered);
end