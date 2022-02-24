function [S,DDD,SEQ]=deltaWij(new,old,shifts)
%%
for session = 1:2;
    if session == 1;
        dat = old;
    else
        dat = new;
    end
    file = [dat.folder,char(dat.siFiles{dat.currentPlane}{1})];
    [hMroiRoiGroup hStimRoiGroups] = scanimage.util.readTiffRoiData(file);
    header = scanimage.util.opentif(file);
    seq = header.SI.hPhotostim.sequenceSelectedStimuli;
    seq = repmat(seq,1,10);
    if session == 1
        seq = seq(header.SI.hPhotostim.sequencePosition+shifts(1):end);
    else
        seq = seq(header.SI.hPhotostim.sequencePosition+shifts(2):end);
    end
    seq = seq(1:length(dat.siFiles{dat.currentPlane})-1);
    dat.stimGroup = hStimRoiGroups;
    
    base = dat.bases{dat.currentPlane};
    inds = cell2mat(cellfun(@(x) ~isempty(strfind(x,base)),...
        dat.intensityFile,'uni',0));
    inds = (find(inds==1));
    clear Fstim
    fun = @(xx) cell2mat(cellfun(@(x) x',xx,'uni',0));    
    for i = 1:length(dat.roi);
%         bl = prctile(fun([dat.roi(i).intensity]),20);
        bl = std(fun([dat.roi(i).intensity]));
        for j = 1:length(inds)-1;
            if j > 1
                a = dat.roi(i).intensity{inds(j-1)}(end-10:end);
            else
                a = nan(11,1);
            end
            Fstim(:,i,j) = [a; dat.roi(i).intensity{inds(j)}(1:20)];
        end
        siOffset = dat.siHeader.SI.hChannels.channelOffset(1);
        a = Fstim(:,i,:) - siOffset;
%         bl = nanstd(a(:));
%         bl = nanmean(a(:));
        bl = prctile(a(:),20);
        Fstim(:,i,:) = (Fstim(:,i,:)-bl)/bl;
    end
    Fs{session} = Fstim;
    SEQ{session} = seq;
    Gx = [];Gy = [];resp = [];x=[];y=[];
    clear ff ddd
    for si = 1:length(dat.stimGroup)
        ind = find(seq==si);
        slm = hStimRoiGroups(si).rois(2).scanfields.slmPattern;
        sg = units_to_pixels(hStimRoiGroups(si).rois(2).scanfields,dat.siHeader,dat.dim);
        pix = sg.SLM_pix;
        galvo = sg.centerXY_pix;
        clear XY distance
        for i = 1:length(dat.roi);
            XY(i,:) = dat.roi(i).centroid;
        end
        for cl = 1:length(dat.roi);
            minDist(cl) = min(sqrt(sum((bsxfun(@minus,pix,XY(cl,:)')).^2,1)));
            gDist(cl) = min(sqrt(sum((bsxfun(@minus,galvo,XY(cl,:)')).^2,1)));
        end
        clf
        f = nanmean(Fstim(:,:,ind),3);
        del = nanmean(f(16:21,:)) - nanmean(f(1:8,:));
        
        a = Fstim(:,:,ind);
        aft = squeeze(nanmean(a(15:21,:,:))-nanmean(a(1:8,:,:)));
        [h,p] = ttest(aft');
        P(:,si) = p;
              
        delta_activity(:,:,si,session) = f;
        S{si,session} = Fstim(:,:,ind);
        ddd(:,si) = minDist;
        dddd(:,si,session) = gDist;
        DDD{session} =ddd;
    end
end
