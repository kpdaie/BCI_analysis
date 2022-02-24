function find_a_conditioned_neuron(number,dat,F,tsta,df,type);
figure(899);
% set(gcf,'position',[ 7.0729    1.2292   11.8021   10.1979]);
set(gcf,'color','w')
clf
type
marg = .2;
if ~exist('F')
    [df,dist,F,epoch,tsta,raw] = BCI_dat_extract(dat);
end
n = length(dat.roi);

p = nanmean(F(1:200,:,:),3);
p = p - repmat(mean(p(1:20,:)),size(p,1),1);
[a,b] = sort(sum(p),'ascend');
if length(number) == 1
    if exist('type')
        if strcmp(type,'down');
            [a,b] = sort(sum(p(40:100,:)),'ascend');
        elseif strcmp(type,'up');
            [a,b] = sort(sum(p(40:100,:)),'descend');
        elseif strcmp(type,'none');
            [a,b] = sort(sum(abs(p(40:100,:))),'ascend');
        elseif strcmp(type,'bright');
            [a,b] = sort(sum(df),'descend');
        elseif strcmp(type,'big');
            [a,b] = sort(prctile(df,99.9),'descend');
        end
    end
    b = b(1:number);
else
    b = number;
    number = length(b);
end
bs = ceil(sqrt(number));
for i = 1:number;
    
        KDsubplot(bs,bs*2,(i)*2,marg);
    win = 30;
    x = dat.roi(b(i)).centroid;
    imagesc(dat.IM,[0 prctile(dat.IM(:),99.9)]);hold on;
%     showROIsPatch(gcf,'r',dat.roi,b(i),0)
    plot(x(1),x(2),'ro','markersize',40)
    xlim([x(1)-win x(1)+win]);
    ylim([x(2)-win,x(2)+win]);
    set(gca,'visible','off')
    text(max(xlim)-30,min(ylim)+10,num2str(b(i)),'color','w','fontsize',16);
    
    
    KDsubplot(bs,bs*2,(i-1)*2 +1,marg);
    plot(tsta(1:size(p,1)),p(:,b(i)),'k');box off
    set(gca,'visible','off')
    a = get(gca,'position');
    a(2) = a(2)+a(4)/2;
    a(end) = a(end)/2;
    
    set(gca,'position',a);

    a = df(:,b(i));
    ind = find(diff(a)>std(a)*3);
    ind(ind>(length(a)-300))=[];
    ind(ind<50)=[];
    ind = repmat(ind',240,1) + repmat((1:240)'-40,1,length(ind));
    KDsubplot(bs,bs*2,(i-1)*2 +1,marg);
    plot(tsta(1:240),mean(a(ind),2))
    set(gca,'visible','off')
    a = get(gca,'position');
    a(end) = a(end)/2;
    set(gca,'position',a);
    

    
    
end
colormap(gray)

