clear
folder = 'D:\bScope\BCI_03\030621\';
wsFiles = {'neuron5_0001.h5'};
rois = readtable('D:\bScope\BCI_03\030621\neuron5_IntegrationRois_00001.csv');
wsData = ws_bitcode_append(wsFiles,folder);
%%
s = wsData.scans;
window = 2*60*wsData.header.AcquisitionSampleRate;
dt = 1/wsData.header.AcquisitionSampleRate;
t = 0:dt:(length(s)-1)*dt;
%%
near = t(find(s(:,4)>1));
near(diff(near)<.001) = [];
tone = t(find(s(:,1)>1));
tone(diff(tone)<1) = [];
far  = t(find(s(:,5)>1));
far(diff(far)<.001) = [];
lick = t(find(s(:,4)>.3));
rew  = far;
iti = 2;
far = far+iti;;
% lick(diff(lick)<.001) = [];
% lick = t(find(diff(medfilt1(s(:,6),100))>.02));
lick(diff(lick)<.1) = [];
ts = 0:.1:t(end);
ts2 = sort([ts rew tone near far lick]);
pos = 0;
rw  = 0;
tn  = 0;
lk  = 0;
for i = 1:length(ts2)-1
    pos(i+1) = pos(i) + ismember(ts2(i),near) - ismember(ts2(i),far)*100;
    pos(pos<0) = 0;
    rw(i+1) = ismember(ts2(i),rew);
    lk(i+1) = ismember(ts2(i),lick);
    tn(i+1) = ismember(ts2(i),tone);
    i/length(ts2)
end
%%
figure(2);clf
marg = [.7 .05];
KDsubplot(2.2,1.1,[1.1 1.1],marg);
fun = @(x) medfilt1(x,3);
% fun = @(x) x;
df = rois{:,5+2};
bl = prctile(df,20);
df = (df - bl)/bl;
labs=fliplr({['Conditioned',char(10),'neuron (\DeltaF/F)'],...
    ['Lickport',char(10),'position'],['Reward'],'Steps'});
colors=([0 .7 0;0 0 0;0 0 1;0 .7 0]);
plot([0 30],[0 0],'linewidth',3,'color',[1 1 1]*.8);hold on;
plot(rois{:,1}/60,fun(df),'b');
set(gca,'xticklabel',[]);
axis tight
yl1 = ylim;

KDsubplot(6.4,1.1,[5.2,1.1],marg);
plot(ts2/60,pos,'r');set(gca,'xticklabel',[]);
KDsubplot(6.4,1.1,[6.2,1.1],marg);
plot((ts2)/60,rw,'c');set(gca,'xticklabel',[]);
KDsubplot(6.4,1.1,[4.2,1.1],marg);
plot(ts2/60,lk,'m');set(gca,'xticklabel',[]);
ch = get(gcf,'Children');
I = 0;
for i = 1:length(ch);
    if strcmp(ch(i).Type,'axes')==1;
        I = I+1;
        axes(ch(i))
        a = get(ch(i),'Children');
        a(1).Color = colors(I,:);
        set(gca,'visible','off');
        xlim([8.56 9.15]);
        box off
        hold on;
        set(gca,'tickdir','out','fontsize',8);
        for j = 1:length(far);
            plot([1 1]*(tone(j))/60,ylim,'k:');
        end
        yl = ylim;
        tt = text(8.5,yl(1) + diff(yl)/2,labs{I},'color',colors(I,:));
        tt.HorizontalAlignment = 'center';
        tt.FontSize = 12;
        if I == 2;
            plot([9.1 9.1+(1/60)],[1 1]*.25,'k');  
            tt = text([9.1],[1]*.12,'1 s');tt.FontSize = 12;           
        end
    end
end
yl = ylim;
tt = text(8.582,yl(2)-5.8,['Trial',char(10),'start']);
tt.HorizontalAlignment = 'center';
tt.FontSize = 12;

KDsubplot(2.2,100,[1.1 14],[0 .05]);

cla
ylim(yl1);
plot([0 0],[0 4],'k');hold on;
plot([-.1/2 0],[0 0],'k');text(-.13,0.03,'0');
plot([-.1/2 0],[0 0]+4,'k');text(-.13,4.03,'4');
plot([-.1/2 0],[0 0]+2,'k');text(-.13,2.03,'2');
xlim([-.1 0]);
set(gca,'visible','off');