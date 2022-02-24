
clf
t = 0:.01:50;
stimTimes = [6:.01:9 15:.01:19 27:.01:31 40:.01:44];
ftimes    = [1:.01:1.1 4:.01:4.1 21:.05:22 16:.2:16.4 27:.2:31 40:.05:42];
F = ismember(t,ftimes);
T = ismember(t,stimTimes);
F = conv(F,exp(-t/.3));F = F(1:length(t));
ker = conv(T,exp(-t/.003));ker = ker(1:length(t));
vel = F.*ker;
subplot(312);
plot(t,F/2+randn(1,length(t))/5,'color',[0 .7 0],'linewidth',2);
set(gca,'xtick',[0 50]);
subplot(311);
plot(t,ker,'m','linewidth',2);hold on;
plot(t,T,'k:');
set(gca,'xtick',[0 50]);
set(gca,'xtick',[0 50]);
subplot(313);
plot(t,cumsum(vel)/210,'b','linewidth',2);
set(gca,'xtick',[0 50]);
figure_finalize
% hold on;plot(t,T,'k:','linewidth',1);
%%
clf
n = 6;
in = 2;
w = zeros(n);
for i = 1:6;
    cls = randsample(1:n,in);
    w(i,cls) = 1;
end
w = w+eye(n);
w = w~=0;

t = 0:.01:2;
k = 0*t;k(20) = 1;
for i = 1:n;
    for j = 1:n
        col = 'k';
        if i == j;col = 'm';end
        KDsubplot(n,n,[i,j],[.15]);
        a = conv(k,exp(-t/(4-rand*3)));a=a(1:length(t));
        plot(t,(1-.8*rand)*a*w(i,j)+randn(1,length(t))/10,col);
        ylim([-.2 1.2]);
        set(gca,'visible','off');
    end
end




