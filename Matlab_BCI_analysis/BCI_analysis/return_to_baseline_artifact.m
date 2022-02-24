
f = nanmean(F(1:240,:,10:end-20),3);f = f-repmat(mean(f(1:20,:)),size(f,1),1);
fo = nanmean(Fo(1:240,:,10:end-20),3);fo = fo-repmat(mean(fo(1:20,:)),size(fo,1),1);
nt = size(F,3);
iters = 20;
clear shuff;
cn = dat.conditioned_neuron;
a = dfo(2:end,cn);
b = dfo(1:end-1,cn);
bl = prctile(a,50);
inds = find(a<bl & b>bl);
for iter = 1:iters
    for ti = 1:nt;
        if rand > .2
            ind = randsample(inds,1);
        else
            ind = randsample(1:length(a),1);
        end
        ind = ind-40:ind+200;
        ind = ind + round(rand*00);
        ind(ind>size(dfo,1)) = size(dfo,1);
        ind(ind<1) = 1;
        shuff(:,ti,iter) = dfo(ind,cn);
    end
end
clf
subplot(211);

ff = @(x) sgolayfilt(x,3,11);
% subplot(311);
plot(tsta(1:240),ff(fo(:,cn)));hold on;
ylim([-.4 .7])
xlabel('Time from trial start (s)')
ylabel('\DeltaF/F')
title(old.folder)

plot(tsta(1:240),ff(f(:,cn)));hold on;
ylim([-.4 .7])
xlabel('Time from trial start (s)')
title(dat.folder)
% subplot(313);
shuff = mean(mean(shuff,2),3);
shuff = shuff - mean(shuff(1:20));
plot(tsta(1:241),ff(shuff))
    ylim([-.3 .3])
xlim([-2 5])
legend('day 1','day 2','day 1 shuffle')

del = mean(f(40:150,:))-mean(fo(40:150,:));
del(cn) = mean(f(40:150,cn)) - mean(shuff(40:150));
subplot(212)
scatter(dist,del,'k');