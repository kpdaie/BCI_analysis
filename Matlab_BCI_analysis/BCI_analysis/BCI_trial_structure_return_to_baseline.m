
clf
t = 0:.01:120;
stimTimes = [];
ftimes    = [1:.01:1.1 4:.01:4.1 21:.05:22 16:.05:16.4 37:.2:48 60:.05:62];
F = ismember(t,ftimes);
T = ismember(t,stimTimes);
F = conv(F,exp(-t/.3));F = F(1:length(t));
ker = conv(T,exp(-t/.003));ker = ker(1:length(t));
vel = F;
strt = 800000000;
pos = 0;
plot(F)
lastRew = 0;
lastBl  = 8000;
BCION   = 1;
for i = 1:length(t)
    if i - lastBl > 200;
        BCION = 1;
    else
        BCION = 0;
    end
    if BCION == 1
        pos(i+1) = pos(i) + vel(i)/500;
    end
end
        