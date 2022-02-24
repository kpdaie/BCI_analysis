function show_conditioned_neuron(varagin)
global dat
% figure(3476);clf
if nargin == 0
    scl = 99.95;
else
    scl = varagin(1);
end
imagesc(dat.IM,[0 prctile(dat.IM(:),scl)]);
colormap(gray);
hold on;
plot(dat.conditioned_coordinates(1),dat.conditioned_coordinates(2),'ro','markersize',20)
set(gca,'units','normalized','position',[0 0 1 1]);
