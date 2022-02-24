

%%
clf
cn = dat.conditioned_neuron;
for i = 1:length(dat.roi);
    ind = find(ddd(i,:)>30);
    a = mean(ff(:,i,ind),3);
    del(i) = mean(a(16:20))-mean(a(1:7));
    numIn(i) = sum(P(i,ind)<.1);
end
clf
% subplot(121);

% subplot(122);
scatter(dist,del,'k')
hold on;
scatter(dist(cn),del(cn),'k','markerfacecolor','r');
% clf
% scatter(dist,numIn)
xlabel('Distance from CN');
ylabel('Summed effective connection inputs (\DeltaF/F)');
xlim([-20 1000])
%%
    
    
    
    
    
    
