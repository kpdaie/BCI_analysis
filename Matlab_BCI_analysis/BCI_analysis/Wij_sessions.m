%% BCI19 control 2 days no training
clear
old = load('F:\BCI2\BCI19\110921\session_110921_analyzed_dat_small__slm322110921.mat');
new = load('F:\BCI2\BCI19\111021\session_111021_analyzed_dat_small__slm3111021.mat');
shifts = [0 0];
%% BCI22 two days 20 groups
clear
new = load('F:\BCI2\BCI22\111321\session_111321_analyzed_dat_small_neuron8_2111321.mat');
old = load('F:\BCI2\BCI22\111121\session_111121_analyzed_dat_small_neuron1_222111121.mat');
old.currentPlane=2;
new.currentPlane = 5;
shifts = [1 0];
%%
clear
dat = load('F:\BCI2\BCI21\111121\session_111121_analyzed_dat_small__slm2222111121.mat');
new = dat;
old = dat;
new.currentPlane = 2;
old.currentPlane = 1;
shifts = [0 1];
%% 
clear
new = load('F:\BCI2\BCI21\111121\session_111121_analyzed_dat_small__slm2222111121.mat');
old =load('F:\BCI2\BCI21\111021\session_111021_analyzed_dat_small_neuron1111021.mat');
new.currentPlane = 2;
old.currentPlane = 2;
shifts = [0 1];
%% BCI 19 same 20 groups each day with training in between
clear
new = load('H:\BCI19\102521\session_102521_analyzed_dat_small_neuron1102621.mat');
old = load('H:\BCI19\102221\session_102221_analyzed_dat_small_slm22102521.mat');
new.folder = 'H:\BCI19\102521\';
old.folder = 'H:\BCI19\102221\';
dat = new;[dat.conditioned_coordinates,dat.conditioned_neuron] = manual_conditioned_neuron_coords(dat,1);new = dat;
new.currentPlane = 4;
old.currentPlane = 4;
shifts = [0 0];
%% BCI 22 same before and after training same day different groups
clear
dat = load('F:\BCI2\BCI22\111121\session_111121_analyzed_dat_small_neuron1_222111121.mat');
new = dat;
old = dat;
new.currentPlane = 2;
old.currentPlane = 1;
shifts = [1 1];
%% BCI21
clear
new=load('F:\BCI2\BCI21\111621\session_111621_analyzed_dat_small_slmPost111721.mat');
old = load('F:\BCI2\BCI21\111521\session_111521_analyzed_dat_small_neuron3111521.mat');
old.currentPlane = 3;
new.currentPlane = 2;
shifts = [0 0];
%% BCI19 one day many trials
clear
new = load('F:\BCI2\BCI19\120121\session_120121_analyzed_dat_small__slm21201211546.mat');old = new;
shifts = [0 0];
%%
close all
[x1,y1,id1,g1,e1] = effective_connections_vs_space(old,old.currentPlane,shifts(1));
[x2,y2,id2,g2,e2] = effective_connections_vs_space(new,new.currentPlane,shifts(2));
%%
[S,DDD,SEQ]=deltaWij(new,old,shifts);