# %%
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from tqdm import tqdm

sys.path.append("/home/labadmin/Github/BCI_analysis/BCI_analysis/")
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#%% 

dlc_base_dir = os.path.abspath("../../bucket/Data/Behavior_videos/DLC_output/Bergamo-2P-Photostim/")
bpod_path = os.path.abspath("../../bucket/Data/Behavior/BCI_exported/Bergamo-2P-Photostim/")
suite2p_path = os.path.abspath("../../bucket/Data/Calcium_imaging/suite2p/Bergamo-2P-Photostim/")
sessionwise_data_path = os.path.abspath("../../bucket/Data/Calcium_imaging/sessionwise_tba/")
plt_save_path = os.path.abspath("../../Plots/")

mouse = "BCI_26"
FOV = "FOV_04"
camera = "side" 
session = "041022"

def interpolate_ca_data(dlc_trial, F_trial, plot=False):
    x = np.linspace(0, dlc_trial.shape[0], F_trial.shape[1])
    xnew = np.arange(0, dlc_trial.shape[0])
    F_ret = np.zeros((F_trial.shape[0], dlc_trial.shape[0]))
    for i in range(F_trial.shape[0]):
        y = F_trial[i, :]
        f = interpolate.interp1d(x, y)
        ynew = f(xnew)
        F_ret[i, :] = ynew
    if plot:
        plt.plot(x, y, 'o', xnew, ynew, '-')
        plt.show()
    return F_ret

bpod_filepath = os.path.join(bpod_path, mouse, session+"-bpod_zaber.npy")
bpod_data = np.load(bpod_filepath, allow_pickle=True).tolist()
behavior_movie_names = bpod_data['behavior_movie_name_list'][:-1]
trial_start_times = bpod_data['trial_start_times']

ca_data = np.load(os.path.join(sessionwise_data_path, mouse, mouse+"-"+session+"-"+FOV+".npy"), allow_pickle=True).tolist()
F = ca_data['F_sessionwise']
fs = ca_data['sampling_rate']
lick_times = ca_data['lick_times']
reward_times = ca_data['reward_times']
trial_times = ca_data['trial_times']

with open(os.path.join(suite2p_path, mouse, FOV, session, "filelist.json")) as f:
    filelist = json.load(f)

cl_trial_list = [filelist['file_name_list'][i] for i in range(len(filelist['frame_num_list'])) if filelist['file_name_list'][i].startswith("neuron")]
print(len(cl_trial_list),len(behavior_movie_names), len(trial_start_times))

#%%
F_behavior = []
dff_behavior = []
lt = []
rt = []
ltrel = []
rtrel = []
dlc_data = None

for i, bm_name in tqdm(enumerate(behavior_movie_names)):

    if type(bm_name) == str:
        print(f"{camera} camera not found for trial {i}, skipping")
        continue
    
    camera_movies = []
    for video_file in bm_name:
        if camera in video_file: 
            camera_movies.append(video_file)
    
    if len(camera_movies) == 0:
        print(f"{camera} camera not found for trial {i}, skipping")
        continue
    elif len(camera_movies) > 1:
        print(f"Multiple {camera} camera files found for trial {i}, skipping")
        continue
    
    video_path = camera_movies[0]
    dlc_file_name = video_path[video_path.find(camera)+len(camera)+1:].split("/") #[mouse, session_id, trial_id]
    dlc_folder = os.path.join(dlc_base_dir, camera, dlc_file_name[0], dlc_file_name[1])
    trial_id = dlc_file_name[2][:-5]

    trial_json = os.path.join(dlc_folder, trial_id+".json")
    with open(trial_json) as f:
        trial_metadata = json.load(f)
    
    frame_times_rel0 = (trial_start_times[i] - trial_start_times[0]).total_seconds() + np.asarray(trial_metadata['frame_times'])
    F_trial = F[:, int(frame_times_rel0[0]*fs):int(frame_times_rel0[-1]*fs)]


    trial_csv = [k for k in next(os.walk(dlc_folder))[2] if k.startswith(trial_id) and k.endswith("csv")][0]
    # dlc_trial = pd.read_csv(os.path.join(dlc_folder, trial_csv), header=[1,2], index_col=0).drop('likelihood', level=1, axis=1)
    dlc_trial = pd.read_csv(os.path.join(dlc_folder, trial_csv), header=[1,2], index_col=0)
    # dlc_trial = collapse_dlc_data(dlc_trial, F_trial.shape[1], mode='edge')

    if i == 0:
        ltimes = list((lick_times[i])*(dlc_trial.shape[0]/trial_times[i]))
        rtimes = list((reward_times[i])*(dlc_trial.shape[0]/trial_times[i]))

        ltimes_rel = list((lick_times[i])*(dlc_trial.shape[0]/trial_times[i]))
        rtimes_rel = list((reward_times[i])*(dlc_trial.shape[0]/trial_times[i]))

    else:
        ltimes = list((lick_times[i])*(dlc_trial.shape[0]/trial_times[i]) + dlc_data.shape[0])
        rtimes = list((reward_times[i])*(dlc_trial.shape[0]/trial_times[i]) + dlc_data.shape[0])

        ltimes_rel = list((lick_times[i])*(dlc_trial.shape[0]/trial_times[i]))
        rtimes_rel = list((reward_times[i])*(dlc_trial.shape[0]/trial_times[i]))

    dlc_data = pd.concat([dlc_data, dlc_trial], ignore_index=True)
    F_trial = interpolate_ca_data(dlc_trial, F_trial)

    sd = np.nanstd(F_trial, axis=1).reshape(-1, 1)
    dff_trial = (F_trial - sd)/sd
    dff_trial = dff_trial - np.nanmean(dff_trial[:, :800], axis=1).reshape(-1, 1)

    F_behavior.append(F_trial)
    dff_behavior.append(dff_trial)
    lt.append([int(k) for k in ltimes])
    rt.append([int(k) for k in rtimes])

    ltrel.append([int(k) for k in ltimes_rel])
    rtrel.append([int(k) for k in rtimes_rel])

trial_lengths = [F_behavior[i].shape[1] for i in range(len(F_behavior))]

dff_behavior = np.hstack(dff_behavior)
F_behavior = np.hstack(F_behavior)
# dff_behavior = np.hstack(dff_behavior)

#%% 

plt.figure(figsize=(16, 10))
for i in range(len(lt)):
    plt.plot(ltrel[i], [i]*len(ltrel[i]), 'go', markersize=1.5)    
    plt.plot(rtrel[i], [i]*len(rtrel[i]), 'ro', markersize=1.5)

plt.ylabel("Trial #")
plt.xlabel("Time from trial start")
plt.show()

#%% 

def segment(arr, max=20):
    clusters = []
    eps = max
    points_sorted = np.asarray(arr)
    curr_point = points_sorted[0]
    curr_cluster = [curr_point]
    for point in points_sorted[1:]:
        if point <= curr_point + eps:
            curr_cluster.append(point)
        else:
            clusters.append(curr_cluster)
            curr_cluster = [point]
            curr_point = point
    clusters.append(curr_cluster)

    return clusters

ttip = []
lport = []
tmid = []
c_lengths = [0] + list(np.cumsum(trial_lengths))

for i in range(len(c_lengths)-2):
    k = dlc_data["TongueTip"][c_lengths[i]:c_lengths[i+1]]
    ttip.append(k[k["likelihood"] > 0.90])

    k = dlc_data["TongueMid"][c_lengths[i]:c_lengths[i+1]]
    tmid.append(k[k["likelihood"] > 0.90])

    k = dlc_data["Lickport"][c_lengths[i]:c_lengths[i+1]]
    lport.append(k)

def plot_licks(ttip, tmid, lport, trial):

    k = segment(ttip[trial].index.values, max=200)
    tongue_start_end = np.array([[k[i][0], k[i][-1]] for i in range(len(k))])

    plt.scatter(ttip[trial].index.values, ttip[trial]["x"],  marker='.', alpha=0.2, c='red')
    # plt.scatter(tmid[trial].index.values, tmid[trial]["x"],  marker='.', alpha=0.2, c='blue')

    for i in range(tongue_start_end.shape[0]):
        plt.plot([tongue_start_end[i, 0], tongue_start_end[i, 0]], [ttip[trial]["x"][tongue_start_end[i, 0]], ttip[trial]["x"][tongue_start_end[i, 1]]], '-', color='black', alpha=1)
    plt.scatter(lport[trial]["x"].index.values, lport[trial]["x"],  marker='.', alpha=0.1, c='green')
    plt.show()
    # plt.xlim([0, 500])
plot_licks(ttip, tmid, lport, 1)


ctr=0
lick_starts = []
avg = [0]*1001
for trial in range(len(ttip)):
    arr = ttip[trial].index.values
    if len(arr) == 0:
        continue
    k = segment(arr, max=200)
    tongue_start_end = np.array([k[i][0] for i in range(len(k))])
    for i in range(tongue_start_end.shape[0]):
        movement = lport[trial]["x"].loc[tongue_start_end[i] - 500: tongue_start_end[i] + 500].values
        if np.mean(movement[500:]) < 300:
            ignore = True
            continue
        if movement.shape[0] != 1001:
            continue
        
        lick_starts.append(tongue_start_end[i])
        ctr = ctr + 1

        # movement = (movement - np.mean(movement))/np.std(movement)
        avg = avg + movement.flatten()

        plt.plot(movement, alpha=0.1, color='black')
    # plt.show()
plt.plot(avg/ctr)

cn = ca_data["cn"][0]
print(ctr)
#%%
import matplotlib.colors as cl

dff_avg = np.zeros((dff_trial.shape[0], 1000, len(c_lengths)-2))
for tl in range(len(c_lengths)-2):
    dff_trial = dff_behavior[:, c_lengths[tl]:c_lengths[tl+1]]
    reward = rtrel[tl]

    if len(reward) == 0:
        print(f"No reward trial, {tl}")
        continue
    
    k = dff_trial[:, (reward[0])-500: (reward[0])+500]
    if k.shape[1] != dff_avg.shape[1]:
        print("Not found")
        continue
    dff_avg[:,:,tl] = k

dff_avg = np.mean(dff_avg, axis=-1)
means = np.mean(dff_avg[:, 500:], axis=1)
sorted_m = np.argsort(means)
plt.imshow(dff_avg[sorted_m], aspect="auto", cmap='seismic', norm=cl.TwoSlopeNorm(0))
plt.colorbar()
plt.show()

# %%
