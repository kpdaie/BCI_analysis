#%%
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# First take the session you want to look at, then bpod has bpod['behavior_movie_name_list']
# throw out all the bpod that have more than one movie_name, there seems to be an issue
# use that to align each video file to the corresponding trial in closed_loop_filenames and all_si_filenames
# This all should probably align.

dlc_base_dir = os.path.abspath("bucket/Data/Behavior_videos/DLC_output/Bergamo-2P-Photostim/")
session_bpod_file = os.path.abspath("bucket/Data/Behavior/BCI_exported/Bergamo-2P-Photostim/")
suite2p_data = os.path.abspath("bucket/Data/Calcium_imaging/suite2p/Bergamo-2P-Photostim/")
sessionwise_data_path = os.path.abspath("bucket/Data/Calcium_imaging/sessionwise_tba/")

mouse = "BCI_26"
FOV = "FOV_04"
camera = "side"
session = "041022"

bpod_filepath = os.path.join(session_bpod_file, mouse, session+"-bpod_zaber.npy")
bpod_data = np.load(bpod_filepath, allow_pickle=True).tolist()
behavior_movie_names = bpod_data['behavior_movie_name_list']
trial_start_times = bpod_data['trial_start_times']

ca_data = np.load(os.path.join(sessionwise_data_path, mouse, mouse+"-"+session+"-"+FOV+".npy"), allow_pickle=True).tolist()
F = ca_data['F_sessionwise']
fs = ca_data['sampling_rate']

with open(os.path.join(suite2p_data, mouse, FOV, session, "filelist.json")) as f:
    filelist = json.load(f)

cl_trial_list = [filelist['file_name_list'][i] for i in range(len(filelist['frame_num_list'])) if filelist['file_name_list'][i].startswith("neuron")]
print(len(cl_trial_list),len(behavior_movie_names), len(trial_start_times))

frame_times = []
F_behavior = []
dlc_data = None

for i, bm_name in enumerate(behavior_movie_names):

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
    print(trial_id)

    trial_json = os.path.join(dlc_folder, trial_id+".json")
    with open(trial_json) as f:
        trial_metadata = json.load(f)
    
    trial_csv = [k for k in next(os.walk(dlc_folder))[2] if k.startswith(trial_id) and k.endswith("csv")][0]
    dlc_data = pd.concat([dlc_data, pd.read_csv(os.path.join(dlc_folder, trial_csv), header=[1,2], index_col=0)], ignore_index=True) 
    frame_times_rel0 = (trial_start_times[i] - trial_start_times[0]).total_seconds() + np.asarray(trial_metadata['frame_times'])

    F_trial = F[:, int(frame_times_rel0[0]*fs):int(frame_times_rel0[-1]*fs)]
    F_behavior.append(F_trial)


# %%
F_behavior = np.hstack(F_behavior)
# %%

def collapse_dlc_data(dlc_dict, target_length=None):

    pad_length = target_length - dlc_dict.shape[0]%target_length
    dlc_dict
