#%%
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from tqdm import tqdm

def collapse_dlc_data(dlc_data: pd.DataFrame, target_length=None, mode='edge'):
    pad_width = target_length - dlc_data.shape[0]%target_length
    mean_window = (dlc_data.shape[0] + pad_width)//target_length
    bodyparts = list(dlc_data.columns.levels[0])
    
    df2 = pd.DataFrame(0, index=range(target_length), columns=dlc_data.columns)
    for bp in bodyparts:
        for dim in ["x", "y"]:
            if mode=='zero':
                list_bp = list(dlc_data[bp][dim]) + [0]*pad_width
            elif mode=='edge':
                list_bp = list(dlc_data[bp][dim]) + [np.mean(dlc_data[bp][dim][-3:])]*pad_width
            df2[bp, dim] = np.asarray(list_bp).reshape(-1, mean_window).mean(axis=1)
    return df2

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

def get_aligned_data(suite2p_path,
                     dlc_base_dir,
                     bpod_path,
                     sessionwise_data_path,
                     mouse = "BCI_26",
                     FOV = "FOV_04",
                     camera = "side",
                     session = "041022"):
    """
    This script returns aligned F (raw flouroscence trace) and DLC data. 
    a. If there are multiple movie files a trial they are thrown out. 
    b. F is interpolated for each trial to match DLC data shape
    ----------
    suite2p_path : str
        path to suite2p data directory
    bpod_path: str
        path to DLC bpod file data
    sessionwise_data_path: str
        path where npy files are saved
    mouse : string
        Mouse name 
    fov: str
        Field of View
    camera: str
        #TODO: Add suport for multiple cameras
    session: str
        Session to look at

    Returns
    -------
    None:
        Saves file to save_path

    Example
    -------
    dlc_base_dir = os.path.abspath("bucket/Data/Behavior_videos/DLC_output/Bergamo-2P-Photostim/")
    bpod_path = os.path.abspath("bucket/Data/Behavior/BCI_exported/Bergamo-2P-Photostim/")
    suite2p_path = os.path.abspath("bucket/Data/Calcium_imaging/suite2p/Bergamo-2P-Photostim/")
    sessionwise_data_path = os.path.abspath("bucket/Data/Calcium_imaging/sessionwise_tba/")
    mouse = "BCI_26",
    FOV = "FOV_04",
    camera = "side",
    session = "041022"
    F_aligned, DLC_aligned = get_aligned_data(suite2p_path, dlc_base_dir, bpod_path, 
                                    sessionwise_data_path, mouse, FOV, camera, session)
    """

    bpod_filepath = os.path.join(bpod_path, mouse, session+"-bpod_zaber.npy")
    bpod_data = np.load(bpod_filepath, allow_pickle=True).tolist()
    behavior_movie_names = bpod_data['behavior_movie_name_list']
    trial_start_times = bpod_data['trial_start_times']

    ca_data = np.load(os.path.join(sessionwise_data_path, mouse, mouse+"-"+session+"-"+FOV+".npy"), allow_pickle=True).tolist()
    F = ca_data['F_sessionwise']
    fs = ca_data['sampling_rate']

    with open(os.path.join(suite2p_path, mouse, FOV, session, "filelist.json")) as f:
        filelist = json.load(f)

    cl_trial_list = [filelist['file_name_list'][i] for i in range(len(filelist['frame_num_list'])) if filelist['file_name_list'][i].startswith("neuron")]
    print(len(cl_trial_list),len(behavior_movie_names), len(trial_start_times))

    F_behavior = []
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
        dlc_trial = pd.read_csv(os.path.join(dlc_folder, trial_csv), header=[1,2], index_col=0).drop('likelihood', level=1, axis=1)
        # dlc_trial = collapse_dlc_data(dlc_trial, F_trial.shape[1], mode='edge')
        dlc_data = pd.concat([dlc_data, dlc_trial], ignore_index=True) 

        F_trial = interpolate_ca_data(dlc_trial, F_trial)
        F_behavior.append(F_trial)

    trial_lengths = [len(F_behavior[i]) for i in range(len(F_behavior))]
    F_behavior = np.hstack(F_behavior)
    return F_behavior, dlc_data, trial_lengths