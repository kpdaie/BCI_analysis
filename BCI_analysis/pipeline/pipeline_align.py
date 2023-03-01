#%%
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from tqdm import tqdm
from BCI_analysis.io_bci.io_suite2p import sessionwise_to_trialwise, sessionwise_to_trialwise_simple
import json

def collapse_dlc_data(dlc_data: pd.DataFrame, 
                      sample_interval,
                      target_length: int=0, 
                      window=100, 
                      functions=['std'], #std,diff,mean
                      convolve_tau = 0,
                      mode='edge'):
    """
    # TODO - add just position with downsampling
    This function takes the dlc_data and converts the shape to target_length, 
    1. Calculates a moving standard deviation with a window
    2. Downsample
    ----------
    dlc_data: pd.Dataframe
        DeepLabCut data for a neuron
    target_length: int
        Length to convert the array size to
    mode: string
        how to pad        

    Returns
    -------
    dataframe: pd.Dataframe
        dataframe of the desired target_length
    """
    pad_width = target_length - dlc_data.shape[0]%target_length
    cols_new = []
    for function in functions:
        for col in dlc_data.columns:
            cols_new.append((col[0],function,col[1]))
    df2 = pd.DataFrame(0, index=range(target_length), columns=cols_new)#dlc_data.columns)
    bodyparts = list(dlc_data.columns.levels[0])
    
    calcium_transient_t = np.arange(0,1,sample_interval)# ms 
    if convolve_tau>0:
        amplitude = 1
        #rise = 1- np.exp(calcium_transient_t/tau_rise*-1)
        decay = np.exp(calcium_transient_t/-convolve_tau)
        calcium_transient = decay #rise*decay
        calcium_transient = amplitude*calcium_transient/np.max(calcium_transient)
        calcium_transient = np.concatenate([np.zeros_like(calcium_transient),calcium_transient])
    for function in functions:
        for bp in bodyparts:
            for dim in ["x", "y"]:
                df_bp = dlc_data[bp][dim]
                if function == 'std':
                    f_sd = df_bp.rolling(window=window).std()
                elif function == 'mean':
                    f_sd = df_bp.rolling(window=window).mean()
                elif function == 'diff': # absolute speed averaged in a window
                    f_sd = pd.Series(np.concatenate([[0],np.diff(df_bp)])).rolling(window=window).mean().abs()
                elif function == 'diff_signed': # absolute speed averaged in a window
                    f_sd = pd.Series(np.concatenate([[0],np.diff(df_bp)])).rolling(window=window).mean()#.abs()
                elif function == 'raw':
                    f_sd = df_bp
                pad_width = target_length - len(f_sd)%target_length
                step_size = (len(f_sd) + pad_width)//target_length
                if mode=='zero':
                    f_sd =  f_sd.append(pd.Series([0]*pad_width))
                elif mode=='nan':
                    f_sd =  f_sd.append(pd.Series([np.nan]*pad_width))
                elif mode=='edge':
                    f_sd =  pd.concat([f_sd,pd.Series([f_sd[-100:].mean()]*pad_width)])#f_sd.append(pd.Series([f_sd[-100:].mean()]*pad_width))


                f_sd.fillna(inplace=True, method='bfill')
                len_orig = len(f_sd)
                if convolve_tau>0: #(function != 'raw') and (
                    if len(calcium_transient)>len(f_sd): # double tapering to avoid edge effects
                        f_sd_ = pd.Series(np.convolve(np.concatenate([f_sd,f_sd[::-1],f_sd,f_sd[::-1],f_sd]),calcium_transient,'same'))
                        f_sd = f_sd_[len(f_sd)*2:len(f_sd)*2+len_orig]
                    else:
                        f_sd_ = pd.Series(np.convolve(np.concatenate([f_sd[::-1],f_sd,f_sd[::-1]]),calcium_transient,'same'))
                        f_sd = f_sd_[len(f_sd):len(f_sd)+len_orig]

                f_sd_downsample = f_sd.iloc[::step_size]
                df2[bp, function,dim] = f_sd_downsample.values
                assert target_length == len(f_sd_downsample)
            #asdasd
    return df2

def interpolate_ca_data(dlc_trial, F_trial, plot=False):
    """
    This function takes the dlc data for a trial and interpolates the data to the length of flouroscence trace
    ----------
    dlc_data: pd.Dataframe
        DeepLabCut data for a neuron
    F_trial: np.array
        flouroscence trace for a trial
    plot: bool 

    Returns
    -------
    F_ret: np.array
        flouroscence trace of the desired target length
    """
    t = np.linspace(0, dlc_trial.shape[0], F_trial.shape[1])
    tnew = np.arange(0, dlc_trial.shape[0])
    F_ret = np.zeros((F_trial.shape[0], dlc_trial.shape[0]))
    for i in range(F_trial.shape[0]):
        y = F_trial[i, :]
        f = interpolate.interp1d(t, y)
        ynew = f(tnew)
        F_ret[i, :] = ynew
    if plot:
        plt.plot(t, y, 'o', tnew, ynew, '-')
        plt.show()
    return F_ret

def get_aligned_data(suite2p_path,
                     dlc_base_dir,
                     bpod_path,
                     sessionwise_data_path,
                     aligned_data_path,
                     mouse = "BCI_26",
                     FOV = "FOV_04",
                     camera = "side",
                     session = "041022",
                     sampling = "up",
                     functions = ['mean','diff'],
                     convolve_tau = 0,
                     plot=False,
                     overwrite=False,
                     use_provided_data=False,
                     source_data = None):
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
    mouse = "BCI_26"
    FOV = "FOV_04"
    camera = "side"
    session = "041022"
    F_aligned, DLC_aligned = get_aligned_data(suite2p_path, dlc_base_dir, bpod_path, 
                                    sessionwise_data_path, mouse, FOV, camera, session)
    """
    os.makedirs(os.path.join(aligned_data_path, mouse), exist_ok=True)
    dict_save_path = os.path.join(aligned_data_path, mouse, f"{session}-dict_aligned-{sampling}sampled.npy")
    if os.path.isfile(dict_save_path) and (overwrite == False):
        dict_return = np.load(dict_save_path, allow_pickle=True).tolist()
        print(f"File found at {dict_save_path}")
        return dict_return

    print(f"Aligned data not found at {dict_save_path}, saving")
    bpod_filepath = os.path.join(bpod_path, mouse, session+"-bpod_zaber.npy")
    bpod_data = np.load(bpod_filepath, allow_pickle=True).tolist()

    behavior_movie_names = bpod_data['behavior_movie_name_list']
    # print(behavior_movie_names)
    # print(bpod_data['scanimage_file_names'])
    files_with_movies = []
    for i, k in enumerate(bpod_data['scanimage_file_names']):
        # print(k)
        if str(k) == 'no movie for this trial':
            files_with_movies.append(False)
        else:
            files_with_movies.append(True)                        
    behavior_movie_names = behavior_movie_names[files_with_movies]

    #trial_start_times = bpod_data['trial_start_times']
    if use_provided_data:
        F = source_data['F']
        fs = 1/source_data['si']
        lick_times = []
        reward_times = []
        trial_times = []
        cn = source_data['cn']
        
        print('bpod trials: {}'.format(len(behavior_movie_names)))
        print('imaging trials: {}'.format(sum(source_data['trial_start'])))
        F_trialwise = sessionwise_to_trialwise_simple(F, source_data['trial_start'],max_frames = 'all')
        print(len(F_trialwise))
        print(F_trialwise[0].shape)
        if len(behavior_movie_names) != sum(source_data['trial_start']):
            asdas
            print('unequal trial number, aborting')
            # here assume that trials were skipped from the beginning and/or end
            # so instead of aborting, subtract the trial lengths from each other while
            # pushing the trials in time, and find the one that has the minimum absolute value - that offset is the winner
            return None
        
    else:
        ca_data = np.load(os.path.join(sessionwise_data_path, mouse, mouse+"-"+session+"-"+FOV+".npy"), allow_pickle=True).tolist()
        F = ca_data['F_sessionwise']
        fs = ca_data['sampling_rate']
        lick_times = ca_data['lick_times']
        reward_times = ca_data['reward_times']
        trial_times = ca_data["trial_times"]
        cn = ca_data["cn"][0]
        if cn is None:
            print(f"{session}: CN is None, assigning value 0")
            cn = 0

        with open(os.path.join(suite2p_path, mouse, FOV, session, "filelist.json")) as f:
            filelist = json.load(f)


        F_trialwise = sessionwise_to_trialwise(F, ca_data['all_si_filenames'], ca_data['closed_loop_filenames'], 
                ca_data['all_si_frame_nums'], ca_data['sampling_rate'], align_on="trial_start", max_frames="all")
        

    F_aligned = []
    lt = []
    rt = []
    tt = []
    dlc_data = None
    trials_taken = []
    trial_start_indices = []
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
        F_trial = F_trialwise[i].T
        # print(F_trial.shape)

        trial_csv = [k for k in next(os.walk(dlc_folder))[2] if k.startswith(trial_id) and k.endswith("csv")][0]
        trial_json = [k for k in next(os.walk(dlc_folder))[2] if k.startswith(trial_id) and k.endswith("json")][0]

        frame_times_dlc = np.asarray(trial_metadata['frame_times'])
        frame_times_ca = np.arange(0, F_trial.shape[1], 1, dtype=float)/fs



        dlc_trial = pd.read_csv(os.path.join(dlc_folder, trial_csv), header=[1,2], index_col=0)
        if dlc_trial.shape[0] == 0 or F_trial.shape[1] == 0:
            # print(dlc_trial.shape, F_trial.shape)
            continue
    
        if frame_times_dlc[-1] > frame_times_ca[-1]:
            closest_id = np.argmin(np.abs(frame_times_dlc - frame_times_ca[-1]))
            #print(f"offset shape = {dlc_trial.shape[0] - closest_id}")
            dlc_trial = dlc_trial[:closest_id]
            frame_times_dlc = frame_times_dlc[:closest_id]

        if frame_times_dlc[-1] < frame_times_ca[-1]:
            closest_id = np.argmin(np.abs(frame_times_ca - frame_times_dlc[-1]))
            #print(f"offset shape = {F_trial.shape[1] - closest_id}")
            F_trial = F_trial[:, :closest_id]
        
        if sampling=='up':
            F_trial = interpolate_ca_data(dlc_trial, F_trial, plot=plot)
        if sampling=='down': # TODO have option to have both position and speed
            dlc_trial = collapse_dlc_data(dlc_trial, 
                                          np.nanmedian(np.diff(frame_times_dlc)),
                                          target_length=F_trial.shape[1], 
                                          functions = functions,
                                          convolve_tau = convolve_tau,
                                          window=20)
            
            
            dlc_data = pd.concat([dlc_data, dlc_trial], ignore_index=True) 
            trial_start_indices.append(len(dlc_data))
        
        
        F_aligned.append(F_trial)
        #print(F_trial.shape)
        trials_taken.append(i)
        try:
            lt.append(list((lick_times[i])*(dlc_trial.shape[0]/trial_times[i])))
            rt.append((reward_times[i])*(dlc_trial.shape[0]/trial_times[i]))
            tt.append(trial_times[i])
        except:
            pass
    print(trial_start_indices)                
    if len(F_aligned) == 0:
        print(f"No data found, session {session}")
        return None
    F_aligned_s = np.hstack(F_aligned)

    sd_list = np.nanstd(F_aligned_s, axis=1).reshape(-1, 1)
    dff_aligned = []
    for i, F_trial in enumerate(F_aligned):
        dff_aligned.append((F_trial - sd_list)/sd_list)
    dff_aligned_s = np.hstack(dff_aligned)
    
    # # baseline subtraction: dff = dff - dff[:1000]
    # baseline_sub = np.nanmean(dff_aligned_s[:, :1000], axis=1).reshape(-1, 1)
    # for i, dff_trial in enumerate(dff_aligned):
    #     dff_aligned[i] = dff_trial - baseline_sub

    

    dict_return = {
            "F_aligned": F_aligned,
            "DLC_aligned": dlc_data.to_dict(),
            "dff_aligned": dff_aligned,
            "lick_times_aligned": lt,
            "reward_times_aligned": rt,
            "trial_times_aligned": tt,
            "cn": cn,
            "trials_taken": trials_taken,
            "trial_start_indices":trial_start_indices
            } 
    np.save(dict_save_path, dict_return)
    return dict_return