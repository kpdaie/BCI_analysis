#%%
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from tqdm import tqdm
from BCI_analysis.io_bci.io_suite2p import sessionwise_to_trialwise, sessionwise_to_trialwise_simple
from BCI_analysis.pipeline.pipeline_videography import extract_motion_energy_from_session
try:
    import hdfdict # 
except:
    print('hdfdict not installed (for facerhythm) - skipped')

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
    try:
        bodyparts = list(dlc_data.columns.levels[0])
        dim_list = ['x','y']
        cols_new = []
        for function in functions:
            for col in dlc_data.columns:
                cols_new.append((col[0],function,col[1]))
    except: #facerhythm
        bodyparts = list(dlc_data.columns)
        dim_list = ['x']
        cols_new = []
        for function in functions:
            for col in dlc_data.columns:
                cols_new.append((col,function,'x'))
    df2 = pd.DataFrame(0, index=range(target_length), columns=cols_new)#dlc_data.columns)
    
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
            for dim in dim_list:

                if (bp,dim) in list(dlc_data.keys().values):
                    df_bp = dlc_data[bp][dim]
                elif bp in list(dlc_data.keys().values): # faceRhythm
                    df_bp = dlc_data[bp]
                else:
                    continue

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
                pad_width = target_length - dlc_data.shape[0]%target_length
                if pad_width == target_length:
                    pad_width = 0
                step_size = (len(f_sd) + pad_width)//target_length
                if pad_width>0:
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
                     motion_energy_base_dir,
                     raw_video_path,
                     mouse = "BCI_26",
                     FOV = "FOV_04",
                     camera = "side",
                     session = "041022",
                     sampling = "up",
                     functions = ['mean','diff'],
                     function_window = 20, 
                     convolve_tau = 0,
                     plot=False,
                     overwrite=False,
                     use_provided_data=False,
                     source_data = None,
                     add_motion_energy = False,
                     face_rhythm_base_dir = None,
                     use_face_rhythm=False,
                     match_with_face_rhythm = False):
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
    only_up_to_reward = True
    if use_face_rhythm or match_with_face_rhythm:
        with open(os.path.join(face_rhythm_base_dir,mouse,session,"run_info.json")) as f:
            face_rhythm_metadata = json.load(f)
        face_rhythm_file_list_ = face_rhythm_metadata['Dataset_videos']['metadata']['paths_videos'][:face_rhythm_metadata['TCA']['num_dictElements']]
        face_rhythm_file_list = []
        for file_path_now_ in face_rhythm_file_list_:
            face_rhythm_file_list.append(os.path.split(file_path_now_)[-1])
        face_rhythm_file_list = np.asarray(face_rhythm_file_list)
        data_facerhythm  = hdfdict.load(os.path.join(face_rhythm_base_dir,mouse,session,'analysis_files/TCA.h5'))
        facerhythm_trial_names = list(data_facerhythm['factors_rearranged']['0']['trials'].data.keys())
        trial_data_facerhythm = data_facerhythm['factors_rearranged']['0']['trials'].data
        num_factors = data_facerhythm['factors']['0']['(xy points)'].shape[1]#.data.keys()
        factor_names = []
        for i_ in range(num_factors):
            factor_names.append('Factor {}'.format(i_))
        
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
        reward_trialwise = sessionwise_to_trialwise_simple(source_data['reward'][:,np.newaxis].T, source_data['trial_start'],max_frames = 'all')
        print(len(F_trialwise))
        print(F_trialwise[0].shape)
        if len(behavior_movie_names) != sum(source_data['trial_start']):
            #asdas
            print('unequal trial number, aligning trials - ')
            dt_ca = np.diff(np.where(source_data['trial_start'])[0])*source_data['si']
            helper_td = np.vectorize(lambda x: x.total_seconds())
            dt_behav = helper_td(np.diff(bpod_data['trial_start_times']))

            offsets = range(len(dt_behav)-len(dt_ca)+1)
            errors = []
            for offset in offsets:
                errors.append(np.abs(np.sum(dt_ca-dt_behav[offset:len(dt_ca)+offset])))
            offset = np.argmin(errors)
            print('offset is {} with an error of {} ms'.format(offset,np.round(errors[offset]*1000,3)))
            needed_indices = np.arange(offset,len(dt_ca)+offset+1)
            
            bpod_data_old = bpod_data.copy()
            trial_n_old = len(bpod_data_old['go_cue_times'])
            for k in bpod_data.keys():
                if len(bpod_data[k]) == trial_n_old:
                    bpod_data[k] = np.asarray(bpod_data_old[k])[needed_indices]
                else:
                    print([k,len(bpod_data[k])])
            
            files_with_movies = []
            behavior_movie_names = bpod_data['behavior_movie_name_list']
            for i, k in enumerate(bpod_data['scanimage_file_names']):
                # print(k)
                if str(k) == 'no movie for this trial':
                    files_with_movies.append(False)
                else:
                    files_with_movies.append(True)                        
            behavior_movie_names = behavior_movie_names[files_with_movies]
        
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
        reward_trace = np.zeros(F.shape[1])
        trial_start_trace = np.zeros(F.shape[1])
        trial_start_frame = np.asarray([0] + np.cumsum(ca_data['all_si_frame_nums']).tolist())
        counter = 0
        for i, filename in enumerate(ca_data['all_si_filenames']):
            if not filename in ca_data['closed_loop_filenames']:
                continue
            start_frame = trial_start_frame[i]
            trial_start_trace[start_frame] = 1
            if len(ca_data['reward_times'][counter]) > 0:
                start_frame += int(ca_data['reward_times'][counter]*ca_data['sampling_rate'])
                reward_trace[start_frame] = 1
            counter += 1
        reward_trialwise = sessionwise_to_trialwise_simple(reward_trace[:,np.newaxis].T, trial_start_trace,max_frames = 'all')
        

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
        if only_up_to_reward:
            rew_trial = reward_trialwise[i].flatten()
            if any(rew_trial>0):
               # print('only used {} frames out of {} due to reward'.format(np.argmax(rew_trial),F_trial.shape[1]))
                F_trial = F_trial[:,:np.argmax(rew_trial)]
                
            

        trial_csv = [k for k in next(os.walk(dlc_folder))[2] if k.startswith(trial_id) and k.endswith("csv")][0]
        trial_json = [k for k in next(os.walk(dlc_folder))[2] if k.startswith(trial_id) and k.endswith("json")][0]
        
        
        frame_times_dlc = np.asarray(trial_metadata['frame_times'])
        
        if use_face_rhythm: ## TODO ## HARD-CODED downsampling that Rich used for the pilot data
            frame_times_dlc = frame_times_dlc[::20] 
            
        frame_times_ca = np.arange(0, F_trial.shape[1], 1, dtype=float)/fs



        dlc_trial = pd.read_csv(os.path.join(dlc_folder, trial_csv), header=[1,2], index_col=0)
        if add_motion_energy:
            motion_energy_folder = os.path.join(motion_energy_base_dir, camera, dlc_file_name[0], dlc_file_name[1])
            motion_energy_fname = os.path.join(motion_energy_folder, trial_id+".npy")
            try:
                motion_energy_dict = np.load(motion_energy_fname,allow_pickle = True).tolist()
            except:
                print('extracting motion energy')
                extract_motion_energy_from_session(bpod_path,
                                                  dlc_base_dir,
                                                  raw_video_path,
                                                  motion_energy_base_dir,
                                                  mouse,
                                                  FOV,
                                                  session,
                                                  overwrite=False)
                motion_energy_dict = np.load(motion_energy_fname,allow_pickle = True).tolist()
            for roi in motion_energy_dict['motion_energy_traces'].keys():
                dlc_trial[roi,'x'] = np.concatenate([[motion_energy_dict['motion_energy_traces'][roi][0]],motion_energy_dict['motion_energy_traces'][roi]])
            
            
        if dlc_trial.shape[0] == 0 or F_trial.shape[1] == 0:
            # print(dlc_trial.shape, F_trial.shape)
            continue
        
        if frame_times_dlc[-1] > frame_times_ca[-1]:
            closest_id_dlc = np.argmin(np.abs(frame_times_dlc - frame_times_ca[-1]))
            #print(f"offset shape = {dlc_trial.shape[0] - closest_id}")
            dlc_trial = dlc_trial[:closest_id_dlc]
            frame_times_dlc = frame_times_dlc[:closest_id_dlc]
        else:
            closest_id_dlc = -1
            

        if frame_times_dlc[-1] < frame_times_ca[-1]:
            closest_id = np.argmin(np.abs(frame_times_ca - frame_times_dlc[-1]))
            #print(f"offset shape = {F_trial.shape[1] - closest_id}")
            F_trial = F_trial[:, :closest_id]
            frame_times_ca = frame_times_ca[:closest_id]
        
        if sampling=='up':
            F_trial = interpolate_ca_data(dlc_trial, F_trial, plot=plot)
        if sampling=='down': # TODO have option to have both position and speed
            if use_face_rhythm:
                # print(face_rhythm_file_list == dlc_file_name[-1])
                # print(face_rhythm_file_list)
                # print(dlc_file_name[-1])
                if any(face_rhythm_file_list == dlc_file_name[-1].strip("'")):
                    trial_index_fr = np.where(face_rhythm_file_list == dlc_file_name[-1].strip("'"))[0][0]
                    dlc_trial_facerhythm = pd.DataFrame(data = trial_data_facerhythm[facerhythm_trial_names[trial_index_fr]][:],columns = factor_names)
                    if closest_id_dlc>0:
                        dlc_trial_facerhythm = dlc_trial_facerhythm[:closest_id_dlc]
                    needed_indices = []
                    for t in frame_times_ca: # TODO # downsample facerhythm here.. it was downsampled in a stupid way so I have to do a stupid fix
                        needed_indices.append(np.argmin(np.abs(t-frame_times_dlc)))
                    dlc_trial_facerhythm = dlc_trial_facerhythm.iloc[needed_indices]
                    dlc_trial = collapse_dlc_data(dlc_trial_facerhythm, 
                                              np.nanmedian(np.diff(frame_times_dlc)),#
                                              target_length=F_trial.shape[1], 
                                              functions = functions,
                                              convolve_tau = convolve_tau,
                                              window=function_window)    
                    dlc_data = pd.concat([dlc_data, dlc_trial], ignore_index=True) 
                    trial_start_indices.append(len(dlc_data))
                                        
                                        
                else:
                    print('facerhythm trial not found, skipping {}'.format(dlc_file_name[-1].strip("'")))
                    continue
                    
                
                
                
                
            else:
                if match_with_face_rhythm:
                    if not any(face_rhythm_file_list == dlc_file_name[-1].strip("'")):
                        print('facerhythm trial not found, skipping {}'.format(dlc_file_name[-1].strip("'")))
                        continue
                        
                    
                dlc_trial = collapse_dlc_data(dlc_trial, 
                                              np.nanmedian(np.diff(frame_times_dlc)),
                                              target_length=F_trial.shape[1], 
                                              functions = functions,
                                              convolve_tau = convolve_tau,
                                              window=function_window)
            
            
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
   # print(trial_start_indices)                
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