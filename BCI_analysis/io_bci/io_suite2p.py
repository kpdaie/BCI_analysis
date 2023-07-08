#%%
import numpy as np
import os
import json
from BCI_analysis.pipeline.pipeline_imaging import find_conditioned_neuron_idx
from tqdm import tqdm


def sessionwise_to_trialwise_simple(F, 
                                    trial_start,
                                    max_frames=None,
                                    frames_after=None, 
                                    frames_before=None,
                                    include_next_trial = False):
    
    start_frames = np.where(trial_start)[0]
    end_frames = np.concatenate([np.where(trial_start)[0][1:],[len(trial_start)]])
    if max_frames == "all":
        print("Since max_frames is all, this function will return a list of F trialwise as all trials have different lengths")
        F_trialwise_closed_loop = []
        for start_frame,end_frame in zip(start_frames,end_frames):
            F_trialwise_closed_loop.append(F[:, start_frame:end_frame].T)
    else:
        max_frames = frames_after + frames_before
        F_trialwise_closed_loop = np.ones((max_frames, F.shape[0], len(start_frames)))*np.nan
        counter = 0
        for start_frame,end_frame in zip(start_frames,end_frames):
            
            if include_next_trial:
                end_frame = np.min([start_frame+frames_after,F.shape[1]-1])
            #print([start_frame,end_frame])

            if end_frame - start_frame > frames_after:
                end_frame = start_frame + frames_after
            start_frame = start_frame - frames_before # taking 40 time points before trial starts
            if start_frame<0: # taking care of edge at the beginning
                missing_frames_at_beginning = np.abs(start_frame)
                start_frame = 0
            else:
                missing_frames_at_beginning = 0
            F_trialwise_closed_loop[missing_frames_at_beginning:missing_frames_at_beginning+end_frame-start_frame, :, counter] = F[:, start_frame:end_frame].T
            counter += 1

    return F_trialwise_closed_loop

def sessionwise_to_trialwise_simple_exclude_reward(F, 
                                                   trial_start,
                                                   reward_start,
                                                   max_frames=None,
                                                   frames_after=None, 
                                                   frames_before=None,
                                                   invert = False,
                                                  offset_to_keep = 0):
    
    start_frames = np.where(trial_start)[0]
    end_frames = np.concatenate([np.where(trial_start)[0][1:],[len(trial_start)]])
    reward_frames = np.where(reward_start)[0]
    if max_frames == "all":
        print("Since max_frames is all, this function will return a list of F trialwise as all trials have different lengths")
        F_trialwise_closed_loop = []
        for start_frame,end_frame in zip(start_frames,end_frames):
            if any((reward_frames>start_frame)& (reward_frames<end_frame)):
                end_frame = reward_frames[(reward_frames>start_frame)& (reward_frames<end_frame)][0]
            F_trialwise_closed_loop.append(F[:, start_frame:end_frame].T)
    else:
        max_frames = frames_after + frames_before
        F_trialwise_closed_loop = np.ones((max_frames, F.shape[0], len(start_frames)))*np.nan
        counter = 0
        for start_frame,end_frame in zip(start_frames,end_frames):
            if any((reward_frames>start_frame)& (reward_frames<end_frame)):
                if invert:
                    reward_frame = reward_frames[(reward_frames>start_frame)& (reward_frames<end_frame)][0]-start_frame
                else:
                    end_frame = reward_frames[(reward_frames>start_frame)& (reward_frames<end_frame)][0] + offset_to_keep
            elif invert:
                reward_frame = end_frame
            #print(end_frame)
            if end_frame - start_frame > frames_after:
                end_frame = start_frame + frames_after
            start_frame = start_frame - frames_before # taking 40 time points before trial starts
            if start_frame<0: # taking care of edge at the beginning
                missing_frames_at_beginning = np.abs(start_frame)
                start_frame = 0
            else:
                missing_frames_at_beginning = 0
            F_trialwise_closed_loop[missing_frames_at_beginning:missing_frames_at_beginning+end_frame-start_frame, :, counter] = F[:, start_frame:end_frame].T
            if invert:
                F_trialwise_closed_loop[frames_before:frames_before+reward_frame, :, counter] = np.nan
                
            counter += 1

    return F_trialwise_closed_loop

def sessionwise_to_trialwise(F, all_si_filenames, closed_loop_filenames, frame_num, fs, align_on = "go_cue", go_cue_times=None, reward_times=None, max_frames=None, frames_after=None, frames_before=None):
    """
    this function does not care if there is a movie for the trial or not, handle this somewhere else
    """

    if align_on == "go_cue" or align_on=="reward" or align_on=="trial_start":

        # if not (frames_after  or frames_before):
        #     print("Aligning with go_cue or trial_start requires frame numbers to take")
        #     return 
        
        filename_start_frame = np.asarray([0] + np.cumsum(frame_num).tolist())

        if max_frames == "all":
            print("Since max_frames is all, this function will return a list of F trialwise as all trials have different lengths")
            F_trialwise_closed_loop = []
            counter = 0
            for i, filename in enumerate(all_si_filenames):
                if not filename in closed_loop_filenames:
                    continue
                start_frame = filename_start_frame[i]
                end_frame = filename_start_frame[i+1]
                #print(start_frame, end_frame)

                if align_on == "go_cue":
                    if len(go_cue_times[counter]) == 0:
                        counter += 1
                        continue
                    elif np.isnan(go_cue_times[counter][0]):
                        counter += 1
                        continue
                    start_frame += int(go_cue_times[counter]*fs)
                if align_on == "reward":
                    if len(reward_times[counter]) == 0:
                        counter += 1
                        continue
                    start_frame += int(reward_times[counter]*fs)
                
                F_trialwise_closed_loop.append(F[:, start_frame:end_frame].T)
                counter += 1

            return F_trialwise_closed_loop


        else:
            needed_list = []
            max_frames = frames_after + frames_before
            F_trialwise_closed_loop = np.ones((max_frames, F.shape[0], len(closed_loop_filenames)))*np.nan
            counter = 0
            for i, filename in enumerate(all_si_filenames):
                if not filename in closed_loop_filenames:
                    continue
                start_frame = filename_start_frame[i]
                end_frame = filename_start_frame[i+1]

                if align_on == "go_cue":
                    if len(go_cue_times[counter]) == 0:
                        counter += 1
                        continue
                    elif np.isnan(go_cue_times[counter][0]):
                        counter += 1
                        continue
                    start_frame += int(go_cue_times[counter]*fs)
                if align_on == "reward":
                    if len(reward_times[counter]) == 0:
                        counter += 1
                        continue
                    start_frame += int(reward_times[counter]*fs)
                
                if end_frame - start_frame > frames_after:
                    end_frame = start_frame + frames_after
                start_frame = start_frame - frames_before # taking 40 time points before trial starts
                if start_frame<0: # taking care of edge at the beginning
                    missing_frames_at_beginning = np.abs(start_frame)
                    start_frame = 0
                else:
                    missing_frames_at_beginning = 0
                F_trialwise_closed_loop[missing_frames_at_beginning:missing_frames_at_beginning+end_frame-start_frame, :, counter] = F[:, start_frame:end_frame].T
                needed_list.append(counter)
                counter += 1
            F_trialwise_closed_loop = F_trialwise_closed_loop[:,:,needed_list]
            return F_trialwise_closed_loop





def suite2p_to_npy(suite2p_path, 
                   raw_data_path, 
                   behavior_data_path, 
                   save_path, 
                   overwrite=True, 
                   mice_name=None, 
                   fov_list=None, 
                   session_list=None,
                   max_frames = 240,
                   frames_prev_trial = 40):
    """
    This script saves the relevant data into .npy files that can be analysed later
    ----------
    suite2p_path : str
        path to suite2p data directory
    raw_data_path : str
        path to raw 2p data
    behavior_data_path: str
        path to DLC bpod file data
    save_path: str
        path where npy files should be saved
    overwrite: bool, optional
        If set to True, npy files will be overwritten
    mice_name : list/string
        List of Mouse names to convert to npy
    fov_list: list
        If you want to convert a sessions from a specific FOV only
    session_list: list
        If you want to convert specific sessions only
    max_frames: int
        Number of frames from the current trial that will appear in the trialwise matrices
    frames_prev_trial: int
        Number of frames from previous trial that will appear in the trialwise matrices

    Returns
    -------
    None:
        Saves file to save_path

    Example
    -------
    mice_name = "BCI_26"
    suite2p_path = "bucket/Data/Calcium_imaging/suite2p/Bergamo-2P-Photostim/"
    raw_data_path = "bucket/Data/Calcium_imaging/raw/Bergamo-2P-Photostim/"
    behavior_data_path = "bucket/Data/Behavior/BCI_exported/Bergamo-2P-Photostim/"
    save_path = "/home/jupyter/bucket/Data/Calcium_imaging/sessionwise_tba"
    suite2p_to_npy(suite2p_path, raw_data_path, behavior_data_path, save_path, overwrite=True, mice_name = mice_name)
    """
    #%%
    adjust_channel_offsets = False
    frames_this_trial = max_frames - frames_prev_trial
    if mice_name is None:
        mice_name = os.listdir(suite2p_path)
    if type(mice_name) == str:
        mice_name = mice_name.split()
    
    for mouse in mice_name:
        suite2p_data = os.path.join(suite2p_path, mouse)
        raw_suite2p = os.path.join(raw_data_path, mouse)
        behavior_data_path = os.path.join(behavior_data_path, mouse)
        mice_save_path = os.path.join(save_path, mouse)

        os.makedirs(mice_save_path, exist_ok=True)

        if fov_list is None:
            # fov_list = os.listdir(suite2p_data)
            fov_list = [k for k in os.listdir(suite2p_data) if k[-2:].isdigit()] # For BCI_29 there exists folders other than FOV_0x
            print(fov_list)

        for fov in fov_list:
            fov_path = os.path.join(suite2p_data, fov)
            try:
                mean_image = np.load(os.path.join(fov_path, "mean_image.npy")) #TODO the mean image should come from the session and not from the FOV
            except OSError as e:
                print(f"Files Not present for this {fov}, skipping")
                continue
            max_image = np.load(os.path.join(fov_path, "max_image.npy"))#TODO the max image should come from the session and not from the FOV
            stat = np.load(os.path.join(fov_path, "stat.npy"), allow_pickle=True).tolist()

            if session_list is None:
                session_list = next(os.walk(fov_path))[1]

            for session_date in session_list:
                if session_date == "Z-stacks":
                    continue

                session_save_path = os.path.join(mice_save_path, f"{mouse}-{session_date}-{fov}.npy")
                if os.path.exists(session_save_path) and overwrite==False:
                    print(f"Session already exists at {session_save_path}, and overwrite=False")
                    continue
                
                else: 
                    print(f"FOV: {fov}, Session Date: {session_date}")
                    session_path = os.path.join(fov_path, session_date)
                    if not os.path.isfile(os.path.join(session_path, "ops.npy")):
                        continue
                    ops =  np.load(os.path.join(session_path, "ops.npy") ,allow_pickle = True).tolist()
                    fs = ops['fs']
                    tsta = np.arange(-frames_prev_trial, frames_this_trial, 1).astype(float)/fs
                    
                    F = np.load(os.path.join(session_path, "F.npy"), allow_pickle=True)
                    F0 = np.load(os.path.join(session_path, "F0.npy"), allow_pickle=True)
                    photon_counts_dict=np.load(os.path.join(session_path,'photon_counts.npy'),allow_pickle=True).tolist()
					#f0_scalar = np.mean(np.load(os.path.join(session_path,'F0.npy')),1)                    
                    f0_scalar = np.percentile(F0[:,:int(F0.shape[1]/2)],10,axis = 1)
                    
                    if adjust_channel_offsets: # this is optional, might not be important
                        channel_offset_dict = np.load(os.path.join(session_path,'channel_offset.npy'),allow_pickle=True).tolist()
                        F+= channel_offset_dict['channel_offset']
                        F0+=channel_offset_dict['channel_offset']                    
                    dff = (F-F0)/F0

                    with open(os.path.join(session_path, "filelist.json")) as json_file:
                        filelist = json.load(json_file)   
                    
                    
                    all_si_filenames = filelist['file_name_list']
                    behavior_fname = os.path.join(behavior_data_path, f"{session_date}-bpod_zaber.npy")
                    cn_idx,_closed_loop_trial,_scanimage_filenames = find_conditioned_neuron_idx(behavior_fname, 
                                                                                                 os.path.join(session_path, "ops.npy"), 
                                                                                                 os.path.join(fov_path, "stat.npy"), 
                                                                                                 plot=False)
                    
                    try:
                        clt = np.concatenate(np.asarray(_scanimage_filenames)[_closed_loop_trial])
                    except:
                        clt = []
                    
                    closed_loop_filenames = [k for k in filelist['file_name_list'] if k.lower().startswith("neuron") or k in clt or k.lower().startswith("condition") ] # TODO, we should pull out this information from the scanimage tiff header
                    #closed_loop_filenames = [k[0] for k in np.asarray(_scanimage_filenames)[_closed_loop_trial] if k[0].lower().startswith("neuron")]

                    frame_num = np.asarray(filelist['frame_num_list'])
                    filename_start_frame = np.asarray([0] + np.cumsum(frame_num).tolist())

                    F_trialwise_all = np.ones((max_frames, F.shape[0], len(all_si_filenames)))*np.nan
                    F_trialwise_closed_loop = np.ones((max_frames, F.shape[0], len(closed_loop_filenames)))*np.nan
                    dff_trialwise_all = np.ones((max_frames, F.shape[0], len(all_si_filenames)))*np.nan
                    dff_trialwise_closed_loop =  np.ones((max_frames, F.shape[0], len(closed_loop_filenames)))*np.nan

                    counter = 0
                    for i, filename in tqdm(enumerate(all_si_filenames)):
                        start_frame = filename_start_frame[i]
                        end_frame = filename_start_frame[i+1]
                        
                        if end_frame - start_frame > frames_this_trial:
                            end_frame = start_frame + frames_this_trial
                        start_frame = start_frame - frames_prev_trial # taking 40 time points before trial starts
                        if start_frame<0: # taking care of edge at the beginning
                            missing_frames_at_beginning = np.abs(start_frame)
                            start_frame = 0
                        else:
                            missing_frames_at_beginning = 0
                            
                        if filename in closed_loop_filenames:
                            F_trialwise_closed_loop[missing_frames_at_beginning:missing_frames_at_beginning+end_frame-start_frame, :, counter] = F[:, start_frame:end_frame].T
                            dff_trialwise_closed_loop[missing_frames_at_beginning:missing_frames_at_beginning+end_frame-start_frame, :, counter] = dff[:, start_frame:end_frame].T
                            counter += 1
                            
                        F_trialwise_all[missing_frames_at_beginning:missing_frames_at_beginning+end_frame-start_frame, :, i] = F[:, start_frame:end_frame].T
                        dff_trialwise_all[missing_frames_at_beginning:missing_frames_at_beginning+end_frame-start_frame, :, i] = dff[:, start_frame:end_frame].T
                                                
                    
                    #print(f"cn idx: {np.unique(cn_idx)}")
                    roi_centers = [(stat[i]['xpix'].mean(), stat[i]['ypix'].mean()) for i in range(len(stat))]
                    roi_centers = np.asarray(roi_centers)
                    idxi = 0
                    try:
                        while cn_idx[idxi] == None:
                            idxi += 1
                        if type(cn_idx[idxi]) == int or type(cn_idx[idxi]) ==np.int64 :
                            roi_centers_cn = roi_centers - roi_centers[cn_idx[0]]
                            dist = [np.sqrt(roi_centers_cn[i][0]**2 + roi_centers_cn[i][1]**2) for i in range(len(roi_centers_cn))]
                        else:
                            dist = []
                            for cn_idx_ in cn_idx[idxi]:
                                roi_centers_cn = roi_centers - roi_centers[cn_idx_]
                                dist_ = [np.sqrt(roi_centers_cn[i][0]**2 + roi_centers_cn[i][1]**2) for i in range(len(roi_centers_cn))]
                                dist.append(dist_)
                    except:
                        print('cn not found??')
                        dist = []
                        cn_idx=np.nan
                                
                    if not os.path.isfile(behavior_fname):
                        print("No corresponding behavior data found for {}".format(session_date))
                        break
                    else:
                        bpod_zaber_data = np.load(behavior_fname, allow_pickle=True).tolist()
                        files_with_movies = []
                        for k in bpod_zaber_data['scanimage_file_names']:
                            if str(k) == 'no movie for this trial':
                                files_with_movies.append(False)
                            else:
                                if len(k)== 1:
                                    files_with_movies.append(True)                        
                                else:
                                    print('multiple movies for a behavior trial, skipped: {}'.format(k))
                                    files_with_movies.append(False)                        
                        trial_st = bpod_zaber_data['trial_start_times'][files_with_movies]    
                        trial_et = bpod_zaber_data['trial_end_times'][files_with_movies]
                        gocue_t = bpod_zaber_data['go_cue_times'][files_with_movies]
                        trial_times = [(trial_et[i]-trial_st[i]).total_seconds() for i in range(len(trial_st))]
                        trial_hit = bpod_zaber_data['trial_hit'][files_with_movies]
                        lick_L = bpod_zaber_data['lick_L'][files_with_movies]

                        threshold_crossing_times = bpod_zaber_data['threshold_crossing_times'][files_with_movies]

                        scanimage_filenames = np.asarray(bpod_zaber_data["scanimage_file_names"])[files_with_movies]
                        needed_cn_indices = []
                        for clt_ in _scanimage_filenames:
                            if clt_ in scanimage_filenames:
                                needed_cn_indices.append(True)
                            else:
                                needed_cn_indices.append(False)
                        cn_idx = np.asarray(cn_idx)[np.asarray(needed_cn_indices)]
                        
                    dict_all = {'F_sessionwise': F,
                                'F_trialwise_all': F_trialwise_all,
                                'F_trialwise_closed_loop': F_trialwise_closed_loop,
                                'dff_sessionwise': dff,
                                'dff_trialwise_all': dff_trialwise_all,
                                'dff_trialwise_closed_loop': dff_trialwise_closed_loop,
                                'cn': cn_idx,
                                'roiX': roi_centers[:, 0],
                                'roiY': roi_centers[:, 1],
                                'dist': dist,
                                'FOV': fov,
                                'session_date': session_date,
                                'session_path': session_path,
                                'mouse': mouse,
                                'mean_image': mean_image,
                                'max_image': max_image,
                                'time_since_trial_start': tsta,
                                'go_cue_times': gocue_t,
                                'lick_times': lick_L,
                                'reward_times': bpod_zaber_data['reward_L'][files_with_movies],
                                'trial_start_times': trial_st,
                                'trial_times': trial_times,
                                'hit': trial_hit,
                                'threshold_crossing_times': threshold_crossing_times, 
                                'zaber_move_forward': bpod_zaber_data['zaber_move_forward'][files_with_movies],
                                'sampling_rate': fs,
                                'all_si_filenames': all_si_filenames,
                                'all_si_frame_nums':frame_num,
                                'closed_loop_filenames': closed_loop_filenames,
                                'scanimage_filenames': scanimage_filenames,
                                'photon_counts' :photon_counts_dict,
                                'f0_scalar':f0_scalar
                            }
                    #%%
                    np.save(session_save_path, dict_all)
                    print(f"Saved to {session_save_path}")