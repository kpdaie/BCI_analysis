#%%
import numpy as np
import os
import json
from BCI_analysis.pipeline.pipeline_imaging import find_conditioned_neuron_idx
from tqdm import tqdm

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        
    Source:
        https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """

    return np.isnan(y), lambda z: z.nonzero()[0]
def remove_stim_artefacts(F,Fneu,frames_per_file):
    """
    removing stimulation artefacts with linear interpolation
    and nan-ing out tripped PMT traces

    Parameters
    ----------
    F : matrix of float
        Fluorescence of ROIs
    Fneu : matrix of float
        fluorescence of neuropil
    frames_per_file : list of int
        # of frames in each file (where the photostim happens)

    Returns
    -------
    F : matrix of float
        corrected fluorescence of ROIs
    Fneu : matrix of float
        corrected fluorescence of neuropil

    """
    artefact_indices = []
    fneu_mean = np.mean(Fneu,0)
    
    for stim_idx in np.concatenate([[0],np.cumsum(frames_per_file)[:-1]]):
        idx_now = []
        if stim_idx>0 and fneu_mean[stim_idx-2]*1.1<fneu_mean[stim_idx-1]:
            idx_now.append(stim_idx-1)
        idx_now.append(stim_idx)
        if stim_idx<len(fneu_mean)-2 and fneu_mean[stim_idx+2]*1.1<fneu_mean[stim_idx+1]:
            idx_now.append(stim_idx+1)
        artefact_indices.append(idx_now)
    
    f_std = np.std(F,0)
    pmt_off_indices = f_std<np.median(f_std)-3*np.std(f_std)
    pmt_off_edges = np.diff(np.concatenate([pmt_off_indices,[0]]))
    pmt_off_indices[pmt_off_edges!=0] = 1 #dilate 1
    pmt_off_edges = np.diff(np.concatenate([[0],pmt_off_indices,[0]]))
    starts = np.where(pmt_off_edges==1)[0]
    ends = np.where(pmt_off_edges==-1)[0]
    lengths = ends-starts
    for idx in np.where(lengths<=10)[0]:
        pmt_off_indices[starts[idx]:ends[idx]]=0
    
    
    F_ = F.copy()
    F_[:,np.concatenate(artefact_indices)]=np.nan
    for f in F_:
        nans, x= nan_helper(f)
        f[nans]= np.interp(x(nans), x(~nans), f[~nans])
        f[pmt_off_indices] = np.nan
    F = F_
    
    Fneu_ = Fneu.copy()
    Fneu_[:,np.concatenate(artefact_indices)]=np.nan
    for f in Fneu_:
        nans, x= nan_helper(f)
        f[nans]= np.interp(x(nans), x(~nans), f[~nans])
        f[pmt_off_indices] = np.nan
    Fneu = Fneu_
    return F, Fneu

def remove_PMT_trips(F):
    """
    nan-ing out tripped PMT traces

    Parameters
    ----------
    F : matrix of float
        Fluorescence of ROIs
        # of frames in each file (where the photostim happens)

    Returns
    -------
    F : matrix of float
        corrected fluorescence of ROIs
    Fneu : matrix of float
        corrected fluorescence of neuropil

    """

    f_std = np.std(F,0)
    pmt_off_indices = f_std<np.median(f_std)-3*np.std(f_std[(f_std>np.percentile(f_std,5)) & (f_std<np.percentile(f_std,95))])
    pmt_off_edges = np.diff(np.concatenate([pmt_off_indices,[0]]))
    pmt_off_indices[pmt_off_edges!=0] = 1 #dilate 1
    pmt_off_edges = np.diff(np.concatenate([[0],pmt_off_indices,[0]]))
    starts = np.where(pmt_off_edges==1)[0]
    ends = np.where(pmt_off_edges==-1)[0]
    lengths = ends-starts
    for idx in np.where(lengths<=10)[0]:
        pmt_off_indices[starts[idx]:ends[idx]]=0
    F[:,pmt_off_indices] = np.nan
    return F

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
                try:
                    F_trialwise_closed_loop[missing_frames_at_beginning:missing_frames_at_beginning+end_frame-start_frame, :, counter] = F[:, start_frame:end_frame].T
                except:
                    print('could not add a reward trial..')
                needed_list.append(counter)
                counter += 1
            F_trialwise_closed_loop = F_trialwise_closed_loop[:,:,needed_list]
            return F_trialwise_closed_loop



        

                
def create_BCI_F(Ftrace,ops,stat):
    F_trial_strt = [];
    Fraw_trial_strt = [];

    strt = 0;
    dff = 0*Ftrace
    for i in range(np.shape(Ftrace)[0]):
        bl = np.std(Ftrace[i,:])
        dff[i,:] = (Ftrace[i,:] - bl)/bl

    for i in range(len(ops['frames_per_file'])):
        ind = list(range(strt,strt+ops['frames_per_file'][i]))   
        f = dff[:,ind]
        F_trial_strt.append(f)
        f = Ftrace[:,ind]
        Fraw_trial_strt.append(f)
        strt = ind[-1]+1


    F = np.full((240,np.shape(Ftrace)[0],len(ops['frames_per_file'])),np.nan)#HARD CODED
    Fraw = np.full((240,np.shape(Ftrace)[0],len(ops['frames_per_file'])),np.nan)#HARD CODED
    pre = np.full((np.shape(Ftrace)[0],40),np.nan)#HARD CODED
    for i in range(len(ops['frames_per_file'])):
        f = F_trial_strt[i]
        fraw = Fraw_trial_strt[i]
        if i > 0:
            pre = F_trial_strt[i-1][:,-40:] #HARD CODED
        pad = np.full((np.shape(Ftrace)[0],200),np.nan) # HARD CODED
        f = np.concatenate((pre,f),axis = 1)
        f = np.concatenate((f,pad),axis = 1)
        f = f[:,0:240] # HARD CODED
        F[:,:,i] = np.transpose(f)


        fraw = np.concatenate((pre,fraw),axis = 1)
        fraw = np.concatenate((fraw,pad),axis = 1)
        fraw = fraw[:,0:240] # HARDCODED
        Fraw[:,:,i] = np.transpose(fraw)
        centroidX = []
        centroidY = []
        dist = []

        for i in range(len(stat)):
            centroidX.append(np.mean(stat[i]['xpix']))
            centroidY.append(np.mean(stat[i]['ypix']))
    return F, Fraw, dff,centroidX, centroidY
os.walk             

def generate_canned_sessions(suite2p_path,
                             raw_data_path,
                             behavior_data_path,
                             save_path,
                             overwrite=False,
                             mouse_list=None,
                             fov_list=None, 
                             session_list=None):
    
    if mouse_list is None:
        mouse_list = os.listdir(suite2p_path)
    if type(mouse_list) == str:
        mouse_list = mouse_list.split()
    
    for mouse in mouse_list:
        suite2p_data = os.path.join(suite2p_path, mouse)
        raw_suite2p = os.path.join(raw_data_path, mouse)
        behavior_data_path = os.path.join(behavior_data_path)
        mouse_save_path = os.path.join(save_path)

        os.makedirs(mouse_save_path, exist_ok=True)

        if fov_list is None:
            fov_list_ = [k for k in os.listdir(suite2p_data) if k[-2:].isdigit()] # For BCI_29 there exists folders other than FOV_0x
            print(fov_list_)
        else:
            fov_list_ = fov_list

        for fov in fov_list_:
            fov_path = os.path.join(suite2p_data, fov)
            try:
                mean_image = np.load(os.path.join(fov_path, "mean_image.npy")) #TODO the mean image should come from the session and not from the FOV
            except OSError as e:
                print(f"Files Not present for this {fov}, skipping")
                continue
            max_image = np.load(os.path.join(fov_path, "max_image.npy"))#TODO the max image should come from the session and not from the FOV
            stat = np.load(os.path.join(fov_path, "stat.npy"), allow_pickle=True).tolist()

            if session_list is None:
                session_list_ = next(os.walk(fov_path))[1]
            else:
                session_list_ = session_list

            for session_date in session_list_:
                if session_date == "Z-stacks":
                    continue
                session_save_path = os.path.join(mouse_save_path, f"{mouse}-{session_date}.npy")
                if os.path.exists(session_save_path) and overwrite==False:
                    print(f"Session already exists at {session_save_path}, and overwrite=False")
                    continue
                print(f"FOV: {fov}, Session Date: {session_date}")
                try:
                    dict_all = generate_canned_session(suite2p_path,
                                                       mouse,
                                                       fov,
                                                       session_date,
                                                       behavior_data_path)
                    if type(dict_all) == dict:
                        np.save(session_save_path, dict_all)
                except:
                    print('error saving {}'.format(os.path.join(fov_path, session_date)))
                    
    
    
def generate_canned_session(suite2p_path,
                            mouse,
                            fov,
                            session_date,
                            behavior_data_path):
    
    fov_path = os.path.join(suite2p_path, mouse,fov)
    session_path = os.path.join(fov_path, session_date)
    if not os.path.isfile(os.path.join(session_path, "ops.npy")):
        print(f"No ops.npy found in {session_path}, aborting")
        return None
    stat =  np.load(os.path.join(fov_path, "stat.npy"),allow_pickle = True).tolist()
    iscell = np.ones(len(stat))
    ops =  np.load(os.path.join(session_path, "ops.npy") ,allow_pickle = True).tolist()
    F = np.load(os.path.join(session_path, "F.npy"), allow_pickle=True)
    F0 = np.load(os.path.join(session_path, "F0.npy"), allow_pickle=True)
    meanimages = np.load(os.path.join(fov_path,'session_mean_images.npy'),allow_pickle = True).tolist()
    meanImg =  meanimages[session_date]['meanImg']
    fs = ops['fs']

    data = dict()
    # basic metadata
    data['dt_si'] = 1/fs
    
    data['iscell'] = iscell
    # metadata
    data['dat_file'] = session_path
    data['session'] = session_date
    data['mouse'] = mouse
    data['mean_image'] = meanImg
    data['fov'] = fov
    data['version'] = '1.0'



    # identify trials with imaging & get CNs

    with open(os.path.join(session_path, "filelist.json")) as json_file:
        filelist = json.load(json_file)   

    all_si_filenames = filelist['file_name_list']
    all_si_frame_nums = np.asarray(filelist['frame_num_list'])
    behavior_fname = os.path.join(behavior_data_path,mouse, f"{session_date}-bpod_zaber.npy")
    cn_idx,_closed_loop_trial,_scanimage_filenames,dist_from_cn = find_conditioned_neuron_idx(behavior_fname, 
                                                                                 os.path.join(session_path, "ops.npy"), 
                                                                                 os.path.join(fov_path, "stat.npy"), 
                                                                                 plot=False,
                                                                                 return_distances = True)


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
    reward_L = bpod_zaber_data['reward_L'][files_with_movies]
    threshold_crossing_times = bpod_zaber_data['threshold_crossing_times'][files_with_movies]
    lickport_steps = bpod_zaber_data['zaber_move_forward'][files_with_movies]

    closed_loop_scanimage_filenames = np.asarray(bpod_zaber_data["scanimage_file_names"])[files_with_movies]
    needed_cn_indices = []
    for clt_ in _scanimage_filenames:
        if clt_ in closed_loop_scanimage_filenames:
            needed_cn_indices.append(True)
        else:
            needed_cn_indices.append(False)
    try:
        cn_idx = np.asarray(cn_idx)[np.asarray(needed_cn_indices)]
    except:
        print('cn is not found: {}'.format(cn_idx))
        cn_idx = np.asarray([None]*sum(needed_cn_indices))
    dist_from_cn = np.asarray(dist_from_cn)[np.asarray(needed_cn_indices)]
    scanimage_filenames = np.asarray(_scanimage_filenames)[np.asarray(needed_cn_indices)]
    uniquecns = np.unique(np.asarray(cn_idx)[(cn_idx==None) ==False])
    if len(uniquecns)>1:
        median_index = []
        for ucn in uniquecns:
            median_index.append(np.nanmedian(np.where(np.asarray(cn_idx)==ucn)[0]))
        uniquecns = uniquecns[np.argsort(median_index)] # now they are ordered
    print('{} conditioned neuron(s) found: {}'.format(len(uniquecns),uniquecns))


    # get the spontaneous
    end_framenum_spontaneous = np.sum(np.asarray(all_si_frame_nums)[:np.where(np.asarray(all_si_filenames) ==closed_loop_scanimage_filenames[0])[0][0]])
    data['spont'] = {'Ftrace': F[:,:end_framenum_spontaneous],
                    'trace_corr' : np.corrcoef(F[:,:end_framenum_spontaneous].T, rowvar=False)}
    
    # get conditioning & preconditioning
    #for minuniquecnnum,cnidx,suffix,parent_dict_name in zip([1,0],[0,-1],['',''],['BCI_precond','BCI']):
    for minuniquecnnum,cnidx,suffix,parent_dict_name in zip([0,1,2,3],[0,1,2,3],['','','',''],['BCI_1','BCI_2','BCI_3','BCI_4']):
        if len(uniquecns)>minuniquecnnum:
            data_ = {}
            cn_prev = uniquecns[cnidx]  
            closed_loop_indices_needed = (np.asarray(cn_idx) == None) == False
            closed_loop_indices = np.asarray(cn_idx)[closed_loop_indices_needed]==cn_prev
            closed_loop_filenames = scanimage_filenames[closed_loop_indices_needed][closed_loop_indices]
            framenums = []
            frame_per_file_now = []
            reward_indices = []
            go_cue_indices = []
            trial_start_indices = []
            threshold_crossing_indices = []
            lick_indices = []
            lickport_step_indices = []
            frames_so_far  = 0
            for filename_now in closed_loop_filenames:
                idx = np.where(all_si_filenames == filename_now)[0][0]
                start_frame = np.sum(all_si_frame_nums[:idx])
                framenums.append(np.arange(all_si_frame_nums[idx])+start_frame)
                frame_per_file_now.append(all_si_frame_nums[idx])
                idx_behavior = np.where(closed_loop_scanimage_filenames == filename_now)[0][0]
                reward_indices.append(np.asarray(np.asarray(reward_L[idx_behavior])*fs,int)+frames_so_far)
                go_cue_indices.append(np.asarray(np.asarray(gocue_t[idx_behavior])*fs,int)+frames_so_far)
                trial_start_indices.append([frames_so_far])
                threshold_crossing_indices.append(np.asarray(np.asarray(threshold_crossing_times[idx_behavior])*fs,int)+frames_so_far)
                lick_indices.append(np.asarray(np.asarray(lick_L[idx_behavior])*fs,int)+frames_so_far)
                lickport_step_indices.append(np.asarray(np.asarray(lickport_steps[idx_behavior])*fs,int)+frames_so_far)
                frames_so_far = len(np.concatenate(framenums))
            framenums = np.concatenate(framenums)
            trial_start_trace = np.zeros_like(framenums)
            trial_start_trace[np.concatenate(trial_start_indices)]+=1
            reward_trace = np.zeros_like(framenums)
            reward_trace[np.concatenate(reward_indices)]+=1
            go_cue_trace = np.zeros_like(framenums)
            go_cue_trace[np.concatenate(go_cue_indices)]+=1
            threshold_crossing_trace = np.zeros_like(framenums)
            threshold_crossing_trace[np.concatenate(threshold_crossing_indices)]+=1
            lick_trace = np.zeros_like(framenums)
            lick_trace[np.concatenate(lick_indices)]+=1
            lickport_step_trace = np.zeros_like(framenums)
            lickport_step_trace[np.concatenate(lickport_step_indices)]+=1
            ops_temp = {'frames_per_file': np.asarray(frame_per_file_now)}
            #print(ops_temp)
            data_['F'+suffix], data_['Fraw'+suffix],data_['df_closedloop'+suffix],data_['centroidX'+suffix],data_['centroidY'+suffix] = create_BCI_F(F[:,framenums],ops_temp,stat);  
            data_['Ftrace'+suffix] = F[:,framenums]
            data_['trace_corr'+suffix] = np.corrcoef(F[:,framenums].T, rowvar=False)
            data_['dist'+suffix] = dist_from_cn[np.where(all_si_filenames == closed_loop_filenames[0])[0][0]]
            data_['conditioned_neuron_coordinates'+suffix] = [stat[cn_prev]['xpix'],stat[cn_prev]['ypix']]
            data_['conditioned_neuron'+suffix] = cn_prev
            data_['reward_time'+suffix] = reward_trace
            data_['step_time'+suffix] = lickport_step_trace
            data_['trial_start'+suffix] = trial_start_trace
            data_['lick_time'+suffix] = lick_trace
            data_['threshold_crossing_time'+suffix] = threshold_crossing_trace
            
            if parent_dict_name == '':
                for key in data_.keys():
                    data[key] = data_[key]
            else:
                data[parent_dict_name] = {}
                for key in data_.keys():
                    data[parent_dict_name][key] = data_[key]
    # get photostim
    if 'photostim' in os.listdir(session_path):
        print('photostim exported')
        photostim_dict = np.load(os.path.join(session_path,'photostim','photostim_dict.npy'),allow_pickle = True).tolist() 
        data['photostim']=photostim_dict
#         photostim_groups = np.load(os.path.join(session_path,'photostim','photostim_groups.npy'),allow_pickle = True).tolist()
#         pg = []
#         for photostim_group in photostim_groups['groups']:
#             if type(pg) == list:
#                 photostim_group.pop('cell_response_distribution')
#                 photostim_group.pop('photostimmed_cells')
#                 pg = {}
#                 for key in photostim_group.keys():
#                     pg[key] = []
#             for key in pg.keys():
#                 pg[key].append(photostim_group[key])
            
#         data['photostim']['stim_group_metadata'] = pg
    
    return data



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
                session_list_ = next(os.walk(fov_path))[1]
            else:
                session_list_ = session_list

            for session_date in session_list_:
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
                                                                                                 plot=True)
                    
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
                    