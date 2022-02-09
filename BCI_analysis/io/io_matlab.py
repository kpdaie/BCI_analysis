import scipy.io as io
import numpy as np
import os

def read_multisession_mat(full_file_path):
    """
    Reads the output of Kayvon's pipeline, multi-session .mat file as a dict.
    
    Parameters
    ----------
    file_path : str
        location of multi-session .mat file to load

    Returns
    -------
    data_dict : dict
        file_path - Path to loaded file
        file_name - Name of loaded file
        n_days - Number of sessions in this dataset
        cni - Conditioned Neuron index
        dFF - DeltaF/F      [time points x neurons]
        dist - Distance from Conditioned Neuron in pixels   [neurons x 1]
        meanImg - Mean Image of the roi
        raw - Flourescence Intensity    [time points x neurons]
        F_trialwise - DeltaF/F reshaped     [time points x neurons x trials]
        tsta - Time steps corresponding to F
        total_steps - Total time steps
        n_trials - Number of trials
        epoch - 3 keys corresponding to spontaneuous Pre Training, closed loop, spontaneous Post Training [trials x 1]
    """
    file_path,file_name = os.path.split(full_file_path)
    data=io.loadmat(full_file_path)
    
    n_days = data['data'].shape[1]    
    cni = [data['data'][:, i]['cn'][0][0][0] - 1 for i in range(n_days)] # -1 to account for matlab indexing
    dFF = [data['data'][:, i]['df'][0] for i in range(n_days)]
    dist = [data['data'][:, i]['dist'][0] for i in range(n_days)]
    meanImg = [data['data'][:, i]['IM'][0] for i in range(n_days)]
    raw = [data['data'][:, i]['raw'][0] for i in range(n_days)]
    F_trialwise = [data['data'][:, i]['F'][0] for i in range(n_days)]
    tsta = [data['data'][:, i]['tsta'][0] for i in range(n_days)]
    total_steps = [dFF[i].shape[0] for i in range(n_days)]
    n_trials = [F_trialwise[i].shape[-1] for i in range(n_days)]
    epoch = [data['data'][:, i]['epoch'][0] for i in range(n_days)]
    lkeys = [[1,3], 1, 4, 2]  #Since i did not know the exact epoch numbers for the BCI_13Sep session, 
    # I hardcoded them here. This should change in the future

    epoch_closedloop = [np.isin(epoch[i],lkeys[i])[0] for i in range(n_days)]
    
    
    data_dict = {'file_path':file_path,
                 'file_name':file_name,
                 'n_days':n_days,
                 'cni':cni,
                 'dFF':dFF,
                 'dist':dist,
                 'meanImg':meanImg,
                 'raw':raw,
                 'F_trialwise':F_trialwise,
                 'tsta':tsta,
                 'total_steps':total_steps,
                 'n_trials':n_trials,
                 'epoch':epoch,
                 'lkeys':lkeys,
                 'epoch_closedloop':epoch_closedloop
                 }
    
    return data_dict