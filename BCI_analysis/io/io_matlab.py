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
        file_path - str,
            Path to loaded file
        file_name - str
            Name of loaded file
        session_file_names - list of str [sessions]
            Names of source session files
        n_days - int
            Number of sessions in this dataset
        cni - int, [sessions]
            Conditioned Neuron indices
        dFF - float [sessions x time points x neurons]
            DeltaF/F      
        dist -float [sessions x neurons x 1]
            Distance from Conditioned Neuron in pixels   
        meanImg - float [sessions x Y_size x X_size]
            Mean Image of the field of view
        raw - float [sessions x time points x neurons]
            Flourescence Intensity
        F_trialwise - float [sessions x time points x neurons x trials]
            DeltaF/F reshaped     
        tsta - float [sessions x time points]
            Time steps corresponding to F_trialwise
        total_steps - int [sessions]
            Total time steps
        n_trials - int [sessions]
            Number of trials
        epoch - int, [sessions x trials]
            3 keys corresponding to spontaneuous Pre Training, closed loop, spontaneous Post Training 
        lkeys - int [sessions]
            which key correspond to actual training trials from epoch
        epoch_closedloop bool [sessions]
            ???
        roi list of dicts [sessions]
            centroid - float [number of ROIs x 2]
                centroid of ROI in pixels
            centerXY - float [number of ROIs x 2]
                centroid of ROI in microns from the center of the FOV
            plane - int [number of ROIs x 1]
                the plane number in which the ROI resides
            pixelList int [number of ROIs x number of pixels in ROI] 
                list of pixels that belong to the ROI
            scalingXY float [number of ROIs x 2]
                ???
            intensity [number of ROIs x ???] 
                ???
            
    """
    file_path,file_name = os.path.split(full_file_path)
    data=io.loadmat(full_file_path)
    
    n_days = data['data'].shape[1]    
    session_file_names = [data['data'][:, i]['file'][0][0] for i in range(n_days)] # -1 to account for matlab indexing
    cni = [data['data'][:, i]['cn'][0][0][0] - 1 for i in range(n_days)] # -1 to account for matlab indexing
    dFF = [data['data'][:, i]['df'][0] for i in range(n_days)]
    dist = [data['data'][:, i]['dist'][0].flatten() for i in range(n_days)]
    meanImg = [data['data'][:, i]['IM'][0] for i in range(n_days)]
    raw = [data['data'][:, i]['raw'][0] for i in range(n_days)]
    F_trialwise = [data['data'][:, i]['F'][0] for i in range(n_days)]
    tsta = [data['data'][:, i]['tsta'][0].flatten() for i in range(n_days)]
    total_steps = [dFF[i].shape[0] for i in range(n_days)]
    n_trials = [F_trialwise[i].shape[-1] for i in range(n_days)]
    epoch = [data['data'][:, i]['epoch'][0].flatten() for i in range(n_days)]
    lkeys = [[1,3], 1, 4, 2]  # TODO this is not defined in the .mat file
    epoch_closedloop = [np.isin(epoch[i],lkeys[i])[0] for i in range(n_days)]

    roi = []
    for i in range(n_days):
        roi_dict = {}
        n_cells = data['data'][:, i]['roi'][0].shape[1]
        roi_keys = data['data'][:, i]['roi'][0].dtype.fields.keys()
        for key in roi_keys:
            key_list = []
            for cell_i in range(n_cells):
                temp_list = data['data'][:, i]['roi'][0][:,cell_i][key]
                while len(temp_list)>0 and hasattr(temp_list[0], '__iter__'):
                    temp_list = np.concatenate(temp_list)
                if key in ['centroid','pixelList']:
                    temp_list -= 1
                key_list.append(temp_list.tolist())
            roi_dict[key]= key_list
        roi.append(roi_dict)
        
    data_dict = {'file_path':file_path,
                 'file_name':file_name,
                 'session_file_names':session_file_names,
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
                 'epoch_closedloop':epoch_closedloop,
                 'roi':roi
                 }
    
    return data_dict