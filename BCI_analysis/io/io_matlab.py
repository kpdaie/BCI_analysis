import scipy.io as io
import numpy as np
import os
import mat73
from datetime import datetime

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
        subject - str
            Mouse water restriction ID.
        session_dates - list of datetime.date, [sessions]
            Dates of sessions.
        session_file_names - list of str, [sessions]
            Names of source session files
        scanimage_file_names - list of str [session x file count]
            Names of original scanimage files.
        closed_loop_scanimage_file_names - list of str [session x file count]
            Names of original scnimage files during closed loop.
        n_days - int
            Number of sessions in this dataset
        conditioned_neuron_idx - list of int, [sessions]
            Conditioned Neuron indices
        dff_sessionwise_all_epochs - float [sessions x time points x neurons]
            DeltaF/F of all trials 
        dff_sessionwise_closed_loop - float [sessions x time points x neurons]
            DeltaF/F of closed loop trials 
        dff_trialwise_closed_loop - float [sessions x time points x neurons x trials]
            DeltaF/F of closed loop trials reshaped by trials
        distance_from_conditioned_neuron -float [sessions x neurons]
            Distance from Conditioned Neuron in pixels   
        mean_image - float [sessions x Y_size x X_size]
            Mean Image of the field of view
        f_sessionwise_closed_loop - float [sessions x time points x neurons]
            Raw flourescence intensity of closed loop trials
        f_trialwise_closed_loop - float [sessions x time points x neurons x trials]
            Raw flourescence intensity of closed loop trials reshaped by trials    
        time_from_trial_start - float [sessions x time points]
            Time steps corresponding to trialwise fluorescence traces
            
        # TO BE ADDED#:
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
    #%%
    
    file_path,file_name = os.path.split(full_file_path)
    data = mat73.loadmat(full_file_path)['data']
    
        
    data_dict = {'file_path':file_path,
                 'file_name':file_name,
                 'subject':data['mouse'][0],
                 #maing this a placeholder for time being...
                  'session_dates' : [['empty'] for i in data['sessionDate']],# 'session_dates':[datetime.strptime(str(int(i)).zfill(6),'%M%d%Y').date() for i in data['sessionDate']],
                 'session_file_names':data['file'],
                 'scanimage_file_names':data['all_si_filenames'],
                 'closed_loop_scanimage_file_names':data['closed_loop_filenames'],
                 'n_days':len(data['mouse']),
                
                 #apparently some mice have inconsistent dimensions for conditioned_neuron_indexes? -- line 85 error for BCI15
                 'dff_sessionwise_all_epochs':data['dff_sessionwise_all_epochs'],
                 'dff_sessionwise_closed_loop':data['dff_sessionwise_closed_loop'],
                 'dff_trialwise_closed_loop':data['dff_trialwise_closded_loop'], # this one is missing from many files, commented out for now
                 'distance_from_conditioned_neuron':data['dist'],
                 'mean_image':data['mean_image'],
                 'f_sessionwise_closed_loop':data['f_sessionwise_closed_loop'],
                 'f_trialwise_closed_loop':data['f_trialwise'],
                 'time_from_trial_start':data['time_from_trial_start'],
                 #'roi':roi
                 }
    
    try: #will try unless there exists a session with no conditioned neurons
        data_dict['conditioned_neuron_idx']=(np.array(data['cn'],int).flatten()-1).tolist(), # matlab to python indexing
    except ValueError:
        print('Some Neurons do not have conditioned neurons')
    #%%
    return data_dict