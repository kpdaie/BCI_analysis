import numpy as np
import matplotlib.pyplot as plt

def plot_trial_averaged_trace_2sessions(data_dict, day_ind, start_time=-2.5, 
                                        end_time=10, baseline_start_time = None,
                                        baseline_end_time = 0):
    """
    Returns a baseline subtracted plot of trial averaged flourescent trace of 
    the conditioned neuron of the selected session (day_ind) and the session 
    before.
    
    Parameters
    ----------
    data_dict : dict
        output of io.io_matlab.read_multisession_mat() function, highly 
        specialized multi-session
    day_ind : int
        Day index, the index of the session that will be plotted against the 
        previous session
    start_time : float, optional
        Time when plot starts. The default is -2.5.
    end_time : float, optional
        Time when plot ends. The default is 10.
    baseline_start_time : Tfloat, optional
        Time when baseline starts. The default is start_time.
    baseline_end_time : float, optional
        Time where baseline ends. The default is 0.

    Returns
    -------
    None
    A matplotlib figure.

    """
  
    if baseline_start_time == None: baseline_start_time = start_time
    
    time_yesterday=np.transpose(data_dict['tsta'][day_ind-1]).flatten()
    idx_yesterday = (time_yesterday>start_time) & (time_yesterday<end_time)
    time_yesterday = time_yesterday[idx_yesterday]
    
    time_today=np.transpose(data_dict['tsta'][day_ind]).flatten()
    idx_today = (time_today>start_time) & (time_today<end_time)
    time_today = time_today[idx_today]
    
    avg_traces_yesterday = np.nanmean(data_dict['F_trialwise'][day_ind-1][idx_yesterday,:,:], axis=2)
    avg_traces_today = np.nanmean(data_dict['F_trialwise'][day_ind][idx_today,:,:], axis=2)
    
    baseline_idx_today = (time_today>baseline_start_time) & (time_today<baseline_end_time)
    baseline_idx_yesterday = (time_yesterday>baseline_start_time) & (time_yesterday<baseline_end_time)
    
    for i in range(avg_traces_today.shape[1]):    
        avg_traces_today[:,i] = avg_traces_today[:,i] - np.nanmean(avg_traces_today[baseline_idx_today,i],0)
        avg_traces_yesterday[:,i] = avg_traces_yesterday[:,i] - np.nanmean(avg_traces_yesterday[baseline_idx_yesterday,i],0)
    
    
    plt.plot(time_today,avg_traces_today[:,data_dict['cni'][day_ind]],color='r')
    plt.plot(time_yesterday,avg_traces_yesterday[:,data_dict['cni'][day_ind]],color='k')