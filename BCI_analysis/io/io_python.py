import numpy as np
import os

def read_sessionwise_npy(file_path):
    """
    Reads and returns the npy file as a dict

    Parameters
    ----------
    file_path : str
        location of multi-session .npy file to load

    Returns
    -------
    data_dict : dict
        file_path: str
            Path to loaded npy file
        file_name:
            Loaded file name        
        F_sessionwise: float [neurons x time_points]
            Fluoroscence trace for the whole session
        F_trialwise_all: float [time points x neurons x trials]
            Fluoroscence trace trialwise, each trial contains 40 timepoints from the previous trial for a better comparison of changes
        F_trialwise_closed_loop: float [time points x neurons x trials]
            Fluoroscence trace trialwise, only for closedloop trials
        dff_sessionwise: float [neurons x time_points]
            Delta F over F sessionwise
        dff_trialwise_all: float [time points x neurons x trials]
            Delta F over F trialwise, all trials
        dff_trialwise_closed_loop: float [time points x neurons x trials]
            Delta F over F trialwise, closedloop trials
        cn: int
            Conditioned neuron index for the session
        roiX: float [neurons x 1]
            X-coordinate of the roi center for each neuron
        roiY: float [neurons x 1]
            Y-coordinate of the roi center for each neuron
        dist: float [neurons x 1] 
            distance from the conditioned neuron
        FOV: str
            Field of view for the session
        session_date: str
            Date of the recording
        session_path: str
            Filepath of the session recording
        mouse: str
            Mouse Name
        mean_image: [X-size x Y-size]
            Mean image of the field of view
        max_image : [X-size x Y-size]
            Max image of the field of view
        time_since_trial_start: [time points x 1]
            Time stamps for each frame
        go_cue_times: float [trials x 1]
            Time relative to the trial start when the go-cue arrives, usually same for all the trials
        lick_times: float
            Lickport licks, time relative to trial start
        reward_times: float
            reward times, time relative to trial start
        trial_times: float
            Time(in s) of the trials, we usually cut off the trace for F_trialwise
        hit: bool, [trial x 1]
            Boolean to represent if the trial is successful or not
        threshold_crossing_times: float [trial x 1]
            Time from the trial start when the lickport crosses the threshold
        zaber_move_forward: float,
            Motor step time, relative to trial start
        sampling_rate: float
            sampling rate for the camera
        all_si_filenames: str
            list of filenames corresponding to all trials
        closed_loop_filenames: str 
            list of filenames corresponding to closedloop trials
    """
    file_path,file_name = os.path.split(file_path)
    data = np.load(file_path, allow_pickle=True).tolist()

    data_dict ={'file_path':file_path,
                'file_name':file_name,
                'subject':data['mouse'],
                'F_sessionwise': data["F_sessionwise"],
                'F_trialwise_all': data["F_trialwise_all"],
                'F_trialwise_closed_loop': data["F_trialwise_closed_loop"],
                'dff_sessionwise': data["dff_sessionwise"],
                'dff_trialwise_all': data["dff_trialwise_all"],
                'dff_trialwise_closed_loop': data["dff_trialwise_closed_loop"],
                'cn': data["cn"],
                'roiX': data["roiX"],
                'roiY': data["roiY"],
                'dist': data["dist"],
                'FOV': data["FOV"],
                'session_date': data["session_date"],
                'session_path': data["session_path"],
                'mouse': data["mouse"],
                'mean_image': data["mean_image"],
                'max_image': data["max_image"],
                'time_since_trial_start': data["time_since_trial_start"],
                'go_cue_times': data["go_cue_times"],
                'lick_times': data["lick_times"],
                'reward_times': data["reward_times"],
                'trial_times': data["trial_times"],
                'hit': data["hit"],
                'threshold_crossing_times': data["threshold_crossing_times"],
                'zaber_move_forward': data["zaber_move_forward"],
                'sampling_rate': data["sampling_rate"],
                'all_si_filenames': data["all_si_filenames"],
                'closed_loop_filenames': data["closed_loop_filenames"],
            }
        
    return dict_all
