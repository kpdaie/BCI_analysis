#%% load DLC traces and align to traces
import numpy as np
import os, json
import pandas as pd
import matplotlib.pyplot as plt


def compare_dlc_bpod_timing(dlc_base_dir,session_bpod_file,trial_needed):
    """
    Compares lickport movement on the side camera to lickport steps in the bpod output.
    produces a single plot

    Parameters
    ----------
    dlc_base_dir : str
        where the dlc files are for this setup
    session_bpod_file : str
        bpod file location
    trial_needed : int
        trial number to plot
        
    Example
    -------
    dlc_base_dir = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Behavior_videos/DLC_output/Bergamo-2P-Photostim/'
    session_bpod_file = '/mnt/Data/Behavior/BCI_exported/KayvonScope/BCI_26/041522-bpod_zaber.npy'
    trial_needed = 66
    compare_dlc_bpod_timing(dlc_base_dir,session_bpod_file,trial_needed)
    Returns
    -------
    None.

    """

    camera = 'side'
    behavior_dict = np.load(session_bpod_file,allow_pickle = True).tolist()
    behavior_video_list = behavior_dict['behavior_movie_name_list']
    
    for trial_i, video_files in enumerate(behavior_video_list):
        if trial_needed>trial_i:
            continue
        if not type(video_files) == list:
            video_files = video_files.tolist()
        #%
        for video_file_path in video_files:
            video_found = False
            video_file_path.strip("' ")
            #
            
            print(video_file_path)
            if camera in video_file_path:
                video_found = True
                #
                break
    
        if not video_found:
            print('{} camera video not found in trial {}'.format(camera,trial_i))
            continue
        video_path,video_file = os.path.split(video_file_path)
        video_path = video_path[video_path.find(camera)+len(camera)+1:]
        dlc_files = os.listdir(os.path.join(dlc_base_dir,camera,video_path))
        for dlc_file in dlc_files:
            if video_file[:-5] in dlc_file:
                if '.json' in dlc_file:
                    with open(os.path.join(dlc_base_dir,camera,video_path,dlc_file)) as f:
                        trial_metadata = json.load(f)
                elif '.csv' in dlc_file:
                    dlcdata = pd.read_csv(os.path.join(dlc_base_dir,camera,video_path,dlc_file),index_col = 0, header = None)
                    bodyparts = np.unique(dlcdata.loc['bodyparts'])
                    dlc_dict = {}
                    for bodypart in bodyparts:
                        dlc_dict[bodypart] = dict()
                    for colname in dlcdata.keys():
                        dlc_dict[dlcdata[colname]['bodyparts']][dlcdata[colname]['coords']] = np.asarray(dlcdata[colname][3:].values,float)
                    
        
        #break
        #print(trial_i)
        if trial_i == trial_needed:
            video_frame_times  = np.asarray(trial_metadata['frame_times'])#+.05
            fig = plt.figure()
            ax_lickport = fig.add_subplot(1,1,1)
            ax_lickport_calculated = ax_lickport.twinx()
            ax_lickport.plot(video_frame_times,dlc_dict['Lickport']['x'],'r-',label = 'lickport X position (DLC)')
            lickport_steps = behavior_dict['zaber_move_forward'][trial_i]
            calculated_position = np.zeros_like(video_frame_times)
            for step in lickport_steps:
                idx = np.argmax(video_frame_times>step)
                
                calculated_position[idx:]+=1
                
            ax_lickport_calculated.plot(video_frame_times,calculated_position,'k',label = 'calculated lickport position')
            ax_lickport_calculated.legend()
            ax_lickport.legend()
            ax_lickport.set_title('trial {}'.format(trial_i))
            break
        if trial_i > trial_needed:
            break
