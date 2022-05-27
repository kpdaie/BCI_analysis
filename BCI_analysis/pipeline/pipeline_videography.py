#%% load DLC traces and align to traces
import numpy as np
import os, json, re
import pandas as pd
dlc_base_dir = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Behavior_videos/DLC_output/Bergamo-2P-Photostim/'
camera = 'side'
session_bpod_file = '/mnt/Data/Behavior/BCI_exported/KayvonScope/BCI_26/040522-bpod_zaber.npy'
behavior_dict = np.load(session_bpod_file,allow_pickle = True).tolist()
behavior_video_list = behavior_dict['behavior_movie_name_list']
for trial_i, video_files in enumerate(behavior_video_list):
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
    if trial_i == 4:
        video_frame_times  = np.asarray(trial_metadata['frame_times'])#+.05
        fig = plt.figure()
        ax_lickport = fig.add_subplot(1,1,1)
        ax_lickport.plot(video_frame_times,dlc_dict['Lickport']['x'],'r-',label = 'lickport X position (DLC)')
        lickport_steps = behavior_dict['zaber_move_forward'][trial_i]
        ax_lickport.plot(lickport_steps, np.zeros_like(lickport_steps)+250,'k|')
        break