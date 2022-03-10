#%%
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import time
import os
import pickle
import shutil
import json
#%%
paths = ['/home/rozmar/Data/Behavior/Behavior_rigs/KayvonScope',
         r'C:\Users\bpod\Documents\Pybpod',
         r'C:\Users\labadmin\Documents\Pybpod']

def load_dir_stucture(projectdir,projectnames_needed = None, experimentnames_needed = None,  setupnames_needed=None):
    """
    Explores the directory structure of a pybpod project.

    Parameters
    ----------
    projectdir : str or pathdir.Path()
        the path to the pybpod project of interest
    projectnames_needed : list of str, optional
        constrain project name(s). The default is None.
    experimentnames_needed : list of str, optional
        constrain experiment name(s). The default is None.
    setupnames_needed : list of str, optional
        constrain setup name(s). The default is None.

    Returns
    -------
    out_dict : dict
        Dictionary containing directory structure and all unique project, 
        experiment, setup, session and subject names.

    """
    dirstructure = dict()
    projectnames = list()
    experimentnames = list()
    setupnames = list()
    sessionnames = list()
    subjectnames = list()
    if type(projectdir) != type(Path()):
        projectdir = Path(projectdir)
    for projectname in projectdir.iterdir():
        if projectname.is_dir() and (not projectnames_needed or projectname.name in projectnames_needed):
            dirstructure[projectname.name] = dict()
            projectnames.append(projectname.name)
            
            for subjectname in (projectname / 'subjects').iterdir():
                if subjectname.is_dir() : 
                    subjectnames.append(subjectname.name)            
            
            for experimentname in (projectname / 'experiments').iterdir():
                if experimentname.is_dir() and (not experimentnames_needed or experimentname.name in experimentnames_needed ): 
                    dirstructure[projectname.name][experimentname.name] = dict()
                    experimentnames.append(experimentname.name)
                    
                    for setupname in (experimentname / 'setups').iterdir():
                        if setupname.is_dir() and (not setupnames_needed or setupname.name in setupnames_needed ): 
                            setupnames.append(setupname.name)
                            dirstructure[projectname.name][experimentname.name][setupname.name] = list()
                            
                            for sessionname in (setupname / 'sessions').iterdir():
                                if sessionname.is_dir(): 
                                    sessionnames.append(sessionname.name)
                                    dirstructure[projectname.name][experimentname.name][setupname.name].append(sessionname.name)
    out_dict = {'dir_structure':dirstructure,
                'project_names':projectnames,
                'experiment_names':experimentnames,
                'setup_names':setupnames,
                'session_names':sessionnames,
                'subject_names':subjectnames}
    return out_dict

def pybpod_csv_to_dataframe(csvfilename,subject_needed = ''):
    """
    Reads a pybpod .csv file, handles exceptions, assigns trial number to each 
    line, fills in blanks, returns a pandas dataframe.

    Parameters
    ----------
    csvfilename : str
        path to .csv file of interest
    subject_needed : str, optional
        If passed, the csv file will be parsed only if it belongs to the
        specified subject, otherwise retunrs  None.

    Returns
    -------
    df : pandas.DataFrame()
        slightly digested csv file.

    """
    df = pd.read_csv(csvfilename,delimiter=';',skiprows = 6)
    df = df[df['TYPE']!='|'] # delete empty rows
    df = df[df['TYPE']!= 'During handling of the above exception, another exception occurred:'] # delete empty rows
    df = df[df['MSG']!= ' '] # delete empty rows
    df = df[df['MSG']!= '|'] # delete empty rows
    df = df.reset_index(drop=True) # resetting indexes after deletion
    try:
        df['PC-TIME']=df['PC-TIME'].apply(lambda x : datetime.strptime(x,'%Y-%m-%d %H:%M:%S.%f')) # converting string time to datetime
    except ValueError: # sometimes pybpod don't write out the whole number...
        badidx = df['PC-TIME'].str.find('.')==-1
        if len(df['PC-TIME'][badidx]) == 1:
            df['PC-TIME'][badidx] = df['PC-TIME'][badidx]+'.000000'
        else:
            df['PC-TIME'][badidx] = [df['PC-TIME'][badidx]+'.000000']
        df['PC-TIME']=df['PC-TIME'].apply(lambda x : datetime.strptime(x,'%Y-%m-%d %H:%M:%S.%f')) # converting string time to datetime
    tempstr = df['+INFO'][df['MSG']=='CREATOR-NAME'].values[0]
    experimenter = tempstr[2:tempstr[2:].find('"')+2] #+2
    tempstr = df['+INFO'][df['MSG']=='SUBJECT-NAME'].values[0]
    subject = tempstr[2:tempstr[2:].find("'")+2] #+2
    setup = df['+INFO'][df['MSG']=='SETUP-NAME'].values[0]
    if len(subject_needed)>0 and subject.lower() != subject_needed.lower():
        return None
    df['experimenter'] = experimenter
    df['subject'] = subject
    df['setup'] = setup
    # adding trial numbers in session
    idx = (df[df['TYPE'] == 'TRIAL']).index.to_numpy()
    idx = np.concatenate(([0],idx,[len(df)]),0)
    idxdiff = np.diff(idx)
    Trialnum = np.array([])
    for i,idxnumnow in enumerate(idxdiff): #zip(np.arange(0:len(idxdiff)),idxdiff):#
        Trialnum  = np.concatenate((Trialnum,np.zeros(idxnumnow)+i),0)
    df['Trial_number_in_session'] = Trialnum
    indexes = df[df['MSG'] == 'Trialnumber:'].index + 1 #+2
    if len(indexes)>0:
        if 'Trial_number' not in df.columns:
            df['Trial_number']=np.NaN
        trialnumbers_real = df['MSG'][indexes]
        trialnumbers = df['Trial_number_in_session'][indexes].values
        for trialnumber_real,trialnum in zip(trialnumbers_real,trialnumbers):
            #df['Trial_number'][df['Trial_number_in_session'] == trialnum] = int(blocknumber)
            try:
                df.loc[df['Trial_number_in_session'] == trialnum, 'Trial_number'] = int(trialnumber_real) - 1  # Note that in the pybpod protocol the trial number comes BEFORE the go cue, so the numbering is off by one
            except:
                df.loc[df['Trial_number_in_session'] == trialnum, 'Block_number'] = np.nan
    # saving variables (if any)
    variableidx = (df[df['MSG'] == 'Variables:']).index.to_numpy()
    if len(variableidx)>0:
        d={}
        exec('variables = ' + df['MSG'][variableidx+1].values[0], d)
        for varname in d['variables'].keys():
            if isinstance(d['variables'][varname], (list,tuple)):
                templist = list()
                for idx in range(0,len(df)):
                    templist.append(d['variables'][varname])
                df['var:'+varname]=templist
            else:
                df['var:'+varname] = d['variables'][varname]
    # updating variables
    variableidxs = (df[df['MSG'] == 'Variables updated:']).index.to_numpy()
    for variableidx in variableidxs:
        d={}
        exec('variables = ' + df['MSG'][variableidx+1], d)
        for varname in d['variables'].keys():
            if isinstance(d['variables'][varname], (list,tuple)):
                templist = list()
                idxs = list()
                for idx in range(variableidx,len(df)):
                    idxs.append(idx)
                    templist.append(d['variables'][varname])
                df['var:'+varname][variableidx:]=templist.copy()
            else:
                #df['var:'+varname][variableidx:] = d['variables'][varname]
                df.loc[range(variableidx,len(df)), 'var:'+varname] = d['variables'][varname]

    return df


def pybpod_dataframe_to_dict(data):
    """
    Extracts trial based information from pandas dataframes

    Parameters
    ----------
    data : pandas dataframe
        the output of pybpod_csv_to_dataframe() function

    Returns
    -------
    data_dict : TYPE
        Dictionary with trial based information..
    """
    Zaber_moves_channel = ['Wire1High','Wire1Low'] # TODO this is hard coded, should be in the variables
    trial_start_idxs = data.loc[data['TYPE'] == 'TRIAL'].index.to_numpy()
    trial_end_idxs = data.loc[data['TYPE'] == 'END-TRIAL'].index.to_numpy()
    
    #threshold_passed_idx =
    if len(trial_start_idxs) > len(trial_end_idxs):
        trial_end_idxs = np.concatenate([trial_end_idxs,[len(data)-1]])
    
    data_dict = {'go_cue_times':list(),
                 'trial_start_times':list(),
                 'trial_end_times':list(),
                 'lick_L':list(),
                 'lick_R':list(),
                 'lick_L_end':list(),
                 'lick_R_end':list(),
                 'reward_L':list(),
                 'reward_R':list(),
                 'autowater_L':list(),
                 'autowater_R':list(),
                 'zaber_move_forward': list(),
                 'trial_hit':list(),
                 'time_to_hit':list(),
                 'trial_num':list(),
                 'threshold_crossing_times':list(),    
                 'behavior_movie_name_list':list(),
                 'scanimage_message_list':list()
                 }

    for key_now in data.keys():
        if 'var:'in key_now:
            data_dict[key_now.replace(':','_')]= list()
    
    for trial_num,(trial_start_idx, trial_end_idx) in enumerate(zip(trial_start_idxs,trial_end_idxs)):
        df_trial =  data[trial_start_idx:trial_end_idx]
        try:
            df_past_trial = data[trial_end_idx:trial_start_idxs[trial_num+1]]
        except:
            df_past_trial = data[trial_end_idx:]
            #%
        behavior_movie_names = 'no behavior video'
        scanimage_message = 'no scanimage message'
        for past_trial_line in  df_past_trial.iterrows():
            past_trial_line = past_trial_line[1]
            if 'Movie names for trial:' in past_trial_line['MSG']:
                behavior_movie_names = past_trial_line['MSG'][23:].strip('[]').split(',')
            if 'scanimage file' in past_trial_line['MSG']:
                scanimage_message = past_trial_line['MSG'][16:]
            
            
        #%
        #TODO df_past_trial contains the scanimage file name and the camera file names
        trial_start_time = data['PC-TIME'][trial_start_idx]
        trial_end_time = data['PC-TIME'][trial_end_idx]
        go_cue_time = df_trial.loc[(df_trial['MSG'] == 'GoCue') & (df_trial['TYPE'] == 'TRANSITION'),'BPOD-INITIAL-TIME'].values#[0]#.index.to_numpy()[0]
        threshold_crossing_time = df_trial.loc[(df_trial['MSG'] == 'ResponseInRewardZone') & (df_trial['TYPE'] == 'TRANSITION'),'BPOD-INITIAL-TIME'].values#[0]#.index.to_numpy()[0]
        if len(go_cue_time) == 0:
            continue # no go cue no trial
            
            
        lick_left_times = df_trial.loc[data['var:WaterPort_L_ch_in'] == data['+INFO'],'BPOD-INITIAL-TIME'].values
        lick_right_times = df_trial.loc[data['var:WaterPort_R_ch_in'] == data['+INFO'],'BPOD-INITIAL-TIME'].values
        lick_left_times_end = df_trial.loc[data['var:WaterPort_L_ch_out'] == data['+INFO'],'BPOD-INITIAL-TIME'].values
        lick_right_times_end = df_trial.loc[data['var:WaterPort_R_ch_out'] == data['+INFO'],'BPOD-INITIAL-TIME'].values
        reward_left_times = df_trial.loc[(data['MSG'] == 'Reward_L') & (data['TYPE'] == 'TRANSITION'),'BPOD-INITIAL-TIME'].values
        reward_right_times = df_trial.loc[(data['MSG'] == 'Reward_R') & (data['TYPE'] == 'TRANSITION'),'BPOD-INITIAL-TIME'].values
        autowater_left_times = df_trial.loc[(data['MSG'] == 'Auto_Water_L') & (data['TYPE'] == 'TRANSITION'),'BPOD-INITIAL-TIME'].values
        autowater_right_times = df_trial.loc[(data['MSG'] == 'Auto_Water_R') & (data['TYPE'] == 'TRANSITION'),'BPOD-INITIAL-TIME'].values
        try:
            ITI_start_times = df_trial.loc[(data['MSG'] == 'ITI') & (data['TYPE'] == 'TRANSITION'),'BPOD-INITIAL-TIME'].values[0]
        except:
            ITI_start_times = np.nan
        zaber_motor_movement_times = df_trial.loc[(Zaber_moves_channel[0] == data['+INFO']) | (Zaber_moves_channel[1] == data['+INFO']),'BPOD-INITIAL-TIME'].values
        trial_number = df_trial.loc[(df_trial['MSG'] == 'GoCue') & (df_trial['TYPE'] == 'TRANSITION'),'Trial_number'].values[0]
        
        data_dict['trial_num'].append(trial_number)
        data_dict['go_cue_times'].append(go_cue_time)
        data_dict['trial_start_times'].append(trial_start_time)
        data_dict['trial_end_times'].append(trial_end_time)
        data_dict['lick_L'].append(lick_left_times)
        data_dict['lick_R'].append(lick_right_times)
        data_dict['lick_L_end'].append(lick_left_times_end)
        data_dict['lick_R_end'].append(lick_right_times_end)
        data_dict['reward_L'].append(reward_left_times)
        data_dict['reward_R'].append(reward_right_times)
        data_dict['autowater_L'].append(autowater_left_times)
        data_dict['autowater_R'].append(autowater_right_times)
        data_dict['zaber_move_forward'].append(zaber_motor_movement_times)
        data_dict['threshold_crossing_times'].append(threshold_crossing_time)
        data_dict['behavior_movie_name_list'].append(behavior_movie_names)
        data_dict['scanimage_message_list'].append(scanimage_message)

        for key_now in data.keys():
            if 'var:'in key_now:
                data_dict[key_now.replace(':','_')].append(df_trial.loc[(df_trial['MSG'] == 'GoCue') & (df_trial['TYPE'] == 'TRANSITION'),key_now].values[0])
            
        
            
        reward_times = np.concatenate([reward_left_times,reward_right_times])
        if len(reward_times)>0:
            data_dict['trial_hit'].append(True)
            data_dict['time_to_hit'].append(reward_times[0]-go_cue_time)
        else:
            data_dict['trial_hit'].append(False)
            data_dict['time_to_hit'].append(np.nan)
    for key_now in data_dict.keys():
        data_dict[key_now] = np.asarray(data_dict[key_now])
            
    return data_dict
        
def add_zaber_info_to_pybpod_dict(behavior_dict,
                                   subject_name,
                                   setup_name,
                                   zaber_folder_root):
    """
    extracts zaber and BCI settings (generated by the Zaber gui) from zaber json files and aligns it to trials.
    Works properly if zaber GUI and pybpod are running on the same PC.
    
    Parameters
    ----------
    behavior_dict : dictionary
        output o pybpod_dataframe_to_dict() function
    subject_name : str
        subject name as appears in pybod and zaber software
    setup_name : str
        name of the setup the recording was made on
    zaber_folder_root : str
        path to zaber data

    Returns
    -------
    zaber_vars_dict : TYPE
        DESCRIPTION.

    """
    if setup_name == 'DOM3':
        setup_dirname = 'DOM3-MMIMS'
    else:
        setup_dirname = setup_name
    try:
        zaberdir = os.path.join(zaber_folder_root,setup_dirname,'subjects',subject_name)
        zaberfiles = np.sort(os.listdir(zaberdir))[::-1]
    except:
        zaberdir = os.path.join(r'W:\Users\labadmin\Documents\BCI_Zaber_data','subjects',subject_name)
        zaberfiles = np.sort(os.listdir(zaberdir))[::-1]
    zabertimes = list()
    for zaberfile in zaberfiles:
        zabertime = datetime.strptime(zaberfile[:-5],'%Y-%m-%d_%H-%M-%S')
        zabertimes.append(zabertime)
    zabertimes = np.asarray(zabertimes)
    zaber_file_idx_prev= np.nan
    zaber_vars_dict = {'acceleration':list(),
                       'direction':list(),
                       'limit_close':list(),
                       'limit_far':list(),
                       'reward_zone':list(),
                       'speed':list(),
                       'trigger_step_size':list(),
                       'max_speed':list(),
                       'microstep_size':list()}
    for trial_start_time in behavior_dict['trial_start_times']:
        zaber_file_idx = np.argmax(zabertimes<trial_start_time)
        if zaber_file_idx  != zaber_file_idx_prev:
            with open(os.path.join(zaberdir,zaberfiles[zaber_file_idx]), "r") as read_file:
                zaber_dict= json.load(read_file)
            zaber_file_idx_prev = zaber_file_idx
        for zaber_key in zaber_vars_dict.keys():
            zaber_vars_dict[zaber_key].append(zaber_dict['zaber'][zaber_key])
    for zaber_key in zaber_vars_dict.keys():
        zaber_vars_dict[zaber_key] = np.asarray(zaber_vars_dict[zaber_key])
        
    return zaber_vars_dict  
