from pybpodgui_api.models.project import Project
from scipy.io import savemat
from pathlib import Path
import os
from statistics import mode
import numpy as np

import datetime



def find_pybpod_sessions(subject_names_list,
                         date_now,
                         projects):
    """
    This scrip uses the pybpodgui-api package to sweep through projects and 
    find sessions for a given subject on a given date.
    
    Parameters
    ----------
    subject_names_list : str or list if str
        subject names of interest
    date_now : str
        date of the session, pybpod style: %Y%m%d
    projects : list of pybpodgui_api.models.project.Project
        pybpod projects that we need to go through.

    Returns
    -------
    outdict : dictionary
        dictionary that contains the sessions of interest, session start times,
        and experiment names.
    """
    if type(subject_names_list) != list:
        subject_names_list = [subject_names_list]
    sessions_now = list()
    session_start_times_now = list()
    experimentnames_now = list()
    for proj in projects: #
        exps = proj.experiments
        for exp in exps:
            stps = exp.setups
            for stp in stps:
                #sessions = stp.sessions
                for session in stp.sessions:
                    for subject_now in subject_names_list:
                        if session.subjects and session.subjects[0].find(subject_now) >-1 and session.name.startswith(date_now):
                            sessions_now.append(session)
                            session_start_times_now.append(session.started)
                            experimentnames_now.append(exp.name)
    order = np.argsort(session_start_times_now)
    outdict = {'sessions':np.asarray(sessions_now)[order],
               'session_start_times':np.asarray(session_start_times_now)[order],
               'experiment_names':np.asarray(experimentnames_now)[order]}
    return outdict


def export_pybpod_files_core(bpod_session_dict,
                             calcium_imaging_raw_session_dir,
                             zaber_root_folder): # 
    """
    Core function that aligns bpod, zaber and imaging modalities.

    Parameters
    ----------
    bpod_session_dict : dict
        list of sessions to be used, output of find_pybpod_sessions()
    calcium_imaging_raw_session_dir : str
        path to the tiff files 
    zaber_root_folder : str
        path to root zaber data

    Returns
    -------
    behavior_dict : dict
        trial-based behavior dictinary with aligned zaber settings and 
        scanimage tiff files

    """
    behavior_dict_list = list()
    sessionfile_start_times = list()
    for sessionfile in bpod_session_dict['sessions']:
        csvfile = sessionfile.filepath
        csv_data = io.io_pybpod.pybpod_csv_to_dataframe(csvfile)
        behavior_dict = io.io_pybpod.pybpod_dataframe_to_dict(csv_data)
        subject_name = csv_data['subject'][0]
        setup_name = csv_data['setup'][0]
        experimenter_name = csv_data['experimenter'][0]
        zaber_dict = io.io_pybpod.add_zaber_info_to_pybpod_dict(behavior_dict,
                                                                subject_name,
                                                                setup_name,
                                                                zaber_root_folder)
        
        for key in zaber_dict.keys():
            behavior_dict['zaber_{}'.format(key)] = zaber_dict[key]
        trialnum = len(behavior_dict['trial_num'])
        behavior_dict['bpod_file_names'] = np.asarray([sessionfile.filepath.split('/')[-1]]*trialnum)
        
        behavior_dict['subject_name'] = np.asarray([subject_name]*trialnum)
        behavior_dict['setup_name'] = np.asarray([setup_name]*trialnum)
        behavior_dict['experimenter_name'] = np.asarray([experimenter_name]*trialnum)
        if trialnum>0:
            behavior_dict_list.append(behavior_dict)
            sessionfile_start_times.append(behavior_dict['trial_start_times'][0])
        #%
    if len(behavior_dict_list) == 0:
        print('no behavior found')
        #timer.sleep(10)
        return None
    #if  len(behavior_dict_list)>1:
        #%
    order = np.argsort(sessionfile_start_times)
    behavior_dict_list = np.asarray(behavior_dict_list)[order]
    behavior_dict = {}
    for key in behavior_dict_list[0].keys():
        keylist = list()
        for behavior_dict_now in behavior_dict_list:
            for element in behavior_dict_now[key]:
                keylist.append(element)
        behavior_dict[key] = np.asarray(keylist)
        
    
    files = io.io_utils.extract_files_from_dir(calcium_imaging_raw_session_dir)
    tiffiles = files['exts']=='.tif'
    uniquebasenames = np.unique(files['basenames'][tiffiles])
    filenames_all = list()
    frame_timestamps_all = list()
    nextfile_timestamps_all = list()
    acqtrigger_timestamps_all = list()
    trigger_arrived_timestamps_all = list()
    scanimage_integration_roi_data_all = list()
    scanimage_metadata_dict_list = list()
    for basename in uniquebasenames:
        #%
        file_idxs_now = (files['exts']=='.tif') & (files['basenames']==basename)
        filenames = files['filenames'][file_idxs_now]
        fileindices = files['fileindices'][file_idxs_now]
        order = np.argsort(fileindices)
        filenames = filenames[order]
        
        for filename in filenames:
            try:
                metadata = io.io_scanimage.extract_scanimage_metadata(os.path.join(calcium_imaging_raw_session_dir,filename))
                
            except:
                print('tiff file read error: {}'.format(os.path.join(calcium_imaging_raw_session_dir,filename)))
                continue
            
            movie_start_time = metadata['movie_start_time']
            
            if float(metadata['description_first_frame']['frameTimestamps_sec'])<0:
                valence=-1
            else:
                valence = 1
            frame_timestamp = movie_start_time+valence*datetime.timedelta(seconds = np.abs(float(metadata['description_first_frame']['frameTimestamps_sec'])))
            
            if float(metadata['description_first_frame']['acqTriggerTimestamps_sec'])<0:
                valence=-1
            else:
                valence = 1
            acqtrigger_timestamp = movie_start_time+valence*datetime.timedelta(seconds = np.abs(float(metadata['description_first_frame']['acqTriggerTimestamps_sec'])))
            if float(metadata['description_first_frame']['nextFileMarkerTimestamps_sec'])<0:
                valence=-1
            else:
                valence = 1
            nextfile_timestamp = movie_start_time+valence*datetime.timedelta(seconds = np.abs(float(metadata['description_first_frame']['nextFileMarkerTimestamps_sec'])))
            if float(metadata['description_first_frame']['nextFileMarkerTimestamps_sec']) == -1:
                trigger_arrived_timestamp =acqtrigger_timestamp
            else:
                trigger_arrived_timestamp = nextfile_timestamp
                
            filenames_all.append(filename)
            frame_timestamps_all.append(frame_timestamp)
            nextfile_timestamps_all.append(nextfile_timestamp)
            acqtrigger_timestamps_all.append(acqtrigger_timestamp)
            scanimage_metadata_dict_list.append(metadata)
            
            #trignextstopenable = 'true' in metadata['metadata']['hScan2D']['trigNextStopEnable'].lower()
            if metadata['metadata']['extTrigEnable'] == '0':
                trigger_arrived_timestamps_all.append(np.nan)
                print('{} is not triggered'.format(filename))
            else:
                trigger_arrived_timestamps_all.append(trigger_arrived_timestamp)
            try:
                integration_roi_data = {}
                integration_roi_data['outputChannelsEnabled'] = np.asarray(metadata['metadata']['hIntegrationRoiManager']['outputChannelsEnabled'].strip('[]').split(' '))=='true'
                integration_roi_data['outputChannelsNames'] = metadata['metadata']['hIntegrationRoiManager']['outputChannelsNames'].strip("{}").replace("'","").split(' ')
                integration_roi_data['outputChannelsRoiNames'] = eval(metadata['metadata']['hIntegrationRoiManager']['outputChannelsRoiNames'].replace('{','[').replace('}',']').replace(' ',','))
                integration_roi_data['outputChannelsFunctions'] = eval(metadata['metadata']['hIntegrationRoiManager']['outputChannelsFunctions'].replace('{','[').replace('}',']').replace(' ',','))
            except:
                integration_roi_data = {}
                integration_roi_data['outputChannelsEnabled'] = []
                integration_roi_data['outputChannelsNames'] = []
                integration_roi_data['outputChannelsRoiNames'] = []
                integration_roi_data['outputChannelsFunctions'] = []
                print('no ROI data')
            scanimage_integration_roi_data_all.append(integration_roi_data)
                #print('triggered')
            
    #%
    filenames_all = np.asarray(filenames_all)
    frame_timestamps_all= np.asarray(frame_timestamps_all)
    nextfile_timestamps_all= np.asarray(nextfile_timestamps_all)
    acqtrigger_timestamps_all= np.asarray(acqtrigger_timestamps_all)
    trigger_arrived_timestamps_all = np.asarray(trigger_arrived_timestamps_all)
    scanimage_integration_roi_data_all = np.asarray(scanimage_integration_roi_data_all)
    scanimage_metadata_dict_list= np.asarray(scanimage_metadata_dict_list)
    
    
    file_order = np.argsort(frame_timestamps_all)
    
    filenames_all = filenames_all[file_order]
    frame_timestamps_all = frame_timestamps_all[file_order]
    nextfile_timestamps_all = nextfile_timestamps_all[file_order]
    acqtrigger_timestamps_all = acqtrigger_timestamps_all[file_order]
    trigger_arrived_timestamps_all = trigger_arrived_timestamps_all[file_order]
    scanimage_integration_roi_data_all = scanimage_integration_roi_data_all[file_order]
    scanimage_metadata_dict_list = scanimage_metadata_dict_list[file_order]
    
    istriggered = list()
    for stamp in trigger_arrived_timestamps_all: istriggered.append(type(stamp)==datetime.datetime)
    
    
    bpod_trial_start_times = behavior_dict['trial_start_times']
    #%
    dist_list = list()
    for i,t_now in enumerate(bpod_trial_start_times):
        for t_next in trigger_arrived_timestamps_all[istriggered]:
            dt = (t_next-t_now).total_seconds()
            if np.abs(dt)<50:
                dist_list.append(dt)
    dist_list = np.asarray(dist_list)
    #%
    residual_filenames = filenames_all
    residual_timestamps = trigger_arrived_timestamps_all
    residual_metadata = scanimage_metadata_dict_list
    if len(dist_list)>0:
        center_sec = mode(np.asarray(dist_list,int))
        dist_list = dist_list[(dist_list>center_sec-1) &  (dist_list<center_sec+1)]
        time_offset = np.median(dist_list)
        print('time_offset: {} s'.format(time_offset))
        #%
        bpod_trial_file_names = list()
        bpod_scanimage_time_offset = list()
        scanimage_frame_time_offset = list()
        scanimage_metadata_list = list()
        for roikey in integration_roi_data.keys():
            behavior_dict['scanimage_roi_{}'.format(roikey)] = list()
            
        for trial_start_time,trial_end_time in zip(behavior_dict['trial_start_times'],behavior_dict['trial_end_times']):
            trial_start_time = trial_start_time +datetime.timedelta(seconds = time_offset-.5) #gets a 0.5 second extra
            trial_end_time = trial_end_time +datetime.timedelta(seconds = time_offset)
            movie_idxes = (trigger_arrived_timestamps_all[istriggered]>trial_start_time) & (trigger_arrived_timestamps_all[istriggered]<trial_end_time)
            if any(movie_idxes):
                if sum(movie_idxes) == 1:
                    moviename = np.asarray(filenames_all[istriggered][movie_idxes])
                    scanimage_metadata = np.asarray(scanimage_metadata_dict_list[istriggered][movie_idxes])
                else:
                    moviename = np.asarray(filenames_all[istriggered][movie_idxes])
                    scanimage_metadata = np.asarray(scanimage_metadata_dict_list[istriggered][movie_idxes])
                movie_trial_time_offset = (trigger_arrived_timestamps_all[istriggered][movie_idxes][0]-(trial_start_time- datetime.timedelta(seconds = time_offset-.5))).total_seconds()
                trigger_to_frame_offset = (frame_timestamps_all[istriggered][movie_idxes][0]-trigger_arrived_timestamps_all[istriggered][movie_idxes][0]).total_seconds()
                roidata = scanimage_integration_roi_data_all[istriggered][movie_idxes][0]
            else:
                moviename = 'no movie for this trial'
                movie_trial_time_offset = np.nan
                trigger_to_frame_offset  = np.nan
                scanimage_metadata = np.nan
                roidata=np.nan
            bpod_trial_file_names.append(moviename)
            scanimage_metadata_list.append(scanimage_metadata)
            bpod_scanimage_time_offset.append(movie_trial_time_offset)
            scanimage_frame_time_offset.append(trigger_to_frame_offset)
            for moviename_now in moviename:
                idx = residual_filenames !=moviename_now
                residual_filenames = residual_filenames[idx]
                residual_timestamps = residual_timestamps[idx]
                residual_metadata = residual_metadata[idx]
            for roikey in integration_roi_data.keys():
                try:
                    behavior_dict['scanimage_roi_{}'.format(roikey)].append(roidata[roikey])
                except:
                    behavior_dict['scanimage_roi_{}'.format(roikey)].append([])
            #
            #%
        behavior_dict['scanimage_file_names'] = bpod_trial_file_names
        behavior_dict['scanimage_bpod_time_offset'] = np.asarray(bpod_scanimage_time_offset)
        behavior_dict['scanimage_first_frame_offset'] = np.asarray(scanimage_frame_time_offset)
        behavior_dict['scanimage_tiff_headers'] = np.asarray(scanimage_metadata_list)
    else:
        time_offset = np.nan
        print('no movie-behavior correspondance found ')
        behavior_dict['scanimage_file_names'] = 'no movie files found'

    residual_tiff_files = {'median_bpod_si_time_offset':time_offset,
                           'triggered':list(),
                           'time_from_previous_trial_start' : list(),
                           'time_to_next_trial_start' : list(),
                           'previous_trial_index': list(),
                           'next_trial_index' : list(),
                           'scanimage_file_names':list(),
                           'scanimage_tiff_headers':list()}
    skip_metadata = False
    for scanimage_fname, scanimage_timestamp,scanimage_metadata in zip(residual_filenames,residual_timestamps,residual_metadata):
        if type(scanimage_timestamp) == float:
            residual_tiff_files['triggered'].append(False)
            movie_start_time_now = frame_timestamps_all[filenames_all==scanimage_fname][0] - datetime.timedelta(seconds = time_offset)
        else:
            residual_tiff_files['triggered'].append(True)
            movie_start_time_now = scanimage_timestamp - datetime.timedelta(seconds = time_offset)
        if any(behavior_dict['trial_start_times']>movie_start_time_now):
            next_trial_idx = np.where(behavior_dict['trial_start_times']>movie_start_time_now)[0][0]
            time_to_next_trial = (behavior_dict['trial_start_times'][next_trial_idx]-movie_start_time_now).total_seconds()# + time_offset
        else:
            next_trial_idx = np.nan
            time_to_next_trial = np.nan
        
        if any(behavior_dict['trial_start_times']<movie_start_time_now):
            prev_trial_idx = np.where(behavior_dict['trial_start_times']<movie_start_time_now)[0][-1]
            time_from_prev_trial = (movie_start_time_now-behavior_dict['trial_start_times'][prev_trial_idx]).total_seconds() #- time_offset
        else:
            prev_trial_idx = np.nan
            time_from_prev_trial = np.nan
        
        residual_tiff_files['time_from_previous_trial_start'].append( time_from_prev_trial)
        residual_tiff_files['time_to_next_trial_start'].append( time_to_next_trial)
        
        residual_tiff_files['previous_trial_index'].append( prev_trial_idx)
        residual_tiff_files['next_trial_index'].append( next_trial_idx)
        residual_tiff_files['scanimage_file_names'].append(scanimage_fname)
        if 'slm' in scanimage_fname and not skip_metadata:
            residual_tiff_files['scanimage_tiff_headers'].append(scanimage_metadata)
            skip_metadata = True
        elif 'slm' in scanimage_fname and skip_metadata:
            residual_tiff_files['scanimage_tiff_headers'].append(None)
        else:
            residual_tiff_files['scanimage_tiff_headers'].append(scanimage_metadata)
    
    behavior_dict['residual_tiff_files'] = residual_tiff_files
    
    return behavior_dict

    
 
def export_single_pybpod_session(session,
                              subject_names,
                              save_dir,
                              calcium_imaging_raw_session_dir,
                              raw_behavior_dirs,
                              zaber_root_folder):
    """
    Main function that exports a single pybpod csv file, pairs them with scanimage
    tiffs and zaber json files and exports them all in a trial-based manner.
    Files are read out from a given directory structure and saved in a similar
    directory structure.

    Parameters
    ----------
    session : str
        session to export, it's a date formatted as %m%d%y or %Y-%m-%d
    subject_names : list of str
        names of subjects to export, or if the subject has multiple aliases, add all
    save_dir : str
        path where the file should be saved
    calcium_imaging_raw_session_dir : str
        path where the scanimage files can be found.
    raw_behavior_dirs : list
        list of directories where the behavior data should be shearched
    zaber_root_folder : str
        path to zaber data (one above the subjects folder).
    
    Example
    -------
    import BCI_analysis
    BCI_analysis.pipeline_bpod.export_single_pybpod_session(session = '040122',
                                 subject_names = ['BCI26'],
                                 save_dir= '/home/rozmar/Data/',
                                 calcium_imaging_raw_session_dir = '/mnt/Data/Calcium_imaging/raw/KayvonScope/BCI_26/040122',
                                 raw_behavior_dirs = ['/mnt/Data/Behavior/raw/KayvonScope/BCI'],
                                 zaber_root_folder = '/mnt/Data/Behavior/BCI_Zaber_data/KayvonScope')

    Returns
    -------
    None.

    """
    try:
        session_date = datetime.datetime.strptime(session,'%m%d%y')
    except:
        try:
            session_date = datetime.datetime.strptime(session,'%Y-%m-%d')
        except:
            try:
                session_date = datetime.datetime.strptime(session[:6],'%m%d%y')
            except:

                print('cannot understand date for session dir: {} - should be a date'.format(session))
            #continue
    
    projects = list()
    for projectdir in raw_behavior_dirs:
        projects.append(Project())
        projects[-1].load(projectdir)
    
    bpod_session_dict = find_pybpod_sessions(subject_names,session_date.strftime('%Y%m%d'),projects)
    behavior_dict = export_pybpod_files_core(bpod_session_dict,calcium_imaging_raw_session_dir,zaber_root_folder) 
    bpod_export_file = '{}-bpod_zaber.npy'.format(session)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(save_dir,bpod_export_file),behavior_dict)
    bpod_export_file = '{}-bpod_zaber.mat'.format(session)
    behavior_dict_matlab = behavior_dict.copy()
    behavior_dict_matlab['trial_start_times'] = np.asarray(behavior_dict['trial_start_times'],str)
    behavior_dict_matlab['trial_end_times'] = np.asarray(behavior_dict['trial_end_times'],str)
    behavior_dict_matlab.pop('scanimage_tiff_headers')
    try:
        behavior_dict_matlab['residual_tiff_files'].pop('scanimage_tiff_headers')
    except:
        pass
    savemat(os.path.join(save_dir,bpod_export_file),behavior_dict_matlab)
    

#%% this script will export behavior and pair it to imaging, then save it in a neat directory structure
def export_pybpod_files(behavior_export_basedir,
                        calcium_imaging_raw_basedir,
                        raw_behavior_dirs,
                        zaber_root_folder,
                        overwrite = False):
    """
    Main function that exports pybpod csv files, pairs them with scanimage
    tiffs and zaber json files and exports them all in a trial-based manner.
    Files are read out from a given directory structure and saved in a similar
    directory structure.
    Parameters
    ----------
    behavior_export_basedir : str
        path to directory where data will be exported
    calcium_imaging_raw_basedir : str
        path to directory where imagig data is found
    raw_behavior_dirs : list of str
        paths to directoies where pybpod projects are found
    zaber_root_folder : str
        path to directory where zaber data is found
    overwrite : Boolean, optional
        If set to False (default), existing files are skipped
    
    Example
    -------
    import BCI_analysis
    behavior_export_basedir = '/mnt/Data/Behavior/BCI_exported'
    calcium_imaging_raw_basedir = '/mnt/Data/Calcium_imaging/raw'
    raw_behavior_dirs = ['/mnt/Data/Behavior/raw/DOM-3/BCI',
                         '/mnt/Data/Behavior/raw/DOM3-MMIMS/BCI',
                         '/mnt/Data/Behavior/raw/KayvonScope/BCI']
    zaber_root_folder = '/mnt/Data/Behavior/BCI_Zaber_data'
    BCI_analysis.pipeline_bpod.export_pybpod_files(behavior_export_basedir,
                                                    calcium_imaging_raw_basedir,
                                                    raw_behavior_dirs, 
                                                    zaber_root_folder,
                                                    overwrite=False)
    Returns
    -------
    None.

    """

    projects = list()
    for projectdir in raw_behavior_dirs:
        projects.append(Project())
        projects[-1].load(projectdir)
    setups = os.listdir(calcium_imaging_raw_basedir)
    for setup in setups[::-1]:
        calcium_imaging_raw_setup_dir = os.path.join(calcium_imaging_raw_basedir,setup)
        subjects = np.sort(os.listdir(calcium_imaging_raw_setup_dir))
        for subject in subjects:
            if not ('bci' in subject.lower() or 'pkj' in subject.lower()):
                continue
            calcium_imaging_raw_subject_dir = os.path.join(calcium_imaging_raw_setup_dir,subject)
            if 'bci' in subject.lower():
                if subject.startswith('KH_'):
                    subject_wr_name = 'BCI{}'.format(subject[subject.find('_',3)+1:])
                else:
                    subject_wr_name = 'BCI{}'.format(subject[subject.find('_')+1:])
            elif 'pkj' in subject.lower():
                subject_wr_name = 'PKJ{}'.format(subject[subject.find('_')+1:])
            sessions = np.sort(os.listdir(calcium_imaging_raw_subject_dir))
            for session in sessions:
                try:
                    session_date = datetime.datetime.strptime(session,'%m%d%y')
                except:
                    try:
                        session_date = datetime.datetime.strptime(session,'%Y-%m-%d')
                    except:
                        try:
                            session_date = datetime.datetime.strptime(session[:6],'%m%d%y')
                        except:
                            print('cannot understand date for session dir: {}'.format(session))
                            continue
        
                
                calcium_imaging_raw_session_dir = os.path.join(calcium_imaging_raw_subject_dir,session)
                bpod_export_dir = os.path.join(behavior_export_basedir,setup,subject)
                Path(bpod_export_dir).mkdir(parents=True, exist_ok=True)
                bpod_export_file = '{}-bpod_zaber.npy'.format(session)
                if not overwrite and os.path.exists(os.path.join(bpod_export_dir,bpod_export_file)):
                    print('session already exported: {}'.format(os.path.join(bpod_export_dir,bpod_export_file)))
                    continue
                
                bpod_session_dict = find_pybpod_sessions([subject_wr_name,subject,subject_wr_name.lower(),subject.lower()],session_date.strftime('%Y%m%d'),projects)
               
                
                
                print('exporting behavior from {}/{}'.format(subject,session))
                try:
                    behavior_dict = export_pybpod_files_core(bpod_session_dict,calcium_imaging_raw_session_dir,zaber_root_folder) 
                except:
                    print('COULD NOT EXPORT SESSION: {}'.format(os.path.join(bpod_export_dir,bpod_export_file)))
                    behavior_dict = None
                if type(behavior_dict) != dict:
                    print('skipping this one')
                    continue
                
                bpod_export_dir = os.path.join(behavior_export_basedir,setup,subject)
                Path(bpod_export_dir).mkdir(parents=True, exist_ok=True)
                bpod_export_file = '{}-bpod_zaber.npy'.format(session)
                np.save(os.path.join(bpod_export_dir,bpod_export_file),behavior_dict)
                bpod_export_file = '{}-bpod_zaber.mat'.format(session)
                behavior_dict_matlab = behavior_dict.copy()
                behavior_dict_matlab['trial_start_times'] = np.asarray(behavior_dict['trial_start_times'],str)
                behavior_dict_matlab['trial_end_times'] = np.asarray(behavior_dict['trial_end_times'],str)
                behavior_dict_matlab.pop('scanimage_tiff_headers')
                try:
                    behavior_dict_matlab['residual_tiff_files'].pop('scanimage_tiff_headers')
                except:
                    pass
                savemat(os.path.join(bpod_export_dir,bpod_export_file),behavior_dict_matlab)
                print('{}/{} saved'.format(subject,session))

# =============================================================================
# #%%
# def check_discrepancies_in_behavior_export():
#     behavior_export_basedir = '/home/rozmar/Data/Behavior/BCI_exported'
#     calcium_imaging_raw_basedir = dj.config['locations.imagingdata_raw']
#     setups = os.listdir(behavior_export_basedir)
#     for setup in setups:
#         if '.' in setup:
#             continue
#         subjects = os.listdir(os.path.join(behavior_export_basedir,setup))
#         for subject in subjects:
#             sessions = os.listdir(os.path.join(behavior_export_basedir,setup,subject))
#             for session in sessions:
#                 if '.npy' not in session:
#                     continue
#                 print([subject,session])
#                 #%
#                 bpoddata = np.load(os.path.join(behavior_export_basedir,setup,subject,session),allow_pickle = True).tolist()
#                 raw_session_dir = os.path.join(calcium_imaging_raw_basedir,setup,subject,session[:-15])
#                 tiffiles = list()
#                 tiff_basenames = list()
#                 tiff_indices = list()
#                 h5files = list()
#                 h5_basenames = list()
#                 h5_indices = list()
#                 files = os.listdir(raw_session_dir)
#                 
#                 for tiffile in files:
#                     if '.tif' in tiffile:
#                        tiffiles.append(tiffile) 
#                        tiff_basenames.append(tiffile[:tiffile.rfind('_')])
#                        idx_string = tiffile[tiffile.rfind('_')+1:tiffile.rfind('.')]
#                        if '-' in idx_string:
#                             idxes = np.arange(int(idx_string[:idx_string.find('-')]),int(idx_string[idx_string.find('-')+1:])+1)
#                        elif ' (' in idx_string:
#                             idxes = [int(idx_string[:idx_string.find(' (')])]
#                        else:
#                             idxes = [int(idx_string)]
#                        tiff_indices.append(np.asarray(idxes))
#                     elif 'h5' in tiffile:
#                         h5files.append(tiffile) 
#                         h5_basenames.append(tiffile[:tiffile.rfind('_')])
#                         idx_string = tiffile[tiffile.rfind('_')+1:tiffile.rfind('.')]
#                         if '-' in idx_string:
#                             idxes = np.arange(int(idx_string[:idx_string.find('-')]),int(idx_string[idx_string.find('-')+1:])+1)
#                         elif ' (' in idx_string:
#                             idxes = [int(idx_string[:idx_string.find(' (')])]
#                         else:
#                             idxes = [int(idx_string)]
#                         h5_indices.append(np.asarray(idxes))
#                             
#                      #
#                 h5data = dict()
#                 
#                 for h5file in h5files:
#                     ephysdata_list = utils_ephys.load_wavesurfer_file(os.path.join(raw_session_dir,h5file))
#                     ephysdata_list_new = list()
#                     for ephysdata in ephysdata_list:
#                         ephysdata_list_new.append(utils_ephys.decode_bitcode(ephysdata))
#                     h5data[h5file] = np.asarray(ephysdata_list_new)
#                         #%
#                 tiffiles = np.asarray(tiffiles)
#                 residual_tiffiles = tiffiles
#                 try:
#                     print('median time offset: {}, std: {}'.format(np.nanmedian(bpoddata['scanimage_bpod_time_offset']),np.nanstd(bpoddata['scanimage_bpod_time_offset'])))
#                 except:
#                     print('no movie?')
#                     continue
#                 for trialnum, scanimagefile,timeoffset in zip(bpoddata['trial_num'],bpoddata['scanimage_file_names'],bpoddata['scanimage_bpod_time_offset']):
#                     if type(scanimagefile)!=str:#='no movie for this trial':
#                         for scanimagefile_now in scanimagefile:
#                             residual_tiffiles = residual_tiffiles[residual_tiffiles!=scanimagefile_now]
#                         scanimagefile = scanimagefile[0]
#                         tiff_basename =scanimagefile[:scanimagefile.rfind('_')]
#                         idx_string = scanimagefile[scanimagefile.rfind('_')+1:scanimagefile.rfind('.')]
#                         if ' (' in idx_string:
#                             tiff_idx = [int(idx_string[:idx_string.find(' (')])]
#                         else:
#                             tiff_idx = int(idx_string)
#                         for h5file,h5_base,h5_idx in zip(h5files,h5_basenames,h5_indices):
#                             if tiff_basename==h5_base:
#                                 if tiff_idx in h5_idx:
#                                     h5data[h5file][h5_idx==tiff_idx]
#                                     bitcode_trial_num = h5data[h5file][h5_idx==tiff_idx][0]['bitcode_trial_nums'][0]
#                                     if trialnum != bitcode_trial_num and bitcode_trial_num > 0:
#                                         print('trialnum does not line up: {} vs {} in {} with {}'.format(trialnum,bitcode_trial_num,os.path.join(raw_session_dir,scanimagefile),h5file))
#                                         print('timeoffset is {}'.format(timeoffset))
#                                         
# 
# =============================================================================
