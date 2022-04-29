import BCI_analysis
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt
file_path = '/mnt/Data/Calcium_imaging/Analysis_Kayvon_pipeline/multi-session/BCI22_030222v8.mat'

data_dict = BCI_analysis.io_matlab.read_multisession_mat(file_path)

#BCI_analysis.plot.plot_imaging.plot_trial_averaged_trace_2sessions(data_dict,1)

#%% behavior pipeline
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
#%% behavior -single session
import BCI_analysis

BCI_analysis.pipeline_bpod.export_single_pybpod_session(session = '040122',
                             subject_names = ['BCI26'],
                             save_dir= '/home/rozmar/Data/',
                             calcium_imaging_raw_session_dir = '/mnt/Data/Calcium_imaging/raw/KayvonScope/BCI_26/040122',
                             raw_behavior_dirs = ['/mnt/Data/Behavior/raw/KayvonScope/BCI'],
                             zaber_root_folder = '/mnt/Data/Behavior/BCI_Zaber_data/KayvonScope')
#%% reading in suite2p output - multi-session registration
import BCI_analysis
rigid = BCI_analysis.suite2p_tools.registration.rigid
import numpy as np
import os, json
from pathlib import Path

import matplotlib.pyplot as plt

def align_sessions(current_session,  previous_session,previous_sessions=None):
    
    print('aligning session {} to session {}'.format(current_session['session'],previous_session['session']))
    ops = current_session['ops']
    mean_img = current_session['mean_img']
    stat = current_session['stat']
    stat_noncell = current_session['stat_noncell']
    
    nframes = ops['nframes']
    batch_size = ops['batch_size']
    ops['nframes'] = 2
    ops['batch_size'] = 2
    
    maskMul, maskOffset = rigid.compute_masks(refImg=mean_img,
                                              maskSlope=1)
    cfRefImg = rigid.phasecorr_reference(refImg=mean_img,
                                         smooth_sigma=1)
    ymax, xmax, cmax = rigid.phasecorr(data=np.complex64(np.float32(np.asarray([previous_session['mean_img']]*2)) * maskMul + maskOffset),
                                       cfRefImg=cfRefImg,
                                       maxregshift=50,
                                       smooth_sigma_time=0)
    regimage = rigid.shift_frame(frame=previous_session['mean_img'], dy=ymax[0], dx=xmax[0])
    if np.max(np.abs(np.concatenate([ymax,xmax])))>100: # for some bizarre reason, sometimes the registration fails on the first time..
        maskMul, maskOffset = rigid.compute_masks(refImg=mean_img,
                                                  maskSlope=1)
        cfRefImg = rigid.phasecorr_reference(refImg=mean_img,
                                             smooth_sigma=1)
        ymax, xmax, cmax = rigid.phasecorr(data=np.complex64(np.float32(np.asarray([previous_session['mean_img']]*2)) * maskMul + maskOffset),
                                           cfRefImg=cfRefImg,
                                           maxregshift=50,
                                           smooth_sigma_time=0)
        regimage = rigid.shift_frame(frame=previous_session['mean_img'], dy=ymax[0], dx=xmax[0])
 
    
    ops['yblock'], ops['xblock'], ops['nblocks'], ops['block_size'], ops['NRsm'] = BCI_analysis.suite2p_tools.registration.register.nonrigid.make_blocks(Ly=ops['Ly'], Lx=ops['Lx'], block_size=[128,128])#ops['block_size'])
    ops['nframes'] = nframes 
    ops['batch_size']=batch_size 
    maskMulNR, maskOffsetNR, cfRefImgNR = BCI_analysis.suite2p_tools.registration.register.nonrigid.phasecorr_reference(refImg0=mean_img,
                                                                                                                        maskSlope=ops['spatial_taper'] if ops['1Preg'] else 3 * ops['smooth_sigma'], # slope of taper mask at the edges
                                                                                                                        smooth_sigma=ops['smooth_sigma'],
                                                                                                                        yblock=ops['yblock'],
                                                                                                                        xblock=ops['xblock'])
    ymax1, xmax1, cmax1 = BCI_analysis.suite2p_tools.registration.register.nonrigid.phasecorr(data=np.complex64(np.float32(np.array([regimage]*2))),
                                                                                              maskMul=maskMulNR.squeeze(),
                                                                                              maskOffset=maskOffsetNR.squeeze(),
                                                                                              cfRefImg=cfRefImgNR.squeeze(),
                                                                                              snr_thresh=ops['snr_thresh'],
                                                                                              NRsm=ops['NRsm'],
                                                                                              xblock=ops['xblock'],
                                                                                              yblock=ops['yblock'],
                                                                                              maxregshiftNR=ops['maxregshiftNR'])
    
    
    # match ROIs based on overlap

    stat_new = []
    for cell_i,cell in enumerate(previous_session['stat']): # this loop goes through the previous session and check every cellular ROI if present
    
        if type(cell) == type(None): # the ROI is not present in the previous session
            print('looking for previous sessions for ROI because it is NONE')
            if type(previous_sessions)==type(None):
                print('there is no previous session - probably an edge hitting ROI, skipping')
                stat_new.append(None)
                continue
            for old_session_dict in previous_sessions:
                if not type(old_session_dict['stat'][cell_i]) == type(None): # the last session where the cell was identified

                    ymax_, xmax_, cmax_ = rigid.phasecorr(data=np.complex64(np.float32(np.asarray([old_session_dict['mean_img']]*2)) * maskMul + maskOffset),
                                                       cfRefImg=cfRefImg,
                                                       maxregshift=50,
                                                       smooth_sigma_time=0)
                    regimage_ = rigid.shift_frame(frame=old_session_dict['mean_img'], dy=ymax_[0], dx=xmax_[0])
                    ymax1_, xmax1_, cmax1_ = BCI_analysis.suite2p_tools.registration.register.nonrigid.phasecorr(data=np.complex64(np.float32(np.array([regimage_]*2))),
                                                                                                              maskMul=maskMulNR.squeeze(),
                                                                                                              maskOffset=maskOffsetNR.squeeze(),
                                                                                                              cfRefImg=cfRefImgNR.squeeze(),
                                                                                                              snr_thresh=ops['snr_thresh'],
                                                                                                              NRsm=ops['NRsm'],
                                                                                                              xblock=ops['xblock'],
                                                                                                              yblock=ops['yblock'],
                                                                                                              maxregshiftNR=ops['maxregshiftNR'],
                                                                                                              )
                    mask_original = np.zeros(mean_img.shape)
                    mask_original[old_session_dict['stat'][cell_i]['ypix'],old_session_dict['stat'][cell_i]['xpix']] = old_session_dict['stat'][cell_i]['lam']/np.sum(old_session_dict['stat'][cell_i]['lam'])
                    mask_original = rigid.shift_frame(frame=mask_original, dy=ymax_[0], dx=xmax_[0])
                    mask_original = BCI_analysis.suite2p_tools.registration.register.nonrigid.transform_data(
                        data=np.float32(np.stack([mask_original,mask_original])),
                        nblocks=ops['nblocks'],
                        xblock=ops['xblock'],
                        yblock=ops['yblock'],
                        ymax1=ymax1,
                        xmax1=xmax1)
                    mask_original = mask_original[0,:,:].squeeze()
                    break
            if type(old_session_dict['stat'][cell_i]) == type(None):
                print('could not find ROI in older sessions..')
                stat_new.append(None)
                continue
                
            
        else:
            mask_original = np.zeros(mean_img.shape)
            mask_original[cell['ypix'],cell['xpix']] = cell['lam']/np.sum(cell['lam'])
            mask_original = rigid.shift_frame(frame=mask_original, dy=ymax[0], dx=xmax[0])
            mask_original = BCI_analysis.suite2p_tools.registration.register.nonrigid.transform_data(
                data=np.float32(np.stack([mask_original,mask_original])),
                nblocks=ops['nblocks'],
                xblock=ops['xblock'],
                yblock=ops['yblock'],
                ymax1=ymax1,
                xmax1=xmax1,
            )
            mask_original = mask_original[0,:,:].squeeze()
        #%
        residual = []
        #%
        for cell_new in stat:
            if type(cell_new) == type(None):
                residual.append(np.inf)
                continue

            #%
            mask_now = np.zeros(mean_img.shape)
            mask_now[cell_new['ypix'],cell_new['xpix']] = cell_new['lam']/np.sum(cell_new['lam'])
            
            diff = mask_original-mask_now
            residual.append(np.sum(np.abs(diff)))
            
        residual.append(np.inf)
        if np.min(residual)<ROI_residual_threshold:
            stat_new.append(stat.pop(np.argmin(residual)))
            if len(current_session['ROI_residuals'])==cell_i:
                current_session['ROI_residuals'].append(np.min(residual))
            #residual_matrix[cell_i,session_i-1] = np.min(residual)
        else:
            min_residual_cell = np.min(residual)
            residual = []
            for cell_new in stat_noncell:
                mask_now = np.zeros(mean_img.shape)
                mask_now[cell_new['ypix'],cell_new['xpix']] = cell_new['lam']/np.sum(cell_new['lam'])
                diff = mask_original-mask_now
                residual.append(np.sum(np.abs(diff)))
            if np.min(residual)<ROI_residual_threshold:
                stat_new.append(stat_noncell.pop(np.argmin(residual)))
                if len(current_session['ROI_residuals'])==cell_i:
                    current_session['ROI_residuals'].append(np.min(residual))
                #residual_matrix[cell_i,session_i-1] = np.min(residual)
            else: # carry cell over from previous session
                 #%            
                if len(current_session['ROI_residuals'])==cell_i:
                    current_session['ROI_residuals'].append(np.nan)
                mask_original = np.zeros(mean_img.shape)
                if type(cell) == type(None): #in case it's coming from the previous sessions
                    mask_original[old_session_dict['stat'][cell_i]['ypix'],old_session_dict['stat'][cell_i]['xpix']] = old_session_dict['stat'][cell_i]['lam']
                    mask_original = rigid.shift_frame(frame=mask_original, dy=ymax_[0], dx=xmax_[0])
                    mask_original = BCI_analysis.suite2p_tools.registration.register.nonrigid.transform_data(
                        data=np.float32(np.stack([mask_original,mask_original])),
                        nblocks=ops['nblocks'],
                        xblock=ops['xblock'],
                        yblock=ops['yblock'],
                        ymax1=ymax1_,
                        xmax1=xmax1_,
                    )
                else:
                    mask_original[cell['ypix'],cell['xpix']] = cell['lam']#/np.sum(cell['lam'])
                    mask_original = rigid.shift_frame(frame=mask_original, dy=ymax[0], dx=xmax[0])
                    mask_original = BCI_analysis.suite2p_tools.registration.register.nonrigid.transform_data(
                        data=np.float32(np.stack([mask_original,mask_original])),
                        nblocks=ops['nblocks'],
                        xblock=ops['xblock'],
                        yblock=ops['yblock'],
                        ymax1=ymax1,
                        xmax1=xmax1,
                    )
                mask_original = mask_original[0,:,:].squeeze()
                yidx,xidx = np.where(mask_original>0)
                lam = mask_original[yidx,xidx]
                stat_newly_fabricated = {'ypix':yidx,
                                         'xpix':xidx,
                                         'lam':lam}
                if any(xidx ==0) or any(yidx ==0):
                    stat_new.append(None) # ROI touches the edge, skipping
                else:
                    stat_new.append(stat_newly_fabricated) 
                
                #print('cell not found, best residuals: {}'.format([min_residual_cell,np.min(residual)]))
    for s_i,s in enumerate(stat):
        if not type(s) == type(None):
            stat_new.append(s) # the ROIs that were not present in the previous sessions(s) just gets at the end
            current_session['ROI_residuals'].append(np.nan)
    stat = stat_new

    current_session['stat']= stat
    current_session['stat_noncell']= stat_noncell
    return current_session

def load_suite2p_sessions(setup,subject,sessions,ROI_residual_threshold):
    session_data_list = []
    for session_i, session_now in enumerate(sessions): #create session data list
        session_dir_now = os.path.join(suite2p_base_dir,setup,subject,session_now)
        ops = np.load(os.path.join(session_dir_now,'ops.npy'),allow_pickle=True).tolist()
        #%
        z_plane_indices = np.argmax(ops['zcorr_list'],1)
        # generate a clean mean image from trials where the Z position is the same
        needed_trials = z_plane_indices == np.median(z_plane_indices) #
        meanimage_all = np.load(os.path.join(session_dir_now,'meanImg.npy'))
        mean_img = np.mean(meanimage_all[needed_trials,:,:],0)
        
        
        
        # load ROIs
        stat = np.load(os.path.join(session_dir_now,'stat.npy'),allow_pickle=True).tolist()
        iscell = np.load(os.path.join(session_dir_now,'iscell.npy'))
        stat_noncell = np.asarray(stat)[iscell[:,0]==0].tolist()
        stat = np.asarray(stat)[iscell[:,0]==1].tolist()
        
        with open(os.path.join(session_dir_now,'filelist.json')) as f:
            filelist_dict = json.load(f)
        with open(os.path.join(session_dir_now,'s2p_params.json')) as f:
            s2p_params = json.load(f)
        ops = np.load(os.path.join(session_dir_now,'ops.npy'),allow_pickle = True).tolist()
        
        
        
        session_data_dict = {'stat':stat,
                             'stat_noncell':stat_noncell,
                             'mean_img':mean_img,
                             'session_directory':session_dir_now,
                             'setup':setup,
                             'subject':subject,
                             'session':session_now,
                             'filelist_dict':filelist_dict,
                             's2p_params':s2p_params,
                             'ops':ops,
                             'ROI_residuals':[]} 
        
        
        session_data_list.append(session_data_dict)
    return session_data_list

def remove_missing_rois(session_data_list):
    #%
    print('removing Nones')
    none_list = []
    for session_data_dict in session_data_list:
        none_list.extend(np.where(np.asarray(session_data_dict['stat']) == None)[0])
    none_list = np.unique(none_list)
    #%
    for i in range(len(session_data_list)):
        for idx_to_pop in none_list[::-1]:
            try:
                session_data_list[i]['stat'].pop(idx_to_pop)
                session_data_list[i]['ROI_residuals'].pop(idx_to_pop)
            except:
                print('session {} done'.format(i))
                pass
    none_list = []
    for session_data_dict in session_data_list:
        none_list.extend(np.where(np.asarray(session_data_dict['stat']) == None)[0])
    none_list = np.unique(none_list)
    print('nones left: {}'.format(none_list))
    return session_data_list
    #%

def align_ROIS_in_sessions(session_data_list):
    #initialize ROI residuals in the first session
    session_data_list[0]['ROI_residuals'] = list(np.ones(len(session_data_list[0]['stat']))*np.nan) 
    # forward pass
    session_data_list_new = []
    for session_i, session_data_dict in enumerate(session_data_list):    
        if session_i>0:
            previous_session_data_dict = session_data_list[session_i-1]
            if session_i>1:
                previous_sessions = session_data_list[session_i-1::-1]
            else:
                previous_sessions = None
            print(len(session_data_dict['stat']))
            session_data_dict = align_sessions(session_data_dict,  previous_session_data_dict,previous_sessions)
            print(len(session_data_dict['stat']))
        session_data_list_new.append(session_data_dict)
        
    session_data_list = session_data_list_new.copy()
    session_data_list = remove_missing_rois(session_data_list)
    #aligning ROIs, backward pass
    session_data_list_new = []
    for i in np.arange(len(session_data_list)-1,-1,-1): 
        if i < len(session_data_list)-1:
            previous_session_data_dict = session_data_list[i+1]
            session_data_dict = session_data_list[i]
            if i<len(session_data_list)-2:
                previous_sessions = session_data_list[2+i:]
            else:
                previous_sessions  = None
            #print(len(session_data_dict['stat']))
            session_data_dict = align_sessions(session_data_dict,  previous_session_data_dict,previous_sessions)
            #print(len(session_data_dict['stat']))
        session_data_list_new.append(session_data_dict)
    session_data_list = session_data_list_new[::-1].copy()
    session_data_list = remove_missing_rois(session_data_list)
    return session_data_list

def compare_session_z_positions(session_data_list,scanimage_base_dir):
    #% compare z-stacks
    ScanImageTiffReader = BCI_analysis.io_scanimage.ScanImageTiffReader
    session_data_list_temp = []
    for session_i,session_data_dict in enumerate(session_data_list):#compare Z positions
        zstack_tiff = os.path.join(scanimage_base_dir,
                                    session_data_dict['setup'],
                                    session_data_dict['subject'],
                                    session_data_dict['session'],
                                    session_data_dict['s2p_params']['z_stack_name'])
        zstack_tiff = os.path.join(scanimage_base_dir,
                                    session_data_dict['session_directory'],
                                    session_data_dict['s2p_params']['z_stack_name'])
        try:
            zstack_metadata = BCI_analysis.io_scanimage.extract_scanimage_metadata(zstack_tiff)
            z_step = float(zstack_metadata['metadata']['hStackManager']['stackZStepSize'])
        except:
            z_step = 1
        try:
            try:
                reader=ScanImageTiffReader(zstack_tiff)
                stack=reader.data()
            except:
                reader=ScanImageTiffReader(zstack_tiff+'f')
                stack=reader.data()
        except:
            stack = tifffile.imread(zstack_tiff)
        #%
        ops_orig, zcorr = BCI_analysis.suite2p_tools.registration.zalign.compute_zpos_single_frame(stack, session_data_dict['mean_img'][np.newaxis,:,:], session_data_dict['ops'])
        z_max_pos = np.argmax(zcorr)
        if session_i>0:
            ops_orig, zcorr = BCI_analysis.suite2p_tools.registration.zalign.compute_zpos_single_frame(stack, session_data_list[session_i-1]['mean_img'][np.newaxis,:,:], session_data_dict['ops'])
            z_max_pos_prev_session =  np.argmax(zcorr)
            session_data_dict['z_diff_prev_session'] = (z_max_pos-z_max_pos_prev_session)*z_step
        else:
            session_data_dict['z_diff_prev_session'] = np.nan
            
        if session_i<len(session_data_list)-1:
            ops_orig, zcorr = BCI_analysis.suite2p_tools.registration.zalign.compute_zpos_single_frame(stack, session_data_list[session_i+1]['mean_img'][np.newaxis,:,:], session_data_dict['ops'])
            z_max_pos_next_session =  np.argmax(zcorr)
            session_data_dict['z_diff_next_session'] = (z_max_pos-z_max_pos_next_session)*z_step
        else:
            session_data_dict['z_diff_next_session'] = np.nan
        
        session_data_list_temp.append(session_data_dict)
    session_data_list = session_data_list_temp
    return session_data_list

ROI_residual_threshold = 1 #0-2
suite2p_base_dir = '/mnt/Data/Calcium_imaging/suite2p/'
scanimage_base_dir = '/mnt/Data/Calcium_imaging/raw/'
#%
save_base_dir = '/mnt/Data/Calcium_imaging/Analysis_python_pipeline/multi_session/'
fov_name = 'FOV_1'
#%
setup = 'DOM3-MMIMS'
subject = 'BCI_14'
sessions = ['2021-06-12',
            '2021-06-13',
            '2021-06-14',
            '2021-06-16']
# =============================================================================
# sessions = [#'2021-06-23',
#             '2021-06-24',
#             '2021-06-25',
#             '2021-06-27',
#             '2021-06-28']
# =============================================================================
setup = 'KayvonScope'
subject = 'BCI_26'
sessions = ['040622',
            '040722',
            '040822']
session_data_list = load_suite2p_sessions(setup,subject,sessions,ROI_residual_threshold)
#%
session_data_list = compare_session_z_positions(session_data_list,scanimage_base_dir)
session_data_list = align_ROIS_in_sessions(session_data_list)

session_data_list_original = session_data_list.copy()
#% redefine ROIs and neuropil, save data
session_data_list = session_data_list_original.copy()
roi_stats = BCI_analysis.suite2p_tools.detection.stats.roi_stats
create_masks_and_extract = BCI_analysis.suite2p_tools.extraction.extract.create_masks_and_extract
needed_cells = np.ones_like(session_data_list[0]['stat'],dtype=bool)
for session_data_dict in session_data_list:
    needed_cells = needed_cells & (np.asarray(session_data_dict['stat']) != None)
    
for i in range(len(session_data_list)):
    for cell_i in np.where(needed_cells==False)[0]:
        session_data_list[i]['stat_noncell'].append(session_data_list[i]['stat'].pop(cell_i))
    session_data_list[i]['ROI_residuals'] = np.asarray(session_data_list[i]['ROI_residuals'])[np.where(needed_cells)[0]]
    
    #%
session_data_list_save = []
for session_data_dict in session_data_list:
    # create a stat where both cells and non-cells are present, so neuropil will be neuropil (???) - or include only cells??
    dy, dx = int(session_data_dict['ops']['aspect'] * 10), 10
    stat = session_data_dict['stat']
    cells_needed = []
    for s in stat:
        if type(s) == type(None):
            cells_needed.append(False)
        else:
            cells_needed.append(True)
    stat = np.asarray(stat)[np.asarray(cells_needed)]
    stat_new = []
    for s in stat:
        if len(s.keys())>5:
            non_original_roi = False
        else:
            non_original_roi = True
        stat_new.append({'ypix':s['ypix'],
                         'xpix':s['xpix'],
                         'lam' :s['lam'],
                         'carried_over_roi':non_original_roi})
    for s in session_data_dict['stat_noncell']:
        if not s is None:
            stat_new.append({'ypix':s['ypix'],
                             'xpix':s['xpix'],
                             'lam' :s['lam']})
    stat = roi_stats(np.asarray(stat_new), 
                     dy,
                     dx,
                     session_data_dict['ops']['Ly'],
                     session_data_dict['ops']['Lx'],
                     max_overlap=session_data_dict['ops']['max_overlap'],
                     do_crop=session_data_dict['ops']['soma_crop']) # add the missing stats fields
    session_data_dict['ops']['reg_file'] = os.path.join(session_data_dict['session_directory'],'data.bin')
    
    ops, stat, F, Fneu, F_chan2, Fneu_chan2 = create_masks_and_extract(session_data_dict['ops'], stat) # extract fluorescence and stuff
    #%
    session_data_dict['stat'] = stat[:sum(cells_needed)]
    session_data_dict['F']  = F[:sum(cells_needed),:]
    session_data_dict['Fneu']  = Fneu[:sum(cells_needed),:]
    session_data_list_save.append(session_data_dict)
    #%
savedir = os.path.join(save_base_dir,subject)
Path(savedir).mkdir(parents = True, exist_ok = True)
np.save(os.path.join(savedir,'{}_{}.npy'.format(subject,fov_name)),session_data_list_save,allow_pickle = True)
    
#%% show ROIs
   
def rollingfun(y, window = 10, func = 'mean'):
    """
    rollingfun
        rolling average, min, max or std
    
    @input:
        y = array, window, function (mean,min,max,std)
    """
    if len(y)<=window:
        if func =='mean':
            out = np.ones(len(y))*np.nanmean(y)
        elif func == 'min':
            out = np.ones(len(y))*np.nanmin(y)
        elif func == 'max':
            out = np.ones(len(y))*np.nanmaxn(y)
        elif func == 'std':
            out = np.ones(len(y))*np.nanstd(y)
        elif func == 'median':
            out = np.ones(len(y))*np.nanmedian(y)
        else:
            print('undefinied funcion in rollinfun')
    else:
        y = np.concatenate([y[window::-1],y,y[:-1*window:-1]])
        ys = list()
        for idx in range(window):    
            ys.append(np.roll(y,idx-round(window/2)))
        if func =='mean':
            out = np.nanmean(ys,0)[window:-window]
        elif func == 'min':
            out = np.nanmin(ys,0)[window:-window]
        elif func == 'max':
            out = np.nanmax(ys,0)[window:-window]
        elif func == 'std':
            out = np.nanstd(ys,0)[window:-window]
        elif func == 'median':
            out = np.nanmedian(ys,0)[window:-window]
        else:
            print('undefinied funcion in rollinfun')
    return out

#%% multiple ROIs, multiple timepoints
import random
import matplotlib.patches as patches
import scipy
session_data_dict = session_data_list_save[0]
stat = session_data_dict['stat']
rois_to_plot = 4
timepoints = 3
original_roi_idxs = []
carried_over_roi_idxs = []
carried_over_roi = False #set to false to plot original rois
rois = np.zeros_like(session_data_dict['mean_img'])
for i,s in enumerate(stat):
    neurpil_coord = np.unravel_index(s['neuropil_mask'],rois.shape)
    if 'carried_over_roi' in s.keys():
        if s['carried_over_roi']:
            rois[s['ypix'][s['overlap']==False],s['xpix'][s['overlap']==False]] -= 1#cell['lam']/np.sum(cell['lam'])
            carried_over_roi_idxs.append(i)
        else:
            rois[s['ypix'][s['overlap']==False],s['xpix'][s['overlap']==False]] += 1#cell['lam']/np.sum(cell['lam'])
            original_roi_idxs.append(i)
    else:
        rois[s['ypix'][s['overlap']==False],s['xpix'][s['overlap']==False]] += 5#cell['lam']/np.sum(cell['lam'])
    rois[neurpil_coord[0],neurpil_coord[1]] = .5#cell['lam']/np.sum(cell['lam'])

if carried_over_roi:
    roi_indices = carried_over_roi_idxs
else:
    roi_indices = original_roi_idxs
random.shuffle(roi_indices)


fig = plt.figure()
ax_meanimage = fig.add_subplot(rois_to_plot+1,timepoints+3,1)
ax_meanimage.set_title('mean image')
ax_meanimage.imshow(session_data_dict['mean_img'])
ax_maximage = fig.add_subplot(rois_to_plot+1,timepoints+3,2,sharex = ax_meanimage,sharey = ax_meanimage)
ax_maximage.set_title('max projection')
maxproj = np.max(mov,0)
ax_maximage.imshow(maxproj)
ax_roi = fig.add_subplot(rois_to_plot+1,timepoints+3,3,sharex = ax_meanimage,sharey = ax_meanimage)
ax_roi.set_title('ROIs')
ax_roi.imshow(rois)
#roi_indices = [0,1]
for i,roi_idx in enumerate(roi_indices[:rois_to_plot]):
    ax_meanim_roi = fig.add_subplot(rois_to_plot+1,timepoints+3,(timepoints+3)*(i+1)+1)
    ax_meanim_roi.set_title('roi {}'.format(roi_idx))
    im = ax_meanim_roi.imshow(maxproj)
    x_lims = [stat[roi_idx]['med'][1]-stat[roi_idx]['radius']*2,stat[roi_idx]['med'][1]+stat[roi_idx]['radius']*2]
    y_lims = [stat[roi_idx]['med'][0]-stat[roi_idx]['radius']*2,stat[roi_idx]['med'][0]+stat[roi_idx]['radius']*2]
    ax_meanim_roi.set_xlim(x_lims)
    ax_meanim_roi.set_ylim(y_lims)
    ax_roi_now = fig.add_subplot(rois_to_plot+1,timepoints+3,(timepoints+3)*(i+1)+2)
    ax_roi_now.imshow(rois)
    ax_roi_now.set_xlim(x_lims)
    ax_roi_now.set_ylim(y_lims)
    im_crop = maxproj[np.max([0,int(y_lims[0])]):int(y_lims[1]),np.max([0,int(x_lims[0])]):int(x_lims[1])]
    im.set_clim([np.min(im_crop),np.max(im_crop)])
    rect = patches.Rectangle(stat[roi_idx]['med'][::-1]-stat[roi_idx]['radius']*2, stat[roi_idx]['radius']*4, stat[roi_idx]['radius']*4, linewidth=.5, edgecolor='r', facecolor='none')
    ax_meanimage.add_patch(rect)
    rect = patches.Rectangle(stat[roi_idx]['med'][::-1]-stat[roi_idx]['radius']*2, stat[roi_idx]['radius']*4, stat[roi_idx]['radius']*4, linewidth=.5, edgecolor='r', facecolor='none')
    ax_maximage.add_patch(rect)
    rect = patches.Rectangle(stat[roi_idx]['med'][::-1]-stat[roi_idx]['radius']*2, stat[roi_idx]['radius']*4, stat[roi_idx]['radius']*4, linewidth=.5, edgecolor='r', facecolor='none')
    ax_roi.add_patch(rect)
    
    cell = stat[roi_idx]
    mov_array = np.reshape(mov, (mov.shape[0],-1)).astype(np.float32)
    cell_mask = np.ravel_multi_index((cell['ypix'], cell['xpix']), session_data_dict['mean_img'].shape)
    f = np.dot(mov_array[:, cell_mask], cell['lam']/np.sum(cell['lam']))
    ax_f = fig.add_subplot(rois_to_plot+1,timepoints+3,(timepoints+3)*(i+1)+3)
    ax_f.plot(f)
    
    peaks,props = scipy.signal.find_peaks(f,distance = 10,prominence=np.median(f)*.1)
    order =np.argsort(props['prominences'])
    peaks = peaks[order]
    props['prominences'] = props['prominences'][order]
    ax_f.plot(peaks,f[peaks],'rx')
    for peak_i in range(timepoints):
        peak = peaks[int((len(peaks)-1)*peak_i/(timepoints-1))]
        ax_f.plot(peak,f[peak],'bo')
        ax_f_im = fig.add_subplot(rois_to_plot+1,timepoints+3,(timepoints+3)*(i+1)+4+peak_i)
        im = ax_f_im.imshow(mov[peak,:,:])
        im.set_clim([0,np.max(f)])
        ax_f_im.set_xlim(x_lims)
        ax_f_im.set_ylim(y_lims)
   #%% a
session_data_dict = session_data_list_save[0]
stat = session_data_dict['stat']
fig = plt.figure()
rois_to_plot = 2
ax_meanimage = fig.add_subplot(rois_to_plot+1,4,1)
ax_rois = fig.add_subplot(rois_to_plot+1,4,2,sharex = ax_meanimage,sharey = ax_meanimage)
ax_meanimage.imshow(session_data_dict['mean_img'])
ax_maximage = fig.add_subplot(rois_to_plot+1,4,3,sharex = ax_meanimage,sharey = ax_meanimage)
ax_maximage.imshow(np.max(mov,0))
rois = np.zeros_like(session_data_dict['mean_img'])
original_rois_plotted = 0
carried_over_rois_plotted = 0


ops = session_data_dict['ops']

bin_size = int(max(1, ops['nframes'] // ops['nbinned'], np.round(ops['tau'] * ops['fs'])))
for i,cell in enumerate(session_data_dict['stat']):
    neurpil_coord = np.unravel_index(cell['neuropil_mask'],rois.shape)
    try:
        if cell['carried_over_roi']:
            rois[cell['ypix'][cell['overlap']==False],cell['xpix'][cell['overlap']==False]] -= 1#cell['lam']/np.sum(cell['lam'])
            rois[neurpil_coord[0],neurpil_coord[1]] = .5#cell['lam']/np.sum(cell['lam'])
            if carried_over_rois_plotted<rois_to_plot:
                ax_now = fig.add_subplot(rois_to_plot+1,4,7+carried_over_rois_plotted*4)
                
                ax_now .plot(session_data_dict['Fneu'][i,:],'r-',alpha = .5)
                ax_now .plot(session_data_dict['F'][i,:],'g-',alpha = .5)
                t = np.arange(len(session_data_dict['F'][i,:]))
                f_rolling_mean = rollingfun(session_data_dict['F'][i,:],bin_size,'mean')
                t = rollingfun(t,bin_size,'mean')
                ax_now .plot(t,f_rolling_mean,'g-',alpha = 1)
                idx = np.argmax(f_rolling_mean)
                ax_now .plot(t[idx],f_rolling_mean[idx],'ro',alpha = 1)
                ax_img = fig.add_subplot(rois_to_plot+1,4,8+carried_over_rois_plotted*4,sharex = ax_meanimage,sharey = ax_meanimage)
                mov_array = np.reshape(mov, (mov.shape[0],-1)).astype(np.float32)
                cell_mask = np.ravel_multi_index((cell['ypix'], cell['xpix']), session_data_dict['mean_img'].shape)
                f = np.dot(mov_array[:, cell_mask], cell['lam']/np.sum(cell['lam']))
                idx_mov = np.argmax(f)
                
                im = ax_img.imshow(mov[idx_mov,:,:])
                im.set_clim([0,np.max(f)])
                rois[cell['ypix'],cell['xpix']] -= 1#cell['lam']/np.sum(cell['lam'])
                carried_over_rois_plotted +=1
            
        else:
            rois[cell['ypix'][cell['overlap']==False],cell['xpix'][cell['overlap']==False]] += 1#cell['lam']/np.sum(cell['lam'])
            rois[neurpil_coord[0],neurpil_coord[1]] = .5#cell['lam']/np.sum(cell['lam'])
            if original_rois_plotted<rois_to_plot:
                ax_now = fig.add_subplot(rois_to_plot+1,4,6+original_rois_plotted*4)
                
                ax_now .plot(session_data_dict['Fneu'][i,:],'r-',alpha = .5)
                ax_now .plot(session_data_dict['F'][i,:],'g-',alpha = .5)
                
                t = np.arange(len(session_data_dict['F'][i,:]))
                f_rolling_mean = rollingfun(session_data_dict['F'][i,:],bin_size,'mean')
                t = rollingfun(t,bin_size,'mean')
                ax_now .plot(t,f_rolling_mean,'g-',alpha = 1)
                idx = np.argmax(f_rolling_mean)
                ax_now .plot(t[idx],f_rolling_mean[idx],'ro',alpha = 1)
                ax_img = fig.add_subplot(rois_to_plot+1,4,5+original_rois_plotted*4,sharex = ax_meanimage,sharey = ax_meanimage)
                mov_array = np.reshape(mov, (mov.shape[0],-1)).astype(np.float32)
                cell_mask = np.ravel_multi_index((cell['ypix'], cell['xpix']), session_data_dict['mean_img'].shape)
                f = np.dot(mov_array[:, cell_mask], cell['lam']/np.sum(cell['lam']))
                idx_mov = np.argmax(f)
                im = ax_img.imshow(mov[idx_mov,:,:])
                im.set_clim([0,np.max(f)])
                original_rois_plotted +=1
                rois[cell['ypix'],cell['xpix']] += 1#cell['lam']/np.sum(cell['lam'])
                
               
    except:
        pass
    
    
    
    
ax_rois.imshow(rois)    
#%%

#%%
from BCI_analysis.suite2p_tools.io.binary import BinaryFile
ops = session_data_dict['ops'].copy()


bin_size = 50#int(max(1, ops['nframes'] // ops['nbinned'], np.round(ops['tau'] * ops['fs'])))
print('Binning movie in chunks of length %2.2d' % bin_size)
with BinaryFile(read_filename=ops['reg_file'], Ly=ops['Ly'], Lx=ops['Lx']) as f:
    mov = f.bin_movie(
        bin_size=bin_size,
        bad_frames=None,
        y_range=None,
        x_range=None,
    )


#%% visualize matches
plt.rcParams['font.size'] = 6
session_index= 0
sorted_residuals = np.sort(session_data_list_save[session_index]['ROI_residuals'])[::-1]
sorted_residuals = sorted_residuals[np.isnan(sorted_residuals)==False] 
cells_to_show = 3

fig_roimatch = plt.figure()
for session_i,session_data in enumerate(session_data_list):
    if session_i ==0:
        ax_meanimage = fig_roimatch.add_subplot(cells_to_show+1,len(session_data_list),session_i+1)
        ax_master = ax_meanimage
    else:
        ax_meanimage = fig_roimatch.add_subplot(cells_to_show+1,len(session_data_list),session_i+1,sharex = ax_master,sharey = ax_master)
    
    
    ax_meanimage.imshow(session_data['mean_img'])
    try:
        ax_meanimage.set_title('{} \n z drift: {} and {} um'.format(session_data['session'],
                                                                                 session_data['z_diff_prev_session'],
                                                                                 session_data['z_diff_next_session']))
    except:
        pass
    plt.axis('off')
    for cell_i in np.arange(cells_to_show):
        
        ax_roi = fig_roimatch.add_subplot(cells_to_show+1,len(session_data_list),session_i+len(session_data_list)*(cell_i+1)+1,sharex = ax_master,sharey = ax_master)
        plt.axis('off')
        #cell_idx = np.unravel_index(np.argmax(residual_matrix==sorted_residuals[cell_i+0]),residual_matrix.shape)[0]
        cell_idx = np.argmax(session_data_list[session_index]['ROI_residuals']==sorted_residuals[cell_i+0])
        try:
            cell = session_data['stat'][cell_idx]
            mask_now = np.zeros(session_data['mean_img'].shape)
            mask_now[cell['ypix'],cell['xpix']] = cell['lam']/np.sum(cell['lam'])
            ax_roi.imshow(mask_now)
            
            ax_roi.set_title('roi {}, residual = {}'.format(cell_idx,np.round(session_data_list[session_i]['ROI_residuals'][cell_idx],2)))
        
        except:
            pass
#%% show last session with ALL the ROIs
session_data_dict = session_data_list[-1]
fig_allrois = plt.figure()
ax_meanimg = fig_allrois.add_subplot(1,2,1)
ax_meanimg.imshow(session_data_dict['mean_img'])
rois = np.zeros_like(session_data_dict['mean_img'])
for cell in session_data_dict['stat']:
    try:
        if 'npix' in cell.keys():
            rois[cell['ypix'],cell['xpix']] += 1#cell['lam']/np.sum(cell['lam'])
        else:
            rois[cell['ypix'],cell['xpix']] -= 1#cell['lam']/np.sum(cell['lam'])
    except:
        pass

for cell in session_data_dict['stat_noncell']:
    try:
        rois[cell['ypix'],cell['xpix']] += .5#cell['lam']/np.sum(cell['lam'])
    except:
        pass
ax_rois = fig_allrois.add_subplot(1,2,2,sharex = ax_meanimg,sharey = ax_meanimg)
ax_rois.imshow(rois)
#%% Lucas scripts
import BCI_analysis
multi_session_dir = '/mnt/Data/Calcium_imaging/Analysis_Kayvon_pipeline/multi-session/'
data_dict = BCI_analysis.plot_imaging.grab_data(multi_session_dir,'bci18')

#%% plot behavior - TODO UNFINISHED !!!


def plot_behavior_session(behavior_file = '/mnt/Data/Behavior/BCI_exported/KayvonScope/BCI_22/111321-bpod_zaber.npy',moving_window = 10):

    #%%
    import numpy as np
    behavior_file = '/mnt/Data/Behavior/BCI_exported/KayvonScope/BCI_22/111321-bpod_zaber.npy'
    behavior_file = '/mnt/Data/Behavior/BCI_exported/KayvonScope/BCI_26/033122-bpod_zaber.npy'
    moving_window = 10
    behav_dict = np.load(behavior_file,allow_pickle = True).tolist()
    #%%
    # decide on which side the mouse got the rewards
    if len(np.concatenate(behav_dict['reward_L']))>len(np.concatenate(behav_dict['reward_R'])): 
        side = 'L'
    else:
        side = 'R'
    # iterate over trials and gather all relevant information
    valve_time = []
    valve_time_trial = []
    go_cue_time = []
    go_cue_time_trial = []
    threshold_crossing_time = []
    threshold_crossing_time_trial = []
    reward_time = []
    reward_time_trial = []
    
    for trial_i in range(len(behav_dict['trial_num'])):
        valve_time_trial.append(trial_i)
        valve_time.append(behav_dict['var_ValveOpenTime_{}'.format(side)][trial_i])
        
        if len(behav_dict['go_cue_times'][trial_i])>0:
            go_cue_time.append(behav_dict['go_cue_times'][trial_i][0])
            go_cue_time_trial.append(trial_i)
        if len(behav_dict['threshold_crossing_times'][trial_i])>0:
            threshold_crossing_time.append(behav_dict['threshold_crossing_times'][trial_i][0])
            threshold_crossing_time_trial.append(trial_i)
        if len(behav_dict['reward_{}'.format(side)][trial_i])>0:
            reward_time.append(behav_dict['reward_{}'.format(side)][trial_i][0])
            reward_time_trial.append(trial_i)
    
    valve_time = np.asarray(valve_time)
    valve_time_trial = np.asarray(valve_time_trial)
    go_cue_time = np.asarray(go_cue_time)
    go_cue_time_trial = np.asarray(go_cue_time_trial)
    threshold_crossing_time = np.asarray(threshold_crossing_time)
    threshold_crossing_time_trial = np.asarray(threshold_crossing_time_trial)
    reward_time = np.asarray(reward_time)
    reward_time_trial = np.asarray(reward_time_trial)
    
    
    time_to_threshold_cross = []
    time_to_threshold_cross_trial = threshold_crossing_time_trial
    for trial,threshold_t in zip(threshold_crossing_time_trial,threshold_crossing_time):
        idx = np.argmax(go_cue_time_trial==trial)
        time_to_threshold_cross.append(threshold_t-go_cue_time[idx])
    time_to_collect_reward = []
    time_to_collect_reward_trial = reward_time_trial
    for trial,reward_t in zip(reward_time_trial,reward_time):
        idx = np.argmax(threshold_crossing_time_trial==trial)
        time_to_collect_reward.append(reward_t-threshold_crossing_time[idx])
    
    
    
    #%% TODO this is the original datajoint function, the plot function to rewrite is above
        
    trial_num,outcome,lickport_auto_step_freq,task_protocol,lickport_step_size,trial_start_time,trial_end_time,lickport_maximum_speed,lickport_limit_far = (experiment.SessionTrial()*experiment.LickPortSetting()*experiment.BehaviorTrial()&session_key).fetch('trial','outcome','lickport_auto_step_freq','task_protocol','lickport_step_size','trial_start_time','trial_end_time','lickport_maximum_speed','lickport_limit_far')
    lickport_step_size = np.asarray(lickport_step_size,float)
    trial_lengths = np.asarray(trial_end_time-trial_start_time,float)
    trial_num_bci,threshold_low,threshold_high,bci_conditioned_neuron_sign,bci_conditioned_neuron_idx = (experiment.BCISettings().ConditionedNeuron()&session_key).fetch('trial','bci_threshold_low','bci_threshold_high','bci_conditioned_neuron_sign','bci_conditioned_neuron_idx')
    hits = outcome == 'hit'
    hit_rate = np.zeros(len(outcome))*np.nan
    
    trial_average_speed = np.abs(np.asarray(lickport_limit_far,float))/time_to_hit
    trial_max_speed = np.zeros(len(outcome))*np.nan
    for trial_i, step_size in zip(trial_num,lickport_step_size):
        step_ts = (experiment.TrialEvent()&session_key&'trial_event_type = "lickport step"'&'trial = {}'.format(trial_i)).fetch('trial_event_time')
        dts = np.diff(np.asarray(step_ts,float))
        if len(dts)>0:
            trial_max_speed[trial_i] =step_size/np.min(dts) 
        
        
    
    rewardsperminute = np.zeros(len(outcome))*np.nan
    trial_length_moving = np.zeros(len(outcome))*np.nan
    timetohit_moving = np.zeros(len(outcome))*np.nan
    timetocollectreward_moving = np.zeros(len(outcome))*np.nan
    max_speed_moving = np.zeros(len(outcome))*np.nan
    trial_average_speed_moving = np.zeros(len(outcome))*np.nan
    task_change_idx = np.where(np.abs(np.diff(task_protocol))>0)[0]
    task_change_idx = np.concatenate([[0],task_change_idx,[len(task_protocol)]])
    for idx in np.arange(len(task_change_idx)-1):
        hit_rate[task_change_idx[idx]:task_change_idx[idx+1]] = rollingfun(hits[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
        rewardsperminute[task_change_idx[idx]:task_change_idx[idx+1]] = rollingfun((hits/trial_lengths*60)[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
        trial_length_moving[task_change_idx[idx]:task_change_idx[idx+1]] = rollingfun(trial_lengths[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
        timetohit_moving[task_change_idx[idx]:task_change_idx[idx+1]] = rollingfun(time_to_hit[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
        timetocollectreward_moving[task_change_idx[idx]:task_change_idx[idx+1]] = rollingfun(time_to_collect_reward[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
        trial_average_speed_moving[task_change_idx[idx]:task_change_idx[idx+1]] = rollingfun(trial_average_speed[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
        max_speed_moving[task_change_idx[idx]:task_change_idx[idx+1]] = rollingfun(trial_max_speed[task_change_idx[idx]:task_change_idx[idx+1]],moving_window)
    
    
    #%
    fig_behavior = plt.figure(figsize = [8,15])
    ax_movement_speed = fig_behavior.add_subplot(412)
    ax_distance = ax_movement_speed.twinx()
    ax_distance.set_ylabel('Dis