import numpy as np
import matplotlib.pyplot as plt
def find_conditioned_neuron_idx(session_bpod_file,session_ops_file,fov_stats_file, plot = False):
    """
    This script matches the scanimage conditioned neuron to the suite2p ROI
    Parameters
    ----------
    session_bpod_file : str
        path to extracted behavior file
    session_ops_file : str
        path to ops file that contains session-related suite2p info
    fov_stats_file : str
        path to ops file that contains spatial ROI information..
    plot : bool, optional
        If set to true, mean image is shown along with suite2p ROIs, the found conditioned ROI is yellow.
        Red dots are scanimage roi integration centers, one should overlap with the conditioned ROI.
    Returns
    -------
    cond_s2p_idx : list of int
        one number for each frame, which was the conditioned neuron -  should be the same for the whole session
    
    Example
    -------
    session_bpod_file = '/mnt/Data/Behavior/BCI_exported/KayvonScope/BCI_29/042922-bpod_zaber.npy'
    session_ops_file = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p/Bergamo-2P-Photostim/BCI_29/FOV_02/042922/ops.npy'
    fov_stats_file = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p/Bergamo-2P-Photostim/BCI_29/FOV_02/stat.npy'
    cond_s2p_idx = find_conditioned_neuron_idx(session_bpod_file,session_ops_file,fov_stats_file, plot = True)
    """
    #%
    behavior_dict = np.load(session_bpod_file,allow_pickle = True).tolist()
    ops =  np.load(session_ops_file,allow_pickle = True).tolist()
    stat =  np.load(fov_stats_file,allow_pickle = True).tolist()
    conditioned_neuron_name_list = []
    roi_indices = []
    for i,(filename,tiff_header) in enumerate(zip(behavior_dict['scanimage_file_names'],behavior_dict['scanimage_tiff_headers'])):    #find names of conditioned neurons
        
        if len(behavior_dict['scanimage_roi_outputChannelsRoiNames'][i]) >0:
            metadata  = tiff_header.tolist()[0]
            for roi_fcn_i, roi_fnc_name in enumerate(behavior_dict['scanimage_roi_outputChannelsNames'][i]):
                bpod_analog_idx = np.nan
                if 'analog' in roi_fnc_name:
                    bpod_analog_idx = roi_fcn_i
                    break
            try:
                conditioned_neuron_name = (behavior_dict['scanimage_roi_outputChannelsRoiNames'][i][bpod_analog_idx])[0]
            except:
                conditioned_neuron_name = (behavior_dict['scanimage_roi_outputChannelsRoiNames'][i][bpod_analog_idx])
            if len(conditioned_neuron_name) == 0:
                conditioned_neuron_name = ''
            rois = metadata['metadata']['json']['RoiGroups']['integrationRoiGroup']['rois']   
            if type(rois) is not list:
                rois = [rois]
            roinames_list = list() 
            for roi in rois:
                print(roi)
                roinames_list.append(roi['name'])
            try:
                roi_idx = np.where(np.asarray(roinames_list)==conditioned_neuron_name)[0][0]+1
            except:
                print('ROI names in scanimage header does not match up: {}'.format(conditioned_neuron_name))
                conditioned_neuron_name = ' '.join(conditioned_neuron_name.split(","))
                roi_idx = np.where(np.asarray(roinames_list)==conditioned_neuron_name)[0][0]+1
        else:
            conditioned_neuron_name  =''
            roi_idx = None
        conditioned_neuron_name_list.append(conditioned_neuron_name)
        roi_indices.append(roi_idx)

    x_offset = np.median(ops['xoff_list'][:100])
    y_offset  =np.median(ops['yoff_list'][:100])
    fovdeg = list()
    for s in metadata['metadata']['hRoiManager']['imagingFovDeg'].strip('[]').split(' '): fovdeg.extend(s.split(';'))
    fovdeg = np.asarray(fovdeg,float)
    fovdeg = [np.min(fovdeg),np.max(fovdeg)]
    rois = metadata['metadata']['json']['RoiGroups']['integrationRoiGroup']['rois']    
    if type(rois) == dict:
        rois = [rois]    
    centerXY_list = list()
    roinames_list = list()
    Lx = float(metadata['metadata']['hRoiManager']['pixelsPerLine'])
    Ly = float(metadata['metadata']['hRoiManager']['linesPerFrame'])
    #
    for roi in rois:
        try:
            centerXY_list.append((roi['scanfields']['centerXY']-fovdeg[0])/np.diff(fovdeg))
        except:
            print('multiple scanfields for {}'.format(roi['name']))
            centerXY_list.append((roi['scanfields'][0]['centerXY']-fovdeg[0])/np.diff(fovdeg))
        
        roinames_list.append(roi['name'])
    cond_s2p_idx = list()
    for roi_idx_now in roi_indices:    
        if roi_idx_now is None:
            cond_s2p_idx.append(None)
            continue
        med_list = list()
        dist_list = list()
        for cell_stat in stat:
            
            dist = np.sqrt((centerXY_list[roi_idx_now-1][0]*Lx-x_offset-cell_stat['med'][1])**2+(centerXY_list[roi_idx_now-1][1]*Lx-y_offset-cell_stat['med'][0])**2)
            dist_list.append(dist)
            med_list.append(cell_stat['med'])
            #break
        cond_s2p_idx.append(np.argmin(dist_list))
        
        
    #%  show ROIS
    if plot:
        fig_meanimage = plt.figure()
        ax_meanimage =fig_meanimage.add_subplot(2,1,1)
        ax_rois = fig_meanimage.add_subplot(2,1,2,sharex = ax_meanimage,sharey = ax_meanimage)
        ax_meanimage.imshow(ops['meanImg'])#,cmap = 'gray')
        mask = np.zeros_like(ops['meanImg'])
        for i,roi_stat in enumerate(stat):
            if i == np.unique(np.asarray(cond_s2p_idx)[np.asarray(cond_s2p_idx) != None])[0]:
                mask[roi_stat['ypix'],roi_stat['xpix']] = 2
            else:
                mask[roi_stat['ypix'],roi_stat['xpix']] = 1
            #break
        maskimg = ax_rois.imshow(mask)#,alpha = .5)    #cmap = 'hot',
        ax_rois.plot(np.asarray(centerXY_list)[:,0]*Lx-x_offset,np.asarray(centerXY_list)[:,1]*Ly-y_offset,'ro')
    #%
    return cond_s2p_idx
