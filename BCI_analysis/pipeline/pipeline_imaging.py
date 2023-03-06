import numpy as np
import matplotlib.pyplot as plt
import math
import os
from suite2p.registration.nonrigid import upsample_block_shifts
from ..plot import plot_utils 
import json

def find_conditioned_neuron_idx(session_bpod_file,
                                session_ops_file,
                                fov_stats_file, 
                                plot = False,
                                return_distances = False,
                                calculate_cn_lickport_correlation = False):
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
    return_distances : bool, optional
        ***************
    calculate_cn_lickport_correlation : bool, optional
        if set to true, calculates the correlation between lickport movement and the conditioned neuron, including the cells in 100 pixels vicinity of the CN
    Returns
    -------
    cond_s2p_idx : list of int
        one number for each trial, which was the conditioned neuron -  should be the same for the whole session
    closed_loop_trial : list of bool
        one True/False for each trial, if this was a closed loop trial
    scanimage_filenames : list of array of str
        scanimage filenames for each trial
    
    Example
    -------
    session_bpod_file = '/mnt/Data/Behavior/BCI_exported/KayvonScope/BCI_35/072122-bpod_zaber.npy'
    session_ops_file = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p/Bergamo-2P-Photostim/BCI_35/FOV_05/072122/ops.npy'
    fov_stats_file = '/home/rozmar/Network/GoogleServices/BCI_data/Data/Calcium_imaging/suite2p/Bergamo-2P-Photostim/BCI_35/FOV_05/stat.npy'
    cond_s2p_idx,closed_loop_trial,scanimage_filenames = find_conditioned_neuron_idx(session_bpod_file,session_ops_file,fov_stats_file, plot = True)
    """



    behavior_dict = np.load(session_bpod_file,allow_pickle = True).tolist()
    ops =  np.load(session_ops_file,allow_pickle = True).tolist()
    meanimg_dict = np.load(os.path.join(os.path.dirname(session_ops_file),'mean_image.npy'),allow_pickle = True).tolist()
    stat =  np.load(fov_stats_file,allow_pickle = True).tolist()

    conditioned_neuron_name_list = []
    roi_indices = []
    closed_loop_trial = []
    scanimage_filenames = []
    distances_all = []
    for i,(filename,tiff_header) in enumerate(zip(behavior_dict['scanimage_file_names'],behavior_dict['scanimage_tiff_headers'])):    #find names of conditioned neurons
        scanimage_filenames.append(filename)
        #print(behavior_dict['scanimage_roi_outputChannelsRoiNames'])
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
                try:
                    roinames_list.append(roi['name'])
                except:
                    roinames_list.append(None)
            try:
                roi_idx = np.where(np.asarray(roinames_list)==conditioned_neuron_name)[0][0]+1
            except:
                try:
                    print('ROI names in scanimage header does not match up: {}'.format(conditioned_neuron_name))
                    conditioned_neuron_name = ' '.join(conditioned_neuron_name.split(","))
                    roi_idx = np.where(np.asarray(roinames_list)==conditioned_neuron_name)[0][0]+1
                except:
                    print('no usable ROI idx, skipping')
                    roi_idx = None

        else:
            conditioned_neuron_name  =''
            roi_idx = None
        if roi_idx != None and metadata['metadata']['hIntegrationRoiManager']['enable'] == metadata['metadata']['hIntegrationRoiManager']['outputChannelsEnabled'] == 'true':
            closed_loop_trial.append(True)
        else:
            closed_loop_trial.append(False)

        conditioned_neuron_name_list.append(conditioned_neuron_name)
        roi_indices.append(roi_idx)
    x_offset = np.median(ops['xoff_list'][:5000])
    y_offset  =np.median(ops['yoff_list'][:5000])
    fovdeg = list()
    for s in metadata['metadata']['hRoiManager']['imagingFovDeg'].strip('[]').split(' '): fovdeg.extend(s.split(';'))
    fovdeg = np.asarray(fovdeg,float)
    fovdeg = [np.min(fovdeg),np.max(fovdeg)]
    rois = metadata['metadata']['json']['RoiGroups']['integrationRoiGroup']['rois']    
    if type(rois) == dict:
        rois = [rois]    
    centerXY_list = list()
    roinames_list = list()
    Lx = int(metadata['metadata']['hRoiManager']['pixelsPerLine'])
    Ly = int(metadata['metadata']['hRoiManager']['linesPerFrame'])
    try:
        yup,xup = upsample_block_shifts(Lx, Ly, ops['nblocks'], ops['xblock'], ops['yblock'], np.median(ops['yoff1_list'][:5000,:],0)[np.newaxis,:], np.median(ops['xoff1_list'][:5000,:],0)[np.newaxis,:])
        xup=xup.squeeze()#+x_offset 
        yup=yup.squeeze()#+y_offset 
    except:
        xup = np.zeros([Ly,Lx])
        yup = np.zeros([Ly,Lx])
    # =============================================================================
    #     plt.figure()
    #     img = plt.imshow(ops['meanImg'])#meanimg_dict['refImg_original'])
    #     img.set_clim(np.percentile(ops['meanImg'],[1,99]))
    # =============================================================================
    for roi in rois:
        if 'name' not in roi.keys():
            continue
        try:
            px,py = roi['scanfields']['centerXY']
            if 'rotation_deg' in meanimg_dict.keys():

                angle = -1*np.mean(np.asarray([np.arccos(meanimg_dict['rotation_matrix'][0,0]),
                                            np.arcsin(meanimg_dict['rotation_matrix'][1,0]),
                                            -1*np.arcsin(meanimg_dict['rotation_matrix'][0,1]),
                                            np.arccos(meanimg_dict['rotation_matrix'][1,1])]))
                #print('offset: {} pixels, rotation {} degrees'.format([x_offset,y_offset], np.degrees(angle)))
                ox = oy = 0
                qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
                qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
            else:
                qx = px
                qy = py
            coordinates_now = (np.asarray([qx,qy])-fovdeg[0])/np.diff(fovdeg)
            #centerXY_list.append((roi['scanfields']['centerXY']-fovdeg[0])/np.diff(fovdeg))
        except:
            print('multiple scanfields for {}'.format(roi['name']))
            px,py = roi['scanfields'][0]['centerXY']
            if 'rotation_deg' in meanimg_dict.keys():
                angle = -1*np.mean(np.asarray([np.arccos(meanimg_dict['rotation_matrix'][0,0]),
                                            np.arcsin(meanimg_dict['rotation_matrix'][1,0]),
                                            -1*np.arcsin(meanimg_dict['rotation_matrix'][0,1]),
                                            np.arccos(meanimg_dict['rotation_matrix'][1,1])]))
                print('rotating ROIs with {} degrees'.format(np.degrees(angle)))
                ox = oy = 0
                qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
                qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
            else:
                qx = px
                qy = py
            coordinates_now =(np.asarray([qx,qy])-fovdeg[0])/np.diff(fovdeg)
        #%

        coordinates_now = coordinates_now[::-1] # go to yx
        coordinates_now[0] = coordinates_now[0]*Ly
        coordinates_now[1] = coordinates_now[1]*Lx

        yoff_now = yup[int(coordinates_now[0]),int(coordinates_now[1])]
        xoff_now = xup[int(coordinates_now[0]),int(coordinates_now[1])]

        #lt.plot(coordinates_now[1],coordinates_now[0],'ro')        
        coordinates_now[0]-=yoff_now
        coordinates_now[1]-=xoff_now
        #plt.plot(coordinates_now[1],coordinates_now[0],'yo')

        centerXY_list.append(coordinates_now[::-1]) # go back to xy
        roinames_list.append(roi['name'])
    cond_s2p_idx = list()
    for roi_idx_now in roi_indices:    
        if roi_idx_now is None:
            cond_s2p_idx.append(None)
            continue
        med_list = list()
        dist_list = list()
        for cell_stat in stat:
            dist = np.sqrt((centerXY_list[roi_idx_now-1][0]-cell_stat['med'][1])**2+(centerXY_list[roi_idx_now-1][1]-cell_stat['med'][0])**2)# - cell_stat['radius']
            dist_list.append(dist)
            med_list.append(cell_stat['med'])
            #break
        cond_s2p_idx.append(np.argmin(dist_list))
        distances_all.append(dist_list)

    if calculate_cn_lickport_correlation:
        # calculate correlation between lickport steps and CN, could go to another function TODO
        reward_skip_frames = 0

        directory = os.path.split(session_ops_file)[0]
        F = np.load(os.path.join(directory,'F.npy'))
        F0 = np.load(os.path.join(directory,'F0.npy'))
        #Fneu = np.load(os.path.join(directory,'Fneu.npy'))
        dff = (F-F0)/F0
        with open(os.path.join(directory, "filelist.json")) as json_file:
            filelist_dict = json.load(json_file)  
        roi_idx_now = int(np.median(roi_indices))
        med_list = list()
        dist_list = list()
        for cell_stat in stat:

            dist = np.sqrt((centerXY_list[roi_idx_now-1][0]-cell_stat['med'][1])**2+(centerXY_list[roi_idx_now-1][1]-cell_stat['med'][0])**2)# - cell_stat['radius']
            dist_list.append(dist)
            med_list.append(cell_stat['med'])
        potential_cn = np.where(np.asarray(dist_list)<50)[0]    
        order = np.argsort(np.asarray(dist_list)[potential_cn])
        potential_cn = potential_cn[order]
        lickport_movements = np.zeros(dff.shape[1])
        reward_indices = []
        dts = []
        for trial_i, scanimage_filename in enumerate(behavior_dict['scanimage_file_names']):
            file_i = np.where(np.asarray(filelist_dict['file_name_list'])==scanimage_filename[0])[0][0]
            start_idx = np.sum(filelist_dict['frame_num_list'][:file_i])
            fr = float(behavior_dict['scanimage_tiff_headers'][trial_i][0]['frame_rate'])
            zaber_move_times = behavior_dict['zaber_move_forward'][trial_i]
            zaber_move_indices = np.asarray(zaber_move_times*fr,int)+start_idx
            zaber_move_indices = np.asarray(zaber_move_indices,int)
            lickport_movements[zaber_move_indices]+=1
            reward_time = behavior_dict['reward_L'][trial_i]
            if len(reward_time)>0:
                reward_indices.append(int(reward_time[0]*fr)+start_idx)
               # asdas
                dts.append(int(reward_time[0]*fr)+start_idx-zaber_move_indices[-1])
                #dts.append(reward_time[0]-zaber_move_times[-1])
        fr = float(behavior_dict['scanimage_tiff_headers'][0][0]['frame_rate'])
        # t = np.arange(0,5,1/fr)# ms 
        # decay = np.exp(t/-.1)
        # decay = decay/np.sum(decay)
        # convolved_lickport_movement = np.convolve(lickport_movements,decay,'full')
        convolved_lickport_movement = lickport_movements#convolved_lickport_movement[:len(lickport_movements)]
        for i in np.asarray(reward_indices,int):
            convolved_lickport_movement[i:i+reward_skip_frames] = 0


        c_list = []
        df_list = []
        for cn_i in potential_cn:
            maxv = np.percentile(dff[cn_i,:],99)
            df_now = dff[cn_i,:]/maxv
            df_now= plot_utils.rollingfun(df_now,4)
            for i in np.asarray(reward_indices,int):
                df_now[i:i+reward_skip_frames] = 0
            a = convolved_lickport_movement
            b = np.asarray(list(df_now)*3)
            # norm_a = np.linalg.norm(a)
            # a = a / norm_a
            # norm_b = np.linalg.norm(b)
            # b = b / norm_b

            c= np.correlate(a,b,'valid')
            df_list.append(df_now)
            c_list.append(c)

        step = 50
        idx = np.asarray(np.arange(-step,step) + np.round(len(c)/2),int)
        t = np.arange(-step,step)/fr
        fig = plt.figure(figsize = [20,20])
        ax1 = fig.add_subplot(2,2,1)

        ax2 = fig.add_subplot(2,2,2)
        ax_meanimage = fig.add_subplot(2,2,3)
        ax_rois = fig.add_subplot(2,2,4,sharex = ax_meanimage,sharey = ax_meanimage)
        im = ax_meanimage.imshow(ops['meanImg'])
        im.set_clim(np.percentile(ops['meanImg'].flatten(),[1,99.5]))
        mask = np.zeros_like(ops['meanImg'])
        for i,roi_stat in enumerate(stat):
                mask[roi_stat['ypix'],roi_stat['xpix']] = 1


        maskimg = ax_rois.imshow(mask)#,alpha = .5)    #cmap = 'hot',
        for i in potential_cn:
            ax_rois.plot(stat[i]['med'][1],stat[i]['med'][0],'.')
            ax_meanimage.plot(stat[i]['med'][1],stat[i]['med'][0],'.')

        maxval = 0
        for i_,(c,df,i) in enumerate(zip(c_list,df_list,potential_cn)):
            ax1.plot(t,c[idx],label = 'neuron {}'.format(i))#-c_auto[idx])
            if i_==0:
                idx_ = np.argmax(c[idx])
                delay = t[idx_]
                ax1.plot(t[idx_],c[idx][idx_],'ro')
                ax1.set_title('conditioned neuron - lickport delay: {0:.3f} s'.format(delay))

            maxval -=np.max(df[5000:6000])
            ax2.plot(df[5000:6000]-maxval)
        ax2.plot(convolved_lickport_movement[5000:6000],'k-')
        ax1.set_ylabel('cross correlation with lickport steps')
        ax1.set_xlabel('time (s)')
        ax1.legend()
        fig.savefig(os.path.join(directory,'lickport_cn_correlation.pdf'), format="pdf")




    #%  show ROIS
    if plot:
        fig_meanimage = plt.figure(figsize = [20,20])
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
        ax_rois.plot(np.asarray(centerXY_list)[:,0],np.asarray(centerXY_list)[:,1],'r.')
    if return_distances:
        return cond_s2p_idx,closed_loop_trial,scanimage_filenames,distances_all
    else:
        return cond_s2p_idx,closed_loop_trial,scanimage_filenames

