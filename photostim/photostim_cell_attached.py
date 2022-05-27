# this script loads the ephys from all the cell attached experiments and checks
# how the firing rate changes before and after the start of imaging
# not pretty but correct

import utils_ephys

import matplotlib.pyplot as plt
import datetime
import time as timer
import numpy as np
import os
from pywavesurfer import ws
from pathlib import Path
import time



#% load data
data_list = []
base_dir = '/mnt/Data/Calcium_imaging/raw/DOM3-MMIMS/cell_attached/'
subjects = os.listdir(base_dir)
counter = 0
for subject in subjects:
    if '.' in subject:
        continue
    
    sessions = os.listdir(os.path.join(base_dir,subject))
    if 'cel' in sessions[0]:
        sessions = ['']
    for session in sessions:
        cells = os.listdir(os.path.join(base_dir,subject,session))
        for cell in cells:
            if '.' in cell:
                continue
            print([subject,cell])
            cell_dict = {'subject':subject,
                         'cell':cell}
            ephysdata_cell = []
            dirs = os.listdir(os.path.join(base_dir,subject,session,cell))
            for dir_ in dirs:
                if '.' in dir_ or 'other' in dir_:
                    continue
                print(dir_)
                dir_now = Path(os.path.join(base_dir,subject,session,cell,dir_))
                ephysfiles = os.listdir(dir_now)
                ephysfiles_real = []
                
                for ephysfile in ephysfiles:
                    if '.h5' in ephysfile:
                        ephysfiles_real.append(ephysfile)
                for ephysfile in ephysfiles_real:
                    try:
                        ephysdata = utils_ephys.load_wavesurfer_file(dir_now.joinpath(ephysfile))#voltage, frame_trigger, sRate, recording_mode, timestamp, response_unit
                        
                    except OSError:
                        print('file not readable, skipping: {}'.format(dir_now.joinpath(ephysfile)))
                        continue
                    for sweep_i in range(len(ephysdata)):
                        trace = ephysdata[sweep_i]['AI-ephys-primary']
                        sample_rate =  ephysdata[sweep_i]['sampling_rate']
                        #ap_dict = utils_ephys.findAPs_cell_attached(trace, sample_rate,recording_mode= 'current clamp', SN_min = 5,method = 'diff') 
                        #ephysdata[sweep_i]['ap_dict'] = ap_dict
                    ephysdata_cell.extend(ephysdata)
            cell_dict['data'] = ephysdata_cell
            data_list.append(cell_dict)
       
# =============================================================================
#     if len(ephysdata_cell)>0:
#         counter+=1
#         if counter>10:
#             break
# =============================================================================
#%% - look at the start of imaging
import scipy.signal as signal
freq_before = []
freq_after = []
for data_now in data_list:
    if len(data_now['data'])==0:
        continue
    for i,sweep in enumerate(data_now['data']):
        print([data_now['subject'],data_now['cell']])
        repeat = True
        start_idx = start_idx_new = 0
        while repeat:
            exposure = sweep['DI-FrameClock'][start_idx:]
            exposure = np.concatenate([[0],np.diff(exposure)])
            peaks = signal.find_peaks(exposure,height = 1)
            peaks_idx = peaks[0]
            #%
            difi = np.diff(peaks_idx)
            #difi[-1]=1500
            median_dif = np.median(difi)
            if sum(difi>10*median_dif)>0:
                needed_idx_end = np.argmax(np.concatenate([[0],difi])>10*median_dif)
                start_idx_new = start_idx+ peaks_idx[needed_idx_end-1]
                peaks_idx=peaks_idx[:needed_idx_end]
                print('imaging has stopped transiently !! check frame times')
                repeat = True
                
            else:
                repeat = False
            if len(peaks_idx) == 0:
                print('no frames, skipping')
                break
            baseline_length = peaks_idx[0]
            imaging_length = peaks_idx[-1] - peaks_idx[0]
            len_to_plot = np.min([baseline_length,imaging_length,int(sweep['sampling_rate']*30)])-1
            if len_to_plot<sweep['sampling_rate']:
                print('less than 1 sec, aborting')
                break
            plot_range_idx = [peaks_idx[0]-len_to_plot,peaks_idx[0]+len_to_plot]
            
            time = np.arange(-len_to_plot,len_to_plot,1)/sweep['sampling_rate']
            voltage = sweep['AI-ephys-primary'][start_idx+plot_range_idx[0]:start_idx+plot_range_idx[1]]
            ap_dict = utils_ephys.findAPs_cell_attached(voltage, sweep['sampling_rate'],recording_mode= 'current clamp', SN_min = 5,method = 'diff') 
            frames = sweep['DI-FrameClock'][start_idx+plot_range_idx[0]:start_idx+plot_range_idx[1]]
            
            
            fig = plt.figure()
            ax_ephys = fig.add_subplot(3,1,1)
            ax_frames = fig.add_subplot(3,1,2,sharex = ax_ephys)
            ax_snr = fig.add_subplot(3,1,3)
            ax_ephys.plot(time,voltage)
            ax_ephys.set_title('{} - cell {} - sweep {}'.format(data_now['subject'],data_now['cell'],i))
            ax_frames.plot(time,frames)
            #aps_needed = (ap_dict['peak_idx']>plot_range_idx[0]+start_idx) & (ap_dict['peak_idx']<plot_range_idx[1]+start_idx)
            ax_snr.hist(ap_dict['peak_snr_v'][np.isinf(ap_dict['peak_snr_v']) == False])
            #if len(aps_needed)>0:
            ap_idxs = ap_dict['peak_idx']#[aps_needed]
            ax_ephys.plot(time[ap_idxs],voltage[ap_idxs],'ro')
            start_idx = start_idx_new
            ap_num_before = sum(ap_idxs<len_to_plot)
            ap_num_after  = sum(ap_idxs>len_to_plot)
            freq_before.append(ap_num_before/(len_to_plot/sweep['sampling_rate']))
            freq_after.append(ap_num_after/(len_to_plot/sweep['sampling_rate']))
        #break
    #break
#%%
fig = plt.figure()
ax_compare = fig.add_subplot(2,1,1)
ax_d = fig.add_subplot(2,1,2)
diff = []
for before,after in zip(freq_before,freq_after):
    if before>10 or before<.25:
        continue
    ax_compare.plot(['before imaging','during imaging'],[before,after],'ro-',alpha = .8)
    diff.append(after/before)
ax_d.hist(diff,20)
ax_compare.set_ylabel('firing rate (Hz)')
ax_d.set_xlabel('Firing rate fold-change during imaging')