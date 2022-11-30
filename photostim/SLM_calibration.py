#%% read movie file - SLM calibration with pyspincapture - synchronized photostim and capturing with the sub-stage camera - camera triggers photostim!
import numpy as np
import os
import tifffile
from scipy.signal import find_peaks, medfilt2d
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import imageio
def gauss(x, a, sigma,c):
    c=0
    return a*np.exp(-(x)**2/(sigma**2))+c
def sigmoid(x, var , shift):
    y = 1-np.exp(var*(x-shift)) / (1+np.exp(var*(x-shift)))
    return (y)
def hypSec(x,a,b):
    return a/np.cosh(x/b)#+c



degrees_per_micron = 0.0189


dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_random_points_with_comp_sigma_offset_low_power/'
dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_random_points_with_comp_sigma_offset/'
dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_random_points_with_comp_sigma/'
dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_random_points_with_comp_sigma_offset_low_power/'

dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_random_points_no_comp_low_power/'
dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_random_points_no_comp/'

dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_random_spiral_points_no_comp_5percent/'

dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_random_spiral_points_no_comp_4percent/'
dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_random_spiral_points_with_comp_4percent/'
dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_random_spiral_points_with_comp_4percent_no_offset/'


dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_random_spiral_points_no_comp_3percent/'
dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_random_spiral_points_with_comp_3percent_3sigma/'
dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_random_spiral_points_with_comp_3percent_2sigma/'
dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_random_spiral_points_with_comp_3percent_1sigma/'

dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_random_spiral_points_with_comp_3percent_big_fov/'
dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_random_spiral_points_with_comp_3percent_big_fov_more_power/'

dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_random_spiral_points_no_comp_3percent/'

dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_points_around_brightness_center_no_correction/'
dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_points_around_brightness_center_with_correction/'
dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_points_around_brightness_center_with_correction_no_offset/'
dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_points_around_brightness_center_with_correction_no_offset_smaller_sigma_2.875/'
dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_points_around_brightness_center_with_correction_smaller_sigma_2.875/'

dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/5_points_10_groups_power_ramp_0_to_10_percent_in_21_steps/'



#dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/fluorescein_5_random_spots/'
#dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/fluorescein_5_random_spots_correction/'
#dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/fluorescein_5_random_spots_correction_no_offset/'

dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/fluorescein_5_random_spots_2/'
dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/fluorescein_5_random_spots_correction_2/'


dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/fluorescein_5_random_spots_10Mhz/'
dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/fluorescein_5_random_spots_10Mhz_correction/'
dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/fluorescein_5_random_spots_10Mhz_correction_no_offset/'


dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/fluorescein_5_random_spots_10Mhz_40pc/'
dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/fluorescein_5_random_spots_10Mhz_40pc_correction/'
#dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/fluorescein_5_random_spots_10Mhz_40pc_correction_no_offset/'

#dirname = '/home/rozmar/Network/BergamoVideo/Behavior_videos/bottom/slm_test/fluorescein_5_random_spots_10Mhz_30pc_again/'

files = os.listdir(dirname)
for file in files:
    if '.mp4' in file:
        break
filename  =os.path.join(dirname,file)
vid = imageio.get_reader(filename,  'ffmpeg')
idx = 0
im_list = []
while True:
    try:
        im = vid.get_data(idx)
        im_list.append(np.mean(im,2))
        idx+=1
    except:
        break
vid.close()
im_array = np.asarray(im_list)
#%


minimum_distance_from_zeroth_order_spot = 10 # to ignore 0th order spot
anticipated_spots_ = 6
neigbourhood_pixel_num= 3
neighborhood_to_remove = 20
calibration_image_num = 3
center_indices = []
snippets = []
intensities = []
max_intensities = []
calibration_indices = []
frame_indices = []
above_noise_sd_level = 10
im_montage = np.zeros(im_array[0,:,:].shape)*np.nan
for i_frame,average_image in enumerate(im_array):
    average_image -=np.nanmedian(average_image.flatten())# remove background if any
    if i_frame<calibration_image_num:
        anticipated_spots = 1
    else:
        anticipated_spots  = anticipated_spots_
    im_f = ndimage.gaussian_filter(average_image,2)
    for i in range(anticipated_spots):
        center_idx = np.unravel_index(np.argmax(im_f),im_f.shape)
        snippet_small = average_image[center_idx[0]-1:center_idx[0]+2,
                                      center_idx[1]-1:center_idx[1]+2].copy()
        center_val =  snippet_small[1,1]
        snippet_small[1,1] = np.nan
        surround_ratio= center_val/np.nanmean(snippet_small)
        
            
            
        
        if np.max(im_f)>np.mean(im_f)+above_noise_sd_level*np.std(average_image) and not (np.isnan(surround_ratio) or surround_ratio>10):
            if i_frame<calibration_image_num:
                calibration_indices.append(center_idx)
            else:
                center_indices.append(center_idx)
                snippet = average_image[center_idx[0]-neigbourhood_pixel_num:center_idx[0]+neigbourhood_pixel_num,
                           center_idx[1]-neigbourhood_pixel_num:center_idx[1]+neigbourhood_pixel_num]-np.mean(average_image)
                snippets.append(snippet)
                im_montage[center_idx[0]-neigbourhood_pixel_num:center_idx[0]+neigbourhood_pixel_num,
                           center_idx[1]-neigbourhood_pixel_num:center_idx[1]+neigbourhood_pixel_num]=np.mean(snippet)
                intensities.append(np.mean(snippet))
                max_intensities.append(np.max(snippet))
                frame_indices.append(i_frame)
                
        step_left = 1
        step_right = 1
        step_up = 1
        step_down = 1
        while center_idx[0]-step_left>0 and im_f[center_idx[0]-step_left,center_idx[1]]<im_f[center_idx[0]-step_left+1,center_idx[1]]:
            step_left+=1
        while center_idx[0]+step_right<average_image.shape[0] and im_f[center_idx[0]+step_right,center_idx[1]]<im_f[center_idx[0]+step_right-1,center_idx[1]]:
            step_right+=1
        while center_idx[0]-step_down>0 and im_f[center_idx[0]-step_down,center_idx[1]]<im_f[center_idx[0]-step_down+1,center_idx[1]]:
            step_down+=1
        while center_idx[0]+step_up<average_image.shape[0] and im_f[center_idx[0]+step_up,center_idx[1]]<im_f[center_idx[0]+step_up-1,center_idx[1]]:
            step_up+=1
        im_f[center_idx[0]-step_left:center_idx[0]+step_right,
            center_idx[1]-step_down:center_idx[1]+step_up] = 0
#%
center_indices = np.asarray(center_indices)  
mean_intensities = np.asarray(intensities)    
max_intensities = np.asarray(max_intensities)    
frame_indices = np.asarray(frame_indices)

calibration_indices = np.asarray(calibration_indices)
center_idx = calibration_indices[0]

dist_zeroth = np.sqrt((center_indices[:,0]-center_idx[0])**2+(center_indices[:,1]-center_idx[1])**2)
indices = dist_zeroth<minimum_distance_from_zeroth_order_spot # 10 pixel is the minimum distance from the 0th order spot
zeroth_indices = center_indices[indices]
zeroth_mean_intensities = mean_intensities[indices]
zeroth_max_intensities = max_intensities[indices]

center_indices = center_indices[indices==False]
mean_intensities = mean_intensities[indices==False]
max_intensities = max_intensities[indices==False]
frame_indices  = frame_indices[indices ==False]
#%
x_dir = calibration_indices[1]-calibration_indices[0]
y_dir = calibration_indices[2]-calibration_indices[0]

#% find the center of intensity

#%
offset_list = np.arange(-50,50,1)*3
rsq_image = np.zeros([len(offset_list),len(offset_list)])

for x_i, offset_x in enumerate(offset_list):
    for y_i,offset_y in enumerate(offset_list):
        #%
        #%
        center_spot = np.asarray(center_idx) + np.asarray([offset_x,offset_y])


        distance = np.sqrt((center_indices[:,0]-center_spot[0])**2+(center_indices[:,1]-center_spot[1])**2)
        
        popt, pcov = curve_fit(gauss, 
                               np.concatenate([distance,distance*-1]), 
                               np.asarray(list(mean_intensities)*2),
                               p0 = [1,300,0])
    
        rsq = np.mean((mean_intensities-gauss(distance, popt[0], popt[1],popt[2]))**2)
        
        #%
        rsq_image[y_i,x_i] = rsq
optimal_offset_idx =         [np.unravel_index(np.argmin(rsq_image),rsq_image.shape)[1],
                              np.unravel_index(np.argmin(rsq_image),rsq_image.shape)[0]]
optimal_offset = [offset_list[np.unravel_index(np.argmin(rsq_image),rsq_image.shape)[1]],
                  offset_list[np.unravel_index(np.argmin(rsq_image),rsq_image.shape)[0]]]

#%
fig = plt.figure()
ax_rsq = fig.add_subplot(1,1,1)
im = ax_rsq.imshow(rsq_image,extent=[offset_list[0],offset_list[-1],offset_list[-1],offset_list[0]])
im .set_clim(np.percentile(rsq_image,[1,99]))
ax_rsq.set_xlabel('x offset')
ax_rsq.set_ylabel('y offset')
ax_rsq.set_title('center offset to minimize error in gaussian fit: {} pixels'.format(optimal_offset))
ax_rsq.plot(optimal_offset[0],optimal_offset[1],'ro')
#%%

correct_for_center = False
nan_window = 5
im_montage_corr = im_montage.copy()
for i in range(30):
    for center_x in range(int(nan_window),im_montage.shape[0],nan_window*2):
        for center_y in range(int(nan_window),im_montage.shape[1],nan_window*2):
            offset = int(np.random.uniform(nan_window)-nan_window/2)

            center_x += offset
            center_y += offset
            if any(np.isnan(im_montage_corr[center_x-nan_window:center_x+nan_window,center_y-nan_window:center_y+nan_window].flatten())) and any(np.isnan(im_montage_corr[center_x-nan_window:center_x+nan_window,center_y-nan_window:center_y+nan_window].flatten())==False):
                val = np.nanmedian(im_montage_corr[center_x-nan_window:center_x+nan_window,center_y-nan_window:center_y+nan_window].flatten())
                for x in np.arange(center_x-nan_window,np.min([im_montage_corr.shape[0],center_x+nan_window])):
                    for y in np.arange(center_y-nan_window,np.min([im_montage_corr.shape[1],center_y+nan_window])):
                        if np.isnan(im_montage_corr[x,y]):
                            im_montage_corr[x,y] = val
                        #im_montage_corr[center_x-nan_window:center_x+nan_window,center_y-nan_window:center_y+nan_window]= 
        
    
#%
#im_montage[nans] = 0
if correct_for_center:
    center_idx_corrected = center_idx + optimal_offset
else:
    center_idx_corrected = center_idx
distance = np.sqrt((center_indices[:,0]-center_idx_corrected[0])**2+(center_indices[:,1]-center_idx_corrected[1])**2)

fig = plt.figure(figsize = [10,10])
ax_calibration = fig.add_subplot(3,2,1)
ax_calibration.set_title(dirname.split('/')[-2])
ax_calibration.imshow(im_montage_corr.T,aspect = 'auto')#plot(calibration_indices[:,0],calibration_indices[:,1],'k.',alpha = .5)
ax_point_dist = fig.add_subplot(3,2,2,sharex = ax_calibration,sharey = ax_calibration)
ax_point_dist.plot(center_indices[:,0],center_indices[:,1],'k.',alpha = .1)
ax_point_dist.plot(center_idx[0],center_idx[1],'ko')

ax_point_dist.plot(center_idx_corrected[0],center_idx_corrected[1],'ro')
ax_calibration.plot(center_idx[0],center_idx[1],'ko')
ax_calibration.plot(center_idx_corrected[0],center_idx_corrected[1],'ro')
ax_distance_dep = fig.add_subplot(3,2,3)
ax_distance_dep .plot(distance,mean_intensities,'k.',alpha = .5)
var_per_mean_needed = distance<200
var_per_mean = np.nanvar(mean_intensities[var_per_mean_needed])/np.nanmean(mean_intensities[var_per_mean_needed])
ax_distance_dep.set_title('var/mean up to 200 pix: {}'.format(np.round(var_per_mean,3)))

ax_distance_dep_max = fig.add_subplot(3,2,4,sharex = ax_distance_dep)
ax_distance_dep_max .plot(distance,max_intensities,'r.',alpha = .5)


popt_gauss, pcov = curve_fit(gauss, 
                       np.concatenate([distance,distance*-1]), 
                       np.asarray(list(mean_intensities)*2),
                       p0 = [1,300,0])
gauss_x = np.arange(0, np.percentile(distance,99),1)
gauss_y = gauss(gauss_x, popt_gauss[0], popt_gauss[1], popt_gauss[2])
ax_distance_dep.plot(gauss_x,gauss_y,'r-')

popt_sigm, pcov = curve_fit(hypSec, 
                       np.concatenate([distance,distance*-1]), 
                       np.asarray(list(mean_intensities)*2),
                       p0 = [1,300])

hypsec_x = np.arange(0, np.percentile(distance,99),1)
hypsec_y = hypSec(hypsec_x, popt_sigm[0], popt_sigm[1])
ax_distance_dep.plot(hypsec_x,hypsec_y,'b-')

fitted_F = gauss(distance, 1, popt_gauss[1],0)
multiplier = 1/fitted_F
corrected_f = (mean_intensities-popt_gauss[2])*multiplier
corrected_f[corrected_f>255]=255
ax_corrected   = fig.add_subplot(3,2,5,sharex = ax_distance_dep)
ax_corrected .plot(distance,corrected_f,'b.',alpha = .5)
var_per_mean_needed = distance<200
var_per_mean = np.nanvar(corrected_f[var_per_mean_needed])/np.nanmean(corrected_f[var_per_mean_needed])
ax_corrected.set_title('var/mean up to 200 pix: {}'.format(np.round(var_per_mean,3)))


corrected_f = (max_intensities-popt_gauss[2])*multiplier
corrected_f[corrected_f>255]=255
ax_corrected_max   = fig.add_subplot(3,2,6,sharex = ax_distance_dep)
ax_corrected_max .plot(distance,corrected_f,'b.',alpha = .5)


offset_si = np.linalg.inv(np.array([x_dir, y_dir])).dot(optimal_offset)*-1
sigma_si = popt_gauss[1]/np.mean([np.sqrt(np.sum(x_dir**2)),np.sqrt(np.sum(y_dir**2))])

ax_point_dist.set_title('offset: {}, sigma: {}'.format(np.round(offset_si,3),np.round(sigma_si,3)))
#%
histbins = np.arange(.5,1.5,.05)
fig_hists = plt.figure(figsize = [10,10])
ax_uncorrected = fig_hists.add_subplot(2,1,1)
ax_uncorrected.hist(mean_intensities[var_per_mean_needed]/np.nanmean(mean_intensities[var_per_mean_needed]),histbins,color = 'black')
ax_uncorrected.set_xlabel('mean normalized intensity')
ax_uncorrected.set_title(dirname.split('/')[-2])
ax_corrected = fig_hists.add_subplot(2,1,2)
ax_corrected.hist(corrected_f[var_per_mean_needed]/np.nanmean(corrected_f[var_per_mean_needed]),histbins,color = 'blue')
ax_corrected.set_xlabel('mean normalized intensity')
#%% check if there are just messed up groups
group_variances = []
group_indices = []
group_maxes = []
fig = plt.figure(figsize = [10,10])
ax_1 = fig.add_subplot(2,1,1)
ax_2 = fig.add_subplot(2,1,2)
for i in np.unique(frame_indices[var_per_mean_needed]):
    intensity_now = mean_intensities[var_per_mean_needed][frame_indices[var_per_mean_needed] == i]
    distance_now = distance[var_per_mean_needed][frame_indices[var_per_mean_needed] == i]
    order = np.argsort(distance_now)
    ax_1.plot(distance_now[order],intensity_now[order])
    group_variances.append(np.var(intensity_now))
    group_indices.append(i)
    group_maxes.append(np.max(intensity_now))
ax_2.plot(group_indices,group_variances,'ko')
ax_2.set_xlabel('group idx')
ax_2.set_ylabel('variance')
ax_1.set_ylabel('mean intensity')
ax_1.set_xlabel('distance from center')

#%% calculate offset of center and sigma of decay

offset_si = np.linalg.inv(np.array([x_dir, y_dir])).dot(optimal_offset)*-1
sigma_si = popt_gauss[1]/np.mean([np.sqrt(np.sum(x_dir**2)),np.sqrt(np.sum(y_dir**2))])
#%% visualize power ramps
from statistics import mode
power_percentages = np.arange(0,10.1,.5)
#center_indices = center_indices[indices==False]
#mean_intensities = mean_intensities[indices==False]
#max_intensities = max_intensities[indices==False]
max_distance = 1
center_indices_ramp = []
mean_intensities_ramp =[]
max_intensities_ramp = []
ramp_len = []
for center_idx in center_indices :
    indices = np.sqrt(np.sum((center_indices-center_idx)**2,1))<=max_distance
    indices_now = np.nanmean(center_indices[indices,:],0)
    try:
        if np.any(np.sqrt(np.sum((np.asarray(center_indices_ramp)-indices_now)**2,1))<=max_distance):
            continue
    except:
        pass
    center_indices_ramp.append(indices_now)
    mean_intensities_ramp.append(mean_intensities[indices])
    max_intensities_ramp.append(max_intensities[indices])
    ramp_len.append(np.sum(indices))
    

needed = np.arange(len(ramp_len))#np.asarray(ramp_len)==mode(ramp_len)
fig = plt.figure()
ax0 = fig.add_subplot(2,1,1)
ax = fig.add_subplot(2,1,2)
ramps = []
for i in np.where(needed)[0]:
    ramp_now = max_intensities_ramp[i]#mean_intensities_ramp[i]
    if np.max(ramp_now) == ramp_now[-1]:
        if len(power_percentages)==len(ramp_now):
            ramps.append(ramp_now)
        elif len(power_percentages)<len(ramp_now):
            ramps.append(ramp_now[-len(power_percentages):])
        else:
            ramp_with_nans = np.ones(len(power_percentages))*np.nan
            ramp_with_nans[len(power_percentages)-len(ramp_now):] = ramp_now
            ramps.append(ramp_with_nans)
        
ramps = np.asarray(ramps)
sqrt_ramps = np.sqrt(ramps/ramps[:,-1][:,np.newaxis])

y_vals = np.nanmedian(sqrt_ramps,0)
x_vals = power_percentages
needed = np.isnan(y_vals)==False
y_vals = y_vals[needed]
x_vals = x_vals[needed]
p = np.polyfit(x_vals,y_vals,1)
y_fit = np.polyval(p,power_percentages)
scale_val = 1#1/np.polyval(p,100)*100#np.polyval(p,100)
scale_val = (100*power_percentages[-1])/(100+power_percentages[-1]*np.polyval(p,0))
offset = scale_val*np.polyval(p,0)
scale_val = 1
#power_percentages[-1]*(100/scale_val)
for r in ramps:
    ax.plot(power_percentages,np.sqrt(r/np.nanmax(r))*scale_val)
    ax0.plot(power_percentages,(r/np.nanmax(r))*scale_val)

ax.plot(power_percentages,y_fit*scale_val,'k-',linewidth = 4)
ax0.plot(power_percentages,scale_val*y_fit**2,'k-',linewidth = 2)
#ax.set_ylim([0,power_percentages[-1]])
#ax.set_xlim([0,power_percentages[-1]])
ax.set_xlabel('power %')
ax.set_ylabel('normalized sqrt(intensity)')
ax0.set_ylabel('normalized intensity')
ax0.set_title('offset error: {}%'.format(np.round(offset,3)))
#%%  slmscan
import numpy as np
import os
import tifffile
from scipy.signal import find_peaks, medfilt2d
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
source_dir = '/home/rozmar/Network/Bergamo2P/Imaging/test/substage_psf/slmscan_monaco_power/slmscan_32x32_3power_1xmag/'
source_dir = '/home/rozmar/Network/Bergamo2P/Imaging/test/substage_psf/slmscan_scanimage_power/slmscan_32x_1xpower_with_0th_order_spot/'
#microns_per_pixel = 1.557 #50 micron grid
microns_per_pixel = 1.246 #40 micron grid

files = os.listdir(source_dir)
tiff_stack = []
for file in files:
    tiff_orig = tifffile.imread(os.path.join(source_dir,file))
    if len(tiff_orig.shape) ==3:
        tiff_corr = np.mean(tiff_orig,2)    
        tiff_stack.append(tiff_corr)
    else:
        tiff_stack.append(tiff_orig)
    print(file)
tiff_stimport imageio
filename = '/tmp/file.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')
nums = [10, 287]
for num in nums:
    image = vid.get_data(num)ack = np.asarray(tiff_stack)
average_image = np.mean(tiff_stack,0)
std_image = np.std(tiff_stack,0)



#% grab point from each image
neigbourhood_pixel_num = 2
above_noise_sd_level = 3
sample_image = np.zeros(average_image.shape)
center_list = []
intensiy_list = []
snippet_list = []
for i,im in enumerate(tiff_stack):
    print(i)
    #im = tiff_stack[300,:,:]
    im[:10,:] = 0
    im[-10:,:] = 0
    im[:,:10] = 0
    im[:,-10:] = 0
    im_f = ndimage.gaussian_filter(im,2)
    
    center_idx = np.unravel_index(np.argmax(im_f),im_f.shape)
    if np.max(im_f)>np.mean(im_f)+above_noise_sd_level*np.std(im_f):
        sample_image[center_idx[0]-neigbourhood_pixel_num:center_idx[0]+neigbourhood_pixel_num,
                     center_idx[1]-neigbourhood_pixel_num:center_idx[1]+neigbourhood_pixel_num] = im[center_idx[0]-neigbourhood_pixel_num:center_idx[0]+neigbourhood_pixel_num,
                                                                                                     center_idx[1]-neigbourhood_pixel_num:center_idx[1]+neigbourhood_pixel_num]
        print(np.max(im_f)/(np.mean(im_f)+10*np.std(im_f)))
        center_list.append(center_idx)
        point_snippet = im[center_idx[0]-neigbourhood_pixel_num:center_idx[0]+neigbourhood_pixel_num,
                           center_idx[1]-neigbourhood_pixel_num:center_idx[1]+neigbourhood_pixel_num]
        intensiy_list.append(np.mean(point_snippet))
        snippet_list.append(point_snippet)
    else:
        print('noise')
        
        
#% select brightest point in case there are multiple measuerements for a spot
max_distance = 10 #pixels
center_array = np.asarray(center_list)
intensity_array = np.asarray(intensiy_list)
center_list_clean = []
intensity_list_clean = []
snippet_list_clean = []
sample_image = np.zeros(average_image.shape)
for i,(center,intensity,snippet) in enumerate(zip(center_list,intensiy_list,snippet_list)):
    add_point =  False
    distances = np.sqrt((center[0]-center_array[:,0])**2 + (center[1]-center_array[:,1])**2)
    needed = distances<max_distance
    if np.sum(needed)>1:
        
        if i == np.where(needed)[0][np.argmax(intensity_array[needed])]:
            add_point = True
        else:
            print('got bigger')
    else:
        add_point = True
    if add_point:
        center_list_clean.append(center)
        intensity_list_clean.append(intensity)
        snippet_list_clean.append(snippet)
        sample_image[center[0]-neigbourhood_pixel_num:center[0]+neigbourhood_pixel_num,
                     center[1]-neigbourhood_pixel_num:center[1]+neigbourhood_pixel_num] = snippet
        
ps = np.percentile(intensity_list_clean,[5,95])        
intensity_list_clean = list((intensity_list_clean-ps[0])/np.diff(ps))
#%%
#% plot mean and std images
std_per_average = std_image/average_image
fig =  plt.figure()
ax_mean = fig.add_subplot(2,2,1)
im = ax_mean.imshow(average_image)
im.set_clim(np.percentile(average_image.flatten(),[1,99]))
ax_std = fig.add_subplot(2,2,2,sharex  = ax_mean,sharey = ax_mean)
im = ax_std.imshow(std_image)
im.set_clim(np.percentile(std_image.flatten(),[1,99]))
ax_mean_per_std = fig.add_subplot(2,2,3,sharex  = ax_mean,sharey = ax_mean)
im = ax_mean_per_std.imshow(std_per_average)
im.set_clim(np.percentile(std_per_average.flatten(),[1,99]))

ax = fig.add_subplot(2,2,4,sharex  = ax_mean,sharey = ax_mean)
ax.invert_yaxis()
center_array = np.asarray(center_list_clean)
for i,center in enumerate(center_array):
    ax.plot(center[1],center[0],'k.',alpha=.4)
    ax.text(center[1],center[0],str(i))

#%% PICK CENTER SPOT based on the images:
center_spot_orig = [622,516]
x_scanning_drection_points  = [20,21]
y_scanning_direction_points = [20,36]
x_dir = center_array[x_scanning_drection_points[1],:] - center_array[x_scanning_drection_points[0],:]
x_dir = x_dir/np.sqrt(np.sum(x_dir**2))
y_dir = center_array[y_scanning_direction_points[1],:] - center_array[y_scanning_direction_points[0],:]
y_dir = y_dir/np.sqrt(np.sum(y_dir**2))
#%%

from scipy.optimize import curve_fit

def gauss(x, a, sigma):
    return a*np.exp(-(x)**2/(sigma**2))
def sigmoid(x, var , shift):
    y = 1-np.exp(var*(x-shift)) / (1+np.exp(var*(x-shift)))
    return (y)
def hypSec(x,a,b):
    return a/np.cosh(x/b)#+c
#%
offset_list = np.arange(-100,100,1)*1
rsq_image = np.zeros([len(offset_list),len(offset_list)])

for x_i, offset_x in enumerate(offset_list):
    for y_i,offset_y in enumerate(offset_list):
        center_spot = np.asarray(center_spot_orig) + np.asarray([offset_x,offset_y])


        center_array = np.asarray(center_list_clean)
        distances = np.sqrt((center_spot[1]-center_array[:,0])**2 + (center_spot[0]-center_array[:,1])**2)*microns_per_pixel
        
        
        popt, pcov = curve_fit(gauss, 
                               np.concatenate([distances,distances*-1]), 
                               np.asarray(intensity_list_clean*2),
                               p0 = [1,300])
        rsq = np.mean((intensity_list_clean-gauss(distances, popt[0], popt[1]))**2)
        
        
# =============================================================================
#         popt, pcov = curve_fit(sigmoid,
#                                distances,
#                                intensity_list_clean,
#                                p0 = [1,300])
#         rsq = np.mean((intensity_list_clean-sigmoid(distances, popt[0], popt[1]))**2)
# =============================================================================
        
        rsq_image[y_i,x_i] = rsq
optimal_offset_idx =         [np.unravel_index(np.argmin(rsq_image),rsq_image.shape)[1],
                              np.unravel_index(np.argmin(rsq_image),rsq_image.shape)[0]]
optimal_offset = [offset_list[np.unravel_index(np.argmin(rsq_image),rsq_image.shape)[1]],
                  offset_list[np.unravel_index(np.argmin(rsq_image),rsq_image.shape)[0]]]
#%%
fig = plt.figure()
ax_rsq = fig.add_subplot(1,1,1)
im = ax_rsq.imshow(rsq_image)
im .set_clim(np.percentile(rsq_image,[1,99]))
ax_rsq.set_xlabel('x offset')
ax_rsq.set_ylabel('y offset')
ax_rsq.set_title('center offset to minimize error in gaussian fit: {} pixels'.format(optimal_offset))
ax_rsq.plot(optimal_offset_idx[0],optimal_offset_idx[1],'ro')
#%%

center_spot = np.asarray(center_spot_orig)+optimal_offset
#center_spot = [631,576]

center_array = np.asarray(center_list_clean)
distances = np.sqrt((center_spot[1]-center_array[:,0])**2 + (center_spot[0]-center_array[:,1])**2)*microns_per_pixel
popt_gauss, pcov = curve_fit(gauss, 
                       np.concatenate([distances,distances*-1]), 
                       np.asarray(intensity_list_clean*2),
                       p0 = [1,300])

fig = plt.figure()
ax_montage = fig.add_subplot(2,1,1)
im = ax_montage.imshow(sample_image)
im.set_clim(np.percentile(sample_image.flatten(),[.1,99.9]))
ax_montage.plot(center_spot[0],center_spot[1],'ro')
ax_distance_dependence = fig.add_subplot(2,1,2)
ax_distance_dependence.plot(distances,intensity_list_clean,'ko')
gauss_x = np.arange(0, np.max(distances),1)
gauss_y = gauss(gauss_x, popt_gauss[0], popt_gauss[1])
ax_distance_dependence.plot(gauss_x,gauss_y,'r-')
#%
popt_sigm, pcov = curve_fit(hypSec, 
                       np.concatenate([distances,distances*-1]), 
                       np.asarray(intensity_list_clean*2),
                       p0 = [1,300])

hypsec_x = np.arange(0, np.max(distances),1)
hypsec_y = hypSec(hypsec_x, popt_sigm[0], popt_sigm[1])
ax_distance_dependence.plot(hypsec_x,hypsec_y,'b-')



popt, pcov = curve_fit(sigmoid,
                       distances,
                       intensity_list_clean,
                       p0 = [1,300])
#popt=[-2,3,0]
sigmoid_x = np.arange(0, np.max(distances),1)
sigmoid_y = sigmoid(sigmoid_x, popt[0], popt[1])
ax_distance_dependence.plot(sigmoid_x,sigmoid_y,'y-')

#%% transform offset to imaging space
offset_microns = np.asarray(optimal_offset)*microns_per_pixel
offset_microns[1]= offset_microns[1]*-1
offset_microns_scanimage = np.linalg.inv(np.array([x_dir, y_dir])).dot(offset_microns)
length_constant = popt_gauss[1]

#%%
average_image = np.mean(tiff_stack,0)
average_image = ndimage.gaussian_filter(average_image,2)
average_image[:100,:] = 0
average_image[-100:,:] = 0
average_image[:,:100] = 0
average_image[:,-100:] = 0
center_idx = np.unravel_index(np.argmax(average_image),average_image.shape)

center_trace = np.mean(tiff_stack[:,
                                  center_idx[0]-neigbourhood_pixel_num:center_idx[0]+neigbourhood_pixel_num,
                                  center_idx[1]-neigbourhood_pixel_num:center_idx[1]+neigbourhood_pixel_num],
                       axis = (1,2))
t_max = np.argmax(center_trace)
average_image = np.mean(tiff_stack[t_max:t_max+50],0)
average_image = ndimage.gaussian_filter(average_image,2)

#%%SLM videos to PSF
import numpy as np
import os
import tifffile
from scipy.signal import find_peaks, medfilt2d
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
#%matplotlib qt


# load images
#microns_per_pixel = 1.557 #50 micron grid
microns_per_pixel = 1.246 #40 micron grid
source_dir = '/home/rozmar/Network/Bergamo2P/Imaging/test/substage_psf/zstack_7xgrid_power3-no_comp/'
#source_dir = '/home/rozmar/Network/Bergamo2P/Imaging/test/substage_psf/zstack_7xgrid_power3_to9comp/'
#source_dir = '/home/rozmar/Network/Bergamo2P/Imaging/test/substage_psf/zstack_7xgrid_power3_to9comp_with_offset/'
Z_rounds = 3
zrange = 100 #microns
files = os.listdir(source_dir)
tiff_stack = []
for file in files:
    tiff_orig = tifffile.imread(os.path.join(source_dir,file))
    try:
        tiff_corr = np.mean(tiff_orig,2)
    except:
        tiff_corr = tiff_orig

    #tiff_corr = tiff_corr - medfilt2d(tiff_corr,21)
    tiff_stack.append(tiff_corr)
    print(file)
tiff_stack = np.asarray(tiff_stack)
#%%
z_range_to_show = [-10,10]
anticipated_spots = 51
neigbourhood_pixel_num = 4
center_indices = []
center_traces = []
average_image = np.mean(tiff_stack,0)
average_image = ndimage.gaussian_filter(average_image,2)
average_image[:100,:] = 0
average_image[-100:,:] = 0
average_image[:,:100] = 0
average_image[:,-100:] = 0
center_idx = np.unravel_index(np.argmax(average_image),average_image.shape)

center_trace = np.mean(tiff_stack[:,
                                  center_idx[0]-neigbourhood_pixel_num:center_idx[0]+neigbourhood_pixel_num,
                                  center_idx[1]-neigbourhood_pixel_num:center_idx[1]+neigbourhood_pixel_num],
                       axis = (1,2))
#%
t_max = np.argmax(ndimage.gaussian_filter(center_trace,10))
average_image = np.mean(tiff_stack[t_max-50:t_max+50],0)
average_image = ndimage.gaussian_filter(average_image,2)
#%
average_image_ = average_image.copy()
average_image_[:100,:] = 0
average_image_[-100:,:] = 0
average_image_[:,:100] = 0
average_image_[:,-100:] = 0
for i in range(anticipated_spots):
    center_idx = np.unravel_index(np.argmax(average_image_),average_image.shape)
    center_trace = np.mean(tiff_stack[:,
                                      center_idx[0]-neigbourhood_pixel_num:center_idx[0]+neigbourhood_pixel_num,
                                      center_idx[1]-neigbourhood_pixel_num:center_idx[1]+neigbourhood_pixel_num],
                           axis = (1,2))
    center_indices.append(center_idx)
    center_traces.append(center_trace)
    average_image_[center_idx[0]-neigbourhood_pixel_num*4:center_idx[0]+neigbourhood_pixel_num*4,
                   center_idx[1]-neigbourhood_pixel_num*4:center_idx[1]+neigbourhood_pixel_num*4] = 0
#%
mean_trace = ndimage.gaussian_filter(np.mean(center_traces,0),1)
idx,peak_dict = find_peaks((mean_trace-np.min(mean_trace))/(np.max(mean_trace)-np.min(mean_trace)),prominence = .10)
order = np.argsort(peak_dict['prominences'])[::-1]
if Z_rounds == 1:
    peak_idx = idx[order[0]]
    end_idx = idx[order[1]]
    while mean_trace[end_idx-1]<mean_trace[end_idx]:
        end_idx -= 1
    height = np.min([end_idx-peak_idx,peak_idx])
else:
    height = int(np.mean(np.diff(idx[:-1])[:Z_rounds-1]))
    peak_idx = idx[1]
    
z = zrange*(np.arange(height*2)-height)/height/2
z_start_idx =  peak_idx-height
z_end_idx = peak_idx+height
useful_indices = []
z_proj = []
stack_list =  []
good_center_indices = []
for i,(t,c) in enumerate(zip(center_traces,center_indices)):
    
    offset = int(np.argmax(t[z_start_idx:z_end_idx])-(z_end_idx-z_start_idx)/2)
    
    t = t[z_start_idx+offset:z_end_idx+offset]
    
    t_filt = ndimage.gaussian_filter(t,5)
    
    t = (t-np.min(t_filt))/(np.max(t_filt)-np.min(t_filt))
    t_filt = (t_filt-np.min(t_filt))/(np.max(t_filt)-np.min(t_filt))
    
    idx,peak_dict = find_peaks(t_filt,height = .50)
    if len(idx)<3:
        z_proj.append(t)
        
        useful_indices.append(i)
        center_idx = center_indices[i]
        
        big_local_stack = np.asarray(tiff_stack[z_start_idx+offset:z_end_idx+offset,
                                 center_idx[0]-4*neigbourhood_pixel_num:center_idx[0]+4*neigbourhood_pixel_num,
                                 center_idx[1]-4*neigbourhood_pixel_num:center_idx[1]+4*neigbourhood_pixel_num],float)
        big_local_stack[:,
                        neigbourhood_pixel_num:-neigbourhood_pixel_num,
                        neigbourhood_pixel_num:-neigbourhood_pixel_num] = np.nan
        bg = []
        for plane in big_local_stack: bg.append(np.nanmean(plane))
        
        local_stack = tiff_stack[z_start_idx+offset:z_end_idx+offset,
                                 center_idx[0]-neigbourhood_pixel_num:center_idx[0]+neigbourhood_pixel_num,
                                 center_idx[1]-neigbourhood_pixel_num:center_idx[1]+neigbourhood_pixel_num]
        local_stack = local_stack-np.nanmean(bg)#np.asarray(bg)[:,np.newaxis,np.newaxis]
        
        stack_list.append(local_stack)
        good_center_indices.append(center_idx)
plotnum_per_cell = 3

fig_mean = plt.figure()
#useful_indices = useful_indices[:10]
fig = plt.figure()
col_num = round(np.sqrt(len(useful_indices)*plotnum_per_cell))+1
row_num = round(np.sqrt(len(useful_indices)*plotnum_per_cell))
ax_meanimage = fig_mean.add_subplot(3,2,1)
ax_z_profiles = fig_mean.add_subplot(3,2,2)
ax_dist_vs_hw = fig_mean.add_subplot(3,2,3)
ax_int_vs_hw = fig_mean.add_subplot(3,2,4)
ax_int_vs_dist = fig_mean.add_subplot(3,2,5)
im = ax_meanimage.imshow(average_image)
im.set_clim(np.percentile(average_image.flatten(),[1,99]))
slm_center_idx = center_indices[0]#
ax_meanimage.plot(slm_center_idx[1],slm_center_idx[0],'ro')
#%
hw_list = []
dist_list = []
mean_intensity_list = []
for i,(idx,z_projection,local_stack) in enumerate(zip(useful_indices,z_proj,stack_list)):
    ax_xy = fig.add_subplot(col_num,row_num,i*plotnum_per_cell+1)
    ax_xy.imshow(np.max(local_stack[np.argmax(z>z_range_to_show[0]):np.argmax(z>z_range_to_show[1]),:,:],0))
    ax_xy.set_title('spot {} xy'.format(i))
    ax_xy.axis('off')
    ax_xz = fig.add_subplot(col_num,row_num,i*plotnum_per_cell+2)
    ax_xz.imshow(np.max(local_stack[np.argmax(z>z_range_to_show[0]):np.argmax(z>z_range_to_show[1]),:,:],2),aspect = 'auto')
    ax_xz.set_title('spot {} xz'.format(i))
    ax_xz.axis('off')
    ax_yz = fig.add_subplot(col_num,row_num,i*plotnum_per_cell+3)
    ax_yz.imshow(np.max(local_stack[np.argmax(z>z_range_to_show[0]):np.argmax(z>z_range_to_show[1]),:,:],1),aspect = 'auto')
    ax_yz.set_title('spot {} yz'.format(i))
    ax_yz.axis('off')
    ax_meanimage.text(center_indices[idx][1],center_indices[idx][0],i)
    z_projection_ =  ndimage.gaussian_filter(z_projection,1)
    z_projection_ = z_projection_[int(height/4):-int(height/4)]
    hw_indices = np.asarray([np.argmax(z_projection_>.5),len(z_projection_)-np.argmax(z_projection_[::-1]>.5)-1])+int(height/4)
    #z_projection = z_projection_
    hw = np.diff(z[hw_indices])
    hw_list.append(hw)
    dist = np.sqrt((center_indices[idx][0]-slm_center_idx[0])**2+(center_indices[idx][1]-slm_center_idx[1])**2)
    dist_list.append(dist)
    ax_z_profiles.plot(z,z_projection)
    ax_z_profiles.plot(z[hw_indices],z_projection[hw_indices],'ko')
    mean_intensity_list.append(np.mean(local_stack[hw_indices[0]:hw_indices[1],:,:]))
dist_list = np.asarray(dist_list)*microns_per_pixel
ax_dist_vs_hw .plot(dist_list,hw_list,'ko') 
ax_dist_vs_hw.set_xlabel('distance from slm center')
ax_dist_vs_hw.set_ylabel('half-width (microns)')
ax_z_profiles.set_xlabel('z position (microns)')
ax_z_profiles.set_ylabel('mean intensity')

ax_int_vs_hw.plot(mean_intensity_list,hw_list,'ko')
ax_int_vs_hw.set_xlabel('spot intensity')
ax_int_vs_hw.set_ylabel('hw')

ax_int_vs_dist.plot(dist_list,mean_intensity_list,'ko')
ax_int_vs_dist.set_xlabel('distance from slm center')
ax_int_vs_dist.set_ylabel('intensity')
#%%
plt.plot(np.asarray(center_indices)[useful_indices,0],mean_intensity_list,'ko')
plt.plot(np.asarray(center_indices)[useful_indices,1],mean_intensity_list,'ro')








#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ---- First scripts
#%% 
from ScanImageTiffReader import ScanImageTiffReader
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
x = [121.045, 118.341, 116.048, 116.048, 118.811, 118.458, 121.045,
       121.045, 121.045, 118.517, 116.224, 113.932, 113.402, 111.051,
       111.168, 111.168, 118.399, 121.221, 113.461, 113.696, 116.165,
       113.638, 111.051, 115.93 , 116.283, 111.11 , 120.927, 118.399,
       113.638, 123.632, 123.573, 123.338, 123.338, 121.162, 123.514,
       123.514]
y = [156.024, 155.906, 155.789, 153.437, 153.378, 150.909, 150.968,
       153.437, 158.317, 158.434, 158.317, 158.434, 155.847, 155.906,
       158.375, 160.844, 160.962, 160.844, 160.786, 151.027, 150.85 ,
       153.437, 153.261, 160.903, 163.314, 163.196, 163.314, 163.314,
       163.372, 158.434, 155.789, 153.496, 150.792, 148.558, 163.137,
       160.903]
x_dist = []
y_dist = []
for x_i in x:
    for x_ii in x:
       x_dist.append(np.abs(x_ii-x_i)) 
for y_i in y:
    for y_ii in y:
       y_dist.append(np.abs(y_ii-y_i)) 
       
fig = plt.figure()
ax_x = fig.add_subplot(2,1,1)
counts,distances = np.histogram(np.concatenate([x_dist,y_dist]),np.arange(0,100,.2))
ax_x.hist(np.concatenate([x_dist,y_dist]),np.arange(0,100,.2))
dists = distances[:-1]+np.median(np.diff(distances))/2
ax_x.plot(dists,counts,'r-')
peaks = find_peaks(counts)[0]
ax_x.plot(dists[peaks],counts[peaks],'ro')
distance_per_ablation = np.median(np.diff(peaks[:10]))*.2
expected_distance = 231.42/64
FOV_factor = distance_per_ablation/expected_distance

print('the FOV of the SLM is {} compared to the galvo-galvo'.format(FOV_factor))

#%% look at the PSF of the SLM 
from ScanImageTiffReader import ScanImageTiffReader
import BCI_analysis
import numpy as np
import matplotlib.pyplot as plt
#stimfile = '/home/rozmar/Network/Bergamo2P/Imaging/Marton/photostim_slm_grid/grid_range_20_10steps_00001.stim'


file = '/mnt/Data/Calcium_imaging/raw/KayvonScope/tests/photostim_slm_grid_4_thick_sample/galvo_grid_00001.tif'
file = '/mnt/Data/Calcium_imaging/raw/KayvonScope/tests/photostim_slm_grid_5_thick_sample_compensation/galvo_grid_00001.tif'
file = '/mnt/Data/Calcium_imaging/raw/KayvonScope/tests/photostim_slm_grid_6_thick_sample_compensation/galvo_grid_00001.tif'
metadata  = BCI_analysis.io_scanimage.extract_scanimage_metadata(file)
reader=ScanImageTiffReader(file)
movie=reader.data()
trace_galvo = np.mean(np.mean(movie[:,:800,:],2),1)
baseline_galvo = np.mean(np.mean(movie[:,800:,:],2),1)
trace_galvo = trace_galvo-np.median(baseline_galvo)

file = '/mnt/Data/Calcium_imaging/raw/KayvonScope/tests/photostim_slm_grid_4_thick_sample/slm_grid_00001.tif' #file = '/mnt/Data/Calcium_imaging/raw/KayvonScope/tests/photostim_slm_grid_3_thick_sample/range_20_00001.tif'
file = '/mnt/Data/Calcium_imaging/raw/KayvonScope/tests/photostim_slm_grid_5_thick_sample_compensation/slm_grid_00001.tif'
file = '/mnt/Data/Calcium_imaging/raw/KayvonScope/tests/photostim_slm_grid_6_thick_sample_compensation/slm_grid_00001.tif'
metadata  = BCI_analysis.io_scanimage.extract_scanimage_metadata(file)
reader=ScanImageTiffReader(file)
movie=reader.data()
trace_slm = np.mean(np.mean(movie[:,:800,:],2),1)
baseline_slm = np.mean(np.mean(movie[:,800:,:],2),1)
trace_slm = trace_slm-np.median(baseline_slm)
#%
file = '/mnt/Data/Calcium_imaging/raw/KayvonScope/tests/photostim_slm_grid_5_thick_sample_compensation/slm_grid_compensated_00001.tif'
file = '/mnt/Data/Calcium_imaging/raw/KayvonScope/tests/photostim_slm_grid_6_thick_sample_compensation/slm_grid_compensated_00002.tif'
#file = '/mnt/Data/Calcium_imaging/raw/KayvonScope/tests/photostim_slm_grid_6_thick_sample_compensation/slm_grid_compensated_nonsqrt_00003.tif'

metadata  = BCI_analysis.io_scanimage.extract_scanimage_metadata(file)
reader=ScanImageTiffReader(file)
movie=reader.data()
trace_slm_compensated = np.mean(np.mean(movie[:,:800,:],2),1)
baseline_slm_compensated = np.mean(np.mean(movie[:,800:,:],2),1)
trace_slm_compensated = trace_slm_compensated-np.median(baseline_slm_compensated)
#%

start_list = np.arange(0,901,100)
end_list = np.arange(100,1001,100)

file = '/mnt/Data/Calcium_imaging/raw/KayvonScope/tests/photostim_slm_grid_4_thick_sample/resonant_mean_00001.tif'
file = '/mnt/Data/Calcium_imaging/raw/KayvonScope/tests/photostim_slm_grid_5_thick_sample_compensation/resonant_mean_00001.tif'
file = '/mnt/Data/Calcium_imaging/raw/KayvonScope/tests/photostim_slm_grid_6_thick_sample_compensation/resonant_mean_00001.tif'
reader=ScanImageTiffReader(file)
res_image=reader.data()

fovdeg = list()
for s in metadata['metadata']['hRoiManager']['imagingFovDeg'].strip('[]').split(' '): fovdeg.extend(s.split(';'))
fovdeg = np.asarray(fovdeg,float)
fovdeg = [np.min(fovdeg),np.max(fovdeg)]
# =============================================================================
# #%%
# file = '/mnt/Data/Calcium_imaging/raw/KayvonScope/tests/photostim_slm_grid_5_thick_sample_compensation/resonant_mean_post_00001.tif'
# reader=ScanImageTiffReader(file)
# res_image_post=reader.data()
# fig = plt.figure()
# ax_1 = fig.add_subplot(2,1,1)
# ax_1.imshow(res_image)
# ax_2 = fig.add_subplot(2,1,2)
# ax_2.imshow(res_image_post)
# =============================================================================
#%% xy intensities
import matplotlib
from  scipy.ndimage import gaussian_filter 
matplotlib.rcParams.update({'font.size': 6})
fig = plt.figure(figsize = [15,9])
ax_meanimg = fig.add_subplot(3,4,1)
ax_meanimg.set_title('image on imaging path')
ax_meanimg.imshow(res_image)
traces_small_galvo = []
traces_small_slm = []
traces_small_slm_comp = []
for s,e in zip(start_list,end_list):
    traces_small_galvo.append(trace_galvo[int(s):int(e)])
    traces_small_slm.append(trace_slm[int(s):int(e)])
    traces_small_slm_comp.append(trace_slm_compensated[int(s):int(e)])
    #plt.plot(traces_small[-1])
#ax_profile.plot(np.mean(traces_small,0),'k',linewidth = 4)
ax_reconstruction_galvo = fig.add_subplot(3,4,2)
ax_reconstruction_galvo.set_title('galvo image')
ax_reconstruction_slm = fig.add_subplot(3,4,3)
ax_reconstruction_slm.set_title('SLM image')
ax_reconstruction_slm_comp = fig.add_subplot(3,4,4)
ax_reconstruction_slm_comp.set_title('SLM image with power compensation')
ax_distance_dependence_resonant = fig.add_subplot(3,4,5)
ax_distance_dependence_resonant.set_xlabel('distance from SLM center (microns)')
ax_distance_dependence_resonant.set_ylabel('normalized intensity')

ax_distance_dependence_galvo = fig.add_subplot(3,4,6)
ax_distance_dependence_galvo.set_xlabel('distance from SLM center (microns)')
ax_distance_dependence_galvo.set_ylabel('normalized intensity')

ax_distance_dependence_slm = fig.add_subplot(3,4,7)
ax_distance_dependence_slm.set_xlabel('distance from SLM center (microns)')
ax_distance_dependence_slm.set_ylabel('normalized intensity')

ax_distance_dependence_slm_corr = fig.add_subplot(3,4,8)
ax_distance_dependence_slm_corr.set_xlabel('distance from SLM center (microns)')
ax_distance_dependence_slm_corr.set_ylabel('normalized intensity')

ax_correction_deg= fig.add_subplot(3,4,12)

#ax_distance_dependence_resonant = ax_distance_dependence.twinx()
ax_correction_galvo = fig.add_subplot(3,4,10)
ax_correction_slm = fig.add_subplot(3,4,11)
mean_vals_galvo = np.mean(np.asarray(traces_small_galvo),0).flatten()
std_vals_galvo = np.std(np.asarray(traces_small_galvo),0).flatten()
mean_vals_slm = np.mean(np.asarray(traces_small_slm),0).flatten()
std_vals_slm = np.std(np.asarray(traces_small_slm),0).flatten()
mean_vals_slm_comp = np.mean(np.asarray(traces_small_slm_comp),0).flatten()
std_vals_slm_comp = np.std(np.asarray(traces_small_slm_comp),0).flatten()
mean_img_galvo = np.zeros([10,10])
mean_img_slm = np.zeros([10,10])
mean_img_slm_comp = np.zeros([10,10])
idx = -1
#%
range_ = 20
step_num = 10
degree_steps = (np.arange(step_num)+1)/step_num*range_ - range_/2 #10*(np.arange(11)-5)/10
step_size =np.mean(np.diff(degree_steps))
FOV_fraction_positions = (degree_steps-fovdeg[0])/np.diff(fovdeg)[0]
zoomfactor = float(metadata['metadata']['hRoiManager']['scanZoomFactor'])
FOV_size = 1/(zoomfactor*6.87150356e-04+ 2.15504336e-06) # in microns
microns_per_fovdeg = FOV_size/np.diff(fovdeg)[0]#(np.max(degree_steps)-np.min(degree_steps))#
#%
center = [4,4]
distances = []
#%
res_x_distances = res_x_distances = np.asarray(np.abs(np.arange(res_image.shape[0])-res_image.shape[0]/2).T.tolist()*res_image.shape[1]).reshape(res_image.shape)
res_y_distances = np.asarray(np.abs(np.arange(res_image.shape[0])-res_image.shape[0]/2).T.tolist()*res_image.shape[1]).reshape(res_image.shape).T
res_distances = np.sqrt(res_x_distances**2+res_y_distances**2)*FOV_size/res_image.shape[0]
ax_distance_dependence_resonant.plot(res_distances.flatten()[::100],gaussian_filter(res_image,[5,5]).flatten()[::100]/np.max(gaussian_filter(res_image,[5,5]).flatten()),'ko',alpha = .01)
ax_distance_dependence_resonant.set_ylabel('drop in intensity on resonant path')
#%
for x in range(10):
    for y in range(10):
        idx+=1
        if [x,y] == center:
            center_idx = idx
        mean_img_galvo[x,y] = mean_vals_galvo[idx]
        mean_img_slm[x,y] = mean_vals_slm[idx]
        mean_img_slm_comp[x,y] = mean_vals_slm_comp[idx]
        distances.append(step_size*np.sqrt((x-center[0])**2+(y-center[1])**2))
        if FOV_fraction_positions[x] == FOV_fraction_positions[y] ==.5:
            marker = 'ro'
        else:
            marker = 'r.'
        ax_meanimg.plot(FOV_fraction_positions[x]*res_image.shape[0],FOV_fraction_positions[y]*res_image.shape[1],marker)

distances = np.asarray(distances)        
ax_distance_dependence_slm.plot(distances*microns_per_fovdeg,mean_vals_slm/np.max(mean_vals_slm),'ko',alpha = .5)
ax_distance_dependence_galvo.plot(distances*microns_per_fovdeg,mean_vals_galvo/np.max(mean_vals_galvo),'ko',alpha = .5)
from scipy.optimize import curve_fit
def Gauss(x, A, B,C):
    #A=1
    y = A*np.exp(-1*B*x**2) +C
    return y
def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b
def hypSec(x,a,b,c):
    c=0
    return a/np.cosh(x/b)+c
parameters_gauss_galvo, covariance = curve_fit(Gauss, np.concatenate([distances,distances*-1])*microns_per_fovdeg, np.concatenate([mean_vals_galvo,mean_vals_galvo])/mean_vals_galvo[center_idx],p0 =[1,.00005,0])
dist_vals = np.arange(0,800,10)
ax_distance_dependence_galvo.plot(dist_vals,Gauss(dist_vals,parameters_gauss_galvo[0],parameters_gauss_galvo[1],parameters_gauss_galvo[2]),'b--',label = 'Gaussian',linewidth = 4)

mean_img_galvo = mean_img_galvo/np.max(mean_img_galvo)  
ax_reconstruction_galvo.imshow(mean_img_galvo)
mean_img_slm = mean_img_slm/np.max(mean_img_slm)  
ax_reconstruction_slm.imshow(mean_img_slm)
mean_img_slm_comp = mean_img_slm_comp/np.max(mean_img_slm_comp[center])  
ax_reconstruction_slm_comp.imshow(mean_img_slm_comp)


ax_correction_galvo.plot(dist_vals,np.sqrt(1/Gauss(dist_vals,parameters_gauss_galvo[0],parameters_gauss_galvo[1],parameters_gauss_galvo[2])),'b--',linewidth = 4)

correction_vals = 1/Gauss(distances*microns_per_fovdeg,parameters_gauss_galvo[0],parameters_gauss_galvo[1],parameters_gauss_galvo[2])
mean_vals_slm_corrected = mean_vals_slm*correction_vals
ax_distance_dependence_slm.plot(distances*microns_per_fovdeg,mean_vals_slm_corrected/np.max(mean_vals_slm_corrected),'kx',alpha = .5)




needed_for_fit = distances*microns_per_fovdeg<800
parameters_gauss_slm, covariance = curve_fit(Gauss, np.concatenate([distances[needed_for_fit],distances[needed_for_fit]*-1])*microns_per_fovdeg, np.concatenate([mean_vals_slm[needed_for_fit],mean_vals_slm[needed_for_fit]])/np.max(mean_vals_slm[center_idx]),p0 =[1,.00005,0])
ax_distance_dependence_slm.plot(dist_vals,Gauss(dist_vals,parameters_gauss_slm[0],parameters_gauss_slm[1],parameters_gauss_slm[2]),'b-',label = 'Gaussian-uncorrected',linewidth = 1)

parameters_gauss_slm, covariance = curve_fit(Gauss, np.concatenate([distances[needed_for_fit],distances[needed_for_fit]*-1])*microns_per_fovdeg, np.concatenate([mean_vals_slm_corrected[needed_for_fit],mean_vals_slm_corrected[needed_for_fit]])/np.max(mean_vals_slm_corrected[center_idx]),p0 =[1,.00005,0])
ax_distance_dependence_slm.plot(dist_vals,Gauss(dist_vals,parameters_gauss_slm[0],parameters_gauss_slm[1],parameters_gauss_slm[2]),'b--',label = 'Gaussian',linewidth = 4)



parameters, covariance = curve_fit(hypSec,distances[needed_for_fit]*microns_per_fovdeg,mean_vals_slm[needed_for_fit]/np.max(mean_vals_slm[center_idx]),p0 = [1,100,0])
ax_distance_dependence_slm.plot(dist_vals,hypSec(dist_vals,parameters[0],parameters[1],parameters[2]),'r-', label = 'hyperbolic secant - uncorrected',linewidth = 1)

parameters, covariance = curve_fit(hypSec,distances[needed_for_fit]*microns_per_fovdeg,mean_vals_slm_corrected[needed_for_fit]/np.max(mean_vals_slm_corrected[center_idx]),p0 = [1,100,0])
ax_distance_dependence_slm.plot(dist_vals,hypSec(dist_vals,parameters[0],parameters[1],parameters[2]),'r--', label = 'hyperbolic secant',linewidth = 4)

ax_distance_dependence_slm.legend()


ax_distance_dependence_slm_corr.plot(distances*microns_per_fovdeg,mean_vals_slm_comp/np.max(mean_vals_slm_comp[center_idx]),'ko',alpha = .5)

ax_correction_slm.plot(dist_vals,np.sqrt(1/Gauss(dist_vals,parameters_gauss_slm[0],parameters_gauss_slm[1],parameters_gauss_slm[2])),'b--',linewidth = 2)

ax_correction_slm.plot(dist_vals,np.sqrt(1/hypSec(dist_vals,parameters[0],parameters[1],parameters[2])),'r-',label = 'sqrt((cosh(x/{1:.3g})+{2:.3g})/{0:.3g})'.format(*parameters))

slm_correction_values = 1/hypSec(distances*microns_per_fovdeg,parameters[0],parameters[1],parameters[2])
ax_distance_dependence_slm_corr.plot(distances*microns_per_fovdeg,mean_vals_slm_corrected/np.max(mean_vals_slm_corrected[center_idx])*slm_correction_values,'rx',alpha = .5)

#ax_correction.set_yscale('log')
ax_correction_galvo.set_xlabel('distance from SLM center (microns)')
ax_correction_galvo.set_ylabel('-fold power correction needed')
parameters_deg, covariance = curve_fit(hypSec,distances[needed_for_fit],mean_vals_slm_corrected[needed_for_fit]/np.max(mean_vals_slm_corrected[center_idx]),p0 = [1,2,0])

dist_vals_deg = np.arange(0,11,.1)
parameters_deg[2] = 0
ax_correction_deg.plot(dist_vals_deg,
                       np.sqrt(1/hypSec(dist_vals_deg,parameters_deg[0],parameters_deg[1],parameters_deg[2])),
                       'r-',
                       label = 'sqrt((cosh(x/{1:.5g})+{2:.5g})/{0:.5g})'.format(*parameters_deg) )

#ax_correction_deg.set_yscale('log')
ax_correction_deg.set_xlabel('distance from SLM center (degree)')
ax_correction_deg.set_ylabel('-fold power correction needed')
ax_correction_deg.legend()
#%% fit fov from magnification data
mag = [5,
3,
2,
1.5,
1.5,
1,]
fov = [291.5438778,
481.8739543,
724.8122578,
963.8076351,
963.0055284,
1487.755102]
plt.plot(mag,1/np.asarray(fov),'ko')
mag = 1
fov = 1/(mag*6.87150356e-04+ 2.15504336e-06)


#%% Z-stack
#%% look at the PSF of the SLM 
from ScanImageTiffReader import ScanImageTiffReader
import BCI_analysis
import numpy as np
import matplotlib.pyplot as plt
import os
#stimfile = '/home/rozmar/Network/Bergamo2P/Imaging/Marton/photostim_slm_grid/grid_range_20_10steps_00001.stim'
base_dir = '/mnt/Data/Calcium_imaging/raw/KayvonScope/tests/photostim_slm_grid_Z/'

res_file = os.path.join(base_dir,'res_zstack_00001.tif')
reader=ScanImageTiffReader(res_file)
#%
res_stack=reader.data().copy()
#%
#reader.close()
#%

metadata  = BCI_analysis.io_scanimage.extract_scanimage_metadata(res_file)

fovdeg = list()
for s in metadata['metadata']['hRoiManager']['imagingFovDeg'].strip('[]').split(' '): fovdeg.extend(s.split(';'))
fovdeg = np.asarray(fovdeg,float)
fovdeg = [np.min(fovdeg),np.max(fovdeg)]


start_list = np.arange(0,901,100)
end_list = np.arange(100,1001,100)
slm_planes = []
for z in range(50,155,10):
    slm_planes.append('slm_{}_00001.tif'.format(str(z).zfill(3)))

slm_traces = []
slm_traces_small = []
slm_mean_images = []
for slm_plane in slm_planes:
    print(slm_plane)
    file = os.path.join(base_dir,slm_plane)
    reader=ScanImageTiffReader(file)
    movie=reader.data()
    trace_slm = np.mean(np.mean(movie[:,:800,:],2),1)
    baseline_slm = np.mean(np.mean(movie[:,800:,:],2),1)
    trace_slm = trace_slm-np.median(baseline_slm)
    slm_traces.append(trace_slm)
    #%
    traces_small_slm = []
    for s,e in zip(start_list,end_list):
        traces_small_slm.append(trace_slm[int(s):int(e)])
        mean_vals_slm = np.mean(np.asarray(traces_small_slm),0).flatten()
        
        
        
        mean_img_slm = np.zeros([10,10])
        mean_img_slm_comp = np.zeros([10,10])
        idx = -1
        #%
        range_ = 20
        step_num = 10
        degree_steps = (np.arange(step_num)+1)/step_num*range_ - range_/2 #10*(np.arange(11)-5)/10
        step_size =np.mean(np.diff(degree_steps))
        FOV_fraction_positions = (degree_steps-fovdeg[0])/np.diff(fovdeg)[0]
        zoomfactor = float(metadata['metadata']['hRoiManager']['scanZoomFactor'])
        FOV_size = 1/(zoomfactor*6.87150356e-04+ 2.15504336e-06) # in microns
        microns_per_fovdeg = FOV_size/np.diff(fovdeg)[0]#(np.max(degree_steps)-np.min(degree_steps))#
        #%%
        center = [4,4]
        distances = []
        #%
        
        for x in range(10):
            for y in range(10):
                idx+=1
                if [x,y] == center:
                    center_idx = idx
                mean_img_slm[x,y] = mean_vals_slm[idx]
                distances.append(step_size*np.sqrt((x-center[0])**2+(y-center[1])**2))
                if FOV_fraction_positions[x] == FOV_fraction_positions[y] ==.5:
                    marker = 'ro'
                else:
                    marker = 'r.'
    slm_mean_images.append(mean_img_slm)
        
        
    slm_traces_small.append(traces_small_slm)
#%%
slm_matrix = np.asarray(slm_mean_images)
slm_matrix_norm = slm_matrix/np.max(slm_matrix,0)
z_profiles = []
for distance in np.unique(distances):
    xs,ys = np.unravel_index(np.where(distance == distances)[0],np.shape(slm_matrix_norm[5,:,:].squeeze()))
    profiles_list = []
    for x,y in zip(xs,ys):
        profiles_list.append(slm_matrix_norm[:,x,y].squeeze())
    z_profiles.append(np.mean(profiles_list,0))
        
fig = plt.figure()
ax=fig.add_subplot(1,1,1)    
res_z = np.mean(np.mean(res_stack,1),1)
for distance,z_profile in zip(np.unique(distances),z_profiles):
    ax.plot(np.arange(-50,55,10),z_profile)

ax.plot(np.arange(-50,55,10),(res_z-np.min(res_z))/(np.max(res_z)-np.min(res_z)),'k-')  
ax.set_xlabel('Z position (microns)')
ax.set_ylabel('relative intensity')