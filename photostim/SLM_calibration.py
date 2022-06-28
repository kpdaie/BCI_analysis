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