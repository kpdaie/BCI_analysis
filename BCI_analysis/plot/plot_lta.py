# %%
import os

import matplotlib.colors as cl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append("/home/labadmin/Github/BCI_analysis/BCI_analysis")
from pipeline.pipeline_align import get_aligned_data
from tqdm import tqdm

from plot.plot_utils import rollingfun


def segment(arr, max=20):
    """
    This function takes an array and clusters them according to the max value. We use these to
    cluster licks from TongueTip positions
    """
    clusters = []
    eps = max
    points_sorted = np.asarray(arr)
    curr_point = points_sorted[0]
    curr_cluster = [curr_point]
    for point in points_sorted[1:]:
        if point <= curr_point + eps:
            curr_cluster.append(point)
        else:
            clusters.append(curr_cluster)
            curr_cluster = [point]
            curr_point = point
    clusters.append(curr_cluster)

    return clusters

def plot_lt_rt(lt, rt):
    """
    This function plots licks and reward times for all trials"""
    plt.figure(figsize=(16, 10))
    for i in range(len(lt)):
        plt.plot(lt[i], [i]*len(lt[i]), 'go', markersize=1.5)    
        plt.plot(rt[i], [i]*len(rt[i]), 'ro', markersize=1.5)
    plt.ylabel("Trial #")
    plt.xlabel("Time from trial start")
    plt.show()

def plot_lick_lickport(ttip, lport, trial):
    """
    This function plots tongue position, the segmented licks and the lickport steps
    """

    k = segment(ttip[trial].index.values, max=200)
    tongue_start_end = np.array([[k[i][0], k[i][-1]] for i in range(len(k))])

    plt.scatter(ttip[trial].index.values, ttip[trial]["x"],  marker='.', alpha=0.2, c='red')
    for i in range(tongue_start_end.shape[0]):
        plt.plot([tongue_start_end[i, 0], tongue_start_end[i, 0]], [ttip[trial]["x"][tongue_start_end[i, 0]], ttip[trial]["x"][tongue_start_end[i, 1]]], '-', color='black', alpha=1)
    plt.scatter(lport[trial]["x"].index.values, lport[trial]["x"],  marker='.', alpha=0.1, c='green')
    plt.show()

def find_nearest_above(my_array, target):
    """
    Utility function that finds nearest value that is greater than target
    """
    diff = my_array - target
    mask = np.ma.less_equal(diff, 0)
    # We need to mask the negative differences and zero
    # since we are looking for values above
    if np.all(mask):
        return None # returns None if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()

def get_bpod_reward_times_aligned(lport, rt):

    rtrel_dlc  = []
    frame_since_start = 0
    for trial in range(len(lport)):
        if len(rt[trial]) == 0:
            rtrel_dlc.append([])
            continue
        closest = frame_since_start + int(rt[trial][0])
        rtrel_dlc.append([closest])
        frame_since_start += lport[trial].shape[0]

    return rtrel_dlc

def dlc_approx_reward_time(lport, rt):
    """
    This function tries to estimate reward time using lickport position. Just for sanity checks
    """
    trials_ignore = []
    rtrel_dlc  = []
    frame_since_start = 0
    for trial in range(len(lport)):
        if lport[trial]["x"][500:].max() > 300 and len(rt[trial]) == 0:
            trials_ignore.append(trial)
            rtrel_dlc.append([])
            frame_since_start += lport[trial].shape[0]
            continue
        
        if len(rt[trial]) == 0:
            rtrel_dlc.append([])
            frame_since_start += lport[trial].shape[0]
            continue

        if lport[trial]["x"][1000:].max() < 300 and len(rt[trial]) != 0:
            rtrel_dlc.append([])
            print(trial, rt[trial])

        nearest = find_nearest_above(rollingfun(lport[trial]['x'][500:].values, 1000), 301)
        if nearest is None:
            rtrel_dlc.append([])

        else:
            closest = frame_since_start + 500 + nearest
            rtrel_dlc.append([closest])

        frame_since_start += lport[trial].shape[0]


        # print(closest, closest - rt[trial][0], lport[trial]['x'].iloc[closest], trial)

    return rtrel_dlc

def plot_trace(dff_mean, dff_sd, indices=None, dim=(2,4)):

    fig = plt.figure(figsize=tuple(i * 4 for i in dim)[::-1])
    for i, id in enumerate(indices):
        ax = plt.subplot(*dim, i + 1)
        ax.plot(dff_mean[id])
        ax.fill_between(np.arange(len(dff_mean[id])), dff_mean[id]-dff_sd[id], dff_mean[id]+dff_sd[id], alpha=0.2)
    plt.show()
    

def plot_population_lta(suite2p_path,
                        dlc_base_dir,
                        bpod_path,
                        sessionwise_data_path,
                        plt_save_path,
                        aligned_data_path,
                        mouse="BCI_26",
                        FOV="FOV_04",
                        camera="side",
                        session="041022",
                        plot=True):
    """
    This function plots and saves the plot to lick triggered population average. We take population
    activity nearby licks throughout the session and then average over the licks. We then plot a
    sorted heatmap for all neurons.

    Example:
    dlc_base_dir = os.path.abspath("../../bucket/Data/Behavior_videos/DLC_output/Bergamo-2P-Photostim/")
    bpod_path = os.path.abspath("../../bucket/Data/Behavior/BCI_exported/Bergamo-2P-Photostim/")
    suite2p_path = os.path.abspath("../../bucket/Data/Calcium_imaging/suite2p/Bergamo-2P-Photostim/")
    sessionwise_data_path = os.path.abspath("../../bucket/Data/Calcium_imaging/sessionwise_tba/")
    plt_save_path = os.path.abspath("../../Plots/")
    aligned_data_path = os.path.abspath("/home/labadmin/Github/BCI_analysis/data_aligned")

    mouse = "BCI_26"
    FOV = "FOV_04"
    camera = "side" 
    session = "041522"
    plot_population_lta(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, 
                        plt_save_path, mouse, FOV, camera, session)
    """

    dict_aligned = get_aligned_data(suite2p_path, dlc_base_dir, bpod_path, 
            sessionwise_data_path, aligned_data_path, mouse, 
            FOV, camera, session, plot=False, overwrite=False)

    if dict_aligned is None:
        return
    F_aligned = dict_aligned["F_aligned"]
    dff_aligned = dict_aligned["dff_aligned"]

    rt = dict_aligned["reward_times_aligned"]
    # F_aligned_s = np.hstack(F_aligned)
    dff_aligned_s = np.hstack(dff_aligned)
    DLC_aligned = pd.DataFrame.from_dict(dict_aligned["DLC_aligned"])
    trial_lengths = [F_aligned[i].shape[1] for i in range(len(F_aligned))]
    cn = dict_aligned["cn"]

    ttip = []
    lport = []
    c_lengths = [0] + list(np.cumsum(trial_lengths))

    for i in range(len(c_lengths)-2):
        k = DLC_aligned["TongueMid"][c_lengths[i]:c_lengths[i+1]]
        ttip.append(k[k["likelihood"] > 0.90])

        k = DLC_aligned["Lickport"][c_lengths[i]:c_lengths[i+1]]
        lport.append(k)


    ctr=0
    lick_starts = []
    for trial in range(len(ttip)):
        arr = ttip[trial].index.values
        if len(arr) == 0:
            continue
        k = segment(arr, max=400)
        tongue_start = np.array([k[i][0] for i in range(len(k))])
        tongue_end = np.array([k[i][-1] for i in range(len(k))])

        for i in range(tongue_start.shape[0]):
            tongue_start_end = (tongue_start[i] + tongue_end[i])//2
            movement = lport[trial]["x"].loc[tongue_start_end - 500: tongue_start_end + 500].values
            if np.mean(movement[:500]) > 300:
                continue

            lick_starts.append(tongue_start[i])
            ctr = ctr + 1

    # rtrel_dlc = dlc_approx_reward_time(lport, rt)
    rtrel_dlc = get_bpod_reward_times_aligned(lport, rt)
    

    tframes = 2000
    ctr = 0
    dff_avg = np.zeros((dff_aligned_s.shape[0], tframes, len(lick_starts)))
    print("licks, ", len(lick_starts))
    # for tl, ls in enumerate(lick_starts):
    # print(rtrel_dlc, dict_aligned['reward_times_aligned'])
    for tl, ls in enumerate(rtrel_dlc):
        if len(ls) == 0:
            continue
        else:
            ls = int(ls[0])
        print(ls)
        k = dff_aligned_s[:, ls-tframes//2:ls+tframes//2]
        if k.shape[1] != dff_avg.shape[1]:
            # print("Not found")
            continue
        dff_avg[:,:,tl] = k
        ctr += 1
    print(ctr)
    dff_avg = dff_avg - np.mean(dff_avg[:, :200, :], axis=1, keepdims=True)
    dff_sd = np.std(dff_avg, axis=-1)
    sem = dff_sd/np.sqrt(dff_avg.shape[2])
    dff_avg = np.mean(dff_avg, axis=-1)

    means = np.mean(dff_avg[:, tframes//2:tframes//2+500], axis=1) - np.mean(dff_avg[:, tframes//2-500:tframes//2], axis=1)
    # means = np.mean(dff_avg - dff_sd, axis=1)
    # means = np.mean(dff_avg[:, tframes//2:], axis=1) - np.mean(dff_avg[:, :tframes//2], axis=1)
    sorted_m = np.argsort(means)[::-1]
    cn_sorted = np.argwhere(sorted_m == cn)

    if plot == True:
        plt.figure(figsize=(16, 8))
        plt.subplot(121)
        plt.imshow(dff_avg[sorted_m[:]], aspect="auto", cmap='seismic', norm=cl.TwoSlopeNorm(0))
        plt.axvline(x=tframes//2, color='black')
        plt.yticks(cn_sorted[0], [f'{cn}: CN'])
        plt.title(f'{mouse}-{session}')
        plt.colorbar()

        plt.subplot(122)
        plt.plot(dff_avg[cn])
        plt.title(f"{cn}: Conditioned Neuron")
        plt.axvline(x=1000, ymin=0.25, ymax=0.75, color='black', linestyle='--')

        os.makedirs(os.path.join(plt_save_path, mouse), exist_ok=True)
        save_path = os.path.join(plt_save_path, mouse, f"lick_triggered_population-{session}")
        plt.tight_layout()
        # plt.savefig(save_path)
        plt.show()

    plot_trace(dff_avg, sem, indices=sorted_m[:8], dim=(4, 2))
    
    return dff_avg, sorted_m

def plot_sessionwise_change(suite2p_path,
                            dlc_base_dir,
                            bpod_path,
                            sessionwise_data_path,
                            plt_save_path,
                            aligned_data_path,
                            mouse="BCI_26",
                            FOV="FOV_04",
                            camera="side",
                            session_list=["041022", "041122"]):
    
    delta_change = []
    dffs = []
    sorts = []
    for session in session_list:
        dff_avg, sorted_m = plot_population_lta(suite2p_path, dlc_base_dir, 
                            bpod_path, sessionwise_data_path, plt_save_path, 
                            aligned_data_path, mouse, FOV, camera, session, plot=False)
        
        dffs.append(dff_avg)
        sorts.append(sorted_m)
        print(session)

    tframes = 2000
    means = [np.sum(k, axis=1) for k in dffs]

    # print(np.mean(delta_change[0]), np.mean(delta_change[1]))
    # plt.scatter(delta_change[0], delta_change[1], marker='.', alpha=0.4, c='black')
    # plt.scatter(np.mean(delta_change[0]), np.mean(delta_change[1]), c='red', marker='.')
    # plt.xlabel(f'{session_list[0]}')
    # plt.ylabel(f'{session_list[1]}')
    # plt.show()

    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(dffs[0][sorts[0]], aspect="auto", cmap='seismic', norm=cl.TwoSlopeNorm(vcenter=0, vmin=-0.5, vmax=0.5))
    plt.axvline(x=tframes//2, color='black')
    # plt.yticks(cn_sorted[0], [f'{cn}: CN'])
    plt.title(f'{mouse}-{session_list[0]}')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(dffs[1][sorts[0]], aspect="auto", cmap='seismic', norm=cl.TwoSlopeNorm(vcenter=0, vmin=-0.5, vmax=0.5))
    plt.axvline(x=tframes//2, color='black')
    # plt.yticks(cn_sorted[0], [f'{cn}: CN'])
    plt.title(f'{mouse}-{session_list[1]}')
    plt.colorbar()

    os.makedirs(os.path.join(plt_save_path, mouse), exist_ok=True)
    save_path = os.path.join(plt_save_path, mouse, f"{session_list[0]}-{session_list[1]}-heatmap")
    plt.tight_layout()
    plt.savefig(save_path)


# %%

dlc_base_dir = os.path.abspath("../../bucket/Data/Behavior_videos/DLC_output/Bergamo-2P-Photostim/")
bpod_path = os.path.abspath("../../bucket/Data/Behavior/BCI_exported/Bergamo-2P-Photostim/")
suite2p_path = os.path.abspath("../../bucket/Data/Calcium_imaging/suite2p/Bergamo-2P-Photostim/")
sessionwise_data_path = os.path.abspath("../../bucket/Data/Calcium_imaging/sessionwise_tba/")
plt_save_path = os.path.abspath("../../Plots/")
aligned_data_path = os.path.abspath("/home/labadmin/Github/BCI_analysis/data_aligned")


mouse = "BCI_26"
FOV = "FOV_04"
camera = "side" 
# session_list = ["041322", "041322_2"]

# # for session in session_list:
# #     plot_population_lta(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, 
# #                         plt_save_path, aligned_data_path, mouse, FOV, camera, session)

# plot_sessionwise_change(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, plt_save_path, aligned_data_path, mouse, FOV, camera, session_list)

# session_list = ["041422", "041522"]
# plot_sessionwise_change(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, plt_save_path, aligned_data_path, mouse, FOV, camera, session_list)

# session_list = ["041122", "041222"]
# plot_sessionwise_change(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, plt_save_path, aligned_data_path, mouse, FOV, camera, session_list)

# session_list = ["041922", "042022"]
# plot_sessionwise_change(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, plt_save_path, aligned_data_path, mouse, FOV, camera, session_list)

# session_list = ["042022", "042122"]
# plot_sessionwise_change(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, plt_save_path, aligned_data_path, mouse, FOV, camera, session_list)

# session_list = ["042722", "042822"]
# plot_sessionwise_change(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, plt_save_path, aligned_data_path, mouse, FOV, camera, session_list)
# session_list = ["041922", "042022", "042122", "042222", "042722", "042822", "042922"]
# plot_sessionwise_change(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, plt_save_path, aligned_data_path, mouse, FOV, camera, session_list)
# for session in session_list:
#     plot_population_lta(suite2p_path, dlc_base_dir, 
#                         bpod_path, sessionwise_data_path, plt_save_path, 
#                         aligned_data_path, mouse, FOV, camera, session, plot=False)

session = "041022"
plot_population_lta(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, 
                    plt_save_path, aligned_data_path, mouse, FOV, camera, session, plot=True)
