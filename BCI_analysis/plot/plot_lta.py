# %%
import os
from BCI_analysis.pipeline.pipeline_align import get_aligned_data

import seaborn as sns
import matplotlib.colors as cl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from BCI_analysis.plot.plot_utils import rollingfun

cmap_cl = 'bwr'
norm_cl = cl.TwoSlopeNorm(vmin=-5.0, vcenter=0, vmax=5.0)

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

def get_paw_changes(DLC_paw, start=None, end=None, plot=False):

    '''
       r: Discounting rate
       order: AR model order
       smooth: smoothing window size T
    '''
    if start == None:
        start = 0
    if end == None:
        end = len(DLC_paw)
    # cf = changefinder.ChangeFinder(r=r, order=order, smooth=smooth)
    # ts_score = [cf.update(p) for p in DLC_paw[start:end].values]
    # ts_change_loc1 = pd.Series(ts_score).nlargest(nlargest)
    # ts_change_loc1 = ts_change_loc1.index.values
    # if plot:
    #     plt.figure(figsize=(16,4))
    #     plt.plot(DLC_paw[start:end])
    #     plt.figure(figsize=(16,4))
    #     [plt.axvline(ts_change_loc1[i], linewidth=1, color='g') for i in range(len(ts_change_loc1))]
    #     plt.plot(ts_score, color='red')
    #     plt.show()
    p = 10
    s = DLC_paw[start:end]
    diffs = s.diff(periods=p)
    std = diffs.std()
    significant_changes = diffs.loc[diffs > std].index
    seg = segment(significant_changes, 200)
    significant_changes = [seg[i][0] for i in range(len(seg))]
    if plot:
        plt.plot(DLC_paw)
        [plt.axvline(significant_changes[i], linewidth=1, color='g') for i in range(len(significant_changes))]
        plt.show()
    return significant_changes

def get_bpod_reward_times_aligned(lport, rt):

    print(len(lport), len(rt))

    rtrel_dlc  = []
    frame_since_start = 0
    for trial in range(len(lport)):
        if len(rt[trial]) == 0:
            frame_since_start += lport[trial].shape[0]
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
    
    mlp = []
    for trial in range(len(rt)):
        if len(rt[trial]) == 0:
            continue
        mlp.append(np.median(lport[trial][-10:]))

    cutoff_val = int(np.median(mlp))
    print(cutoff_val)

    for trial in range(len(lport)):
        if lport[trial]["x"][500:].max() > cutoff_val and len(rt[trial]) == 0:
            trials_ignore.append(trial)
            rtrel_dlc.append([])
            frame_since_start += lport[trial].shape[0]
            continue
        
        if len(rt[trial]) == 0:
            rtrel_dlc.append([])
            frame_since_start += lport[trial].shape[0]
            continue

        if lport[trial]["x"][1000:].max() < cutoff_val and len(rt[trial]) != 0:
            rtrel_dlc.append([])
            frame_since_start += lport[trial].shape[0]
            print(trial, rt[trial])
            continue

        nearest = find_nearest_above(rollingfun(lport[trial]['x'][500:].values, 1000), cutoff_val+1)
        if nearest is None:
            rtrel_dlc.append([])

        else:
            closest = frame_since_start + 500 + nearest
            rtrel_dlc.append([closest])

        frame_since_start += lport[trial].shape[0]


        # print(closest, closest - rt[trial][0], lport[trial]['x'].iloc[closest], trial)

    return rtrel_dlc

def plot_trace(dff_mean, dff_sd, session=None, indices=None, dim=(2,4), ax=None):

    # fig = plt.figure(figsize=tuple(i * 4 for i in dim)[::-1])
    color = None
    if ax==None:
        fig = plt.figure(figsize=(20, 12))
        ax = plt.subplot(1, 1, 1)
    for _, id in enumerate(indices):
        # ax = plt.subplot(*dim, i + 1)
        ax.plot(dff_mean[id], color=color)
        ax.fill_between(np.arange(len(dff_mean[id])), dff_mean[id]-dff_sd[id], dff_mean[id]+dff_sd[id], alpha=0.2, color=color)
        ax.set_title(f'Lick Triggered Neurons: {session}')
        ax.set_ylabel(f"$\Delta$F/F")
        ax.set_xlabel('Frames')
    ax.axvline(x=1000, ymin=0.25, ymax=0.75, color='black', linestyle='--', label='Lick')
    ax.legend()
    save_path = os.path.join(plt_save_path, f'lta_example_traces-{session}')
    if ax==None:
        plt.savefig(save_path)
        plt.show()
    
def plot_daycompare(dffs, sems, sessions_list, indices, align_at='lick', dim=(2, 4)):
    # fig = plt.figure(figsize=tuple(i * 4 for i in dim)[::-1])
    # fig, ax = plt.figure(figsize=(10, 10))
    color = None
    for i, id in enumerate(indices):
        # ax = plt.subplot(*dim, i + 1)
        for sl in range(len(session_list)):
            plt.plot(dffs[sl][id], color=color, label=f'{session_list[sl]}')
            plt.fill_between(np.arange(len(dffs[sl][id])), dffs[sl][id]-sems[sl][id], dffs[sl][id]+sems[sl][id], alpha=0.2, color=color)

        # ax.set_xlim([-0.5, 3])
        plt.axvline(x=1000, ymin=0.25, ymax=0.75, color='black', linestyle='--', label='Lick')
        # ax.set_title(f'Lick Triggered Neurons: {session}')
        # ax.set_title(f'Neuron:{id}')
        # if i == (dim[0]-1)*dim[1]:
        #     ax.set_xlabel('Frames')
        #     sns.despine(ax=ax)
        #     ax.set_ylabel(f"$\Delta$F/F")
        # else:
        #     sns.despine(bottom=True, left=True, ax=ax)
        #     plt.xticks([])
        #     plt.yticks([])
        # if i == dim[1]-1:
        #     ax.legend()
    save_path = os.path.join(plt_save_path, f'{align_at}-ta_example_traces-{session_list}')
    # if ax==None:
    # plt.savefig(save_path)
    plt.show()


def plot_population_lta(suite2p_path,
                        dlc_base_dir,
                        bpod_path,
                        sessionwise_data_path,
                        plt_save_path,
                        aligned_data_path,
                        align_at="lick",
                        mouse="BCI_26",
                        FOV="FOV_04",
                        camera="side",
                        session="041022",
                        plot=True,
                        overwrite=False):
    """
    This function plots and saves the plot to lick triggered population average. We take population
    activity nearby licks throughout the session and then average over the licks. We then plot a
    sorted heatmap for all neurons.

    Example:
    dlc_base_dir = os.path.abspath("/home/labadmin/Github/BCI_analysis/bucket/Data/Behavior_videos/DLC_output/Bergamo-2P-Photostim/")
    bpod_path = os.path.abspath("/home/labadmin/Github/BCI_analysis/bucket/Data/Behavior/BCI_exported/Bergamo-2P-Photostim/")
    suite2p_path = os.path.abspath("/home/labadmin/Github/BCI_analysis/bucket/Data/Calcium_imaging/suite2p/Bergamo-2P-Photostim/")
    sessionwise_data_path = os.path.abspath("/home/labadmin/Github/BCI_analysis/bucket/Data/Calcium_imaging/sessionwise_tba/")
    plt_save_path = os.path.abspath("/home/labadmin/Github/BCI_analysis/Plots/")
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
            FOV, camera, session, plot=False, overwrite=overwrite)

    if dict_aligned is None:
        return
    F_aligned = dict_aligned["F_aligned"]
    dff_aligned = dict_aligned["dff_aligned"]

    rt = dict_aligned["reward_times_aligned"]
    # F_aligned_s = np.hstack(F_aligned)
    dff_aligned_s = np.hstack(dff_aligned)
    DLC_aligned = pd.DataFrame.from_dict(dict_aligned["DLC_aligned"])
    trial_lengths = [F_aligned[i].shape[1] for i in range(len(F_aligned))]
    print(trial_lengths)
    cn = dict_aligned["cn"]
    trials_taken = np.asarray(dict_aligned["trials_taken"])
    print(trials_taken)

    ttip = []
    lport = []
    c_lengths = [0] + list(np.cumsum(trial_lengths))
    changetimes = []

    for i in range(len(c_lengths)-1):
        k = DLC_aligned["TongueTip"][c_lengths[i]:c_lengths[i+1]]
        ttip.append(k[k["likelihood"] > 0.90])

        k = DLC_aligned["Lickport"][c_lengths[i]:c_lengths[i+1]]
        lport.append(k)

    # plot_lick_lickport(ttip, lport, 0)
    # plot_lick_lickport(ttip, lport, 6)
    # plot_lick_lickport(ttip, lport, 8)

    if align_at == "lick":
        ctr=0
        lick_starts = []
        for trial in range(len(ttip)):
            arr = ttip[trial].index.values
            if len(arr) == 0:
                continue
            k = segment(arr, max=200)
            tongue_start = np.array([k[i][0] for i in range(len(k))])
            tongue_end = np.array([k[i][-1] for i in range(len(k))])

            for i in range(tongue_start.shape[0]):
                tongue_start_end = (tongue_start[i] + tongue_end[i])//2
                movement = lport[trial]["x"].loc[tongue_start_end - 500: tongue_start_end + 500].values
                if np.mean(movement[:500]) > 300:
                    continue

                lick_starts.append((trial, tongue_start[i]))
                ctr = ctr + 1
        print("licks, ", len(lick_starts))
        changetimes = lick_starts       # lick_starts is a list of tuples (trial_num, lick_time)

    elif align_at == "reward":
        # rtrel_dlc = dlc_approx_reward_time(lport, rt)
        rtrel_dlc = get_bpod_reward_times_aligned(lport, rt)
        rtrel_dlc = [(ctr, int(k[0])) for ctr, k in enumerate(rtrel_dlc) if len(k) != 0]
        changetimes = rtrel_dlc     # rtrel_dlc is a list of tuples (trial_num, reward_time)

    elif align_at == "PawL" or "PawR" or "EyeDown" or "EyeRight":
        global norm_cl 
        norm_cl = cl.TwoSlopeNorm(vmin=-1, vmax=1, vcenter=0)
        changetimes = []
        for i in range(len(c_lengths)-1):
            DLC_paw = DLC_aligned[align_at]["x"][c_lengths[i]:c_lengths[i+1]]
            ts_change_loc1 = get_paw_changes(DLC_paw)
            [changetimes.append((i, j)) for j in ts_change_loc1]

    tframes = 2000
    ctr = 0
    dff_lw = np.zeros((dff_aligned_s.shape[0], tframes, len(changetimes)))
    # for tl, ls in enumerate(lick_starts):
    # print(rtrel_dlc, dict_aligned['reward_times_aligned'])
    for i, (tl, ls) in enumerate(changetimes):
        print(tl)
        if tl not in trials_taken:
            print("Trial not taken ", tl)
            continue
        print(c_lengths[tl], ls, ls - c_lengths[tl])
        k = dff_aligned_s[:, ls-tframes//2:ls+tframes//2]
        if k.shape[1] != dff_lw.shape[1]:
            print("Not found")
            continue
        dff_lw[:,:,i] = k
        ctr += 1
    print(ctr)
    dff_sd = np.std(dff_lw, axis=-1)
    sem = dff_sd/np.sqrt(dff_lw.shape[2])
    dff_avg = np.mean(dff_lw, axis=-1)
    dff_avg = dff_avg - np.mean(dff_avg[:, :tframes//2], axis=1, keepdims=True)

    means = np.mean(dff_avg[:, tframes//2:tframes//2+500], axis=1) - np.mean(dff_avg[:, tframes//2-500:tframes//2], axis=1)
    # plt.hist(means, bins=100)
    # plt.show()
    sorted_m = np.argsort(means)[::-1]
    cn_sorted = np.argwhere(sorted_m == cn)


    if plot == True:
        plt.figure(figsize=(16, 8))
        plt.subplot(121)
        plt.imshow(dff_avg[sorted_m[:]], aspect="auto", cmap=cmap_cl, norm=norm_cl)
        plt.axvline(x=tframes//2, color='black')
        plt.yticks(cn_sorted[0], [f'{cn}: CN'])
        plt.title(f'{mouse}-{session}')
        plt.colorbar()

        plt.subplot(122)
        plt.plot(dff_avg[cn])
        plt.title(f"{cn}: Conditioned Neuron")
        plt.axvline(x=1000, ymin=0.25, ymax=0.75, color='black', linestyle='--')

        os.makedirs(os.path.join(plt_save_path, mouse), exist_ok=True)
        save_path = os.path.join(plt_save_path, mouse, f"lick_triggered_population-{session}-aligned-{align_at}")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"saved to {save_path}")
        plt.show()

        plot_trace(dff_avg, sem, session, indices=sorted_m[-8:], dim=(4, 2))
    
    return dff_avg, sem, sorted_m

def linear_regression(suite2p_path,
                    dlc_base_dir,
                    bpod_path,
                    sessionwise_data_path,
                    plt_save_path,
                    aligned_data_path,
                    mouse="BCI_26",
                    FOV="FOV_04",
                    camera="side",
                    session="041022"):

    dff_avg, sorted_m = plot_population_lta(suite2p_path, dlc_base_dir, 
                        bpod_path, sessionwise_data_path, plt_save_path, 
                        aligned_data_path, mouse, FOV, camera, session, plot=False)

    plt.plot()


def plot_sessionwise_change(suite2p_path,
                            dlc_base_dir,
                            bpod_path,
                            sessionwise_data_path,
                            plt_save_path,
                            aligned_data_path,
                            align_at="lick",
                            mouse="BCI_26",
                            FOV="FOV_04",
                            camera="side",
                            session_list=["041022", "041122"]):
    
    delta_change = []
    dffs = []
    sorts = []
    sems = []
    for session in session_list:
        dff_avg, sem, sorted_m = plot_population_lta(suite2p_path, dlc_base_dir, 
                            bpod_path, sessionwise_data_path, plt_save_path, 
                            aligned_data_path, align_at, mouse, FOV, camera, session, plot=False)
        
        dffs.append(dff_avg)
        sems.append(sem)
        sorts.append(sorted_m)
        print(session, sorted_m[:10])

    tframes = 2000
    means = [np.sum(k, axis=1) for k in dffs]

    # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20, 10))
    # plot_trace(dffs[0], sems[0], session_list[0], sorts[0][:10], ax=ax1)
    # plot_trace(dffs[1], sems[1], session_list[1], sorts[0][:10], ax=ax2)
    # save_path = os.path.join(plt_save_path, f'lta_example_traces-{session_list}')
    # plt.savefig(save_path)
    # plt.show()

    plot_daycompare(dffs, sems, session_list, sorts[0][:10], dim=(2, 5), align_at=align_at)

    # print(np.mean(delta_change[0]), np.mean(delta_change[1]))
    # plt.scatter(delta_change[0], delta_change[1], marker='.', alpha=0.4, c='black')
    # plt.scatter(np.mean(delta_change[0]), np.mean(delta_change[1]), c='red', marker='.')
    # plt.xlabel(f'{session_list[0]}')
    # plt.ylabel(f'{session_list[1]}')
    # plt.show()

    fig, axes = plt.subplots(1, len(session_list), figsize=(16, 8), sharey=True)
    for i, session in enumerate(session_list):
        if i == 0:
            axes[i].set_ylabel(f'Neurons')
            axes[i].set_xlabel(f'Frames')
        else:
            axes[i].set_yticks([])
            axes[i].set_xticks([])
        sns.despine(bottom=True, left=True)
        im = axes[i].imshow(dffs[i][sorts[0]], aspect="auto", cmap=cmap_cl, norm=norm_cl)
        axes[i].axvline(x=tframes//2, color='black')
        # plt.yticks(cn_sorted[0], [f'{cn}: CN'])
        axes[i].set_title(f'{mouse}-{session_list[i]}')

    plt.tight_layout()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.10, 0.02, 0.8])
    fig.colorbar(im, cax=cbar_ax)
    # fig.colorbar(im, ax=axes.ravel().tolist())
    os.makedirs(os.path.join(plt_save_path, mouse), exist_ok=True)
    save_path = os.path.join(plt_save_path, mouse, f"{'-'.join(session_list)}-{align_at}-heatmap.png")
    plt.savefig(save_path)
    plt.show()


# %%
if __name__ == "__main__":
    dlc_base_dir = os.path.abspath("/home/labadmin/Github/BCI_analysis/bucket/Data/Behavior_videos/DLC_output/Bergamo-2P-Photostim/")
    bpod_path = os.path.abspath("/home/labadmin/Github/BCI_analysis/bucket/Data/Behavior/BCI_exported/Bergamo-2P-Photostim/")
    suite2p_path = os.path.abspath("/home/labadmin/Github/BCI_analysis/bucket/Data/Calcium_imaging/suite2p/Bergamo-2P-Photostim/")
    sessionwise_data_path = os.path.abspath("/home/labadmin/Github/BCI_analysis/bucket/Data/Calcium_imaging/sessionwise_tba/")
    plt_save_path = os.path.abspath("/home/labadmin/Github/BCI_analysis/Plots/")
    aligned_data_path = os.path.abspath("/home/labadmin/Github/BCI_analysis/data_aligned")


    mouse = "BCI_26"
    FOV = "FOV_04"
    camera = "side"
    align_at = "reward"
# session_list = ["041322", "041322_2"]

# # for session in session_list:
# #     plot_population_lta(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, 
# #                         plt_save_path, aligned_data_path, mouse, FOV, camera, session)

# plot_sessionwise_change(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, plt_save_path, aligned_data_path, mouse, FOV, camera, session_list)

# session_list = ["041422", "041522"]
# plot_sessionwise_change(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, plt_save_path, aligned_data_path, mouse, FOV, camera, session_list)

# session_list = ["042522", "042722"]
# plot_sessionwise_change(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, plt_save_path, aligned_data_path, align_at, mouse, FOV, camera, session_list)

# session_list = ["041922", "042022"]
# plot_sessionwise_change(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, plt_save_path, aligned_data_path, mouse, FOV, camera, session_list)

    session_list = ["041022", "041122"]
    plot_sessionwise_change(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, plt_save_path, aligned_data_path, align_at, mouse, FOV, camera, session_list)

# session_list = ["042722", "042822"]
# plot_sessionwise_change(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, plt_save_path, aligned_data_path, mouse, FOV, camera, session_list)
# session_list = ["041922", "042022", "042122", "042222", "042722", "042822", "042922"]
# plot_sessionwise_change(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, plt_save_path, aligned_data_path, mouse, FOV, camera, session_list)
    # for session in ["041222", "041222_2", "041322"]:
    #     plot_population_lta(suite2p_path, dlc_base_dir,
    #                         bpod_path, sessionwise_data_path, plt_save_path,
    #                         aligned_data_path, align_at, mouse, FOV, camera, session, overwrite=True)

    # session = "050422"
    # plot_population_lta(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path,
    #                     plt_save_path, aligned_data_path, align_at, mouse, FOV, camera, session, plot=True)

    # bpod_path = os.path.abspath("/home/labadmin/Github/BCI_analysis/bucket/Data/Behavior/BCI_exported/Bergamo-2P-Photostim/")
    # suite2p_path = os.path.abspath("/home/labadmin/Github/BCI_analysis/bucket/Data/Calcium_imaging/suite2p/Bergamo-2P-Photostim/")
    # mice_name = "BCI_26"
    # raw_data_path = os.path.abspath("/home/labadmin/Github/BCI_analysis/bucket/Data/Calcium_imaging/raw/Bergamo-2P-Photostim/")
    # save_path = os.path.abspath("/home/labadmin/Github/BCI_analysis/bucket/Data/Calcium_imaging/sessionwise_tba")
    # suite2p_to_npy(suite2p_path, raw_data_path, bpod_path, save_path, overwrite=True, mice_name = mice_name)
