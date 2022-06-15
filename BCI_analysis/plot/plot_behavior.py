#%%
import sys
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import os


sys.path.append("/home/labadmin/Github/BCI_analysis/BCI_analysis/")
from pipeline.pipeline_align import get_aligned_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from plot_utils import rollingfun
from sklearn.decomposition import PCA

dlc_base_dir = os.path.abspath("bucket/Data/Behavior_videos/DLC_output/Bergamo-2P-Photostim/")
bpod_path = os.path.abspath("bucket/Data/Behavior/BCI_exported/Bergamo-2P-Photostim/")
suite2p_path = os.path.abspath("bucket/Data/Calcium_imaging/suite2p/Bergamo-2P-Photostim/")
sessionwise_data_path = os.path.abspath("bucket/Data/Calcium_imaging/sessionwise_tba/")
plt_save_path = os.path.abspath("Plots/")

mouse = "BCI_26"
FOV = "FOV_04"
camera = "side"
session = "041522"
# F_aligned, DLC_aligned, trial_lengths = get_aligned_data(suite2p_path, dlc_base_dir, bpod_path, 
#                                 sessionwise_data_path, mouse, FOV, camera, session)
# ca_data = np.load(os.path.join(sessionwise_data_path, mouse, "-".join([mouse, session, FOV])+".npy"), allow_pickle=True).tolist()
#%%

def pc_dataframe(DLC_aligned: pd.DataFrame, drop_pole=False) -> pd.DataFrame:

    dlc_cpy = DLC_aligned.copy(deep=True)
    bodyparts = DLC_aligned.columns.levels[0]
    for bp in bodyparts:
        if bp == 'Pole' and drop_pole == True:
            dlc_cpy.drop((bp, 'x'), axis=1, inplace=True)
            dlc_cpy.drop((bp, 'y'), axis=1, inplace=True)
            continue
        pca = PCA(n_components=2)
        X_t = pca.fit_transform(dlc_cpy[bp].values)
        print(f"{bp}: {pca.explained_variance_ratio_}")    
        if pca.explained_variance_ratio_[0] > 0.6:
            dlc_cpy.drop((bp, 'y'), axis=1, inplace=True)
            dlc_cpy.loc[:, (bp, 'x')] = X_t[:, 0].reshape(-1, 1)
    return dlc_cpy

def linear_regression(F_aligned, DLC_aligned):

    print(f"Calculating regression fit for {F_aligned.shape[0]} neurons, May take time")
    n_features = DLC_aligned.shape[1]
    beta_vals = np.zeros((F_aligned.shape[0], n_features))
    scores = []
    intercept = []
    for neuron in tqdm(range(F_aligned.shape[0])):
        lr = LinearRegression()
        lr.fit(DLC_aligned, F_aligned[neuron])
        scores.append(lr.score(DLC_aligned, F_aligned[neuron]))
        beta_vals[neuron] = lr.coef_
        intercept.append(lr.intercept_)

    return scores, beta_vals, intercept

def normalize_and_window(DLC_aligned, F_aligned, start, end, window, save=False):
    normalize = StandardScaler()
    dlc_cpy = pc_dataframe(DLC_aligned[start:end], drop_pole=True)
    DLC_aligned_n = np.zeros_like(dlc_cpy)
    F_aligned_n = np.zeros_like(F_aligned[:, start:end])

    print(f"Normalize F and Apply a sliding window of {window}")
    for neu in tqdm(range(F_aligned.shape[0])):
        temp = normalize.fit_transform(F_aligned[neu, start:end].reshape(-1, 1))
        F_aligned_n[neu] = rollingfun(temp, window=window).flatten()

    print(f"Normalize DLC data and apply a sliding window of {window}")
    for i in tqdm(range(0, dlc_cpy.shape[1])):
        temp = normalize.fit_transform(dlc_cpy.values[start:end, i].reshape(-1, 1))
        DLC_aligned_n[:, i] = rollingfun(temp, window=window).flatten()


    return dlc_cpy, DLC_aligned_n, F_aligned_n


def apply_lr_plot(DLC_aligned, F_aligned, ca_data, start=0, end=100000, window=200):

    dlc_cpy, DLC_aligned_n, F_aligned_n = normalize_and_window(DLC_aligned, F_aligned, start, end, window)
#%%
    print(f"Using Linear regression on {F_aligned_n.shape[0]} neurons")
    scores, beta_vals, intercept = linear_regression(F_aligned_n, DLC_aligned_n)
    scores = np.asarray(scores)
# %%
    cn = ca_data['cn'][0]
    _, ax = plt.subplots(2, 2, figsize=(16,10))
    plt.suptitle(f"{mouse} - {session} - {camera} - {FOV}-window={window}")
    ax[0,0].plot(F_aligned_n[cn, :], label=f'neuron {cn} Flourescence Trace')
    ax[0,0].plot((beta_vals[cn]@DLC_aligned_n.T) + intercept[cn], label='Fitted Data')
    ax[0,0].set_title(f'{end-start} timepoints, score {scores[cn]:0.2f}')
    ax[0,0].set_xlabel("Time")
    ax[0,0].set_ylabel("Normalized F")
# ax[0].set_xlim([0, 30000])
    ax[0,0].legend()

    ax[0,1].hist(scores, bins=100)
    ax[0,1].set_xlabel("LR score")
    ax[0,1].set_ylabel("Number of neurons")

    ax[1,0].imshow(beta_vals*(np.abs(beta_vals)>0.1), norm=cl.TwoSlopeNorm(0), cmap='seismic', aspect='auto')
    ax[1,0].set_xticks(np.arange(0, dlc_cpy.shape[1]), dlc_cpy.columns.values, rotation ='vertical')
# %%

    im = ax[1,1].imshow(np.expand_dims(beta_vals[ca_data['cn'][0]], axis=0), norm=cl.TwoSlopeNorm(0), cmap='seismic', aspect='auto')
    ax[1,1].set_xticks(np.arange(0, dlc_cpy.shape[1]), dlc_cpy.columns.values, rotation ='vertical')
    ax[1,1].set_title(f"cn = {ca_data['cn'][0]}")
    cbar = plt.colorbar(im)

    save_path = os.path.join(plt_save_path, f"{mouse}-{session}-{camera}-{FOV}-window={window}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show(block=False)

# %%

def sliding_fit(DLC_aligned, F_aligned, start=0, end=100000, window=100, slide=[0]):

    _, DLC_aligned_n, F_aligned_n = normalize_and_window(DLC_aligned, F_aligned, start, end, window)

    cn = ca_data['cn'][0]
    _, ax = plt.subplots(len(slide), 2, figsize=(16,10))
    plt.suptitle(f"{mouse} - {session} - {camera} - {FOV}-window={window}")

    for i, k in enumerate(list(slide)):
        F_ = np.zeros_like(F_aligned_n)
        if k>0:
            F_[:, k:] = F_aligned_n[:, :-k]
            F_[:, :k] = np.mean(F_aligned_n[:, :k])
        elif k<0:
            F_[:, :-np.abs(k)] = F_aligned_n[:, np.abs(k):]
            F_[:, -np.abs(k):] = np.mean(F_aligned_n[:, -np.abs(k):])
        else:
            F_ = F_aligned_n

        print(f"Using Linear regression on {F_.shape[0]} neurons")
        scores, beta_vals, intercept = linear_regression(F_, DLC_aligned_n)
        scores = np.asarray(scores)
# %%
        ax[i,0].plot(F_[cn, 0:10000], label=f'neuron {cn} Flourescence Trace')
        ax[i,0].plot(((beta_vals[cn]@DLC_aligned_n.T) + intercept[cn])[0:10000], label='Fitted Data')
        ax[i,0].set_title(f'slide = {k}')
        ax[i,0].set_xlabel("Time")
        ax[i,0].set_ylabel("Normalized F")
        ax[i,0].legend()

        ax[i,1].hist(scores, bins=100, label=f'average R^2 = {np.mean(scores):0.5f}')
        ax[i,1].set_xlabel("LR score")
        ax[i,1].set_ylabel("Number of neurons")
        ax[i,1].legend()

    save_path = os.path.join(plt_save_path, f"{mouse}-{session}-{camera}-{FOV}-window={window}-slide={slide}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show(block=False)

# # for window in [1, 10, 100, 200, 400, 1000]:
# #     apply_lr_plot(DLC_aligned, F_aligned, ca_data, window=window)
#
# sliding_fit(DLC_aligned, F_aligned, window=100, slide=[-2000, -1000, -400, 0, 400, 1000, 2000])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None) 
    parser.add_argument('--save', type=str, default=None) 
    args = parser.parse_args()
    
    print(args.path)
    if args.path:
        a = np.load(args.path)
        ca_data = np.load(os.path.join(sessionwise_data_path, mouse, "-".join([mouse, session, FOV])+".npy"), allow_pickle=True).tolist()

        DLC_aligned_n = a['DLC_aligned_n']
        F_aligned_n = a['F_aligned_n']
        dlc_cpy = a['dlc_cpy']
        scores = a['scores']
        beta_vals = a['beta_vals']
        intercept = a['intercept']
        cn = ca_data['cn'][0]

        print(scores)

    if args.save:
        mouse = "BCI_26"
        FOV = "FOV_04"
        camera = "side"
        session = args.save
        F_aligned, DLC_aligned, trial_lengths = get_aligned_data(suite2p_path, dlc_base_dir, bpod_path,
                                        sessionwise_data_path, mouse, FOV, camera, session)
        ca_data = np.load(os.path.join(sessionwise_data_path, mouse, "-".join([mouse, session, FOV])+".npy"), allow_pickle=True).tolist()
        start = 0
        end = DLC_aligned.shape[0]
        window = 100

        dlc_cpy, DLC_aligned_n, F_aligned_n = normalize_and_window(DLC_aligned, F_aligned, start, end, window)
        scores, beta_vals, intercept = linear_regression(F_aligned_n, DLC_aligned_n)

        t = time.localtime()
        current_time = time.strftime("%H%M%S", t)
        os.makedirs(os.path.join(plt_save_path, "use_data", mouse), exist_ok=True)
        save_path = os.path.join(plt_save_path, "use_data", mouse, f"window{window}-{end}-"+current_time)
        np.savez(save_path, DLC_aligned_n=DLC_aligned_n, F_aligned_n=F_aligned_n, dlc_cpy=dlc_cpy, scores=scores, beta_vals=beta_vals, intercept=intercept)
