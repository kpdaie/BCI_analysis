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
import seaborn as sns


from BCI_analysis.pipeline.pipeline_align import get_aligned_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from BCI_analysis.plot.plot_utils import rollingfun
from sklearn.decomposition import PCA

dlc_base_dir = os.path.abspath("bucket/Data/Behavior_videos/DLC_output/Bergamo-2P-Photostim/")
bpod_path = os.path.abspath("bucket/Data/Behavior/BCI_exported/Bergamo-2P-Photostim/")
suite2p_path = os.path.abspath("bucket/Data/Calcium_imaging/suite2p/Bergamo-2P-Photostim/")
sessionwise_data_path = os.path.abspath("bucket/Data/Calcium_imaging/sessionwise_tba/")
plt_save_path = os.path.abspath("Plots/")
aligned_data_path = os.path.abspath("/home/labadmin/Github/BCI_analysis/data_aligned")


# F_aligned, DLC_aligned, trial_lengths = get_aligned_data(suite2p_path, dlc_base_dir, bpod_path, 
#                                 sessionwise_data_path, mouse, FOV, camera, session)
# ca_data = np.load(os.path.join(sessionwise_data_path, mouse, "-".join([mouse, session, FOV])+".npy"), allow_pickle=True).tolist()
#%%

def pc_dataframe(DLC_aligned: pd.DataFrame, drop_pole=False) -> pd.DataFrame:

    dlc_cpy = DLC_aligned.copy(deep=True)
    dlc_cpy.drop('likelihood', level=1, axis=1, inplace=True)
    bodyparts = DLC_aligned.columns.levels[0]
    for bp in bodyparts:
        if bp == 'Pole' and drop_pole == True:
            dlc_cpy.drop((bp, 'x'), axis=1, inplace=True)
            dlc_cpy.drop((bp, 'y'), axis=1, inplace=True)
            continue
        if bp == 'EyeUp' and drop_pole == True:
            dlc_cpy.drop((bp, 'x'), axis=1, inplace=True)
            dlc_cpy.drop((bp, 'y'), axis=1, inplace=True)
            continue
        if bp == 'EyeLeft' and drop_pole == True:
            dlc_cpy.drop((bp, 'x'), axis=1, inplace=True)
            dlc_cpy.drop((bp, 'y'), axis=1, inplace=True)
            continue
        if bp == 'EyeDown' and drop_pole == True:
            dlc_cpy.drop((bp, 'x'), axis=1, inplace=True)
            dlc_cpy.drop((bp, 'y'), axis=1, inplace=True)
            continue
        if bp == 'EyeRight' and drop_pole == True:
            dlc_cpy.drop((bp, 'x'), axis=1, inplace=True)
            dlc_cpy.drop((bp, 'y'), axis=1, inplace=True)
            continue
        pca = PCA(n_components=2)
        X_t = pca.fit_transform(dlc_cpy[bp].values)
        print(f"{bp}: {pca.explained_variance_ratio_}")    
        if pca.explained_variance_ratio_[0] > 0.6:
            dlc_cpy.drop((bp, 'y'), axis=1, inplace=True)
            dlc_cpy.loc[:, (bp, 'x')] = X_t[:, 0].reshape(-1, 1)
    print(dlc_cpy.columns.levels[0])
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
    print(F_aligned.shape, start, end)
    F_aligned_n = np.zeros_like(F_aligned[:, start:end])

    print(f"Normalize DLC data and apply a sliding window of {window}")
    for i in tqdm(range(0, dlc_cpy.shape[1])):
        temp = normalize.fit_transform(dlc_cpy.values[:, i].reshape(-1, 1))
        dlc_cpy.values[:, i] = rollingfun(temp, window=window).flatten()

    print(f"Normalize F and Apply a sliding window of {window}")
    for neu in tqdm(range(F_aligned.shape[0])):
        temp = normalize.fit_transform(F_aligned[neu, start:end].reshape(-1, 1))
        F_aligned_n[neu] = rollingfun(temp, window=window).flatten()


    print(dlc_cpy)
    print(pd.DataFrame.from_dict(dlc_cpy.to_dict()))
    return dlc_cpy, F_aligned_n


def apply_lr_plot(DLC_aligned, F_aligned, cn, scores=None, beta_vals=None, intercept=None, start=50000, end=150000, window=200):

#     DLC_aligned_n, F_aligned_n = normalize_and_window(DLC_aligned, F_aligned, start, end, window)
# #%%
#     print(f"Using Linear regression on {F_aligned_n.shape[0]} neurons")
#     scores, beta_vals, intercept = linear_regression(F_aligned_n, DLC_aligned_n)
#     scores = np.asarray(scores)
# %%
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
    ax[1,0].set_xticks(np.arange(0, DLC_aligned_n.shape[1]), DLC_aligned_n.columns.values, rotation ='vertical')
# %%

    im = ax[1,1].imshow(np.expand_dims(beta_vals[cn], axis=0), norm=cl.TwoSlopeNorm(0), cmap='seismic', aspect='auto')
    ax[1,1].set_xticks(np.arange(0, DLC_aligned_n.shape[1]), DLC_aligned_n.columns.values, rotation ='vertical')
    ax[1,1].set_title(f"cn = {cn}")
    cbar = plt.colorbar(im)

    save_path = os.path.join(plt_save_path, f"{mouse}-{session}-{camera}-{FOV}-window={window}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show(block=False)

# %%

def sliding_fit(DLC_aligned, F_aligned, normalize=True, start=0, end=100000, window=100, slide=[0]):

    if normalize:
        DLC_aligned_n, F_aligned_n = normalize_and_window(DLC_aligned, F_aligned, start, end, window)
    else:
        DLC_aligned_n, F_aligned_n = DLC_aligned, F_aligned

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
        ax[i,0].plot(F_[cn, :], label=f'neuron {cn} Flourescence Trace')
        ax[i,0].plot(((beta_vals[cn]@DLC_aligned_n.T) + intercept[cn]), label='Fitted Data')
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
    # plt.show(block=False)

# # for window in [1, 10, 100, 200, 400, 1000]:
# #     apply_lr_plot(DLC_aligned, F_aligned, ca_data, window=window)
# sliding_fit(DLC_aligned, F_aligned, window=100, slide=[-2000, -1000, -400, 0, 400, 1000, 2000])
from scipy.stats import pearsonr

def corrfunc(x, y, hue=None, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None) 
    parser.add_argument('--save', type=str, default=None) 
    parser.add_argument('--apply_lr', type=str, default=False)
    args = parser.parse_args()
    mouse = "BCI_26"
    FOV = "FOV_04"
    camera = "side"
    session = "041022"
    
    print(args.path)
    if args.path:
        a = np.load(args.path, allow_pickle=True)
        ca_data = np.load(os.path.join(sessionwise_data_path, mouse, "-".join([mouse, session, FOV])+".npy"), allow_pickle=True).tolist()

        DLC_aligned_n = pd.DataFrame.from_dict(a["DLC_aligned_n"].item())
        # F_aligned_n = a['F_aligned_n']
        # dlc_cpy = a['dlc_cpy']
        # scores = a['scores']
        # beta_vals = a['beta_vals']
        # intercept = a['intercept']
        cn = ca_data['cn'][0]
        dlc_headers = a['dlc_headers']
        print(dlc_headers)
        DLC_aligned_n.columns = dlc_headers

        # sliding_fit(DLC_aligned_n, F_aligned_n, normalize=False, window=200, slide=[-2000, -1000, 0, 1000, 2000])
        good_neuron = cn
        # fig, ax = plt.subplots(len(dlc_headers)+1, 1, figsize=(16,12), sharex=True)

        # ax[0].plot(F_aligned_n[good_neuron, :])
        # for i, bp in enumerate(np.argsort(np.abs(beta_vals[good_neuron]))[::-1]):
        #     ax[i+1].plot(dlc_cpy[:, bp], label=f"beta_val = {beta_vals[good_neuron, bp]:0.3f}")
        #     ax[i+1].set_ylabel(f"{dlc_headers[bp]}", rotation='horizontal', ha='right')
        #     ax[i+1].legend()
        # plt.show()
        # print(DLC_aligned_n)
        plt.figure()
        
        # g = sns.pairplot(DLC_aligned_n.loc[:100000])
        p = 30000
        DLC_aligned_n.columns = ['-'.join(DLC_aligned_n.columns.to_flat_index()[i]) for i in range(len(DLC_aligned_n.columns))]
        print(DLC_aligned_n)
        g = sns.PairGrid(DLC_aligned_n.sample(p), palette="crest")
        g.map_upper(plt.scatter, s=0.1)
        g.map_diag(sns.histplot, kde=True, color='0.3')
        g.map_lower(sns.kdeplot, cmap="Blues_d", fill=True)
        g.map_lower(corrfunc)
        for ax in g.axes.flatten():
            ax.set_ylabel(ax.get_ylabel(), rotation = 60)
            ax.set_xlabel(ax.get_xlabel(), rotation = 0)
        # plt.savefig("fancy_plot")
        plt.tight_layout()
        plt.show()


    if args.save:
        mouse = "BCI_26"
        FOV = "FOV_04"
        camera = "side"
        session = args.save
        dict_aligned = get_aligned_data(suite2p_path, dlc_base_dir, bpod_path, 
                sessionwise_data_path, aligned_data_path, mouse, 
                FOV, camera, session, plot=False, overwrite=False)

        if dict_aligned is None:
            pass
        dff_aligned = dict_aligned["dff_aligned"]
        dff_aligned_s = np.hstack(dff_aligned)
        DLC_aligned = pd.DataFrame.from_dict(dict_aligned["DLC_aligned"])

        ca_data = np.load(os.path.join(sessionwise_data_path, mouse, "-".join([mouse, session, FOV])+".npy"), allow_pickle=True).tolist()
        start = 0
        end = DLC_aligned.shape[0]
        window = 10

        DLC_aligned_n, dff_aligned_n = normalize_and_window(DLC_aligned, dff_aligned_s, start, end, window)
        scores, beta_vals, intercept = linear_regression(dff_aligned_n, DLC_aligned_n)

        t = time.localtime()
        current_time = time.strftime("%H%M%S", t)
        os.makedirs(os.path.join(plt_save_path, "use_data", mouse), exist_ok=True)
        save_path = os.path.join(plt_save_path, "use_data", mouse, f"{session}-window{window}-{end}")
        np.savez(save_path, DLC_aligned_n=DLC_aligned_n.to_dict(), F_aligned_n=dff_aligned_n, dlc_headers=np.asarray(DLC_aligned_n.columns.values), scores=scores, beta_vals=beta_vals, intercept=intercept)

    if args.apply_lr:

        mouse = "BCI_26"
        FOV = "FOV_04"
        camera = "side"
        session = "041022"
        a = np.load(args.apply_lr, allow_pickle=True)
        ca_data = np.load(os.path.join(sessionwise_data_path, mouse, "-".join([mouse, session, FOV])+".npy"), allow_pickle=True).tolist()

        DLC_aligned_n = pd.DataFrame.from_dict(a["DLC_aligned_n"].item())
        F_aligned_n = a['F_aligned_n']
        scores = a['scores']
        beta_vals = a['beta_vals']
        intercept = a['intercept']
        cn = ca_data['cn'][0]
        dlc_headers = a['dlc_headers']
        print(dlc_headers)

        start = 0
        end = DLC_aligned_n.shape[0]
        print(start, end)
        window = 100

        # for window in [1, 10, 100, 200, 400, 1000]:
        apply_lr_plot(DLC_aligned_n, F_aligned_n, cn, scores, beta_vals, intercept, window=200)
