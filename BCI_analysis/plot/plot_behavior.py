#%%
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.append('../')
import os

from pipeline.pipeline_align import get_aligned_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from plot_utils import rollingfun


#%%

dlc_base_dir = os.path.abspath("../../bucket/Data/Behavior_videos/DLC_output/Bergamo-2P-Photostim/")
bpod_path = os.path.abspath("../../bucket/Data/Behavior/BCI_exported/Bergamo-2P-Photostim/")
suite2p_path = os.path.abspath("../../bucket/Data/Calcium_imaging/suite2p/Bergamo-2P-Photostim/")
sessionwise_data_path = os.path.abspath("../../bucket/Data/Calcium_imaging/sessionwise_tba/")
mouse = "BCI_26"
FOV = "FOV_04"
camera = "side"
session = "041022"
F_aligned, DLC_aligned = get_aligned_data(suite2p_path, dlc_base_dir, bpod_path, 
                                sessionwise_data_path, mouse, FOV, camera, session)
ca_data = np.load(os.path.join(sessionwise_data_path, mouse, "-".join([mouse, session, FOV])+".npy"), allow_pickle=True).tolist()
#%%

from sklearn.decomposition import PCA

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
            drop = "True"
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


# %%
start = 0
end = 100000
window = 100
normalize = StandardScaler()
# F_aligned_n = normalize.fit_transform(F_aligned[:, start:end])
dlc_cpy = pc_dataframe(DLC_aligned[start:end], drop_pole=True)
DLC_aligned_n = np.zeros_like(dlc_cpy)
F_aligned_n = np.zeros_like(F_aligned[:, start:end])
for neu in tqdm(range(F_aligned.shape[0])):
    temp = normalize.fit_transform(F_aligned[neu, start:end].reshape(-1, 1))
    F_aligned_n[neu] = rollingfun(temp, window=window).flatten()
print("F rolled")
for i in tqdm(range(0, dlc_cpy.shape[1])):
    temp = normalize.fit_transform(dlc_cpy.values[start:end, i].reshape(-1, 1))
    DLC_aligned_n[:, i] = rollingfun(temp, window=window).flatten()
print("DLC rolled")

#%%
scores, beta_vals, intercept = linear_regression(F_aligned_n, DLC_aligned_n)
scores = np.asarray(scores)
# %%
cn = ca_data['cn'][0]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(F_aligned_n[cn, :], label=f'neuron {cn} Flourescence Trace')
ax[0].plot((beta_vals[cn]@DLC_aligned_n.T) + intercept[cn], label='Fitted Data')
ax[0].set_title(f'{end-start} timepoints, score {scores[cn]:0.2e}')
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Normalized F")
# ax[0].set_xlim([0, 30000])
ax[0].legend()

ax[1].hist(scores, bins=100)
ax[1].set_xlabel("LR score")
ax[1].set_ylabel("Number of neurons")
plt.show()

# %%

plt.imshow(beta_vals*(np.abs(beta_vals)>0.1), norm=cl.TwoSlopeNorm(0), cmap='seismic', aspect='auto')
plt.xticks(np.arange(0, dlc_cpy.shape[1]), dlc_cpy.columns.values, rotation ='vertical')
plt.colorbar()
plt.show()
# %%

plt.imshow(np.expand_dims(beta_vals[ca_data['cn'][0]], axis=0), norm=cl.TwoSlopeNorm(0), cmap='seismic', aspect='auto')
plt.xticks(np.arange(0, dlc_cpy.shape[1]), dlc_cpy.columns.values, rotation ='vertical')
plt.title(f"cn = {ca_data['cn'][0]}")
plt.colorbar()
plt.show()

# %%
h_neu = np.argmax(scores)
(bp, dim) = DLC_aligned.columns[np.argmax(beta_vals[h_neu, :])]
plt.plot(DLC_aligned_n[:, 13:15], label=[f'{bp} x', f'{bp} y'])
plt.plot(F_aligned_n[h_neu, :], label=f'neuron {h_neu}, score {scores[h_neu]}')
plt.legend()
plt.show()

# %%
