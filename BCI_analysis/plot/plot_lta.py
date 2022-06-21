import numpy as np
import matplotlib.pyplot as plt
import os, sys

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
session = "041022"

from pipeline.pipeline_align import align_licks

align_licks(suite2p_path, dlc_base_dir, bpod_path, sessionwise_data_path, mouse, FOV, camera, session)
