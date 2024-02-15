# We run this function after every session to export behavior data and align it with imaging

import BCI_analysis
behavior_export_basedir = '/mnt/Data/Behavior/BCI_exported'
calcium_imaging_raw_basedir = '/mnt/Data/Calcium_imaging/raw'
raw_behavior_dirs = ['/mnt/Data/Behavior/raw/DOM-3/BCI',
                     '/mnt/Data/Behavior/raw/DOM3-MMIMS/BCI',
                     '/mnt/Data/Behavior/raw/KayvonScope/BCI',
                     '/mnt/Data/Behavior/raw/SLAP2/BCI']
zaber_root_folder = '/mnt/Data/Behavior/BCI_Zaber_data'
BCI_analysis.pipeline_bpod.export_pybpod_files(behavior_export_basedir,
                                                calcium_imaging_raw_basedir,
                                                raw_behavior_dirs, 
                                                zaber_root_folder,
                                                overwrite=False)