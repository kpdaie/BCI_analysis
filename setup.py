from setuptools import find_packages, setup

setup(
    name='BCI_analysis',
    packages=find_packages(),
    install_requires=['suite2p',
                      'scipy',
                      'pandas',
                      'scanimage_tiff_reader',
                      'pybpod-gui-api',#'pybpod',
                      'scikit-learn',
                      'matplotlib']
                      #'mat73',# for loading old matlab files, 
                      #'umap-learn',# for axon imaging, temporary
                      #'rastermap'] # problems with numpy
    )
