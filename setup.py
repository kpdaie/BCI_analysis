from setuptools import find_packages, setup

setup(
    name='BCI_analysis',
    packages=find_packages(),
    install_requires=['suite2p',
                      'scipy',
                      'pandas',
                      'mat73',
                      'scanimage_tiff_reader',
                      'pybpod',
                      'scikit-learn',
                      'matplotlib']
                      #'umap-learn',# for axon imaging, temporary
                      #'rastermap'] # problems with numpy
    )
