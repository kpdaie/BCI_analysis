from setuptools import find_packages, setup

setup(
    name='BCI_analysis',
    packages=find_packages(),
    install_requires=['scipy',
                      'pandas',
                      'mat73',
                      'scanimage_tiff_reader',
                      'pybpod',
                      'umap-learn',# for axon imaging, temporary
                      'scikit-learn',
                      'matplotlib',
                      'rastermap']
    )
