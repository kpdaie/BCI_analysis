import os
from pathlib import Path
import numpy as np

def extract_files_from_dir(basedir):
    #%
    files = os.listdir(basedir)
    exts = list()
    basenames = list()
    fileindexs = list()
    dirs = list()
    for file in files:
        if Path(os.path.join(basedir,file)).is_dir():
            exts.append('')
            basenames.append('')
            fileindexs.append(np.nan)
            dirs.append(file)
        else:
            if '_' in file:# and ('cell' in file.lower() or 'stim' in file.lower()):
                basenames.append(file[:-1*file[::-1].find('_')-1])
                try:
                    fileindexs.append(int(file[-1*file[::-1].find('_'):file.find('.')]))
                except:
                    print('weird file index: {}'.format(file))
                    fileindexs.append(-1)
            else:
                basenames.append(file[:file.find('.')])
                fileindexs.append(-1)
            exts.append(file[file.find('.'):])
    tokeep = np.asarray(exts) != ''
    files = np.asarray(files)[tokeep]
    exts = np.asarray(exts)[tokeep]
    basenames = np.asarray(basenames)[tokeep]
    fileindexs = np.asarray(fileindexs)[tokeep]
    out = {'dir':basedir,
           'filenames':files,
           'exts':exts,
           'basenames':basenames,
           'fileindices':fileindexs,
           'dirs':dirs
           }
    #%
    return out