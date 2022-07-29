import numpy as np

def rollingfun(y, window = 10, func = 'mean'):
    """
    rollingfun
        rolling average, min, max or std
    
    @input:
        y = array, window, function (mean,min,max,std)
    """
    if len(y)<=window:
        if func =='mean':
            out = np.ones(len(y))*np.nanmean(y)
        elif func == 'min':
            out = np.ones(len(y))*np.nanmin(y)
        elif func == 'max':
            out = np.ones(len(y))*np.nanmax(y)
        elif func == 'std':
            out = np.ones(len(y))*np.nanstd(y)
        elif func == 'median':
            out = np.ones(len(y))*np.nanmedian(y)
        else:
            print('undefinied funcion in rollinfun')
    else:
        y = np.concatenate([y[window::-1],y,y[:-1*window:-1]])
        ys = list()
        for idx in range(window):    
            ys.append(np.roll(y,idx-round(window/2)))
        if func =='mean':
            out = np.nanmean(ys,0)[window:-window]
        elif func == 'min':
            out = np.nanmin(ys,0)[window:-window]
        elif func == 'max':
            out = np.nanmax(ys,0)[window:-window]
        elif func == 'std':
            out = np.nanstd(ys,0)[window:-window]
        elif func == 'median':
            out = np.nanmedian(ys,0)[window:-window]
        else:
            print('undefined funcion in rollinfun')
    return out
