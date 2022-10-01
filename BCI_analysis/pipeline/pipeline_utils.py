import numpy as np
import warnings
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        
    Source:
        https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def rollingfun(y, window = 10, func = 'mean'):
    """
    rollingfun
        rolling average, min, max or std
    
    @input:
        y = array, window, function (mean,min,max,std)
    """
    


    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
    
        window = int(window)
        if window >len(y):
            window = len(y)-1
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
            print('undefinied funcion in rollinfun')
    return out 