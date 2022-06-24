# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:18:01 2022

@author: Lucas
"""

## Imports
import matplotlib.pyplot as plt
import numpy as np
from grabData import *
dataDict = grabData()

## Set Up Mean Image
dayOfInterest = input('Day of interest (day 1 = -1... dont do day 1): ')
day = int(dayOfInterest) + 1 #this is why you shouldn't do day 1
print('Interested in day '+dayOfInterest)
F_arr = np.array(dataDict['mean_image'][day]) #np.array turns consecutive brackets in python (example [[],[]]) into a matlab-type array

## Plot
plt.imshow(F_arr)
plt.colorbar()
plt.title('Mean ROI On Day ' + dayOfInterest)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()