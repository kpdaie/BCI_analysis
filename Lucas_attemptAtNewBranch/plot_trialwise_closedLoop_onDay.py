# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 07:49:57 2022

@author: Lucas
"""

## Imports
import matplotlib.pyplot as plt
import numpy as np

from grabData import *
dataDict = grabData()

## Set Up Trialwise Averages
dayOfInterest = input('Day of interest (day 1 = -1... dont do day 1): ')
day = int(dayOfInterest) + 1 #this is why you shouldn't do day 1
print('Interested in day '+dayOfInterest)
F_arr = np.array(dataDict['f_trialwise_closed_loop'][day]) #np.array turns consecutive brackets in python (example [[],[]]) into a matlab-type array
Fo_arr = np.array(dataDict['f_trialwise_closed_loop'][day-1])
F = np.nanmean(F_arr, axis = 2)
# Fo = np.nanmean(Fo_arr, axis = 2)

## Plot
plt.imshow(F)
plt.colorbar()
plt.title('Neuron Flourescent Intensity Averaged Across All Trials On Day ' + dayOfInterest)
plt.xlabel('Trial Time (s)')
plt.ylabel('Neuron')
plt.show()