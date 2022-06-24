# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:29:19 2022

@author: Lucas
"""
## Imports
import matplotlib.pyplot as plt
import numpy as np
from grabData import *

## grab ALL THE DATA!
dataDict = grabData()
print('Note, BCI11 has 98 neurons on day 0, so dont use day 1 against day 0 for BCI11')

## Set Up Scatter Data
dayOfInterest = input('Day of interest (day 1 = -1... dont do day 1): ')
day = int(dayOfInterest) + 1 #this is why you shouldn't do day 1
dayBefore = int(dayOfInterest)-1
print('Interested in day '+dayOfInterest)
neurons = np.array(dataDict['distance_from_conditioned_neuron'][day]) #np.array turns consecutive brackets in python (example [[],[]]) into a matlab-type array
dayFlo = np.array(dataDict['f_trialwise_closed_loop'][day])
dayB4Flo = np.array(dataDict['f_trialwise_closed_loop'][dayBefore]) #NOTE, dff_trialwise_closed_loop does not work... why?
meanFlo = np.nanmean(np.nanmean(dayFlo, axis = 2), axis=0)
meanFloB4 = np.nanmean(np.nanmean(dayB4Flo, axis = 2),axis=0)

diff_Flo = meanFlo - meanFloB4

#Showing Conditioned Neuron as Blue
CN = min(neurons)
CN_flo = diff_Flo[np.where(neurons==CN)[0][0]]

## go on... SCATTER! 
xAx = 'Pixels'
yAx = 'Difference In Flourescence From ' + dayOfInterest + ' to ' + str(dayBefore)
plt.plot(neurons,diff_Flo, 'o', markeredgecolor = 'grey', markerfacecolor = 'white')
plt.plot(CN,CN_flo, 'o', markeredgecolor = 'blue', markerfacecolor = 'blue')
plt.xlabel(xAx)
plt.ylabel(yAx)
plt.title('Distance from Conditioned Neuron Against Change in Average Flourescence Across Session')
plt.legend('Neurons', 'Conditioned Neuron')
plt.show()


