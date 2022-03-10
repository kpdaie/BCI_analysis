# -*- coding: utf-8 -*-
"""
Lucas Kinsey
20220309
grabData File
Quickly calls read mat file function and returns dictionary of a mouse
mouse is specified based on the response of the user
"""


import BCI_analysis as bci

def grabData():
    ## Specify Mouse and load its data
    mouseID = input("What is the Mouse's Name? ")
    dataDir = 'C:\\Users\\Lucas\\BCIAnalysis\\BCI_data\\'
    mousePath = dataDir + mouseID + '_030222v8.mat'
    dataDict = bci.io_matlab.read_multisession_mat(mousePath) #currently loads file every time script is run
    return dataDict