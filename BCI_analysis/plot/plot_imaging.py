import numpy as np
import matplotlib.pyplot as plt
import time
import os
from .. import io
from scipy.stats.stats import pearsonr as cor

def plot_trial_averaged_trace_2sessions(data_dict, day_ind, start_time=-2.5, 
                                        end_time=10, baseline_start_time = None,
                                        baseline_end_time = 0):
    """
    Returns a baseline subtracted plot of trial averaged flourescent trace of 
    the conditioned neuron of the selected session (day_ind) and the session 
    before.
    
    Parameters
    ----------
    data_dict : dict
        output of io.io_matlab.read_multisession_mat() function, highly 
        specialized multi-session
    day_ind : int
        Day index, the index of the session that will be plotted against the 
        previous session
    start_time : float, optional
        Time when plot starts. The default is -2.5.
    end_time : float, optional
        Time when plot ends. The default is 10.
    baseline_start_time : Tfloat, optional
        Time when baseline starts. The default is start_time.
    baseline_end_time : float, optional
        Time where baseline ends. The default is 0.

    Returns
    -------
    None
    A matplotlib figure.

    """
  
    if baseline_start_time == None: baseline_start_time = start_time
    
    time_yesterday=np.transpose(data_dict['tsta'][day_ind-1]).flatten()
    idx_yesterday = (time_yesterday>start_time) & (time_yesterday<end_time)
    time_yesterday = time_yesterday[idx_yesterday]
    
    time_today=np.transpose(data_dict['tsta'][day_ind]).flatten()
    idx_today = (time_today>start_time) & (time_today<end_time)
    time_today = time_today[idx_today]
    
    avg_traces_yesterday = np.nanmean(data_dict['F_trialwise'][day_ind-1][idx_yesterday,:,:], axis=2)
    avg_traces_today = np.nanmean(data_dict['F_trialwise'][day_ind][idx_today,:,:], axis=2)
    
    baseline_idx_today = (time_today>baseline_start_time) & (time_today<baseline_end_time)
    baseline_idx_yesterday = (time_yesterday>baseline_start_time) & (time_yesterday<baseline_end_time)
    
    for i in range(avg_traces_today.shape[1]):    
        avg_traces_today[:,i] = avg_traces_today[:,i] - np.nanmean(avg_traces_today[baseline_idx_today,i],0)
        avg_traces_yesterday[:,i] = avg_traces_yesterday[:,i] - np.nanmean(avg_traces_yesterday[baseline_idx_yesterday,i],0)
    
    
    plt.plot(time_today,avg_traces_today[:,data_dict['cni'][day_ind]],color='r')
    plt.plot(time_yesterday,avg_traces_yesterday[:,data_dict['cni'][day_ind]],color='k')


def grab_data(data_dir,mouse_ID=None):
    """
    Returns the data dictionary speicified in
    BCI_analysis.io_matlab.read_multisession_mat, but now with the mouse specified
    Parameters
    ----------
    mouse_ID : str, optional
        Water restriction ID of mouse, if not provided, user can type it in.
    data_dir : str, optional
        Folder for multi-session .mat files.

    Returns
    -------
    data_dict : dictionary
        Multi-session imaging data, output of 
        BCI_analysis.io_matlab.read_multisession_mat()

    """
    if mouse_ID==None: ## Specify Mouse 
        mouse_ID = input("What is the Mouse's Name? ")
        
    potential_files = os.listdir(data_dir)
    mouse_file = None
    for potential_file in potential_files:
        if potential_file.lower().startswith(mouse_ID.lower()) and potential_file.lower().endswith('.mat'):
            mouse_file = potential_file
    if mouse_file == None:
        print('ERROR: no appropriate file found - choose from the following: {}'.format(potential_files))
        return None
    
    mouse_path = os.path.join(data_dir,mouse_file)
    data_dict = io.io_matlab.read_multisession_mat(mouse_path) 
    print('{} loaded'.format(mouse_path))
    return data_dict

def delta_activity_plot(dataDict, dayOfInterest, baseline_correction_range = 1.5,
                        what_to_compare = 1, ):
    """
    delta_activity_plot plots a scatter plot by comparing the distances of 
    each neuron from the conditioned neuron to the average flourescent activity
    of each neuron in a session
    
    -------
    
    dataDict : Dictionary
        dictionary of data produced from experiment. Specified in 
        BCI_analysis.io_matlab.read_multisession_mat
    dayOfInterest : int
        The session of interest out of the total number of sessions
    baseline_correction_range : float
        The range of time from before the go cue
    what_to_compare : int
        Specifies what computation to compare between days
        1 = compares the average of peaks between days
        2 = compares the max of peaks bewteen days
        3 = compares the integrals of peaks between days
        
    -------
    Returns
    
    Scatter Plot that represents the distance of each neuron from the 
    conditioned neuron against the change in flourescence across specified session
    and previous session
    
    """
    
    ## Setting day and day before to be an integer
    day = int(dayOfInterest) + 1 #this is why you shouldn't do day 1
    dayBefore = day-1
    print('Delta Activity plot for day '+dayOfInterest)
    
    ## Define Distances from CN
    distances_from_conditioned_neuron = np.array(dataDict['distance_from_conditioned_neuron'][day]) #np.array turns consecutive brackets in python (example [[],[]]) into a matlab-type array
    
    ## Define Time Epochs of Each Frame
    frame_time_stamps = dataDict['time_from_trial_start'][1]
    
    ## Establish Baseline Correction For Each Neuron
    # baseline_correction_range = float(input('Enter range of time of baseline correction in seconds from go cue (max = 2.00) '))
    baseline_range_values = sorted( t for t in frame_time_stamps if t < 0 and t > -baseline_correction_range)
    baseline_range_max_index = np.where(frame_time_stamps==max(baseline_range_values))[0][0]
    baseline_range_min_index = np.where(frame_time_stamps==min(baseline_range_values))[0][0] #for example, if 1.5 is specified, then indexes will be [10:38]
    
    
    # Calculating Flourescent Values of each neuron on a specified day
    if np.array(dataDict['f_trialwise_closed_loop'][day]).shape[0] == np.array(dataDict['f_trialwise_closed_loop'][dayBefore]).shape[0]: #need to make sure we are comparing the same number neurons between days
        
        #Setting up the average response of each neuron across the entire session
        average_neuron_response_curves = np.nanmean(np.array(dataDict['dff_trialwise_closed_loop'][day]),axis=2) #averaging the flourescent response of each neuron across all trials
        previous_average_neuron_response_curves = np.nanmean(np.array(dataDict['dff_trialwise_closed_loop'][dayBefore]),axis=2) #averaging neuron flourescent curves from previous day
        
        #average the initial "baseline_correction_range" times of flourescent responses 
        average_baseline_today = np.nanmean(np.nanmean(average_neuron_response_curves[baseline_range_min_index:baseline_range_max_index], axis =1))
        average_baseline_yesterday = np.nanmean(np.nanmean(previous_average_neuron_response_curves[baseline_range_min_index:baseline_range_max_index], axis = 1)) #averaged twice so we can repeat 1 number to subtract from everything
        
        #subtracting average neuron responses by the baseline 
        todays_baseline = np.repeat(average_baseline_today, len(average_neuron_response_curves[:]))
        yesterdays_baseline = np.repeat(average_baseline_yesterday, len(previous_average_neuron_response_curves[:]))
        
        ## The baseline correction
        todays_baseline_correction = []
        previous_baseline_correction = []
        for i in range(len(average_neuron_response_curves[1,:])): 
            curve = average_neuron_response_curves[:,i] 
            todays_baseline_correction.append(np.subtract(curve,todays_baseline)) 
        for i in range(len(previous_average_neuron_response_curves[1,:])):
            curve = previous_average_neuron_response_curves[:,i]
            previous_baseline_correction.append(np.subtract(curve, yesterdays_baseline))
         
        todays_baseline_correction = np.array(todays_baseline_correction)[:,40:160]#hardcoded [40:60] similarly to in matlab script
        previous_baseline_correction = np.array(previous_baseline_correction)[:,40:160]#hardcoded [40:60] similarly to in matlab script
        # what_to_compare = input('What do you want to compare? (1 =  averages in peaks, 2 = summs of peaks, 3 = integrals of peaks)')
        if int(what_to_compare) == 1:
            difference_in_f = np.mean(todays_baseline_correction,axis=1)-np.mean(previous_baseline_correction,axis=1)
        else:
            print('Ending Analysis until further notice...') # Need to ask about other possible ways to compare flourescent intensities
            time.sleep(1.5)
            exit()
    else:
        print('Not comparing the same amount of neurons between days...')
        time.sleep(1.5)
        exit() #temporarily leaving this here...
    
    ## Showing Conditioned Neuron as Blue When Plotted
    conditioned_neuron_index = dataDict['conditioned_neuron_idx'][day] # calls the index of the conditioned neuron from the dictionary
    conditioned_neuron_location = distances_from_conditioned_neuron[conditioned_neuron_index] #this must be specified because the conditioned neuron still shows up as being >0 pixels away from itself

    ## and... SCATTER! 
    xAx = 'Pixels'
    yAx = 'Difference In Flourescence From ' + dayOfInterest + ' to ' + str(dayBefore)
    plt.plot(distances_from_conditioned_neuron,difference_in_f, 'o', markeredgecolor = 'grey', markerfacecolor = 'white')
    plt.plot(conditioned_neuron_location, difference_in_f[conditioned_neuron_index], 'o', markeredgecolor = 'blue', markerfacecolor = 'blue')
    plt.xlabel(xAx)
    plt.ylabel(yAx)
    plt.title('Distance from Conditioned Neuron Against Change in Average Flourescence Across Session')
    plt.legend('Neurons', 'Conditioned Neuron')
    plt.show()
    
    
    
    
def mean_ROI(dataDict, dayOfInterest):
    """
    Parameters
    ----------
    dataDict : dict
        dictionary of data produced from experiment. Specified in 
        BCI_analysis.io_matlab.read_multisession_mat
    dayOfInterest : int
        The session of interest out of the total number of sessions

    Returns
    -------
    An iamshow plot that acts as a heat map of the ROI, more yellow means more
    average activity during session of interest

    """
    # dayOfInterest = input('Day of interest (day 1 = -1... dont do day 1): ')
    day = int(dayOfInterest) + 1 #this is why you shouldn't do day 1
    print('Interested in day '+dayOfInterest)
    F_arr = np.array(dataDict['mean_image'][day])
    plt.imshow(F_arr)
    plt.colorbar()
    plt.title('Mean ROI On Day ' + dayOfInterest)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    
    
    
def plot_trialwise_closed_loop(dataDict, dayOfInterest):
    """
    Parameters
    ----------
    dataDict : Dict
        dictionary of data produced from experiment. Specified in 
        BCI_analysis.io_matlab.read_multisession_mat
    dayOfInterest : Int
        The session of interest out of the total number of sessions

    Returns
    -------
    plot of the neural activity for the AVERAGE TRIAL
    for each neuron in a session, more yellow means more activity, y axis is 
    neurons x axis is time

    """
    # dayOfInterest = input('Day of interest (day 1 = -1... dont do day 1): ')
    day = int(dayOfInterest) + 1 #this is why you shouldn't do day 1
    print('Interested in day '+dayOfInterest)
    F_arr = np.array(dataDict['f_trialwise_closed_loop'][day]) #np.array turns consecutive brackets in python (example [[],[]]) into a matlab-type array
    F = np.nanmean(F_arr, axis = 2)
    ## Plot
    plt.imshow(F.T)
    plt.colorbar()
    plt.title('Neuron Flourescent Intensity Averaged Across All Trials On Day ' + dayOfInterest)
    plt.xlabel('Frames') #translate frames to seconds eventually using frame rate (~20Hz?)
    plt.ylabel('Neuron')
    plt.show()


def correlation_deltaActivity(dataDict, dayOfInterest):
    ## Set Up Day of Session
    # dayOfInterest = input('Day of interest (day 1 = -1... dont do day 1): ')
    if int(dayOfInterest) == 0:
        while int(dayOfInterest) <= 0: #this is so we dont get stupid values
            print('Incorrect Day Entry, redo please')
            dayOfInterest = input('Day of interest (Day 1 = "1") ')
    day = int(dayOfInterest) - 1 #Since python starts at 0, we are subtracting one to account for indexing
    print('Interested in day '+dayOfInterest)


    #Grab Day-Specific Data
    today_dff_allTrials = dataDict['dff_trialwise_closed_loop'][day]
    yesterday_dff_allTrials = dataDict['dff_trialwise_closed_loop'][day-1]

    if 'conditioned_neuron_idx' in dataDict.keys(): #some sessions dont have a conditioned neuron
        todays_conditioned_neuron_index = dataDict['conditioned_neuron_idx'][0][day] #this is the 'cn' field in mat file
        # yesterdays_conditioned_neuron_index = dataDict['conditioned_neuron_idx'][0][day-1]#this is the 'cn' field in mat file
    else:
        print('Some Sessions do not include conditioned neurons -- cannot use dataset')
        quit()

    #Average All Trials Together Including CNs
    avg_trial_dff_today = np.nanmean(today_dff_allTrials, axis=2)
    avg_trial_dff_yesterday = np.nanmean(yesterday_dff_allTrials, axis=2)

    #Pick Out the Avg CN
    avg_cn_today = avg_trial_dff_today[:,todays_conditioned_neuron_index]
    # avg_cn_yesterday = avg_trial_dff_yesterday[:, yesterdays_conditioned_neuron_index]

    #Correlate Each Avg Neuron dff to the Avg CN dff
    todays_correlation = [cor(avg_trial_dff_today[:,i], avg_cn_today)[0] for i in range(avg_trial_dff_today.shape[1])] #0 is the right index for corr because when corr conditioned with itself it should end up being nearly ~1
    #length of todays correlation should be the number of neurons in session, ex: BCI11 has 245 neurons for day 3
    todays_correlation.sort() #dont forget to sort to make sure its all in order!


    #Bin All Correlated Values into 5 bins
    binning_size = len(todays_correlation)//5 
    corr_bin1 = todays_correlation[0:binning_size]
    corr_bin2 = todays_correlation[(binning_size+1):binning_size*2]
    corr_bin3 = todays_correlation[(binning_size*2)+1:binning_size*3]
    corr_bin4 = todays_correlation[(binning_size*3)+1:binning_size*4]
    corr_bin5 = todays_correlation[(binning_size*4)+1:len(todays_correlation)]

    #Averaging Each corr bin
    avg_bin1 = np.nanmean(corr_bin1)
    avg_bin2 = np.nanmean(corr_bin2)
    avg_bin3 = np.nanmean(corr_bin3)
    avg_bin4 = np.nanmean(corr_bin4)
    avg_bin5 = np.nanmean(corr_bin5)


    X = [avg_bin1, avg_bin2, avg_bin3, avg_bin4, avg_bin5] #this will be the 5 bins of our x axis


    # Now lets go back and subtract avg dff of today from the avg dff of yesterday
    if len(avg_trial_dff_today[1,:]) == len(avg_trial_dff_yesterday[1,:]):
        difference_in_dff = np.subtract(avg_trial_dff_today, avg_trial_dff_yesterday) #this is basically doing today - yesterday
        #then lets average each neuron to one value per neuron for easier plotting
        avg_delta_dff = np.nanmean(difference_in_dff,axis=0)
    else:
        print('Error: You are attempting to compare two days with different ROIs')
        quit()

    # Now we shall bin each delta_dff
    delta_bin1 = avg_delta_dff[0:binning_size]
    delta_bin2 = avg_delta_dff[(binning_size+1):binning_size*2]
    delta_bin3 = avg_delta_dff[(binning_size*2)+1:binning_size*3]
    delta_bin4 = avg_delta_dff[(binning_size*3)+1:binning_size*4]
    delta_bin5 = avg_delta_dff[(binning_size*4)+1:binning_size*5]

    #now lets average each delta_dff bin
    avg_dff_bin1 = np.nanmean(delta_bin1)
    avg_dff_bin2 = np.nanmean(delta_bin2)
    avg_dff_bin3 = np.nanmean(delta_bin3)
    avg_dff_bin4 = np.nanmean(delta_bin4)
    avg_dff_bin5 = np.nanmean(delta_bin5)
    #Calculate Error of each avg dff bin
    err1 = np.std(delta_bin1)/(np.sqrt(len(delta_bin1)))
    err2 = np.std(delta_bin2)/(np.sqrt(len(delta_bin2)))
    err3 = np.std(delta_bin3)/(np.sqrt(len(delta_bin3)))
    err4 = np.std(delta_bin4)/(np.sqrt(len(delta_bin4)))
    err5 = np.std(delta_bin5)/(np.sqrt(len(delta_bin5)))

    Y = [avg_dff_bin1, avg_dff_bin2, avg_dff_bin3, avg_dff_bin4, avg_dff_bin5]
    Yerr = [err1, err2, err3, err4, err5]

    plt.errorbar(X, Y, yerr=Yerr)
    plt.ylabel(' Day2 Corr - Day1 Corr ')
    plt.xlabel(' Average Correlations with Conditioned Neuron ')
    plt.title(' Delta_Activity Against Correlation with CN ')
    plt.show()