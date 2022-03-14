import numpy as np
import matplotlib.pyplot as plt

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


def grabData(mouseID):
    '''
    

    Returns the data dictionary speicified in
    BCI_analysis.io_matlab.read_multisession_mat, but now with the mouse specified
    -------
    dataDict : dictionary
        description of 
    mouseID : str
        name of mouse
    '''
    
    ## Specify Mouse and load its data
    mouseID = input("What is the Mouse's Name? ")
    dataDir = 'C:\\Users\\Lucas\\BCIAnalysis\\BCI_data\\'
    mousePath = dataDir + mouseID + '_030222v8.mat'
    dataDict = bci.io_matlab.read_multisession_mat(mousePath) #currently loads file every time script is run
    return dataDict

def delta_activity_plot(dataDict, dayOfInterest, baseline_correction_range = 1.5,
                        what_to_compare = 1, ):
    import time
    '''
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
    
    '''
    
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
    '''
    

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

    '''
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
    '''
    

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

    '''
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


