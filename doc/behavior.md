## Output of BCI_analysis.pipeline_bpod.export_pybpod_files()

There is a single .npy file for each session, which contain the same data. The sessions are in the same folder structure as the raw data: {setup}/{subject}/{session}

The content of the files are explained below. There are multiple variables in a given file. Each variable has n fields, where n is the number of trials during that session for that mouse. By default all timing is in seconds and relative to trial start (when trigger is sent to scanimage), distance is in mm, voltage in Volts, frequency in Hz, speed in mm/s (unless stated otherwise).
variables starting with var_ are the bpod variables that are set by the experimenter during the experiment. The repertoire of these variables might change as we developed the behavior.
variables starting with zaber_ are variables of the zaber arduino GUI except for the zaber_move_forward variable, that is the movements of the lickport.


### Variables explained:
- autowater_L: timing of autowater relative to trial start in seconds
- autowater_R: timing of autowater relative to trial start in seconds
- bpod_file_names: the original .csv file this trial originates from
- experimenter_name: name of experimenter specified in bpod file
- go_cue_times: time of the go cue relative to trial start in seconds
- lick_L: time of the left licks relative to trial start in seconds
- lick_R: time of the right licks relative to trial start in seconds
- residual_tiff_files: this is a struct that contains information abot the scanimage files that couldn't be paired to the trials (fields explained below)
	- residual_tiff_files.median_bpod_si_time_offset : median time offset between scanimage and bpod PC, this is used to calculate time from bpod trials below
	- residual_tiff_files.triggered: true if it is a triggered acquisition
	- residual_tiff_files.time_from_previous_trial_start: time in seconds from the start of the previous trial
	- residual_tiff_files.time_to_next_trial_start: time in seconds to the start of the next trial in time
	- residual_tiff_files.previous_trial_index: index of closest previous bpod trial in time
	- residual_tiff_files.next_trial_index: index of closest next bpod trial in time
	- residual_tiff_files.scanimage_file_names: the name of each tiff file
- reward_L: time of left reward delivery relative to trial start in seconds
- reward_R: time of right reward delivery relative to trial start in seconds
- setup_name: name of setup specified in bpod file
- subject_name: name of subject specified in bpod file
- scanimage_bpod_time_offset: time offset calculated between the bpod and scanimage computer for that particular trial in seconds - variance under 1 s is expected
- scanimage_file_names: scanimage file names associated with the trial. All scanimage files that were started during the trial are listed here
- scanimage_first_frame_offset: time in seconds from registered start trigger to the start of the first frame of the movie
- scanimage_tiff_headers: extracted scanimage tiff header for each trial
- threshold_crossing_times: timing of lickport threshold crossing from trial start in seconds
- time_to_hit: time from go cue to reward in seconds 
- trial_start_times: date and time of trial start on the bpod computer
- trial_end_times: date and time of trial end on the bpod computer
- trial_hit: true on rewarded trials, false on unrevarded trials
- trial_num: bpod trial number (it restarts from 1 every time a new session is started), this is sent to wavesurfer as a bitcode
- zaber_move_forward:timing of steps when the lickport moves forward one step, in seconds from trial start-
</br>

- var_AutoWater: true if autowater is turned on in the trial
- var_AutoWaterTimeMultiplier: valve time multiplier for autowater
- var_BaselineZaberForwardStepFrequency: forward step frequency during open loop trials. if set to 0, the lickport moves only on scanimage input
- var_Bias_expected_camera_num, var_Bias_ip, var_Bias_port_base, var_Bias_port_stride: BIAS related variables
- var_BitCode_ch_out: bitcode channel on bpod
- var_CameraFrameRate: framerate sent out by bpod to the face cameras
- var_CameraTriggerOut: channel for camera frame triggering
- var_EnforceStopLicking: if true mouse is forced to withstand licking for a var_RewardConsumeTime long time period before the next trial starts
- var_GoCue_ch: bpod channel fro go cuie
- var_ITI: inter trial interval length in seconds
- var_LickResponseTime: time in seconds available for mouse to lick the lickport once it crosses the threshold  
- var_LowActivityCheckAtTheBeginning: if true, low activity is checked at the beginning of the trial
- var_LowActivityTime: time in seconds at lowactivitycheck. This is the duration the ROI has to be silent so the trial can start.
- var_MotorInRewardZone: bpod channel that receives the reward zone signal from the zaber motor
- var_NeuronResponseTime: time in seconds available for the mouse to modulate the ROI activity and move the lickport past the threshold
- var_RecordMovies: face camera movies are triggered and recorded if set to True
- var_ResetTrial_ch_out: bpod channel for sending the zaber motor to home position
- var_ResponseEligibilityChannel: bpod channel that sends out the eligibility trace to arduino when lickport movement can start
- var_RewardConsumeTime: time in seconds granted for mouse to consume the reward
- var_RewardZoneCue_ch: bpod channel where threshold crossing TTL is received from zaber
- var_Scanimage_trial_start_ch_out: bpod channel that sends trigger out to scanimage
- var_ScanimageROIisActive_ch_in: bpod channel that receives 10Hz input from the arduino while the scanimage roi is active
- var_SoundOnRewardZoneEntry: if set to True, a sound cue is provided for the mouse when the lickport crosses the threshold
- var_StepZaberForwardManually_ch_out: bpod channel that steps the zaber motor forward by zaber_trigger_step_size microns
- var_UDP_IP_bpod, var_UDP_PORT_bpod: variables for UDP communication with scanimage
- var_ValveOpenTime_L: valve open time that determines the reward size (seconds)
- var_ValveOpenTime_R: valve open time that determines the reward size (seconds)
- var_WaterPort_L_ch_in: bpod channel where licks are recorded
- var_WaterPort_L_ch_out: bpod channel for opening valve
- var_WaterPort_L_PWM: bpod channel which sends valve opening info to wavesurfer
- var_WaterPort_R_ch_in:bpod channel where licks are recorded
- var_WaterPort_R_ch_out:bpod channel for opening valve 
- var_WaterPort_R_PWM:bpod channel which sends valve opening info to wavesurfer
- var_WhiteNoise_ch: bpod channel for white noise punishment during lowactivity time in the beginning of the trial 

- zaber_acceleration: acceleration of the zaber motor that carries the lickport (mm/s^2) 
- zaber_direction: the direction of the animal relative to the start position. "+" means that the mouse is in the positive direction relative to the home position
- zaber_limit_close: hard limit on lickport position in mm (so it won't crash in the mouse)
- zaber_limit_far: home position of the lickport position, this is where it starts in the beginning of the trial
- zaber_max_speed: maximum speed of the lickport during BCI movement
- zaber_microstep_size: microstep size of the motor in MICROMETERS - this is not the step size on a trigger from the arduino, this refers to the resolution of the motor itself
- zaber_reward_zone: start location of the reward zone (mm) 
- zaber_speed: max speed of the zaber motor that carries the lickport (mm/s)
- zaber_trigger_step_size: step size that the lickport makes on a TTL trigger from the arduino or bpod, it's in MICROMETERS
- zaber_trigger_step_time: time needed for the motor to do the zaber_trigger_step_size given the acceleration and speed parameters
