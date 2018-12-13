# -*- coding: utf-8 -*-
"""
Created on Tue Dec 04 15:24:50 2018

@author: tbeleyur
"""

import os 
import glob 
import matplotlib.pyplot as plt
import numpy as np 
plt.rcParams['agg.path.chunksize'] = 10000
import peakutils as peak
import scipy.signal as signal 
import scipy.io.wavfile as WAV
import warnings


def get_on_off_peaks(sync_signal, fs, video_fps):
    '''Gets the LED flashing signal as recorded in a voltage signal from the Rpi. 
    
    Parameters:
    
        sync_signal : np.array. Voltage signal which is a copy of the LED flashing

        fs : integer. sampling rate of the sync_signal in Hz
    
        video_fps : integer. Recording rate of the video recording that captured the LED. 

    Returns : 

        on_off : np.array. The video frame numbers are which the on-off signal is expected to occur. +1 indicates
                 switching on, and -1 indicates switching off. 

    '''
    sync = np.float32(sync_signal.copy())
    sync *= 1.0/np.max(sync)
    
    # isolate the peaks into positive and negative peaks 
    pos_sync = sync.copy()
    pos_sync[sync<0] = 0
    neg_sync = sync.copy()
    neg_sync[sync>0] = 0
    # get indices of positive and negative peaks 
    min_distance = int(fs*0.07*2) # HARD CODED - CORRECT 
    pos_peaks = peak.indexes(pos_sync, 0.5, min_distance)
    neg_peaks = peak.indexes(abs(neg_sync), 0.5, min_distance)

    print('calculating indices for video frame rate')
    # calculate indices wrt to video fps 
    pos_peaks_time = pos_peaks/float(fs)
    neg_peaks_time = neg_peaks/float(fs)
    
    pospeaks_vidfps = np.int32(video_fps*pos_peaks_time)
    negpeaks_vidfps = np.int32(video_fps*neg_peaks_time)
    
    # create a binary on/off array to show when the light went on and off:
    rec_duration = sync_signal.size/float(fs)
    rec_samples = int(np.around(video_fps*rec_duration))
    print(rec_samples, video_fps, rec_duration)
    on_off = np.zeros(rec_samples)

    for upslope in pospeaks_vidfps:
        # get closest downslope in the available array:
        downslope = get_closest_downslope(upslope, negpeaks_vidfps)
        if downslope is not None:
            on_off[upslope:downslope] = 1 
    return(on_off)

def get_closest_downslope(upslope, all_downslopes):
    '''finds closest downslope index to the given upslope index
    '''
    try:
        relevant_downslopes = all_downslopes[all_downslopes>=upslope] 
        closest_index = np.argmin(np.abs(relevant_downslopes-upslope))
        closest_downslope = relevant_downslopes[closest_index]
        return(closest_downslope)
    except ValueError:
        return(None)
    
    


def get_start_stop(LED_signal, on_off):
    '''
    '''
    crosscor = np.correlate(LED_signal, on_off, 'same')
    peak_ind = np.argmax(crosscor)
        
    even_numsamples = np.divmod(on_off.size, 2)[1] == 0
    samples_toleftandright = int(on_off.size/2.0)

    if even_numsamples:
        start, stop = peak_ind-samples_toleftandright, peak_ind+samples_toleftandright-1
    else:
        start, stop = peak_ind-samples_toleftandright, peak_ind+samples_toleftandright
    
    return(start, stop, crosscor)   


def align_audio_to_video(LED_signal, fps, audiosync, fs):
    '''Gives the start and stop time for an audio file that has a voltage copy of the 
	LED bulb on/off pattern. 
	 
    
    Parameters:
    
        sync_signal : np.array. Voltage signal which is a copy of the LED flashing

        fs : integer. sampling rate of the sync_signal in Hz
    
        video_fps : integer. Recording rate of the video recording that captured the LED. 

    Returns : 

        audio_start : integer. Frame in the video at which the audio starts

        audio_stop : integer. Frame in the video at which the audio stops
    
    	crosscor : np.array. cross correlation of the video LED on/off signal to the 
		   obtained on/off audio sync signal. 
          
        on_off : np.array. The reconstructed On/Off signal from the input sync

    Note: A Warning is thrown if there is no real peak found !! 	

    '''
    on_off = get_on_off_peaks(audiosync, fs, fps)
    audio_start, audio_stop, crosscor = get_start_stop(LED_signal, on_off)
    # check if there's a strong peak
    first_diff = np.diff(crosscor)
    peaks = peak.indexes(first_diff, 0.9)
    
    if len(peaks) >1:
        warnings.warn('There may not be a strong peak !!')
    else:
        print('there might be a peak')

    return(audio_start, audio_stop, crosscor, on_off)  

