# -*- coding: utf-8 -*-
"""Systematic attempts at getting the AV sync in place

Created on Wed Jun 26 13:50:37 2019

@author: tbeleyur
"""
import glob
import multiprocessing
import time
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
import numpy as np 
import peakutils as peak
import scipy.signal as signal 
import soundfile as sf



def get_audio_sync_signal(sync_channel, min_durn=0.08,fs=250000, **kwargs):
    '''Creates a square type wave that takes on 3 values: -1,0,1.
    The pulse durations (ON/OFF) is calculated by the time gap between 
    adjacent peaks. If the pulse duration is < min_durn, then this pulse
    is set to 0. If the pulse duration is >= min_durn then it is set to 
    either +1/-1 depending on which type of pulse it is. 

    Parameters
    ----------
    
    sync_channel : array-like. 1x Nsamples

    min_durn : float>0.
               Any pulses that are below min_durn will be set to a 
               bunch of zeros of equivalent duration. Defaults to 0.08 sec.

    fs : int >0
        Frequency of sampling in Hz. Defaults to 250 kHz.

    Keyword Arguments
    -----------------

    parallel : Boolean. 
               If True then the peak finding is done with pool.map 
               , else it is done serially with plain map syntax. 

    Returns
    --------
    reassembled_signal : 1 x Nsamples np.array. 
                        The reassembled set of ON/OFF pulses with 
                        0 being the 'suppressed' pulses. 
                        Valid ON pulses have +1 value, 
                        and valid OFF pulses have -1 value. 

    *Note*
    ----
    When running the script in a visual editor like spyder it's best to 
    disable the parallel processing as everything gets stuck...
    '''
    sync = sync_channel.copy()
    # normalise the sync channel
    sync *= 1/np.max(sync)
    
    # identify positive and negative peaks 
    # isolate the peaks into positive and negative peaks 
    pos_sync = sync.copy()
    pos_sync[sync<0] = 0
    neg_sync = sync.copy()
    neg_sync[sync>0] = 0
    # get indices of positive and negative peaks 

#    pos_peaks = peak.indexes(pos_sync, 0.5, min_distance)
#    neg_peaks = peak.indexes(abs(neg_sync), 0.5, min_distance)

    
    print('begin peak  processing')
    
    
    if kwargs['parallel']:
        pool = multiprocessing.Pool(2)
        pos_peaks, neg_peaks = pool.map(get_peaks_parallelised, 
                                            [pos_sync, neg_sync])
    else:
        pos_peaks, neg_peaks = map(get_peaks_parallelised, 
                                            [pos_sync, neg_sync])
        
    print('peak finding processing done')

    all_peaks = np.concatenate((neg_peaks, pos_peaks))
    
    all_peaks_argsort = np.argsort(all_peaks)
    sorted_peaks = all_peaks[all_peaks_argsort]
    multiply_by_minus1 = np.isin(sorted_peaks, neg_peaks)
    sorted_peaks[multiply_by_minus1] *= -1 # tag the off with a -1 

    # now go peak to peak and figure out if its on or off:
    first_peak_is_rising = sorted_peaks[0] > 0
    if first_peak_is_rising:
        first_segment = np.zeros(abs(sorted_peaks[0]))
    else:
        first_segment = np.ones(abs(sorted_peaks[0]))
    
    many_segments = []
    many_segments.append(first_segment)
    for one_peak, adjacent_peak in zip(sorted_peaks[:-1],sorted_peaks[1:]):
        pulse_is_on, length = get_pulse_state(one_peak, adjacent_peak)
        pulse_segment = make_onoff_segment(pulse_is_on, length)
        many_segments.append(pulse_segment)
        
    # check if the last peak:
    last_peak_is_rising = sorted_peaks[-1] > 0
    nsamples_lastsegment = int(sync.size - np.abs(sorted_peaks[-1]))
    if last_peak_is_rising:
        last_segment = np.ones(nsamples_lastsegment)
    else:
        last_segment = np.zeros(nsamples_lastsegment)
    many_segments.append(last_segment)
    
        
        
    
    
#    
#    durations_proper = np.concatenate(( np.array([0]),sorted_peaks,np.array([sync.size])))
#    pulse_durations = np.diff( abs(durations_proper)) /float(fs)
#    cumulative_durations = np.cumsum(pulse_durations)
#    reassembled_signal = np.array([])    
#
#    all_pulses = []
#    for pulse_durn, peak_type, cuml_durn in zip(pulse_durations,
#                                                np.append(sorted_peaks,sorted_peaks[-1]),
#                                                          cumulative_durations):
#        if pulse_durn>=min_durn:
#            pulse = np.ones(int(pulse_durn*fs))*multiplyby(peak_type)
#        elif pulse_durn<=min_durn:
#            pulse = np.zeros(int(pulse_durn*fs))
#            
#        all_pulses.append(pulse)
#    # re-assmeble the signal after suppressing the very short pulses:
    reassembled_signal = np.concatenate(many_segments)
    
    return(reassembled_signal)

def get_pulse_state(index1, index2):
    '''index1 and index2 are two integers
    
    Parameters
    ---------
    index1, index2 : integers
                     index1 and index2 may be +ve or -ve depending on whether 
                     they are rising peaks (+ve) - or dropping peaks (-ve)
    Returns
    ------
    pulse_state: Boolean
    
    length : integer. 
             the length of the pulse state
    '''
    length = np.diff(np.abs([index1,index2]))
    
    peak_types = tuple([each_peak >0 for each_peak in [index1, index2]])
    pulse_state = pulse_state_dictionary[peak_types]
    return(pulse_state, length)


pulse_state_dictionary = {(True, False): True, (False,True): False}

def make_onoff_segment(pulse_state_ison, length):
    '''
    '''
    if pulse_state_ison:
        return(np.ones(length))
    else:
        return(np.zeros(length))


def video_sync_signal(vid_sync, fs=22.0):
    '''
    '''


def figure_out_if_ONorOFF(first_peak, second_peak):
    '''
    '''
    


def multiplyby(peaktype):
    if peaktype > 0 :
        return(1)
    else:
        return(0)

def get_peaks_parallelised(X, min_distance=int(250000*0.07*2)):
        det_peaks = peak.indexes(np.abs(X), 0.7, min_distance)
        return(det_peaks)

def make_time_axis(X,fs=250000):
    t = np.linspace(0, X.size/float(fs), X.size)
    return(t)



if __name__ == '__main__':
    durns = np.arange(0.2, 2.0, 0.08)
    ons=[];offs=[];
    for  i in range(50):
        ons.append(np.random.choice(durns,1))
        offs.append(np.random.choice(durns,1))
    
    actual_LED_onoff = []
    fs = 250000
    for on,off in zip(ons,offs):
        actual_LED_onoff.append(np.ones(int(on*fs)))
        actual_LED_onoff.append(np.zeros(int(off*fs)))
    
    full_onoff = np.concatenate(actual_LED_onoff)
    just_peaks = np.diff(full_onoff)
    reconstr = get_audio_sync_signal(just_peaks, parallel=False)
  
