#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Tests for audio_video_matcher
Created on Tue Apr 30 08:55:54 2019

@author: tbeleyur
"""
from __future__ import division

import unittest

from peakutils import peak
import numpy as np 
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000


def make_onoff_durations(segment_durationlimits):
    '''mimics the LED signal I made in the RPi
    Parameters:
        segment_durationlimits : tuple. with min and max duration limits
    '''
    min_bout= 1/25.0 
    max_bout = 0.5
    all_poss_boutlengths = np.arange(min_bout, max_bout, 0.0001)

    min_durn, max_durn = segment_durationlimits
    all_poss_durations = np.arange(min_durn, max_durn, 0.04)
    # choose an overall LED signal duration:
    segment_durn = np.random.choice(all_poss_durations)
    
    onoff_durations = []
    bouts_sum = 0
    while bouts_sum < segment_durn:
        off = np.random.choice(all_poss_boutlengths)
        on = np.random.choice(all_poss_boutlengths)
        bouts_sum += off + on
        onoff_durations.append(on)
        onoff_durations.append(off)
      
    return(onoff_durations)
        
def make_binary_fromonoff(onoff_durns, fps=22):
    '''Generate an onoff with higher sampling rate and high frequency signals
    >12.5 Hz, and then downsample it to 22 Hz - which has a Nyquist frequency of
    11 Hz. 
    '''
    double_fps = fps*2
    total_durn = sum(onoff_durns)
    max_samples = int(double_fps*total_durn)
    # randomdecide whether the first value is on or off:
    fill_ones = np.random.random() < 0.5

    led_bouts = []
    for each_bout in onoff_durns:       
        numsamples = int(each_bout*double_fps)
        if numsamples == 0 :
            bout_signal = np.zeros(1)
        else:
            bout_signal = np.ones(numsamples)*fill_ones
         
        led_bouts.append(bout_signal)
        fill_ones = np.invert(fill_ones)
    led_signal = np.concatenate(led_bouts)
    if led_signal.size >= max_samples:
        led_signal = led_signal[:max_samples]

    resampled_signal = signal.resample(led_signal,
                                       int(np.around(led_signal.size/2.0)) )  
    return(led_signal, resampled_signal)

def make_spike_signal_from_onoff(onoff_durns, first_on=True, fs=250000):
    '''Generate spikes at the left and right edges of the onoff signal 
    '''
    
    tau = 2*10**-3
    t = np.linspace(0,tau,int(fs*tau))
    spike = np.exp(-3000*t)

    if first_on:
        rising = True
    else:
        rising = False

    make_pulse = {True: spike, 
                  False: spike*-1}
    whole_spike_signal = np.zeros(int(sum(onoff_durns)*fs))

    cum_time = 0
    for bout_durn in onoff_durns:
        rising = np.invert(rising)
        cum_time += bout_durn
        edge_sample = int(cum_time*fs)
        start, stop = edge_sample, edge_sample+spike.size
        try:
            whole_spike_signal[start:stop] = make_pulse[rising]
            
        except:
            pass
    return(whole_spike_signal)

def remake_LED_from_spikes(spike_signal, fs=250000, fps=22):
    '''Given a set of voltage spikes make the LED on off signal
    '''
    min_distance = (1/25.0)*fs
    skeleton_signal = np.zeros(spike_signal.size)
    
    pos_region = spike_signal>0
    pos_spikes = skeleton_signal.copy()
    pos_spikes[pos_region] = spike_signal[pos_region]
    neg_spikes = skeleton_signal.copy()
    neg_spikes[np.invert(pos_region)] = spike_signal[np.invert(pos_region)]
    
    
    positive_spikes = peak.indexes(pos_spikes, thres=0.5,
                                   min_dist = min_distance) 
    negative_spikes = peak.indexes(abs(neg_spikes), thres=0.5,
                                   min_dist = min_distance)
    # create a dictioanry with True for +ve spike, False for -ve spikes
    index_to_spiketype = {}
    for each in positive_spikes:
        index_to_spiketype[int(each)] = True 
    for each in negative_spikes:
        index_to_spiketype[int(each)] = False

    on_off = make_on_off_from_spikes(index_to_spiketype, skeleton_signal.copy())
                
    
    
    pass

def make_on_off_from_spikes(index_to_spiketype, envelope_signal):
    '''
    TODO : 
        1) do a proper study of all possible situations of 1 and mandy spikes, 
            a) what happens when tehre's a spike at the 0th or -1th index
            b) what happens if there's only one +ve/-ve spike in the snippet
            c) what happens if an empty index to spiketype dictionary is received
            d) what happens if there's only one spike
            
    Parameters:

        index_to_spiketype : dictioanry  with spike indices as keys and
                             Booleans to indicate if the spikes are positve.
                             If it's True - the spike is +ve, else it;s -ve.

        envelope_signal : 1 x Nsamples np.array. Typically an array with 0's.
    
    Returns :

        envelope_signal : 1 x Nsamples np.array. A binary array with 1/0 values
                          showing the on off envelope as expected from the LED
                          flashing 
    '''
    for i, (index, positive_spike) in enumerate(sorted(index_to_spiketype.iteritems())):
        print(i, index, positive_spike)
    
        if i == 0 and index>0:
            envelope_signal[:index] = np.ones(index)*np.invert(positive_spike)
        elif i == 0 and index==0:
            envelope_signal
            
    
    
    
    pass

if __name__ == '__main__':
    onoff = make_onoff_durations((0.5,2.0))
    orig_signal, led_signal = make_binary_fromonoff(onoff)
    make_time = lambda X, sum_durn : np.linspace(0,sum_durn,X.size)
#    plt.figure()
#    plt.plot(make_time(orig_signal, sum(onoff)), orig_signal)
#    plt.plot(make_time(led_signal, sum(onoff)), led_signal)
#    
    #### 
    spikes = make_spike_signal_from_onoff(onoff,
                                          first_on=True)
    plt.figure()
    plt.plot(make_time(led_signal, sum(onoff)), led_signal)
    plt.plot(make_time(spikes, sum(onoff)), spikes)
    plt.plot(make_time(spikes_resample, sum(onoff)), spikes_resample*10)
    plt.grid()
