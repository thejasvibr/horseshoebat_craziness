#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Cross correlate the data from test experiments of 2019-07-04
Created on Mon Jul 15 15:01:51 2019

@author: tbeleyur
"""
import glob
import numpy as np 
import pandas as pd
import scipy.signal as signal 
import matplotlib.pyplot as plt 
plt.rcParams['agg.path.chunksize'] = 100000


make_time = lambda X, fs : np.linspace(0,X.size/float(fs),X.size)


audio_folder  = 'audio_sync_signals/'
audio_sync_file = glob.glob(audio_folder+'*07.WAV.npy')


# load ON/OFf signals from audio sync channel
audio_sync_signal = np.float32(np.load(audio_sync_file[0]))
audio_samplingrate = 500*10**3

# load the video signal and upsample it!
uniform_sampling_file = 'uniformsampling_LED_and_timestamp_from2000_DVRecorder_03_20190704_16.49.45-16.56.42[R][@da37][0].a.csv'
led_signal = np.float32(np.array(pd.read_csv(uniform_sampling_file)['led_intensity']))[275:1500]
led_signal -= np.min(led_signal)
led_signal /= np.max(led_signal)


led_samplingrate = 25.0
timestamps_and_led = pd.read_csv(uniform_sampling_file)
# resample by simple replacement. The current led intensity signal is at 25 Hz. 
# repeat every sample 10,000 times and then concatentate.
num_frames = led_signal.shape[0]
num_repeats = int(audio_samplingrate/led_samplingrate)
upsampled_led = np.ones((num_frames, num_repeats))

for framenum, each_framevalue in enumerate(led_signal):
    upsampled_led[framenum,:] *= each_framevalue

led_signal_forcrosscor = np.float32(upsampled_led.reshape(-1))

# a weird thing I'm trying .. but 
window_durn = int(0.02*audio_samplingrate)
smoothing_window  =  np.ones(window_durn)/window_durn
smoother_ledsignal = signal.convolve(led_signal_forcrosscor, smoothing_window,'same')
smoother_ledsignal /= np.max(smoother_ledsignal)
np.save('smoother_ledsignal', smoother_ledsignal)

# now begin crosscorrelation of audio and video signals
print('Starting cross correlation now')
#av_cc = signal.correlate(smoother_ledsignal[:20*audio_samplingrate], audio_sync_signal[:20*audio_samplingrate], 'same')
av_cc = signal.correlate(led_signal_forcrosscor[:20*audio_samplingrate], audio_sync_signal[:20*audio_samplingrate], 'same')
np.save('av_cc', av_cc)
print(av_cc.shape)

## run in local computer
av_cc = np.load('av_cc.npy')
smoother_ledsignal = np.load('smoother_ledsignal.npy')

# get max of the cc and try to overlay the audio and video sync signals to 
# figure out if the alignment happened properly 
ind = np.argmax(av_cc)
padding = int(ind - av_cc.size/2.0)
#
num_seconds = 10
t = np.linspace(0,num_seconds,num_seconds*audio_samplingrate)
plt.figure()
plt.plot(t,led_signal_forcrosscor[:int(num_seconds*audio_samplingrate)], label='LED signal')
plt.plot(t+ float(padding)/audio_samplingrate,audio_sync_signal[:int(num_seconds*audio_samplingrate)], label='Reconstructed Audio sync')
plt.legend()
   