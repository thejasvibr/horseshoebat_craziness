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


audio_folder  = 'audio_sync_signals/'
audio_sync_file = glob.glob(audio_folder+'*.npy')


# load ON/OFf signals from audio sync channel
audio_sync_signal = np.float32(np.load(audio_sync_file[0]))
audio_samplingrate = 250*10**3

# load the video signal and upsample it!
uniform_sampling_file = 'uniformsamplingLED_and_timestamp_DVRecorder_03_20190704_15.24.35-15.29.45[R][@d966][3].csv'
led_signal = np.float32(np.array(pd.read_csv(uniform_sampling_file)['led_intensity']))
led_samplingrate = 25.0

# resample by simple replacement. The current led intensity signal is at 25 Hz. 
# repeat every sample 10,000 times and then concatentate.
num_frames = led_signal.shape[0]
num_repeats = int(audio_samplingrate/led_samplingrate)
upsampled_led = np.ones((num_frames, num_repeats))

for framenum, each_framevalue in enumerate(led_signal):
    upsampled_led[framenum,:] *= each_framevalue

led_signal_forcrosscor = np.float32(upsampled_led.reshape(-1))

# now begin crosscorrelation of audio and video signals
print('Starting cross correlation now')
av_cc = signal.correlate(led_signal_forcrosscor, audio_sync_signal, 'same')
np.save('av_cc', av_cc)
print(av_cc.shape)

## run in local computer
avcc = np.load('av_cc.npy')

# get max of the cc and try to overlay the audio and video sync signals to 
# figure out if the alignment happened properly 
ind = np.argmax(avcc)
padding = int(ind - audio_sync_signal.size/2.0)
audio_w_padding = np.concatenate((np.zeros(padding), audio_sync_signal))*128

# overlay:
plt.figure()
plt.plot(led_signal_forcrosscor)
plt.plot(np.arange(padding, padding+audio_sync_signal.size),
         audio_sync_signal*128)



