#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Cross correlate the audio and video data for 2018-08-20 2:00
Created on Thu Jul 18 11:57:15 2019

@author: tbeleyur
"""
import glob
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
import numpy as np 
import pandas as pd
import scipy.signal as signal 
import soundfile as sf

fs = 250000.0
fps = 25.0

video_upsampling_factor = int(fs/fps)

# upsample the LED signal to the audio sampling rate 
led_df = pd.read_csv('25fps_LED_and_timestamp_to400_OrlovaChukaDome_01_20180820_02.00.00-03.00.00[R][@45f1][2].avi_.csv')[:25*5]

video_sync = np.ones((led_df.shape[0], video_upsampling_factor))

for i,each_value in enumerate(led_df['led_intensity']):
    video_sync[i,:] *= each_value

video_sync = video_sync.reshape(-1)
video_sync -= np.min(video_sync)
video_sync /= np.max(video_sync)
sf.write('ledsync_000096.WAV', video_sync, int(fs))

# load the reconstructed audio sync signal ;
multi_audio_syncs = []
for each in sorted(glob.glob('corrected_sync*3*.WAV.npy')):
    multi_audio_syncs.append(np.load(each))
multi_audio_syncs = np.concatenate(multi_audio_syncs)


# cross correlate audio and video :
av_cc = signal.correlate(multi_audio_syncs, video_sync, 'same')

ind = np.argmax(av_cc)
shift = int(np.round((ind - video_sync.size/2.0)))
print(shift/fs, ':seconds delay')
#
num_seconds = 90
num_samples = int(fs*num_seconds)
make_time = lambda X : np.linspace(0,X.size/fs, X.size)
#
plt.figure(10)
plt.plot(multi_audio_syncs[:shift+video_sync.size], label='audio sync')
plt.plot(np.arange(shift, shift+video_sync.size), video_sync, label='video sync')
