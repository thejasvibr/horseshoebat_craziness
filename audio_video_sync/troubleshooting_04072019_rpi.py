#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Try syncing the audio and video from 04-07-2019
Created on Wed Jul 10 10:31:50 2019

@author: tbeleyur
"""
import glob 
import os
import sys
sys.path.append('audio_sync_signals/')
import numpy as np 
import scipy.signal as signal 
import soundfile as sf
import matplotlib.pyplot as plt 
plt.rcParams['agg.path.chunksize'] = 10000

from av_sync import get_audio_sync_signal


folder = '/media/tbeleyur/PEN_DRIVe/AV synchronisation experiment_4July2019/'
video_folder = folder + 'video/' 
audio_folder = folder + 'audio/'

wav_files = glob.glob(audio_folder+'*.WAV')

audio, fs = sf.read(wav_files[0])
sync = audio[:,3]
file_name = os.path.split(wav_files[0])[-1]
corrected_sync = get_audio_sync_signal(sync, **{'parallel':True})        
np.save('corrected_sync_'+file_name+'.npy',corrected_sync)