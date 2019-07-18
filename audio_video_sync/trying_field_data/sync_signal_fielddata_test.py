#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Get sync signal for  audio around 2018-08-20 2:00:00 to 2018-08-20 2:00:20
Created on Thu Jul 18 11:29:46 2019

@author: tbeleyur
"""
import sys 
sys.path.append('../audio_sync_signals/')
import soundfile as sf
import numpy as np
from av_sync import get_audio_sync_signal

audio_folder = '../../field_audio/'
audio_files = ['T0000094.WAV','T0000095.WAV','T0000096.WAV']

for audio_file in audio_files: 
    print('getting..'+audio_file)
    audio, fs = sf.read(audio_folder+audio_file)
    sync = audio[:,3]
    corrected_sync = get_audio_sync_signal(sync, **{'parallel':True})
    sf.write('corrected_sync_'+audio_file, corrected_sync, fs)
    np.save('corrected_sync'+audio_file, corrected_sync)
    print('Done with  '+audio_file)
