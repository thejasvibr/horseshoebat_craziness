#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Module which find the appropriate audio data segment for a given 
video data segment
Created on Tue Apr 30 08:53:19 2019

@author: tbeleyur
"""

import numpy as np 
import scipy.signal as signal
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000


def get_audio_segment(video_LED, video_segment, audio_folder,
                                                          **kwargs):
    '''Given user input on start and stop of video segment - the corresponding
    time synchronised audio segment is extracted. 

    This function works by taking out the relevant LED on/off and correlating
    it with the voltage signal in all of the audio segments in the audio_folder. 
    The audio file with the highest cross correlation is chosen.

    Parameters:
        video_LED : 1 x Nsamples np.array. The LED ON/OFF signal for the video file.

        video_segment : tuple with floats. The burnt timestamps in the videofile
                          with the following format. 
                          YYYY-mm-DD HH:MM:SS:FF where 
                          YYYY - year
                          mm - month
                          DD - day
                          
                          HH - hour
                          MM - minute
                          SS - second
                          FF - frame. FF can take values between 1 and the frames
                               per second    

        audio_folder : string. The path to the relevant audio folder. 

    Keyword Arguments:

        audio_fs : integer. Frequency of sampling in Hertz of the audio system

        video_fps : integer. Frames per second of the video system in Hertz. 

        audio_sync_channel : integer. The channel to which the reference voltage
                             copy is fed. Channel numbering starts from 0.   

    Returns:
        timealigned_audio : Nchannels x samples audio array. Depending ont he 
                            dimensions of the audio.    
                                
    '''   

    segment_LED = extract_segment_LEDsignal(video_LED, video_segment)

    best_audio = crosscorrelate_LED_with_audio(segment_LED, audio_folder,
                                                   **kwargs)
    
    return(best_audio)

    