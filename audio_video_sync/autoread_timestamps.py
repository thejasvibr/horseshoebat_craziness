#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Load a video snippet and get the time stamps from it using Pytesseract
Created on Mon Apr 29 18:00:36 2019

@author: tbeleyur
"""
import os
import cv2
import glob
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageOps
from skimage.color import rgb2gray
from skimage.filters import threshold_local
start = time.time()
# now ty the same for all the frames in this video :
#folder_path2 = os.path.join('//media//','tbeleyur//','THEJASVI_DATA_BACKUP_3//',
#                            'horseshoebat_analysis//','videoanalysis//','Clipped videos_to be shared with Vivek//',
#                            '14//')



def get_data_from_video(video_path, **kwargs):
    '''
    
    Keyword Argument
    
    border = (550, 50, 70, 990) # left, up, right, bottom
    led_border = (867,1020,40,30)
    max_numframes
    '''
    video = cv2.VideoCapture(video_path)
    numframes = get_numframes_to_iterate(video, **kwargs)
        
        
    print('starting frame reading')
    timestamps = []
    led_intensities = []
    numframes = get_numframes_to_iterate(video, **kwargs)
    for i in  range(1, numframes+1):
        successful, frame = video.read()
        if np.remainder(i,10)==0:
            print('reading '+str(i)+'th frame')
        if not successful:
            frame = np.zeros((1080,944,3))
            print('Couldnt read frame number' + str(i))
        timestamp, intensity = get_lamp_and_timestamp(frame ,**kwargs)
        timestamps.append(timestamp)
        led_intensities.append(intensity)

    print('Done with frame conversion')
    return(timestamps, led_intensities)

def get_numframes_to_iterate(videocap, **kwargs):
    if 'max_numframes' in kwargs.keys():
        return(kwargs['max_numframes'])
    else:
        return(int(videocap.get(cv2.CAP_PROP_FRAME_COUNT)))
        
    



def get_lamp_and_timestamp(each_img, **kwargs):
    '''
    '''
    try:
        im = Image.fromarray(each_img)
        # CROP THE TIMESTAMP OUT
        cropped_img = ImageOps.crop(im, kwargs['border']).resize((1600,200))
        P = np.array(cropped_img)
        P_mono = rgb2gray(P)
        
        block_size = 11
        P_bw = threshold_local(P_mono, block_size,
                                             method='mean')
        thresh = 0.65
        P_bw[P_bw>=thresh] = 1
        P_bw[P_bw<thresh] = 0
        
        input_im = np.uint8(P_bw*255)
        
        
        text = pytesseract.image_to_string(Image.fromarray(input_im),
                                           config='digits')
        # calculate LED buld intensity:
        led_intensity = np.max(ImageOps.crop(im,kwargs['led_border']))
        return(text, led_intensity)
    except:
         print('Failed reading' + 'file:')
         return(np.nan, np.nan)
    
def separate_out(ts_and_intensity):
    timestamps = []
    intensity = []
    
    for each in ts_and_intensity:
        ts, brightness = each
        timestamps.append(ts)
        intensity.append(brightness)
    return(timestamps, intensity)

if __name__ == '__main__':
    start = time.time()
    video_path = 'video/OrlovaChukaDome_01_20180816_23.00.00-00.00.00[R][@f6b][1].avi'
    kwargs={'':10,'led_border':(867,1020,40,30), 'border':(550, 50, 70, 990),
            'max_numframes':2000}
    ts, intensity = get_data_from_video(video_path, **kwargs)
    df = pd.DataFrame(data=[], index=range(1,len(ts)+1), 
                      columns=['frame_number','led_intensity',
                               'timestamp','timestamp_verified'])
    df['led_intensity'] = intensity
    df['timestamp'] = ts
    print('It took:', time.time()-start)
    video_filename = os.path.split(video_path)[-1]
    df.to_csv('LED_and_timestamp_'+video_filename+'_.csv')
    
    

    