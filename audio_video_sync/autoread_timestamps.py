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
    
    timestamp_border = (550, 50, 70, 990) # left, up, right, bottom
    led_border = (867,1020,40,30)
    end_frame : optional. End frame for reading the timestamp + LED signal
    start_frame : optional. can get the timestamp reading to start from any arbitrary points
    '''
    video = cv2.VideoCapture(video_path)

    print('starting frame reading')
    timestamps = []
    led_intensities = []
    
    start_frame = kwargs.get('start_frame',1)
    end_frame = kwargs.get('end_frame', int(video.get(cv2.CAP_PROP_FRAME_COUNT))+1)    
    video.set(1, start_frame)
    for i in  range(start_frame, end_frame):
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

def get_lamp_and_timestamp(each_img, **kwargs):
    '''
    '''
    try:
        im = Image.fromarray(each_img)
        # CROP THE TIMESTAMP OUT
        cropped_img = ImageOps.crop(im, kwargs['timestamp_border']).resize((1600,200))
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
    video_path = 'video/DVRecorder_03_20190704_16.49.45-16.56.42[R][@da37][0].avi'
    kwargs={'led_border':(414,800,490,240), 'timestamp_border':(550, 50, 70, 990),
            'start_frame':2000,'end_frame':2500}
        
#    video = cv2.VideoCapture(video_path)
#    video.set(1,3591)
#    success, frame = video.read()
#    plt.figure()
#    plt.imshow(frame)
    #    plt.ginput(n=4)
        
    ts, intensity = get_data_from_video(video_path, **kwargs)
    df = pd.DataFrame(data=[], index=range(1,len(ts)+1), 
                      columns=['frame_number','led_intensity',
                               'timestamp','timestamp_verified'])
    df['led_intensity'] = intensity
    df['timestamp'] = ts
    print('It took:', time.time()-start)
    video_filename = os.path.split(video_path)[-1]
    df.to_csv('LED_and_timestamp_from2000_to2500_'+video_filename+'_.csv', encoding='utf-8')
    
    

    