#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Trying out AV sync for a random video snippet
Created on Thu Jul 18 10:24:08 2019

@author: tbeleyur
"""
import os 
import sys 
sys.path.append('..//')
import time 
import cv2
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from PIL import Image, ImageOps

from autoread_timestamps import get_data_from_video

start = time.time()
# choose a 1minute video segment 
folder_location = '../../field_video/'
file_name = 'OrlovaChukaDome_01_20180820_02.00.00-03.00.00[R][@45f1][2].avi'

video_path = folder_location+file_name

# load video and check out the LED to get its border location - uncomment onlce
# you've figured out the location 

#video = cv2.VideoCapture(video_path)
#video.set(1,400)
#success, frame = video.read()
#plt.figure()
#plt.imshow(frame)
#plt.ginput(n=4)

led_border = (438,962,488,95)

# extract LED signal and timestamps for the first 400 seconds. 
kwargs={'led_border':led_border, 'timestamp_border':(550, 50, 70, 990),
        'end_frame':400}
#        
#plt.figure()
#im = Image.fromarray(frame)
#plt.imshow(ImageOps.crop(im,(550, 50, 70, 990)))




ts, intensity = get_data_from_video(video_path, **kwargs)
df = pd.DataFrame(data=[], index=range(1,len(ts)+1), 
                  columns=['frame_number','led_intensity',
                           'timestamp','timestamp_verified'])
df['led_intensity'] = intensity
df['timestamp'] = ts
print('It took:', time.time()-start)
video_filename = os.path.split(video_path)[-1]
df.to_csv('LED_and_timestamp_to400_'+video_filename+'_.csv', encoding='utf-8')
