#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Load a video snippet and get the time stamps from it using Pytesseract
Created on Mon Apr 29 18:00:36 2019

@author: tbeleyur
"""
import glob
import os
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
folder_path2 = 'video_frames//'
images = sorted(glob.glob(folder_path2+'OrlovaChuka*.jpg'))


def read_text_from_images(each_img):
    try:
        im = Image.open(each_img)
        border = (550, 50, 70, 990) # left, up, right, bottom
        # CROP THE TIMESTAMP OUT
        cropped_img = ImageOps.crop(im, border).resize((1600,200))
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
        led_border = (867,1020,40,30)
        led_intensity = np.max(ImageOps.crop(im,led_border))
        return(text, led_intensity)
    except:
         print('Failed reading' + 'file:' + each_img)
         return(np.nan, np.nan)
    
def separate_out(ts_and_intensity):
    timestamps = []
    intensity = []
    
    for each in ts_and_intensity:
        ts, brightness = each
        timestamps.append(ts)
        intensity.append(brightness)
    return(timestamps, intensity)

timestamp_and_intensity = map(read_text_from_images, images)
timestamps, intensity = separate_out(timestamp_and_intensity)
print(np.unique(timestamps))


print('It took '+str(time.time()-start)+' seconds ro complete'+str(len(images))+' images')
frame_date = pd.DataFrame(data=[], index=range(len(timestamps)), 
                          columns=['clip_name','timestamp','filenames',
                                   'led_intensity'])
frame_date['clip_name'] = 'miaow'
frame_date['filenames'] = images[:len(timestamps)]
frame_date['timestamp']= timestamps
frame_date['led_intensity'] = intensity
frame_date.to_csv('manyJPGs_frametimes.csv')
    

cor = pd.read_csv('manyJPGs_frametimes_checked.csv')['corrected_timestamps'][:len(timestamps)]
print(np.sum(cor!=frame_date['timestamp'].astype(str))/float(cor.shape[0]))


