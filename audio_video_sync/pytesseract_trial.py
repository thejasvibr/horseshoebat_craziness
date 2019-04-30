#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Load a video snippet and get the time stamps from it using Pytesseract
Created on Mon Apr 29 18:00:36 2019

@author: tbeleyur
"""
import glob
import os

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageOps

file_path  = '//home//tbeleyur//Desktop//Untitled 1.png'
text = pytesseract.image_to_string(Image.open(file_path))


# now ty the same for all the frames in this video :
folder_path2 = os.path.join('//media//','tbeleyur//','THEJASVI_DATA_BACKUP_3//',
                            'horseshoebat_analysis//','videoanalysis//','Clipped videos_to be shared with Vivek//',
                            '14//')
images = sorted(glob.glob(folder_path2+'frame-*.png'))

frame_date = pd.DataFrame(data=[], index=range(len(images)),
                          columns=['frame_number','timestamp','clip_name'])

for i,each_img in enumerate(images):
    im = Image.open(each_img)
    border = (550, 50, 70, 990) # left, up, right, bottom
    cropped_img = ImageOps.crop(im, border).resize((1000,150))
    P = np.array(cropped_img)
    P_mono = np.sum(P,2)
    thresh = np.percentile(P_mono.flatten(), 90)
    P_bw = np.zeros(P.shape);P_bw[P_mono>thresh] = 1
    input_im = np.uint8(P_bw*255)
    im = Image.fromarray(input_im)
    text = pytesseract.image_to_string(cropped_img, config='digits')
    if np.remainder(i, 100)==0:
        print(i)
    frame_date['frame_number'][i] = i
    frame_date['timestamp'][i] = text

frame_date['clip_name'] = 'vlc-record-2018-10-12-14h29m16s-OrlovaChukaDome_01_20180815_02.00.00-03.00.00[R][@75e][0].avi-.mp4'
frame_date.to_csv('manyPNGs_frametimes.csv')
    
    
