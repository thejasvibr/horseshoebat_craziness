# -*- coding: utf-8 -*-
"""Module that handles getting video sync signal data
Created on Wed Jul 24 09:41:06 2019

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

def generate_videodata_from_annotations(annotations_df, **kwargs):
    '''
    
    Parameters
    ----------
    annotations_df : pandas DataFrame with at elast the following columns
                    'video_path'
    '''
    
    #generate the timestamps and led signals for each video file that 
    # has annotations
    unique_videofiles = np.unique(annotations_df['video_path'])
    
    for each_video in unique_videofiles:
        video_name = os.path.split(each_video)[-1]
        kwargs['video_name'] = video_name
        print('gettin raw video data from '+video_name+'  now....')
        get_syncdata_for_a_videofile(each_video, **kwargs)
        print('doen w getting raw video data ')


def get_syncdata_for_a_videofile(video_path,**kwargs):
    '''
    
    Parameters
    ----------
    video_annotation : pandas DataFrame row with at least the following 
                       columns. 
                       video_path : full file path to the video file

    Returns
    --------
    None 
    
    A side effect of this function is the csv which follows the naming
    convention:
        'videosync_{video_name_here}_.csv'
                 
    '''
    
    timestamps, intensity = get_data_from_video(video_path, 
                                                           **kwargs)
    df = pd.DataFrame(data=[], index=range(1,len(timestamps)+1), 
                      columns=['frame_number','led_intensity',
                               'timestamp','timestamp_verified'])
    df['led_intensity'] = intensity
    df['timestamp'] = timestamps
    df['frame_number'] = range(len(timestamps))
   
    
    df.to_csv('videosync_'+kwargs['video_name']+'_.csv',
              encoding='utf-8')


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
        if np.remainder(i,50)==0:
            print('reading '+str(i)+'th frame')
        if not successful:
            frame = np.zeros((1080,944,3))
            print('Couldnt read frame number' + str(i))

        timestamp, intensity = get_lamp_and_timestamp(frame ,**kwargs)
        timestamps.append(timestamp)
        try:
            led_intensities.append(float(intensity))
        except ValueError:
            print('Unable to read LED intensity at :', i)

    print('Done with frame conversion')
    return(timestamps, led_intensities)

def get_lamp_and_timestamp(each_img, **kwargs):
    '''
    
    Keyword Arguments
    ------------------
    timestamp_border, led_border : tuple with 4 entries
                       Defines the border area where the timestamp/led data
                       can be extracted from.

                       The number of pixels to crop in the following order:
                       to the left of, above, to the right of and below. 
    
    measure_led : function
                  A custom function to measure the led intensity of
                  the cropped patch. 
                  Defaults to np.max if not given. 
                  eg. if there are saturated patches and very dark patches in 
                  and the led intensity tends to be somewhere in between 
                  tracking the median value with np.median
                  could show when the led goes on and 
                  off. 

    bw_threshold : 1 > float >0
                  Sets the threshold for binarisation after a color image is turned to 
                  grayscale. Defaults to 0.65.
    '''
    try:
        im = Image.fromarray(each_img)
        # CROP THE TIMESTAMP OUT
        timestamp_region = kwargs.get('timestamp_border')
        cropped_img = ImageOps.crop(im, timestamp_region).resize((1600,200))
        P = np.array(cropped_img)
        P_mono = rgb2gray(P)
        
        block_size = 11
        P_bw = threshold_local(P_mono, block_size,
                                             method='mean')
        thresh = kwargs.get('bw_threshold', 0.65)
        P_bw[P_bw>=thresh] = 1
        P_bw[P_bw<thresh] = 0
        input_im = np.uint8(P_bw*255)
        
        text = pytesseract.image_to_string(Image.fromarray(input_im),
                                           config='digits')
        # calculate LED buld intensity:
        measure_led_intensity = kwargs.get('measure_led', np.max)
        led_intensity = measure_led_intensity(ImageOps.crop(im,kwargs['led_border']))
        return(text, led_intensity)
    except:
         print('Failed reading' + 'file:')
         return(np.nan, np.nan)


if __name__ == '__main__':
    df = pd.read_csv('DEV_file1.csv')
    #video_folder = '/media/tbeleyur/THEJASVI_DATA_BACKUP_3/fieldwork_2018_002/horseshoe_bat/video/Horseshoe_bat_2018-08/2018-08-16/cam01/'
    video_folder = '../field_video/'
    filename = 'OrlovaChukaDome_01_20180816_23.00.00-00.00.00[R][@f6b][1].avi'
    df['video_path'] = video_folder+filename
#
#    df2 = df.copy()
#    fname2 = 'OrlovaChukaDome_01_20180817_04.00.00-05.00.00[R][@2240][0].avi'
#    df2['video_path'] = video_folder + fname2

    kwargs= {}
    kwargs['timestamp_border'] = (550, 50, 70, 990)
    kwargs['led_border'] = (874, 1025, 45, 38)
    kwargs['end_frame'] = 7000
    video_path = video_folder+filename
    generate_videodata_from_annotations(df, **kwargs)


