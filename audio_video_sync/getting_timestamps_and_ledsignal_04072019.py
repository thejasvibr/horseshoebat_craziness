#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Reading the timestamps and the LED intensities 
for video recordings made on 04-07-2019

Created on Tue Jul  9 11:05:21 2019

@author: tbeleyur
"""
import glob
import os 
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps


from autoread_timestamps import get_data_from_video







if __name__ == '__main__':
    folder = '/media/tbeleyur/PEN_DRIVe/AV synchronisation experiment_4July2019/video/'
    file_paths = glob.glob(folder+'*.avi')
    video_path = file_paths[1]
    video_names = [os.path.split(each_video)[-1] for each_video in file_paths]
    kwargs={'border':(550, 50, 70, 990), 'print_frequency':100}
#    
#    
    frame_num = 9614
    video = cv2.VideoCapture(video_path)
    video.set(1, frame_num-1)
    _ , frame = video.read()
    
    plt.figure(1)
    plt.title('Frame number'+str(frame_num))
    plt.imshow(frame)
#    
#    plt.figure(2)
#    img = Image.fromarray(frame)
#    crop_img = ImageOps.crop(img, (425,800,944-440,1080-835))
#    plt.imshow(crop_img)
#    
#    
    
    
    led_borders_per_video = {}
    led_borders_per_video[video_names[0]] = (180, 525,727, 515)
    led_borders_per_video[video_names[1]] = (425,800,944-440,1080-835)
    led_borders_per_video[video_names[2]] = (425,800,944-440,1080-835)
    
    
    
    start = time.time()
    for video_name, each_video in zip(video_names,file_paths):
        print('processing' +video_name+'  now....')
        kwargs['led_border'] = led_borders_per_video[video_name]
        ts, intensity = get_data_from_video(each_video, **kwargs)
        df = pd.DataFrame(data=[], index=range(1,len(ts)+1), 
                          columns=['frame_number','led_intensity',
                                   'timestamp','timestamp_verified'])
        df['led_intensity'] = intensity
        df['timestamp'] = ts
        saved_file_name = folder+'LED_and_timestamp_'+video_name
            
        print('It took:', time.time()-start)
        try:
            df.to_csv(saved_file_name+'.csv',encoding = 'utf-8')
            print('SAVING ' + video_name+' FAILED...NOW TRYING TO SAVE AS PKL FILE')
        except:
            # save to pkl file just in case ! 
            with open(saved_file_name+'.pkl', 'wb') as pklfile:
                pickle.dump({'led_intensity':intensity, 'timestamp':ts}, pklfile)