#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 19:42:54 2019

@author: tbeleyur
"""
from __future__ import division

import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize']=10000
from matplotlib.widgets import TextBox, RadioButtons
import numpy as np 
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import soundfile as sf

from make_CFcall_training_data import make_9number_to_class_converter

ninenum_to_2letterclass = make_9number_to_class_converter()
number_from_2leters = {entry: key  for key, entry in ninenum_to_2letterclass.iteritems()}


############# LOAD DATA and calculate features 
# load audio snippets and calculate features 
labs = pd.read_csv('manually_checked_audiolabels_cleaned.csv')
date_2_folder_paths = {
        '2018-08-16' : os.path.join('audio', '2018-08-16/'),
        '2018-08-18' : os.path.join('audio', '2018-08-18/'),
        '2018-08-19' : os.path.join('audio', '2018-08-19', 'ch1/',)
                        }
path_to_datafolder = os.path.join('/media', 'tbeleyur', 'THEJASVI_DATA_BACKUP_3',
                                  'fieldwork_2018_002', 'horseshoe_bat',
                                  )
global rec_num, all_snippets

all_snippets = []
all_classes = []
fs = 250000

for i, row in labs.iterrows():
    t_start, t_stop = row['time_start'], row['time_end']
    file_name = os.path.join(path_to_datafolder, 
                             date_2_folder_paths[row['date_recorded']],
                             row['file_name'])
    ch_num = int(row['channel_num'])
    snippet, _ = sf.read(file_name, start=int(np.round(t_start*fs)), 
                                     stop=int(np.round(t_stop*fs)))
    num_samples = 50000
    input_snippet = snippet[:,ch_num][:num_samples]
    all_classes.append(str(labs.loc[i,'class_name'][:-1]))
    all_snippets.append(input_snippet)

# now make 
human_labelled = ['' for i in range(len(all_snippets))]
rec_num = 0
fig, ax = plt.subplots()
ax.specgram(all_snippets[rec_num], Fs=250000)
plt.subplots_adjust(left=0.3)

axcolor = 'lightgoldenrodyellow'
rax = plt.axes([0.02, 0.6, 0.15, 0.25], facecolor=axcolor)
radio = RadioButtons(rax, ('00','01','0m',
                           '10','m0',
                           '11', '1m',
                           'm1', 'mm','pass'))

global start_confusionanalysis, rand_index
start_confusionanalysis = False

rand_index = np.arange(len(all_snippets))
for k in range(10):
    np.random.shuffle(rand_index)

def hzfunc(label):
    global rec_num, start_confusionanalysis, rand_index

    if rec_num < len(all_snippets):
        
        rand_index_chosen = int(rand_index[rec_num])
        rand_recording = all_snippets[rand_index_chosen]
        vmin_val = 20*np.log10(np.max(rand_recording)) -80.0
        print(vmin_val)
        ax.cla()

        ax.specgram(rand_recording, Fs=250000,
                NFFT=512, noverlap=256, vmin = vmin_val)
        plt.draw()
        human_labelled[rand_index_chosen] = label
        rec_num += 1 
    else:
        ax.cla()
        rax.cla()
        start_confusionanalysis = True

radio.on_clicked(hzfunc)

all_classes_wzeros = [entry.replace('O','0') for entry in all_classes]
if start_confusionanalysis:
    ## do the analyses 
    cf_matrix = confusion_matrix(human_labelled, all_classes_wzeros)
    plt.figure()
    plt.imshow(cf_matrix, aspect='equal', cmap=plt.cm.get_cmap('viridis',18))
    outer=8.5;
    plt.vlines(np.arange(0.5,outer),-0.5,outer)
    plt.hlines(np.arange(0.5,outer),-0.5,outer)
    plt.xlabel('Labelled this time, -80dB range')
    plt.ylabel('Previously labelled, -60dB range')
    plt.xticks(np.arange(0,9), number_from_2leters.keys(), rotation=45)
    plt.yticks(np.arange(0,9), number_from_2leters.keys())
    cbar = plt.colorbar()
    cbar.set_ticks(np.arange(0,np.max(cf_matrix),2))
    cbar.set_ticklabels(np.arange(0,np.max(cf_matrix),2))
    
    labs['-80dB_threshold'] = [ each.replace('0','O') for each in human_labelled]

    labs.to_csv('manually_checked_audiolabels_cleaned.csv')

    



