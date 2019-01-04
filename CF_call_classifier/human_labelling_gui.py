#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Script which allows a human to label the simulated data 
that allows checking if the simulated data matches what we expect. 

Created on Fri Jan  4 18:02:55 2019

@author: tbeleyur
"""

from __future__ import division

import matplotlib.pyplot as plt 
plt.rcParams['agg.path.chunksize'] = 10000
from matplotlib.widgets import TextBox, RadioButtons
import numpy as np 
import pandas as pd
from sklearn.metrics import confusion_matrix

from make_CFcall_training_data import generate_audio_snippet

import time 
np.warnings.filterwarnings('ignore')

from make_CFcall_training_data import calculate_features_fft_style, generate_audio_snippet
from make_CFcall_training_data import make_9number_to_class_converter

ninenum_to_2letterclass = make_9number_to_class_converter()
number_from_2leters = {entry: key  for key, entry in ninenum_to_2letterclass.iteritems()}


def gen_random_situation():
    situation = np.random.choice(['0','1','m'],1).tolist() + np.random.choice(['0','1','m'],1).tolist()+np.random.choice(['0','1'],1).tolist()
    situation_name =''.join(situation)
    return(situation_name)

human_labelled = []
generated_labels = []

first_audio_situation = gen_random_situation()
generated_labels.append(first_audio_situation)
fig, ax = plt.subplots()
ax.specgram(generate_audio_snippet(first_audio_situation), Fs=250000)
plt.subplots_adjust(left=0.3)

axcolor = 'lightgoldenrodyellow'
rax = plt.axes([0.02, 0.6, 0.15, 0.25], facecolor=axcolor)
radio = RadioButtons(rax, ('00','01','0m',
                           '10','m0',
                           '11', '1m',
                           'm1', 'mm','pass'))

start_confusion = False
def hzfunc(label):
    global start_confusion

    if len(human_labelled) <= 50:
        human_labelled.append(label)
        situation = np.random.choice(['0','1','m'],1).tolist() + np.random.choice(['0','1','m'],1).tolist()+np.random.choice(['0','1'],1).tolist()
        situation_name =''.join(situation) 
        generated_labels.append(situation_name[:-1])
        rand_audio = generate_audio_snippet(situation_name)
        vmin_val = 20*np.log10(np.max(rand_audio)) - 80
        ax.cla()
        ax.specgram(rand_audio, Fs=250000,
                NFFT=512, noverlap=256, vmin = vmin_val)
        plt.draw()
    else:
        ax.cla()
        rax.cla()
       
        start_confusion = True
        
        

radio.on_clicked(hzfunc)

while start_confusion:
    ## do the analyses 
    cf_matrix = confusion_matrix(human_labelled, generated_labels[:-1],
                                 labels=number_from_2leters.keys())
    plt.figure()
    plt.imshow(cf_matrix, aspect='equal', cmap=plt.cm.get_cmap('viridis',18))
    outer=8.5;
    plt.vlines(np.arange(0.5,outer),-0.5,outer)
    plt.hlines(np.arange(0.5,outer),-0.5,outer)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(np.arange(0,9), number_from_2leters.keys(), rotation=45)
    plt.yticks(np.arange(0,9), number_from_2leters.keys())
    cbar = plt.colorbar()
    cbar.set_ticks(np.arange(0,np.max(cf_matrix),2))
    cbar.set_ticklabels(np.arange(0,np.max(cf_matrix),2))

    
