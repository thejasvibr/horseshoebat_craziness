#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Using a set of networks to explore what architecture works best 
for my bad synthesised data of the Cf party
Created on Fri Dec 21 09:41:50 2018

@author: tbeleyur
"""
import numpy as np 
import keras
from keras import layers
from keras import models
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import optimizers

import time


# load the training and data labels:
train_specgm_shuf = np.load('training_data/all_shuf_training_specs.npy')
train_onehot_shuf = np.load('training_data/all_shuf_training_labels.npy')

numsamples, rows, cols , nchannels = train_specgm_shuf.shape

# the first model which seemed to work 
time_start_n0 = time.time()
network = models.Sequential()
network.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(rows,cols,1)))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(32, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Conv2D(16, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
network.add(layers.Flatten())
network.add(layers.Dense(32, activation='relu'))
network.add(layers.Dense(18, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy', metrics=['acc'])

# checkpoint
filepath="network_weights/network0_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, \
                          verbose=1, mode='auto')

callbacks_list = [checkpoint, earlystop]
network_info = network.fit(train_specgm_shuf, train_onehot_shuf,
                           validation_split=0.25, epochs=120, batch_size=150,
                           callbacks=callbacks_list, )

n0_training = time.time() - time_start_n0 
print(n0_training)
    
time.sleep(10)
########### Network 2 : half the number of channels..a weird scaled down version
n2_start = time.time()
network2 = models.Sequential()
network2.add(layers.Conv2D(16,(3,3), activation='relu', input_shape=(rows,cols,1)))
network2.add(layers.MaxPooling2D((2, 2)))
network2.add(layers.Conv2D(32, (3, 3), activation='relu'))
network2.add(layers.MaxPooling2D((2, 2)))
network2.add(layers.Conv2D(16, (3, 3), activation='relu'))
network2.add(layers.MaxPooling2D((2, 2)))
network2.add(layers.Conv2D(8, (3, 3), activation='relu'))
network2.add(layers.MaxPooling2D((2, 2)))
network2.add(layers.Flatten())
network2.add(layers.Dense(32, activation='relu'))
network2.add(layers.Dense(18, activation='softmax'))

network2.compile(optimizer='rmsprop',
                loss='categorical_crossentropy', metrics=['acc'])

# checkpoint network 2 
filepath2="network_weights/network2_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint2 = ModelCheckpoint(filepath2, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, \
                          verbose=1, mode='auto')

callbacks_list = [checkpoint2, earlystop]
network2_info = network2.fit(train_specgm_shuf, train_onehot_shuf,
                           validation_split=0.25, epochs=120, batch_size=150,
                           callbacks=callbacks_list, )

n2_training = time.time() -  n2_start 
print(n2_training)

#################

# Network 3 : 
n3_start = time.time()
network3 = models.Sequential()
network3.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(rows,cols,1)))
network3.add(layers.MaxPooling2D((2, 2)))
network3.add(layers.Conv2D(16, (3, 3), activation='relu'))
network3.add(layers.MaxPooling2D((2, 2)))
network3.add(layers.Conv2D(8, (3, 3), activation='relu'))
network3.add(layers.MaxPooling2D((2, 2)))
network3.add(layers.Flatten())
network3.add(layers.Dense(32, activation='relu'))
network3.add(layers.Dense(18, activation='softmax'))

network3.compile(optimizer='rmsprop',
                loss='categorical_crossentropy', metrics=['acc'])

# checkpoint network 2 
filepath3="network_weights/network3_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint3 = ModelCheckpoint(filepath3, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, \
                          verbose=1, mode='auto')

callbacks_list = [checkpoint3, earlystop]
network3_info = network3.fit(train_specgm_shuf, train_onehot_shuf,
                           validation_split=0.25, epochs=120, batch_size=150,
                           callbacks=callbacks_list, )

n3_training = time.time() -  n3_start 
print(n3_training)
