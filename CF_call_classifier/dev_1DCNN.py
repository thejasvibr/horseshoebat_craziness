#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Making a 1D CNN to detect the various cases.

Created on Thu Dec 27 13:34:02 2018

@author: tbeleyur
"""
import matplotlib.pyplot as plt 
plt.rcParams['agg.path.chunksize'] = 10000
import numpy as np 
import keras
from keras import layers
from keras import models
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import optimizers
np.random.seed(121)
import time


# load the training and data labels:
#train_audio = np.load('training_data/sim_audio_snippets.npy', mmap_mode='r')
#train_onehot = np.load('training_data/sim_audio_labels.npy',  mmap_mode='r')
train_labels = np.load('training_data/labels_multifeatures.npy',  mmap_mode='r')
train_onehot = to_categorical(train_labels)
train_audio_features = np.load('training_data/features_multifeatures.npy', mmap_mode='r')

# shuffle it up  
inds = np.arange(train_onehot.shape[0])
for i in xrange(1000):
    np.random.shuffle(inds)



#train_audio_shuf = np.float16(train_audio[inds,:].copy().reshape(train_audio.shape[0],
#                              train_audio.shape[1],1))
train_features_shuf = train_audio_features.copy()[inds,:,:]
## normalise all values so they are >0 
#
#train_audio_shuf += 1
#train_audio_shuf *= 0.5

train_label_shuf = train_onehot[inds]


### FOR AUDIO RAW DATA:
#
## setup the model
#n = models.Sequential()
#n.add(layers.Conv1D(filters=16, kernel_size=250,activation='relu',
#                    strides=125, input_shape=(train_audio.shape[1],1)))
#n.add(layers.Conv1D(filters=16, kernel_size=125,
#                    strides=64,
#                    activation='relu',
#                    ))
#n.add(layers.MaxPool1D(2))
#n.add(layers.Flatten())
#n.add(layers.Dense(32, activation='relu'))
#n.add(layers.Dense(18, activation='softmax'))
#n.summary()
#n.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
#              metrics=['accuracy'])
#
## fit network
#n.fit(train_audio_shuf, train_label_shuf,
#      epochs=50, batch_size=25, validation_split=0.1)
#
## check out how the trained network does : 
#preds = n.predict(train_audio_shuf[:100])
#preds_argmax = np.apply_


## For audio feature  data : 

# normalise the data accroding to each channel:
for i in range(train_features_shuf.shape[2]):
    train_features_shuf[:,:,i] *= 1/np.max(train_features_shuf[:,:,i])

# setup the model
n = models.Sequential()
n.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu',
                    strides=1, input_shape=(train_features_shuf.shape[1],
                                            train_features_shuf.shape[2])))
n.add(layers.MaxPool1D(2))
n.add(layers.Conv1D(filters=16, kernel_size=2,
                    strides=1,
                    activation='relu',
                    ))
n.add(layers.MaxPool1D(2))
n.add(layers.Flatten())
n.add(layers.Dense(32, activation='relu'))
n.add(layers.Dense(18, activation='softmax'))
n.summary()
n.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

# fit network
n_history = n.fit(train_features_shuf, train_label_shuf,
                  epochs=75, batch_size=72, validation_split=0.2)
n_histdict = n_history.history




## 
n2 = models.Sequential()
n2.add(layers.Conv1D(filters=64, kernel_size=2, activation='relu',
                    strides=1, input_shape=(train_features_shuf.shape[1],
                                            train_features_shuf.shape[2])))
n2.add(layers.MaxPool1D(2))
n2.add(layers.Conv1D(filters=32, kernel_size=2,
                    strides=1,
                    activation='relu',
                    ))
n2.add(layers.MaxPool1D(2))
n2.add(layers.Conv1D(filters=16, kernel_size=2,
                    strides=1,
                    activation='relu',
                    ))
n2.add(layers.Flatten())
n2.add(layers.Dense(32, activation='relu'))
n2.add(layers.Dense(18, activation='softmax'))
n2.summary()
n2.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])
n2_history = n2.fit(train_features_shuf, train_label_shuf,
                  epochs=75, batch_size=72, validation_split=0.2)

n2_histdict = n2_history.history

plt.figure()
a1 = plt.subplot(211)
plt.plot(n_histdict['loss'], label='training loss')
plt.plot(n_histdict['val_loss'], label='validation loss')
plt.plot(n_histdict['acc'], label='training accuracy')
plt.plot(n_histdict['val_acc'], label='accuracy')
plt.legend();plt.grid();plt.ylim(0,2)
plt.subplot(212, sharey=a1, sharex=a1)
plt.plot(n2_histdict['loss'], label='training loss')
plt.plot(n2_histdict['val_loss'], label='validation loss')
plt.plot(n2_histdict['acc'], label='training accuracy')
plt.plot(n2_histdict['val_acc'], label='validation accuracy')
plt.grid();plt.ylim(0,2)


### checking how well they do on an independent test data set 
test_labels = np.load('training_data/labels_multifeatures_test.npy',  mmap_mode='r')
test_onehot = to_categorical(test_labels)
test_features = np.load('training_data/features_multifeatures_test.npy', mmap_mode='r+')

# run evaluations
score_n, acc_n = n.evaluate(test_features, test_onehot, batch_size=72)
score_n2, acc_n2 = n2.evaluate(test_features, test_onehot, batch_size=72)

print('accuracy', acc_n, acc_n2)
