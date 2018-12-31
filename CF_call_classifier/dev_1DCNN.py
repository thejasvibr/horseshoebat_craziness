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
np.random.seed(12)
import time


# load the training and data labels:
train_labels = np.load('training_data/labels_multifeatures.npy')
train_onehot = to_categorical(train_labels)
train_audio_features = np.load('training_data/features_multifeatures.npy')

# shuffle it up  
inds = np.arange(train_onehot.shape[0])
for i in xrange(1000):
    np.random.shuffle(inds)


## For audio feature  data : 




train_features_shuf = train_audio_features.copy()[inds,:,:][:-1000]
train_label_shuf = train_onehot[inds][:-1000]
test_features_shuf = train_audio_features.copy()[inds,:,:][-1000:]
test_label_shuf = train_onehot[inds][-1000:]



# normalise the data accroding to each channel:
for i in range(train_features_shuf.shape[2]):
    train_features_shuf[:,:,i] *= 1/np.max(train_audio_features[:,:,i])
    test_features_shuf[:,:,i] *= 1/np.max(train_audio_features[:,:,i])
    


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

num_epochs = 45



### 
#n2 = models.Sequential()
#n2.add(layers.Conv1D(filters=64, kernel_size=2, activation='relu',
#                    strides=1, input_shape=(train_features_shuf.shape[1],
#                                            train_features_shuf.shape[2])))
#n2.add(layers.MaxPool1D(2))
#n2.add(layers.Conv1D(filters=32, kernel_size=2,
#                    strides=1,
#                    activation='relu',
#                    ))
#n2.add(layers.MaxPool1D(2))
#n2.add(layers.Conv1D(filters=16, kernel_size=2,
#                    strides=1,
#                    activation='relu',
#                    ))
#n2.add(layers.Flatten())
#n2.add(layers.Dense(32, activation='relu'))
#n2.add(layers.Dense(18, activation='softmax'))
#n2.summary()
#n2.compile(loss='categorical_crossentropy', optimizer='adam', 
#              metrics=['accuracy'])
#





# fit networks
n_history = n.fit(train_features_shuf, train_label_shuf,
                  epochs=num_epochs, batch_size=72, validation_split=0.3)
n_histdict = n_history.history
#
#n2_history = n2.fit(train_features_shuf, train_label_shuf,
#                  epochs=num_epochs, batch_size=72, validation_split=0.3)
#
#n2_histdict = n2_history.history

plt.figure()
a1 = plt.subplot(211)
plt.plot(n_histdict['loss'], label='training loss')
plt.plot(n_histdict['val_loss'], label='validation loss')
plt.plot(n_histdict['acc'], label='training accuracy')
plt.plot(n_histdict['val_acc'], label='accuracy')
plt.legend();plt.grid();plt.ylim(0,2)
#plt.subplot(212, sharey=a1, sharex=a1)
#plt.plot(n2_histdict['loss'], label='training loss')
#plt.plot(n2_histdict['val_loss'], label='validation loss')
#plt.plot(n2_histdict['acc'], label='training accuracy')
#plt.plot(n2_histdict['val_acc'], label='validation accuracy')
#plt.grid();plt.ylim(0,2)


### checking how well they do on an independent test data set 
test_labels = np.load('training_data/labels_multifeatures_test.npy')
test_onehot = to_categorical(test_labels)
test_features = np.load('training_data/features_multifeatures_test.npy')

# normalise the test data :
for each_channel in range(5):
    test_features[:,:,each_channel] *= 1.0/np.max(train_audio_features[:,:,each_channel])

# run evaluations
score_n, acc_n = n.evaluate(test_features_shuf, test_label_shuf)
score_n_dt2, acc_n_dt2 = n.evaluate(test_features, test_onehot)

#score_n2, acc_n2 = n2.evaluate(test_features, test_onehot, batch_size=72)

print('accuracy', acc_n, acc_n_dt2)

for i in range(5):
    print('')
    print(np.max(train_features_shuf[:,:,i])-np.max(test_features[:,:,i]))
    print('')
    print(np.min(train_features_shuf[:,:,i])-np.min(test_features[:,:,i]))