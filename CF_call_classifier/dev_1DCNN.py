#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Making a 1D CNN to detect the various cases trained with simulated data 

Created on Thu Dec 27 13:34:02 2018

@author: tbeleyur
"""
from __future__ import division
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
from sklearn.metrics import confusion_matrix
np.random.seed(12)
import time
from make_CFcall_training_data import make_18number_to_9number_converter
from make_CFcall_training_data import get_categorywise_accuracy,make_9number_to_class_converter

# load the training and data labels:
# train_labels = np.load('training_data/labels_multifeatures_2019-1-1_training.npy')
train_labels = np.load('training_data/labels_multifeatures_2019-3-1_fftfeatures_train.npy')
# convert the 18 class labels to 9 class labels. 
eighteen_to_9 = make_18number_to_9number_converter()
train_labels_9classes = map(lambda X : eighteen_to_9[X], train_labels)
train_onehot = to_categorical(train_labels_9classes)
#train_audio_features = np.load('training_data/features_multifeatures_2019-1-1_training.npy')
train_audio_features = np.load('training_data/features_multifeatures_2019-3-1_fftfeatures_train.npy')


# JUST REMOVE THE FIRST 30 kHZ:
train_audio_features = train_audio_features[:,:,30:]

# normalise each channel by its own max value for each exampl:
num_features = train_audio_features.shape[-1]
for each_example in xrange(train_audio_features.shape[0]):  
    for each_feature in range(5):
        train_audio_features[each_example,:,each_feature] /= np.max(train_audio_features[each_example,:,each_feature])

# shuffle it up  
inds = np.arange(train_onehot.shape[0])
for i in xrange(1000):
    np.random.shuffle(inds)


## split into training and test data sets

num_test = 100

train_features_shuf = train_audio_features.copy()[inds,:,:][:-num_test]
train_label_shuf = train_onehot[inds][:-num_test]

test_features_shuf = train_audio_features.copy()[inds,:,:][-num_test:]
test_label_shuf = train_onehot[inds][-num_test:]






# setup the model
n = models.Sequential()
n.add(layers.Conv1D(filters=64, kernel_size=1, activation='relu',
                    strides=1, input_shape=(train_features_shuf.shape[1],
                                            train_features_shuf.shape[2])))
n.add(layers.MaxPool1D(2))
n.add(layers.Conv1D(filters=32, kernel_size=2,
                    strides=1,
                    activation='relu',
                    ))
n.add(layers.MaxPool1D(2))
n.add(layers.Conv1D(filters=8, kernel_size=2,
                    strides=1,
                    activation='relu',
                    ))
n.add(layers.MaxPool1D(2))

n.add(layers.Flatten())
n.add(layers.Dense(32, activation='relu'))
n.add(layers.Dense(9, activation='softmax'))
n.summary()
n.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

num_epochs = 40

# fit networks
n_history = n.fit(train_features_shuf, train_label_shuf,
                  epochs=num_epochs, batch_size=72, validation_split=0.2)
n_histdict = n_history.history
#
#n2_history = n2.fit(train_features_shuf, train_label_shuf,
#                  epochs=num_epochs, batch_size=72, validation_split=0.3)
#
#n2_histdict = n2_history.history

plt.figure()
plt.plot(n_histdict['loss'], label='training loss')
plt.plot(n_histdict['val_loss'], label='validation loss')
plt.plot(n_histdict['acc'], label='training accuracy')
plt.plot(n_histdict['val_acc'], label='accuracy')
plt.legend();plt.grid()


# run evaluations
score_n, acc_n = n.evaluate(test_features_shuf, test_label_shuf)
test_predns = n.predict_classes(test_features_shuf)
#score_n2, acc_n2 = n2.evaluate(test_features, test_onehot, batch_size=72)

print('accuracy', acc_n)
##
### save mdoel 
#model_json = n.to_json()
#with open("CF_classifier_1jandata_9class_fftfeatures.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#n.save_weights("CF_classifier1jandata_9class_fftfeatures.h5")
#print("Saved model to disk")


# analyse where the model went wrong : 
ninenumber_tosniptype = make_9number_to_class_converter()
test_predns_type = map(lambda X: ninenumber_tosniptype[X], test_predns)
actual_type = map(lambda X: ninenumber_tosniptype[X], np.argmax(test_label_shuf,1))
all_labels = np.unique(actual_type)
cf_matrix = confusion_matrix(actual_type, test_predns_type,
                             labels=all_labels)
plt.figure()
plt.imshow(cf_matrix, aspect='equal', cmap=plt.cm.get_cmap('viridis',18))
outer=8.5;
plt.vlines(np.arange(0.5,outer),-0.5,outer)
plt.hlines(np.arange(0.5,outer),-0.5,outer)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(np.arange(0,9), all_labels, rotation=45)
plt.yticks(np.arange(0,9), all_labels)
cbar = plt.colorbar()
cbar.set_ticks(np.arange(0,np.max(cf_matrix),4))
cbar.set_ticklabels(np.arange(0,np.max(cf_matrix),4))

cat_accuracy =  get_categorywise_accuracy(cf_matrix)
print(get_categorywise_accuracy(cf_matrix))
