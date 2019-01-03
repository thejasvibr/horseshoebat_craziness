#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Testing the Cf cocktail party 
classifier success with manually annotated data 

Created on Tue Jan  1 11:55:09 2019

@author: tbeleyur
"""
from __future__ import division
import keras 
from keras.utils import to_categorical
from keras.models import model_from_json
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize']=10000
import numpy as np 
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import soundfile as sf

from make_CFcall_training_data import calculate_snippet_features, get_categorywise_accuracy
from make_CFcall_training_data import make_snippettype_to_9number_converter, make_9number_to_class_converter

from make_CFcall_training_data import calculate_features_fft_style


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

all_features = np.zeros((labs.shape[0],200,95))
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
    all_features[i,:,:] = calculate_features_fft_style(input_snippet)[:,30:]
    # max normalise each channel :
    for each_ch in range(all_features.shape[-1]):
        all_features[i,:,each_ch] /= np.max(all_features[i,:,each_ch])

    all_classes.append(str(labs.loc[i,'class_name']))


# remove the O's and change them to zeros
classes_noOs = [ each.replace('O','0')for each in all_classes]
conv_classes_to_9 = make_snippettype_to_9number_converter()
classes_numeric_labels = map(lambda X: conv_classes_to_9[X], classes_noOs)
all_classes_onehot = to_categorical(classes_numeric_labels)
num_cols = 9 - all_classes_onehot.shape[1]
# rectify the shape to 18 columns:
if num_cols >0:
    class_labels_onehot = np.column_stack((all_classes_onehot,
                                      np.zeros((all_classes_onehot.shape[0],
                                               num_cols))))
else:
    class_labels_onehot = all_classes_onehot.copy()

#
## save the data : 
#np.save('real_data_5channel_features.npy', all_features)
#np.save('real_data_5channel_XYZ_labels.npy', all_classes)
#np.save('real_data_5channel_18class_numeric_labels.npy', all_features)

######################################### 
# load neural network and weights:
with open('CF_classifier_1jandata_9class_fftfeatures.json' ,'r') as model_json:
    n = model_from_json(model_json.read())
n.load_weights('CF_classifier1jandata_9class_fftfeatures.h5')
n.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

loss, acc = n.evaluate(all_features, class_labels_onehot)

#  generate predictions:
pred_classes =n.predict_classes(all_features)
ninenumber_tosniptype = make_9number_to_class_converter()

pred_classes_str = map(lambda X : ninenumber_tosniptype[X], pred_classes)
all_classes_str = map(lambda X : ninenumber_tosniptype[X], classes_numeric_labels)
####### ANALYSE WHERE THE NETWORK GOES WRONG:

# get the ones that are CORRECT :
correct_inds = pred_classes == classes_numeric_labels
correct_predns = np.array(all_classes)[correct_inds]
correct_classes, count_correct = np.unique(correct_predns, return_counts=True)
print(correct_classes, count_correct)

# get the ones that are wrong : 
wrong_preds_original = np.array(all_classes)[~correct_inds]
wrong_preds_classes = np.array(map(lambda X : ninenumber_tosniptype[X], pred_classes))[~correct_inds]
wrong_classes, count_wrong = np.unique(wrong_preds_original, return_counts=True)
print(wrong_classes, count_wrong)
# which combination of errors is most frequency:
combine = lambda X,Y : X+'_'+Y
error_combis = []
for a,b in zip(wrong_preds_original,wrong_preds_classes):
    error_combis.append(combine(a,b))
confusion, counts = np.unique(error_combis, return_counts=True)
print(confusion, counts)
# are any classes always being predicted wrong 
only_wrong = set(wrong_classes) - set(correct_classes)
print(only_wrong)

#  save as csv file for easier viewing:
wrong_preds_inds = ~correct_inds
wrong_predicted_labs = labs[wrong_preds_inds]
#for each_wrong_pred, each_actual in zip(wrong_preds_classes, wrong_original):
wrong_predicted_labs['actual_class']    = wrong_preds_original
wrong_predicted_labs['predicted_class'] = wrong_preds_classes

wrong_predicted_labs.to_csv('analysis_wrong_predictions_1jandata.csv')    


## get the confusion matrix : 
all_labels = np.unique([all_classes_str,pred_classes_str])
cf_matrix = confusion_matrix(all_classes_str,pred_classes_str, labels=all_labels)
plt.figure()
plt.imshow(cf_matrix, aspect='equal', cmap=plt.cm.get_cmap('terrain',18))
outer=5.5;
plt.vlines(np.arange(0.5,outer),-0.5,outer)
plt.hlines(np.arange(0.5,outer),-0.5,outer)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(np.arange(0,6), all_labels, rotation=45)
plt.yticks(np.arange(0,6), all_labels)
cbar = plt.colorbar()
cbar.set_ticks(np.arange(0,np.max(cf_matrix),4))
cbar.set_ticklabels(np.arange(0,np.max(cf_matrix),4))

cat_accuracy =  get_categorywise_accuracy(cf_matrix)
print(get_categorywise_accuracy(cf_matrix))
