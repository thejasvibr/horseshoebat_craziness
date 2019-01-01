#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Testing the Cf cocktail party 
classifier success with manually annotated data 

Created on Tue Jan  1 11:55:09 2019

@author: tbeleyur
"""
import keras 
from keras.utils import to_categorical
from keras.models import model_from_json
import numpy as np 
import os
import pandas as pd
import soundfile as sf

from make_CFcall_training_data import calculate_snippet_features


############# LOAD DATA and calculate features 
# load audio snippets and calculate features 
labs = pd.read_csv('audio_labels_cleaned.csv')
date_2_folder_paths = {
        '2018-08-16' : os.path.join('audio', '2018-08-16/'),
        '2018-08-18' : os.path.join('audio', '2018-08-18/'),
        '2018-08-19' : os.path.join('audio', '2018-08-19', 'ch1/',)
                        }
path_to_datafolder = os.path.join('/media', 'tbeleyur', 'THEJASVI_DATA_BACKUP_3',
                                  'fieldwork_2018_002', 'horseshoe_bat',
                                  )

all_features = np.zeros((labs.shape[0],200,5))
all_classes = []
fs = 250000
for i, row in labs.iterrows():
    t_start, t_stop = row['time_start'], row['time_end']
    file_name = os.path.join(path_to_datafolder, 
                             date_2_folder_paths[row['date_recorded']],
                             row['file_name'])
    ch_num = int(row['channel_num'])
    snippet, _ = sf.read(file_name, start=int(t_start*fs), 
                                     stop=int(t_stop*fs))
    num_samples = 50000
    input_snippet = snippet[:,ch_num][:num_samples]
    all_features[i,:,:] = calculate_snippet_features(input_snippet)
    all_classes.append(str(labs.loc[i,'class_name']))
   
# make the reverse translation dictionary 
cf_bat_states = ['O','1','m']

convert_snippet_type_to_categorical = {}

i = 0 
for ferrum in cf_bat_states:
    for eume in cf_bat_states:
        for fm in ['O','1']:
            convert_snippet_type_to_categorical[ferrum+eume+fm] = i
            i+=1
all_classes_number = map(lambda X: convert_snippet_type_to_categorical[X],
                              all_classes)
classes_number_to_str = { entry: key for key, entry in convert_snippet_type_to_categorical.iteritems()}
all_classes_onehot = to_categorical(all_classes_number)
num_cols = 18 - all_classes_onehot.shape[1]
# rectify the shape to 18 columns:
if num_cols >0:
    class_labels_onehot = np.column_stack((all_classes_onehot,
                                      np.zeros((all_classes_onehot.shape[0],
                                               num_cols))))
else:
    class_labels_onehot = all_classes_onehot.copy()

######################################### 
# load neural network and weights:
with open('CF_classifier.json' ,'r') as model_json:
    n = model_from_json(model_json.read())
n.load_weights('CF_classifier.h5')
n.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

loss, acc = n.evaluate(all_features, class_labels_onehot)

#  generate predictions:
pred_classes =n.predict_classes(all_features)

####### ANALYSE WHERE THE NETWORK GOES WRONG:

# get the ones that are CORRECT :
correct_inds = pred_classes == all_classes_number
correct_predns = np.array(all_classes)[correct_inds]
correct_classes, count_correct = np.unique(correct_predns, return_counts=True)
print(correct_classes, count_correct)

# get the ones that are wrong : 
wrong_preds_original = np.array(all_classes)[~correct_inds]
wrong_preds_classes = np.array(map(lambda X : classes_number_to_str[X], pred_classes))[~correct_inds]
wrong_classes, count_wrong = np.unique(wrong_preds_original, return_counts=True)
print(wrong_classes, count_wrong)
# are any classes always being predicted wrong 
only_wrong = set(wrong_classes) - set(correct_classes)
print(only_wrong)

#  save as csv file for easier viewing:
wrong_preds_inds = ~correct_inds
wrong_predicted_labs = labs[wrong_preds_inds]
#for each_wrong_pred, each_actual in zip(wrong_preds_classes, wrong_original):
wrong_predicted_labs['predicted_class'] = wrong_preds_classes
wrong_predicted_labs['actual_class']    = wrong_preds_original
wrong_predicted_labs = wrong_predicted_labs.drop(columns=['Ferrum_call','BlEuMi_call',
                                                  'Myotis_call', 'cf_sum',
                                                  'class_name','groupsize_label'])
wrong_predicted_labs.to_csv('analysis_wrong_predictions.csv')    
