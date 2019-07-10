#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Upsample the LED signal from 04072019 to a uniform 25 fps 

Created on Wed Jul 10 17:53:30 2019

@author: tbeleyur
"""

import pandas as pd
import os 
import glob 

fname = ''
df = pd.read_csv(fname)
resampled_df = convert_to_common_fs(df, 25)

plt.figure()
plt.plot(np.array(resampled_df['led_intensity'])/np.max(resampled_df['led_intensity']),'g')
resampled_df.to_csv('uniform_sampling_rate_LED_signal.csv')
   


