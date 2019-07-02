#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Re-sample each second to have the same sampling rate
Created on Tue Jul  2 14:05:59 2019

@author: tbeleyur
"""
import pandas as pd
import numpy as np 
import scipy.signal as signal 
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000



def resample_signal(input_signal, orig_fs, target_fs=25):
    '''Resamples the LED signal into the target frequency of sampling. 

    Parameters
    ----------
    input_signal : 1x Nsamples array -like

    orig_fs : int >0
              original frequency of sampling

    target_fs : int>0
                final sampling rate in Hz to which the input signal 
                must be resampled to. 
                Defaults to 25 Hz.

    Returns
    --------
    input_resampled : 1x Msamples np.array
                     resampled input signal 
  
    '''
    if orig_fs == target_fs:
        return(np.array(input_signal))
    else:
        durn = (1.0/orig_fs)*input_signal.size
        new_numsamples = int(durn*target_fs)
        input_resampled = signal.resample(np.array(input_signal), new_numsamples)
        return(input_resampled)


def resample_and_reformat(subdf, target_fs=25):
    '''Resamples a 
    '''
    if subdf.shape[0]==target_fs:
        return(subdf)
    else:
        resampled_LED_signal = resample_signal(subdf['led_intensity'],
                                           subdf.shape[0], target_fs)
        new_df = pd.DataFrame(data=[], columns=subdf.columns,
                              index=range(target_fs))
        new_df['timestamp_verified'] = subdf['timestamp_verified'].iloc[0]
        new_df['led_intensity'] = resampled_LED_signal
        return(new_df)

   
    



def convert_to_common_fs(df, common_fs=25):
    '''Changes a DataFrame with timestamps and led intensities sampled 
    with varying frequencies to a DataFrame with timestamps and columns with
    a uniformly sampled frequency. 

    Parameters
    ---------
    df : a pd.DataFrame with at least the following columns
         timestamp_verified
         led_intensity

    '''
    all_timestamps = df['timestamp_verified'].unique()
    # remove the first and last time stamps because there's no guarantee about
    # their frame rate
    valid_timestamps = all_timestamps[1:-1]
    with_validtimestamps = df['timestamp_verified'].isin(valid_timestamps)
    valid_df = df[with_validtimestamps]
    dfs_by_timestamp = valid_df.groupby('timestamp_verified')
    all_resampled_subdfs = dfs_by_timestamp.apply(lambda X : resample_and_reformat(X, common_fs))
    all_resampled_subdfs = all_resampled_subdfs.loc[:,['timestamp_verified','led_intensity']]
    return(all_resampled_subdfs)

if __name__ == '__main__':
    fname = 'LED_and_timestamp_OrlovaChukaDome_01_20180816_23.00.00-00.00.00[R][@f6b][1].avi_.csv'
    df = pd.read_csv(fname)
    resampled_df = convert_to_common_fs(df, 25)
    plt.figure()
    plt.plot(np.array(resampled_df['led_intensity'])/np.max(resampled_df['led_intensity']),'g')
    resampled_df.to_csv('uniform_sampling_rate_LED_signal.csv')
   