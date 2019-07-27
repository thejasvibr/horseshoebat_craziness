#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Handle varying fps and set it video feed to common fps
Created on Wed Jul 24 13:03:20 2019

@author: tbeleyur
"""
import warnings
import datetime as dt
from datetime import timedelta
from dateutil.parser import parse
import time
import numpy as np
import pandas as pd
import scipy.signal as signal


def video_sync_over_annotation_block(annotation, video_sync_data, 
                                     **kwargs):
    '''Generates a long video sync signal 
    within a 'block' time window set by the annotation.
    
    This longer segment is called a 'block'.
    eg. The annotation only covers 10 frames from 2018-08-18 22:30:12:10 to
    2018-08-18 22:30:12:19  at 22Hz -- but the minimum duration for a good cross
    correlation is 5 seconds, then a minimum of at least 125 frames is taken 
    as the block. 

    This function requires the presence of the corresponding 
    video_sync_{video name}_.csv file for each annotated video.


    Parameters
    ----------
    annotation : pandas DataFrame row with at least the following columns
                start_timestamp, end_timestamp,
                start_framenumebr, end_framenumber
                annotation_id : string. Unique identifier that identifies
                                this particular annotation    

    
    timestamp_pattern : string
                       The format of the timestamps in the frames. 
                       The inputs should match those expected by
                       datetime.strptime
                       eg. see table here 
                       https://www.journaldev.com/23365/python-string-to-datetime-strptime

    video_sync_data : pandas.DataFrame with at least the 
                      following columns. 
                      
                      led_intensity
                      timestamp_verified


    Keyword Arguments
    -----------------
    target_commonfps : int >0.
                       If the number of frames in each second deviates from
                       the target_commonfps - the LED signal is resampled 
                       to match it. 
                       eg. if a second has 22 points , and the target is 
                       25 - then this second is upsampled.

    min_fps : int>0. 
              The lowest fps expected in the video
              Defaults to 20 fps. Anything lower than this is considered to be 
              unreliable for resampling to a common fps and the LED signal is
              replaced with zeros. 

    min_durn : float>0. 
              The minimum duration for a reliable cross-correlation in seconds.
              This is set by the duration over which *at least* 10 transitions will 
              have occured. 
              Defaults to 5 seconds. 
              eg. is the sync signal goes ON/OFF at between 0.08-0.5 seconds. 
              >= 10 transitions will have happened in 5 seconds. 
    
    Returns
    -------
    success : Boolean. 
               Whether the video sync signal could be brought to the common video 
               fps. True indicates a success, False indicates a failure. 
               If True , a .npy file with the common fps signal should 
               be output as a side effect. 
    '''
    # choose a block of video sync signal that is of a minimum length
    target_timestamps = make_video_sync_block(annotation, **kwargs)

    # check if there's variation in fps in the video sync block
    # and bring the video sync signal to a common fps
    videosync_commonfps, success = bring_video_sync_to_commonfps(target_timestamps,
                                                                                    video_sync_data,
                                                         **kwargs)
    if success:
        # save the common fps video sync signal 
        unique_id = str(annotation['annotation_id'])
        output_filename = 'common_fps_video_sync'+unique_id+'.csv'
        videosync_commonfps.to_csv(output_filename)
        
        print('SAVED IT ALL!!!' + unique_id)
    else:
        print(annotation)
        warnings.warn('could not bring above annotation to common fps')   
    return(success)


def make_video_sync_block(annotation,  **kwargs):
    '''Checks if the annotated behavious is of the minimum duration 
    and otherwise chooses a longer block than the annotation itself for audio 
    video sync. 
    
    If the annotated behaviour is above the minimum duration the sync_block 
    and annotated_block are the same. If the annotation_block is smaller
    than the minimum duration, then the the sync_block shares the same start, 
    but its ending is extended till the minmum duration is satisfied.

    Parameters
    -----------
    annotation : pandas DataFrame row with at least the following columns
                
                 start_timestamp, end_timestamp   : string.
                                     Timestamp needs to be
                                     in the exact same format
                                     that matches the timestamp present on
                                     the video frames.
                     
                 start_framenumber, end_framenumber : int.
                                     Frame number with the timestamp, index starting
                                     with 1. 

    Returns
    --------
    video_sync_block : dictionary with two keys 'annotation_block', 'sync_block'
                          


    Example
    --------
    If the annotated behaviour starts at the 3rd frame of 2018-09-23 22:00:01
    and ends at the 9th frame with 2019-09-23 22:01:02 then the annotation row
    must have the following values:
        start_timestamp : 2018-09-23 22:00:01
        start_framenumber : 3
        end_timestamp : 2018-09-23 22:01:02
        end_framenumber : 9

    '''
    annotation_long_enough = check_annotation_longenough(annotation,
                                                          **kwargs)
    video_sync_block  = {}
    if annotation_long_enough:
        video_sync_block['sync_block'] = annotation
    else:
        sync_block_row = annotation.copy()
        sync_block_row['end_timestamp'] = make_upto_min_duration(annotation,**kwargs)
        video_sync_block['sync_block'] = sync_block_row
    
    video_sync_block['annotation_block'] = annotation
    return(video_sync_block)


def check_annotation_longenough(annotation,  **kwargs):
    '''Check that the timegap between the start and end of an annotation
    is more than the minimum required for cross correlation , to the nearest seconds

    Keyword Arguments
    ------------------
    min_durn : float>0. 
               Minimum duration a video sync signal needs to be in seconds. 
    '''
    annotation_length = get_timegap(annotation, **kwargs)
    if 0<annotation_length<= kwargs['min_durn']:
        return(False)
    else:
        return(True)
        
def get_timegap(annotation, **kwargs):
    ''' 
    ACHTUNG!! :: If the start and end of an annotation occur within the
    same second eg. 2018-09-10 22:10:05  - then the timegap is by default 
    output as 0.0001.
    
    Keyword Arguments
    ------------------
    timestamp_pattern
    
    '''
    
    time_diff = dt.datetime.strptime(annotation['end_timestamp'], kwargs['timestamp_pattern']) - dt.datetime.strptime(annotation['start_timestamp'], kwargs['timestamp_pattern'])
    annotation_length = time_diff.total_seconds()   
    if annotation_length<=0:
        # check if the frame numbers are valid:
        framenumbers_valid = annotation['end_framenumber']>annotation['start_framenumber']
        if framenumbers_valid:
            annotation_length= 0.0001
        else:
            print(annotation['start_timestamp'],annotation['end_timestamp'] )
            raise ValueError('annotation length cannot be <0 seconds')

    return(annotation_length)
   
def make_upto_min_duration(annotation, **kwargs):
    '''
    
    Parameters
    ------------
    annotation : pd.DataFrame row
    
    Keyword Arguments
    ----------------
    min_durn

    timestamp_patter : the pattern in which the 

    Returns
    -------
    end_timestamp
    '''
    end_time = dt.datetime.strptime(annotation['start_timestamp'], kwargs['timestamp_pattern']) + timedelta(seconds=kwargs['min_durn'])
    end_timestamp = end_time.strftime(kwargs['timestamp_pattern']) 
    return(end_timestamp)
    

def bring_video_sync_to_commonfps(target_timestamps, video_sync_data,
                                  **kwargs):
    '''Bring the whole sync block into a common fps signal, and check
    to see if there are missing timestamps or many dropped frames.

    TODO:                                                
    1.Bring led signal with varying timestmaps to common fps
    2.Check for missing timestamps and raise a warning if there's 
      a missing second - and ignore this annotation  

    

    Keyword Arguments
    ----------------
    min_fps 
    
    common_fps
    

    Returns
    --------
    videosync_commonfps : pd.DataFrame with the following columns
                          led_signal_commonfps
                          annotation_block
                          
    success
    '''
    
    # check for missing timestamps between the start and end 
    timestamps_missing, absent_timestamps = check_for_missing_timestamps(target_timestamps,
                                                      video_sync_data,
                                                      **kwargs)
    
    if timestamps_missing:
        print(absent_timestamps)
        warnings.warn('missing timestamps found in video')
        return(None, False)
    #check for odd fps variation within each second 
    allabove_minfps = check_allabove_minimum_fps(target_timestamps,
                                                      video_sync_data, **kwargs)
    
    if allabove_minfps:
        syncblock_subset = extract_subset_for_syncblock(video_sync_data,
                                                            target_timestamps,
                                                            **kwargs) 

        videosync_commonfps =  convert_to_common_fps(syncblock_subset,
                                                         **kwargs)
        attach_annotation_start_and_end(syncblock_subset, target_timestamps,
                                        videosync_commonfps,
                                        **kwargs)
        return(videosync_commonfps, True)
    else:
        
        print(target_timestamps['annotation_block'])
        warnings.warn('FPS fell below the minimum required - ignoring this annotation')
        return(None, False)
        
        
        
        
        

def check_for_missing_timestamps(target_timestamps, video_sync_data,**kwargs):
    '''
    Parameters
    ----------
    target_timestamps  : dictioanry w two keys
                        key1 : 'sync_block' 
                                A pd.DataFrame row with following columns
                                'start_timestamp', 'end_timestamp', 'start_framenumber'
                                , 'end_framenumber'

                        key2 : 'annotation_block'
                                pd.DataFrame row with same columns as above

    video_sync_data : pd.DataFrame with following columns.
                    'led_intensity' : brightness of video sync signal 
                    'timestamp_verified' : the timestamp on each frame after manual verification

    Keyword Arguments
    --------------------

    '''
    syncblock_start = target_timestamps['sync_block']['start_timestamp']
    syncblock_end = target_timestamps['sync_block']['end_timestamp']
    timestamps_present = get_timestamps_in_between(video_sync_data, syncblock_start,
                                                   syncblock_end, **kwargs)

    expected_timestamps = make_timestamps_in_between(syncblock_start, 
                                                     syncblock_end,
                                                     **kwargs)
    
    matching_timestamps = set(expected_timestamps).intersection(set(timestamps_present))

    absent_timestamps = set(expected_timestamps).difference(set(timestamps_present))

    all_timestamps_match = len(matching_timestamps) == len(set(expected_timestamps))
    if all_timestamps_match:
        return(False, None)
    else:
        # convert the posix based timestamps back to human readable format
        absent_timestamps_readable = [datetime_from_posix(each,**kwargs) for each in absent_timestamps]
        return(True, absent_timestamps_readable)


def check_allabove_minimum_fps(target_timestamps,  video_sync_data, **kwargs):
    '''
    
    Keyword Arguments
    -----------------
    min_fps : int.
              minimum frames per second for the video sync signal to have 
              been digitised correctly. 
              
    '''
    syncblock_df = extract_subset_for_syncblock(video_sync_data, 
                                                target_timestamps,**kwargs)
    timestamps, num_frames = np.unique(syncblock_df['timestamp_verified'], 
                                       return_counts=True)
    frames_above_min = num_frames >= kwargs['min_fps']
    all_fps_above_min = np.all(frames_above_min)
    
    if all_fps_above_min:
        return(True)
    else:
        warnings.warn('Some timestamps below min fps!')
        print('timestamps below min fps: ', timestamps[np.invert(frames_above_min)])
        return(False)



def extract_subset_for_syncblock(video_sync_data, target_timestamps,
                                                            **kwargs):
    '''subset the video sync df for the time window of interest
    and also attach references to where the ANNOTATION BLOCK BEGINS and ENDS


    Returns
    -------
    syncblock_data : pd.DataFrame with the rows that fall within the target_timestamps.
   
    '''
    start = make_posix_time(target_timestamps['sync_block']['start_timestamp'],**kwargs)
    end = make_posix_time(target_timestamps['sync_block']['end_timestamp'], **kwargs)
    
    all_timestamps = video_sync_data['timestamp_verified'].apply(make_posix_time,
                                                                    0,**kwargs)
    relevant_rows = np.logical_and(all_timestamps>=start, all_timestamps<=end)
    
    syncblock_data = video_sync_data[relevant_rows].reset_index()
    return(syncblock_data)



def convert_to_common_fps(df, **kwargs):
    '''Changes a DataFrame with timestamps and led intensities sampled 
    with varying frequencies to a DataFrame with timestamps and columns with
    a uniformly sampled frequency. 

    Parameters
    ---------
    df : a pd.DataFrame with at least the following columns
         timestamp_verified
         led_intensity

    '''
    common_fps = kwargs['common_fps']
    dfs_by_timestamp = df.groupby('timestamp_verified')
    all_resampled_subdfs = dfs_by_timestamp.apply(lambda X : resample_and_reformat(X, common_fps))
    final_resampled_subdfs = all_resampled_subdfs[['timestamp_verified','led_intensity']].reset_index(drop=True)
    return(final_resampled_subdfs)    

def attach_annotation_start_and_end(video_sync_data, target_timestamps,
                                        videosync_commonfps,
                                        **kwargs):
    '''get relative positions of annotation block within the
    sync block 

    '''
    annotation_start = target_timestamps['annotation_block']['start_timestamp']
    start_frame = target_timestamps['annotation_block']['start_framenumber']
    annotation_end = target_timestamps['annotation_block']['end_timestamp']
    end_frame = target_timestamps['annotation_block']['end_framenumber']
    
    time_diff = dt.datetime.strptime(annotation_end, kwargs['timestamp_pattern'])- dt.datetime.strptime(annotation_start, kwargs['timestamp_pattern'])
    total_syncblock_durn = time_diff.total_seconds()

    numframes_start_second = np.sum(video_sync_data['timestamp_verified']==annotation_start)
    relative_start_infirstsecond = start_frame/float(numframes_start_second)
    
    
    numframes_end_second = np.sum(video_sync_data['timestamp_verified']==annotation_end)
    relative_time_in_lastsecond = end_frame/float(numframes_end_second)
    
    total_commonfps_frames = videosync_commonfps.shape[0]
    start_frame = int(kwargs['common_fps']*relative_start_infirstsecond)
    end_frame = int(total_commonfps_frames - kwargs['common_fps']*relative_time_in_lastsecond)
    
    rows, _ = videosync_commonfps.shape
    videosync_commonfps['annotation_block'] = np.tile(False, rows)
    videosync_commonfps.loc[start_frame:end_frame+1,'annotation_block'] = True
    return(videosync_commonfps)

def make_timestamps_in_between(start, end, **kwargs):
    '''Get timestamps in one second granularity and output POSIX tiemstamps
    for nice matching and sequence generation. 

    ACHTUNG : THIS ONLY HANDLES ONE SECOND GRANULARITY RIGHT NOW !!!!
    '''
    # generate posix timestamps between start and end times
    posix_start = make_posix_time(start, **kwargs)
    posix_end = make_posix_time(end, **kwargs)
    posix_values = np.arange(posix_start, posix_end+1)
    return(posix_values)


def get_timestamps_in_between(df, start, end, **kwargs):
    '''
    Parameters
    ----------
    df : pd.DataFrame with at least the following columns
         timestamp_verified.
         This is the DataFrame with LED signal and the timestamps

    start : string. 
            starting timestamp of the time period

    end  : string.
           end timestamps of the time period

    Keyword Arguments
    ------------------
    timestamp_pattern : string. 
                        A format to describe what the timestamps look like. 
                        This is in the form of a datetime.strptime entry eg 
                        %Y-%m-%d for year-month-day formatting etc.

    Returns
    --------
    timestamps_in_between : pd.Series?
                            An array-like/pd.Series? with all the timestamps
                            that fall within start and end in POSIX 
                            format.
    '''
    posix_start = make_posix_time(start, **kwargs)
    posix_end = make_posix_time(end, **kwargs)
    posix_verified_timestamps = df['timestamp_verified'].apply(make_posix_time,0, **kwargs)
    
    timestamps_in_window = np.logical_and(posix_verified_timestamps>=posix_start,
                                         posix_verified_timestamps<=posix_end)
    timestamps_in_between = posix_verified_timestamps[timestamps_in_window]
    
    return(timestamps_in_between)


def make_posix_time(timestamp, **kwargs):
    '''
    '''
    dt_timstamp = dt.datetime.strptime(timestamp, kwargs['timestamp_pattern'])
    posix_time = time.mktime(dt_timstamp.timetuple())
    return(posix_time)

def datetime_from_posix(posix_ts, **kwargs):
    '''thanks https://stackoverflow.com/a/3682808/4955732
    '''
    readable_datetime = dt.datetime.fromtimestamp(posix_ts).strftime(kwargs['timestamp_pattern'])
    return(readable_datetime)





def resample_and_reformat(subdf, target_fs):
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

def resample_signal(input_signal, orig_fs, target_fs):
    '''Resamples the LED signal into the target frequency of sampling. 

    Parameters
    ----------+
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



if __name__ == '__main__':
    raw_ann = pd.read_csv('DEV_file1.csv')
    ann = raw_ann[['annotation_id','start_timestamp', 'start_framenumber','end_timestamp','end_framenumber']][:5]
    video_sync_raw = pd.read_csv('videosync_OrlovaChukaDome_01_20180816_23.00.00-00.00.00[R][@f6b][1].avi_.csv')
    kwargs = {'timestamp_pattern': '%Y-%m-%d %H:%M:%S'}
    kwargs['min_fps']= 22
    kwargs['min_durn'] = 5.0
    kwargs['target_commonfps'] = 25.0
    kwargs['common_fps'] = 25
    
    Y = ann.apply(video_sync_over_annotation_block,1, video_sync_data=video_sync_raw, 
                                     **kwargs)
    
    
  