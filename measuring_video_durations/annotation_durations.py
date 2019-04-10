# -*- coding: utf-8 -*-
"""Function which calculates the duration between the 
start and end of a video annotation upto millisecond resolution 

Created on Wed Apr 10 10:11:33 2019

@author: tbeleyur
"""

import warnings
import numpy as np
import pandas as pd
import datetime as dt


def calc_durations(row):
    '''Takes in a ROW of a pandas DF with the columns 
    'start_milliseconds' and 'end_milliseconds' and calculates the duration
    from these two points of time. 

    Note : 
        If the duration is <=0 a warning is thrown once and the rows are printed.

    Parameters:
        
        row : a single row pandas DataFrame with at least the following columns,
            start_milliseconds : string. The start timestamp of an annotation in 
                                HH:MM:SS.milliseconds format
            end_milliseconds : string. The end timestamp of an annotation in 
                                HH:MM:SS.milliseconds format
    Returns:

        duration : float/np.nan. The duration of time between the start and end 
                   in seconds. If the duration is >=0 then a float is returned, 
                   else a warning is thrown and np.nan is instead returned. 
    '''
    time_format = '%H:%M:%S.%f'
    start = dt.datetime.strptime(row['start_milliseconds'], time_format)
    end =  dt.datetime.strptime(row['end_milliseconds'], time_format)

    duration = (end - start).total_seconds()

    if duration <=0:
        warnings.warn('The duration of a row is <=0 !! this is not possible!')
        print(row)
        return(np.nan)
    else:
        return(duration)

if __name__ == '__main__':
    # set the folder and file paths 
    folder_path = 'C:\\Users\\tbeleyur\\Downloads\\'
    fname = 'Copy of Flight behaviour annotation_for Aditya vs Neetash_compared on 27032019_modified for R upload_in milliseconds.txt'
    # load the .txt file and set the line separator as ' '
    d = pd.read_csv(folder_path+fname, sep=' ')

    # apply the get_durations function onto each row 
    d['duration_seconds'] = d.apply(calc_durations, 1)   

    # set the destination txt file and save it 
    destination_file = 'w_durations.txt'
    d.to_csv(destination_file, sep=' ', na_rep='NA')