#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:10:32 2018

@author: tbeleyur
"""
from __future__ import division
import unittest 
import numpy as np 
import scipy.signal as signal 
from getting_sync_pseaks import *

class TestingGettingSyncPeaks(unittest.TestCase):
    
    def setUp(self):
        ons = np.array([0.05, 0.220, 0.500])
        offs = np.array([0.130,0.320, 0.700])

        self.vid_fps = 25
        self.led_signal = np.zeros(int(np.max(offs*self.vid_fps)+int(0.1*self.vid_fps)))

        self.fs = 250000
        sampling_rate_factor = int(self.fs/self.vid_fps)
        self.sync = np.zeros(self.led_signal.size*sampling_rate_factor)

        drop_durn = 0.001
        t = np.linspace(0,drop_durn,int(self.fs*drop_durn))
        exponential = np.exp(-t*10000)

        for each_on, each_off in zip(ons,offs):
            on_sample = int(self.vid_fps*each_on)
            off_sample = int(self.vid_fps*each_off)

            self.led_signal[on_sample:off_sample] = 1 

            onsample_audio = int(self.fs*each_on)
            offsample_audio = int(self.fs*each_off)

            self.sync[onsample_audio:onsample_audio+int(self.fs*drop_durn)] = exponential
            self.sync[offsample_audio:offsample_audio+int(self.fs*drop_durn)] = exponential*-1

   
    def test_get_downslope(self):
        numsamples = 100
        t = np.linspace(0,1,numsamples)
        sq_signal = signal.square(np.sin(2*np.pi*25*t))
        # get up and down slope indices:
        diff_sig = np.diff(sq_signal)
        pos_peaks = np.argwhere(diff_sig==2)
        neg_peaks = np.argwhere(diff_sig==-2)
        
        obtained_neg_peaks = []
        for each_pospeak in pos_peaks:
            negpeak_result = get_closest_downslope(each_pospeak, neg_peaks)
            if negpeak_result is not None:
                obtained_neg_peaks.append(negpeak_result)

        valid_negpeaks = neg_peaks[neg_peaks>np.min(pos_peaks)]
        result = np.array_equal(np.array(obtained_neg_peaks), valid_negpeaks)
        self.assertTrue(result)
    
    def test_get_on_off_peaks(self):   
       
        obtained_onoff = get_on_off_peaks(self.sync, self.fs, self.vid_fps)

        matching_results = np.array_equal(obtained_onoff, self.led_signal)   
        self.assertTrue(matching_results)
        
    def test_align_audio_to_video(self):
        
        t_bigsignal = np.linspace(0,1,self.vid_fps)
        freqs = [10, 9, 8]
        duty_cycle = [0.8, 0.5, 0.1]
        # generate different square signals : 
        all_square_sigs = []
        for f, dc in zip(freqs, duty_cycle):
            raw_squaresig = signal.square(np.sin(2*np.pi*t_bigsignal*f), dc)
            raw_squaresig += 1 
            raw_squaresig *= 1/np.max(raw_squaresig)
            
            all_square_sigs.append(raw_squaresig)
        
                    
        big_LED_signal = np.concatenate((all_square_sigs[0],
                                         all_square_sigs[1],
                                         self.led_signal,
                                         all_square_sigs[2]))
        
        expected_start = t_bigsignal.size*2 -1
        expected_end =  t_bigsignal.size*2 -1 + self.led_signal.size
        
        start, stop, cc, onoff = align_audio_to_video(big_LED_signal,
                                                      self.vid_fps,
                                                      self.sync,
                                                      self.fs)
        
        matching_onoff = np.array_equal(onoff, self.led_signal)
        self.assertTrue(matching_onoff)
        
        
        
if __name__ == '__main__':
    unittest.main()
        
        
            
        
        
        