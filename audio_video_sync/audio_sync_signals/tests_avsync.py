# -*- coding: utf-8 -*-
"""Tests for the av_sync module
Created on Tue Jul 16 15:01:28 2019

@author: tbeleyur
"""
import unittest
import numpy as np 
import scipy.signal as signal 

from av_sync import get_audio_sync_signal


class BasicTest(unittest.TestCase):
    
    def setUp(self):
        ons = []
        offs = []
        
        durns = np.arange(0.2, 2.0, 0.08)
        for  i in range(20):
            ons.append(np.random.choice(durns,1))
            offs.append(np.random.choice(durns,1))
        
        actual_LED_onoff = []
        fs = 250000
        for on,off in zip(ons,offs):
            actual_LED_onoff.append(np.ones(int(on*fs)))
            actual_LED_onoff.append(np.zeros(int(off*fs)))
        
        self.full_onoff = np.concatenate(actual_LED_onoff)
        self.just_peaks = np.diff(self.full_onoff)
        
    
    
    def test1(self):
        
        output = get_audio_sync_signal(self.just_peaks, parallel=False)
        difference = np.unique(self.full_onoff[1:]-output)
        print(difference)
        self.assertTrue(np.array_equal(difference, np.array([0.])))
    
    def test2(self):
        pass
        
    

if __name__ == '__main__':
    unittest.main()