#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Script to generate all ten cases of the CF call classifier.

Created on Thu Dec 20 13:52:18 2018

@author: tbeleyur
"""
from __future__ import division
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
import numpy as np 
import scipy.signal as signal 

def generate_audio_snippet(snippet_type, **kwargs):
    '''Generates a snippet of given duration with the snippter type. 
    
    Parameters:
        
        snippet_type : tuplestring with three characters. Each character stands for
                       the absence/presence of a call type and the number of calling bats. 
                       
                       The first position refers to the presence of R. ferrumequinum calls. 
                       R. ferrumequinum calls can be 0(none), 1(one bat), or m (multi).
                       The second position refers to R. euryale or mehelyi calls.
                       The third refers to the position of M. myotis or FM calls. 
                       
                       eg1. 0m1 refers to a snippet with multiple bats of EuMe (Euryale/Mehelyi)
                       calling and the presence of FM calls in the snippet. 
                       
                       eg2. mm0 refers to multiple ferrumequinum and multiple EuMe bats
                       calling with no FM calls in the snippet. 
        
    Keyword arguments:
        
        snippet_duration : float >=0. Duration of the snippet in seconds. Defaults to 
                   200 milliseconds. 

    Returns:
        
        snippet: 1 x Nsamples numpy array. The audio snippet with the required
                types of bat calls. 
    '''
    ferrum, eume, fm = snippet_type 
    fs = 250000
    if 'snippet_duration' in kwargs.keys():
        snippet_duration = kwargs['snippet_duration']
    else:
        snippet_duration = 0.2

    # make ferrumequinum calls:
    ferrum_call = make_CF_call(ferrum, 'ferrum', snippet_duration, fs)
    # make eume calls:
    eume_call = make_CF_call(eume, 'eume', snippet_duration, fs)
    # make FM call:
    fm_call = make_FM_call(fm, snippet_duration, fs)

    # randomly choose a ratio to amplify/attenuate the three types of calls:
    rand_nums = np.random.choice(range(1,11),3)
    amp_ratios = rand_nums/np.sum(rand_nums)

    snippet = np.zeros(int(fs*snippet_duration))
    
    for  amp_ratio, calltype in zip(amp_ratios, [ferrum_call, eume_call, fm_call]):
        snippet += amp_ratio*ferrum_call
    return(snippet)
    

def make_FM_call(fm_in, durn, fs):
    '''Make fm call 
    
    Parameters:
        fm_in : string that is either '0' or '1'
        
    Returns:
        
    TODO:
        *1 Implement FM call generation with second harmonic in place ! 
    '''
    
    num_calls = np.random.choice(xrange(1,6),1)
    
    
    
    if fm_in is '0':
        # return zeros array:
        no_fm = make_zeros_array(durn,fs)
        return(no_fm)
    else:
        all_calls = []
        for each_call in xrange(num_calls):
            start_f = np.random.choice(xrange(50000,95001),1)
            end_f = np.random.choice(xrange(20000,40001),1)
            fm_shape = np.random.choice(['linear',
                                         'log',
                                         'hyperbolic'],1)[0]
            call_duration = np.random.choice(np.arange(0.001,0.006,0.001),1)
            window_type = np.random.choice(['hann',
                                            'triangle',
                                            'bartlett',
                                            ], 1)[0]
            
            t = np.linspace(0, call_duration, int(fs*call_duration))
            chirp = signal.chirp(t, start_f, t[-1], end_f, method=fm_shape)
           
            chirp *= signal.get_window(window_type, chirp.size)
            chirp *= np.random.choice(np.linspace(0.2,0.9,100),1)
            
            all_calls.append(chirp)
        # place each call in the snippet:
        numsamples = int(durn*fs)
        fmsnippet_locations = np.random.choice(xrange(0, numsamples), 
                                             num_calls)
        fm_snippet = np.zeros(numsamples)
        for each_call, each_position in zip(all_calls, fmsnippet_locations):
            call_samples = each_call.size
            # if the end positions of the calls can't be defined, then define the
            # position by the start of the calls.
            try:
                fm_snippet[each_position-call_samples:each_position] += each_call
            except: 
                fm_snippet[each_position:each_position+call_samples] += each_call
        
        fm_snippet *= 1/np.max(fm_snippet)
        return(fm_snippet)


def make_zeros_array(duration,fs):
    '''
    '''
    zeros_array = np.zeros(int(duration*fs))
    return(zeros_array)

def make_CF_call(numbats, cf_type, snippet_duration, fs):
    '''
    '''
    if numbats is '0':
        # return zeros array:
        noCF = make_zeros_array(snippet_duration,fs)
        return(noCF)

    elif numbats is '1':
        singleCFbat = make_singleCFbat_sequence(cf_type, snippet_duration, fs)
        return(singleCFbat)

    elif numbats is 'm' : 
        multiCFbat = np.zeros(int(snippet_duration*fs))
        num_cfbats = np.random.choice(xrange(1,4),1)
        for each_callingbat in xrange(num_cfbats):
            multiCFbat += make_singleCFbat_sequence(cf_type, snippet_duration, fs)
        
        multiCFbat *= 1/np.max(multiCFbat)
        return(multiCFbat)
            
def make_singleCFbat_sequence(cf_type, durn, fs):
    ''' Makes a Cf bat call sequence by randomly choosing a 
    call duration, call shape and other parameters. 
    
    '''
    if cf_type is 'ferrum':
        CF_range = np.arange(79000,82500,500)
    elif cf_type is 'eume':
        CF_range = np.arange(99500,106500,500)

    CF_value = np.random.choice(CF_range)
    call_shape = np.random.choice(['staplepin',
                                   'rightangle',
                                   'onlyCF'],1)[0]
    
    call_durationrange = np.arange(30,100) * 10**-3

    duty_cycle = np.random.choice(xrange(50,80),1) * 10**-2
    
    baseline_calldurn = np.random.choice(call_durationrange,1)

    approx_numcalls = np.ceil(duty_cycle*durn/baseline_calldurn)
    # set the call durations of each 
    call_durns = np.random.choice(np.arange(0,0.006,0.001),approx_numcalls) + baseline_calldurn
    # set the interpulse interval following each call
    sum_ipis = sum(call_durns) * ((1-duty_cycle)/duty_cycle)
    baseline_ipi = sum_ipis/approx_numcalls
    ipis = baseline_ipi + np.random.choice(np.arange(0,0.010,0.001),approx_numcalls)
    long_snippet_durn = sum(ipis) + sum(call_durns)

    longCFcall_seq = create_CF_call_sequence(call_durns, ipis, CF_value,
                                         call_shape, 
                                         long_snippet_durn, fs)
    differnce = longCFcall_seq.size - int(durn*fs)
    if difference > 0:
        valid_startind = np.random.choice(xrange(0,differnce),1)
        CFcall_seq = longCFcall_seq[valid_startind:valid_startind+int(durn*fs)]
    elif difference < 0:
        padding = np.zeros(difference)
        CFcall_seq = np.concatenate((longCFcall_seq, padding))
    elif difference == 0 :
        CFcall_seq = longCFcall_seq

    return(CFcall_seq)

def create_CF_call_sequence(call_durns, ipis, CF_value, call_shape, durn, fs):
    '''
    '''
    baseline_reclevel = np.random.choice(np.arange(10**-4,0.9, 10**-4),1)
    
    for calldurn, ipi in zip(call_durns, ipis):
        seg_length = calldurn+ipi
        callipi_segment = np.zeros(int(seg_length*fs))

        if call_shape is 'staplepin':
            staplepin_call = make_staplepin(calldurn, CF_value)
            
        pass
        

    
    
    
    pass


def make_staplepin(call_durn, cf_freq):
    '''
    '''
    pass

def make_rightangle():
    pass

def make_onlyCF():
    pass
        
        
            
        
        
   
