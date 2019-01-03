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
import warnings

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
    rand_nums = np.random.choice(range(1,21),3)
    amp_ratios = rand_nums/np.sum(rand_nums)

    snippet = np.zeros(int(fs*snippet_duration))
    
    for  amp_ratio, calltype in zip(amp_ratios, [ferrum_call, eume_call, fm_call]):
        snippet += amp_ratio*calltype
    
    # add background noise or
    bkg_noise_dB = np.random.choice(np.arange(-60,-40),1)
    snippet += np.random.normal(0, 10**(bkg_noise_dB/20.0), snippet.size)
    correct_clipping(snippet)
    return(snippet)
    

def make_FM_call(fm_in, durn, fs):
    '''Make fm call 
    
    Parameters:
        fm_in : string that is either '0' or '1'
        
    Returns:

    '''
    
    num_calls = np.random.choice(xrange(1,6),1)
    
    
    
    if fm_in is '0':
        # return zeros array:
        no_fm = make_nobat_snippet(durn,fs)
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
            # implement 2nd harmonic in FM chirp
            if np.random.random()<=0.3:
                chirp_hm2 = signal.chirp(t, start_f*2, t[-1], end_f*2, method=fm_shape)
                if start_f*2 >= fs/2.0:
                    chirp_hm2 = anti_aliasing_filter(chirp_hm2)
                chirp += chirp_hm2
                chirp *= 1/np.max(chirp)
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

def anti_aliasing_filter(X, level=0.95):
    b,a = signal.butter(8, level, 'lowpass')
    lp_X = signal.lfilter(b,a,X)
    return(lp_X)

def make_nobat_snippet(duration,fs):
    '''
    '''
    nobat_snippet = np.zeros(int(duration*fs))
    
    # ad backgroun noise in dB ;
    bkg_noise = np.random.choice(np.arange(-80, -40),1)
    nobat_snippet += np.random.normal(0, 10**(bkg_noise/20.0), nobat_snippet.size)
   
    correct_clipping(nobat_snippet);
    # mimic insect noises by bandpassing between 0-40 kHz:
    if np.random.random() <= 0.3:
        nobat_snippet = add_insect_noise(nobat_snippet/np.max(nobat_snippet), fs)
    return(nobat_snippet)

def add_insect_noise(snippet, fs):
    insect_range = np.arange(0.1,40,10)*10**3
    lower, higher = np.sort(np.random.choice(insect_range, 2))
    b,a = signal.butter(4, [(2*lower)/fs,(2*higher)/fs], 'bandpass')
    insect_snippet = signal.lfilter(b,a,snippet)
    return(insect_snippet)
    

def correct_clipping(X):
    '''
    '''
    if sum(abs(X)>1) > 0 :
        X[X>1.0] = 1.0
        X[X<-1.0] = -1.0
       
    return(X)

def make_CF_call(numbats, cf_type, snippet_duration, fs):
    '''
    '''
    if numbats is '0':
        # return zeros array:
        noCF = make_nobat_snippet(snippet_duration,fs)
        return(noCF)

    elif numbats is '1':
        singleCFbat = make_singleCFbat_sequence(cf_type, snippet_duration, fs)
        return(singleCFbat)

    elif numbats is 'm' : 
        multiCFbat = np.zeros(int(snippet_duration*fs))
        num_cfbats = np.random.choice(xrange(2,5),1)
        for each_callingbat in xrange(num_cfbats):
            multiCFbat += make_singleCFbat_sequence(cf_type, snippet_duration,
                                                    fs,multibat=True)
        
        multiCFbat *= 1/np.max(multiCFbat)
        return(multiCFbat)
            
def make_singleCFbat_sequence(cf_type, durn, fs, multibat=False):
    ''' Makes a Cf bat call sequence by randomly choosing a 
    call duration, call shape and other parameters. 
    
    '''
    if cf_type is 'ferrum':
        CF_range = np.arange(79000,82500,1000)
    elif cf_type is 'eume':
        CF_range = np.arange(99500,106500,1000)

    CF_value = np.random.choice(CF_range)
    call_shape = np.random.choice(['staplepin',
                                   'rightangle',
                                   ],1, p=[0.6,0.4])[0]
    
    

    if multibat:
        duty_cycle = np.random.choice(xrange(75,95),1) * 10**-2
        call_durationrange = np.arange(5,20) * 10**-3
    else:
        duty_cycle = np.random.choice(xrange(50,95),1) * 10**-2
        call_durationrange = np.arange(10,50) * 10**-3
    
    baseline_calldurn = np.random.choice(call_durationrange,1)

    approx_numcalls = int(np.ceil(duty_cycle*durn/baseline_calldurn))
    # set the call durations of each 
    call_durns = np.random.choice(np.arange(0,0.003,0.001),approx_numcalls) + baseline_calldurn
    # set the interpulse interval following each call
    sum_ipis = sum(call_durns) * ((1-duty_cycle)/duty_cycle)
    baseline_ipi = sum_ipis/approx_numcalls
    ipi_variation = np.random.choice(np.arange(0,0.003,0.001),approx_numcalls)
    ipis = baseline_ipi + ipi_variation

    longCFcall_seq = create_CF_call_sequence(call_durns, ipis, CF_value,
                                         call_shape, 
                                         fs)

    difference = longCFcall_seq.size - int(durn*fs)
    if difference > 0:
        valid_startind = int(np.random.choice(xrange(0,difference),1))
        CFcall_seq = longCFcall_seq[valid_startind:valid_startind+int(durn*fs)]
    elif difference < 0:
        padding = np.zeros(difference)
        CFcall_seq = np.concatenate((longCFcall_seq, padding))
    elif difference == 0 :
        CFcall_seq = longCFcall_seq

    return(CFcall_seq)

def create_CF_call_sequence(call_durns, ipis, CF_value, call_shape, fs):
    '''
    '''
    baseline_reclevel = np.random.choice(np.linspace(0.01,0.9, 100),1)
    call_sequence = np.array([])
    
    for calldurn, ipi in zip(call_durns, ipis):
        seg_length = calldurn+ipi
        callipi_segment = np.zeros(int(seg_length*fs))
        one_call = make_one_CFcall(calldurn, CF_value, fs, call_shape)       
        one_call *= baseline_reclevel 
        callipi_segment[-one_call.size:] = one_call
        call_sequence = np.concatenate((call_sequence, callipi_segment))

    return(call_sequence)



def make_one_CFcall(call_durn, cf_freq, fs, call_shape):
    '''
      
    TODO : make harmonics
    '''
    # choose an FM duration
    FM_durnrange = np.arange(0.001, 0.0031, 10**-4)
    fm_durn = np.random.choice(FM_durnrange,1)

    # choose an Fm start/end fr equency :
    FM_bandwidth= xrange(5,25)
    fm_bw = np.random.choice(FM_bandwidth, 1)*10.0**3
    start_f = cf_freq - fm_bw
    # 
    polynomial_num = 25
    t = np.linspace(0, call_durn, int(call_durn*fs))
    # define the transition points in the staplepin
    freqs = np.tile(cf_freq, t.size)
    numfm_samples = int(fs*fm_durn)
    if call_shape == 'staplepin':       
        freqs[:numfm_samples] = np.linspace(start_f,cf_freq,numfm_samples, endpoint=True)
        freqs[-numfm_samples:] = np.linspace(cf_freq,start_f,numfm_samples, endpoint=True)
        p = np.polyfit(t, freqs, polynomial_num)

    elif call_shape == 'rightangle':
        # alternate between rising and falling right angle shapes
        rightangle_type = np.random.choice(['rising','falling'],1)
        if rightangle_type == 'rising':
            freqs[:numfm_samples] = np.linspace(cf_freq,start_f,numfm_samples, endpoint=True)
        elif rightangle_type == 'falling':
            freqs[-numfm_samples:] = np.linspace(cf_freq,start_f,numfm_samples, endpoint=True)
        p = np.polyfit(t, freqs, polynomial_num)

    else: 
        raise ValueError('Wrong input given')
      
    cfcall = signal.sweep_poly(t, p)
    # randomly decide if the call will have harmonics or not
    cfcall = generate_cf_harmonics(cfcall, t, p)

    windowing = np.random.choice(['hann', 'nuttall', 'bartlett'], 1)[0]
    cfcall *= signal.get_window(windowing, cfcall.size)
    return(cfcall)

def generate_cf_harmonics(main_call, t, p):
    '''
    ''' 
    main_hmonic_freq = np.polyval(p,t)
    if np.random.random()<0.4:
        lower_hmonic = main_hmonic_freq/2.0
        p_lower = np.polyfit(t, lower_hmonic, 15)
        lower_chirp = signal.sweep_poly(t, p_lower)
        lower_hmonic_attenuation = np.random.choice(np.arange(-40,-10),1)
        lower_chirp *= 10**(lower_hmonic_attenuation/20.0)
        main_call += lower_chirp
        
    if np.random.random()<0.1:
        # upsample to 3*fs
        t_upsample = np.linspace(np.min(t), np.max(t), t.size*3)
        harmonic_range = np.random.choice(np.arange(1.2,1.8,0.1),1)
        upper_hmonics = np.polyval(p, t_upsample)*harmonic_range
        p_upper = np.polyfit(t_upsample, upper_hmonics, 15)
        upper_chirp = signal.sweep_poly(t_upsample, p_upper)
        # lowpass filter below 125 kHz
        lp_upper_chirp = anti_aliasing_filter(upper_chirp, 0.32)
        # downsample back to previous rate 
        downsamples = main_call.size
        upper_chirp_downsample = signal.resample(lp_upper_chirp, downsamples)
        upper_hmonic_atten = np.random.choice(np.arange(-40,-10),1)
        upper_chirp_downsample *= 10**(upper_hmonic_atten/20.0)
        main_call += upper_chirp_downsample

    main_call *= 1/np.max(main_call)
    return(main_call)
    

def calculate_snippet_features(snippet, chunksize=250, **kwargs):
    ''' bandpasses a snippet audio and calculates the rms fo the signal 
    for the five pred-determined frequency bands. 
    
    Parameters:

        snippet : Nsamples np.array. audio snippet
        chunksize : int. number of samples in each 'chunk' for which rms is calculated
    
    Keyword arguments:
        fs : int. sampling rate in Hz. Defaults to 250000. 
        channel_norm :  Boolean. Defaults to False. If True then the rms within each channel
                      is divided by the maximum value. 
    Returns:

        rms_bands : 5 x Nchunks np.array. The rms of all chunks across the differen
                    t frequency bands. 
    '''
    eume_cf = np.array([100000.00,110000])
    eume_fm = np.array([90000.00,99000])
    ferrum_cf = np.array([77000.00,83000])
    ferrum_fm =  np.array([60000.00,76000])
    myotis_fm = np.array([20000.0,50000.0])
    
    all_bands = [eume_cf, eume_fm, ferrum_cf, ferrum_fm, myotis_fm]

    # calculate rms in each of the bands:
    if 'fs' not in kwargs.keys():
        fs = 250000
    else: 
        fs = kwargs['fs']

    # normalise the rms values of each channel if required:
    if 'channel_norm' in kwargs.keys():
        channel_norm = kwargs['channel_norm']
    else:
        channel_norm = False

    rms_bands = []
    for each_band in all_bands:
        # bandpass 
        b,a = signal.butter(8, 2*each_band/fs, 'bandpass')
        bp_snippet = signal.filtfilt(b,a, snippet)
        # calculate rms for all chunks
        band_chunkrms = calc_rms_of_chunks(bp_snippet, chunksize)
        if channel_norm:
            band_chunkrms *= 1.0/np.max(band_chunkrms)
        rms_bands.append(band_chunkrms)
    
    all_rms_bands = np.array(rms_bands).reshape(5,-1)
    # transpose to make it a channels last format for keras 
    all_rms_bands = all_rms_bands.T
    return(all_rms_bands)

def calc_features_pll(audio_snippets):
    '''
    Parameters:
        audio_snippets : Nexamples x Nsamples np.array. 
    
    Returns: 
        features : Nexamples x Nchunks x 5 np.array
    '''
    features = np.apply_along_axis(calculate_snippet_features,1,audio_snippets,
                                   chunksize=250)
    return(features)

def calc_features_fft_pll(audio_snippets):
    '''
    Parameters:
        audio_snippets : Nexamples x Nsamples np.array. 
    
    Returns: 
        features : Nexamples x Nchunks x 5 np.array
    '''
    features = np.apply_along_axis(calculate_features_fft_style,1,
                                   audio_snippets)
    return(features)


def calculate_features_fft_style(snippet, chunksize=250, **kwargs):
    '''calculate bandwise features of a snippet along each of the chunks
    In case the snippet size and chunksize do not result in an integer number
    of chunks, then the floor of the division is the result. 

    Parameters:
        snippet: Nsamples np.array of audio recordings

        chunksize: int. number of samples in a single chunk for which
                    the fft is done. Defaults to 250 samples.

    Returns:
        band_features : Nbands x Nchunks np.array. Nbands is given 
                        by the resulting number of bands 
                        based on the fft of the chunksize. 
                        Nchunks is Nsamples/chunksize. 

    '''
    
    num_chunks = int(np.floor(snippet.size/chunksize))
    if np.remainder(snippet.size, chunksize) >0:
        raise ValueError('Non integer number of chunks! ')
    
    starts = np.arange(0, snippet.size, chunksize)
    stops =  starts+chunksize
    
    num_bands = sum(np.fft.fftfreq(chunksize)>=0)
    all_ffts = np.zeros((num_chunks, num_bands))
    i = 0
    for start, stop in zip(starts, stops):
        audio_chunk = snippet[start:stop]
        all_ffts[i,:] = np.abs(np.fft.fft(audio_chunk))[:num_bands]
        i += 1 

    return(all_ffts)        


def calc_rms_of_chunks(snippet, chunksize):
    '''
    '''
    all_rms = []
    
    for start_ind in range(0, snippet.size, chunksize):
        chunk = snippet[start_ind:start_ind +chunksize-1]
        all_rms.append(rms(chunk))
    
    rms_values = np.array(all_rms)
    return(rms_values)
    

def rms(chunk):
    '''
    '''
    sq = np.square(chunk)
    mean_Sq = np.mean(sq)
    root_mean_sq = np.sqrt(mean_Sq)
    return(root_mean_sq)
    

def generate_audio_and_calc_features():
    ferrum =  np.random.choice(['0','1','m'],1).tolist() 
    eume   =  np.random.choice(['0','1','m'],1).tolist()
    myotis =  np.random.choice(['0','1'],1).tolist()
    situation = ferrum + eume + myotis
    situation_name =''.join(situation)
    
    audio = generate_audio_snippet(situation_name)
    chunk_size=250
    features = calculate_snippet_features(audio,chunk_size)
    return(situation_name, features)

def generate_audio_and_calc_fft_features():
    '''
    '''
    ferrum =  np.random.choice(['0','1','m'],1).tolist() 
    eume   =  np.random.choice(['0','1','m'],1).tolist()
    myotis =  np.random.choice(['0','1'],1).tolist()
    situation = ferrum + eume + myotis
    situation_name =''.join(situation)

    audio = generate_audio_snippet(situation_name)
    features = calculate_features_fft_style(audio)
    return(situation_name, features)

def make_100_examples_fftfeatures(X):
    '''
    '''
    all_situations = []
    all_features = []
    for i in xrange(100):
        situation, feature = generate_audio_and_calc_fft_features()
        all_situations.append(situation)
        all_features.append(feature)
    all_features = np.array(all_features)
    print('Done with 1 x100 examples')
    return(all_situations, all_features)
    
    
    

def make_100_examples(X):
    '''calculates 100 examples with different scenarios and features from
    scenarios
    '''
    all_situations = []
    all_features = []
    for i in xrange(100):
        situation, feature = generate_audio_and_calc_features()
        all_situations.append(situation)
        all_features.append(feature)
    all_features = np.array(all_features)
    print('Done with 1 x100 examples')
    return(all_situations, all_features)

def make_18number_to_9number_converter():
    '''dictionary that converts the 18 numeric classes
    to a 9 numeric classes. this system ignores the presence of 
    absence of myotis FM calls
    '''
    eighteen_classes_dict = make_snippettype_to_18number_converter()
    nine_classes_dict = make_snippettype_to_9number_converter()
    
    eighteen_to_9classes = {}
    for key, numeric_18classes in eighteen_classes_dict.iteritems():
        eighteen_to_9classes[numeric_18classes] = nine_classes_dict[key]
    return(eighteen_to_9classes)

def make_snippettype_to_18number_converter():
    '''Converts a string with Ferrum,Eume,Myotis 
    to one of the 18 numbers. 
    '''
    # make the reverse translation dictionary 
    cf_bat_states = ['0','1','m']

    convert_snippet_type_to_categorical = {}

    i = 0 
    for ferrum in cf_bat_states:
        for eume in cf_bat_states:
            for fm in ['0','1']:
                convert_snippet_type_to_categorical[ferrum+eume+fm] = i
                i+=1
    return(convert_snippet_type_to_categorical)


    
def make_snippettype_to_9number_converter():
    # make the reverse translation dictionary 
    cf_bat_states = ['0','1','m']

    convert_snippet_type_to_9categorical = {}

    i = 0 
    for ferrum in cf_bat_states:
        for eume in cf_bat_states:
            
            for fm in ['0','1']:
                convert_snippet_type_to_9categorical[ferrum+eume+fm] = i
            i+=1   
    return(convert_snippet_type_to_9categorical)

def make_9number_to_class_converter():
    '''convert 9 numbers to 2 place snippet type
    '''
    class_to_9number = make_snippettype_to_9number_converter()
    nine_number_to_class = {}
    for key, entry in class_to_9number.iteritems():
        nine_number_to_class[entry] = key[:2]
    return(nine_number_to_class)


# calculate category wise accuracy:
def get_categorywise_accuracy(conf_matrix):
    '''Takes in a confusion matrix and calculates the per 
    category accuracy 
    '''
    diags = np.diag(conf_matrix)
    num_per_cat = np.sum(conf_matrix,1)
    accuracy = diags/num_per_cat
    return(accuracy)

def max_normalise_each_channel(all_features):
    '''takes in a Nsamples x Nchannels array
    and divides each channel by the max value
    '''
    numsamples, numchannels = all_features.shape
    for i in xrange(numchannels):
        all_features[:,i] /= np.max(all_features[:,i])
    return(all_features)
if __name__ == '__main__':
    
    situation = np.random.choice(['0','1','m'],1).tolist() + np.random.choice(['0','1','m'],1).tolist()+     np.random.choice(['0','1'],1).tolist()
    situation_name =''.join(situation)

    sn = generate_audio_snippet(situation_name)
    snip_features = calculate_snippet_features(sn, 250)

    print(situation_name, 'situation name')
    plt.figure()
    plt.subplot(311)
    plt.title(situation_name)
    plt.specgram(sn, Fs=250000, NFFT=256, noverlap=100)
    plt.subplot(312)
    plt.plot(sn)
    plt.subplot(313)
    for i in range(5):
        plt.plot(snip_features[:,i])
   
