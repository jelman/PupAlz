# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 12:00:45 2016

@author: jelman

This script takes Tobii .gazedata file from auditory oddball as input. It 
first performs interpolation and filtering, Then peristimulus timecourses 
are created for target and standard trials after baselining. Trial-level data 
and average PSTC waveforms data are output for further group processing using 
(i.e., with oddball_proc_group.py). 

Some procedures and parameters adapted from:
Jackson, I. and Sirois, S. (2009), Infant cognition: going full factorial 
    with pupil dilation. Developmental Science, 12: 670-679. 
    doi:10.1111/j.1467-7687.2008.00805.x

Hoeks, B. & Levelt, W.J.M. Behavior Research Methods, Instruments, & 
    Computers (1993) 25: 16. https://doi.org/10.3758/BF03204445
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pupil_utils


"""
TODO: 
    Use ReadLetter or BeginFile as baseline?  Change get_trial_events if needed.
    Verify low pass filter for fluency data to avoid removing linear trend
"""

    


def plot_trials(df, fname):
    outfile = pupil_utils.get_outfile(fname, "_PupilPlot.png")
    p = df.groupby(level='Trial').DiameterPupilLRFilt.plot()
    p.savefig(outfile)  
    plt.close()
    
    
def clean_trials(df, trialevents):
    resampled_dict = {}
    gradient = pupil_utils.get_gradient(df)
    for trialnum in trialevents.Trial.unique():
        starttime, stoptime =  trialevents.loc[trialevents.Trial==trialnum,'TETTime'].iloc[[0,-1]]
        rawtrial = df.loc[(df.TETTime>=starttime) & (df.TETTime<=stoptime)]
        clean_pupil_left, blinks_left = pupil_utils.chap_deblink(rawtrial.DiameterPupilLeftEye, 
                                                                 gradient,
                                                                 zeros_outliers = 20)       
        clean_pupil_right, blinks_right = pupil_utils.chap_deblink(rawtrial.DiameterPupilRightEye, gradient,
                                                                   zeros_outliers = 20)
        rawtrial = rawtrial.assign(DiameterPupilLeftEyeDeblink=clean_pupil_left, 
                                   BlinksLeft=blinks_left,
                                   DiameterPupilRightEyeDeblink=clean_pupil_right, 
                                   BlinksRight=blinks_right) 
        rawtrial['BlinksLR'] = np.where(rawtrial.BlinksLeft+rawtrial.BlinksRight>=1, 1, 0)        
        trial_resamp = pupil_utils.resamp_filt_data(rawtrial, filt_type='low', string_cols=['CurrentObject'])
        resampled_dict[trialnum] = trial_resamp        
    dfresamp = pd.concat(resampled_dict, names=['Trial','Timestamp'])
    return dfresamp
    

def get_trial_events(df):
    """
    Create dataframe of trial events. This includes:
        Condition: ['Letter', 'Category']
        Trial: [1, 2, 3, 4, 5, 6]
        TrialPhase = ['Baseline', 'Response']
        StartStop = ['Start', 'Stop']
    First finds timestamps where CurrentObject changes to determine starts and stops.
    Combines these and defines trial, phase and whether it is start or stop time. 
    """
    startidx = df['CurrentObject'].ne(df['CurrentObject'].shift().ffill()).astype(bool)
    stopidx = df['CurrentObject'].ne(df['CurrentObject'].shift(-1).bfill()).astype(bool)
    trialevents_start = pd.DataFrame(df.loc[startidx, ['TETTime','CurrentObject']])
    trialevents_stop = pd.DataFrame(df.loc[stopidx, ['TETTime','CurrentObject']])
    trialevents_start = trialevents_start.loc[(trialevents_start.CurrentObject=="ReadLetter") | 
                                              (trialevents_start.CurrentObject == "RecordLetter")]
    trialevents_stop = trialevents_stop.loc[(trialevents_stop.CurrentObject=="BeginFile") | 
                                            (trialevents_stop.CurrentObject == "RecordLetter")]
    trialevents_start['TrialPhase'] = np.tile(['Baseline','Response'], 6)
    trialevents_start['StartStop'] = 'Start'
    trialevents_stop['TrialPhase'] = np.tile(['Baseline','Response'], 6)
    trialevents_stop['StartStop'] = 'Stop'
    trialevents = trialevents_start.append(trialevents_stop).sort_index()
    trialevents['Trial'] = np.repeat(range(1,7), 4)
    trialevents['Condition'] = np.repeat(['Letter', 'Category'], 12)
    return trialevents

    
def proc_subject(fname):
    """Given an infile of raw pupil data, saves out:
        1. Session level data with dilation data summarized for each trial
        2. Dataframe of average peristumulus timecourse for each condition
        3. Plot of average peristumulus timecourse for each condition
        4. Percent of samples with blinks """
    df = pd.read_csv(fname, sep="\t")
    df[['DiameterPupilLeftEye','DiameterPupilRightEye']] = df[['DiameterPupilLeftEye','DiameterPupilRightEye']].replace(-1, np.nan)
    trialevents = get_trial_events(df)
    dfresamp = clean_trials(df, trialevents)
    blinkpct = pd.DataFrame(dfresamp.groupby(level='Trial').BlinksLR.mean())
    blink_outname = pupil_utils.get_outfile(fname, "_BlinkPct.csv")
    blinkpct.to_csv(blink_outname, index=True)
    dfresamp1s = dfresamp.groupby(level='Trial').apply(lambda x: x.resample('1s', level='Timestamp').mean())
    pupildf = dfresamp1s.reset_index()[['Subject','Trial','Timestamp','DiameterPupilLRFilt']]
    pupil_outname = pupil_utils.get_outfile(fname, '_ProcessedPupil.csv')
    pupildf.to_csv(pupil_outname, index=True)
    plot_trials(dfresamp1s, fname)


    
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print 'USAGE: %s <raw pupil file> ' % os.path.basename(sys.argv[0])
        print 'Takes eye tracker data text file (*recoded.gazedata) as input.'
        print 'Removes artifacts, filters, and calculates peristimulus dilation'
        print 'for target vs. non-targets. Processes single subject data and'
        print 'outputs csv files for use in further group analysis.'
    else:
        fname = sys.argv[1]
        proc_subject(fname)

