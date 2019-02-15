# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 12:00:45 2016

@author: jelman

This script takes Tobii .gazedata file from digit span as input. It 
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
import Tkinter,tkFileDialog
from __future__ import division, print_function, absolute_import


def plot_trials(pupildf, fname):
    palette = sns.cubehelix_palette(len(pupildf.Load.unique()))
    p = sns.lineplot(data=pupildf, x="Timestamp",y="Dilation", hue="Load", palette=palette, legend="brief", ci=None)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_outname = pupil_utils.get_proc_outfile(fname, "_PupilPlot.png")
    p.figure.savefig(plot_outname)
    plt.close()
    
    
def clean_trials(trialevents):
    resampled_dict = {}
    for trial in trialevents.Trial.unique():
        starttime, stoptime =  trialevents.loc[trialevents.Trial==trial,'TETTime'].iloc[[0,-1]]
        rawtrial = trialevents.loc[(trialevents.TETTime>=starttime) & (trialevents.TETTime<=stoptime)]
        cleantrial = pupil_utils.deblink(rawtrial)
        string_cols = ['Load', 'Trial', 'Condition']
        trial_resamp = pupil_utils.resamp_filt_data(cleantrial, filt_type='low', string_cols=string_cols)
        baseline = trial_resamp.loc[trial_resamp.Condition=='Ready', 'DiameterPupilLRFilt'].mean()
        trial_resamp['Baseline'] = baseline
        trial_resamp['Dilation'] = trial_resamp['DiameterPupilLRFilt'] - trial_resamp['Baseline']
        trial_resamp = trial_resamp[trial_resamp.Condition=='Record']
        trial_resamp.index = pd.DatetimeIndex((trial_resamp.index - trial_resamp.index[0]).values)
        resampled_dict[trial] = trial_resamp        
    dfresamp = pd.concat(resampled_dict, names=['Trial','Timestamp'])
    return dfresamp
    
def define_condition(trialdf):
    trialdf['Condition'] = np.where((trialdf.TETTime - trialdf.TETTime.iloc[0]) / 1000. > 1., 'Record', 'Ready')
    return trialdf


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
    loads = [x for x in df.CurrentObject.unique() if "Recall" in str(x)]
    # Find last index of each load
    lastidx = [df[df.CurrentObject==x].index[-1] for x in loads]
    stopidx = [x + 1 for x in lastidx]
    startidx = stopidx[:-1]
    startidx.insert(0,0)
    triallist = []
    for i, cond in enumerate(loads):
        loaddf = df[startidx[i]:stopidx[i]]
        loaddf = loaddf[loaddf['CurrentObject']=='Ready']
        loaddf['Load'] = cond.replace('RecallDS','')
        loaddf['Trial'] = cond + '_' + loaddf['TrialId'].astype(str)
        loaddf = loaddf.groupby('Trial').apply(define_condition)
        triallist.append(loaddf)
    trialevents = pd.concat(triallist)    
    return trialevents

   
def proc_subject(filelist):
    """Given an infile of raw pupil data, saves out:
        1. Session level data with dilation data summarized for each trial
        2. Dataframe of average peristumulus timecourse for each condition
        3. Plot of average peristumulus timecourse for each condition
        4. Percent of samples with blinks """
    for fname in filelist:
        df = pd.read_csv(fname, sep="\t")
        trialevents = get_trial_events(df)
        dfresamp = clean_trials(trialevents)
        dfresamp = dfresamp.reset_index(level='Timestamp').set_index(['Load','Trial'])
#        df_load_avg = dfresamp.groupby(['Load','Timestamp']).mean().reset_index()
        pupildf = dfresamp.groupby(level=['Load','Trial']).apply(lambda x: x.resample('1s', on='Timestamp', closed='right', label='right').mean()).reset_index()
        pupilcols = ['Subject', 'Trial', 'Load', 'Timestamp', 'Dilation',
                     'Baseline', 'DiameterPupilLRFilt', 'BlinksLR']
        pupildf = pupildf[pupilcols].rename(columns={'DiameterPupilLRFilt':'Diameter',
                                                 'BlinksLR':'BlinkPct'})
        pupildf['Timestamp'] = pupildf.Timestamp.dt.strftime('%H:%M:%S')
        pupil_outname = pupil_utils.get_proc_outfile(fname, '_ProcessedPupil.csv')
        pupildf.to_csv(pupil_outname, index=False)
        plot_trials(pupildf, fname)


    
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print ''
        print 'USAGE: %s <raw pupil file> ' % os.path.basename(sys.argv[0])
        print 'Processes single subject data from digit span task and outputs' 
        print 'csv files for use in further group analysis.'
        print 'Takes eye tracker data text file (*.gazedata) as input.'
        print 'Removes artifacts, filters, and calculates dilation per 1sec.'
        print ''
        root = Tkinter.Tk()
        root.withdraw()
        # Select files to process
        filelist = tkFileDialog.askopenfilenames(parent=root,
                                              title='Choose Digit Span pupil gazedata file to process',
                                              filetypes = (("gazedata files","*.gazedata"),
                                                           ("all files","*.*")))
        filelist = list(filelist)
        
        # Run script
        proc_subject(filelist)

    else:
        filelist = [os.path.abspath(f) for f in sys.argv[1:]]
        proc_subject(filelist)

