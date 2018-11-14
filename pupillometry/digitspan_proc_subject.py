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



def plot_trials(pupildf, fname):
    palette = sns.cubehelix_palette(len(df_load_avg.Load.unique()), start=0, rot=-.25)
    p = sns.lineplot(data=pupildf, x="Timestamp",y="Dilation", hue="Load", palette=palette,legend="brief")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_outname = pupil_utils.get_outfile(fname, "_PupilPlot.png")
    p.figure.savefig(plot_outname)
    plt.close()
    
    
def clean_trials(trialevents):
    resampled_dict = {}
    for trial in trialevents.Trial.unique():
        starttime, stoptime =  trialevents.loc[trialevents.Trial==trial,'TETTime'].iloc[[0,-1]]
        rawtrial = trialevents.loc[(trialevents.TETTime>=starttime) & (trialevents.TETTime<=stoptime)]
        cleantrial = pupil_utils.deblink(rawtrial)
        cleantrial = cleantrial[cleantrial.Condition=='Record']
        string_cols = ['Load', 'Trial', 'Condition']
        trial_resamp = pupil_utils.resamp_filt_data(cleantrial, filt_type='low', string_cols=string_cols)
        trial_resamp['Dilation'] = trial_resamp['DiameterPupilLRFilt'] - trial_resamp['DiameterPupilLRFilt'].iat[0]
#        trial_resamp.index = trial_resamp.index - trial_resamp.index[trial_resamp.Condition.searchsorted('Record')][0]
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
    startidx = lastidx[:-1]
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

   
def proc_subject(fname):
    """Given an infile of raw pupil data, saves out:
        1. Session level data with dilation data summarized for each trial
        2. Dataframe of average peristumulus timecourse for each condition
        3. Plot of average peristumulus timecourse for each condition
        4. Percent of samples with blinks """
    df = pd.read_csv(fname, sep="\t")
    trialevents = get_trial_events(df)
    dfresamp = clean_trials(trialevents)
    blinkpct = pd.DataFrame(dfresamp.groupby(level='Trial').BlinksLR.mean())
    blink_outname = pupil_utils.get_outfile(fname, "_BlinkPct.csv")
    blinkpct.to_csv(blink_outname, index=True)
    good_trials = blinkpct.index[blinkpct.BlinksLR<.33]
    dfresamp = dfresamp.loc[good_trials]
    dfresamp = dfresamp.reset_index(level='Trial', drop=True).reset_index()
    df_load_avg = dfresamp.groupby(['Load','Timestamp']).mean().reset_index()
#    sns.lineplot(data=df_load_avg, x="Timestamp",y="Dilation", hue="Load", palette=palette,legend="brief")
    pupildf = df_load_avg.groupby('Load').apply(lambda x: x.resample('1s', on='Timestamp', closed='right', label='right').mean()).reset_index()
    pupildf = pupildf[['Subject','Load','Timestamp','Dilation','DiameterPupilLRFilt']]
    pupildf['Timestamp'] = pupildf.Timestamp.dt.strftime('%H:%M:%S')
    pupil_outname = pupil_utils.get_outfile(fname, '_ProcessedPupil.csv')
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
        fname = tkFileDialog.askopenfilenames(parent=root,
                                              title='Choose pupil gazedata file to process',
                                              filetypes = (("gazedata files","*.gazedata"),
                                                           ("all files","*.*")))[0]

        
        # Run script
        proc_subject(fname)

    else:
        fname = sys.argv[1]
        proc_subject(fname)

