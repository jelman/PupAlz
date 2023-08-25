# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 12:00:45 2016

@author: jelman

This script takes Tobii .gazedata file from HVLT recall-rocgnition as input. It 
first performs interpolation and filtering. The first 250ms of each trial is
used as baseline and subtracted from remaining samples in the trial. Mean dilation 
is then calculated for every trial. Conditions labels corresponding to a hard-coded 
trial order are added to each trial. If the trial order of conditions is changed, 
the function hvlt_conditiona_df() should be updated.

Some procedures and parameters adapted from:
Jackson, I. and Sirois, S. (2009), Infant cognition: going full factorial 
    with pupil dilation. Developmental Science, 12: 670-679. 
    doi:10.1111/j.1467-7687.2008.00805.x

Hoeks, B. & Levelt, W.J.M. Behavior Research Methods, Instruments, & 
    Computers (1993) 25: 16. https://doi.org/10.3758/BF03204445
"""

from __future__ import division, print_function, absolute_import
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pupil_utils
try:
    # for Python2
    import Tkinter as tkinter
    import tkFileDialog as filedialog
except ImportError:
    # for Python3
    import tkinter
    from tkinter import filedialog


def plot_trials(df, fname):
    sns.set_style("ticks")
    p = sns.lineplot(data=df, x="Timestamp", y="Dilation", hue="Condition")
    plt.ylim(-.2, .5)
    p.axvline(x=2.5, color='black', linestyle='--')
    plt.tight_layout()
    plot_outname = pupil_utils.get_proc_outfile(fname, "_PSTCPlot.png")
    plot_outname = plot_outname.replace("HVLT_Recall-Recognition","HVLT_Recognition")    
    plot_outname = plot_outname.replace("-Delay","-Recognition")
    p.figure.savefig(plot_outname)
    plt.close()
    
    
def hvlt_conditions_df():
    """
    Creates a dataframe with hard-coded trial conditions for HVLT recognition. 
    """    
    trialids = np.arange(1,25)
    words = ["Horse","Ruby","Cave","Balloon","Coffee","Lion","House","Opal",
             "Tiger","Boat","Scarf","Pearl","Hut","Emerald","Sapphire","Dog",
             "Apartment","Penny","Tent","Mountain","Cat","Hotel","Cow","Diamond"]
    condition = ["old","new","old","new","new","old","new","old","old","new",
                 "new","old","old","old","old","new","new","new","old","new",
                 "new","old","old","new"]
    hvlt_conds = pd.DataFrame({'TrialId':trialids, 'Words':words, 'Condition':condition})
    return hvlt_conds


def clean_trials(df):
        dfresamp = pupil_utils.resamp_filt_data(df, filt_type='low', string_cols=['CurrentObject'])
        # Resampling fills forward fills Current Object when missing. This
        # results in values of "Response" at beginning of trials. Reaplce these
        # by backfilling from first occurrence of "Fixation" in every trial. 
        for i in dfresamp.TrialId.unique():
            trialstartidx = (dfresamp.TrialId==i).idxmax()
            fixstartidx = (dfresamp.loc[dfresamp.TrialId==i,"CurrentObject"]=="Fixation").idxmax()
            dfresamp.loc[trialstartidx:fixstartidx, "CurrentObject"] = "Fixation"
        dfresamp.groupby("TrialId").apply(lambda x: x.index[-1] - x.index[0]).sum()
        return dfresamp


def proc_all_trials(dfresamp):
    """ 
    Process single trial using the following steps:
        1) Get mean diameter of first 250ms as baseline 
        2) Calculate dilation by subtracting baseline from diameter at all samples
        3) Get mean dilation for each trial
        4) Get mean blink pct per trial
        5) Get duration of each trial
    """
    dfresamp = dfresamp[~(dfresamp.CurrentObject=="Fixation")]
    dfresamp['Baseline'] = dfresamp.groupby('TrialId')['DiameterPupilLRFilt'].transform(lambda x: x.first('250ms').mean(numeric_only=True))
    dfresamp['Dilation'] = dfresamp['DiameterPupilLRFilt'] - dfresamp['Baseline']
    dfresamp['Duration'] = dfresamp.groupby('TrialId')['TrialId'].transform(lambda x: (x.index[-1] - x.index[0]) / np.timedelta64(1, 's'))
    dfresamp['BlinkPct'] = dfresamp.groupby('TrialId')['BlinksLR'].transform(lambda x: x.mean(numeric_only=True))
    alltrialsdf = dfresamp.reset_index()
    alltrialsdf['Timestamp'] = alltrialsdf.groupby('TrialId')['Timestamp'].transform(lambda x: (x - x.iat[0]) / np.timedelta64(1, 's'))
    conditions = hvlt_conditions_df()
    alltrialsdf = alltrialsdf.merge(conditions[['TrialId','Condition']], on='TrialId') 
    return alltrialsdf    
    
def proc_subject(filelist):
    """
    Given an infile of raw pupil data, saves out:
        1) Session level data with dilation data summarized for each trial
        2) Dataframe of average peristumulus timecourse for each condition
        3) Plot of average peristumulus timecourse for each condition
        4) Percent of samples with blinks 
    """
    for fname in filelist:
        print('Processing {}'.format(fname))
        if (os.path.splitext(fname)[-1] == ".gazedata") | (os.path.splitext(fname)[-1] == ".csv"):
            df = pd.read_csv(fname, sep="\t")
        elif os.path.splitext(fname)[-1] == ".xlsx":
            df = pd.read_excel(fname)
        else: 
            raise IOError('Could not open {}'.format(fname))
        subid = pupil_utils.get_subid(df['Subject'], fname)
        timepoint = pupil_utils.get_timepoint(df['Session'], fname)
        # Keep only samples after last sample of Recall
        df = df[df[df.CurrentObject=="Recall"].index[-1]+1:]
        df = pupil_utils.deblink(df)
        dfresamp = clean_trials(df)      
        alltrialsdf = proc_all_trials(dfresamp)
        # Remove trials with >50% blinks
        alltrialsdf = alltrialsdf[alltrialsdf.BlinkPct<.50]              
        
        plot_trials(alltrialsdf, fname)
        pupildf = alltrialsdf.groupby(['Condition', 'Timestamp'])[['Baseline','DiameterPupilLRFilt','Dilation','BlinksLR','Duration']].mean(numeric_only=True)
        pupildf['ntrials'] = alltrialsdf.groupby(['Condition', 'Timestamp']).size()
        pupildf = pupildf.reset_index()  
        pupildf['Subject'] = subid
        pupildf['Session'] = timepoint
        pupildf = pupildf.rename(columns={'DiameterPupilLRFilt':'Diameter',
                                                  'BlinksLR':'BlinkPct'})
        # Reorder columns
        cols = ['Subject', 'Session', 'Baseline', 'Timestamp','Diameter', 
                'Dilation', 'BlinkPct', 'Duration','Condition','ntrials']
        pupildf = pupildf[cols]
        pupil_outname = pupil_utils.get_proc_outfile(fname, '_ProcessedPupil.csv')
        pupil_outname = pupil_outname.replace("HVLT_Recall-Recognition","HVLT_Recognition")    
        pupil_outname = pupil_outname.replace("-Delay","-Recognition")
        pupildf.to_csv(pupil_outname, index=False)
        print('Writing processed data to {0}'.format(pupil_outname))
 

    
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('')
        print('USAGE: {} <raw pupil file> '.format(os.path.basename(sys.argv[0])))
        print("""Processes single subject data from HVLT task and outputs csv
              files for use in further group analysis. Takes eye tracker data 
              text file (*.gazedata) as input. Removes artifacts, filters, and 
              calculates dilation per 1s.Also creates averages over 15s blocks.""")
        print('')
        root = tkinter.Tk()
        root.withdraw()
        # Select files to process
        filelist = filedialog.askopenfilenames(parent=root,
                                              title='Choose HVLT recall-recognition pupil gazedata file to process')       
        filelist = list(filelist)
        # Run script
        proc_subject(filelist)

    else:
        filelist = [os.path.abspath(f) for f in sys.argv[1:]]
        proc_subject(filelist)

