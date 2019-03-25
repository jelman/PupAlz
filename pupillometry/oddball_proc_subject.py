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

from __future__ import division, print_function, absolute_import
import os
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import fftconvolve
import nitime.timeseries as ts
import nitime.analysis as nta
import nitime.viz as viz
from nistats.regression import ARModel, OLSModel
import pupil_utils
try:
    # for Python2
    import Tkinter as tkinter
    import tkFileDialog as filedialog
except ImportError:
    # for Python3
    import tkinter
    from tkinter import filedialog



def get_sessdf(dfresamp):
    """Create dataframe of session level trial info"""
    sessdf_cols = ['Subject','Session','Condition','TrialId', 'Timestamp',
                   'ACC','RT']
    sessdf = dfresamp.reset_index().groupby('TrialId')[sessdf_cols].first()
    newcols = ['DilationMean','DilationMax','DilationSD','ConstrictionMax']
    sessdf = sessdf.join(pd.DataFrame(index=sessdf.index, columns=newcols))
    return sessdf
    

def save_total_blink_pct(dfresamp, infile):
    """Calculate and save out percent of trials with blinks in session"""
    outfile = pupil_utils.get_outfile(infile, '_BlinkPct.json')
    blink_dict = {}
    blink_dict['BlinkPct'] = float(dfresamp.BlinksLR.mean())
    blink_dict['Subject'] = str(dfresamp.loc[dfresamp.index[0], 'Subject'])
    blink_dict['Session'] = int(dfresamp.loc[dfresamp.index[0], 'Session'])
    blink_json = json.dumps(blink_dict)
    with open(outfile, 'w') as f:
        f.write(blink_json)
        
    
def get_blink_pct(dfresamp, infile=None):
    """Save out percentage of blink samples across the entire session if an infile
    is passed (infile used to determine outfile path and name). Returns percent
    of samples with blinks within each trial for filtering out bad trials."""
    if infile:
        save_total_blink_pct(dfresamp, infile)
    trial_blinkpct = dfresamp.groupby('TrialId')['BlinksLR'].mean()
    return trial_blinkpct


def get_trial_dils(pupil_dils, onset, tpre, tpost, samp_rate):
    """Given pupil dilations for entire session and an onset, returns a 
    normalized timecourse for the trial. Calculates a baseline to subtract from
    trial data."""
    onset_idx = int(pupil_dils.index.get_loc(onset))
    pre_idx = int(onset_idx - (tpre/(1/samp_rate)))
    post_idx = int(onset_idx + (tpost/(1/samp_rate)) + 1)
    baseline = pupil_dils.iloc[pre_idx:onset_idx].mean()
#    baseline = pupil_dils[onset]
    #trial_dils = pupil_dils[onset:post_event] - baseline
    trial_dils = pupil_dils.iloc[pre_idx:post_idx] - baseline
    return trial_dils


def initiate_condition_df(tpre, tpost, samp_rate):
    """Initiate dataframe to hold trial data for target and condition trials. 
    Index will represent time relative to trial start with interval based on 
    sampling rate."""
    postidx = np.arange(0, tpost + .0001, 1/samp_rate)
    preidx = np.arange(0, -1*(tpre + 0.0001), -1/samp_rate)
    trialidx = np.sort(np.unique(np.append(postidx, preidx)))
    targdf = pd.DataFrame(index=trialidx)
    targdf['Condition'] = 'Target'
    standdf = pd.DataFrame(index=trialidx)
    standdf['Condition'] = 'Standard'
    return targdf, standdf
    
    
    
def proc_all_trials(sessdf, pupil_dils, tpre=.5, tpost=2.5, samp_rate=30.):
    """FOr each trial, calculates the pupil dilation timecourse and saves to 
    appropriate dataframe depending on trial condition (target or standard).
    Saves summary metric of max dilation and standard deviation of dilation 
    to session level dataframe."""
    targdf, standdf = initiate_condition_df(tpre, tpost, samp_rate)
    for trial_series in sessdf.itertuples():
        if (trial_series.TrialId.iat[0]==1) | (trial_series.BlinkPct.iat[0]>0.33):
            continue
        onset = trial_series.Timestamp.iat[0]
        trial_dils = get_trial_dils(pupil_dils, onset, tpre, tpost, samp_rate)
        sessdf.loc[sessdf.TrialId==trial_series.TrialId.iat[0],'DilationMax'] = trial_dils.max()
        sessdf.loc[sessdf.TrialId==trial_series.TrialId.iat[0],'DilationMean'] = trial_dils.mean()
        sessdf.loc[sessdf.TrialId==trial_series.TrialId.iat[0],'DilationSD'] = trial_dils.std()
        sessdf.loc[sessdf.TrialId==trial_series.TrialId.iat[0],'ConstrictionMax'] = trial_dils.min()
        if trial_series.Condition=='Standard':
            standdf[trial_series.TrialId] = np.nan
            standdf.loc[standdf.index[:len(trial_dils)], trial_series.TrialId] = trial_dils.values
        elif trial_series.Condition=='Target':
            targdf[trial_series.TrialId] = np.nan
            targdf.loc[standdf.index[:len(trial_dils)], trial_series.TrialId] = trial_dils.values
    return sessdf, targdf, standdf
            

def reshape_df(dfwide):
    """Converts wide dataframe with separate columns for each trial to long format"""
    dfwide['Timepoint'] = dfwide.index
    df_long = pd.melt(dfwide, id_vars=['Timepoint','Condition'], 
                      var_name='TrialId', value_name='Dilation')
    df_long = df_long.sort_values(['TrialId','Timepoint'])
    return df_long
    

def get_event_ts(pupilts, events):
    event_reg = np.zeros(len(pupilts))
    event_reg[pupilts.index.isin(events)] = 1
    event_ts = ts.TimeSeries(event_reg, sampling_rate=30., time_unit='s')
    return event_ts


def plot_event(signal_filt, trg_ts, std_ts, kernel, infile):
    """Plot peri-stimulus timecourse of each event type as well as the 
    canonical pupil response function"""
    outfile = pupil_utils.get_outfile(infile, '_PSTCplot.png')
    plt.ioff()
    all_events = std_ts.data + (trg_ts.data*2)
    all_events_ts = ts.TimeSeries(all_events, sampling_rate=30., time_unit='s')
    all_era = nta.EventRelatedAnalyzer(signal_filt, all_events_ts, len_et=75, correct_baseline=True)
    fig, ax = plt.subplots()
    viz.plot_tseries(all_era.eta, yerror=all_era.ets, fig=fig)
    ax.plot((all_era.eta.time*(10**-12)), kernel)
    ax.legend(['Standard','Target','Pupil IRF'])
    fig.savefig(outfile)
    plt.close(fig)

    
    
def ts_glm(pupilts, trg_onsets, std_onsets, blinks, sampling_rate=30.):
    signal_filt = ts.TimeSeries(pupilts, sampling_rate=sampling_rate)
    trg_ts = get_event_ts(pupilts, trg_onsets)    
    std_ts = get_event_ts(pupilts, std_onsets)
    kernel_end_sec = 2.5
    kernel_length = kernel_end_sec / (1/sampling_rate)
    kernel_x = np.linspace(0, kernel_end_sec, int(kernel_length))
    trg_reg, trg_td_reg = pupil_utils.regressor_tempderiv(trg_ts, kernel_x)
    std_reg, std_td_reg = pupil_utils.regressor_tempderiv(std_ts, kernel_x)
    #kernel = pupil_irf(kernel_x)
    #plot_event(signal_filt, trg_ts, std_ts, kernel, fname)
    intercept = np.ones_like(signal_filt.data)
    X = np.array(np.vstack((intercept, trg_reg, std_reg, blinks.values)).T)
    Y = np.atleast_2d(signal_filt).T
    model = ARModel(X, rho=1.).fit(Y)
    tTrg = float(model.Tcontrast([0,1,0,0]).t)
    tStd = float(model.Tcontrast([0,0,1,0]).t)
    tTrgStd = float(model.Tcontrast([0,1,-1,0]).t)   
    resultdict = {'Target_Beta':tTrg, 'Standard_Beta':tStd, 'ContrastT':tTrgStd}
    return resultdict


def save_glm_results(glm_results, infile):
    """Calculate and save out percent of trials with blinks in session"""
    glm_json = json.dumps(glm_results)
    outfile = pupil_utils.get_outfile(infile, '_GLMresults.json')
    with open(outfile, 'w') as f:
        f.write(glm_json)
        
        
def plot_pstc(allconddf, infile, trial_start=0.):
    """Plot peri-stimulus timecourse across all trials and split by condition"""
    outfile = pupil_utils.get_outfile(infile, '_PSTCplot.png')
    p = sns.lineplot(data=allconddf, x="Timepoint",y="Dilation", hue="Condition", legend="brief")
    plt.axvline(trial_start, color='k', linestyle='--')
    p.figure.savefig(outfile)  
    plt.close()
    

def save_pstc(allconddf, infile, trial_start=0.):
    """Save out peristimulus timecourse plots"""
    outfile = pupil_utils.get_outfile(infile, '_PSTCdata.csv')
    pstcdf = allconddf.groupby(['Subject','Condition','Timepoint']).mean().reset_index()
    pstcdf.to_csv(outfile, index=False)
    

def proc_subject(filelist):
    """Given an infile of raw pupil data, saves out:
        1. Session level data with dilation data summarized for each trial
        2. Dataframe of average peristumulus timecourse for each condition
        3. Plot of average peristumulus timecourse for each condition
        4. Percent of samples with blinks """
    tpre = 0.5
    tpost = 2.5
    samp_rate = 30.
    for fname in filelist:
        df = pd.read_csv(fname, sep="\t")
        df = pupil_utils.deblink(df)
        dfresamp = pupil_utils.resamp_filt_data(df)
        dfresamp['Condition'] = np.where(dfresamp.CRESP==5, 'Standard', 'Target')
        pupil_utils.plot_qc(dfresamp, fname.replace("/raw/","/proc/"))
        sessdf = get_sessdf(dfresamp)
        sessdf['BlinkPct'] = get_blink_pct(dfresamp, fname)
        dfresamp['zDiameterPupilLRFilt'] = pupil_utils.zscore(dfresamp['DiameterPupilLRFilt'])
        sessdf, targdf, standdf = proc_all_trials(sessdf, dfresamp['zDiameterPupilLRFilt'], 
                                                  tpre, tpost, samp_rate)
        targdf_long = reshape_df(targdf)
        standdf_long = reshape_df(standdf)
        glm_results = ts_glm(dfresamp.zDiameterPupilLRFilt, 
                             sessdf.loc[sessdf.Condition=='Target', 'Timestamp'],
                             sessdf.loc[sessdf.Condition=='Standard', 'Timestamp'],
                             dfresamp.BlinksLR)
        glm_results['Session'] = int(dfresamp.loc[dfresamp.index[0], 'Session'])
        glm_results['Subject'] = str(dfresamp.loc[dfresamp.index[0], 'Subject'])
        save_glm_results(glm_results, fname)
        allconddf = standdf_long.append(targdf_long).reset_index(drop=True)
        allconddf['Subject'] = sessdf.Subject.iat[0]
        allconddf['Session'] = sessdf.Session.iat[0]    
        plot_pstc(allconddf, fname)
        save_pstc(allconddf, fname)
        sessout = pupil_utils.get_outfile(fname, '_SessionData.csv')    
        sessdf.to_csv(sessout, index=False)

    
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('USAGE: {} <raw pupil file> '.format(os.path.basename(sys.argv[0])))
        print('Takes eye tracker data text file (*recoded.gazedata) as input.')
        print('Removes artifacts, filters, and calculates peristimulus dilation')
        print('for target vs. non-targets. Processes single subject data and')
        print('outputs csv files for use in further group analysis.')
        print('')
        
        root = tkinter.Tk()
        root.withdraw()
        # Select files to process
        filelist = filedialog.askopenfilenames(parent=root,
                                                    title='Choose Oddball pupil gazedata file to process',
                                                    filetypes = (("gazedata files","*recoded.gazedata"),("all files","*.*")))
        filelist = list(filelist)
        # Run script
        proc_subject(filelist)

    else:
        filelist = [os.path.abspath(f) for f in sys.argv[1:]]
        proc_subject(filelist)


