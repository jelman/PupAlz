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
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import fftconvolve
import nitime.timeseries as ts
import nitime.analysis as nta
import nitime.viz as viz
from nipy.modalities.fmri.glm import GeneralLinearModel
import pupil_utils


"""
TODO: 
    Use ReadLetter or BeginFile as baseline?  Change get_trial_events if needed.
    Verify low pass filter for fluency data to avoid removing linear trend
"""

    

def split_df(dfresamp):
    """Create separate dataframes:
        1. Session level df with trial info
        2. Pupil data for all samples in target trials
        3. Pupil data for all samples in standard trials"""
    sessdf_cols = ['Subject','Session','Timestamp']
    sessdf = dfresamp.reset_index().groupby('CurrentObject')[sessdf_cols].first()
    newcols = ['TrialMean','TrialMax','TrialSD']
    
    sessdf = sessdf.join(pd.DataFrame(index=sessdf.index, columns=newcols))
    max_samples = dfresamp.reset_index().groupby('TrialId').size().max()
    trialidx = [x*.033 for x in range(max_samples)]
    targdf = pd.DataFrame(index=trialidx)
    targdf['Condition'] = 'Target'
    standdf = pd.DataFrame(index=trialidx)
    standdf['Condition'] = 'Standard'
    return sessdf, targdf, standdf
    

def save_total_blink_pct(dfresamp, infile):
    """Calculate and save out percent of trials with blinks in session"""
    outfile = pupil_utils.get_outfile(infile, '_BlinkPct.json')
    blink_dict = {}
    blink_dict['BlinkPct'] = dfresamp.BlinksLR.mean()
    blink_dict['Subject'] = dfresamp.loc[dfresamp.index[0], 'Subject']
    blink_dict['Session'] = dfresamp.loc[dfresamp.index[0], 'Session']
    blink_json = json.dumps(blink_dict)
    with open(outfile, 'w') as f:
        f.write(blink_json)
        
def get_trial_dils(pupil_dils, onset, tpre=.5, tpost=2.5):
    """Given pupil dilations for entire session and an onset, returns a 
    normalized timecourse for the trial. Calculates a baseline to subtract from
    trial data."""
    pre_event = onset - pd.to_timedelta(tpre, unit='s')
    post_event = onset + pd.to_timedelta(tpost, unit='s')    
    baseline = pupil_dils[pre_event:onset].mean()
    trial_dils = pupil_dils[onset:post_event] - baseline
    return trial_dils


def proc_all_trials(sessdf, pupil_dils, targdf, standdf):
    """FOr each trial, calculates the pupil dilation timecourse and saves to 
    appropriate dataframe depending on trial condition (target or standard).
    Saves summary metric of max dilation and standard deviation of dilation 
    to session level dataframe."""
    for trial_series in sessdf.itertuples():
        if (trial_series.TrialId==1) | (trial_series.BlinkPct>0.33):
            continue
        onset = trial_series.Timestamp
        trial_dils = get_trial_dils(pupil_dils, onset)
        sessdf.loc[sessdf.TrialId==trial_series.TrialId,'TrialMean'] = trial_dils.mean()
        sessdf.loc[sessdf.TrialId==trial_series.TrialId,'TrialMax'] = trial_dils.max()
        sessdf.loc[sessdf.TrialId==trial_series.TrialId,'TrialSD'] = trial_dils.std()
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


def pupil_irf(x):
    s1 = 50000.
    n1 = 10.1
    tmax1 = 0.930
    return s1 * ((x**n1) * (np.e**((-n1*x)/tmax1)))


def convolve_reg(event_ts, kernel):
    return fftconvolve(event_ts, kernel, 'full')[:-(len(kernel)-1)]


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
    kernel_x = np.linspace(0, kernel_end_sec, kernel_length)
    kernel = pupil_irf(kernel_x)
    trg_reg = convolve_reg(trg_ts, kernel)
    std_reg = convolve_reg(std_ts, kernel)
    plot_event(signal_filt, trg_ts, std_ts, kernel, fname)
    intercept = np.ones_like(signal_filt.data)
    X = np.array(np.vstack((intercept, trg_reg, std_reg, blinks.values)).T)
    Y = np.atleast_2d(signal_filt).T
    model = GeneralLinearModel(X)
    model.fit(Y, model='ar1')    
    int_beta, trg_beta, std_beta, blinks_beta = model.get_beta().T[0] 
    cval = [0,1,-1, 0]
    con = model.contrast(cval)    
    zval = con.z_score()[0]
    resultdict = {'Target_Beta':trg_beta, 'Standard_Beta':std_beta, 'ContrastZ':zval}
    return resultdict


def save_glm_results(glm_results, infile):
    """Calculate and save out percent of trials with blinks in session"""
    glm_json = json.dumps(glm_results)
    outfile = pupil_utils.get_outfile(infile, '_GLMresults.json')
    with open(outfile, 'w') as f:
        f.write(glm_json)
        
        
def plot_pstc(allconddf, infile):
    """Plot peri-stimulus timecourse across all trials and split by condition"""
    outfile = pupil_utils.get_outfile(infile, '_PSTCplot.png')
    p = sns.tsplot(data=allconddf, time="Timepoint",condition="Condition", unit="TrialId", value="Dilation").figure
    p.savefig(outfile)  
    plt.close()
    

def save_pstc(allconddf, infile):
    """Save out peristimulus timecourse plots"""
    outfile = pupil_utils.get_outfile(infile, '_PSTCdata.csv')
    pstcdf = allconddf.groupby(['Subject','Condition','Timepoint']).mean().reset_index()
    pstcdf.to_csv(outfile, index=False)
    


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
    

def plot_trials(df, fname):
    outfile = pupil_utils.get_outfile(fname, "_PupilPlot.png")
    p = df.groupby(level='Trial').DiameterPupilLRFilt.plot()
    p.savefig(outfile)  
    plt.close()

    
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
       
    
    

    allconddf['Subject'] = sessdf.Subject.iat[0]
    allconddf['Session'] = sessdf.Session.iat[0]    
    save_pstc(allconddf, fname)
    sessout = pupil_utils.get_outfile(fname, '_SessionData.csv')    
    sessdf.to_csv(sessout, index=False)

    
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

