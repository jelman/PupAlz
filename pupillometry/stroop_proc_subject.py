# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 12:00:45 2016

@author: jelman

This script takes Tobii .gazedata file from stroop task as input. It 
first performs interpolation and filtering, Then peristimulus timecourses 
are created for incongruent, congruent, and neutral trials after baselining.
Baseline is the mean dilation 250ms prior to stimulus onset. Three sets of 
data are produced:
    - Dilation measures for each individual trial
    - Averaged timecourse for each condition
    - Results from an fmri-style GLM with contrasts betwen conditions 

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
import nitime.timeseries as ts
import nitime.analysis as nta
import nitime.viz as viz
from nilearn.glm import ARModel, OLSModel
import pupil_utils
import re
try:
    # for Python2
    import Tkinter as tkinter
    import tkFileDialog as filedialog
except ImportError:
    # for Python3
    import tkinter
    from tkinter import filedialog
    
    
def get_sessdf(dfresamp, eprime):
    """Create separate dataframes:
        1. Session level df with trial info
        2. Pupil data for all samples in target trials
        3. Pupil data for all samples in standard trials"""
    dfresamp = dfresamp.loc[dfresamp['CurrentObject']=="Stimulus"]
    sessdf_cols = ['Subject','Session','TrialId', 'Timestamp']
    sessdf = dfresamp.reset_index().groupby(['TrialId'])[sessdf_cols].first()
    assert (sessdf.shape[0] == eprime.shape[0]), "Number of trials in pupil data and eprime data do not match!"
    eprimecols = ['TrialList.Sample','Condition'] + [i for i in eprime.columns if re.search(r'Stimulus.*RT$', i)]
    eprimesub = eprime[eprimecols]
    eprimesub.columns = ['TrialId','Condition','RT']
    sessdf = sessdf.join(eprimesub.set_index('TrialId'))    
    sessdf['PrevCondition'] = sessdf['Condition'].shift()
    return sessdf
    
def get_eprime_fname(pupil_fname):
    pupildir, pupilfile = os.path.split(pupil_fname)
    edatdir = pupildir.replace('Gaze data', 'Edat')   
    edatfile = os.path.splitext(pupilfile)[0] + '-edat.csv'
    edat_fname = os.path.join(edatdir, edatfile)  
    return edat_fname
    
def save_total_blink_pct(dfresamp, infile):
    """Calculate and save out percent of trials with blinks in session"""
    outfile = pupil_utils.get_proc_outfile(infile, '_BlinkPct.json')

    blink_dict = {}
    blink_dict['TotalBlinkPct'] = float(dfresamp.BlinksLR.mean(numeric_only=True))
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
    trial_blinkpct = dfresamp.groupby(['TrialId'])['BlinksLR'].mean(numeric_only=True)
    return trial_blinkpct


def get_trial_dils(pupil_dils, onset, tpre, tpost, samp_rate):
    """Given pupil dilations for entire session and an onset, returns a 
    normalized timecourse for the trial. Calculates a baseline to subtract from
    trial data."""
    onset_idx = int(pupil_dils.index.get_loc(onset))
    pre_idx = int(onset_idx - (tpre/(1/samp_rate)))
    post_idx = int(onset_idx + (tpost/(1/samp_rate)) + 1)
    baseline = pupil_dils.iloc[pre_idx:onset_idx].mean(numeric_only=True)
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
    condf = pd.DataFrame(index=trialidx)
    condf['Condition'] = 'Congruent'
    incondf = pd.DataFrame(index=trialidx)
    incondf['Condition'] = 'Incongruent'
    neutraldf = pd.DataFrame(index=trialidx)
    neutraldf['Condition'] = 'Neutral'
    return condf, incondf, neutraldf


def proc_all_trials(sessdf, pupil_dils, tpre=.5, tpost=2.5, samp_rate=30.):
    """For each trial, calculates the pupil dilation timecourse and saves to 
    appropriate dataframe depending on trial condition."""
    condf, incondf, neutraldf = initiate_condition_df(tpre, tpost, samp_rate)
    # Filter trials for subjects with RT data
    # Some subjects have RT==0 for almost all trials, skip these subjects
    # Threshold is arbitrarily set to 90% of trials with value of 0
    if np.mean(sessdf.RT==0) < .9:
        # Filter out trials that are too short
        sessdf = sessdf.loc[sessdf.RT>=250]
        # Filter out trials that are too long (more than 3 SDs above the mean)
        sessdf = sessdf.loc[sessdf.RT < sessdf.RT.mean(numeric_only=True) + (3*sessdf.RT.std())]
    for trial_series in sessdf.itertuples():
        if trial_series.BlinkPct>0.33:
            continue
        onset = trial_series.Timestamp
        trial_dils = get_trial_dils(pupil_dils, onset, tpre, tpost, samp_rate)
        # Depending on sampling rate, trial_dils and condf may not be identical length
        # If off by 1 sample, cut first sample (from tpre period) from trial_dils
        if len(trial_dils)==len(condf)+1:
            trial_dils = trial_dils[1:]

        if trial_series.Condition=='C':
            condf[trial_series.TrialId] = np.nan
            condf.loc[condf.index[:len(trial_dils)], trial_series.TrialId] = trial_dils.values
        elif trial_series.Condition=='I':
            incondf[trial_series.TrialId] = np.nan
            incondf.loc[incondf.index[:len(trial_dils)], trial_series.TrialId] = trial_dils.values
        elif trial_series.Condition=='N':
            neutraldf[trial_series.TrialId] = np.nan
            neutraldf.loc[neutraldf.index[:len(trial_dils)], trial_series.TrialId] = trial_dils.values 
    return condf, incondf, neutraldf
            

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


def plot_event(signal_filt, con_ts, incon_ts, neut_ts, kernel, infile, plot_kernel=True):
    """Plot peri-stimulus timecourse of each event type as well as the 
    canonical pupil response function"""
    outfile = pupil_utils.get_proc_outfile(infile, '_PSTCplot.png')
    plt.ioff()
    all_events = con_ts.data + (incon_ts.data*2) + (neut_ts.data*3)
    all_events_ts = ts.TimeSeries(all_events, sampling_rate=30., time_unit='s')
    all_era = nta.EventRelatedAnalyzer(signal_filt, all_events_ts, len_et=90, correct_baseline=False)
    fig, ax = plt.subplots()
    viz.plot_tseries(all_era.eta, yerror=all_era.ets, fig=fig)
    if plot_kernel:
        ax.plot((all_era.eta.time*(10**-12)), kernel)
    ax.legend(['Congruent','Incongruent','Neutral'])
    fig.savefig(outfile)
    plt.close(fig)

    
def ts_glm(pupilts, con_onsets, incon_onsets, neut_onsets, blinks, sampling_rate=30.):
    """
    Currently runs the following contrasts:
        Incongruent: [0,1,0,0,0]
        Congruent:   [0,0,1,0,0]
        Neutral:     [0,0,0,1,0]  
        Incon-Neut:  [0,1,0,-1,0]
        Con-Neut:    [0,0,1,-1,0]
        Incon-Con:   [0,1,-1,0,0]
    """
    signal_filt = ts.TimeSeries(pupilts, sampling_rate=sampling_rate)
    con_ts = get_event_ts(pupilts, con_onsets)    
    incon_ts = get_event_ts(pupilts, incon_onsets)
    neut_ts = get_event_ts(pupilts, neut_onsets)    
    kernel_end_sec = 3.
    kernel_length = kernel_end_sec / (1/sampling_rate)
    kernel_x = np.linspace(0, kernel_end_sec, int(kernel_length)) 
    con_reg, con_td_reg = pupil_utils.regressor_tempderiv(con_ts, kernel_x, s1=1000., tmax=1.30)
    incon_reg, incon_td_reg = pupil_utils.regressor_tempderiv(incon_ts, kernel_x, s1=1000., tmax=1.30)
    neut_reg, neut_td_reg = pupil_utils.regressor_tempderiv(neut_ts, kernel_x, s1=1000., tmax=1.30)
    #kernel = pupil_utils.pupil_irf(kernel_x, s1=1000., tmax=1.30)
    #plot_event(signal_filt, con_ts, incon_ts, neut_ts, kernel, pupil_fname)
    intercept = np.ones_like(signal_filt.data)
    X = np.array(np.vstack((intercept, incon_reg, con_reg, neut_reg, blinks.values)).T)
    Y = np.atleast_2d(signal_filt).T
    model = ARModel(X, rho=1.).fit(Y)
    tIncon = float(model.Tcontrast([0,1,0,0,0]).t)
    tCon = float(model.Tcontrast([0,0,1,0,0]).t)
    tNeut = float(model.Tcontrast([0,0,0,1,0]).t)
    tIncon_Neut = float(model.Tcontrast([0,1,0,-1,0]).t)
    tCon_Neut = float(model.Tcontrast([0,0,1,-1,0]).t)
    tIncon_Con = float(model.Tcontrast([0,1,-1,0,0]).t)   
    resultdict = {'Incon_t':tIncon, 'Con_t':tCon, 'Neut_t':tNeut,
                  'Incon_Neut_t':tIncon_Neut, 'Con_Neut_t':tCon_Neut, 
                  'Incon_Con_t':tIncon_Con}
    return resultdict


def save_glm_results(glm_results, infile):
    """Calculate and save out percent of trials with blinks in session"""
    glm_json = json.dumps(glm_results)
    outfile = pupil_utils.get_proc_outfile(infile, '_GLMresults.json')
    with open(outfile, 'w') as f:
        f.write(glm_json)
        
        
def plot_pstc(allconddf, infile, trial_start=0.):
    """Plot peri-stimulus timecourse across all trials and split by condition"""
    outfile = pupil_utils.get_proc_outfile(infile, '_PSTCplot.png')
    p = sns.lineplot(data=allconddf, x="Timepoint",y="Dilation", hue="Condition", legend="brief")
    plt.axvline(trial_start, color='k', linestyle='--')
    p.figure.savefig(outfile)  
    plt.close()
    

def save_pstc(allconddf, infile):
    """Save out peristimulus timecourse plots"""
    outfile = pupil_utils.get_proc_outfile(infile, '_PSTCdata.csv')
    pstcdf = allconddf.groupby(['Subject','Condition','Timepoint']).mean(numeric_only=True).reset_index()
    pstcdf.to_csv(outfile, index=False)
    

def proc_subject(filelist):
    """Given an infile of raw pupil data, saves out:
        1. Session level data with dilation data summarized for each trial
        2. Dataframe of average peristumulus timecourse for each condition
        3. Plot of average peristumulus timecourse for each condition
        4. Percent of samples with blinks """
    tpre = 0.250
    tpost = 2.5
    samp_rate = 30.
    for pupil_fname in filelist:
        print('Processing {}'.format(pupil_fname))
        if (os.path.splitext(pupil_fname)[-1] == ".gazedata") | (os.path.splitext(pupil_fname)[-1] == ".csv"):
            df = pd.read_csv(pupil_fname, sep="\t")
        elif os.path.splitext(pupil_fname)[-1] == ".xlsx":
            df = pd.read_excel(pupil_fname, parse_dates=False)
        else: 
            raise IOError('Could not open {}'.format(pupil_fname))
        subid = pupil_utils.get_subid(df['Subject'],pupil_fname)
        timepoint = pupil_utils.get_timepoint(df['Session'], pupil_fname)
        df = pupil_utils.deblink(df)
        df.CurrentObject.replace('StimulusRecord','Stimulus',inplace=True)
        dfresamp = pupil_utils.resamp_filt_data(df, filt_type='band', string_cols=['TrialId','CurrentObject'])
        dfresamp = dfresamp.drop(columns='TrialId_x').rename(columns={'TrialId_y':'TrialId'})
        eprime_fname = get_eprime_fname(pupil_fname)
        eprime = pd.read_csv(eprime_fname, sep='\t', encoding='utf-16', skiprows=0)
        if not np.array_equal(eprime.columns[:3], ['ExperimentName', 'Subject', 'Session']):
            eprime = pd.read_csv(eprime_fname, sep='\t', encoding='utf-16', skiprows=1)
        eprime = eprime.rename(columns={"Congruency":"Condition"})
        pupil_utils.plot_qc(dfresamp, pupil_fname)
        sessdf = get_sessdf(dfresamp, eprime)
        sessdf['BlinkPct'] = get_blink_pct(dfresamp, pupil_fname)
        dfresamp['zDiameterPupilLRFilt'] = pupil_utils.zscore(dfresamp['DiameterPupilLRFilt'])
        condf, incondf, neutraldf = proc_all_trials(sessdf, dfresamp['zDiameterPupilLRFilt'], 
                                                    tpre, tpost, samp_rate)
        condf_long = reshape_df(condf)
        incondf_long = reshape_df(incondf)
        neutraldf_long = reshape_df(neutraldf)
        glm_results = ts_glm(dfresamp.zDiameterPupilLRFilt, 
                             sessdf.loc[sessdf.Condition=='C', 'Timestamp'],
                             sessdf.loc[sessdf.Condition=='I', 'Timestamp'],
                             sessdf.loc[sessdf.Condition=='N', 'Timestamp'],
                             dfresamp.BlinksLR)
        # Set subject ID and session as (as type string)
        glm_results['Subject'] = subid
        glm_results['Session'] = timepoint
        save_glm_results(glm_results, pupil_fname)
        allconddf = condf_long.append(incondf_long).reset_index(drop=True)
        allconddf = allconddf.append(neutraldf_long).reset_index(drop=True)
        # Set subject ID and session as (as type string)
        allconddf['Subject'] = subid
        allconddf['Session'] = timepoint   
        allconddf = allconddf[allconddf.Timepoint<3.0]
        plot_pstc(allconddf, pupil_fname)
        save_pstc(allconddf, pupil_fname)
        sessdf['Subject'] = subid
        sessdf['Session'] = timepoint
        sessout = pupil_utils.get_proc_outfile(pupil_fname, '_SessionData.csv')    
        sessdf.to_csv(sessout, index=False)

    
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('')
        print('USAGE: {} <raw pupil file> '.format(os.path.basename(sys.argv[0])))
        print("""Takes eye tracker data text file (*.gazedata/*.xlsx/*.csv) as input.
              Uses filename and path of eye tracker data to additionally identify 
              and load eprime file (must already be converted from .edat to .csv. 
              Removes artifacts, filters, and calculates peristimulus dilation
              for congruent, incongruent, and the contrast between the two.
              Processes single subject data and outputs csv files for use in
              further group analysis.""")
        print('')
        root = tkinter.Tk()
        root.withdraw()
        # Select files to process
        filelist = filedialog.askopenfilenames(parent=root,
                                                    title='Choose Stroop pupil gazedata file to process',
                                                    filetypes = (("xlsx files","*.xlsx"),("all files","*.*")))
        filelist = list(filelist)
        # Run script
        proc_subject(filelist)

    else:
        filelist = [os.path.abspath(f) for f in sys.argv[1:]]
        proc_subject(filelist)
