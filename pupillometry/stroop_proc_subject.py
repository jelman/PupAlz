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
import nitime.timeseries as ts
import nitime.analysis as nta
import nitime.viz as viz
from nistats.regression import ARModel, OLSModel
import pupil_utils
import re
import Tkinter,tkFileDialog

def split_df(dfresamp, eprime):
    """Create separate dataframes:
        1. Session level df with trial info
        2. Pupil data for all samples in target trials
        3. Pupil data for all samples in standard trials"""
    sessdf_cols = ['Subject','Session','TrialId', 'Timestamp','CurrentObject']
    sessdf = dfresamp.reset_index().groupby(['TrialId','CurrentObject'])[sessdf_cols].first()
    sessidx = (sessdf.index.get_level_values("CurrentObject")=='Fixation') | (sessdf.index.get_level_values("CurrentObject")=='Stimulus')
    sessdf = sessdf.loc[sessidx]
    assert (sessdf.shape[0] == 2*eprime.shape[0]), "Number of trials in pupil data and eprime data do not match!"
    eprimecols = ['TrialList.Sample','Condition'] + [i for i in eprime.columns if re.search(r'Stimulus.*RT$', i)]
    eprimesub = eprime[eprimecols]
    eprimesub.columns = ['TrialId','Condition','RT']
    sessdf = sessdf.join(eprimesub.set_index('TrialId'))    
    newcols = ['TrialMean','TrialMax','TrialSD']    
    sessdf = sessdf.join(pd.DataFrame(index=sessdf.index, columns=newcols))
    #max_samples = dfresamp.reset_index().groupby(['TrialId','CurrentObject']).size().max()
    max_samples = dfresamp.reset_index().groupby('TrialId').size().max()
    trialidx = [x*.033 for x in range(max_samples)]
    condf = pd.DataFrame(index=trialidx)
    condf['Condition'] = 'Congruent'
    incondf = pd.DataFrame(index=trialidx)
    incondf['Condition'] = 'Incongruent'
    neutraldf = pd.DataFrame(index=trialidx)
    neutraldf['Condition'] = 'Neutral'
    return sessdf, condf, incondf, neutraldf
    

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
        
    
def get_blink_pct(dfresamp, infile=None):
    """Save out percentage of blink samples across the entire session if an infile
    is passed (infile used to determine outfile path and name). Returns percent
    of samples with blinks within each trial for filtering out bad trials."""
    if infile:
        save_total_blink_pct(dfresamp, infile)
    trial_blinkpct = dfresamp.groupby(['TrialId','CurrentObject'])['BlinksLR'].mean()
    return trial_blinkpct


def get_trial_dils(pupil_dils, fix_onset, stim_onset, tpre=-.5,tpost=3.0):
    """Given pupil dilations for entire session and an onset, returns a 
    normalized timecourse for the trial. Calculates a baseline to subtract from
    trial data."""
    post_event = stim_onset + pd.to_timedelta(tpost, unit='s')   
    pre_event = stim_onset + pd.to_timedelta(tpre, unit='s')
    #baseline = pupil_dils[fix_onset:stim_onset].mean()
    baseline = pupil_dils[pre_event:stim_onset].mean()
    #baseline = pupil_dils[fix_onset]
    #trial_dils = pupil_dils[stim_onset:post_event] - baseline
    trial_dils = pupil_dils[stim_onset:post_event] - baseline
    #trial_dils = pupil_dils[fix_onset:post_event]
    return trial_dils


def proc_all_trials(sessdf, pupil_dils, condf, incondf, neutraldf):
    """FOr each trial, calculates the pupil dilation timecourse and saves to 
    appropriate dataframe depending on trial condition (target or standard).
    Saves summary metric of max dilation and standard deviation of dilation 
    to session level dataframe."""
    for trial_number in sessdf.TrialId.unique():
        trial_series = sessdf.loc[sessdf.TrialId==trial_number]
        if trial_series.loc[(slice(None),'Stimulus'),'BlinkPct'].values>0.33:
            continue
        fix_onset, stim_onset = trial_series.Timestamp
        trial_dils = get_trial_dils(pupil_dils, fix_onset, stim_onset, tpost=3.)
        sessdf.loc[sessdf.TrialId==trial_number,'TrialMean'] = trial_dils.mean()
        sessdf.loc[sessdf.TrialId==trial_number,'TrialMax'] = trial_dils.max()
        sessdf.loc[sessdf.TrialId==trial_number,'TrialSD'] = trial_dils.std()
        if (trial_series.Condition=='C').all():
            condf[trial_number] = np.nan
            condf.loc[condf.index[:len(trial_dils)], trial_number] = trial_dils.values
        elif (trial_series.Condition=='I').all():
            incondf[trial_number] = np.nan
            incondf.loc[incondf.index[:len(trial_dils)], trial_number] = trial_dils.values
        elif (trial_series.Condition=='N').all():
            neutraldf[trial_number] = np.nan
            neutraldf.loc[neutraldf.index[:len(trial_dils)], trial_number] = trial_dils.values 
    return sessdf, condf, incondf, neutraldf
            

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
    outfile = pupil_utils.get_outfile(infile, '_PSTCplot.png')
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
    con_ts = get_event_ts(pupilts, con_onsets.xs('Stimulus', level=1))    
    incon_ts = get_event_ts(pupilts, incon_onsets.xs('Stimulus', level=1))
    neut_ts = get_event_ts(pupilts, neut_onsets.xs('Stimulus', level=1))    
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
                  'InconNeut_t':tIncon_Neut, 'ConNeut_t':tCon_Neut, 
                  'InconCon_t':tIncon_Con}
    return resultdict


def save_glm_results(glm_results, infile):
    """Calculate and save out percent of trials with blinks in session"""
    glm_json = json.dumps(glm_results)
    outfile = pupil_utils.get_outfile(infile, '_GLMresults.json')
    with open(outfile, 'w') as f:
        f.write(glm_json)
        
        
def plot_pstc(allconddf, infile, trial_start=0.):
    """Plot peri-stimulus timecourse across all trials and split by condition"""
    outfile = pupil_utils.get_outfile(infile, '_PSTCplotStimBaseline.png')
    p = sns.lineplot(data=allconddf, x="Timepoint",y="Dilation", hue="Condition", legend="brief")
    kernel = pupil_utils.pupil_irf(allconddf.Timepoint.unique(), s1=1000., tmax=1.30)
    plt.plot(allconddf.Timepoint.unique(), kernel, color='dimgrey', linestyle='--')
    plt.axvline(trial_start, color='k', linestyle='--')
    p.figure.savefig(outfile)  
    plt.close()
    

def save_pstc(allconddf, infile):
    """Save out peristimulus timecourse plots"""
    outfile = pupil_utils.get_outfile(infile, '_PSTCdata.csv')
    pstcdf = allconddf.groupby(['Subject','Condition','Timepoint']).mean().reset_index()
    pstcdf.to_csv(outfile, index=False)
    

def proc_subject(pupil_fname, eprime_fname):
    """Given an infile of raw pupil data, saves out:
        1. Session level data with dilation data summarized for each trial
        2. Dataframe of average peristumulus timecourse for each condition
        3. Plot of average peristumulus timecourse for each condition
        4. Percent of samples with blinks """
    if os.path.splitext(pupil_fname)[-1] == ".gazedata":
        df = pd.read_csv(pupil_fname, sep="\t")
    elif os.path.splitext(pupil_fname)[-1] == ".xlsx":
        df = pd.read_excel(pupil_fname)
    else: 
        raise IOError, 'Could not open {}'.format(pupil_fname)
    df = pupil_utils.deblink(df)
    df.CurrentObject.replace('StimulusRecord','Stimulus',inplace=True)
    dfresamp = pupil_utils.resamp_filt_data(df, filt_type='band', string_cols=['TrialId','CurrentObject'])
    dfresamp = dfresamp.drop(columns='TrialId_x').rename(columns={'TrialId_y':'TrialId'})
    eprime = pd.read_csv(eprime_fname, sep='\t', encoding='utf-16', skiprows=1)
    eprime = eprime.rename(columns={"Congruency":"Condition"})
    pupil_utils.plot_qc(dfresamp, pupil_fname)
    sessdf, condf, incondf, neutraldf = split_df(dfresamp, eprime)
    sessdf['BlinkPct'] = get_blink_pct(dfresamp, pupil_fname)
    dfresamp['zDiameterPupilLRFilt'] = pupil_utils.zscore(dfresamp['DiameterPupilLRFilt'])
    sessdf, condf, incondf, neutraldf = proc_all_trials(sessdf, 
                                                        dfresamp['zDiameterPupilLRFilt'],
                                                        condf, incondf, neutraldf)
    condf_long = reshape_df(condf)
    incondf_long = reshape_df(incondf)
    neutraldf_long = reshape_df(neutraldf)

    glm_results = ts_glm(dfresamp.zDiameterPupilLRFilt, 
                         sessdf.loc[sessdf.Condition=='C', 'Timestamp'],
                         sessdf.loc[sessdf.Condition=='I', 'Timestamp'],
                         sessdf.loc[sessdf.Condition=='N', 'Timestamp'],
                         dfresamp.BlinksLR)
    glm_results['Session'] = dfresamp.loc[dfresamp.index[0], 'Session']
    glm_results['Subject'] = dfresamp.loc[dfresamp.index[0], 'Subject']
    save_glm_results(glm_results, pupil_fname)
    allconddf = condf_long.append(incondf_long).reset_index(drop=True)
    allconddf = allconddf.append(neutraldf_long).reset_index(drop=True)
    allconddf['Subject'] = sessdf.Subject.iat[0]
    allconddf['Session'] = sessdf.Session.iat[0]    
    allconddf = allconddf[allconddf.Timepoint<3.0]
    plot_pstc(allconddf, pupil_fname, trial_start=.0)
    save_pstc(allconddf, pupil_fname)
    sessout = pupil_utils.get_outfile(pupil_fname, '_SessionData.csv')    
    sessdf.to_csv(sessout, index=False)

    
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print ''
        print 'USAGE: %s <raw pupil file> <eprime file>' % os.path.basename(sys.argv[0])
        print 'Takes eye tracker data text file (*.gazedata) and eprime' 
        print 'converted from .edat to .csv as input. '
        print 'Removes artifacts, filters, and calculates peristimulus dilation'
        print 'for congruent, incongruent, and the contrast between the two.'
        print 'Processes single subject data and outputs csv files for use in'
        print 'further group analysis.'
        print ''
        root = Tkinter.Tk()
        root.withdraw()
        # Select files to process
        pupil_fname = tkFileDialog.askopenfilenames(parent=root,
                                                    title='Choose Stroop pupil gazedata file to process',
                                                    filetypes = (("gazedata files","*.gazedata"),("all files","*.*")))[0]
        eprime_fname = tkFileDialog.askopenfilenames(parent=root,
                                                    title='Choose Stroop eprime file to process',
                                                    filetypes = (("eprime files","*.csv"),("all files","*.*")))[0]
        
        # Run script
        proc_subject(pupil_fname, eprime_fname)

    else:
        pupil_fname = os.path.abspath(sys.argv[1])
        eprime_fname = os.path.abspath(sys.argv[2])
        proc_subject(pupil_fname, eprime_fname)


"""
TODO:
    Add previous trial condition to sessdf
    Create regressor for previous trial condition in glm
    
"""