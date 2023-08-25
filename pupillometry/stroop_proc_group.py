#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:34:58 2018

@author: jelman

Takes as input datafiles created by stroop_proc_subject.py. This includes:
    <session>-<subject>_SessionData.csv
    <session>-<subject>_PSTCdata.csv
    <session>-<subject>_BlinkPct.txt
    
Calculates subject level measures of pupil dilation.
Plots group level PTSC. Output can be used for statistical analysis.
"""

from __future__ import division, print_function, absolute_import
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from glob import glob
import json
import pupil_utils
try:
    # for Python2
    import Tkinter as tkinter
    import tkFileDialog as filedialog
except ImportError:
    # for Python3
    import tkinter
    from tkinter import filedialog
    
    
def glob_files(datadir, suffix):
    globstr = os.path.join(datadir, '*'+suffix)
    return glob(globstr)
    
    
def get_sess_data(datadir):
    """
    Gather session data. Merge with total percentage of blinks across entire session. 
    Filter our trial 1 and any trials with >33% blinks. Filter out subjects with
    >50% blinks across the entire session. Add number of trials contributing to averages.
    """
    sess_filelist = glob_files(datadir, suffix='_SessionData.csv')
    sess_list = []
    for sub_file in sess_filelist:
        subdf = pd.read_csv(sub_file)
        sess_list.append(subdf)
    sessdf = pd.concat(sess_list).reset_index(drop=True)
    sessdf = sessdf.astype({"Subject": str, "Session": str})  
    total_blink_df = get_blink_data(datadir)
    total_blink_df = total_blink_df.astype({"Subject": str, "Session": str}) 
    sessdf = pd.merge(sessdf, total_blink_df, on=['Subject','Session'])
    sessdf = sessdf[(sessdf.BlinkPct<0.33) & (sessdf.TotalBlinkPct<0.50)]
    sessdf = sessdf.drop(columns=['TrialId'])
    sessdf_grp = sessdf.groupby(['Subject','Session','Condition']).mean(numeric_only=True)
    ntrials = sessdf.groupby(by=['Subject','Session','Condition']).size()
    ntrials.name = 'ntrials'
    sessdf_grp = sessdf_grp.join(ntrials)
    return sessdf_grp.reset_index()


def get_blink_data(datadir):
    blink_filelist = glob_files(datadir, suffix='_BlinkPct.json')
    blink_list = []
    for sub_file in blink_filelist:
        with open(sub_file, 'r') as f:
            subdict = json.load(f)
        subdf = pd.DataFrame.from_records([subdict])
        blink_list.append(subdf)
    blinkdf = pd.concat(blink_list).reset_index(drop=True)
    return blinkdf    


def get_pstc_data(datadir):
    pstc_filelist = glob_files(datadir, suffix='_PSTCdata.csv')
    pstc_list = []
    for sub_file in pstc_filelist:
        subdf = pd.read_csv(sub_file)
        subdf['Session'] = pupil_utils.get_tpfolder(sub_file)
        pstc_list.append(subdf)
    pstcdf = pd.concat(pstc_list).reset_index(drop=True)
    pstcdf.Session = pstcdf.Session.astype(int)
    pstcdf.Subject = pstcdf.Subject.astype('str')
    return pstcdf


def get_glm_data(datadir):
    glm_filelist = glob_files(datadir, suffix='GLMresults.json')
    glm_list = []
    for sub_file in glm_filelist:
        with open(sub_file, 'r') as f:
            subdict = json.load(f)
        subdf = pd.DataFrame.from_records([subdict])
        glm_list.append(subdf)
    glmdf = pd.concat(glm_list).reset_index(drop=True)
    return glmdf    


def unstack_conditions(dflong):
    df = dflong.pivot_table(index=["Subject","Session"], columns="Condition")
    df.columns = ['_'.join([col[1],col[0]]).strip() for col in df.columns.values]
    df = df.reset_index()
    return df


def calc_pstc_stats(pstc_df, window = [.5, 2.0]):
    """
    Parameters
    ----------
    pstc_df : pandas dataframe
        Dataframe in long format containing dilation averaged over trials 
        at each sample for each condition.
    window : list, optional
        DESCRIPTION. The default is [.5, 2.0].
    Returns
    -------
    Dataframe with mean and maximum dilation for all conditions.
    Considers dilation only within the specified window.
    """
    pstc_win = pstc_df[(pstc_df.Timepoint>=window[0]) & (pstc_df.Timepoint<=window[1])]
    pstc_stats_long = pstc_win.groupby(['Subject','Session','Condition'])['Dilation'].agg(['mean','max']).reset_index()
    pstc_stats_wide = unstack_conditions(pstc_stats_long)
    return pstc_stats_wide


def plot_group_pstc(pstcdf, outfile, trial_start=0.):
    pstcdf['Subject_Session'] =  pstcdf.Subject.astype('str') + "_" + pstcdf.Session.astype('str')
    p = sns.lineplot(data=pstcdf, x="Timepoint",y="Dilation", hue="Condition")
    plt.axvline(trial_start, color='k', linestyle='--')
    p.figure.savefig(outfile, dpi=300)  
    plt.close()    
    
    
def proc_group(datadir):
    tstamp = datetime.now().strftime("%Y%m%d")
    sessdf = get_sess_data(datadir)
    sessdf_wide = unstack_conditions(sessdf)
    sessdf_wide = sessdf_wide.rename(columns={'C_TotalBlinkPct':'TotalBlinkPct'}).drop(columns=['I_TotalBlinkPct','N_TotalBlinkPct'])
    glm_df = get_glm_data(datadir)
    
    alldat = pd.merge(sessdf_wide, glm_df, on=['Subject','Session'])
    # alldat = calc_cnr(alldat)
    # # Average across A and B sessions
    # alldat = alldat.groupby(['Subject','Session']).mean(numeric_only=True).reset_index()
    pstc_df = get_pstc_data(datadir)    
    pstc_df = pstc_df.astype({"Subject": str, "Session": str})
    pstc_outfile = os.path.join(datadir, 'stroop_group_pstc_' + tstamp + '.png')
    plot_group_pstc(pstc_df, pstc_outfile)
    pstc_stats = calc_pstc_stats(pstc_df, window = [.5, 2.0])
    alldat = pd.merge(alldat, pstc_stats, on=['Subject','Session'])  
    outfile = os.path.join(datadir, 'stroop_group_' + tstamp + '.csv')
    print('Writing processed data to {0}'.format(outfile))
    alldat.to_csv(outfile, index=False)
    alldat_redcap = alldat
    alldat_redcap.columns = ['_'.join(['stroop',str(col)]).lower() for col in alldat_redcap.columns.values]
    alldat_redcap = alldat_redcap.rename(columns={'stroop_subject':'subject', 'stroop_session':'session'})
    redcap_outfile = os.path.join(datadir, 'stroop_REDCap_' + tstamp + '.csv')
    print('Writing processed data for REDCap to {0}'.format(redcap_outfile)) 
    alldat_redcap.to_csv(redcap_outfile, index=False)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('USAGE: {} <data directory> '.format(os.path.basename(sys.argv[0])))
        print("""Searches for datafiles created by stroop_proc_subject.py for use as input.')
              This includes:
                  <session>-<subject>_SessionData.csv
                  <session>-<subject>_PSTCdata.csv
                  <session>-<subject>_BlinkPct.json
                  <session>-<subject>_GLMresults.json
              Calculates subject level measures of pupil dilation. Plots group 
              level PTSC. Output can be used for statistical analysis.""")
        print('')
        
        root = tkinter.Tk()
        root.withdraw()
        # Select folder containing all data to process
        datadir = filedialog.askdirectory(title='Choose directory containing subject data')
        proc_group(datadir)

    else:
        datadir = sys.argv[1]
        proc_group(datadir)
