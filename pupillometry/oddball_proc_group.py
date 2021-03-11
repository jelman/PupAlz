#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:34:58 2018

@author: jelman

Takes as input datafiles created by oddball_proc_subject.py. This includes:
    <session>-<subject>_SessionData.csv
    <session>-<subject>_PSTCdata.csv
    <session>-<subject>_BlinkPct.txt
    
Calculates subject level measures of pupil dilation and contrast to noise ratios.
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
import pupil_utils
import json
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
    sess_filelist = glob_files(datadir, suffix='_SessionData.csv')
    sess_list = []
    for sub_file in sess_filelist:
        subdf = pd.read_csv(sub_file)
        sess_list.append(subdf)
    sessdf = pd.concat(sess_list).reset_index(drop=True)
    sessdf = sessdf.drop(columns=['TrialId', 'BlinkPct'])
    sessdf = sessdf.groupby(['Subject','Session','OddballSession','Condition']).median()
    return sessdf.reset_index()


def get_oddball_session(infile):
    """Returns session as listed in the infile name (1=A, 2=B). If not listed, 
    default to SessionA."""
    if infile.find("Session") == -1:
        session = 'A'
    else:
        session = infile.split("Session")[1][0]
        session = session.replace('1','A').replace('2','B')
    return (session)


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
        subdf['OddballSession'] = get_oddball_session(sub_file)
        pstc_list.append(subdf)
    pstcdf = pd.concat(pstc_list).reset_index(drop=True)
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
    glmdf.Subject = glmdf.Subject.astype('str')
    return glmdf    


def unstack_conditions(dflong):
    df = dflong.pivot_table(index=["Subject","Session","OddballSession"], columns="Condition")
    df.columns = ['_'.join([col[1],col[0]]).strip() for col in df.columns.values]
    df = df.reset_index()
    return df


def calc_cnr(df):
    """
    Calculates multiple measures of CNR. Dataframe must have the following 
    columns: Target_DilationMax, Standard_DilationMax, Target_DilationSD, Standard_DilationSD.
    DIFF: Target max - Standard max
    CNR1: Target max / Standard SD
    CNR2: (Target max - Standard max) / Standard SD
    CNR3: Target SD / Standard SD
    CNR4: (Target max - Standard max) / Standard max
    """
    df['DIFF'] = df.Target_DilationMax - df.Standard_DilationMax
    df['CNR1'] = df.Target_DilationMax / df.Standard_DilationSD
    df['CNR2'] = (df.Target_DilationMax - df.Standard_DilationMax) / df.Standard_DilationSD
    df['CNR3'] = df.Target_DilationSD / df.Standard_DilationSD
    df['CNR4'] = (df.Target_DilationMax - df.Standard_DilationMax) / df.Standard_DilationMax
    return df


def plot_group_pstc(pstcdf, outfile, trial_start=0.):
    pstcdf = pstcdf[pstcdf.BlinkPct<.5]
    pstcdf['Subject_Session'] =  pstcdf.Subject + "_" + pstcdf.Session + "_" + pstcdf.OddballSession
    p = sns.lineplot(data=pstcdf, x="Timepoint",y="Dilation", hue="Condition")
    plt.axvline(trial_start, color='k', linestyle='--')
    p.figure.savefig(outfile, dpi=300)  
    plt.close()    
    
    
def proc_group(datadir):
    sessdf = get_sess_data(datadir)
    sessdf_wide = unstack_conditions(sessdf)
    sessdf_wide = sessdf_wide.astype({"Subject": str, "Session": str})    
    glm_df = get_glm_data(datadir)
    blink_df = get_blink_data(datadir)
    blink_df[['Subject']]
    alldat = pd.merge(sessdf_wide, glm_df, on=['Subject','Session','OddballSession'])
    alldat = pd.merge(alldat, blink_df, on=['Subject','Session','OddballSession'])
    alldat = calc_cnr(alldat)
    # Average across A and B sessions
    alldat = alldat.groupby(['Subject','Session']).mean().reset_index()
    tstamp = datetime.now().strftime("%Y%m%d")
    outfile = os.path.join(datadir, 'oddball_group_' + tstamp + '.csv')
    print('Writing processed data to {0}'.format(outfile))
    alldat.to_csv(outfile, index=False)
    redcap_cols = ['Subject', 'Session', 'Standard_ACC', 'Target_ACC',
       'Standard_ConstrictionMax', 'Target_ConstrictionMax',
       'Standard_DilationMax', 'Target_DilationMax', 'Standard_DilationMean',
       'Target_DilationMean', 'Standard_DilationSD', 'Target_DilationSD', 
       'Target_Beta', 'Standard_Beta', 'ContrastT','BlinkPct']
    alldat_redcap = alldat[redcap_cols]
    alldat_redcap.columns = ['_'.join(['oddball',str(col)]).lower() for col in alldat_redcap.columns.values]
    alldat_redcap = alldat_redcap.rename(columns={'oddball_subject':'subject', 'oddball_session':'session'})
    redcap_outfile = os.path.join(datadir, 'oddball_REDCap_' + tstamp + '.csv')
    print('Writing processed data for REDCap to {0}'.format(redcap_outfile)) 
    alldat_redcap.to_csv(redcap_outfile, index=False)
    pstc_df = get_pstc_data(datadir)    
    pstc_df = pstc_df.astype({"Subject": str, "Session": str})
    blink_df = blink_df.astype({"Subject": str, "Session": str})
    pstc_df = pd.merge(pstc_df, blink_df, on=['Subject','Session','OddballSession'])
    pstc_outfile = os.path.join(datadir, 'oddball_group_pstc_' + tstamp + '.png')
    plot_group_pstc(pstc_df, pstc_outfile)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('USAGE: {} <data directory> '.format(os.path.basename(sys.argv[0])))
        print("""Searches for datafiles created by oddball_proc_subject.py for use as input.
              This includes:
                  <session>-<subject>_SessionData.csv
                  <session>-<subject>_PSTCdata.csv
                  <session>-<subject>_BlinkPct.json
                  <session>-<subject>_GLMresults.json
              Calculates subject level measures of pupil dilation and contrast to noise ratios.
              Plots group level PTSC. Output can be used for statistical analysis.""")
        
        root = tkinter.Tk()
        root.withdraw()
        # Select folder containing all data to process
        datadir = filedialog.askdirectory(title='Choose directory containing subject data')
        proc_group(datadir)

    else:
        datadir = sys.argv[1]
        proc_group(datadir)