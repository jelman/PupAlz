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

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from glob import glob
import json

def glob_files(datadir, suffix):
    pth = os.path.join(datadir, '*/proc')
    globstr = os.path.join(pth, '*'+suffix)
    return glob(globstr)
    
    
def get_sess_data(datadir):
    sess_filelist = glob_files(datadir, suffix='_SessionData.csv')
    sess_list = []
    for sub_file in sess_filelist:
        subdf = pd.read_csv(sub_file)
        sess_list.append(subdf)
    sessdf = pd.concat(sess_list).reset_index(drop=True)
    sessdf = sessdf.drop(columns=['TrialId', 'BlinkPct'])
    sessdf = sessdf.groupby(['Subject','Session','Condition']).median()
    return sessdf.reset_index()


def get_blink_data(datadir):
    blink_filelist = glob_files(datadir, suffix='_BlinkPct.json')
    blink_list = []
    for sub_file in blink_filelist:
        subdict = json.load(open(sub_file))
        subdf = pd.DataFrame.from_records([subdict])
        blink_list.append(subdf)
    blinkdf = pd.concat(blink_list).reset_index(drop=True)
    return blinkdf    


def get_pstc_data(datadir):
    pstc_filelist = glob_files(datadir, suffix='_PSTCdata.csv')
    pstc_list = []
    for sub_file in pstc_filelist:
        subdf = pd.read_csv(sub_file)
        pstc_list.append(subdf)
    pstcdf = pd.concat(pstc_list).reset_index(drop=True)
    return pstcdf


def get_glm_data(datadir):
    glm_filelist = glob_files(datadir, suffix='GLMresults.json')
    glm_list = []
    for sub_file in glm_filelist:
        subdict = json.load(open(sub_file))
        subdf = pd.DataFrame.from_records([subdict])
        glm_list.append(subdf)
    glmdf = pd.concat(glm_list).reset_index(drop=True)
    return glmdf    


def unstack_conditions(dflong):
    df = dflong.pivot_table(index=["Subject","Session"], columns="Condition")
    df.columns = ['_'.join([col[1],col[0]]).strip() for col in df.columns.values]
    df = df.reset_index()
    return df


def calc_cnr(df):
    """
    Calculates multiple measures of CNR. Dataframe must have the following 
    columns: Target_TrialMax, Standard_TrialMax, Target_TrialSD, Standard_TrialSD.
    DIFF: Target max - Standard max
    CNR1: Target max / Standard SD
    CNR2: (Target max - Standard max) / Standard SD
    CNR3: Target SD / Standard SD
    CNR4: (Target max - Standard max) / Standard max
    """
    df['DIFF'] = df.Target_TrialMax - df.Standard_TrialMax
    df['CNR1'] = df.Target_TrialMax / df.Standard_TrialSD
    df['CNR2'] = (df.Target_TrialMax - df.Standard_TrialMax) / df.Standard_TrialSD
    df['CNR3'] = df.Target_TrialSD / df.Standard_TrialSD
    df['CNR4'] = (df.Target_TrialMax - df.Standard_TrialMax) / df.Standard_TrialMax
    df['TargStd_BetaContrast'] = df.Target_Beta - df.Standard_Beta
    return df


def plot_group_pstc(pstcdf, outfile):
    pstcdf = pstcdf[pstcdf.BlinkPct<.5]
    pstcdf['Subject_Session'] =  pstcdf.Subject.astype('str') + "_" + pstcdf.Session.astype('str')
    p = sns.tsplot(data=pstcdf, time="Timepoint",condition="Condition", unit="Subject_Session", value="Dilation").figure
    p.savefig(outfile)  
    plt.close()    
    
    
def proc_group(datadir):
    sessdf = get_sess_data(datadir)
    sessdf_wide = unstack_conditions(sessdf)
    glm_df = get_glm_data(datadir)
    blink_df = get_blink_data(datadir)
    alldat = pd.merge(sessdf_wide, glm_df, on=['Subject','Session'])
    alldat = pd.merge(alldat, blink_df, on=['Subject','Session'])
    alldat = calc_cnr(alldat)
    tstamp = datetime.now().strftime("%Y%m%d")
    outfile = os.path.join(datadir, 'oddball_group_data_' + tstamp + '.csv')
    alldat.to_csv(outfile, index=False)
    pstc_df = get_pstc_data(datadir)    
    pstc_df = pd.merge(pstc_df, blink_df, on=['Subject','Session'])
    pstc_outfile = os.path.join(datadir, 'oddball_group_pstc_' + tstamp + '.png')
    plot_group_pstc(pstc_df, pstc_outfile)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print 'USAGE: %s <data directory> ' % os.path.basename(sys.argv[0])
        print 'Searches for datafiles created by oddball_proc_subject.py for use as input.'
        print 'This includes:'
        print '  <session>-<subject>_SessionData.csv'
        print '  <session>-<subject>_PSTCdata.csv'
        print '  <session>-<subject>_BlinkPct.json'
        print '  <session>-<subject>_GLMresults.json'
        print 'Calculates subject level measures of pupil dilation and contrast to noise ratios.'
        print 'Plots group level PTSC. Output can be used for statistical analysis.'
    else:
        datadir = sys.argv[1]
        proc_group(datadir)
