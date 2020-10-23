#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:57:31 2019

@author: jelman

Script to gather HVLT Recognition data for each subject and output a summary
dataset conditions averaged data per condition (old vs. new). 
"""

import os
import sys
import pandas as pd
from glob import glob
from datetime import datetime


def pivot_wide(dflong):
    colnames = ['Session','Dilation','Baseline','Diameter','BlinkPct','Duration','ntrials']
    dfwide = dflong.pivot(index="Subject", columns='Condition', values=colnames)
    dfwide.columns = ['_'.join([str(col[0]),'hvlt_recognition',str(col[1])]).strip() for col in dfwide.columns.values]
    condition = ['old', 'new']
    neworder = [n+'_hvlt_recognition_'+c for c in condition for n in colnames]
    dfwide = dfwide.reindex(neworder, axis=1)
    dfwide = dfwide.reset_index()
    dfwide.columns = dfwide.columns.str.lower()
    return dfwide
    
    
    
    
def proc_group(datadir):
    # Gather processed fluency data
    globstr = 'HVLT-Recognition-*_ProcessedPupil.csv'
    filelist = glob(os.path.join(datadir, globstr))
    # Initiate empty list to hold subject data
    allsubs = []
    for fname in filelist:
        subdf = pd.read_csv(fname)
        unique_subid = subdf.Subject.unique()
        if len(unique_subid) == 1:
            subid = str(subdf['Subject'].iat[0])
        else:
            raise Exception('Found multiple subject IDs in file {0}: {1}'.format(fname, unique_subid))
        subdf['Subject'] = subid
        allsubs.append(subdf)
    
    # Concatenate all subject date
    alldf = pd.concat(allsubs)
    # Save out concatenated data
    date = datetime.today().strftime('%Y-%m-%d')
    outname_all = ''.join(['HVLT-Recognition_group_AllTrials_',date,'.csv'])
    alldf.to_csv(os.path.join(datadir, outname_all), index=False)
    
    # Filter out trials with >50% blinks
    alldf = alldf[alldf.BlinkPct<.50]
    # Average across trials within quartile and condition
    alldfgrp = alldf.groupby(['Subject','Condition']).mean().reset_index()
    ntrials = alldf.groupby(['Subject','Condition']).size().reset_index(name='ntrials')
    alldfgrp = alldfgrp.merge(ntrials, on=['Subject','Condition'], validate="one_to_one")
    alldfgrp = alldfgrp.drop(columns='TrialId')
    # Save out summarized data
    outname_avg = ''.join(['HVLT-Recognition_group_Averaged_',date,'.csv'])
    alldfgrp.to_csv(os.path.join(datadir, outname_avg), index=False)
    
    alldfgrp_wide = pivot_wide(alldfgrp)
    outname_wide = ''.join(['HVLT-Recognition_group_REDCap_',date,'.csv'])
    alldfgrp_wide.to_csv(os.path.join(datadir, outname_wide), index=False)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('USAGE: {} <data directory> '.format(os.path.basename(sys.argv[0])))
        print('Searches for datafiles created by hvlt_recognition_proc_subject.py for use as input.')
        print('This includes:')
        print('  HVLT-Recognition-<subject>_ProcessedPupil.csv')
        print('Extracts mean dilation an duration of trials in each conditiom (old vs. new).')
        print('')
    else:
        datadir = sys.argv[1]
        proc_group(datadir)
        
