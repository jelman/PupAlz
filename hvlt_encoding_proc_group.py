#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:57:31 2019

@author: jelman

Script to gather HVLT Encoding data summarized by quartiles. Outputs a  a summary 
dataset which averages across trials to give a single value per quartile.
"""

import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
try:
    # for Python2
    import Tkinter as tkinter
    import tkFileDialog as filedialog
except ImportError:
    # for Python3
    import tkinter
    from tkinter import filedialog
    

def pivot_wide(dflong):
    colnames = ['Session','Baseline','Diameter', 'Dilation', 'BlinkPct', 'ntrials']
    dfwide = dflong.pivot(index="Subject", columns='Word', values=colnames)
    dfwide.columns = ['_'.join([str(col[0]),'hvlt_encoding',str(col[1])]).strip() for col in dfwide.columns.values]
    word = np.arange(1,13)
    neworder = [n+'_hvlt_encoding_'+str(w) for w in word for n in colnames]
    dfwide = dfwide.reindex(neworder, axis=1)
    dfwide = dfwide.reset_index()
    dfwide.columns = dfwide.columns.str.lower()
    return dfwide
    

def proc_group(datadir, exclude_file):
    # Gather processed fluency data
    globstr = '*_ProcessedPupil.csv'
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
    # outname_all = ''.join(['fluency_Quartiles_AllTrials_',date,'.csv'])
    # alldf.to_csv(os.path.join(datadir, outname_all), index=False)
    
    # Filter out words with >50% blinks
    alldf = alldf[alldf.BlinkPct<.50]

    # Load QC file and remove visually excluded data
    excludes = pd.read_excel(exclude_file)
    excludes.Subject = excludes.Subject.astype('str')
    alldf = pd.merge(alldf, excludes, how='left', on=['Subject','Session','Trial'], indicator=True)
    if (alldf._merge=='left_only').sum() > 0:
        print('Not all data has been QC\'d!')
    alldf = alldf[alldf.Exclude!=1].drop(columns=["_merge", "Exclude"])

    # Average across trials within quartile and condition
    alldfgrp = alldf.groupby(['Subject','Word']).mean(numeric_only=True).reset_index()
    alldfgrp = alldfgrp.drop(columns='Trial')
    ntrials = alldf.groupby(['Subject','Word']).size().reset_index(name='ntrials')
    alldfgrp = alldfgrp.merge(ntrials, on=['Subject','Word'], validate="one_to_one")
    # Save out summarized data
    outname_avg = ''.join(['HVLT-Encoding_group_',date,'.csv'])
    alldfgrp.to_csv(os.path.join(datadir, outname_avg), index=False)
    
    alldfgrp_wide = pivot_wide(alldfgrp)
    outname_wide = ''.join(['HVLT-Encoding_REDCap_',date,'.csv'])
    print('Writing processed group data to {0}'.format(outname_wide))
    alldfgrp_wide.to_csv(os.path.join(datadir, outname_wide), index=False)



if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('USAGE: {} <data directory> <QC file>'.format(os.path.basename(sys.argv[0])))
        print('Searches for datafiles created by hvlt_encoding_proc_subject.py for use as input.')
        print('This includes:')
        print('  HVLT_<subject>_ProcessedPupil.csv')
        print('Extracts mean dilation from quartiles and aggregates over trials.')
        print('')
        print('Require QC file listing excluded trials. Columns should be named:')
        print('    Subject	Timepoint	Trial	Exclude')
        print('Code Exclude as 0/1 for include/exclude.')
        print('')
        
        root = tkinter.Tk()
        root.withdraw()
        # Select folder containing all data to process
        datadir = filedialog.askdirectory(title='Choose directory containing subject data')
        exclude_file = filedialog.askopenfilename(title='Choose QC file containing exclude data')
        proc_group(datadir, exclude_file)

    else:
        datadir = sys.argv[1]
        exclude_file = sys.argv[2]
        proc_group(datadir, exclude_file)
        
