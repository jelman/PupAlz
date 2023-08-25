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
    dflong = dflong.replace({'Timestamp' : 
                                      {'00:00:10' : '1_10', 
                                       '00:00:20' : '10_20', 
                                       '00:00:30' : '20_30'}})

    dflong = dflong[dflong.Timestamp!='00:00:00']
    colnames = ['Session', 'Baseline','Diameter', 'Dilation', 'BlinkPct']
    dfwide = dflong.pivot(index="Subject", columns='Timestamp', values=colnames)
    dfwide.columns = ['_'.join([str(col[0]),'hvlt_recall',str(col[1])]).strip() for col in dfwide.columns.values]
    quart = ['1_10','10_20','20_30']
    neworder = [n+'_hvlt_recall_'+q for q in quart for n in colnames]
    dfwide = dfwide.reindex(neworder, axis=1)
    dfwide = dfwide.reset_index()
    dfwide.columns = dfwide.columns.str.lower()
    return dfwide
    


def proc_group(datadir, exclude_file):
    # Gather processed fluency data
    globstr = 'HVLT-Recall*_ProcessedPupil_Tertiles.csv'
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
    # Remove time 00:00:00
    
    # Remove subjects who have total blinkpct >50%
    alldf = alldf.groupby('Subject').filter(lambda x: x['BlinkPct'].mean(numeric_only=True) < .50)
    # Remove tertiles with over 50% blinks
    alldf = alldf[alldf.BlinkPct<.50]
    # Save out concatenated data
    date = datetime.today().strftime('%Y-%m-%d')
    
    # Load QC file and remove visually excluded data
    excludes = pd.read_excel(exclude_file)
    excludes.Subject = excludes.Subject.astype('str')
    alldf = pd.merge(alldf, excludes, how='left', on=['Subject','Session'], indicator=True)
    if (alldf._merge=='left_only').sum() > 0:
        print('Not all data has been QC\'d!')
    alldf = alldf[alldf.Exclude!=1].drop(columns=["_merge", "Exclude"])
    
    # outname_all = ''.join(['fluency_Quartiles_AllTrials_',date,'.csv'])
    # alldf.to_csv(os.path.join(datadir, outname_all), index=False)
    
    # Save out summarized data
    outname = ''.join(['HVLT-Recall_Tertiles_group_',date,'.csv'])
    alldf.to_csv(os.path.join(datadir, outname), index=False)

    alldf_wide = pivot_wide(alldf)
    outname_wide = ''.join(['HVLT-Recall_Tertiles_REDCap_',date,'.csv'])
    print('Writing processed group data to {0}'.format(outname_wide))
    alldf_wide.to_csv(os.path.join(datadir, outname_wide), index=False)
    

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('USAGE: {} <data directory> '.format(os.path.basename(sys.argv[0])))
        print('Searches for datafiles created by hvlt_recall_proc_subject.py for use as input.')
        print('This includes:')
        print('  HVLT-Recall_<subject>_ProcessedPupil_Tertiles.csv')
        print('Extracts mean dilation from tertiles of first 30s and aggregates over trials.')
        print('')
        print('Require QC file listing excluded trials. Columns should be named:')
        print('    Subject	Timepoint	Exclude')
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
