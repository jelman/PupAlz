#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:57:31 2019

@author: jelman

Script to gather fluency data summarized by quartiles. Outputs a  a summary 
dataset which averages across trials to give a single value per condition and 
quartile.
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
                                      {'00:00:15' : '1_15', '00:00:30' : '15_30', 
                                       '00:00:45' : '30_45', '00:01:00' : '45_60'},
                              'Condition' : 
                                      {'Category' : 'cat', 'Letter' : 'let'}})

    dflong = dflong[dflong.Timestamp!='00:00:00']
    dflong['ConditionTime'] = dflong.Condition + '_' + dflong.Timestamp
    dflong = dflong.drop(columns=['Condition','Timestamp'])
    colnames = ['Session', 'Dilation', 'Baseline','Diameter', 'BlinkPct', 'ntrials']
    dfwide = dflong.pivot(index="Subject", columns='ConditionTime', values=colnames)
    dfwide.columns = ['_'.join([str(col[0]),'fluency',str(col[1])]).strip() for col in dfwide.columns.values]
    condition = ['cat', 'let']
    quart = ['1_15','15_30','30_45','45_60']
    neworder = [n+'_fluency_'+c+'_'+q for c in condition for q in quart for n in colnames]
    dfwide = dfwide.reindex(neworder, axis=1)
    dfwide = dfwide.reset_index()
    dfwide.columns = dfwide.columns.str.lower()
    return dfwide
    
    
    
    
def proc_group(datadir):
    # Gather processed fluency data
    globstr = '*_ProcessedPupil_Quartiles.csv'
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
    
    # Filter out quartiles with >50% blinks or entire trials with >50% blinks
    exclude = (alldf.groupby('Subject').BlinkPct.transform(lambda x: x.mean())>.50) | (alldf.BlinkPct>.50)
    alldf = alldf[-exclude]
    # Average across trials within quartile and condition
    alldfgrp = alldf.groupby(['Subject','Condition','Timestamp']).mean().reset_index()
    ntrials = alldf.groupby(['Subject','Condition','Timestamp']).size().reset_index(name='ntrials')
    alldfgrp = alldfgrp.merge(ntrials, on=['Subject','Condition','Timestamp'], validate="one_to_one")
    alldfgrp = alldfgrp.drop(columns='Trial')
    # Save out summarized data
    outname_avg = ''.join(['fluency_Quartiles_group_',date,'.csv'])
    alldfgrp.to_csv(os.path.join(datadir, outname_avg), index=False)
    
    alldfgrp_wide = pivot_wide(alldfgrp)
    outname_wide = ''.join(['fluency_Quartiles_REDCap_',date,'.csv'])
    print('Writing processed group data to {0}'.format(outname_wide))
    alldfgrp_wide.to_csv(os.path.join(datadir, outname_wide), index=False)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('USAGE: {} <data directory> '.format(os.path.basename(sys.argv[0])))
        print('Searches for datafiles created by fluency_proc_subject.py for use as input.')
        print('This includes:')
        print('  Fluency_<subject>_ProcessedPupil_Quartiles.csv')
        print('Extracts mean dilation from quartiles and aggregates over trials.')
        print('')

        root = tkinter.Tk()
        root.withdraw()
        # Select folder containing all data to process
        datadir = filedialog.askdirectory(title='Choose directory containing subject data')
        proc_group(datadir)

    else:
        datadir = sys.argv[1]
        proc_group(datadir)


        
