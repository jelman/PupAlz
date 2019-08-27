#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:57:31 2019

@author: jelman

Script to gather fluency data summarized by quartiles. Outputs a concatenated
dataset as well as a summary dataset which averages across trials to give 
a single value per condition and quartile.
"""

import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime

# Gather processed fluency data
datadir = '/home/jelman/netshare/VETSA_NAS/PROJ/PupAlz/data/Pupillometry/Fluency/proc'
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
outname_all = ''.join(['fluency_Quartiles_AllTrials_',date,'.csv'])
alldf.to_csv(os.path.join(datadir, outname_all), index=False)

# Filter out quartiles with >50% blinks
alldf = alldf[alldf.BlinkPct<.50]
# Average across trials within quartile and condition
alldfgrp = alldf.groupby(['Subject','Condition','Timestamp']).mean().reset_index()
ntrials = alldf.groupby(['Subject','Condition','Timestamp']).size().reset_index(name='ntrials')
alldfgrp = alldfgrp.merge(ntrials, on=['Subject','Condition','Timestamp'], validate="one_to_one")
alldfgrp = alldfgrp.drop(columns='Trial')
# Save out summarized data
outname_avg = ''.join(['fluency_Quartiles_Averaged_',date,'.csv'])
alldfgrp.to_csv(os.path.join(datadir, outname_avg), index=False)
