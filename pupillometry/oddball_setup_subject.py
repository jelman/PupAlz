#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:08:15 2018

@author: jelman

Sets up subject directory and recodes data. Takes subject directory as input. 
Directory name must be in the format of "<PupAlz ID>_<ADRC ID>". Performs the 
following actions:
    1. Renames subject ID in the gazadata file
    2. Recodes session based on session number listed in gazedata filename
    3. Swaps correct response in gazedata file
    4. Saves out new .gazedata file with "recoded" suffix
    5. Moves all files to "raw" directory
"""

import pandas as pd
import re
import os, sys, shutil
from glob import glob
import numpy as np


def check_setup(rawdir):
    globstr = os.path.join(rawdir, '*recoded.gazedata')
    files = glob(globstr)
    if os.path.exists(rawdir) & (len(files)>=1):
        setup_status = 1
    else:
        setup_status = 0
    return setup_status

def get_subid(datadir):
    dirname = os.path.split(datadir)[-1]
    return dirname.replace('_', '-')


def get_session(infile):
    """Returns session as listed in the infile name."""
    if infile.find("Session") == -1:
        raise ValueError("No session number in filename!")
    else:
        session = infile.split("Session")[1][0]
        return int(session)
    
    
def swap_response(df):
    """ Correct response is swapped. Recode and calculate accuracy"""
    idx1 = np.where(df.CRESP==1)[0]
    idx5 = np.where(df.CRESP==5)[0]
    df.loc[df.index[idx1],'CRESP'] = 5
    df.loc[df.index[idx5],'CRESP'] = 1
    df['ACC'] = np.where(df.CRESP==df.RESP,1,0)
    return df


def recode_gaze_data(infile, subid):
    df = pd.read_csv(infile, sep="\t")
    df['Subject'] = subid
    df['Session'] = get_session(infile)  
    df = swap_response(df)
    return df
    

def rename_gaze_file(oldfile, subid):
    fname = os.path.basename(oldfile)
    fname = re.sub('[0-9]\.gazedata', 'recoded.gazedata', fname)
    splitname = fname.split('-')
    new_fname = '-'.join([splitname[0], subid, splitname[2]])
    newfile = os.path.join(datadir, new_fname) 
    return newfile       
        
    
def setup_subject(datadir):
    subid = get_subid(datadir)
    rawdir = os.path.join(datadir, 'raw')
    setup_status = check_setup(rawdir)
    if setup_status==1:
        print 'Subject {} already set up, skipping...'.format(subid)
        sys.exit()
    if not os.path.exists(rawdir):
        os.makedirs(rawdir)
    globstr = os.path.join(datadir, '*gazedata')
    gazefiles = glob(globstr)
    for oldfile in gazefiles:
        if oldfile.find('Practice') != -1:
            continue
        df = recode_gaze_data(oldfile, subid)
        newfile = rename_gaze_file(oldfile, subid)
        df.to_csv(newfile, index=False, sep="\t")
    files = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir,f))]
    for f in files:
        src = os.path.join(datadir, f)
        trg = os.path.join(rawdir, f)
        shutil.move(src, trg)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print 'USAGE: %s <subject directory> ' % os.path.basename(sys.argv[0])
        print 'Sets up subject directory and recodes data. Takes subject directory as input.'
        print 'Directory name must be in the format of "<PupAlz ID>_<ADRC ID>".'
        print 'Performs the following actions:'
        print '  1. Renames subject ID in the gazadata file'
        print '  2. Recodes session based on session number listed in gazedata filename'
        print '  3. Swaps correct response in gazedata file'
        print '  4. Saves out new .gazedata file with "recoded" suffix'
        print '  5. Moves all files to "raw" directory'
    else:
        datadir = sys.argv[1]
        setup_subject(datadir)