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

from __future__ import division, print_function, absolute_import
import pandas as pd
import re
import os, sys
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


def get_subid(fname):
    subid = re.findall(r'Oddball-\d{3}\b', fname)[0].replace('Oddball-','')
    return subid


def get_session(infile):
    """Returns session as listed in the infile name."""
    if infile.find("Session") == -1:
        session = 1
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


def recode_gaze_data(fname, session):
    if (os.path.splitext(fname)[-1] == ".gazedata") | (os.path.splitext(fname)[-1] == ".csv"):
        df = pd.read_csv(fname, sep="\t")
    elif os.path.splitext(fname)[-1] == ".xlsx":
        df = pd.read_excel(fname)
    else: 
        raise IOError('Could not open {}'.format(fname))   
    df['Session'] = session
    df = swap_response(df)
    return df
    

def rename_gaze_file(procdir, subid, session):
    new_fname = ''.join(['Oddball-', str(subid), '-', str(session), '_recoded.gazedata'])
    newfile = os.path.join(procdir, new_fname)
    return newfile       
        
    
def setup_subject(fname):
    subid = get_subid(fname)
    session = get_session(fname)
    rawdir = os.path.dirname(fname)
    procdir = rawdir.replace('/raw', '/proc')
    df = recode_gaze_data(fname, session)
    newfile = rename_gaze_file(procdir, subid, session)
    df.to_csv(newfile, index=False, sep="\t")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('USAGE: {} <subject directory> '.format(os.path.basename(sys.argv[0])))
        print('Sets up subject and recodes data. Takes pupil data filename as input.')
        print('Performs the following actions:')
        print('  1. Renames subject ID in the gazadata file')
        print('  2. Recodes session based on session number listed in gazedata filename')
        print('  3. Swaps correct response in gazedata file')
        print('  4. Saves out new .gazedata file with "recoded" suffix')
    else:
        fname = sys.argv[1]
        setup_subject(fname)