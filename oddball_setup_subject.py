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
import pupil_utils

try:
    # for Python2
    import Tkinter as tkinter
    import tkFileDialog as filedialog
except ImportError:
    # for Python3
    import tkinter
    from tkinter import filedialog

def check_setup(rawdir):
    globstr = os.path.join(rawdir, '*recoded.gazedata')
    files = glob(globstr)
    if os.path.exists(rawdir) & (len(files)>=1):
        setup_status = 1
    else:
        setup_status = 0
    return setup_status


def get_subid(fname):
    subid = re.findall(r'Oddball-\d{3}(?:-\d{2})?', fname)[0].replace('Oddball-','')
    return subid


def get_oddball_session(infile):
    """Returns session as listed in the infile name (1=A, 2=B). If not listed, 
    default to SessionA."""
    if infile.find("Session") == -1:
        session = 'A'
    else:
        session = infile.split("Session")[1][0]
        session = session.replace('1','A').replace('2','B')
    return (session)
    
    
def swap_response(df):
    """ Correct response is swapped. Recode and calculate accuracy"""
    idx1 = np.where(df.CRESP==1)[0]
    idx5 = np.where(df.CRESP==5)[0]
    df.loc[df.index[idx1],'CRESP'] = 5
    df.loc[df.index[idx5],'CRESP'] = 1
    df['ACC'] = np.where(df.CRESP==df.RESP,1,0)
    return df


def recode_gaze_data(fname):
    if (os.path.splitext(fname)[-1] == ".gazedata") | (os.path.splitext(fname)[-1] == ".csv"):
        df = pd.read_csv(fname, sep="\t")
    elif os.path.splitext(fname)[-1] == ".xlsx":
        df = pd.read_excel(fname, parse_dates=False)
    else: 
        raise IOError('Could not open {}'.format(fname))   
    df = swap_response(df)
    return df
    

def rename_gaze_file(fname, subid, session):
    procdir = os.path.dirname(pupil_utils.get_proc_outfile(fname, ''))
    new_fname = ''.join(['Oddball-', str(subid), '-Session', str(session), '_recoded.gazedata'])
    newfile = os.path.join(procdir, new_fname)
    return newfile       
        
    
def setup_subject(filelist):
    for fname in filelist:
        print('Processing {}'.format(fname))
        subid = get_subid(fname)
        session = get_oddball_session(fname)
        df = recode_gaze_data(fname)
        newfile = rename_gaze_file(fname, subid, session)
        df.to_csv(newfile, index=False, sep="\t")
        print('Writing recoded data to {0}'.format(newfile))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('USAGE: {} <raw gazedata file> '.format(os.path.basename(sys.argv[0])))
        print("""Sets up subject and recodes data. Takes pupil data filename as input.
              Performs the following actions:
                  1. Renames subject ID in the gazadata file
                  2. Recodes session based on session number listed in gazedata filename
                  3. Swaps correct response in gazedata file
                  4. Saves out new .gazedata file with "recoded" suffix""")
        
        root = tkinter.Tk()
        root.withdraw()
        # Select files to process
        filelist = filedialog.askopenfilenames(parent=root,
                                                    title='Choose raw oddball pupil gazedata file to recode')
        filelist = list(filelist)
        # Run script
        setup_subject(filelist)

    else:
        filelist = [os.path.abspath(f) for f in sys.argv[1:]]
        setup_subject(filelist)