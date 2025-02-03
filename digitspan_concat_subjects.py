# -*- coding: utf-8 -*-
"""
This script will load the selected subject files of processed pupil data and 
concatenate them into one large group file. The resulting file will have dilation
for every second at each load (but averaged across trials within a load). 

Note: Visual QC metrics should be used to exclude bad conditions.
"""

import pandas as pd
import os, sys
from datetime import datetime

try:
    # for Python2
    import Tkinter as tkinter
    import tkFileDialog as filedialog
except ImportError:
    # for Python3
    import tkinter
    from tkinter import filedialog


def main(filelist, outdir):
    """Takes list of files and concatenates. Writes out to outfile."""
    allsubs = []
    for fname in filelist:
        subdf = pd.read_csv(fname)
        allsubs.append(subdf)
        
    alldf = pd.concat(allsubs).reset_index(drop=True)
    tstamp = datetime.now().strftime("%Y-%m-%d")
    outname = 'digitspan_allsubjects_' + tstamp + '.csv'  
    outfile = os.path.join(outdir, outname)    
    alldf.to_csv(outfile, index=False)
    
    
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('')
        print('USAGE: {} <processed subject pupil file> '.format(os.path.basename(sys.argv[0])))
        print("""Concatenate individual subject files. Resulting group file will
              contain dilation at each second, averaged across trials of a 
              given load.""")
        root = tkinter.Tk()
        root.withdraw()
        # Select files to process
        filelist = filedialog.askopenfilenames(parent=root,
                                              title='Choose files to concatenate',
                                              filetypes = (("processed files","*_ProcessedPupil.csv"),("all files","*.*")))
        filelist = list(filelist)
        outdir  = filedialog.askdirectory()
        # Run script
        main(filelist, outdir)

    else:
        filelist = [os.path.abspath(f) for f in sys.argv[1:]]
        proc_subject(filelist)

