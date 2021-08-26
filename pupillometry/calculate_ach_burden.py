# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:42:53 2021

@author: jelma
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
    
def main(subjfile, effectfile):
    """
    Parameters
    ----------
    subjfile : str
        Path to file containing medication info for participants. 
    effectfile : str
        Path to file containing medication code anticholinergic effects. 

    Returns
    -------
    Saves out file with total ACh burden score to same directory as subjfile .
    """
    # Load data files
    effectdf = pd.read_csv(effectfile)
    subjdf = pd.read_csv(subjfile)
    # Get columns containing medication codes
    medcols = list(subjdf.filter(regex='med[0-9]{1,2}_code').columns)
    subjdf = subjdf[['pupalz_id', 'redcap_event_name'] + medcols]
    # Convert to long format
    subjdf_long = subjdf.melt(id_vars=['pupalz_id', 'redcap_event_name'], var_name="med_num", value_name="med_code")
    subjdf_long = subjdf_long[subjdf_long.med_code.notna()]
    # Drugs are listed multiple times due to differen brand names. Keep only unique DRUGID
    effectdf = effectdf.loc[effectdf.ANTICHOL_RATING.notna(),['DRUGID','ANTICHOL_RATING']].drop_duplicates()    
    # Merge participant data with burden ACh effects
    df = pd.merge(subjdf_long, effectdf, how='left', left_on='med_code', right_on='DRUGID')
    df['ANTICHOL_RATING'] = df.ANTICHOL_RATING.fillna(value=0)
    # Calculate sum
    totaldf = df.groupby(['pupalz_id', 'redcap_event_name'])['ANTICHOL_RATING'].sum().reset_index()    
    totaldf = totaldf.rename(columns={'ANTICHOL_RATING':'total_ach'})
    # Save output to same directory as input file
    outdir = os.path.dirname(subjfile)
    tstamp = datetime.now().strftime("%Y-%m-%d")
    outfile = os.path.join(outdir, 'PupAlz_AnticholBurden_' + tstamp + '.csv') 
    totaldf.to_csv(outfile, sep=",", index=False)
    
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('')
        print('USAGE: {} <participant data file> <med code effects file>'.format(os.path.basename(sys.argv[0])))
        print("""'Creates total anti-cholinergic burden score per participants.
                'First two columns should be: pupalz_id, redcap_event_name
                'Medcodes in format dXXXXX should be contained in columnes named medN_code
                'where N ranges from 1 to the total number of medications taken. Creates
                'a file named PupAlz_AnticholBurden_<date>.csv in the same directory 
                'as input file.""")
        print('')

        root = tkinter.Tk()
        root.withdraw()
        # Select files to process
        subjfile = filedialog.askopenfilename(parent=root,
                                                     title='Choose file with participant data',
                                                     filetypes = (("csv files","*.csv"),("all files","*.*")))
        effectfile = filedialog.askopenfilename(parent=root,
                                                     title='Choose file with medication effects',
                                                     filetypes = (("csv files","*.csv"),("all files","*.*")))
        main(subjfile, effectfile)
    else:
        subjfile = sys.argv[1]
        effectfile = sys.argv[2]
        main(subjfile, effectfile)