#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:34:58 2018

@author: jelman

Takes as input datafiles created by digitspan_proc_subject.py. This includes:
    DigitSpan_<subject>_ProcessedPupil.csv
    
Extracts dilation from timepoint of interest (i.e., last second) in each load.
Plots group level PTSC. Output can be used for statistical analysis.
"""

from __future__ import division, print_function, absolute_import
import os
import sys
import pandas as pd
import seaborn as sns
from datetime import datetime
from glob import glob


def glob_files(datadir, suffix):
    globstr = os.path.join(datadir, '*'+suffix)
    return glob(globstr)
    
    
def get_sess_data(datadir):
    sess_filelist = glob_files(datadir, suffix='_ProcessedPupil.csv')
    sess_list = []
    for sub_file in sess_filelist:
        subdf = pd.read_csv(sub_file)
        sess_list.append(subdf)
    sessdf = pd.concat(sess_list).reset_index(drop=True)
    sessdf = sessdf.sort_values(by=['Subject', 'Load', 'Timestamp'])
    sessdf = sessdf.groupby(['Subject','Load']).last().reset_index()
    # Filter for loads that have data at the last second
    idx = sessdf.Timestamp.str.slice(-2).values.astype('int') == sessdf.Load.values+1
    return sessdf.loc[idx,:] 



def unstack_conditions(dflong):
    
    df = dflong.pivot(index="Subject", columns='Load', 
                      values=['Dilation', 'Baseline', 'Diameter', 'BlinkPct', 'ntrials'])
    df.columns = ['_'.join([str(col[1]),str(col[0])]).strip() for col in df.columns.values]
    df = df.reset_index()
    return df

 
    
def proc_group(datadir):
    sessdf_long = get_sess_data(datadir)
    tstamp = datetime.now().strftime("%Y%m%d")
    sessdf_long_outfile = os.path.join(datadir, 'digitspan_long_group_' + tstamp + '.csv')
    sessdf_long.to_csv(sessdf_long_outfile, index=False)
    sessdf_wide = unstack_conditions(sessdf_long)
    sessdf_wide_outfile = os.path.join(datadir, 'digitspan_wide_group_' + tstamp + '.csv')
    sessdf_wide.to_csv(sessdf_wide_outfile, index=False)
    plot_outfile = os.path.join(datadir, 'digitspan_group_plot_' + tstamp + '.png')
    sns.set_context('notebook')
    sns.set_style('ticks')
    p = sns.catplot(x="Load", y="Dilation",
                    kind="point", capsize=.2, aspect=1.5, 
                    data=sessdf_long)
    p.despine()
    p.savefig(plot_outfile)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('USAGE: {} <data directory> '.format(os.path.basename(sys.argv[0])))
        print('Searches for datafiles created by digitspan_proc_subject.py for use as input.')
        print('This includes:')
        print('  DigitSpan_<subject>_ProcessedPupil.csv')
        print('Extracts dilation from timepoint of interest (i.e., last second) in each load.')
        print('Plots group level PTSC. Output can be used for statistical analysis.')
        print('')
    else:
        datadir = sys.argv[1]
        proc_group(datadir)