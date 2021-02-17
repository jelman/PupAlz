# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:42:53 2021

@author: jelma
"""
import pandas as pd
import os, sys

    
def main(fname):
    df = pd.read_csv(fname)
    medcols = list(df.filter(regex='med[0-9]{1,2}_code').columns)
    med = df = df[['pupalz_id', 'redcap_event_name'] + medcols]
    


if __name__ == '__main__':
if len(sys.argv) == 1:
    print('USAGE: {} <file> '.format(os.path.basename(sys.argv[0])))
    print('Creates total anti-cholinergic burden score per participants.')
    print('First to columns should be: pupalz_id, redcap_event_name')
    print('Medcodes in format dXXXXX should be contained in columnes named medN_code')
    print('where N ranges from 1 to the total number of medications taken. Creates')
    print('a file named PupAlz_total_ach_burden_<date>.csv in the same directory ')
    print('as input file.')
else:
    datadir = sys.argv[1]
    main(fname)