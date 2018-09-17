import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
import matlab_wrapper


def zscore(x):
    """ Z-score numpy array or pandas series """
    return (x - x.mean()) / x.std()


def get_proc_outfile(infile, suffix):
    """Take infile to derive outdir. Changes path from raw to proc
    and adds suffix to basename."""
    outdir = os.path.dirname(infile)
    outdir = outdir.replace("raw", "proc")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fname = os.path.splitext(os.path.basename(infile))[0].split('_')[1] + suffix
    outfile = os.path.join(outdir, fname)
    return outfile
    

def get_outfile(infile, suffix):
    """Take infile to derive outdir. Adds suffix to basename."""
    outdir = os.path.dirname(infile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fname = os.path.splitext(os.path.basename(infile))[0] + suffix
    outfile = os.path.join(outdir, fname)
    return outfile

     
def get_blinks(diameter, validity):
    """Get vector of blink or bad trials by combining validity field and any 
    samples with a change in dilation greater than 1mm. Mark ~100ms before 
    blink onset and after blink offset as a blink."""
    invalid = validity==4
    bigdiff = diameter.diff().abs()>1
    blinks = np.where(invalid | bigdiff, 1, 0)
#    startidx = np.where(np.diff(blinks)==1)[0]
#    stopidx = np.where(np.diff(blinks)==-1)[0]
#    startblinks = np.concatenate((startidx, startidx-1,startidx-2))
#    stopblinks = np.concatenate((stopidx+1, stopidx+2,stopidx+3))
#    blinks[np.concatenate((startblinks, stopblinks))] = 1
    return blinks


def deblink(df):
    """ Set dilation of all blink trials to nan."""
    df['BlinksLeft'] = get_blinks(df.DiameterPupilLeftEye, df.ValidityLeftEye)
    df['BlinksRight'] = get_blinks(df.DiameterPupilRightEye, df.ValidityRightEye)
    df.loc[df.BlinksLeft==1, "DiameterPupilLeftEye"] = np.nan
    df.loc[df.BlinksRight==1, "DiameterPupilRightEye"] = np.nan    
    df['BlinksLR'] = np.where(df.BlinksLeft+df.BlinksRight>=2, 1, 0)
    return df


def butter_bandpass(lowcut, highcut, fs, order):
    """Takes the low and high frequencies, sampling rate, and order. Normalizes
    critical frequencies by the nyquist frequency."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(signal, lowcut=0.01, highcut=4., fs=30., order=3):
    """Get numerator and denominator coefficient vectors from Butterworth filter
    and then apply bandpass filter to signal."""

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, signal)
    return y
    

def butter_lowpass(highcut, fs, order):
    """Takes the high frequencies, sampling rate, and order. Normalizes
    critical frequencies by the nyquist frequency."""
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a


def butter_lowpass_filter(signal, highcut=4., fs=30., order=3):
    """Get numerator and denominator coefficient vectors from Butterworth filter
    and then apply higpass filter to signal."""
    b, a = butter_lowpass(highcut, fs, order=order)
    y = filtfilt(b, a, signal)
    return y



def get_gradient(df, gradient_crit=4):
    diffleft = df.DiameterPupilLeftEye.replace(-1,np.nan).diff()
    diffright = df.DiameterPupilRightEye.replace(-1,np.nan).diff()
    diffleftmean = np.nanmean(diffleft)
    diffrightmean = np.nanmean(diffright)
    diffleftstd = np.nanstd(diffleft)
    diffrightstd = np.nanstd(diffright)
    gradientleft = diffleftmean + (gradient_crit*diffleftstd)
    gradientright = diffrightmean + (gradient_crit*diffrightstd)
    gradient = np.mean([gradientleft, gradientright])
    return gradient
    
    
def chap_deblink(raw_pupil, gradient, gradient_crit=4, z_outliers=2.5, zeros_outliers = 20,
                 data_rate=30, linear_interpolation=True, trial2show=0): 
    matlab = matlab_wrapper.MatlabSession()
#    matlab.eval(os.path.abspath(__file__))
    clean_pupil, blinkidx, blinks = matlab.workspace.fix_blinks_PupAlz(np.atleast_2d(raw_pupil).T.tolist(), 
                                                       float(z_outliers), float(zeros_outliers), 
                                                       float(data_rate), linear_interpolation, 
                                                       gradient, 
                                                       trial2show, 
                                                       nout=3)
    return clean_pupil, blinks


def resamp_filt_data(df, bin_length='33ms', filt_type='band', string_cols=None):
    """Takes dataframe of raw pupil data and performs the following steps:
        1. Smooths left and right pupil by taking average of 2 surrounding samples
        2. Averages left and right pupils
        3. Creates a timestamp index with start of trial as time 0. 
        4. Resamples data to 30Hz to standardize timing across trials.
        5. Nearest neighbor interpolation for blinks, trial, and subject level data 
        6. Linear interpolation (bidirectional) of dilation data
        7. Applies Butterworth bandpass filter to remove high and low freq noise
        8. If string columns should be retained, forward fill and merge with resamp data
        """
    df['DiameterPupilLeftEyeSmooth'] = df.DiameterPupilLeftEye.rolling(5, center=True).mean()  
    df['DiameterPupilRightEyeSmooth'] = df.DiameterPupilRightEye.rolling(5, center=True).mean()  
    df['DiameterPupilLRSmooth'] = df[['DiameterPupilLeftEyeSmooth','DiameterPupilRightEyeSmooth']].mean(axis=1, skipna=True)
    df['Time'] = (df.TETTime - df.TETTime.iloc[0]) / 1000.
    df['Timestamp'] = pd.to_datetime(df.Time, unit='s')
    df = df.set_index('Timestamp')
    dfresamp = df.resample(bin_length).mean()
    dfresamp['Subject'] = df.Subject[0]
    nearestcols = ['Subject','Session','TrialId','CRESP','ACC','RT',
                   'BlinksLeft','BlinksRight','BlinksLR'] 
    dfresamp[nearestcols] = dfresamp[nearestcols].interpolate('nearest')
    resampcols = ['DiameterPupilLRSmooth','DiameterPupilLeftEyeSmooth','DiameterPupilRightEyeSmooth']
    newresampcols = [x.replace('Smooth','Resamp') for x in resampcols]
    dfresamp[newresampcols] = dfresamp[resampcols].interpolate('linear', limit_direction='both')
    if filt_type=='band':
        dfresamp['DiameterPupilLRFilt'] = butter_bandpass_filter(dfresamp.DiameterPupilLRResamp)        
        dfresamp['DiameterPupilLeftEyeFilt'] = butter_bandpass_filter(dfresamp.DiameterPupilLeftEyeResamp)
        dfresamp['DiameterPupilRightEyeFilt'] = butter_bandpass_filter(dfresamp.DiameterPupilRightEyeResamp)    
    elif filt_type=='low':
        dfresamp['DiameterPupilLRFilt'] = butter_lowpass_filter(dfresamp.DiameterPupilLRResamp)        
        dfresamp['DiameterPupilLeftEyeFilt'] = butter_lowpass_filter(dfresamp.DiameterPupilLeftEyeResamp)
        dfresamp['DiameterPupilRightEyeFilt'] = butter_lowpass_filter(dfresamp.DiameterPupilRightEyeResamp)           
    dfresamp['Session'] = dfresamp['Session'].astype('int')    
    dfresamp['TrialId'] = dfresamp['TrialId'].astype('int')
    if string_cols:
        stringdf = df[string_cols].resample(bin_length).ffill()
        dfresamp = dfresamp.merge(stringdf, left_index=True, right_index=True)
    return dfresamp



def plot_qc(dfresamp, infile):
    """Plot raw signal, interpolated and filter signal, and blinks"""
    outfile = get_outfile(infile, '_PupilLR_plot.png')
    signal = dfresamp.DiameterPupilLRResamp.values
    signal_bp = dfresamp.DiameterPupilLRFilt.values
    blinktimes = dfresamp.BlinksLR.values
    plt.plot(range(len(signal)), signal, sns.xkcd_rgb["pale red"], 
         range(len(signal_bp)), signal_bp+np.nanmean(signal), sns.xkcd_rgb["denim blue"], 
         blinktimes, sns.xkcd_rgb["amber"], lw=1)
    plt.savefig(outfile)
    plt.close()
