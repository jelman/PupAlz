from __future__ import division, print_function, absolute_import
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
# import matlab_wrapper
from scipy.signal import fftconvolve
from nilearn.glm import ARModel, OLSModel


def get_fname_subid(fname):
    """Given the input files, extract subject ID from basename"""
    fname_base = os.path.basename(fname)  
    try:
        subid = re.search(r'(\d+)-\d.[xlsx|gazedata]', fname_base, re.IGNORECASE).group(1)
        return subid
    except AttributeError:
        print("Could not find valid subject ID in path of input file.")
    
def get_subid(sub_col, fname):
    """Given a column containing subject ID, checks to see that only one
    ID is present. If one ID only is found, returns this ID. Otherwise raises 
    an exception. Checks subject ID in file against ID in filename. Raises 
    an exception if hey do not match"""
    unique_subid = sub_col.unique()
    if len(unique_subid) == 1:
        subid = str(unique_subid[0])
    else:
        raise Exception('Found multiple subject IDs in file: {}'.format(unique_subid))
    fname_subid = get_fname_subid(fname)
    if subid == fname_subid:
        return subid
    else:
        raise Exception('Subject ID in file {0} does not match filename: {1}'.format(unique_subid, fname))   
    
def get_tpfolder(fname):
    """Given a file path of input file, extract timepoint based on Timepoint folder."""
    try:
        tp = re.search(r'Timepoint (\d+)', fname, re.IGNORECASE).group(1)
        return tp
    except AttributeError:
        print("Could not find valid timepoint folder in path of input file.")

    
def get_timepoint(sess_col, fname):
    """Given a column containing session ID, checks to see that only one
    number is present and that it matches timepoint specified in path of input 
    file. If both conditions are satisfied, returns this session number. Otherwise 
    raises an exception."""
    unique_sess = sess_col.unique()
    if len(unique_sess) == 1:
        sess = str(unique_sess[0])
    else:
        raise Exception('Found multiple subject IDs in file: {}'.format(unique_sess))
    tp_folder = get_tpfolder(fname)
    if sess == tp_folder:
        return sess
    else:
        raise Exception("Timepoint folder does not match session number in file.")
    
        
def zscore(x):
    """ Z-score numpy array or pandas series """
    return (x - x.mean()) / x.std()


def get_proc_outfile(infile, suffix):
    """Take infile to derive outdir. Changes path from raw to proc
    and adds suffix to basename."""
    outdir = os.path.dirname(infile)
    outdir = re.sub('Raw Pupil Data', 'Processed Pupil Data', outdir, flags=re.IGNORECASE)
    outdir = re.sub("(Gaze|Edat) data/","", outdir, flags=re.IGNORECASE)
    if not os.path.exists(outdir):
        'Output directory does not exist, creating now: "{0}"'.format(outdir)
        os.makedirs(outdir)
    fname = os.path.splitext(os.path.basename(infile))[0] + suffix
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

def get_iqr(x):
    try:
        q75, q25 = np.percentile(x.dropna(), [75 ,25])
    except IndexError:
        print('Cannot calculate quartile from array of nan')
        q75, q25 = np.nan, np.nan
    iqr = q75 - q25
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    return min, max


def get_blinks(diameter, validity, pupilthresh_hi=5., pupilthresh_lo=1., gradient_crit=4, n_timepoints=1):
    """Get vector of blink or bad trials. Combines validity field, any 
    samples with a change in dilation greater than 1mm, any sample that is 
    outside 2mm from the median."""
    invalid = validity==4
    diffmin, diffmax = get_iqr(diameter.diff(n_timepoints))
    bigdiff = (np.abs(diameter.diff(n_timepoints)) < diffmin) | (np.abs(diameter.diff(-1*n_timepoints)) > diffmax)
    zoutliers = np.abs(zscore(diameter)) > 2.5
    mindiameter, maxdiameter = get_iqr(diameter)
    diameter_outliers = (diameter < mindiameter) | (diameter > maxdiameter) 
    pupil_outlier = (diameter > pupilthresh_hi) | (diameter < pupilthresh_lo)
    blinks = np.where(invalid | bigdiff | zoutliers | diameter_outliers | pupil_outlier, 1, 0)
    return blinks


def deblink(dfraw, **kwargs):
    """ Set dilation of all blink trials to nan."""
    df = dfraw.copy()
    df.loc[df.DiameterPupilLeftEye<0, 'DiameterPupilLeftEye'] = np.nan
    df.loc[df.DiameterPupilRightEye<0, 'DiameterPupilRightEye'] = np.nan
    df['BlinksLeft'] = get_blinks(df.DiameterPupilLeftEye, df.ValidityLeftEye, **kwargs)
    df['BlinksRight'] = get_blinks(df.DiameterPupilRightEye, df.ValidityRightEye, **kwargs)
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
    and then apply lowpass filter to signal."""
    b, a = butter_lowpass(highcut, fs, order=order)
    y = filtfilt(b, a, signal)
    return y



def get_gradient(diameter, gradient_crit=4, n_timepoints=1):
    diff = diameter.replace(-1,np.nan).diff(n_timepoints)
    diffmean = np.nanmean(diff)
    diffstd = np.nanstd(diff)
    gradient = diffmean + (gradient_crit*diffstd)
    return gradient
    
    
# def chap_deblink(raw_pupil, gradient, gradient_crit=4, z_outliers=2.5, zeros_outliers = 20,
#                  data_rate=30, linear_interpolation=True, trial2show=0): 
#     matlab = matlab_wrapper.MatlabSession()
# #    matlab.eval(os.path.abspath(__file__))
#     clean_pupil, blinkidx, blinks = matlab.workspace.fix_blinks_PupAlz(np.atleast_2d(raw_pupil).T.tolist(), 
#                                                        float(z_outliers), float(zeros_outliers), 
#                                                        float(data_rate), linear_interpolation, 
#                                                        gradient, 
#                                                        trial2show, 
#                                                        nout=3)
#     if np.all(clean_pupil==0):
#         clean_pupil.fill(np.nan)
#         blinks.fill(np.nan)
#     return clean_pupil, blinks


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
    # Smooth the pupil diameter data
    df['DiameterPupilLeftEyeSmooth'] = df.DiameterPupilLeftEye.rolling(5, center=True).mean()  
    df['DiameterPupilRightEyeSmooth'] = df.DiameterPupilRightEye.rolling(5, center=True).mean()  
    df['DiameterPupilLRSmooth'] = df[['DiameterPupilLeftEyeSmooth','DiameterPupilRightEyeSmooth']].mean(axis=1, skipna=True)

    # Convert the time to seconds since the start of the experiment
    df['Time'] = (df.TETTime - df.TETTime.iloc[0]) / 1000.

    # Convert the time to a datetime object and set it as the index
    df['Timestamp'] = pd.to_datetime(df.Time, unit='s')
    df = df.set_index('Timestamp')
    # Resample the data to 100 ms bins
    dfresamp = df.select_dtypes(exclude=['object']).resample(bin_length, closed='right', label='right').mean()
    # Fill in missing values by interpolating from nearest value
    dfresamp['Subject'] = df.Subject[0]
    nearestcols = ['Subject','Session','TrialId','CRESP','ACC','RT',
                   'BlinksLeft','BlinksRight','BlinksLR'] 
    dfresamp[nearestcols] = dfresamp[nearestcols].interpolate('nearest')
    # Round the blinks to nearest whole number
    dfresamp[['BlinksLeft','BlinksRight','BlinksLR']] = dfresamp[['BlinksLeft','BlinksRight','BlinksLR']].round()
    # Interpolate the pupil diameter to fill in missing values
    resampcols = ['DiameterPupilLRSmooth','DiameterPupilLeftEyeSmooth','DiameterPupilRightEyeSmooth']
    newresampcols = [x.replace('Smooth','Resamp') for x in resampcols]
    dfresamp[newresampcols] = dfresamp[resampcols].interpolate('linear', limit_direction='both')    
    # Filter the pupil data
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


def pupil_irf(x, s1=50000., n1=10.1, tmax=0.930):
    return s1 * ((x**n1) * (np.e**((-n1*x)/tmax)))


def orthogonalize(y, x):
    """Orthogonalize variable y with respect to variable x. Convert 1-d array
    to 2-d array with shape (n, 1)"""
    yT = np.atleast_2d(y).T
    xT = np.atleast_2d(x).T
    model = OLSModel(xT).fit(yT)
    return model.residuals.squeeze()


def convolve_reg(event_ts, kernel):
    return fftconvolve(event_ts, kernel, 'full')[:-(len(kernel)-1)]

    
def regressor_tempderiv(event_ts, kernel_x, s1=50000., n1=10.1, tmax=0.930):
    """Takes an array of event onset times and an array of timepoints
    within each event. First calculates a kernel based on the pupil irf, as 
    well as the temporal derivative. COnvolves the event onset times with both 
    to get an event regressor and regressor for the temporal derivative. 
    Then orthogonalizes the temporal derivative regressor with respect to the 
    event regressor."""
    kernel = pupil_irf(kernel_x, s1=s1, n1=n1, tmax=tmax)
    dkernel = d_pupil_irf(kernel_x,  s1=s1, n1=n1, tmax=tmax)
    event_reg = convolve_reg(event_ts, kernel)
    td_reg = convolve_reg(event_ts, dkernel)
    td_reg_orth = orthogonalize(td_reg, event_reg)
    return event_reg, td_reg_orth


def d_pupil_irf(x, s1=50000., n1=10.1, tmax=0.930):
    y = pupil_irf(x)
    dy = np.zeros(y.shape,np.float)
    dy[0:-1] = np.diff(y)/np.diff(x)
    dy[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
    return dy


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
