#!/usr/bin/env python3

import pysigproc
import numpy as np

def fil_data(fil_file,tstart,tstop):
    """
    :param fil_file: name of the filterbank file
    :param tstart: start time in seconds
    :param tstop: stop time in seconds
    :return 2d data array, freqs, time series, sampling time 
    """
    fil_obj=pysigproc.SigprocFile(fil_file)
    nstart=int(tstart/fil_obj.tsamp)
    nsamp=int((tstop-tstart)/fil_obj.tsamp)
    img=fil_obj.get_data(nstart=6000,nsamp=1024)[:,0,:]
    freqs=fil_obj.chan_freqs
    ts=np.linspace(tstart,tstop,endpoint=True,num=img.shape[0])
    return(img,freqs,ts,fil_obj.tsamp)

def dedisp(img, dm,freq,t_bin,nchans):
    """
    :param img: 2d array of frequency-time data
    :param dm: dispersion measure
    :param freq: frequency (MHz)
    :param t_bin: Sampling time (s)
    :param nchans: Number of channels
    :return: dedispersed 2d array
    """
    nt, nf = img.shape
    assert nf == nchans
    dmk = 4148808.0
    bw=freq[-1]-freq[0]
    inv_flow_sq = 1.0 / freq[-1] ** 2
    delay_time = np.array([dmk * dm  * (inv_flow_sq - 1/(f_chan**2)) for f_chan in freq])
    delay_bins = np.round(delay_time*1e-3/t_bin).astype('int64')
    dedisp_arr=np.zeros(img.shape)
    for ii, delay in enumerate(delay_bins):
        dedisp_arr[:,ii]=np.roll(img[:,ii],delay)
    return dedisp_arr
