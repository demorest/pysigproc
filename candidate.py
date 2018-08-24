#!/usr/bin/env python3
import numpy as np
import pylab as plt 
import h5py
import filutils as fu
from pysigproc import SigprocFile

class Candidate(SigprocFile):
    def __init__(self,fp=None,dm=None,tcand=0):
	    SigprocFile.__init__(self, fil_file)
	    self.dm=dm
	    self.tcand=tcand
    
    def dispersion_delay(self,dms=None):
        if dms is None:
            dms=self.dm
        if dms is None:
            return None
        else:
            return 4148808.0*dms*(1/np.min(self.chan_freqs)**2 - 1/np.max(self.chan_freqs)**2)/1000

    def get_chunk(self,tstart=None,tstop=None):
        if tstart is None:
            tstart = self.tcand - self.dispersion_delay()
            if tstart < 0:
                tstart = 0
        if tstop is None:
            tstop = self.tcand + self.dispersion_delay()
            if tstop > cand.tend:
                tstop = cand.tend
        nstart=int(tstart/self.tsamp)
        nsamp=int((tstop-tstart)/self.tsamp)
        self.data=self.get_data(nstart=nstart,nsamp=nsamp)[:,0,:]
        return self
    
    def dedisperse(self,dms=None):
        if dms is None:
            dms=self.dm
        if self.data is not None: 
            nt, nf = self.data.shape
            assert nf == len(self.chan_freqs)
            delay_time = 4148808.0*dms*(1/(self.chan_freqs[0])**2 - 1/(self.chan_freqs)**2)/1000
            delay_bins = np.round(delay_time/self.tsamp).astype('int64')
            dedisp_arr=np.zeros(self.data.shape)
            data=np.zeros(self.data.shape)
            for chan in range(nf):
                data[:,chan]-=self.data.mean(1)
            for ii, delay in enumerate(delay_bins):
                dedisp_arr[:,ii]=np.roll(data[:,ii],delay)
            self.dedispersed = dedisp_arr
        else:
            self.dedipersed = None
        return self

    def dmtime(self):
        start=time.time()
        range=2*self.dm
        dm_list=self.dm+np.linspace(-range,range,100)
        dmt=np.zeros((100,self.data.shape[0]))
        for ii,dm in enumerate(dm_list):
            dmt[ii,:]=self.dedisperse(dms=dm).dedispersed.sum(1)
        print('took', start-time.time(),' s')
        return dmt
