#!/usr/bin/env python3
import numpy as np
import pylab as plt 
import h5py
from pysigproc import SigprocFile
from scipy.optimize import golden

class Candidate(SigprocFile):
    def __init__(self,fp=None,dm=None,tcand=0,width=0,label=np.nan,snr=None):
        SigprocFile.__init__(self, fp)
        self.dm=dm
        self.tcand=tcand
        self.width=width
        self.label=label
        self.snr=snr

    def save_h5(self,fnout=None):
        if fnout is None:
            fnout=f'cand_dm_{self.dm}_snr_{self.snr}_time_{self.tcand}.h5'
        with  h5py.File(fnout, 'w') as f:
            freq_time_dset = f.create_dataset('data_freq_time', data=self.dedispersed)
            freq_time_dset.dims[0].label = b"time"
            freq_time_dset.dims[1].label = b"frequency"

            labels_dset =  f.create_dataset('labels', data=[self.label])
            labels_dset.dims[0].label = b"labels"         

            if self.dmt is not None:
                dm_time_dset = f.create_dataset('data_dm_time', data=self.dmt)
                dm_time_dset.dims[0].label = b"dm"
                dm_time_dset.dims[1].label = b"time"
            #if params is not None:
            #    f.create_dataset('params', data=params)
            #if self.snr is not None:
            #    snr_dset = f.create_dataset('snr', data=self.snr)
            #    snr_dset.dim[0].label = b"snr"
                
            # Copy over header information as attributes
            #for key in self._type.keys():
            #    print(getattr(self, key))
            #    if not getattr(self, key): 
            #        freq_time_dset.attrs[key] = getattr(self, key)


        return fnout
    
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
            if tstop > self.tend:
                tstop = self.tend
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
            for ii, delay in enumerate(delay_bins):
                dedisp_arr[:,ii]=np.roll(self.data[:,ii],delay)
            self.dedispersed = dedisp_arr
        else:
            self.dedispersed = None
        return self

    def dmtime(self):
        range=2*self.dm
        dm_list=self.dm+np.linspace(-range,range,100)
        dmt=np.zeros((100,self.data.shape[0]))
        for ii,dm in enumerate(dm_list):
            dmt[ii,:]=self.dedisperse(dms=dm).dedispersed.sum(1)
        self.dmt=dmt
        return self

    def get_snr(self,time_series=None):
        if time_series is None and self.dedispersed is None:
            return None
        if time_series is None:
            x=self.dedispersed.mean(1)
        else:
            x=time_series
        argmax=np.argmax(x)
        mask=np.ones(len(x),dtype=np.bool)
        mask[argmax - 2*self.width:argmax + 2*self.width]=0
        x-=x[mask].mean()
        std=np.std(x[mask])
        return x.max()/std
    
    def optimize_dm(self):
        if self.data is None:
            return None
        def dm2snr(dm):
            time_series=self.dedisperse(dm).dedispersed.sum(1)
            return -self.get_snr(time_series)
        try:
            out=golden(dm2snr,full_output=1,brack=(-self.dm/2,self.dm,2*self.dm),tol=1e-3)
        except ValueError:
            out=golden(dm2snr,full_output=1,tol=1e-3)
        return out[0],-out[1]

    
