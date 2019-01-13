#!/usr/bin/env python3
import numpy as np
import pylab as plt 
import h5py
from pysigproc import SigprocFile
from scipy.optimize import golden
import tqdm

class Candidate(SigprocFile):
    def __init__(self,fp=None,dm=None,tcand=0,width=0,label=-1,snr=0):
        SigprocFile.__init__(self, fp)
        self.dm=dm
        self.tcand=tcand
        self.width=width
        self.label=label
        self.snr=snr
        self.id=f'cand_tstart_{self.tstart:.12f}_tcand_{self.tcand:.7f}_dm_{self.dm:.5f}_snr_{self.snr:.5f}'
        self.data=None
        self.dedispersed=None

    def save_h5(self,out_dir=None,fnout=None):
        cand_id = self.id
        if fnout is None:
            fnout= cand_id+'.h5'
        if out_dir is not None:
            fnout = out_dir + fnout
        with  h5py.File(fnout, 'w') as f:
            f.attrs['cand_id'] = cand_id
            f.attrs['tcand'] = self.tcand
            f.attrs['dm'] = self.dm
            f.attrs['dm_opt'] = self.dm_opt
            f.attrs['snr'] = self.snr
            f.attrs['snr_opt'] = self.snr_opt
            f.attrs['width'] = self.width
            f.attrs['label'] = self.label

            # Copy over header information as attributes
            for key in list(self._type.keys()):
                if getattr(self, key) is not None: 
                    f.attrs[key] = getattr(self,key)
                else:
                    f.attrs[key] = b'None'

            freq_time_dset = f.create_dataset('data_freq_time', data=self.dedispersed)
            freq_time_dset.dims[0].label = b"time"
            freq_time_dset.dims[1].label = b"frequency"

            if self.dmt is not None:
                dm_time_dset = f.create_dataset('data_dm_time', data=self.dmt)
                dm_time_dset.dims[0].label = b"dm"
                dm_time_dset.dims[1].label = b"time"
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
            tstart = self.tcand - self.dispersion_delay() - self.width*self.tsamp
            if tstart < 0:
                tstart = 0
        if tstop is None:
            tstop = self.tcand + self.dispersion_delay() + self.width*self.tsamp
            if tstop > self.tend:
                tstop = self.tend
        nstart=int(tstart/self.tsamp)
        nsamp=int((tstop-tstart)/self.tsamp)
        if self.width < 2:
            min_samp=256
        else:
            min_samp=self.width*256//2
        if nsamp < min_samp:
            #if number of time samples less than 256, make it 256.
            nstart-= (min_samp-nsamp)//2
            nsamp=min_samp
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
        range_dm=self.dm
        dm_list=self.dm+np.linspace(-range_dm,range_dm,256)
        dmt=np.zeros((256,self.data.shape[0]))
        for ii,dm in enumerate(tqdm.tqdm(dm_list)):
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
        mask[argmax - self.width//2:argmax + self.width//2]=0
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
        except (ValueError,TypeError) as e:
            out=golden(dm2snr,full_output=1,tol=1e-3)
        self.dm_opt = out[0]
        self.snr_opt = -out[1]
        return out[0],-out[1]
