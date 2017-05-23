# pysigproc.py -- P. Demorest, 2016/04
#
# Simple functions for generating sigproc filterbank
# files from python.  Not all possible features are implemented.

import sys
import struct
from collections import OrderedDict
import mmap
import numpy

class SigprocFile(object):

    ## List of types
    _type = OrderedDict()
    _type['rawdatafile'] = 'string'
    _type['source_name'] = 'string'
    _type['machine_id'] = 'int'
    _type['barycentric'] = 'int'
    _type['telescope_id'] = 'int'
    _type['src_raj'] = 'double'
    _type['src_dej'] = 'double'
    _type['az_start'] = 'double'
    _type['za_start'] = 'double'
    _type['data_type'] = 'int'
    _type['fch1'] = 'double'
    _type['foff'] = 'double'
    _type['nchans'] = 'int'
    _type['nbeams'] = 'int'
    _type['ibeam'] = 'int'
    _type['nbits'] = 'int'
    _type['tstart'] = 'double'
    _type['tsamp'] = 'double'
    _type['nifs'] = 'int'

    def __init__(self,fp=None,copy_hdr=None):
        # init all items to None
        for k in self._type.keys():
            setattr(self, k, None)
        if copy_hdr is not None:
            for k in self._type.keys():
                setattr(self,k,getattr(copy_hdr,k))
        if fp is not None:
            try:
                self.fp = open(fp,'r')
            except TypeError:
                self.fp = fp
            self.read_header(self.fp)
            self._mmdata = mmap.mmap(self.fp.fileno(), 0, mmap.MAP_PRIVATE, 
                    mmap.PROT_READ)

    ## See sigproc send_stuff.c

    @staticmethod
    def send_string(val,f=sys.stdout):
        f.write(struct.pack('i',len(val)))
        f.write(val)

    def send_num(self,name,val,f=sys.stdout):
        self.send_string(name,f)
        f.write(struct.pack(self._type[name][0],val))

    def send(self,name,f=sys.stdout):
        if not hasattr(self,name): return
        if getattr(self,name) is None: return
        if self._type[name]=='string':
            self.send_string(name,f)
            self.send_string(getattr(self,name),f)
        else:
            self.send_num(name,getattr(self,name),f)

    ## See sigproc filterbank_header.c

    def filterbank_header(self,fout=sys.stdout):
        self.send_string("HEADER_START",f=fout)
        for k in self._type.keys():
            self.send(k,fout)
        self.send_string("HEADER_END",f=fout)

    ## See sigproc read_header.c

    @staticmethod
    def get_string(fp):
        """Read the next sigproc-format string in the file."""
        nchar = struct.unpack('i',fp.read(4))[0]
        if nchar>80 or nchar<1: 
            return (None, 0)
        out = fp.read(nchar)
        return (out, nchar+4)

    def read_header(self,fp=None):
        """Read the header from the specified file pointer."""
        if fp is not None: self.fp = fp
        self.hdrbytes = 0
        (s,n) = self.get_string(self.fp)
        if s != 'HEADER_START':
            raise RuntimeError("File does not start with HEADER_START (read '%s')" % s)
        self.hdrbytes += n
        while True:
            (s,n) = self.get_string(self.fp)
            self.hdrbytes += n
            if s == 'HEADER_END': return
            if self._type[s] == 'string':
                (v,n) = self.get_string(self.fp)
                self.hdrbytes += n
                setattr(self,s,v)
            else:
                datatype = self._type[s][0]
                datasize = struct.calcsize(datatype)
                val = struct.unpack(datatype,self.fp.read(datasize))[0]
                setattr(self,s,val)
                self.hdrbytes += datasize

    @property
    def dtype(self):
        if self.nbits==8:
            return numpy.uint8
        elif self.nbits==16:
            return numpy.uint16
        elif self.nbits==32:
            return numpy.float32
        else:
            raise RuntimeError('nbits=%d not supported' % self.nbits)

    @property
    def bytes_per_spectrum(self):
        return self.nbits * self.nchans * self.nifs / 8

    @property
    def nspectra(self):
        return self._mmdata.size() / self.bytes_per_spectrum

    @property
    def tend(self):
        return self.tstart + self.nspectra*self.tsamp/86400.0

    def get_data(self,nstart,nsamp):
        """Return nsamp time slices starting at nstart."""
        bstart = int(nstart) * self.bytes_per_spectrum
        nbytes = int(nsamp) * self.bytes_per_spectrum
        b0 = self.hdrbytes + bstart
        b1 = b0 + nbytes
        return numpy.frombuffer(self._mmdata[b0:b1],
                dtype=self.dtype).reshape((-1,self.nifs,self.nchans))

    def unpack(self,nstart,nsamp):
        """Unpack nsamp time slices starting at nstart to 32-bit floats."""
        if self.nbits >= 8:
            return self.get_data(nstart,nsamp).astype(numpy.float32)
        bstart = int(nstart) * self.bytes_per_spectrum
        nbytes = int(nsamp) * self.bytes_per_spectrum
        b0 = self.hdrbytes + bstart
        b1 = b0 + nbytes
        # reshape with the frequency axis reduced by packing factor
        fac = 8 / self.nbits
        d = numpy.frombuffer(self._mmdata[b0:b1],
                dtype=numpy.uint8).reshape(
                        (nsamp,self.nifs,self.nchans/fac))
        unpacked = numpy.empty((nsamp,self.nifs,self.nchans),
                dtype=numpy.float32)
        for i in xrange(fac):
            mask = 2**(self.nbits*i)*(2**self.nbits-1)
            unpacked[...,i::fac] = (d & mask) / 2**(i*self.nbits)
        return unpacked

    @property
    def chan_freqs(self):
        return self.fch1 + numpy.arange(self.nchans)*self.foff
