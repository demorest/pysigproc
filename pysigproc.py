# pysigproc.py -- P. Demorest, 2016/04
#
# Simple functions for generating sigproc filterbank
# files from python.  Not all possible features are implemented.

import sys
import struct
from collections import OrderedDict

class SigprocFile(object):

    ## List of types
    _type = OrderedDict()
    _type['rawdatafile'] = 'string'
    _type['source_name'] = 'string'
    _type['machine_id'] = 'int'
    _type['telescope_id'] = 'int'
    _type['src_raj'] = 'double'
    _type['src_dej'] = 'double'
    _type['data_type'] = 'int'
    _type['fch1'] = 'double'
    _type['nchans'] = 'int'
    _type['nbeams'] = 'int'
    _type['ibeam'] = 'int'
    _type['nbits'] = 'int'
    _type['tstart'] = 'double'
    _type['tsamp'] = 'double'
    _type['nifs'] = 'int'

    def __init__(self,fp=None):
        self.fp = fp
        # init all items to None
        for k in self._type.keys():
            setattr(self, k, None)
        if self.fp is not None:
            self.read_header(fp)

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
        nchar = struct.unpack('i',fp.read(4))
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
            raise RuntimeError('File does not start with HEADER_START.')
        self.hdrbytes += n
        while True:
            (s,n) = self.get_string(self.fp)
            self.hdrbytes += n
            if s == 'HEADER_END': return
            datatype = self._type[s][0]
            datasize = struct.calcsize(datatype)
            val = struct.unpack(datatype,self.fp.read(datasize))
            setattr(self,s,val)
            self.hdrbytes += datasize



