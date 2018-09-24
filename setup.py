#!/usr/bin/env python3

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(name='pysigproc',
      version='0.3',
      description='Python reader/writer for sigproc filterbank files (works with python3 as well)',
      author='Paul Demorest, Devansh Agarwal, Kshitij Aggarwal',
      author_email='pdemores@nrao.edu, da0017@mix.wvu.edu, ka0064@mix.wvu.edu',
      url='http://github.com/devanshkv/pysigproc',
      packages=find_packages(),
      py_modules=['pysigproc','candidate']
     )
