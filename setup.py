#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7: Wed Jul  30 18:21:21 2021
"""

##from setuptools import setup
from distutils.core import setup

setup(name='Baroclinic SWEs DG-FEM',
      version= '0.1',
      description= 'Baroclinic SWEs DG-FEM',
      url= 'http://github.com/ml14je/baroclinicSWEs',
      author= 'Joseph Elmes',
      author_email= 'ml14je@leeds.ac.uk',
      license='None',
      install_requires=[
          'numpy', 'scipy', 'pandas', 'matplotlib',
          'sympy', 'bottleneck', 'cython', 'numba', 'dill', 'ppp',
          'oceanmesh'
#          'ChannelWaves1D', 'ppp', 'oceanmesh'
      ],
      packages=['baroclinicSWEs'],
      zip_safe=False)
