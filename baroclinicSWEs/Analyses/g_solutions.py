#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Thu May  5 13:50:24 2022

"""
import numpy as np

class global_solutions(object):
    def __init__(self, bbox, ω, inner_funcs, outer_funcs):
        x0, xN, y0, yN = bbox
        self.u = lambda x, y, t : \
            inner_funcs[0](x, y) * np.exp(-1j * ω * t) * \
                (x > x0) * (x < xN) * (y > y0) * (y < yN) + \
            outer_funcs[0](x, y, t) * \
                (1 - (x > x0) * (x < xN) * (y > y0) * (y < yN))

        self.v = lambda x, y, t : \
            inner_funcs[1](x, y) * np.exp(-1j * ω * t) * \
                (x > x0) * (x < xN) * (y > y0) * (y < yN) + \
            outer_funcs[1](x, y, t) * \
                (1 - (x > x0) * (x < xN) * (y > y0) * (y < yN))

        self.p = lambda x, y, t : \
            inner_funcs[2](x, y) * np.exp(-1j * ω * t) * \
                (x > x0) * (x < xN) * (y > y0) * (y < yN) + \
            outer_funcs[2](x, y, t) * \
                (1 - (x > x0) * (x < xN) * (y > y0) * (y < yN))