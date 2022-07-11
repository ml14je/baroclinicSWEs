#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Fri Jun 24 22:57:58 2022

"""
import numpy as np

if __name__=='__main__':
    import pickle

    folder_name = "Energies"
    data_name = "CanyonWidth=15km_Order=3"
    
    with open(f"{folder_name}/{data_name}.pkl", "rb") as inp:
        energies = pickle.load(inp)
        
    print(energies)
    
    for a in [.2, .8]:
        for b in [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
            del energies[(a, b)]
            
    print(energies)
    
    with open(f"{folder_name}/{data_name}.pkl", "wb") as outp:
        pickle.dump(energies, outp, pickle.HIGHEST_PROTOCOL)
        
    
