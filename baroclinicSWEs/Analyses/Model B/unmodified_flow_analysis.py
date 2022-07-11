#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Fri Jul  8 11:05:02 2022

"""
import numpy as np

if __name__=='__main__':
    import pickle
    alpha_values = np.round(np.linspace(.02, 1, 50), 3) #change here
    beta_values = np.round(np.linspace(.02, 1, 50), 3) #change here
    
    


    folder_name = "Energies"
    data_name = "CanyonWidth=5km_Order=3"
    with open(f"{folder_name}/{data_name}.pkl", "rb") as inp:
        data_canyon = pickle.load(inp)


    data_name += "_unperturbed_barotropic_flow"
    with open(f"{folder_name}/{data_name}.pkl", "rb") as inp:
        data_slope = pickle.load(inp)
        
    canyon_arr = np.zeros((50, 50))
    slope_arr = np.zeros((50, 50))
    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):
            for k in range(4):
                canyon_arr[i, j] += data_canyon[(alpha, beta)][k]
                slope_arr[i, j] += data_slope[(alpha, beta)][k]

    beta_values = np.append(np.array([0]), beta_values, 0)
    A, B = np.meshgrid(alpha_values, beta_values)
        
    