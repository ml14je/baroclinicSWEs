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
    from ppp.Plots import plot_setup, add_colorbar, save_plot
    
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
    
    slope_val = 0
    for k in range(4):
        slope_val += data_canyon["slope"][k]
        
    
    
    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):
            for k in range(4):
                canyon_arr[j, i] += data_canyon[(alpha, beta)][k]
                slope_arr[j, i] += data_slope[(alpha, beta)][k]
                
                
    
    A, B = np.meshgrid(alpha_values, beta_values)
    import matplotlib.pyplot as pt
    slope_arr -= slope_val
    canyon_arr -= slope_val
    
    mags = [35, 35, 25, 33]
    N = [29, 29, 21, 23]
    
    for i, vals in enumerate([canyon_arr, slope_arr, canyon_arr-slope_arr,
                              100*(canyon_arr-slope_arr)/(slope_val+canyon_arr)]):
        fig, ax = plot_setup('$\\alpha$', '$\\beta$', scale=.7)
        if i < 3:
            vals *= 1e-6
        mag = mags[i]
        levels_ = np.linspace(-mag, mag, N[i])
    # print(np.max(100*(slope_arr-canyon_arr)/canyon_arr))
        c = ax.contourf(vals,
                        extent=[alpha_values[0], alpha_values[-1],
                                beta_values[0], beta_values[-1]],
                    cmap='seismic', levels=levels_,
                    extend='both')
        cbar = add_colorbar(c)
        
        if i == 1 :
            c2 = ax.contour(A, B, vals, [0],
                            colors='k')
            cbar.add_lines(c2)
            
        if i < 3:
            y_lab = "Change in tidal dissipation (MW)"
            
        else:
            y_lab = "Relative change in tidal dissipation (%)"
            
        
        
        cbar.ax.tick_params(labelsize=16)
        
        cbar.ax.set_ylabel(y_lab, rotation=270,
                            fontsize=16, labelpad=20)
        ax.set_aspect('equal')
        pt.show()
        
    