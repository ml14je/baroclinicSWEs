#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Wed Dec  8 23:06:06 2021

"""
import numpy as np

@np.vectorize
def sponge1(x, y, param, magnitude=.05, x_padding=.01, y_padding=.01):
    x0, xN, y0, yN = param.bbox
    
    if (x < x0 + x_padding) | (x > xN - x_padding):
        R = + magnitude
    
    elif (y < y0 + y_padding) | (y > yN - y_padding):
        R = + magnitude
        
    else:
        R = 0
        
    return R

def plot_friction(param, func, magnitude=.05, x_padding=.005, y_padding=.005,
                  plot_name=''):
    r = magnitude
    x_pad, y_pad = x_padding, y_padding
    x0, xN, y0, yN = param.bbox
    x, y = np.linspace(x0, xN, 101), np.linspace(y0, yN, 101)
    X, Y = np.meshgrid(x, y)

    R = func(X, Y,
             param,
             magnitude=r,
             x_padding=x_pad,
             y_padding=y_pad)
    
    from ppp.Plots import plot_setup

    fig, ax = plot_setup('Along-shore (km)', 'Cross-shore (km)')
    L_R = param.L_R * 1e-3
    c = ax.matshow(
        R,
        cmap="seismic",
        vmax=r,
        vmin=-r,
        extent=[x0*L_R, xN*L_R,
                y0*L_R, yN*L_R],
        aspect="auto",
        origin="lower",
    )
    fig.colorbar(c, ax=ax)

    if plot_name:
        from ppp.Plots import save_plot
        if plot_name.upper() == 'SAVE':
            plot_name = 'Friction'
        save_plot(fig, ax, plot_name)
        
    else:
        import matplotlib.pyplot as pt
        pt.show()


if __name__ == '__main__':
    import configure
    param, args = configure.main()
    plot_friction(param, sponge1, magnitude=.05, x_padding=.01,
                  y_padding=.005, plot_name='')
    
    