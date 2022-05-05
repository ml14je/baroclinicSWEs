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
    x0, xN, y0, yN = np.array(param.bboxes[1])/param.L_R
    
    if (x < x0 + x_padding) | (x > xN - x_padding):
        R = + magnitude
    
    elif (y < y0 + y_padding) | (y > yN - y_padding):
        R = + magnitude
        
    else:
        R = 0
        
    return R


def sponge2(x, y, param, magnitude=.05, x_padding=.01, y_padding=.01,
            xc=None, yc=None, gauss_parameter=6):
    import numpy as np
    x0, xN, y0, yN = np.array(param.bboxes[1]) * 1e3 /param.L_R
    
    if xc is None:
        xc = .5 * (x0 + xN)
        
    if yc is None:
        yc = .05

    px, py = (xN - x0 - 1.7*x_padding)/2, (yN - y0 - 1.7*y_padding)/2
    n = gauss_parameter
        
    return  magnitude * \
        (1 - (np.exp(-.5*abs((x-xc)/px)**n) * np.exp(-.5*abs((y-yc)/py)**n))**(1/n))

@np.vectorize
def sponge3(x, y, param, magnitude=.05, y_padding=.01,
            yc=None, gauss_parameter=5):
    x0, xN, y0, yN = np.array(param.bboxes[1])/param.L_R
        
    if yc is None:
        yc = .5 * (y0 + yN)

    py = (yN - y0 - 2*y_padding)/2
    n = gauss_parameter
    R = magnitude * (1 - (np.exp(-.5*abs((y-yc)/py)**n))**(1/n))
        
    return R

def plot_friction(param, func, magnitude=.05, x_padding=.005, y_padding=.005,
                  plot_name=''):
    r = magnitude
    x_pad, y_pad = x_padding, y_padding
    x0, xN, y0, yN = np.array(param.bboxes[1]) * 1e3 / param.L_R
    x, y = np.linspace(x0, xN, 101), np.linspace(y0, yN, 101)
    X, Y = np.meshgrid(x, y)

    R = func(.85*X, Y,
             param,
             magnitude=r,
             x_padding=x_pad,
             y_padding=y_pad)
    
    R[abs(X)>.9*.4/2] = 1
    
    from ppp.Plots import plot_setup
    from matplotlib import patches

    fig, ax = plot_setup('Along-shore (km)', 'Cross-shore (km)')
    L_R = param.L_R * 1e-3
    c = ax.matshow(
        R,
        cmap="seismic",
        vmax=r,
        vmin=-r,
        extent= np.array(param.bboxes[1])  / L_R,
        aspect="auto",
        origin="lower",
    )
    
    # x0, xN, y0, yN = np.array(param.bboxes[1])
    rect = patches.Rectangle((x0+x_padding,
                              y0+y_padding ),
                              (xN-x0)-2*x_padding,
                              (yN-y0)-2*y_padding,
                              linewidth=3,
                              edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.set_aspect('equal')
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
    from barotropicSWEs.Configuration import configure     
    param = configure.main()
    
    bbox_barotropic = (-100, 100, 0, 200)
    bbox_baroclinic = (-500, 500, -400, 600)
    param.bboxes = [bbox_barotropic, bbox_baroclinic]
    plot_friction(param, sponge2,
                  magnitude=1, x_padding=.2,
                  y_padding=.2, plot_name='') 
    
    