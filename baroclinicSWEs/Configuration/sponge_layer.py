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
            xc=None, yc=None, gauss_parameter=4):
    import numpy as np
    x0, xN, y0, yN = np.array(param.bboxes[1]) * 1e3 /param.L_R
    
    if xc is None:
        xc = .5 * (x0 + xN)
        
    if yc is None:
        yc = 0

    px, py = (xN - x0 - 2*x_padding)/2, (yN - y0 - 2*y_padding)/2
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

def lavelle_x(x, param, magnitude=.05, x_padding=.01,
            xc=0, D=200):
    D *= 1e3/param.L_R
    x0, xN, y0, yN = np.array(param.bboxes[1])/param.L_R
        

    σx = ((abs(x - xc) - x_padding)/D)**2 * (abs(x-xc) - x_padding > 0)
    σx[σx > 1] = 1

    return σx

def lavelle_y(y, param, magnitude=.05, y_padding=.01,
            yc=None, D=200):
    D *= 1e3/param.L_R
    x0, xN, y0, yN = np.array(param.bboxes[1])/param.L_R
        

    if yc is None:
        yc = .5 * (y0 + yN)
    
    σy = ((abs(y - yc) - y_padding)/D)**2 * (abs(y-yc) - y_padding > 0)
    σy[σy > 1] = 1
    
    return σy
    

def lavelle(x, y, param, magnitude=.05, x_padding=.01, y_padding=.01,
            xc=0, yc=None, D=200):

    σx = lavelle_x(x, param, magnitude, x_padding,
                   xc, D)
    
    σy = lavelle_y(y, param, magnitude, y_padding,
                   yc, D)

    return σx + σy


def plot_uniform_friction(param, func, magnitude=.05, x_padding=.005, y_padding=.005,
                  plot_name='', damping_width=200):
    r = magnitude
    x00, xN0, y00, yN0 = np.array(param.bboxes[0])
    Lx, Ly = xN0 - x00, yN0 - y00
    x0, xN, y0, yN = np.array(param.bboxes[1])# * 1e3 / param.L_R
    x, y = np.linspace(x0, xN, 101) * 1e3 / param.L_R, np.linspace(y0, yN, 101) * 1e3 / param.L_R
    X, Y = np.meshgrid(x, y)
    xc, yc = 0, (param.L_C/param.L_R)

    r = lavelle(X, Y,
                param,
                magnitude=r,
                x_padding=x_padding * 1e3/param.L_R,
                y_padding=y_padding * 1e3/param.L_R,
                xc=xc, yc=yc, D=damping_width)

    
    from ppp.Plots import plot_setup
    from matplotlib import patches
    r[r>1] = 1
    fig, ax = plot_setup('Along-shore (km)', 'Cross-shore (km)')
    L_R = param.L_R * 1e-3
    c = ax.matshow(
        .5 * r,
        cmap="OrRd",
        vmax=1,
        vmin=0,
        extent= np.array(param.bboxes[1]),
        aspect="auto",
        origin="lower",
    )

    for (x_pad, y_pad, col) in zip([Lx/2, x_padding, x_padding+damping_width],
                                 [Ly/2, y_padding, y_padding+damping_width],
                                 ['black', 'red', 'blue']):
        rect = patches.Rectangle((xc*param.L_R*1e-3-x_pad,
                                  yc*param.L_R*1e-3-y_pad),
                                 2*x_pad,
                                 2*y_pad,
                                 linewidth=3,
                                 edgecolor=col, facecolor='none')
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
            
def plot_pretty_good_friction(param, func, magnitude=.05, x_padding=.005, y_padding=.005,
                  plot_name='', damping_width=200):
    r = magnitude
    x00, xN0, y00, yN0 = np.array(param.bboxes[0])
    Lx, Ly = xN0 - x00, yN0 - y00
    x0, xN, y0, yN = np.array(param.bboxes[1])# * 1e3 / param.L_R
    x, y = np.linspace(x0, xN, 101) * 1e3 / param.L_R, np.linspace(y0, yN, 101) * 1e3 / param.L_R
    X, Y = np.meshgrid(x, y)
    xc, yc = 0, (param.L_C/param.L_R)

    R = lavelle(X, Y,
                param,
                magnitude=r,
                x_padding=x_padding * 1e3/param.L_R,
                y_padding=y_padding * 1e3/param.L_R,
                xc=xc, yc=yc, D=damping_width)
    
    Rx = lavelle_x(X,
                   param,
                   magnitude=r,
                   x_padding=x_padding * 1e3/param.L_R,
                   xc=xc, D=damping_width)
    
    Ry = lavelle_y(Y,
                   param,
                   magnitude=r,
                   y_padding=y_padding * 1e3/param.L_R,
                   yc=yc, D=damping_width)
    
    from ppp.Plots import plot_setup, add_colorbar
    from matplotlib import patches

    for r, plot_name in zip([R, Rx, Ry], ['sponge', 'sponge_x', 'sponge_y']):
        print(np.array(param.bboxes[1]))
        fig, ax = plot_setup('Along-shore (km)', 'Cross-shore (km)', scale=.8)
        L_R = param.L_R * 1e-3
        c = ax.imshow(
            .5 * r,
            cmap="OrRd",
            vmax=1,
            vmin=0,
            extent= np.array(param.bboxes[1]),
            aspect="auto",
            origin="lower",
        )
    
        for (x_pad, y_pad, col) in zip([Lx/2, x_padding, x_padding+damping_width],
                                       [Ly/2, y_padding, y_padding+damping_width],
                                       ['black', 'red', 'blue']):
            rect = patches.Rectangle((xc*L_R-x_pad,
                                      yc*L_R-y_pad),
                                     2*x_pad,
                                     2*y_pad,
                                     linewidth=3,
                                     edgecolor=col, facecolor='none')
            ax.add_patch(rect)
        ax.set_aspect('equal')
        cbar = add_colorbar(c, ax=ax)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel('Absorption Coefficient ($\\times10^{-4}\\,\\mathrm{s^{-1}}$)', rotation=270,
                            fontsize=16, labelpad=20)

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
    L = 550
    for padding in [350]:
        bbox_baroclinic = (bbox_barotropic[0] - L,
                           bbox_barotropic[1] + L,
                           bbox_barotropic[2] - L,
                           bbox_barotropic[3] + L)
        print(bbox_baroclinic)
        param.bboxes = [bbox_barotropic, bbox_baroclinic]

    
        plot_uniform_friction(param, lavelle,
                      magnitude=1,
                      x_padding=padding,
                      y_padding=padding,
                      damping_width=150,
                      plot_name='')
        
        plot_pretty_good_friction(param, lavelle,
                      magnitude=1,
                      x_padding=padding,
                      y_padding=padding,
                      damping_width=150,
                      plot_name='')     
    
    