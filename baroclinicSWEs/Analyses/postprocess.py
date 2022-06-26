#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Sat Jan 29 13:39:42 2022

"""
import numpy as np

class animate_solutions(object):
    def __init__(self, vals, grid, wave_frequency=1.4,
                 start_time=0, periods=1, N_period=50, frame_rate=10,
                 repeat=3, bbox=None, file_name='test',
                 folder_dir='Baroclinic Animation', mode=1, titles=None):
        self.grid = grid
        self.x, self.y = grid
        self.values = vals
        self.u, self.v, self.p = np.split(vals, 3)
        self.mode = mode

        labels = ['u', 'v', 'p']
        units = ['m/s', 'm/s', 'Pa']
        self.titles = [f'${labels[i]}_{self.mode}$ ($\\rm{{{units[i]}}}$)' \
                       for i in range(3)] if titles is None else titles
        
        
        self.t0 = start_time
        self.repeat = repeat
        self.wave_frequency = wave_frequency
        self.period = 2*np.pi/self.wave_frequency
        self.tend = self.t0 + periods * self.period
        self.Nt = N_period
        self.time = np.linspace(self.t0, self.tend, self.Nt+1)
        self.fps = frame_rate
        self.file_name, self.folder_dir = file_name, folder_dir
        

        self.x0 = self.x[0, 0]
        self.xN = self.x[-1, -1]
        self.y0 = self.y[0, 0]
        self.yN = self.y[-1, -1]
        self.bbox = (self.x0, self.xN, self.y0, self.yN)
        self.fig_init()
        
        self.start_anim()
        
    def fig_init(self):
        from ppp.Plots import set_axis, subplots

        self.fig, self.axis = subplots(3, 1)
        self.magnitudes = []
        self.plots = []
        

        for i in range(3):
            x_lab = 'Along-shore (km)' if i == 2 else ''
            
            ax = self.axis[i]
            set_axis(ax,
                     x_label=x_lab,
                     y_label='Cross-shore (km)',
                     title=self.titles[i], scale=.75)
            vals = self.values[i]
            self.magnitudes.append(np.nanmax(np.abs(vals)))
            
            c = ax.matshow(vals.real,
                    aspect="auto",
                    cmap="seismic",
                    extent=(self.x0, self.xN, self.y0, self.yN),
                    origin="lower",
                    vmin=-self.magnitudes[-1],
                    vmax=self.magnitudes[-1],
                )
            self.plots.append(c)
            
            self.fig.colorbar(c, ax=ax)
            ax.set_xlim([self.bbox[0], self.bbox[1]])
            ax.set_ylim([self.bbox[2], self.bbox[3]])
            
        ax.set_aspect('equal')
        self.fig.tight_layout()
            
    def animate(self, k):
        sgn = self.wave_frequency/np.abs(self.wave_frequency)
        phase = np.exp(-sgn*2j*np.pi*k/self.Nt)
        for j, plot in enumerate(self.plots):
            vals = self.values[j] * phase
            plot.set_data(vals.real)
            
    def start_anim(self):
        # import os
        import matplotlib.animation as animation
        from ppp.File_Management import dir_assurer

        dir_assurer(self.folder_dir)
        self.anim = animation.FuncAnimation(self.fig,
                        self.animate, frames=self.repeat*self.Nt)
            
        writervideo = animation.FFMpegWriter(fps=self.fps)
        self.anim.save(f'{self.folder_dir}/{self.file_name}.mp4',
                       writer=writervideo)

class animate_solutions2(object):
    def __init__(self, vals, grid, wave_frequency=1.4,
                 start_time=0, periods=1, N_period=50, frame_rate=10,
                 repeat=3, bbox=None, file_name='test',
                 folder_dir='Baroclinic Animation', mode=1, titles=None):
        self.grid = grid
        self.x, self.y = grid
        self.X, self.Y = self.x[::40, ::40], self.y[::40, ::40]
        self.values = vals
        self.u, self.v, self.p = np.split(vals, 3)
        self.p = self.p[0, :, :]
        self.U, self.V = self.u[0, ::40, ::40], self.v[0, ::40, ::40]
        # print(self.U.shape, self.V.shape, self.p.shape, self.X.shape, self.Y.shape)
        self.mode = mode
        
        self.t0 = start_time
        self.repeat = repeat
        self.wave_frequency = wave_frequency
        self.period = 2*np.pi/self.wave_frequency
        self.tend = self.t0 + periods * self.period
        self.Nt = N_period
        self.time = np.linspace(self.t0, self.tend, self.Nt+1)
        self.fps = frame_rate
        self.file_name, self.folder_dir = file_name, folder_dir
        

        self.x0 = self.x[0, 0]
        self.xN = self.x[-1, -1]
        self.y0 = self.y[0, 0]
        self.yN = self.y[-1, -1]
        self.bbox = (self.x0, self.xN, self.y0, self.yN)
        self.fig_init()
        
        self.start_anim()
        
    def fig_init(self):
        from ppp.Plots import plot_setup

        self.fig, self.ax = plot_setup('Along-shore (km)',
                                       'Cross-shore (km)')
        vals = self.p
        magnitude = np.nanmax(np.abs(self.p))
        self.plots = []
            
        c = self.ax.matshow(vals.real,
                            aspect="auto",
                            cmap="seismic",
                            extent=(self.x0, self.xN, self.y0, self.yN),
                            origin="lower",
                            vmin=-magnitude,
                            vmax=magnitude,
                            )
        self.plots.append(c)
        cbar = self.fig.colorbar(c, ax=self.ax)
        cbar.ax.set_ylabel('Pressure (Pa)', rotation=270, fontsize=16)

        if self.mode == 0:
            Q = self.ax.quiver(self.X, self.Y, self.U, self.V,
                               scale=3)
            self.ax.quiverkey(Q, 0.8, .03, .1,
                              r'Horizontal velocity key: $10\,\rm{cm/s}$',
                              labelpos='W', coordinates='figure',
                              fontproperties={'weight': 'bold'})
        else:
            Q = None
        
        self.plots.append(Q)
        self.ax.set_xlim([self.bbox[0], self.bbox[1]])
        self.ax.set_ylim([self.bbox[2], self.bbox[3]])
        self.ax.set_aspect('equal')
        self.fig.tight_layout()
            
    def animate(self, k):
        sgn = self.wave_frequency/np.abs(self.wave_frequency)
        phase = np.exp(-sgn*2j*np.pi*k/self.Nt)
        p_plot, velocity_plot = self.plots
        p_plot.set_data((self.p * phase).real)

        if self.mode == 0:
            velocity_plot.set_UVC((self.U * phase).real, (self.V * phase).real)
            
    def start_anim(self):
        # import os
        import matplotlib.animation as animation
        from ppp.File_Management import dir_assurer

        dir_assurer(self.folder_dir)
        self.anim = animation.FuncAnimation(self.fig,
                        self.animate, frames=self.repeat*self.Nt)
            
        writervideo = animation.FFMpegWriter(fps=self.fps)
        self.anim.save(f'{self.folder_dir}/{self.file_name}.mp4',
                       writer=writervideo)

def post_process(param, barotropic_slns, baroclinic_slns,
                 NN=1000,
                 show_barotropic_sln=False,
                 show_baroclinic_sln=False,
                 show_fluxes=False,
                 show_dissipation=False,
                 nearfield=True,
                 frames=5,
                 save=False
                 ):
    from ppp.Plots import plot_setup, add_colorbar, save_plot
    from ppp.File_Management import dir_assurer
    import matplotlib.pyplot as pt
    folder_dir = 'Figures'
    name_ = f"CanyonWidth={param.canyon_width:.0f}km_alpha={param.alpha:.1f}_beta={param.beta:.1f}"

    bbox_barotropic, bbox_baroclinic = param.bboxes
    dir_assurer(folder_dir)
    
    if nearfield:
        x1 = np.linspace(bbox_barotropic[0], bbox_barotropic[1], NN+1) * 1e3/param.L_R
        y1 = np.linspace(bbox_barotropic[2], bbox_barotropic[3], NN+1) * 1e3/param.L_R
        dx1, dy1 = param.L_R * (x1[-1] - x1[0])/NN, param.L_R * (y1[-1] - y1[0])/NN
        Xg1, Yg1 = np.meshgrid(x1, y1)
        bbox1 = bbox_barotropic
        
    else:
        x1 = np.linspace(bbox_baroclinic[0], bbox_baroclinic[1], NN+1) * 1e3/param.L_R
        y1 = np.linspace(bbox_baroclinic[2], bbox_baroclinic[3], NN+1) * 1e3/param.L_R
        dx1, dy1 = param.L_R * (x1[-1] - x1[0])/NN, param.L_R * (y1[-1] - y1[0])/NN
        Xg1, Yg1 = np.meshgrid(x1, y1)
        bbox1 = bbox_baroclinic
    
    
    ### Barotropic domain ###
    bathymetry = param.canyon_topography(Xg1, Yg1) * param.H_D
    Z1 = baroclinic_slns.Z1(bathymetry)/np.sqrt(param.g)
    # print(np.max(np.abs((Z1 - (param.H_pyc/bathymetry) * np.sqrt(param.reduced_gravity/param.g)))))
    
    # Baroclinic quantities in BL
    p1 = param.ρ_ref * (param.c**2) * Z1 * baroclinic_slns.p(Xg1, Yg1, 0) # * np.exp(-1j * k * Xg)
    u1 = param.c * Z1 * baroclinic_slns.u(Xg1, Yg1, 0)
    v1 = param.c * Z1 * baroclinic_slns.v(Xg1, Yg1, 0)
    

    # Barotropic quantities in BL
    inds = Yg1 >= 0
    p0, u0, v0 = np.zeros(Xg1.shape), np.zeros(Xg1.shape), np.zeros(Xg1.shape)
    p0[inds] = param.ρ_ref * (param.c**2) * barotropic_slns.p(Xg1[inds], Yg1[inds], 0)
    u0[inds] = param.c * barotropic_slns.u(Xg1[inds], Yg1[inds], 0)
    v0[inds] = param.c * barotropic_slns.v(Xg1[inds], Yg1[inds], 0)
    
    if show_barotropic_sln:
        for phase in -np.arange(frames) * 2*np.pi/frames:
            fig, ax = plot_setup('Along-shore (km)',
                         'Cross-shore (km)', scale=.7)

            P = 1e-3 * (p0 * np.exp(1j*phase)).real
            c = ax.contourf(P,
                            cmap='seismic',
                            extent=bbox1,
                            alpha=0.5,
                            vmin=-np.max(np.abs(p0)) * 1e-3, vmax=np.max(np.abs(p1)) * 1e-3,
                            levels=np.linspace(-np.max(np.abs(p0)*1e-3),
                                               +np.max(np.abs(p0)*1e-3),
                                               21)
                            )
            cbar = add_colorbar(c)
            cbar.ax.tick_params(labelsize=16)
            cbar.ax.set_ylabel('Pressure ($\\rm{kPa}$)', rotation=270,
                                fontsize=16, labelpad=20)
        
            U, V = (u0 * np.exp(1j*phase)).real, (v0 * np.exp(1j*phase)).real
            X, Y = Xg1 * param.L_R *1e-3, Yg1 * param.L_R *1e-3
            Q = ax.quiver(
                   X[::40, ::40],
                   Y[::40, ::40],
                   U[::40, ::40], V[::40, ::40],
                   width=0.002,
                   scale=1,
                   )
        
            ax.quiverkey(
                Q,
                .8, .05, #x and y position of key
                .05, #unit
                r"$5\,\rm{cm/s}$",
                labelpos="W",
                coordinates="figure",
                fontproperties={"weight": "bold", "size": 18},
            )
            ax.set_aspect('equal')
            fig.tight_layout()
            
            if not save:
                pt.show()
                
            else:
                dir_assurer(f"{folder_dir}/Barotropic/Nearfield={nearfield}")
                save_plot(fig, ax, name_,
                          folder_name=f"{folder_dir}/Barotropic/Nearfield={nearfield}"
                          )

    if show_baroclinic_sln:
        h1, h2 = param.H_pyc, bathymetry - param.H_pyc
        layers, coeffs = ['Lower', 'Upper'], [1, -h2/h1]
        for phase in -np.arange(frames) * 2*np.pi/frames:
            for k in range(2):
                fig, ax = plot_setup('Along-shore (km)',
                                     'Cross-shore (km)',
                                     scale=.7)
    
                P = (p1 * np.exp(1j*phase)).real
                c = ax.contourf(P * coeffs[k],
                                cmap='seismic',
                                extent=bbox1,
                                alpha=0.5,
                                vmin=-np.max(np.abs(p1 * coeffs[k])), vmax=np.max(np.abs(p1 * coeffs[k])),
                                levels=np.linspace(-np.max(np.abs(p1 * coeffs[k])),
                                                   np.max(np.abs(p1 * coeffs[k])),
                                                   21)
                                )
                cbar = add_colorbar(c)
                cbar.ax.tick_params(labelsize=16)
                cbar.ax.set_ylabel('Pressure ($\\rm{Pa}$)', rotation=270,
                                    fontsize=16, labelpad=20)
            
                U, V = (u1 * np.exp(1j*phase)).real * coeffs[k], (v1 * np.exp(1j*phase)).real * coeffs[k]
                X, Y = Xg1 * param.L_R *1e-3, Yg1 * param.L_R *1e-3
                Q = ax.quiver(
                       X[::40, ::40],
                       Y[::40, ::40],
                       U[::40, ::40], V[::40, ::40],
                       width=0.002,
                       scale=1,
                       )
            
                ax.quiverkey(
                    Q,
                    .8, .05, #x and y position of key
                    .05, #unit
                    r"$5\,\rm{cm/s}$",
                    labelpos="W",
                    coordinates="figure",
                    fontproperties={"weight": "bold", "size": 18},
                )
                ax.set_aspect('equal')
                fig.tight_layout()
                
                if not save:
                    pt.show()
                    
                else:
                    dir_assurer(f"{folder_dir}/Baroclinic/Nearfield={nearfield}")
                    save_plot(fig, ax, name_+f"_{layers[k]}Layer",
                              folder_name=f"{folder_dir}/Baroclinic/Nearfield={nearfield}"
                              )
            
    from barotropicSWEs.Configuration import topography
    h = bathymetry
    h1 = param.H_pyc ; h2 = h - h1
    hx, hy = topography.grad_function(h, dy1, dx1)
    w0 = - (hx * u0 + hy * v0) # barotropic vertical velocity in BL
    D = .5 * (p1 * w0.conjugate()).real #barotropic dissipation
    Jx = .5 * (h * h2/h1) * (p1 * u1.conjugate()).real   # along-shore baroclinic energy flux
    Jy = .5 * (h * h2/h1) * (p1 * v1.conjugate()).real   # cross-shore baroclinic energy flux

    if show_fluxes:
        J = np.sqrt(Jx ** 2 + Jy**2)
        fig, ax = plot_setup('Along-shore (km)',
                              'Cross-shore (km)',
                              scale=.7)
        c = ax.imshow(J,
                      cmap='YlOrRd', aspect='equal',
                      extent=bbox1,
                      origin='lower',
                      vmin=0, vmax=800
                      )
        cbar = add_colorbar(c)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel("Energy Flux ($\\rm{W/m}$)", rotation=270,
                            fontsize=16, labelpad=20)
        X, Y = Xg1 * param.L_R *1e-3, Yg1 * param.L_R *1e-3
        Q = ax.quiver(
               X[::40, ::40],
               Y[::40, ::40],
               1e-4*Jx[::40, ::40], 1e-4*Jy[::40, ::40],
               width=0.002,
               scale=1,
               )
        ax.quiverkey(
            Q,
            .8, .05,
            .04,
            r"$400\,\rm{W/m}$",
            labelpos="W",
            coordinates="figure",
            fontproperties={"weight": "bold", "size": 18},
        )
        ax.set_aspect('equal')
        fig.tight_layout()
        
        if not save:
            pt.show()
            
        else:
            dir_assurer(f"{folder_dir}/Flux/Nearfield={nearfield}")
            save_plot(fig, ax, name_,
                      folder_name=f"{folder_dir}/Flux/Nearfield={nearfield}"
                      )


    if show_dissipation:
        fig, ax = plot_setup('Along-shore (km)',
                              'Cross-shore (km)',
                              scale=.7)
        c = ax.imshow(D*1e3,
                      cmap='seismic', aspect='equal',
                      extent=bbox1,
                      origin='lower',
                      vmin=-80, vmax=80
                      )
        cbar = add_colorbar(c)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel('Tidal Dissipation ($\\rm{mW/m^2}$)', rotation=270,
                            fontsize=16, labelpad=20)
        if not save:
            pt.show()
            
        else:
            dir_assurer(f"{folder_dir}/Dissipation/Nearfield={nearfield}")
            save_plot(fig, ax, name_,
                      folder_name=f"{folder_dir}/Dissipation/Nearfield={nearfield}"
                      )

    D_total = dx1 * dy1 * np.sum(
        ((D[1:] + D[:-1])[:, 1:] + (D[1:] + D[:-1])[:, :-1])/4)
    
    Jx_L = -.5 * dy1 * np.sum(Jx[1:, 0] + Jx[:-1, 0])
    Jx_R = .5 * dy1 * np.sum(Jx[1:, -1] + Jx[:-1, -1])
    Jy_D = .5 * dx1 * np.sum(Jy[-1, 1:] + Jy[-1, :-1])
    Jy_C = -.5 * dx1 * np.sum(Jy[0, 1:] + Jy[0, :-1])

    print(f"\tOffshore: {Jy_D*1e-6:.1f} MW\n\tOnshore: {Jy_C*1e-6:.1f} MW\
\n\tRightward: {Jx_R*1e-6:.1f} MW\n\tLeftward: {Jx_L*1e-6:.1f} MW")

    Jx_total = Jx_R + Jx_L
    Jy_total = Jy_D + Jy_C
    print(f"Jx = {Jx_total*1e-6:.1f} MW, Jy = {Jy_total*1e-6:.1f} MW", flush=True)
    print(f"Jx = {Jx_total/200e3:.1f} W/m, Jy = {Jy_total/200e3:.1f} W/m", flush=True)
    J_total = Jx_total + Jy_total
    print(f"Total Energy Flux in Box: {J_total*1e-6:.2f} MW", flush=True)
    print(f"Total Energy Dissipation in Box: {D_total*1e-6:.2f} MW\n\n", flush=True)

    return Jx_R, Jx_L, Jy_D, Jy_C, D_total

if __name__ == "__main__":
    pass