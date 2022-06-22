#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Sat Jan 29 13:39:42 2022

"""
import numpy as np
    
def post_process1(baroclinic_swes,
                    extent=[-.05, .05, 0, .125],
                    Nx=1000, Ny=1000,
                    folder_dir=''):
    from ppp.Plots import plot_setup, save_plot
    LR = 1e-3 * baroclinic_swes.param.L_R #Rossby radius given in km's
    x0, xN, y0, yN = extent
    folder_dir_ = folder_dir if folder_dir else 'PostProcessing'

    u0, v0, p0 = dimensional_barotropic_solutions(baroclinic_swes,
                                         extent,
                                         Nx, Ny)

    u1, v1, p1 = dimensional_baroclinic_solutions(baroclinic_swes,
                                         extent,
                                         Nx, Ny)

    xg, yg = np.linspace(x0, xN, Nx+1), np.linspace(y0, yN, Ny+1)
    Xg, Yg = np.meshgrid(xg, yg)
    
    # animate_solutions(np.array([u0, v0, p0]), (LR*Xg, LR*Yg),
    #                   file_name='BarotropicAnimation',
    #                   mode=0, folder_dir=folder_dir_)
    
    # animate_solutions2(np.array([u0, v0, p0]), (LR*Xg, LR*Yg),
    #                   file_name='BarotropicAnimation2',
    #                   mode=0, folder_dir=folder_dir_)

    # animate_solutions(np.array([u1, v1, p1]), (LR*Xg, LR*Yg),
    #                   file_name='BaroclinicAnimation',
    #                   mode=1, folder_dir=folder_dir_)
    
    animate_solutions2(np.array([u1, v1, p1]), (LR*Xg, LR*Yg),
                      file_name='BaroclinicAnimation2',
                      mode=1, folder_dir=folder_dir_)

    # bathymetry = baroclinic_swes.h_func(Xg, Yg)
    # baroclinic_wavespeed_sqrd = \
    #     baroclinic_swes.wave_speed_sqrd_functions_apprx[1](bathymetry)
    
    # from barotropicSWEs.topography import grad_function
    # Hx, Hy = grad_function(bathymetry,
    #                         baroclinic_swes.param.L_R * (yN-y0)/Ny, 
    #                         baroclinic_swes.param.L_R * (xN-x0)/Nx)
    
    # # import matplotlib.pyplot as pt
    # values = [bathymetry, Hx, Hy]
    # names = ['Bathymetry',
    #          'Along-shore Bathymetric Gradient',
    #          'Cross-shore Bathymetric Gradient']
    # magnitudes = [(0, baroclinic_swes.param.H_D),
    #               (-2.25, 2.25), (-.15, .15)]
    # cmaps = ['Blues', 'seismic', 'seismic']
    # for i in range(3):
    #     fig, ax = plot_setup('Along-shore (km)',
    #                          'Cross-shore (km)')
    #     c = ax.imshow(values[i], cmap=cmaps[i], aspect='equal',
    #             extent=[LR*x0, LR*xN, LR*y0, LR*yN],
    #             origin='lower',
    #             vmin=magnitudes[i][0], vmax=magnitudes[i][1])
    #     fig.colorbar(c, ax=ax)
    #     save_plot(fig, ax, names[i], folder_name=folder_dir_)
    
    # gradh_dot_u = Hx * u0 + Hy * v0
    
    # for i in range(2):
    #     fig, ax = plot_setup('Along-shore (km)',
    #                      'Cross-shore (km)')
    #     mag = np.nanmax(np.abs(gradh_dot_u))
    #     phase = np.exp(1j * i * np.pi/2)
    #     c = ax.imshow((phase * gradh_dot_u).real,
    #                   cmap='seismic', aspect='equal',
    #                   extent=[LR*x0, LR*xN, LR*y0, LR*yN],
    #                   origin='lower', vmin=-.073, vmax=.073)
    #     fig.colorbar(c, ax=ax)
    #     save_plot(fig, ax, file_name=f'VolumeTransportBathymetricGradient{i}',
    #               folder_name=folder_dir_)

    # time_averaged_drag = .5 * np.real(gradh_dot_u * \
    #                                   p1.conjugate())
    # D = time_averaged_drag

    # Z1 = baroclinic_swes.Z1_func(bathymetry)
    # norm = baroclinic_wavespeed_sqrd * baroclinic_swes.upper_layer_density/\
    #     baroclinic_swes.lower_layer_density
    # Jx = .5 * norm * np.real(u1 * p1.conjugate())/(Z1**2)
    # Jy = .5 * norm * np.real(v1 * p1.conjugate())/(Z1**2)
    
    # J = np.sqrt(Jx**2 + Jy**2)
    # Jx1, Jx2 = Jx[:, 0], Jx[:, -1]
    # Jy1, Jy2 = Jy[0, :], Jy[-1, :]

    # grids = [yg, xg]
    # fluxes = [[Jx1, Jx2], [Jy1, Jy2]]
    # x_labels = ['Cross-shore', 'Along-shore']
    
    # for i in range(2):
    #     fig, ax = plot_setup(x_label=f'{x_labels[i]} (km)',
    #                          y_label='Energy Flux (W/m)')
    #     ax.plot(grids[i]*LR, fluxes[i][0])
    #     ax.plot(grids[i]*LR, fluxes[i][1])
    #     save_plot(fig, ax, f'{x_labels[i ^ 1]} Energy Flux',
    #               folder_name=folder_dir_)
        
    # dx, dy = (xN - x0)/Nx, (yN - y0)/Ny
    
    # Jy_av = dy*np.sum(Jx2-Jx1)/(yN - y0)
    # Jx_av = dx*np.sum(Jy2-Jy1)/(xN - x0)
    
    # xg_coarse, yg_coarse = np.linspace(x0, xN, 26), np.linspace(y0, yN, 26)
    # Xg_coarse, Yg_coarse = np.meshgrid(xg_coarse, yg_coarse)

    # from scipy.interpolate import griddata
    # Jx_coarse = griddata((Xg.flatten(), Yg.flatten()), Jx.flatten(),
    #               (Xg_coarse, Yg_coarse), method="cubic",
    #               fill_value=0)
    # del Jx
    # Jy_coarse = griddata((Xg.flatten(), Yg.flatten()), Jy.flatten(),
    #               (Xg_coarse, Yg_coarse), method="cubic",
    #               fill_value=0)
    # del Jy
    
    # import matplotlib.pyplot as pt
    # fig, ax = plot_setup(x_label='Along-shore (km)',
    #                      y_label='Cross-shore (km)')

    # c = ax.imshow(-D * 1e3,
    #               cmap='seismic', aspect='auto',
    #               extent=[LR*x0, LR*xN, LR*y0, LR*yN],
    #               origin='lower',
    #               vmin=-45, vmax=45)
    # cbar = fig.colorbar(c, ax=ax)
    # cbar.ax.set_ylabel('Tidal Dissipation ($\\rm{mW/m^2}$)', rotation=270,
    #                    fontsize=16, labelpad=20)
    # Q = ax.quiver(LR*xg_coarse, LR*yg_coarse,
    #               Jx_coarse, Jy_coarse)
    
    # ax.quiverkey(Q, .84, .04, 30,
    #              r'Energy flux: $30\,\rm{W/m}$',
    #              labelpos='W', coordinates='figure',
    #              fontproperties={'weight': 'bold'})
    # ax.set_aspect('equal')
    # fig.tight_layout()
    # # pt.show()
    # save_plot(fig, ax, 'BarotropicDissipation',
    #           folder_name=folder_dir_)

    # fig, ax = plot_setup(x_label='Along-shore (km)',
    #                      y_label='Cross-shore (km)')

    # c = ax.imshow(J,
    #               cmap='YlOrRd', aspect='auto',
    #               extent=[LR*x0, LR*xN, LR*y0, LR*yN],
    #               origin='lower',
    #               vmin=0, vmax=45)
    # cbar = fig.colorbar(c, ax=ax)
    # cbar.ax.set_ylabel('Energy Flux ($\\rm{W/m}$)', rotation=270,
    #                    fontsize=16, labelpad=20)
    # Q = ax.quiver(LR*xg_coarse, LR*yg_coarse,
    #               Jx_coarse, Jy_coarse)
    # ax.quiverkey(Q, .84, .04, 30,
    #              r'Energy flux: $30\,\rm{W/m}$',
    #              labelpos='W', coordinates='figure',
    #              fontproperties={'weight': 'bold'})
    # ax.set_aspect('equal')
    # fig.tight_layout()
    # save_plot(fig, ax, 'BaroclinicEnergyFlux',
    #           folder_name=folder_dir_)

    return 1, 1

def dimensional_barotropic_solutions(baroclinic_swes,
                                     extent=[-.1, .1, 0, .2],
                                     Nx=1000, Ny=1000):
    u0, v0, p0 = np.split(baroclinic_swes.barotropic_sols, 3, axis=0)
    X0, Y0 = baroclinic_swes.barotropic_fem.x.flatten('F'), baroclinic_swes.barotropic_fem.y.flatten('F')
    # X1, Y1 = baroclinic_swes.X, baroclinic_swes.Y

    x0, xN, y0, yN = extent
    xg, yg = np.linspace(x0, xN, Nx+1), np.linspace(y0, yN, Ny+1)
    Xg, Yg = np.meshgrid(xg, yg)

    bathymetry = baroclinic_swes.h_func(Xg, Yg)

    from scipy.interpolate import griddata
    
    gridded_barotropic_solutions = []
    Z0 = baroclinic_swes.Z0_func(bathymetry)

    for val in [u0, v0, p0]:
        gridded_barotropic_solutions.append(griddata((X0, Y0), val,
                                                     (Xg, Yg), method="cubic",
                                               fill_value=0)/Z0)    
    u, v, p = gridded_barotropic_solutions
    barotropic_dimensional_slns = [
        baroclinic_swes.c_scale * u * Z0,
        baroclinic_swes.c_scale * v * Z0,
        baroclinic_swes.upper_layer_density * baroclinic_swes.p_scale * \
            p * Z0
            ]
        
    return barotropic_dimensional_slns

def dimensional_baroclinic_solutions(baroclinic_swes,
                                     extent=[-.1, .1, 0, .2],
                                     Nx=1000, Ny=1000):
    u1, v1, p1 = baroclinic_swes.irregular_sols
    X1, Y1 = baroclinic_swes.irregular_grid
    u1, v1, p1 = u1[:, 0], v1[:, 0], p1[:, 0]

    x0, xN, y0, yN = extent
    xg, yg = np.linspace(x0, xN, Nx+1), np.linspace(y0, yN, Ny+1)
    Xg, Yg = np.meshgrid(xg, yg)

    bathymetry = baroclinic_swes.h_func(Xg, Yg)

    from scipy.interpolate import griddata
    
    gridded_baroclinic_solutions = []
    Z1_nd = baroclinic_swes.Z1_func(bathymetry)/np.sqrt(baroclinic_swes.param.g)

    for val in [u1, v1, p1]:
        gridded_baroclinic_solutions.append(griddata((X1, Y1), val,
                                                     (Xg, Yg), method="cubic",
                                                     fill_value=0))
        
    u, v, p = gridded_baroclinic_solutions
    barotropic_dimensional_slns = [
        baroclinic_swes.c_scale * u * Z1_nd,
        baroclinic_swes.c_scale * v * Z1_nd,
        baroclinic_swes.upper_layer_density * baroclinic_swes.p_scale * \
            p * Z1_nd
            ]

    return barotropic_dimensional_slns

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

def yield_fluxes(baroclinic_swes,
                 extent=[-.05, .05, 0, .125],
                 Nx=1000, Ny=1000):
    x0, xN, y0, yN = extent

    u0, v0, p0 = dimensional_barotropic_solutions(baroclinic_swes,
                                         extent,
                                         Nx, Ny)

    u1, v1, p1 = dimensional_baroclinic_solutions(baroclinic_swes,
                                         extent,
                                         Nx, Ny)

    xg, yg = np.linspace(x0, xN, Nx+1), np.linspace(y0, yN, Ny+1)
    Xg, Yg = np.meshgrid(xg, yg)

    bathymetry = baroclinic_swes.h_func(Xg, Yg)
    baroclinic_wavespeed_sqrd = \
        baroclinic_swes.wave_speed_sqrd_functions_apprx[1](bathymetry)
    


    Z1 = baroclinic_swes.Z1_func(bathymetry)
    norm = baroclinic_wavespeed_sqrd * baroclinic_swes.upper_layer_density/\
        baroclinic_swes.lower_layer_density
    Jx = .5 * norm * np.real(u1 * p1.conjugate())/(Z1**2)
    Jy = .5 * norm * np.real(v1 * p1.conjugate())/(Z1**2)

    Jx1, Jx2 = (Jx[1:, 0]+Jx[:-1, 0])/2, (Jx[1:, -1]+Jx[:-1, -1])/2
    Jy1, Jy2 = (Jy[0, 1:]+Jy[0, :-1])/2, (Jy[-1, 1:]+Jy[-1, :-1])/2

    dx, dy = (xN - x0)/Nx, (yN - y0)/Ny
    
    Jy_av = dy*np.sum(Jy2-Jy1)/(yN - y0)
    Jx_av = dx*np.sum(Jx2-Jx1)/(xN - x0)

    return Jx_av, Jy_av

def yield_dissipation(baroclinic_swes,
                      extent=[-.05, .05, 0, .125],
                      Nx=1000, Ny=1000):
    x0, xN, y0, yN = extent # min and max along- and cross-shore positions of bbox
    LR = baroclinic_swes.param.L_R # Rossby radius of deformation in metres

    # Dimensional barotropic solutions in bottom layer on regular grid
    u0, v0, p0 = dimensional_barotropic_solutions(baroclinic_swes,
                                         extent,
                                         Nx, Ny)

    # Dimensional baroclinic solutions in bottom layer on regular grid
    u1, v1, p1 = dimensional_baroclinic_solutions(baroclinic_swes,
                                         extent,
                                         Nx, Ny)

    # Regular grid
    xg, yg = np.linspace(x0, xN, Nx+1), np.linspace(y0, yN, Ny+1)
    Xg, Yg = np.meshgrid(xg, yg)
    dx, dy = (xN - x0)/Nx, (yN - y0)/Ny

    bathymetry = baroclinic_swes.h_func(Xg, Yg)

    # Bathymetrix gradients
    from barotropicSWEs.topography import grad_function
    Hx, Hy = grad_function(bathymetry,
                           baroclinic_swes.param.L_R * (yN-y0)/Ny, 
                           baroclinic_swes.param.L_R * (xN-x0)/Nx)
    
    gradh_dot_u0 = Hx * u0 + Hy * v0
    time_averaged_drag = -.5 * np.real(gradh_dot_u0 * \
                                      p1.conjugate())
    D = time_averaged_drag
    D = (D[1:] + D[:-1])/2 #grid average along-shore
    D_av_cell = (D[:,1:] + D[:,:-1])/2 #grid average over along- and cross-shore
    
    total_dissipation = np.sum(D_av_cell) * dx * dy * (LR**2)
    return total_dissipation

def plot_data(param, order=3, h_min=1e-3, h_max=5e-3,
         wave_frequency=1.4, wavenumber=1.4,
         coastal_lengthscale=0.03,
         coastal_shelf_width=0.02,
         canyon_widths=[1e-10, 1e-3, 5e-3, 1e-2],
         canyon_length=3e-2,
         canyon_depth=.5,
         potential_forcing=False,
         rayleigh_friction=0,
         numerical_flux="Lax-Friedrichs",
         data_dir='Baroclinic Radiating Energy Flux'):
    ω, k, λ = wave_frequency, wavenumber, coastal_lengthscale
    ΔL, h_canyon = canyon_length, canyon_depth
    forcing_, r = potential_forcing, rayleigh_friction
    scheme_ = numerical_flux
    shelf_width=coastal_shelf_width
    LR = param.L_R * 1e-3
    from SWEs import startup

    for background_flow_ in ['KELVIN']:
        for i, w_ in enumerate(canyon_widths):
            barotropic_sols, baroclinic_swes = startup(
                param,
                h_min,
                h_max,
                order,
                mesh_name="",
                canyon_width=w_,
                canyon_length=ΔL,
                canyon_depth=h_canyon,
                coastal_shelf_width=shelf_width,
                plot_domain=False,
                boundary_conditions=["Specified", "Solid Wall"],
                scheme=scheme_,
                potential_forcing=forcing_,
                background_flow=background_flow_,
                θ=0.5,
                rotation=False,
                wave_frequency=ω,
                wavenumber=k,
                coastal_lengthscale=λ,
                rayleigh_friction=r
                )

            Jx, Jy = post_process1(baroclinic_swes,
                                  extent=[-.05, .05, 0, .125],
                                  Nx=1000, Ny=1000,
                                  folder_dir=f'PostProcessing/\
{numerical_flux}/canyonwidth={LR*w_:.0f}km_canyonlength={LR*ΔL:.0f}km_\
canyondepth={h_canyon * param.H_D:.0f}m')

            del baroclinic_swes
            
def plot_energy_fluxes(param, order=3, h_min=1e-3, h_max=5e-3,
         wave_frequency=1.4, wavenumber=1.4,
         coastal_lengthscale=0.03,
         coastal_shelf_width=0.02,
         canyon_widths=[1e-10, 1e-3, 5e-3, 1e-2],
         canyon_length=3e-2,
         canyon_depths=[.25, .5, .75],
         potential_forcing=False,
         rayleigh_friction=0,
         numerical_flux="Lax-Friedrichs",
         data_dir='Baroclinic Radiating Energy Flux',
         colour_scheme='bmh'):
    from ppp.Numpy_Data import save_arrays, load_arrays
    from SWEs import startup

    ω, k, λ = wave_frequency, wavenumber, coastal_lengthscale
    ΔL = canyon_length
    forcing_, r = potential_forcing, rayleigh_friction
    scheme_ = numerical_flux
    shelf_width=coastal_shelf_width

    for background_flow_ in ['KELVIN']:
        from ppp.File_Management import file_exist
        data_name = f'PostProcessing/{scheme_}/EnergyFluxes'
        if not file_exist(f'{data_name}.npz'):
            energy_fluxes = np.zeros((len(canyon_depths),
                                      len(canyon_widths),
                                      2))
            dissipation = np.zeros((len(canyon_depths),
                                    len(canyon_widths)))
            
            for j, d_ in enumerate(canyon_depths):
                if not file_exist(f'{data_name}_{d_ * param.H_D :.0f}m.npz'):
                    flux_depth = np.zeros((len(canyon_widths), 2))
                    dissipation_depth = np.zeros(len(canyon_widths))
    
                    for i, w_ in enumerate(canyon_widths):
                        print(w_, d_)
                        barotropic_sols, baroclinic_swes = startup(
                                    param,
                                    h_min,
                                    h_max,
                                    order,
                                    mesh_name="",
                                    canyon_width=w_,
                                    canyon_length=ΔL,
                                    canyon_depth=d_,
                                    coastal_shelf_width=shelf_width,
                                    plot_domain=False,
                                    boundary_conditions=["Specified",
                                                         "Solid Wall"],
                                    scheme=scheme_,
                                    potential_forcing=forcing_,
                                    background_flow=background_flow_,
                                    θ=0.5,
                                    rotation=False,
                                    wave_frequency=ω,
                                    wavenumber=k,
                                    coastal_lengthscale=λ,
                                    rayleigh_friction=r
                                    )
                        
                        Jx, Jy = yield_fluxes(baroclinic_swes,
                                          extent=[-.05, .05, 0, .125],
                                          Nx=1000, Ny=1000)
                        D = yield_dissipation(baroclinic_swes,
                                              extent=[-.05, .05, 0, .125],
                                              Nx=1000, Ny=1000)

                        flux_depth[i, :] = Jx, Jy
                        dissipation_depth[i] = D
                        del baroclinic_swes
                        
                    save_arrays(f'{data_name}_{d_ * param.H_D :.0f}m',
                                (flux_depth, dissipation_depth))
                    
                else:
                    flux_depth, dissipation_depth = load_arrays(
                        f'{data_name}_{d_ * param.H_D :.0f}m'
                        )
                    
                energy_fluxes[j, :, :] = flux_depth
                dissipation[j, :] = dissipation_depth
            
            save_arrays(data_name, (energy_fluxes, dissipation))

        else:
            from ppp.Numpy_Data import load_arrays
            energy_fluxes, dissipation = load_arrays(data_name)
            
            
        from ppp.Plots import plot_setup, save_plot
        import matplotlib.pyplot as pt
        Jx, Jy = np.split(energy_fluxes, 2, axis=2)
        fig, ax = plot_setup('Canyon Width (km)', 'Energy Flux (W/m)',
                             colour_scheme=colour_scheme)
        ax.plot(param.L_R * 1e-3 * canyon_widths, Jx[:, :, 0].T, 'x-',
                label=(canyon_depths*param.H_D).astype(int))
        ax.legend(title='Canyon Depth (m)', loc=2, fontsize=16,
                  title_fontsize=18)
        fig.tight_layout()
        pt.show()
        # save_plot(fig, ax, f'PostProcessing/{scheme_}/Along-shore')
        
        fig, ax = plot_setup('Canyon Width (km)', 'Energy Flux (W/m)',
                             colour_scheme=colour_scheme)
        ax.plot(param.L_R * 1e-3 * canyon_widths, Jy[:, :, 0].T, 'x-',
                label=(canyon_depths*param.H_D).astype(int))
        ax.legend(title='Canyon Depth (m)', loc=3, fontsize=16,
                  title_fontsize=18)
        fig.tight_layout()
        pt.show()
        # save_plot(fig, ax, f'PostProcessing/{scheme_}/Cross-shore')
        
        fig, ax = plot_setup('Canyon Width (km)', 'Tidal Dissipation (MW)',
                             colour_scheme=colour_scheme)
        ax.plot(param.L_R * 1e-3 * canyon_widths, 1e-6 * dissipation.T, 'x-',
                label=(canyon_depths*param.H_D).astype(int))
        ax.legend(title='Canyon Depth (m)', loc=3, fontsize=16,
                  title_fontsize=18)
        fig.tight_layout()
        pt.show()
        # save_plot(fig, ax, f'PostProcessing/{scheme_}/Dissipation')
        
def post_process(param, barotropic_slns, baroclinic_slns,
                 NN=1000,
                 show_barotropic_sln=False,
                 show_baroclinic_sln=False,
                 show_fluxes=False,
                 show_dissipation=False
                 ):
    from ppp.Plots import plot_setup, add_colorbar
    import matplotlib.pyplot as pt

    bbox_barotropic, bbox_baroclinic = param.bboxes
    x0 = np.linspace(bbox_barotropic[0], bbox_barotropic[1], NN+1) * 1e3/param.L_R
    y0 = np.linspace(bbox_barotropic[2], bbox_barotropic[3], NN+1) * 1e3/param.L_R
    dx0, dy0 = param.L_R * (x0[-1] - x0[0])/NN, param.L_R * (y0[-1] - y0[0])/NN
    Xg0, Yg0 = np.meshgrid(x0, y0)
    
    # x1 = np.linspace(bbox_baroclinic[0], bbox_baroclinic[1], NN+1) * 1e3/param.L_R
    # y1 = np.linspace(bbox_baroclinic[2], bbox_baroclinic[3], NN+1) * 1e3/param.L_R
    # dx1, dy1 = param.L_R * (x1[-1] - x1[0])/NN, param.L_R * (y1[-1] - y1[0])/NN
    # Xg1, Yg1 = np.meshgrid(x1, y1)
    
    
    ### Barotropic domain ###
    bathymetry = param.canyon_topography(Xg0, Yg0) * param.H_D
    Z1 = baroclinic_slns.Z1(bathymetry)/np.sqrt(param.g)
    # print(np.max(np.abs((Z1 - (param.H_pyc/bathymetry) * np.sqrt(param.reduced_gravity/param.g)))))
    
    # Baroclinic quantities in BL
    p1 = param.ρ_ref * (param.c**2) * Z1 * baroclinic_slns.p(Xg0, Yg0, 0) # * np.exp(-1j * k * Xg)
    u1 = param.c * Z1 * baroclinic_slns.u(Xg0, Yg0, 0)
    v1 = param.c * Z1 * baroclinic_slns.v(Xg0, Yg0, 0)
    

    # Barotropic quantities in BL
    p0 = param.ρ_ref * (param.c**2) * barotropic_slns.p(Xg0, Yg0, 0)
    u0 = param.c * barotropic_slns.u(Xg0, Yg0, 0)
    v0 = param.c * barotropic_slns.v(Xg0, Yg0, 0)
    
    if show_barotropic_sln:
        for phase in -np.arange(5) * 2*np.pi/5:
            fig, ax = plot_setup('Along-shore (km)',
                         'Cross-shore (km)')

            P = 1e-3 * (p0 * np.exp(1j*phase)).real
            c = ax.contourf(P,
                            cmap='seismic',
                            extent=[bbox_barotropic[0], bbox_barotropic[1],
                                    bbox_barotropic[2], bbox_barotropic[3]],
                            alpha=0.5,
                            vmin=-np.max(np.abs(p0)) * 1e-3, vmax=np.max(np.abs(p1)) * 1e-3,
                            levels=np.linspace(-np.max(np.abs(p0)*1e-3),
                                               +np.max(np.abs(p0)*1e-3),
                                               21)
                            )
            cbar = add_colorbar(c, ax=ax)
            cbar.ax.tick_params(labelsize=16)
        
            U, V = (u0 * np.exp(1j*phase)).real, (v0 * np.exp(1j*phase)).real
            X, Y = Xg0 * param.L_R *1e-3, Yg0 * param.L_R *1e-3
            Q = ax.quiver(
                   X[::40, ::40],
                   Y[::40, ::40],
                   U[::40, ::40], V[::40, ::40],
                   width=0.002,
                   scale=1,
                   )
        
            ax.quiverkey(
                Q,
                .75, .03, #x and y position of key
                .05, #unit
                r"$5\,\rm{cm/s}$",
                labelpos="W",
                coordinates="figure",
                fontproperties={"weight": "bold", "size": 18},
            )
            ax.set_aspect('equal')
            fig.tight_layout()
            pt.show()

    if show_baroclinic_sln:
        for phase in -np.arange(5) * 2*np.pi/5:
            fig, ax = plot_setup('Along-shore (km)',
                         'Cross-shore (km)')

            P = (p1 * np.exp(1j*phase)).real
            c = ax.contourf(P,
                            cmap='seismic',
                            extent=[bbox_barotropic[0], bbox_barotropic[1],
                                    bbox_barotropic[2], bbox_barotropic[3]],
                            alpha=0.5,
                            vmin=-np.max(np.abs(p1)), vmax=np.max(np.abs(p1)),
                            levels=np.linspace(-np.max(np.abs(p1)),
                                               np.max(np.abs(p1)),
                                               21)
                            )
            cbar = add_colorbar(c, ax=ax)
            cbar.ax.tick_params(labelsize=16)
        
            U, V = (u1 * np.exp(1j*phase)).real, (v1 * np.exp(1j*phase)).real
            X, Y = Xg0 * param.L_R *1e-3, Yg0 * param.L_R *1e-3
            Q = ax.quiver(
                   X[::40, ::40],
                   Y[::40, ::40],
                   U[::40, ::40], V[::40, ::40],
                   width=0.002,
                   scale=1,
                   )
        
            ax.quiverkey(
                Q,
                .75, .03, #x and y position of key
                .05, #unit
                r"$5\,\rm{cm/s}$",
                labelpos="W",
                coordinates="figure",
                fontproperties={"weight": "bold", "size": 18},
            )
            ax.set_aspect('equal')
            fig.tight_layout()
            pt.show()
            
    from barotropicSWEs.Configuration import topography
    h = bathymetry
    h1 = param.H_pyc ; h2 = h - h1
    hx, hy = topography.grad_function(h, dy0, dx0)
    w0 = - (hx * u0 + hy * v0) # barotropic vertical velocity in BL
    D = .5 * (p1 * w0.conjugate()).real #barotropic dissipation
    Jx = .5 * (h * h2/h1) * (p1 * u1.conjugate()).real   # along-shore baroclinic energy flux
    Jy = .5 * (h * h2/h1) * (p1 * v1.conjugate()).real   # cross-shore baroclinic energy flux

    if show_fluxes:
        J = np.sqrt(Jx ** 2 + Jy**2)
        fig, ax = plot_setup('Along-shore (km)',
                              'Cross-shore (km)')
        # J_max = np.max(J)
        c = ax.imshow(J,
                      cmap='YlOrRd', aspect='equal',
                      extent=bbox_barotropic,
                      origin='lower',
                      vmin=0, vmax=800
                      )
        cbar = fig.colorbar(c, ax=ax)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel('Energy Flux ($\\rm{W/m}$)', rotation=270,
                            fontsize=16, labelpad=20)
        X, Y = Xg0 * param.L_R *1e-3, Yg0 * param.L_R *1e-3
        Q = ax.quiver(
               X[::40, ::40],
               Y[::40, ::40],
               1e-4*Jx[::40, ::40], 1e-4*Jy[::40, ::40],
               width=0.002,
               scale=1,
               )
        ax.quiverkey(
            Q,
            .88, .03,
            .04,
            r"$400\,\rm{W/m}$",
            labelpos="W",
            coordinates="figure",
            fontproperties={"weight": "bold", "size": 18},
        )
        ax.set_aspect('equal')
        fig.tight_layout()
        pt.show()


    if show_dissipation:
        fig, ax = plot_setup('Along-shore (km)',
                              'Cross-shore (km)')
        # D_max = 1e3 * np.max(D)
        c = ax.imshow(D*1e3,
                      cmap='seismic', aspect='equal',
                      extent=bbox_barotropic,
                      origin='lower',
                      vmin=-80, vmax=80
                      )
        cbar = fig.colorbar(c, ax=ax)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel('Energy Dissipation ($\\rm{mW/m^2}$)', rotation=270,
                            fontsize=16, labelpad=20)
        pt.show()

    D_total = dx0 * dy0 * np.sum(
        ((D[1:] + D[:-1])[:, 1:] + (D[1:] + D[:-1])[:, :-1])/4)
    
    Jx_L = -.5 * dy0 * np.sum(Jx[1:, 0] + Jx[:-1, 0])
    Jx_R = .5 * dy0 * np.sum(Jx[1:, -1] + Jx[:-1, -1])
    
    # pt.plot(y0 * param.L_R * 1e-3, Jx[:, 0])
    # pt.plot(y0 * param.L_R * 1e-3, Jx[:, -1])
    # pt.show()
    Jy_D = .5 * dx0 * np.sum(Jy[-1, 1:] + Jy[-1, :-1])
    Jy_C = -.5 * dx0 * np.sum(Jy[0, 1:] + Jy[0, :-1])
    
    # pt.plot(x0 * param.L_R * 1e-3, Jy[0])
    # pt.plot(x0 * param.L_R * 1e-3, Jy[-1])
    # pt.show()
    
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
    import configure
    param, args = configure.main()
    args.domain = .4
    args.order = 4

    h_min, h_max = .05, .01 #args.hmin, args.hmax
    order, domain_width = args.order, args.domain
    λ = args.coastal_lengthscale
    LC = args.shelf_width
    LS = λ - LC

    k = param.k * param.L_R  # non-dimensional alongshore wavenumber
    ω = param.ω / param.f  # non-dimensional forcing frequency
    canyon_widths_ = np.linspace(1e-3, 1e-2, 19)
    canyon_widths_ = np.insert(canyon_widths_, 0, 0)
    bbox_barotropic = (-domain_width/2, domain_width/2, 0, .175)
    DLy = .4
    bbox_baroclinic = (-domain_width/2, domain_width/2, λ/2 - DLy, λ/2 + DLy)
    param.bboxes = [bbox_barotropic, bbox_baroclinic]
    plot_energy_fluxes(param, args.order,
                        h_min, h_max,
                        coastal_lengthscale=args.coastal_lengthscale,
                        canyon_length=args.canyon_length,
                        canyon_depths=np.array([.125, .25, .5, .75, .875, 1.0]),
                        coastal_shelf_width=args.shelf_width,
                        canyon_widths=canyon_widths_,
                        numerical_flux="Central",
                        wave_frequency=ω, wavenumber=k)

    # canyon_widths_ = np.linspace(1e-3, 1e-2, 19)#[4::4]
    # canyon_widths_ = np.insert(canyon_widths_, 0, 0)
    # plot_data(param, args.order,
    #                     h_min, h_max,
    #                     coastal_lengthscale=args.coastal_lengthscale,
    #                     canyon_length=args.canyon_length,
    #                     canyon_depth=.5,
    #                     coastal_shelf_width=args.shelf_width,
    #                     canyon_widths=canyon_widths_[[0, 4, 9, 19]],
    #                     numerical_flux="Central",
    #                     wave_frequency=ω, wavenumber=k)
