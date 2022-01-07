#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Tue Mar  9 16:00:06 2021
"""
import numpy as np
import plotly.io as pio

pio.renderers.default = "browser"

class solver(object):
    def __init__(
        self,
        param,
        fem,
        barotropic_sols,
        barotropic_fem,
        barotropic_dir,
        h_func,
        upper_layer_thickness=100,
        upper_layer_density=1025,
        lower_layer_density=1050,
        periods=2,
        flux_scheme="central",
        boundary_conditions="Solid Wall",
        data_dir='Baroclinic Response',
        rotation=True,
        θ=1,
        wave_frequency=1.4,
        rayleigh_friction=0,
        sponge_function=np.vectorize(lambda x, y : 0)
    ):
        from scipy.sparse import diags

        
        self.param = param
        self.fem = fem
        self.barotropic_fem = barotropic_fem
        self.barotropic_sols = barotropic_sols
        self.barotropic_dir = barotropic_dir
        self.rotation = rotation
        self.ω = wave_frequency
        self.sponge_function = sponge_function
        self.rayleigh_friction = rayleigh_friction
        self.X, self.Y = self.fem.x.flatten("F"), self.fem.y.flatten("F")
        self.data_dir = data_dir
        
        from ppp.File_Management import dir_assurer
        dir_assurer(self.data_dir)

        # Normal matrices:
        self.Nx = diags(self.fem.nx.T.flatten(), format="csr")
        self.Ny = diags(self.fem.ny.T.flatten(), format="csr")
        self.scheme, self.boundary_conditions = flux_scheme.upper(), boundary_conditions.upper()

        if self.scheme not in [
            "CENTRAL",
            "UPWIND",
            "LAX-FRIEDRICHS",
            "PENALTY",
            "ALTERNATING",
        ]:
            raise ValueError("Invalid Flux scheme")

        if self.boundary_conditions not in ["SOLID WALL", "OPEN FLOW", "MOVING WALL", "SPECIFIED"]:
            raise ValueError("Invalid Boundary Condition")

        else:
            # Schemes as given in Hesthaven & Warburton
            if self.scheme == "CENTRAL":
                self.α, self.β, self.γ, self.θ = 0.0, 0.0, 0.0, 0.5

            elif self.scheme == "UPWIND":
                self.α, self.β, self.γ, self.θ = 0.0, 1.0, 1.0, 0.5

            elif self.scheme == "PENALTY":
                self.α, self.β, self.γ, self.θ = 1.0, 0.0, 1.0, 0.5

            elif self.scheme == "LAX-FRIEDRICHS":
                self.α, self.β, self.γ, self.θ = 1.0, 1.0, 1.0, 0.5

            else:  # Alternating flux as given in (Shu, 2002) -certainly (Ambati & Bokhove, 2007)
                self.α, self.β, self.γ, self.θ = 0.0, 0.0, 0.0, θ

        if h_func is None:
            self.h_func = np.vectorize(lambda X, Y: self.param.H_D)
        else:
            self.h_func = h_func

        self.bathymetry = self.h_func(self.X, self.Y)
        self.upper_layer_thickness = upper_layer_thickness
        self.upper_layer_density = upper_layer_density
        self.lower_layer_density = lower_layer_density
        
        from modal_decomposition import MultiLayerModes
        self.modes = MultiLayerModes(self.bathymetry,
                                layer_thicknesses=self.upper_layer_thickness,
                                layer_densities=self.upper_layer_density,
                                max_density=self.lower_layer_density,
                                g=param.g,
                                normalisation="Anti-Symmetric")
        self.wave_speed_sqrd_functions_apprx, \
            self.modal_interaction_coefficients_apprx, \
                self.approx_normalisations, \
                    self.vertical_structure_functions_apprx = \
                        self.modes.two_layer_approximations()
                        
        self.barotropic_wavespeed_apprx = \
            self.wave_speed_sqrd_functions_apprx[0](
                self.bathymetry
                )
        self.baroclinic_wavespeed_apprx = \
            self.wave_speed_sqrd_functions_apprx[1](
                self.bathymetry
                )
            
        self.C0_sqrd = diags(self.barotropic_wavespeed_apprx/self.param.c**2)
        self.C1_sqrd = diags(self.baroclinic_wavespeed_apprx/self.param.c**2)
        
        self.matrix_setup()
        self.baroclinic_friction()
        self.generate_barotropic_forcing()

    def timestep(
        self,
        u0,
        v0,
        η0,
        t_final=1,
        Nout=None,
        file_name=None,
        method="RK4",
        ω=None,
        φ=0,
        animate=False,
        N_frames=100,
    ):
        from ppp.Jacobi import JacobiGQ
        from numpy.linalg import norm

        # Compute time-step size
        rLGL = JacobiGQ(0, 0, self.fem.N)[0]
        rmin = norm(rLGL[0] - rLGL[1])
        dtscale = self.fem.dtscale
        dt = np.min(dtscale) * rmin * 2 / 3 #(see Hesthaven and Warburton, 2008)
        dt = rmin/np.sqrt(np.max(self.barotropic_wavespeed_apprx/self.param.c**2))

        # from math import ceil

        # Nt = ceil((t_final / N_frames) / dt)
        # dt = .001 #t_final / (Nt * N_frames)

        if Nout is None:
            Nout = self.fem.N
        if file_name is None:
            file_name = f"Solutions: {self.scheme}"
        u, v, η = np.copy(u0), np.copy(v0), np.copy(η0)
        ω = self.ω if ω is None else ω
        N = self.fem.Np * self.fem.K
        from scipy.sparse import csr_matrix as sp
        from scipy.sparse import block_diag, identity

        i = identity(N)
        I2 = block_diag(2 * [self.R_friction] + [sp((N, N))])
        # I3 = block_diag(3 * [i])

        assert ω is not None

        # Potential forcing φ of frequency ω
        T = 4

        F = (
            lambda t: self.barotropic_forcing
            * np.exp(-1j * ω * t)
            * (np.tanh(t / (5 * T))) ** 4
        )
        
        r_ = lambda t: 1 - np.tanh(t / (10 * T)) ** 4

        x, y = np.round(self.X, 15), np.round(self.Y, 15)
        Nx, Ny = 500, 500
        xg, yg = np.linspace(self.param.bboxes[1][0], self.param.bboxes[1][1], Nx), \
            np.linspace(self.param.bboxes[1][2], self.param.bboxes[1][3], Ny)
        X, Y = np.meshgrid(xg, yg)
        
        y_old = np.concatenate([u, v, η], axis=0)[:, None]
        
        from scipy.sparse.linalg import spsolve
        y_vals = np.zeros((1001, 3*u.shape[0]), dtype=complex)
        t = 0
        for i in range(10000000):
            # print(y_old.shape, (F(t)).shape)
            y_new = y_old + dt * ((self.A - r_(t) * I2) @ y_old + F(t))
            t += dt
            y_old = y_new
            if (i+1) % 1000 == 0:
                j = int((i+1)//1000)
                y_vals[j, :] = y_new[:, 0]
                plot(y_new[:, 0], t, (x, y), (X, Y))

        raise ValueError


    def matrix_setup(self):
        from scipy.sparse import bmat as bsp
        from scipy.sparse import csr_matrix as sp
        from scipy.sparse import identity, diags, block_diag

        vmapM, vmapP = self.fem.vmapM, self.fem.vmapP

        mapB, vmapB = self.fem.mapB, self.fem.vmapB
        N = vmapM.shape[0]
    
        inds = np.arange(N)
        self.Im = sp((N, self.fem.Np * self.fem.K)).tolil()
        self.Ip = sp((N, self.fem.Np * self.fem.K)).tolil()
        
        self.Im[inds, vmapM], self.Ip[inds, vmapP] = 1, 1

        Im, Ip = self.Im.tocsr(), self.Ip.tocsr()
        self.avg = 0.5 * (self.Im + self.Ip)

        # Normal matrices:
        Nx, Ny = self.Nx, self.Ny

        N00, N10 = self.fem.nx.shape
        I, ON = identity(N00 * N10), sp((N00 * N10, N00 * N10))

        self.fscale = diags(self.fem.Fscale.T.flatten())
        self.fscale_inv = diags(1 / self.fem.Fscale.T.flatten())
        self.Fscale = block_diag([self.fscale] * 3)
        self.lift = block_diag([self.fem.LIFT] * self.fem.K)
        self.LIFT = block_diag([self.lift] * 3)

        self.Dx = diags(self.fem.rx.T.flatten()) @ block_diag(
            [self.fem.Dr] * N10
        ) + diags(self.fem.sx.T.flatten()) @ block_diag([self.fem.Ds] * N10)

        self.Dy = diags(self.fem.ry.T.flatten()) @ block_diag(
            [self.fem.Dr] * N10
        ) + diags(self.fem.sy.T.flatten()) @ block_diag([self.fem.Ds] * N10)

        ON = sp((Ny.shape[0], Ip.shape[1]))
        i = identity(self.fem.Np * self.fem.K)
        o = sp((self.fem.Np * self.fem.K, self.fem.Np * self.fem.K))

        # System Equations:
        if self.rotation:
            self.A1 = bsp(
                [
                    [o, i, -self.Dx],
                    [-i, o, -self.Dy],
                    [-self.Dx @ self.C1_sqrd, -self.Dy @ self.C1_sqrd, o],
                ]
            )

        else:
            self.A1 = bsp(
                [
                    [o, o, -self.Dx],
                    [o, o, -self.Dy],
                    [-self.Dx @ self.C1_sqrd, -self.Dy @ self.C1_sqrd, o],
                ]
            )

        # Boundary Conditions:
        Ipu1, Ipu2, Ipu3 = Ip.copy().tolil(), ON.copy().tolil(), ON.copy().tolil()
        Ipv1, Ipv2, Ipv3 = ON.copy().tolil(), Ip.copy().tolil(), ON.copy().tolil()
        Ipη1, Ipη2, Ipη3 = ON.copy().tolil(), ON.copy().tolil(), Ip.copy().tolil()

        N = self.fem.K * self.fem.Nfaces
        self.Un = np.zeros((3 * Nx.shape[0], 1))

        if self.boundary_conditions == "SOLID WALL":
#            if self.rotation and self.scheme not in ['ALTERNATING', 'CENTRAL']:
                # η+ = η-
            # Ipη3[mapB, vmapB] = Im[mapB, vmapB]

                # u+ = (ny^2 - nx^2) * -u- - 2 * nx * ny * v- (non-dimensional impermeability)
            Ipu1[mapB, vmapB] = ((Ny @ Ny - Nx @ Nx) @ Im)[mapB, vmapB]
            Ipu2[mapB, vmapB] = -2 * (Nx @ Ny @ Im)[mapB, vmapB]

                # v+ = (nx^2 - ny^2) * -v- - 2 * nx * ny * u- (non-dimensional impermeability)
            Ipv2[mapB, vmapB] = ((Nx @ Nx - Ny @ Ny) @ Im)[mapB, vmapB]
            Ipv1[mapB, vmapB] = -2 * (Nx @ Ny @ Im)[mapB, vmapB]

  #          else:                #Case of Ambati & Bokhove to constrain central flux on domain boundary
  #              raise ValueError
  #              # η+ = η-
  #              Ipη3[mapB, vmapB] = Im[mapB, vmapB]



        elif self.boundary_conditions == "OPEN FLOW":
            # [[η]]=0 ==> η+ = η-
            Ipη3[mapB, vmapB] = Im[mapB, vmapB]

        elif self.boundary_conditions == "FIXED FLUX":
            # {{h u}} = Q(x) ==> u+ = -u- +2Q(x)/h(x)
            Ipu1[mapB, vmapB] = -Im + 2 * self.Q(self.X, self.Y) / self.h(
                self.X, self.Y
            )

        elif self.boundary_conditions == "SPECIFIED":

            for bc in self.fem.BCs.keys():
                m, vm = self.fem.maps[bc], self.fem.vmaps[bc]
                if "Wall" or "Open" in bc:
                    # η+ = η-
                    # Ipη3[m, vm] = Im[m, vm]

                    # u+ = (ny^2 - nx^2) * -u- - 2 * nx * ny * v- (non-dimensional impermeability)
                    Ipu1[m, vm] = ((Ny @ Ny - Nx @ Nx) @ Im)[m, vm]
                    Ipu2[m, vm] = -2 * (Nx @ Ny @ Im)[m, vm]

                    # v+ = (nx^2 - ny^2) * -v- - 2 * nx * ny * u- (non-dimensional impermeability)
                    Ipv2[m, vm] = ((Nx @ Nx - Ny @ Ny) @ Im)[m, vm]
                    Ipv1[m, vm] = -2 * (Nx @ Ny @ Im)[m, vm]

                if "Open" in bc:
                    N = self.fem.vmapM.shape[0]
                    Norms = bsp([[Nx, Ny]])
                    IO = sp((N, self.fem.Np * self.fem.K)).tolil()
                    IO[m, vm] = 1
                    IO = block_diag([IO] * 2)
                    U = Norms @ IO @ block_diag([self.H] * 2) @ self.background_flow
                    Qx = 2 * Nx @ U
                    Qy = 2 * Ny @ U
                    η_t = -1j * self.ω * sp(U.shape) #zero
                    X = bsp([[Qx], [Qy], [η_t]])
                    self.Un += X

        else:  # 'MOVING WALL'
            # η+ = (-nt^2 + nx^2 + ny^2)η- - 2 * nt * nx * h- u- - 2 * nt * ny * h- v-
            # u+ = (-nt^2 + nx^2 + ny^2)η- - 2 * nt * nx * h- u- - 2 * nt * ny * h- v-
            # v+ = (-nt^2 + nx^2 + ny^2)η- - 2 * nt * nx * h- u- - 2 * nt * ny * h- v-
            pass

        Ipu = bsp([[Ipu1, Ipu2, Ipu3]])
        Ipv = bsp([[Ipv1, Ipv2, Ipv3]])
        Ipη = bsp([[Ipη1, Ipη2, Ipη3]])
        Ip2 = bsp([[Ipu], [Ipv], [Ipη]])
        Im2 = block_diag(3 * [Im])

        self.jump = Im2 - Ip2  # Jump operator
        H_mat = block_diag(
            [self.C1_sqrd, self.C1_sqrd, i]
        )

        α, β, γ, θ = self.α, self.β, self.γ, self.θ
        θ = θ * np.ones(Ny.shape[0])
        θ[mapB] = 0.5

        self.Flux = .5 * bsp(
            [
                [
                    -sp((α * (Nx @ Nx) + β * (Ny @ Ny))),
                    -sp(((α - β) * Nx @ Ny)),
                    2 * diags(1 - θ) @ Nx,
                ],
                [
                    -sp(((α - β) * Nx @ Ny)),
                    -sp((β * Nx @ Nx + α * Ny @ Ny)),
                    2 * diags(1 - θ) @ Ny,
                ],
                [2 * diags(θ) @ Nx, 2 * diags(θ) @ Ny, -γ * I],
            ]
        )

        # Flux Contributiuons
        self.F = self.LIFT @ self.Fscale @ self.Flux @ self.jump @ H_mat
        self.A = self.A1 + self.F

        # Singular Mass Matrix
        self.M = block_diag([self.fem.mass_matrix] * N10)
        ones = np.ones((self.fem.Np * self.fem.K, 1))

        self.norm = (ones.T @ self.M @ ones)[0, 0]

        # Inhomogeneous part from BCs
        self.U_D = self.LIFT @ self.Fscale @ self.Flux @ self.Un

    def evp(self, k_vals=10):
        from scipy.linalg import eig

        vals, vecs = eig(1j * self.A.todense())
        return vals, vecs.T

    def generate_barotropic_forcing(self,
                                    animate_barotropic_solutions=True,
                                    animate_barotropic_forcing=True):
        from ppp.File_Management import dir_assurer, file_exist
        dir_assurer('Barotropic Forcing')
        x0, xN, y0, yN = self.param.bboxes[0]
        LR = self.param.L_R * 1e-3
        file_dir = self.barotropic_dir + \
            f'_domainwidth={LR*(xN-x0):.0f}x{LR*(yN-y0):.0f}'
            
        import matplotlib.pyplot as pt
        Nx, Ny = 1000, 1000
        x0, xN, y0, yN = self.param.bboxes[1]
        dx = (xN - x0)/Nx
        dy = (yN - y0)/Ny
        xg, yg = np.linspace(x0, xN, Nx+1), np.linspace(y0, yN, Ny+1)
        Xg, Yg = np.meshgrid(xg, yg)
        pt.matshow(self.sponge_function(Xg, Yg), cmap="seismic",
                   extent=self.param.bboxes[1], aspect="auto",
                   vmin=-1, vmax=1)
        pt.show()

        if not file_exist(f'Barotropic Forcing/{file_dir}.npz'):
            u0, v0, p0 = np.split(self.barotropic_sols, 3, axis=0)
            X0, Y0 = self.barotropic_fem.x.flatten('F'), self.barotropic_fem.y.flatten('F')
            X1, Y1 = self.X, self.Y
            
            Nx, Ny = 1000, 1000
            x0, xN, y0, yN = self.param.bboxes[1]
            dx = (xN - x0)/Nx
            dy = (yN - y0)/Ny
            xg, yg = np.linspace(x0, xN, Nx+1), np.linspace(y0, yN, Ny+1)
            Xg, Yg = np.meshgrid(xg, yg)
    
            bathymetry = self.h_func(Xg, Yg)
            # bathymetry0 = self.h_func(X0, Y0)
            # bathymetry1 = self.h_func(X1, Y1)

            from barotropicSWEs.topography import grad_function
            
            from ppp.Plots import plot_setup
            import matplotlib.pyplot as pt
            from scipy.interpolate import griddata
            
            gridded_barotropic = []
            for val in [u0, v0, p0]:
                val_r = griddata((X0, Y0), val.real,
                                        (Xg, Yg), method="cubic",
                                        fill_value=0)
                val_i = griddata((X0, Y0), val.imag,
                                        (Xg, Yg), method="cubic",
                                        fill_value=0)
                val = val_r + 1j * val_i
                gridded_barotropic.append(val)
    
            u, v, p = gridded_barotropic #Gridded solution non-dimensional system variables, and fluid depth
            
            if animate_barotropic_solutions:
                print(np.array(gridded_barotropic).shape,
                      Xg.shape)
                animate_solutions(np.array(gridded_barotropic), (Xg, Yg),
                                  wave_frequency=1.4,
                                  bbox=self.param.bboxes[0],
                                  padding=(0, 0),
                                  file_name=self.barotropic_dir,
                                  folder_dir="Barotropic Animation",
                                  mode=0
                                  )
                
            hx, hy = grad_function(bathymetry/self.param.H_D, dx, dy) # bathymetric gradients
    
            L_R = self.param.L_R*1e-3
            T10 = self.modal_interaction_coefficients_apprx[1][0](bathymetry) * \
                self.param.H_D
            c0_sqrd = self.wave_speed_sqrd_functions_apprx[0](bathymetry)/ \
                (self.param.c**2)
            
            
            u1_forcing = -T10 * p * hx * (1 - self.sponge_function(Xg, Yg))
            v1_forcing = -T10 * p * hy * (1 - self.sponge_function(Xg, Yg))
            p1_forcing = -T10 * c0_sqrd * (u * hx + v * hy)  * \
                (1 - self.sponge_function(Xg, Yg))
            
            forcings = [u1_forcing, v1_forcing, p1_forcing]
            
            if animate_barotropic_forcing:
                animate_solutions(np.array(forcings), (Xg, Yg),
                                  wave_frequency=1.4,
                                  bbox=self.param.bboxes[1],
                                  padding=(0, 0),
                                  file_name=f"{self.barotropic_dir}_forcing",
                                  folder_dir="Barotropic Animation",
                                  mode=1
                                  )
            
            
            
            forcing_baroclinic_grid = []
            for forcing, title in zip(forcings,
                                      ['u_1', 'v_1', 'p_1']):
                forcing_r = griddata((Xg.flatten(), Yg.flatten()),
                                      forcing.real.flatten(),
                                      (X1, Y1), method="cubic",
                                      fill_value=0)
                forcing_i = griddata((Xg.flatten(), Yg.flatten()),
                                      forcing.imag.flatten(),
                                      (X1, Y1), method="cubic",
                                      fill_value=0)
                forcing_baroclinic_grid.append(forcing_r + 1j * forcing_i)
    
            from ppp.Numpy_Data import save_arrays
            save_arrays(file_dir, tuple(forcing_baroclinic_grid),
                        wd="Barotropic Forcing")
            
        else:
            from ppp.Numpy_Data import load_arrays
            forcing_baroclinic_grid = load_arrays(file_dir,
                                                  wd="Barotropic Forcing")

        self.barotropic_forcing = np.concatenate(
            forcing_baroclinic_grid)[:, None]

    def baroclinic_friction(self):
        from scipy.sparse import diags

        R = self.sponge_function(self.X, self.Y) * self.rayleigh_friction
        self.R_friction = diags(R)

    def bvp(self, wave_frequency=None, verbose=False):
        from scipy.sparse import identity, block_diag  # , diags
        from scipy.sparse.linalg import spsolve
        from scipy.sparse import csr_matrix as sp
        from ppp.File_Management import file_exist
        
        ω = self.ω if wave_frequency is None else wave_frequency
        LR = 1e-3 * self.param.L_R
        xN, x0, yN, y0 = self.param.bboxes[1]

        name = self.barotropic_dir + \
            f'_domainwidth={LR*(xN-x0):.0f}x{LR*(yN-y0):.0f}_ω={ω:.2f}'
        
        if not file_exist(f'{self.data_dir}/{name}.npz'):
            from ppp.Numpy_Data import save_arrays

            
            # r = self.r if rayleigh_friction is None else rayleigh_friction
            # self.frames = frames
            # self.time = np.linspace(0, 2 * np.pi / ω, self.frames + 1)
    
            N = self.fem.Np * self.fem.K
            i, o = identity(N), sp((N, N))
    
            I = block_diag([i] * 3)
            assert ω is not None
            
            sponge = block_diag(3 * [self.R_friction]) #Alternative: block_diag(2 * [self.R_friction] + o)
    
            A = -self.A - 1j * ω * I + sponge
            if verbose:
                print("Solving BVP using spsolve")
    
            # import time
            # for method in ["MMD_ATA", "MMD_AT_PLUS_A", "COLAMD", "NATURAL"]:
            #     print(method)
            #     start = time.perf_counter()
            #     sols = spsolve(sp(A), self.barotropic_forcing, permc_spec=method)[:, None]
            #     print(f'{method}: {time.perf_counter()-start:.2f} seconds')
                
            sols = spsolve(sp(A), self.barotropic_forcing)[:, None]
            
            x, y = np.round(self.X, 15), np.round(self.Y, 15)
            Nx, Ny = 500, 500
            xg, yg = np.linspace(self.param.bboxes[1][0], self.param.bboxes[1][1], Nx), \
                np.linspace(self.param.bboxes[1][2], self.param.bboxes[1][3], Ny)
            X, Y = np.meshgrid(xg, yg)
            save_arrays(name, (sols, x, y, X, Y),
                        folder_name=self.data_dir)
            
            
        else:
            from ppp.Numpy_Data import load_arrays
            sols, x, y, X, Y = load_arrays(name, folder_name=self.data_dir)
        
        x_padding, y_padding = .125, .125
        
        bathymetry = self.h_func(X, Y)
        
        u, v, p = np.split(sols, 3)
        baroclinic_wavespeed_sqrd = \
            self.wave_speed_sqrd_functions_apprx[1](bathymetry)
        norm = baroclinic_wavespeed_sqrd/self.param.c**2
        
        u, v, p = grid_convert([u, v, p], (x, y), (X, Y))
        Jx = .5 * norm * np.real(u * p.conjugate())
        Jy = .5 * norm * np.real(v * p.conjugate())
        
        from ppp.Plots import plot_setup
        import matplotlib.pyplot as pt
        from matplotlib import patches
        
        x0, xN, y0, yN = (X[0,0], X[-1,-1], Y[0,0], Y[-1,-1])

        for J, title_ in zip([Jx, Jy], ["$J_x$", "$J_y$"]):
            val_max = np.max(np.abs(J))
            fig, ax = plot_setup("x", "y", title=f"Non-dimensional {title_}")
            c = ax.matshow(J,
                    aspect="auto",
                    cmap="seismic",
                    extent=(x0, xN, y0, yN),
                    vmax=val_max, vmin=-val_max,
                    origin="lower"
                )

            rect = patches.Rectangle((x0+x_padding, y0+y_padding),
                                      (xN-x0)-2*x_padding,
                                      (yN-y0)-2*y_padding,
                                      linewidth=3,
                                      edgecolor='black', facecolor='none')
            ax.add_patch(rect)
            fig.colorbar(c, ax=ax)
            
            pt.show()
            
        # return

        # convert to from Rossby radii to km's
        x *= LR
        y *= LR
        X *= LR
        Y *= LR
        x_padding *= LR
        y_padding *= LR
        
        for bbox_, name_ in zip([None,
                                  (X[0,0]+x_padding,
                                  X[-1,-1]-x_padding,
                                  Y[0,0]+y_padding,
                                  Y[-1,-1]-y_padding),
                                  (X[0,0]+x_padding,
                                  X[-1,-1]-x_padding,
                                  0, 100)], ['', 'zoom', 'zoom2']):                                           
            animate_solutions(sols, (x, y), new_grid=(X, Y),
                              wave_frequency=1.4,
                              bbox=bbox_,
                              padding=(x_padding, y_padding),
                              file_name=f'{name}_{name_}'
                              )
        return sols

def grid_convert(vals, old_grid, new_grid):
    from scipy.interpolate import griddata
    
    new_vals = []
    for val in vals:
        val_r = griddata(old_grid, val.real,
                         new_grid, method="cubic")
        val_i = griddata(old_grid, val.imag,
                         new_grid, method="cubic")
        new_vals.append((val_r + 1j * val_i)[:, :, 0])
        
    return new_vals

def plot(vals, t, old_grid, new_grid, padding=(2.5e-2, 5e-2)):
    from scipy.interpolate import griddata
    from ppp.Plots import plot_setup
    import matplotlib.pyplot as pt
    from matplotlib import patches
    x, y = old_grid
    X, Y = new_grid
    x0, xN = X[0, 0], X[-1, -1]
    y0, yN = Y[0, 0], Y[-1, -1]
    u, v, p = np.split(vals, 3)
    time = np.linspace(0, 2*np.pi/1.4, 21)

    for val, label in zip([u, v, p], ['u', 'v', 'p']):
        val_r = griddata((x, y), val.real,
                               (X, Y), method="cubic")
        val_i = griddata((x, y), val.imag,
                               (X, Y), method="cubic")
        
        vals = val_r + 1j * val_i
        max_val = np.max(abs(vals))
        for t in time:
            fig, ax = plot_setup('Along-shore ($\\rm{km}$)',
                                 'Cross-shore ($\\rm{km}$)',
                                 title=f'${label}$, time: {t:.2f}')
            
            c = ax.matshow((vals * np.exp(-1j*1.4 * t)).real,
                aspect="auto",
                cmap="seismic",
                extent=[x0, xN, y0, yN],
                origin="lower",
                vmin=-max_val,
                vmax=max_val,
            )
            
            if padding:
                x_padding, y_padding = padding
                rect = patches.Rectangle((x0+x_padding, y0+y_padding),
                                         (xN-x0)-2*x_padding,
                                         (yN-y0)-2*y_padding,
                                         linewidth=3,
                                         edgecolor='black', facecolor='none')
                ax.add_patch(rect)
            fig.colorbar(c, ax=ax)
            
            pt.show()

class animate_solutions(object):
    def __init__(self, vals, old_grid, new_grid=None, wave_frequency=1.4,
                 start_time=0, periods=1, N_period=50, frame_rate=10,
                 repeat=3, padding=(2.5e-2, 5e-2), bbox=None, file_name='test',
                 folder_dir='Baroclinic Animation', mode=1):
        self.old_grid, self.new_grid = old_grid, new_grid
        self.x, self.y = old_grid
        self.u, self.v, self.p = np.split(vals, 3)
        self.t0 = start_time
        self.repeat = repeat
        self.wave_frequency = wave_frequency
        self.period = 2*np.pi/self.wave_frequency
        self.tend = self.t0 + periods * self.period
        self.Nt = N_period
        self.time = np.linspace(self.t0, self.tend, self.Nt+1)
        self.fps = frame_rate
        self.file_name, self.folder_dir = file_name, folder_dir
        self.padding = padding
        self.mode = mode
        
        if self.new_grid is None:
            self.values = vals
            self.X, self.Y = self.old_grid
            grid = old_grid

        else:
            
            self.values = grid_convert([self.u, self.v, self.p],
                                       self.old_grid,
                                       self.new_grid)
            grid = new_grid
            
        self.x0 = grid[0][0, 0]
        self.xN = grid[0][-1, -1]
        self.y0 = grid[1][0, 0]
        self.yN = grid[1][-1, -1]
        self.bbox=(self.x0, self.xN, self.y0, self.yN) if bbox is None \
            else bbox
            
        
        self.fig_init()
        
        self.start_anim()
        
    def fig_init(self):
        from ppp.Plots import set_axis, subplots
        from matplotlib import patches
    
        self.fig, self.axis = subplots(3, 1, y_share=True)
        self.magnitudes = []
        self.plots = []
        labels = ['u', 'v', 'p']
        for i in range(3):
            title_ = f'${labels[i]}_{self.mode}$'
            ax = self.axis[i]
            set_axis(ax, title=title_, scale=.75)
            vals = self.values[i].real
            self.magnitudes.append(np.max(np.abs(vals)))
            
            c = ax.matshow(vals,
                    aspect="auto",
                    cmap="seismic",
                    extent=(self.x0, self.xN, self.y0, self.yN),
                    origin="lower",
                    vmin=-self.magnitudes[-1],
                    vmax=self.magnitudes[-1],
                )
            self.plots.append(c)
            
            if self.padding:
                x_padding, y_padding = self.padding
                rect = patches.Rectangle((self.x0+x_padding, self.y0+y_padding),
                                          (self.xN-self.x0)-2*x_padding,
                                          (self.yN-self.y0)-2*y_padding,
                                          linewidth=3,
                                          edgecolor='black', facecolor='none')
                ax.add_patch(rect)
            
            self.fig.colorbar(c, ax=ax)
            ax.set_xlim([self.bbox[0], self.bbox[1]])
            ax.set_ylim([self.bbox[2], self.bbox[3]])
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
        print('saving')
        self.anim.save(f'{self.folder_dir}/{self.file_name}.mp4',
                       writer=writervideo)

if __name__ == "__main__":
    pass
