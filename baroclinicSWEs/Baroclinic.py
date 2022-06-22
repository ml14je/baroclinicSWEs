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
        barotropic_name,
        baroclinic_name,
        h_func,
        periods=2,
        flux_scheme="central",
        boundary_conditions="Solid Wall",
        data_dir='Baroclinic',
        rotation=True,
        θ=1,
        wave_frequency=1.4,
        rayleigh_friction=0,
        domain_extension=(200, 200), #in kilometres
        damping_width=50 #in kilometres
    ):
        from scipy.sparse import diags
        
        self.param = param
        self.fem = fem
        self.order = fem.N
        self.barotropic_sols = barotropic_sols
        self.barotropic_dir = barotropic_name
        self.baroclinic_dir = baroclinic_name
        self.domain_extension = domain_extension
        self.damping_width = damping_width
        
        self.rotation = rotation
        self.ω = wave_frequency
        self.rayleigh_friction = rayleigh_friction

        self.X, self.Y = self.fem.x.flatten("F"), self.fem.y.flatten("F")
        self.data_dir = f"{data_dir}/Solutions"
        
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

        assert self.boundary_conditions in ["SOLID WALL", "OPEN FLOW",
                                            "MOVING WALL", "SPECIFIED", "REFLECTING"], \
            "Invalid Boundary Condition"

        if h_func is None:
            self.h_func = np.vectorize(lambda X, Y: self.param.H_D)
        else:
            self.h_func = lambda x, y : self.param.H_D * h_func(x, y)

        self.bathymetry = self.h_func(self.X, self.Y)
        self.upper_layer_thickness = param.H_pyc
        self.upper_layer_density = param.ρ_min
        self.lower_layer_density = param.ρ_max
        self.reference_density = param.ρ_ref
        self.rel_density_diff = (param.ρ_max-param.ρ_min)/param.ρ_max
        
        from baroclinicSWEs.modal_decomposition import MultiLayerModes
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
            
        # Scalings
        self.c_scale = self.param.c
        self.p_scale = self.param.c**2
        self.hor_scale, self.vert_scale = self.param.L_R, self.param.H_D
            
        self.C0_sqrd = diags(self.barotropic_wavespeed_apprx/self.param.c**2)
        self.C1_sqrd = diags(self.baroclinic_wavespeed_apprx/self.param.c**2)
        self.Z0_func, self.Z1_func = self.approx_normalisations #normalisations
        self.Z0, self.Z1 = self.Z0_func(self.bathymetry), self.Z1_func(self.bathymetry)

        # self.T10 = diags(param.H_D * self.modal_interaction_coefficients_apprx[1][0](self.bathymetry))
        # self.T11 = diags(param.H_D * self.modal_interaction_coefficients_apprx[1][1](self.bathymetry))
        
        self.matrix_setup()
        self.baroclinic_friction()
        self.generate_barotropic_forcing()


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

            # u+ = (ny^2 - nx^2) * -u- - 2 * nx * ny * v- (non-dimensional impermeability)
            Ipu1[mapB, vmapB] = ((Ny @ Ny - Nx @ Nx) @ Im)[mapB, vmapB]
            Ipu2[mapB, vmapB] = -2 * (Nx @ Ny @ Im)[mapB, vmapB]

            # v+ = (nx^2 - ny^2) * -v- - 2 * nx * ny * u- (non-dimensional impermeability)
            Ipv2[mapB, vmapB] = ((Nx @ Nx - Ny @ Ny) @ Im)[mapB, vmapB]
            Ipv1[mapB, vmapB] = -2 * (Nx @ Ny @ Im)[mapB, vmapB]
            
        elif self.boundary_conditions == 'REFLECTING':
            Ipu1[mapB, vmapB] = Im[mapB, vmapB]
            Ipv2[mapB, vmapB] = Im[mapB, vmapB]
            Ipη3[mapB, vmapB] = Im[mapB, vmapB]

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

            raise ValueError('Not coded. Choose a different boundary condition or \
code it yourself haha')

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

    def generate_barotropic_forcing(self,
                                    animate_barotropic_solutions=False,
                                    animate_barotropic_forcing=False,
                                    save=False):
        from ppp.File_Management import dir_assurer, file_exist
        x0, xN, y0, yN = self.param.bboxes[0]
        LR = self.param.L_R * 1e-3
        file_dir = self.baroclinic_dir + \
            f'_domainwidth={(xN-x0):.0f}kmx{(yN-y0):.0f}km_baroclinic_order={self.fem.N}'
        dir_assurer('Baroclinic/Barotropic Forcing')

        if not file_exist(f'Baroclinic/Barotropic Forcing/{file_dir}.npz'):
            X1, Y1 = self.X, self.Y
            dx, dy = .5e3/self.param.L_R, .5e3/self.param.L_R
            
            bbox_temp = np.array(self.param.bboxes[1]) * 1e3 /self.param.L_R
            bbox_temp[2] = 0 # y0 = 0 (coastline)
            x0, xN, y0, yN = bbox_temp

            xg, yg = np.arange(x0, xN+dx, dx), np.arange(y0, yN+dy, dy)
            Xg, Yg = np.meshgrid(xg, yg)

            bathymetry = self.h_func(Xg, Yg)

            from barotropicSWEs.Configuration import topography
            from scipy.interpolate import griddata
                    

            # Z0 = self.Z0_func(bathymetry) #dimensional
            Z1 = self.Z1_func(bathymetry) #dimensional
            u = self.barotropic_sols.u(Xg, Yg, 0) # Non-dimensional w.r.t c
            v = self.barotropic_sols.v(Xg, Yg, 0) # Non-dimensional w.r.t c
            p = self.barotropic_sols.p(Xg, Yg, 0) # Non-dimensional w.r.t \rho c^2
            assert not np.all(u - v == 0), "Error in function uniqueness!"
            
            if animate_barotropic_solutions:
                plot_solutions(np.array([self.c_scale * u,
                                         self.c_scale * v,
                                         1e-3 * self.reference_density * \
                                             self.p_scale * p]),
                                  (LR * Xg, LR * Yg),
                                  wave_frequency=1.4,
                                  bbox=[LR * L for L in bbox_temp],
                                  padding=(x0, xN, y0, yN),
                                  file_name=self.barotropic_dir,
                                  folder_dir="Baroclinic/Barotropic Animation",
                                  mode=0,
                                  repeat=1,
                                  )
            
            hx, hy = topography.grad_function(bathymetry/self.vert_scale,
                                              dy, dx)
            
            T10 = self.modal_interaction_coefficients_apprx[1][0](bathymetry)
                
            T10 *= self.vert_scale #Non-dimensional

            c0_sqrd = self.wave_speed_sqrd_functions_apprx[0](bathymetry)/ \
                (self.c_scale**2)
            
            ratio = (self.domain_extension[0] + self.damping_width)/self.domain_extension[0]
            R = self.σ_x(ratio*Xg)
            u1_forcing = -T10 * p * hx * (1-R)
            v1_forcing = -T10 * p * hy * (1-R)
            p1_forcing = -T10 * c0_sqrd * (u * hx + v * hy) * (1-R)

            forcings = [
                u1_forcing,
                v1_forcing,
                p1_forcing]
            
            forcings_dim = [
                u1_forcing * self.p_scale * (1/self.hor_scale) * Z1,
                v1_forcing * self.p_scale * (1/self.hor_scale) * Z1,
                p1_forcing * (self.c_scale**3) * (1/self.hor_scale) * Z1]

            if animate_barotropic_forcing:                    
                bbox_temp = np.array(self.param.bboxes[1]) * 1e3 /self.param.L_R

                plot_solutions(np.array(forcings_dim), (Xg*LR, Yg*LR),
                                  wave_frequency=1.4,
                                  bbox=[LR * L for L in bbox_temp],
                                  padding=(x0, xN, y0, yN),
                                  file_name=f"{self.baroclinic_dir}_forcing",
                                  folder_dir="Baroclinic/Barotropic Animation",
                                  mode=1
                                  )

            forcing_baroclinic_grid = []
            for forcing, title in zip(forcings,
                                      ['u_1', 'v_1', 'p_1']):
                forcing = griddata((Xg.flatten(), Yg.flatten()),
                                   forcing.flatten(),
                                   (X1, Y1), method="cubic",
                                   fill_value=0)
                forcing_baroclinic_grid.append(forcing)
            if save:
                from ppp.Numpy_Data import save_arrays
                save_arrays(file_dir, tuple(forcing_baroclinic_grid),
                            wd="Baroclinic/Barotropic Forcing")
            
        else:
            from ppp.Numpy_Data import load_arrays
            forcing_baroclinic_grid = load_arrays(
                file_dir,
                wd="Baroclinic/Barotropic Forcing"
                )

        self.barotropic_forcing = np.concatenate(
            forcing_baroclinic_grid)[:, None]

    def baroclinic_friction(self):
        from baroclinicSWEs.Configuration import sponge_layer
        from scipy.sparse import block_diag, diags
        self.σ_x = lambda x : sponge_layer.lavelle_x(x,
                                                     self.param,
                                                     magnitude=1,
                                                     x_padding=self.domain_extension[0] * 1e3/self.param.L_R,
                                                     xc=0,
                                                     D=self.damping_width)
        self.σ_y = lambda y : sponge_layer.lavelle_y(y,
                                                     self.param,
                                                     magnitude=1,
                                                     y_padding=self.domain_extension[1] * 1e3/self.param.L_R,
                                                     yc=self.param.L_C/self.param.L_R,
                                                     D=self.damping_width)
        self.σ = lambda x, y : self.σ_x(x) + self.σ_y(y)


        self.R = self.rayleigh_friction * block_diag(
            [diags(self.σ_x(self.X)),
             diags(self.σ_y(self.Y)), 
             diags(self.σ(self.X, self.Y))]
            )
        
    def plot_sponge_layer(self, sponge_padding):
        from ppp.Plots import plot_setup
        from matplotlib import patches
        import matplotlib.pyplot as pt
        x_padding, y_padding = sponge_padding
        x0, xN, y0, yN = self.param.bboxes[1]
        x = np.linspace(x0, xN, 101)
        y = np.linspace(y0, yN, 101)
        X, Y = np.meshgrid(x, y)
        r = .05
        fig, ax = plot_setup('Along-shore (km)', 'Cross-shore (km)')
        R = r * self.sponge_function(X, Y)
        c = ax.matshow(
            R,
            cmap="seismic",
            vmax=r,
            vmin=-r,
            extent=self.param.bboxes[1],
            aspect="auto",
            origin="lower",
        )
        rect = patches.Rectangle((x0+x_padding, y0+y_padding),
                                  (xN-x0)-2*x_padding,
                                  (yN-y0)-2*y_padding,
                                  linewidth=3,
                                  edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        fig.colorbar(c, ax=ax)
        pt.show()

    def boundary_value_problem(self, wave_frequency=None, verbose=True,
                               save_solution=True):
        from scipy.sparse import identity, block_diag  # , diags
        from scipy.sparse.linalg import spsolve
        from scipy.sparse import csr_matrix as sp
        from ppp.File_Management import file_exist

        ω = self.ω if wave_frequency is None else wave_frequency
        # LR = 1e-3 * self.param.L_R
        xN, x0, yN, y0 = self.param.bboxes[1]

        self.name = self.baroclinic_dir
        if not file_exist(f'{self.data_dir}/{self.name}.npz'):
            N = self.fem.Np * self.fem.K
            i = identity(N)
    
            I = block_diag([i] * 3)
            assert ω is not None
            
            # damping = self.R #Alternative: block_diag(2 * [self.R_friction] + o)
            A = -self.A - 1j * ω * I + self.R
            if verbose:
                print("Solving BVP using spsolve", flush=True)
                
            sols = spsolve(sp(A), self.barotropic_forcing)
            
            if save_solution:
                from ppp.Numpy_Data import save_arrays
                save_arrays(self.name, (sols,), folder_name=self.data_dir)

        else:
            from ppp.Numpy_Data import load_arrays
            if verbose:
                print("Data exists", flush=True)
            sols, = load_arrays(self.name, folder_name=self.data_dir)

        return sols
                    
    def animate_baroclinic(self, file_name='', extent=None, Nx=200, Ny=200):
        file_name_ = file_name if file_name else self.barotropic_dir
        LR = self.param.L_R * 1e-3

        if (extent is None) or (extent==self.param.bboxes[1]):
            X, Y = self.regular_grid
            bathymetry = self.h_func(X, Y)
            u, v, p = self.regular_sols
            extent = self.param.bboxes[1]
            
        else:
            x0, xN, y0, yN = extent
            file_name_ += f'_[{(xN-x0)*LR:.0f}km,{(yN-y0)*LR:.0f}km]'
            Xold, Yold = self.regular_grid
            x, y = np.linspace(x0, xN, Nx+1), np.linspace(y0, yN, Ny+1)
            X, Y = np.meshgrid(x, y)
            bathymetry = self.h_func(X, Y)
            old_sols = np.array([sols.flatten() for sols in self.regular_sols])
            new_sols = grid_convert(old_sols, (Xold.flatten(), Yold.flatten()),
                                    (X, Y))
            u, v, p = new_sols
            
    
        Z1 = self.Z1_func(bathymetry) # Baroclinic normalisation coefficient
        Z1_nd = Z1/np.sqrt(self.param.g) # Non-dimensional baroclinic normalisation coefficient
        LR = self.param.L_R * 1e-3 #Rossby Radii in km's

        plot_solutions(np.array([self.c_scale * u * Z1_nd,
                                    self.c_scale * v * Z1_nd,
                                    self.upper_layer_density * self.p_scale * p * Z1_nd]),
                          (LR * X, LR * Y),
                          wave_frequency=1.4,
                          bbox=[LR * L for L in extent],
                          padding=(0, 0),
                          file_name=file_name_,
                          folder_dir="Baroclinic Animation",
                          mode=1
                          )
                    
    
    def energy_flux(self, extent=[-.025, .025, 0, .05]):
        X, Y = self.regular_grid
        bathymetry = self.h_func(X, Y)
        baroclinic_wavespeed_sqrd = \
            self.wave_speed_sqrd_functions_apprx[1](bathymetry)
            
        
        norm = baroclinic_wavespeed_sqrd * \
            (self.upper_layer_density**2)/self.lower_layer_density
        
        u, v, p = self.regular_sols
        
        self.Jx = .5 * norm * np.real(u * p.conjugate()) * self.c_scale * self.vert_scale
        self.Jy = .5 * norm * np.real(v * p.conjugate()) * self.c_scale * self.vert_scale
        self.plot_flux(extent)
        Jx, Jy = self.calculate_radiating_energy_flux(extent)
        
        return Jx, Jy

    def plot_flux(self, extent):
        from ppp.Plots import plot_setup
        from matplotlib import patches
        X, Y = self.regular_grid
        
        x0, xN, y0, yN = extent
        LR = 1e-3 * self.param.L_R
    
        for J, title_ in zip([self.Jx, self.Jy], ["$J_x$", "$J_y$"]):
            val_max = np.nanmax(np.abs(J))
            fig, ax = plot_setup("Alongshore (km)", "Crossshore (km))",
                                 title=f"Energy Flux {title_} (W/m)")
            c = ax.matshow(J,
                    aspect="auto",
                    cmap="seismic",
                    extent=[LR * L for L in self.param.bboxes[1]],
                    vmax=val_max, vmin=-val_max,
                )
    
            rect = patches.Rectangle((LR*x0, LR*y0),
                                     LR*(xN-x0),
                                     LR*(yN-y0),
                                     linewidth=3,
                                     edgecolor='black', facecolor='none')
            ax.add_patch(rect)
            fig.colorbar(c, ax=ax)

            from ppp.Plots import save_plot
            save_plot(fig, ax,
                      f"{title_.replace('$', '').replace('_', '')}_{self.name}",
                          folder_name='Baroclinic Energy Flux Field')                

    def calculate_radiating_energy_flux(self, extent, Nx=100, Ny=100):
        X, Y = self.regular_grid
        x0, xN, y0, yN = extent
        x_grid = np.linspace(x0, xN, Nx+1)
        y_grid = np.linspace(y0, yN, Ny+1)
        X_new, Y_new = np.meshgrid(x_grid, y_grid)
        dx, dy = (xN - x0)/Nx, (yN - y0)/Ny
        
        from scipy.interpolate import griddata
        self.Jx_new = griddata((X.flatten(), Y.flatten()), self.Jx.flatten(), (X_new, Y_new), method='cubic',
                           fill_value=0)
        self.Jy_new = griddata((X.flatten(), Y.flatten()), self.Jy.flatten(), (X_new, Y_new), method='cubic',
                           fill_value=0)
        
        
        Jx_vals1 = self.Jx_new[:, 0]
        Jx_vals2 = self.Jx_new[:, -1]
        
        Jy_vals1 = self.Jy_new[0, :]
        Jy_vals2 = self.Jy_new[-1, :]

        from ppp.Plots import plot_setup, save_plot
        
        fig, ax = plot_setup("y", "Time-Averaged Along-Shore Energy Flux")
        ax.plot(y_grid, Jx_vals1, label='counter-flow')
        ax.plot(y_grid, Jx_vals2, label='contra-flow')
        save_plot(fig, ax,
                  f"Jx_{self.name}",
                      folder_name='Baroclinic Energy Flux')
            
        
        fig, ax = plot_setup("x", "Time-Averaged Cross-Shore Energy Flux")
        ax.plot(x_grid, Jy_vals1, label='shore-ward')
        ax.plot(x_grid, Jy_vals2, label='ocean-ward')
        save_plot(fig, ax,
                  f"Jy_{self.name}",
                      folder_name='Baroclinic Energy Flux')
            
        Jy_t = dy*np.sum(Jx_vals2-Jx_vals1)/(yN - y0)
        Jx_t = dx*np.sum(Jy_vals2-Jy_vals1)/(xN - x0)
        
        return Jx_t, Jy_t

def grid_convert(vals, old_grid, new_grid):
    from scipy.interpolate import griddata
    new_vals = []
    for val in vals:
        new_vals.append(griddata(old_grid, val,
                         new_grid, method="cubic"))
        
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
        vals = griddata((x, y), val,
                               (X, Y), method="cubic")

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

class plot_solutions(object):
    def __init__(self, vals, old_grid, new_grid=None, wave_frequency=1.4,
                 start_time=0, periods=1, N_period=50, frame_rate=10,
                 repeat=3, padding=None, bbox=None, file_name='test',
                 folder_dir='Baroclinic Animation', mode=1, key_value=5e-2,
                 x_pos=.9, y_pos=.06, animate=True):
        self.old_grid, self.new_grid = old_grid, new_grid
        self.x, self.y = old_grid
        self.u, self.v, self.p = np.split(vals, 3)
        self.x_pos, self.y_pos = x_pos, y_pos

        self.u = self.u[0, ::40, ::40]
        self.v = self.v[0, ::40, ::40]
        self.p = self.p[0]
        self.mode = mode
        self.key_value = key_value
        
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
        if animate:
            self.start_anim()
            
        else:
            from ppp.Plots import save_plot
            self.fig.tight_layout()
            # save_plot(self.fig, self.ax, f'{self.folder_dir}/{self.file_name}')
        
    def fig_init(self):
        from ppp.Plots import plot_setup, add_colorbar
        from matplotlib import patches

        self.fig, self.ax = plot_setup(
            x_label="Along-shore (km)", y_label="Cross-shore (km)"
        )
        self.max_p = np.nanmax(np.abs(self.p))
        self.c = self.ax.contourf(
            (self.p).real,
            cmap="seismic",
            alpha=0.5,
            extent=(self.x0, self.xN, self.y0, self.yN),
            origin="lower",
            levels=np.linspace(-self.max_p, self.max_p, 21),
        )
        
        cbar = add_colorbar(self.c)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel('Pressure ($\\rm{Pa}$)', rotation=270,
                            fontsize=16, labelpad=20)

        self.Q = self.ax.quiver(
            self.X[::40, ::40],
            self.Y[::40, ::40],
            (self.u).real,
            (self.v).real,
            width=0.002,
            scale=1,
        )

        self.ax.quiverkey(
            self.Q,
            self.x_pos,
            self.y_pos,
            self.key_value,
            r"$5\,\rm{cm/s}$",
            labelpos="W",
            coordinates="figure",
            fontproperties={"weight": "bold", "size": 18},
        )
        if self.padding:
            x1, x2, y1, y2 = self.padding
            rect = patches.Rectangle((x1, y1),
                                      x2-x1, y2-y1,
                                      linewidth=3,
                                      edgecolor='black', facecolor='none')
            self.ax.add_patch(rect)
        
        self.ax.set_aspect("equal")
        self.ax.set_xlim([self.bbox[0], self.bbox[1]])
        self.ax.set_ylim([self.bbox[2], self.bbox[3]])
        self.fig.tight_layout()

    def animate(self, k):
        sgn = self.wave_frequency/np.abs(self.wave_frequency)
        phase = np.exp(-sgn*2j*np.pi*k/self.Nt)
        
        for c in self.c.collections:
            c.remove()
            
        self.c = self.ax.contourf(
            (self.p * phase).real,
            cmap='seismic',
            alpha=0.5,
            extent=(self.x0, self.xN, self.y0, self.yN),
            origin="lower",
            levels=np.linspace(-self.max_p, self.max_p, 21)
        )
        
        self.Q.set_UVC((self.u * phase).real, (self.v * phase).real)
        self.ax.set_aspect("equal")
        self.ax.set_xlim([self.bbox[0], self.bbox[1]])
        self.ax.set_ylim([self.bbox[2], self.bbox[3]])  
        self.fig.tight_layout()
            
    def start_anim(self):
        # import os
        import matplotlib.pyplot as pt
        import matplotlib.animation as animation
        from ppp.File_Management import dir_assurer

        dir_assurer(self.folder_dir)
        self.anim = animation.FuncAnimation(self.fig,
                        self.animate, frames=self.repeat*self.Nt)
            
        writervideo = animation.FFMpegWriter(fps=self.fps)
        self.anim.save(f'{self.folder_dir}/{self.file_name}.mp4',
                       writer=writervideo)
        
        pt.close(self.fig)

if __name__ == "__main__":
    pass
