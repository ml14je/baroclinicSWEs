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
        fem,
        barotropic_sols,
        param,
        h_func,
        upper_layer_thickness=100,
        upper_layer_density=1025,
        lower_layer_density=1050,
        periods=2,
        flux_scheme="central",
        boundary_conditions="Solid Wall",
        rotation=True,
        θ=1,
        wave_frequency=1.4,
        rayleigh_friction=0.05,
    ):
        from scipy.sparse import diags

        self.rotation = rotation
        self.fem = fem
        self.param = param
        self.ω = wave_frequency
        self.r = rayleigh_friction
        self.X, self.Y = self.fem.x.flatten("F"), self.fem.y.flatten("F")
        self.barotropic_sols = barotropic_sols

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
        self.barotropic_forcing()

        self.matrix_setup()

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
        dt = 0.5 * np.min(dtscale) * rmin * 2 / 3

        from math import ceil

        Nt = ceil((t_final / N_frames) / dt)
        dt = t_final / (Nt * N_frames)

        if Nout is None:
            Nout = self.fem.N
        if file_name is None:
            file_name = f"Solutions: {self.scheme}"
        u, v, η = np.copy(u0), np.copy(v0), np.copy(η0)
        ω = self.ω if ω is None else ω
        N = self.fem.Np * self.fem.K
        from scipy.sparse import csr_matrix as sp
        from scipy.sparse import identity, block_diag

        i, o = identity(N), sp((N, N))
        I2 = block_diag(2 * [i] + [o])
        assert ω is not None

        # Potential forcing φ of frequency ω
        T = 2 * np.pi / ω
        assert not np.all(u0) == 0

        if np.all(u0) == 0:
            F = (
                lambda t: self.forcing(φ)
                * np.exp(-1j * ω * t)
                * np.tanh(t / (4 * T)) ** 4
            )
            r_ = lambda t: self.r * (1 - np.tanh(t / (4 * T)) ** 4)

        else:
            F = lambda t: self.forcing(φ) * np.exp(-1j * ω * t)
            r_ = lambda t: self.r

        def rhs(t, y):
            return (self.A - r_(t) * I2) @ y + F(t)

        y_0 = np.concatenate([u, v, η], axis=0)[:, None]

        if method in [
            "Forward Euler",
            "Explicit Midpoint",
            "Heun",
            "Ralston",
            "RK3",
            "Heun3",
            "Ralston3",
            "SSPRK3",
            "RK4",
            "3/8 Rule",
            "RK5",
        ]:
            from ppp.Explicit import explicit
            from math import ceil

            timestepping = explicit(
                rhs,
                y_0,
                0,
                t_final,
                N=ceil(t_final / dt),
                method=method,
                nt=ceil(((t_final / dt) + 1) / (N_frames + 1)),
                verbose=False,
            )

        elif method in [
            "Heun_Euler",
            "Runge–Kutta–Fehlberg",
            "Bogacki–Shampine",
            "Fehlberg",
            "Cash–Karp",
            "Dormand–Prince",
        ]:
            from ppp.Embedded import embedded

            timestepping = embedded(rhs, y_0, 0, t_final, method=method)

        else:
            raise ValueError("method argument is not defined.")

        self.time = timestepping.t_vals
        sols = timestepping.y_vals[:, :, 0]

        if animate:
            self.animate_sols(np.copy(sols), file_name=file_name, Nout=Nout)

        return timestepping.t_vals, sols

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
            Ipη3[mapB, vmapB] = Im[mapB, vmapB]

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
        )  # allows volume transport flux in mass conservation

        α, β, γ, θ = self.α, self.β, self.γ, self.θ
        θ = θ * np.ones(Ny.shape[0])
        θ[mapB] = 0.5

        # C1_sqrd_av = (self.baroclinic_wavespeed_apprx[vmapP] + \
        #               self.baroclinic_wavespeed_apprx[vmapM]) / 2

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

        assert self.A.shape[0] == self.U_D.shape[0]

    def rhs3(self, sols, t):

        return self.A @ sols

    def evp(self, k_vals=10):
        from scipy.linalg import eig

        vals, vecs = eig(1j * self.A.todense())
        return vals, vecs.T

    def barotropic_forcing(self):
        from scipy.sparse import diags
        u0, v0, p0 = np.split(self.barotropic_sols, 3, axis=0)
        # T10 = self.param.H_D * \
        #     self.modal_interaction_coefficients_apprx[1][0](self.bathymetry)

        Nx, Ny = 1000, 1000
        x0, xN, y0, yN = self.param.bbox
        dx = (xN - x0)/Nx
        dy = (yN - y0)/Ny
        
        xg, yg = np.linspace(x0, xN, Nx+1), np.linspace(y0, yN, Ny+1)
        X, Y = np.meshgrid(xg, yg)
        bathymetry = self.h_func(X, Y)#/self.param.H_D
        
        from barotropicSWEs.topography import grad_function
        
        from ppp.Plots import plot_setup
        import matplotlib.pyplot as pt
        from scipy.interpolate import griddata
        
        gridded_vals = []
        for val in [u0, v0, p0, self.bathymetry]:
            val_r = griddata((self.X, self.Y), val.real,
                                    (X, Y), method="cubic")
            val_i = griddata((self.X, self.Y), val.imag,
                                    (X, Y), method="cubic")
            val = val_r + 1j * val_i
            gridded_vals.append(val)

        # from topography import grad_function
        # ρ = swes.param.ρ_ref
        u, v, p, h = gridded_vals #Gridded solution non-dimensional system variables, and fluid depth
        # u, v, eta, h = u * swes.param.c, v * swes.param.c, swes.param.H_D * p, swes.param.H_D * h #Dimensionalised quantities
        # H = swes.param.H_D * self.h_func(X, Y) #Dimensional fluid depth projected on mesh grid
        # ux, uy = grad_function(u, dx, dy) # spatial derivatives of along-shore velocity
        # vx, vy = grad_function(v, dx, dy) # spatial derivatives of cross-shore velocity
        hx, hy = grad_function(bathymetry/self.param.H_D, dx, dy) # bathymetric gradients
        # Qx, Qy = h * u, h * v # Along-shore and cross-shore volume fluxes, respectivelu
        # vorticity = vx - uy # Vorticity
        # u_gradh = u * hx + v * hy # u . grad(h)
        
        L_R = self.param.L_R*1e-3
        T10 = self.modal_interaction_coefficients_apprx[1][0](bathymetry) * \
            self.param.H_D
        c0_sqrd = self.wave_speed_sqrd_functions_apprx[0](bathymetry)/ \
            (self.param.c**2)
        u1_forcing = -T10 * p * hx
        v1_forcing = -T10 * p * hy
        p1_forcing = -T10 * c0_sqrd * (u * hx + v * hy)
        
        for forcing, title in zip([u1_forcing, v1_forcing, p1_forcing],
                                  ['u_1', 'v_1', 'p_1']):
            print(self.X.shape, forcing.shape, X.shape)
            # force = griddata((self.X,self.Y), forcing, (X, Y))
            v = np.max(np.abs(forcing))
            fig, ax = plot_setup('Along-shore (km)', 'Cross-shore (km)',
                                 title=f'${title}$')
            c = ax.matshow(
                forcing.real,
                cmap="seismic",
                vmax=v,
                vmin=-v,
                extent=[x0*L_R, xN*L_R,
                        y0*L_R, yN*L_R],
                aspect="auto",
                origin="lower",
            )
            fig.colorbar(c, ax=ax)
            pt.show()
            
        raise ValueError
        modal_forcing = np.concatenate(
            [u1_forcing,
             v1_forcing,
             p1_forcing
             ], axis=1)

        # Barotropic forcing + prescription of boundary conditions
        return modal_forcing - self.U_D

    def bvp(
        self,
        φ,
        rayleigh_friction=None,
        wave_frequency=None,
        file_name="BVP Animation",
        animate=True,
        frames=20,
        verbose=True,
    ):
        from scipy.sparse import identity, block_diag  # , diags
        from scipy.sparse.linalg import spsolve
        from scipy.sparse import csr_matrix as sp

        ω = self.ω if wave_frequency is None else wave_frequency
        r = self.r if rayleigh_friction is None else rayleigh_friction
        self.frames = frames
        self.time = np.linspace(0, 2 * np.pi / ω, self.frames + 1)

        N = self.fem.Np * self.fem.K
        i, o = identity(N), sp((N, N))

        I, I2 = block_diag([i] * 3), block_diag(2 * [i] + [o])
        assert ω is not None

        A = -self.A - 1j * ω * I + r * I2
        if verbose:
            print("Solving BVP using spsolve")

        sols = spsolve(A, self.forcing(φ))[None, :]

        return sols


if __name__ == "__main__":
    pass
