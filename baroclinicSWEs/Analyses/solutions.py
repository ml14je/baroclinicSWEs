#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Thu May  5 12:28:01 2022

"""
import numpy as np

class Global_Solutions(object):
    def __init__(self, bbox, ω, inner_funcs, outer_funcs):
        x0, xN, y0, yN = bbox
        self.u = lambda x, y, t : \
            inner_funcs[0](x, y) * np.exp(-1j * ω * t) * \
                (x > x0) * (x < xN) * (y > y0) * (y < yN) + \
            outer_funcs[0](x, y, t) * \
                (1 - (x > x0) * (x < xN) * (y > y0) * (y < yN))

        self.v = lambda x, y, t : \
            inner_funcs[1](x, y) * np.exp(-1j * ω * t) * \
                (x > x0) * (x < xN) * (y > y0) * (y < yN) + \
            outer_funcs[1](x, y, t) * \
                (1 - (x > x0) * (x < xN) * (y > y0) * (y < yN))

        self.p = lambda x, y, t : \
            inner_funcs[2](x, y) * np.exp(-1j * ω * t) * \
                (x > x0) * (x < xN) * (y > y0) * (y < yN) + \
            outer_funcs[2](x, y, t) * \
                (1 - (x > x0) * (x < xN) * (y > y0) * (y < yN))
                
class Baroclinic_Solutions(object):
    def __init__(self, funcs, ω, Z1):
        self.u = lambda x, y, t : \
            funcs[0](x, y) * np.exp(-1j * ω * t)
        self.v = lambda x, y, t : \
            funcs[1](x, y) * np.exp(-1j * ω * t)
        self.p = lambda x, y, t : \
            funcs[2](x, y) * np.exp(-1j * ω * t)
        self.Z1 = Z1
    

def barotropic_flow(mesh,
                    param,
                    order,
                    h_min,
                    h_max,
                    slope_topography,
                    canyon_topography,
                    folder_name,
                    file_name,
                    plot_domain=False,
                    boundary_conditions="SPECIFIED",
                    scheme="Lax-Friedrichs",
                    θ=0.5,
                    rotation=True,
                    show_exact_kelvin=False,
                    save_all=True,
                    verbose=True,
                ):
    from ppp.File_Management import file_exist, dir_assurer
    import pickle, dill
    from DGFEM.dgfem import FEM

    sln_dir = f"{folder_name}/Functions/{file_name}_N={order}.pkl"
    dir_assurer(f"{folder_name}/Functions")
            
    try:
        with open(sln_dir, "rb") as inp:
            barotropic_solution = dill.load(inp)
            
    except (EOFError, FileNotFoundError):
        P, T, mesh_name = mesh
        x0, xN, y0, yN = param.bboxes[0]
        Lx, Ly = xN - x0, yN - y0
        
    
        # Far field barotropic tide in form of perturbed Kelvin Wave
        from barotropicSWEs.SWEs import kelvin_solver
    
        u_kelv, v_kelv, p_kelv, ω, k = kelvin_solver(
            slope_topography,
            param,
            coastal_lengthscale=(param.L_C + param.L_S) / param.L_R,
            tidal_amplitude=1,
            forcing_frequency=param.ω / param.f,
            foldername=f"{folder_name}/Kelvin Flow/{'_'.join(file_name.split('_')[1:-1])}",
            filename=f"KelvinMode_Domain={Lx}kmx{Ly}km_ω={param.ω:.1e}",
        )
        
        file_name += f"_ForcingFrequency={param.ω:.1e}s^{{{-1}}}"
        dir_assurer("Barotropic/FEM Objects")
        fem_dir = f"Barotropic/FEM Objects/{mesh_name}_N={order}.pkl"
        
        
    
        # Non-dimensionalise bbox coordinates
        x0, xN, y0, yN = (
            x0 * 1e3 / param.L_R,
            xN * 1e3 / param.L_R,
            y0 * 1e3 / param.L_R,
            yN * 1e3 / param.L_R,
        )
        s = P.shape
        X, Y = P.T
        X[abs(X - x0) < 1e-15] = x0
        X[abs(X - xN) < 1e-15] = xN
        Y[abs(Y - y0) < 1e-15] = y0
        Y[abs(Y - yN) < 1e-15] = yN
        P = np.array([X, Y]).T
        assert P.shape == s, "Incorrect recompilation of coordinates"

        wall_inds = np.where((Y == y0))[0]
        open_inds2 = np.where((X == x0) | (X == xN))[0]
        open_inds = np.where(Y == yN)[0]
        BC_maps, BC_Types = [wall_inds, open_inds, open_inds2], \
            ["Wall", "Open", "Open2"]
        BCs = dict(zip(BC_Types, BC_maps))
    
        if not file_exist(fem_dir):
            if save_all:
                with open(fem_dir, "wb") as outp:
                    if verbose:
                        print("Creating FEM class")
        
                    fem = FEM(P, T, N=order, BCs=BCs)
                    pickle.dump(fem, outp, pickle.HIGHEST_PROTOCOL)
                    
            else:
                fem = FEM(P, T, N=order, BCs=BCs)   
    
        else:
            if verbose:
                print("Loading FEM class")
    
            with open(fem_dir, "rb") as inp:
                fem = pickle.load(inp)
    
        X, Y = np.round(fem.x.T.flatten(), 15), np.round(fem.y.T.flatten(), 15)
        
        background_flow = (
            u_kelv(X, Y, 0),
            v_kelv(X, Y, 0),
        )
    
        from barotropicSWEs import Barotropic
        swes = Barotropic.solver(
            fem,
            param,
            flux_scheme=scheme,
            θ=θ,
            boundary_conditions="SPECIFIED",
            rotation=True,
            background_flow=background_flow,
            h_func=canyon_topography,
            wave_frequency=ω,
            rayleigh_friction=0,
        )
    
        from barotropicSWEs import SWEs
    
        numerical_sol = SWEs.boundary_value_problem(
            swes,
            np.zeros(X.shape),
            animate=False,
            file_name=f"{scheme.upper()}_order={order}",
            file_dir=f"{folder_name}/Solutions/{file_name}",
            save_solution=save_all,
        )
    
        irregular_solutions = np.split(numerical_sol, 3)
        kelvin_solutions = [u_kelv, v_kelv, p_kelv]
        from scipy.interpolate import CloughTocher2DInterpolator
        
        # Interpolate over irregular barotropic data
        funcs = [CloughTocher2DInterpolator(
            (X, Y), irregular_solutions[i], fill_value=0
            ) for i in range(3)]
        
        eps = 1e3/param.L_R
        bbox_temp = (x0+eps, xN-eps, y0, yN)
        
        barotropic_solution = Global_Solutions(
            bbox_temp, ω, funcs, kelvin_solutions
            )

        with open(sln_dir, "wb") as outp:
            if verbose:
                print("Saving Barotropic solution function")
                
            dill.dump(barotropic_solution, outp)

        
    return barotropic_solution

def baroclinic_flow(param,
                    order,
                    baroclinic_mesh,
                    barotropic_solution,
                    barotropic_name,
                    bathymetry_func,
                    boundary_conditions="Solid Wall",
                    scheme="Lax-Friedrichs",
                    θ=0.5,
                    rotation=True,
                    domain_extension=(500, 500), # in km
                    damping_width=200, # in km
                    rayleigh_friction_magnitude=.6,
                    save_all=True,
                    animate=True,
                    verbose=True
                ):
    

    assert scheme.upper() in [
        "CENTRAL",
        "UPWIND",
        "LAX-FRIEDRICHS",
        "RIEMANN",
        "PENALTY",
        "ALTERNATING",
    ], "Invalid choice of numerical flux."
    import pickle, dill
    from ppp.File_Management import dir_assurer, file_exist

    folder_name = "Baroclinic"
    baroclinic_name = barotropic_name + \
            f"_rho1={param.ρ_min:.0f}_rho2={param.ρ_max:.0f}_\
rho={param.ρ_ref:.0f}_h1={param.H_pyc:.0f}m_\
sponge=({domain_extension[0]:.1f}km,{domain_extension[1]:.0f}km)_\
width={damping_width:.0f}km_N={order}"
    folder_name = "Baroclinic"
    
    sln_dir = f"{folder_name}/Functions/{baroclinic_name}.pkl"
    dir_assurer(f"{folder_name}/Functions")
    
    try:
        with open(sln_dir, "rb") as inp:
            baroclinic_solution = dill.load(inp)
            
    except (EOFError, FileNotFoundError):
        from DGFEM.dgfem import FEM
        P, T, mesh_name = baroclinic_mesh
        x0, xN, y0, yN = param.bboxes[1]
        x0, xN, y0, yN = (
            x0 * 1e3 / param.L_R,
            xN * 1e3 / param.L_R,
            y0 * 1e3 / param.L_R,
            yN * 1e3 / param.L_R,
        )
        s = P.shape
        X, Y = P.T
        X[abs(X - x0) < 1e-15] = x0
        X[abs(X - xN) < 1e-15] = xN
        Y[abs(Y - y0) < 1e-15] = y0
        Y[abs(Y - yN) < 1e-15] = yN
        P = np.array([X, Y]).T
        assert P.shape == s, "Incorrect recompilation of coordinates"
    
        scheme_, boundary_conditions_ = scheme, boundary_conditions
        ω, r = param.ω/param.f, rayleigh_friction_magnitude
    
        from ppp.File_Management import dir_assurer, file_exist
        dir_assurer("Baroclinic/FEM Objects")
        fem_dir = f"Baroclinic/FEM Objects/{mesh_name}_N={order}.pkl"
    
        if not file_exist(fem_dir):
    
            with open(fem_dir, "wb") as outp:
                if verbose:
                    print("Creating Baroclinic FEM class")
    
                fem = FEM(P, T, N=order)
    
                if save_all:
                    pickle.dump(fem, outp, pickle.HIGHEST_PROTOCOL)
    
        else:
            if verbose:
                print("Loading Baroclinic FEM class")
    
            with open(fem_dir, "rb") as inp:
                fem = pickle.load(inp)
    
        from baroclinicSWEs.Baroclinic import solver
        
        baroclinic_swes = solver(
            param,
            fem,
            barotropic_solution,
            barotropic_name,
            baroclinic_name,
            bathymetry_func,
            flux_scheme=scheme_,
            boundary_conditions=boundary_conditions_,
            rotation=True,
            rayleigh_friction=r,
            domain_extension=domain_extension,
            damping_width=damping_width,
        )
    
        sols = baroclinic_swes.boundary_value_problem(wave_frequency=ω)
        u1, v1, p1 = np.split(sols, 3)
        assert not np.all(u1 - v1 == 0), "Error in solutions!"
        
        irregular_sols = np.array([u1, v1, p1])
        x, y = np.round(baroclinic_swes.X, 15), np.round(baroclinic_swes.Y, 15)
        
        from scipy.interpolate import CloughTocher2DInterpolator
        # Interpolate over irregular barotropic data
        solutions = [CloughTocher2DInterpolator(
            (x, y), irregular_sols[i], fill_value=0
            ) for i in range(3)]
        
        baroclinic_solution = Baroclinic_Solutions(solutions, ω, baroclinic_swes.Z1_func)
            
        with open(sln_dir, "wb") as outp:
            if verbose:
                print("Saving Baroclinic solution function")
                
            dill.dump(baroclinic_solution, outp)
    
    if animate:
        for zoom in [False]:
            Nx, Ny = 1000, 1000
            bbox_temp = np.array(param.bboxes[1]) * 1e3 /param.L_R if not \
                zoom else np.array([-100, 100, 0, 200]) * 1e3 /param.L_R
            x0, xN, y0, yN = bbox_temp
            
            xg, yg = np.linspace(x0, xN, Nx+1), np.linspace(y0, yN, Ny+1)
            Xg, Yg = np.meshgrid(xg, yg)
    
            bathymetry = param.H_D * bathymetry_func(Xg, Yg)
            Z1 = baroclinic_solution.Z1(bathymetry)
            # print(self.barotropic_sols)
            u = baroclinic_solution.u(Xg, Yg, 0)
            v = baroclinic_solution.v(Xg, Yg, 0)
            p = baroclinic_solution.p(Xg, Yg, 0)
            LR = 1e-3 * param.L_R
    
            from baroclinicSWEs.Baroclinic import plot_solutions
            plot_solutions(np.array([param.c * u * Z1/param.g,
                                        param.c * v * Z1/param.g,
                                        param.ρ_ref * \
                                            (param.c**2) * p * Z1/param.g]),
                              (LR * Xg, LR * Yg),
                              wave_frequency=1.4,
                              bbox=[LR * L for L in bbox_temp],
                              padding=param.bboxes[0],
                              repeat=1,
                              file_name=f"alpha={param.alpha:.2f}_\
beta={param.beta:.2f}_CanyonWidth={param.canyon_width:.1f}km_order={order}_{tuple(param.bboxes[1])}_\
sponge=({domain_extension[0]:.0f}km,{domain_extension[1]:.0f}km)_zoom={zoom}",
                              folder_dir="Baroclinic/Baroclinic Animation",
                              mode=1
                              )

    return baroclinic_solution