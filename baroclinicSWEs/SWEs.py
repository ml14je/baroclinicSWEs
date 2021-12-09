#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Sun Aug 22 12:27:53 2021

"""
import numpy as np

def startup(
    bbox,
    param,
    h_min,
    h_max,
    order,
    mesh_name="",
    upper_layer_depth=200,
    canyon_width=5e-3,
    canyon_intrusion=1.5e-2,
    plot_domain=False,
    boundary_conditions=["SPECIFIED", 'OPEN FLOW'],
    scheme="Lax-Friedrichs",
    potential_forcing=False,
    background_flow='Kelvin',
    θ=0.5,
    rotation=True,
    wave_frequency=1.4,
    wavenumber=1.4,
    coastal_shelf_width=2e-2,
    coastal_lengthscale=3e-3,
    rayleigh_friction=5e-2,
    sponge_padding=(1e-2, 5e-3),
    verbose=True,
):

    #Barotropic boundary conditions
    assert boundary_conditions[0].upper() in ["SOLID WALL", "OPEN FLOW",
        "MOVING WALL", "SPECIFIED"], "Invalid choice of boundary conditions."
    
    #Baroclinic boundary conditions
    assert boundary_conditions[1].upper() in ["SOLID WALL", "OPEN FLOW",
        "MOVING WALL", "SPECIFIED"], "Invalid choice of boundary conditions."

    assert background_flow.upper() in ['KELVIN', 'CROSSSHORE'], "Invalid \
choice of background flow."

    assert scheme.upper() in [
        "CENTRAL",
        "UPWIND",
        "RIEMANN",
        "LAX-FRIEDRICHS",
        "PENALTY",
        "ALTERNATING",
    ], "Invalid choice of numerical flux."

    x0, xN, y0, yN = bbox
    Lx, Ly = xN - x0, yN - y0

    k, ω, r = wavenumber, wave_frequency, rayleigh_friction
    LC, λ = coastal_shelf_width, coastal_lengthscale
    ΔL, w = canyon_intrusion, canyon_width
    
    scheme_, potential_forcing_ = scheme, potential_forcing
    background_flow_, rotation_, verbose_ = background_flow, rotation, verbose

    param.domain_size = (Lx, Ly)
    param.L_C = coastal_shelf_width * param.L_R
    param.L_S = (coastal_lengthscale - coastal_shelf_width) * param.L_R

    from barotropicSWEs.topography import canyon_func1
    h_func = lambda x, y: canyon_func1(x, y, canyon_width=w, canyon_intrusion=ΔL,
                          coastal_shelf_width=LC, coastal_lengthscale=λ)
    h_min = (λ - LC)/10 if w < 1e-5 else min(w/4, (λ - LC)/10)

    from barotropicSWEs.make_canyon_meshes import mesh_generator
    P, T, mesh_name = mesh_generator(
        bbox,
        param,
        h_min,
        h_max,
        canyon_func=h_func,
        canyon_width=w,
        plot_mesh=False,
        coastal_shelf_width=LC,
        coastal_lengthscale=λ,
        verbose=verbose_,
    )

    barotropic_sols = barotropic_flow(bbox,
                                      (P, T, mesh_name),
                                      param,
                                      order,
                                      canyon_width=w,
                                      plot_domain=plot_domain,
                                      boundary_conditions=boundary_conditions[0],
                                      scheme=scheme_,
                                      potential_forcing=potential_forcing_,
                                      background_flow=background_flow_,
                                      rotation=rotation_,
                                      wave_frequency=ω,
                                      wavenumber=k,
                                      coastal_shelf_width=LC,
                                      coastal_lengthscale=λ,
                                      rayleigh_friction=r,
                                      verbose=verbose_,
                                      )
    
    bathymetry_func = lambda X, Y : param.H_D * h_func(X, Y)
    from baroclinicSWEs.sponge_layer import sponge1
    rayleigh_friction_func = lambda x, y : sponge1(x, y, param,
                                   magnitude=r,
                                   x_padding=sponge_padding[0],
                                   y_padding=sponge_padding[1])
    
    baroclinic_sols = baroclinic_flow(bbox,
                                      (P, T, mesh_name),
                                      param,
                                      order,
                                      barotropic_sols,
                                      bathymetry_func,
                                      boundary_conditions=boundary_conditions[1],
                                      scheme=scheme_,
                                      background_flow=background_flow_,
                                      rotation=rotation_,
                                      rayleigh_friction=rayleigh_friction_func,
                                      verbose=verbose_,
                                      )
    
    return barotropic_sols, baroclinic_sols

def barotropic_flow(bbox,
                    mesh,
                    param,
                    order,
                    canyon_width=5e-3,
                    canyon_intrusion=1.5e-2,
                    plot_domain=False,
                    boundary_conditions="SPECIFIED",
                    scheme="Lax-Friedrichs",
                    potential_forcing=False,
                    background_flow='Kelvin',
                    θ=0.5,
                    rotation=True,
                    show_exact_kelvin=False,
                    wave_frequency=1.4,
                    wavenumber=1.4,
                    coastal_shelf_width=2e-3,
                    coastal_lengthscale=3e-3,
                    rayleigh_friction=5e-3,
                    verbose=True,
                ):
    k, ω, r = wavenumber, wave_frequency, rayleigh_friction
    LC, λ = coastal_shelf_width, coastal_lengthscale
    ΔL, w = canyon_intrusion, canyon_width
    
    mesh_, scheme_, potential_forcing_ = mesh, scheme, potential_forcing
    background_flow_, rotation_, verbose_ = background_flow, rotation, verbose
    
    from barotropicSWEs import SWEs
    swes, φ, file_dir = SWEs.startup(
        bbox,
        param,
        h_min,
        h_max,
        order,
        canyon_width=w,
        canyon_intrusion=ΔL,
        boundary_conditions=boundary_conditions,
        scheme=scheme_,
        potential_forcing=potential_forcing_,
        background_flow=background_flow_,
        θ=θ,
        rotation=rotation_,
        wave_frequency=ω,
        wavenumber=k,
        coastal_shelf_width=LC,
        coastal_lengthscale=λ,
        rayleigh_friction=r,
        verbose=verbose_,
        mesh=mesh_,
        )
    
    sols = SWEs.boundary_value_problem(
        swes,
        φ,
        file_dir="Barotropic Data",
        file_name=file_dir,
        animate=False,
    )
    
    return sols

def baroclinic_flow(bbox,
                    mesh,
                    param,
                    order,
                    barotropic_sols,
                    bathymetry_func,
                    boundary_conditions="Solid Wall",
                    scheme="Lax-Friedrichs",
                    background_flow='Kelvin',
                    θ=0.5,
                    rotation=True,
                    rayleigh_friction=np.vectorize(lambda x, y: 5e-2),
                    verbose=True,
                ):    
    import pickle
    from ppp.FEMDG import FEM
    #Barotropic boundary conditions
    assert boundary_conditions.upper() in ["SOLID WALL", "OPEN FLOW",
        "MOVING WALL", "SPECIFIED"], "Invalid choice of boundary conditions."

    assert background_flow.upper() in ['KELVIN', 'CROSSSHORE'], "Invalid \
choice of background flow."

    assert scheme.upper() in [
        "CENTRAL",
        "UPWIND",
        "LAX-FRIEDRICHS",
        "RIEMANN",
        "PENALTY",
        "ALTERNATING",
    ], "Invalid choice of numerical flux."
    
    #Could be that we need a more refined mesh for baroclinic mode
    P, T, mesh_name = mesh
    X, Y = P.T    
    x0, xN, y0, yN = bbox
    scheme_, boundary_conditions_ = scheme, boundary_conditions
    rayleigh_friction_ = rayleigh_friction
    
    from ppp.File_Management import dir_assurer, file_exist
    dir_assurer("Baroclinic FEM Objects")
    fem_dir = f"Baroclinic FEM Objects/{mesh_name}_N={order}.pkl"      


    wall_inds = np.where((Y == y0))[0]
    open_inds2 = np.where((X == x0) | (X == xN))[0]
    open_inds = np.where(Y == yN)[0]

    BC_maps, BC_Types = [wall_inds, wall_inds, open_inds2], ["Wall", "Wall2", "Wall3"]
    BCs = dict(zip(BC_Types, BC_maps))

    if not file_exist(fem_dir):
        with open(fem_dir, "wb") as outp:
            if verbose:
                print("Creating FEM class")

            fem = FEM(P, T, N=order, BCs=BCs)
            pickle.dump(fem, outp, pickle.HIGHEST_PROTOCOL)

    else:
        if verbose:
            print("Loading FEM class")

        with open(fem_dir, "rb") as inp:
            fem = pickle.load(inp)

    X, Y = np.round(fem.x.T.flatten(), 16), np.round(fem.y.T.flatten(), 16)
    
    from baroclinicSWEs.Baroclinic import solver

    swes = solver(
        fem,
        barotropic_sols,
        param,
        bathymetry_func,
        upper_layer_thickness=param.H_pyc,
        upper_layer_density=param.ρ_min,
        lower_layer_density=param.ρ_max,
        flux_scheme=scheme_,
        boundary_conditions=boundary_conditions_,
        rotation=True,
        rayleigh_friction=rayleigh_friction_,
    )
    raise ValueError

def main(bbox, param, order=3, h_min=1e-3, h_max=5e-3,
             wave_frequency=1.4, wavenumber=1.4, coastal_lengthscale=0.03,
             canyon_widths=[1e-10, 1e-3, 5e-3, 1e-2]):
    ω, k, λ = wave_frequency, wavenumber, coastal_lengthscale

    for background_flow_ in ['KELVIN', 'CROSSSHORE']:
        for w_ in canyon_widths:
            for forcing_, r in zip([False], [0]):
                for scheme_ in ["Lax-Friedrichs"]:
                    startup(
                        bbox,
                        param,
                        h_min,
                        h_max,
                        order,
                        mesh_name="",
                        canyon_width=w_,
                        plot_domain=False,
                        boundary_conditions=["Specified", "Open Flow"],
                        scheme=scheme_,
                        potential_forcing=forcing_,
                        background_flow=background_flow_,
                        θ=0.5,
                        rotation=True,
                        wave_frequency=ω,
                        wavenumber=k,
                        coastal_lengthscale=λ,
                        rayleigh_friction=r,
                    )

if __name__ == "__main__":
    import configure
    param, args = configure.main()

    h_min, h_max = args.hmin, args.hmax
    order, domain_width = args.order, args.domain
    λ = args.coastal_lengthscale

    k = param.k * param.L_R  # non-dimensional alongshore wavenumber
    ω = param.ω / param.f  # non-dimensional forcing frequency
    w_vals = np.linspace(1e-3, 1e-2, 19) #[2::4]
    w_vals = np.insert(w_vals, 0, 0)

    bbox = (-domain_width/2, domain_width/2, 0, domain_width)
    main(bbox, param, order, h_min, h_max,
             coastal_lengthscale=λ, canyon_widths=w_vals[::-1],
             wave_frequency=ω, wavenumber=k)