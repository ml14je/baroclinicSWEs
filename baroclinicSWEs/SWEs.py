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
    param,
    h_min,
    h_max,
    order,
    mesh_name="",
    upper_layer_depth=200,
    canyon_width=5e-3,
    canyon_intrusion=1.5e-2,
    plot_domain=False,
    boundary_conditions=["SPECIFIED", 'SOLID WALL'],
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
    sponge_padding=(1.25e-1, 1.25e-1),
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

    x0, xN, y0, yN = param.bboxes[0]
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
    barotropic_mesh = mesh_generator(
        param.bboxes[0],
        param,
        h_min,
        h_max,
        mesh_dir='Barotropic Meshes',
        canyon_func=h_func,
        canyon_width=w,
        plot_mesh=False,
        coastal_shelf_width=LC,
        coastal_lengthscale=λ,
        verbose=verbose_,
    )

    barotropic_sols, barotropic_fem, barotropic_dir = \
        barotropic_flow(param.bboxes[0],
                        barotropic_mesh,
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
                        rayleigh_friction_magnitude=r,
                        verbose=verbose_,
                        )
    
    bathymetry_func = lambda X, Y : param.H_D * h_func(X, Y)
    from baroclinicSWEs.sponge_layer import sponge2 as sponge

    baroclinic_mesh = mesh_generator(
        param.bboxes[1],
        param,
        h_min*4,
        h_max*4,
        canyon_func=h_func,
        canyon_width=w,
        mesh_dir='Baroclinic Meshes',
        plot_mesh=False,
        coastal_shelf_width=LC,
        coastal_lengthscale=λ,
        verbose=verbose_,
    )
    
    sponge_func = lambda x, y : sponge(x, y, param,
                                       magnitude=1,
                                       x_padding=sponge_padding[0],
                                       y_padding=sponge_padding[1])
    
    baroclinic_sols = baroclinic_flow(param,
                                      order,
                                      baroclinic_mesh,
                                      barotropic_sols,
                                      barotropic_fem,
                                      barotropic_dir,
                                      bathymetry_func,
                                      boundary_conditions=boundary_conditions[1],
                                      scheme=scheme_,
                                      background_flow=background_flow_,
                                      rotation=rotation_,
                                      sponge_function=sponge_func,
                                      rayleigh_friction_magnitude=.5,
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
                    rayleigh_friction_magnitude=5e-2,
                    verbose=True,
                ):
    k, ω, r = wavenumber, wave_frequency, rayleigh_friction_magnitude
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
    
    return sols, swes.fem, file_dir

def baroclinic_flow(param,
                    order,
                    baroclinic_mesh,
                    barotropic_sols,
                    barotropic_fem,
                    barotropic_dir,
                    bathymetry_func,
                    boundary_conditions="Solid Wall",
                    scheme="Lax-Friedrichs",
                    background_flow='Kelvin',
                    θ=0.5,
                    rotation=True,
                    sponge_function=np.vectorize(lambda x, y: 1),
                    rayleigh_friction_magnitude=5e-2,
                    verbose=True,
                ):    
    import pickle
    from ppp.FEMDG import FEM

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

    P, T, mesh_name = baroclinic_mesh

    scheme_, boundary_conditions_ = scheme, boundary_conditions
    r, s = rayleigh_friction_magnitude, sponge_function
    
    
    from ppp.File_Management import dir_assurer, file_exist
    dir_assurer("Baroclinic FEM Objects")
    fem_dir = f"Baroclinic FEM Objects/{mesh_name}_N={order}.pkl"

    if not file_exist(fem_dir):
        with open(fem_dir, "wb") as outp:
            if verbose:
                print("Creating Baroclinic FEM class")

            fem = FEM(P, T, N=order)
            pickle.dump(fem, outp, pickle.HIGHEST_PROTOCOL)

    else:
        if verbose:
            print("Loading Baroclinic FEM class")

        with open(fem_dir, "rb") as inp:
            fem = pickle.load(inp)

    from baroclinicSWEs.Baroclinic import solver

    swes = solver(
        param,
        fem,
        barotropic_sols,
        barotropic_fem,
        barotropic_dir,
        bathymetry_func,
        upper_layer_thickness=param.H_pyc,
        upper_layer_density=param.ρ_min,
        lower_layer_density=param.ρ_max,
        flux_scheme=scheme_,
        boundary_conditions=boundary_conditions_,
        rotation=True,
        rayleigh_friction=r,
        sponge_function=s
    )
    # u1_init = np.zeros(swes.X.shape[0], dtype=complex)
    # v1_init, p1_init = np.copy(u1_init), np.copy(u1_init)
    # swes.timestep(u1_init, v1_init, p1_init, t_final=20, method='Forward Euler')
    swes.bvp(
             wave_frequency=1.4
             )

def main(param, order=3, h_min=1e-3, h_max=5e-3,
             wave_frequency=1.4, wavenumber=1.4, coastal_lengthscale=0.03,
             canyon_widths=[1e-10, 1e-3, 5e-3, 1e-2]):
    ω, k, λ = wave_frequency, wavenumber, coastal_lengthscale

    for background_flow_ in ['KELVIN', 'CROSSSHORE']:
        for w_ in canyon_widths:
            for forcing_, r in zip([False], [0]):
                for scheme_ in ["Lax-Friedrichs"]:
                    startup(
                        param,
                        h_min,
                        h_max,
                        order,
                        mesh_name="",
                        canyon_width=w_,
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
                        rayleigh_friction=r,
                    )

if __name__ == "__main__":
    import configure
    param, args = configure.main()
    args.order = 3
    args.domain = .45

    h_min, h_max = args.hmin, args.hmax
    order, domain_width = args.order, args.domain
    λ = args.coastal_lengthscale

    k = param.k * param.L_R  # non-dimensional alongshore wavenumber
    ω = param.ω / param.f  # non-dimensional forcing frequency
    canyon_widths_ = np.linspace(1e-3, 1e-2, 19) #[2::4]
    canyon_widths_ = np.insert(canyon_widths_, 0, 0)

    bbox_barotropic = (-domain_width/2, domain_width/2, 0, .05)
    bbox_baroclinic = (-domain_width/2, domain_width/2, -.175, .225)
    param.bboxes = [bbox_barotropic, bbox_baroclinic]
    main(param, order, h_min, h_max,
             coastal_lengthscale=λ, canyon_widths=[0],
             wave_frequency=ω, wavenumber=k)