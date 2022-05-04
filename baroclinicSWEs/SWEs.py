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
    upper_layer_depth=100,
    canyon_width=5e-3,
    canyon_depth=1.0,
    canyon_length=3e-2,
    canyon_choice='v-shape',
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
    sponge_padding=(1.5e-1, 3e-1),
    verbose=True,
):
    if hasattr(order, '__len__'):
        barotropic_order, baroclinic_order = order
    else:
        barotropic_order = order
        baroclinic_order = order

    param.H_pyc = upper_layer_depth

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
    ΔL, w, h_canyon = canyon_length, canyon_width, canyon_depth
    h_coastal = param.H_C/param.H_D
    canyon_choice_ = canyon_choice

    scheme_, potential_forcing_ = scheme, potential_forcing
    background_flow_, rotation_, verbose_ = background_flow, rotation, verbose

    param.domain_size = (Lx, Ly)
    param.L_C = coastal_shelf_width * param.L_R
    param.L_S = (coastal_lengthscale - coastal_shelf_width) * param.L_R

    from barotropicSWEs.topography import coastal_topography
    h_func = coastal_topography(slope_choice='smooth',
                                canyon_choice=canyon_choice_,
                                shelf_depth=h_coastal,
                                coastal_shelf_width=LC,
                                coastal_lengthscale=λ,
                                canyon_length=ΔL,
                                canyon_width=w,
                                canyon_depth=h_canyon)

    h_min = (λ - LC)/10 if w < 1e-5 else min(w/10, (λ - LC)/10)

    from barotropicSWEs.make_canyon_meshes import mesh_generator
    barotropic_mesh = mesh_generator(
        param.bboxes[0],
        param,
        h_min,
        h_max,
        mesh_dir='Barotropic Meshes',
        canyon_func=h_func,
        canyon_width=w,
        canyon_depth=h_canyon,
        canyon_length=ΔL,
        plot_mesh=True,
        coastal_shelf_width=LC,
        coastal_lengthscale=λ,
        mesh_gradation=.35,
        slope_parameter=28,
        verbose=verbose_,
    )

    barotropic_sols, barotropic_fem, barotropic_dir = \
        barotropic_flow(param.bboxes[0],
                        barotropic_mesh,
                        param,
                        barotropic_order,
                        h_min,
                        h_max,
                        canyon_width=w,
                        canyon_depth=h_canyon,
                        canyon_length=ΔL,
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
    from baroclinicSWEs.make_baroclinic_canyon_meshes import mesh_generator

    sponge_func = lambda x, y : sponge(x, y, param,
                                        magnitude=1,
                                        x_padding=sponge_padding[0],
                                        y_padding=sponge_padding[1])

    h_min = (λ - LC)/10 if w < 1e-5 else min(w/8, (λ - LC)/10)
    baroclinic_mesh =mesh_generator(param.bboxes[1],
        param,
        h_min,
        h_max,
        canyon_func=h_func,
        canyon_width=w,
        canyon_depth=h_canyon,
        mesh_dir='Baroclinic Meshes',
        plot_mesh=True,
        coastal_shelf_width=LC,
        coastal_lengthscale=λ,
        slope_parameter=15,
        mesh_gradation=.35,
        sponge_function=lambda x, y : sponge_func(.6*x, .3*y),
        verbose=verbose_,
    )

    baroclinic_swes = baroclinic_flow(param,
                                      baroclinic_order,
                                      baroclinic_mesh,
                                      barotropic_sols,
                                      barotropic_fem,
                                      barotropic_dir,
                                      bathymetry_func,
                                      boundary_conditions=boundary_conditions[1],
                                      scheme=scheme_,
                                      background_flow=background_flow_,
                                      rotation=rotation_,
                                      wave_frequency=ω,
                                      sponge_padding=sponge_padding,
                                      sponge_function=sponge_func,
                                      rayleigh_friction_magnitude=.05,
                                      verbose=verbose_
                                      )
    
    return barotropic_sols, baroclinic_swes

def barotropic_flow(bbox,
                    mesh,
                    param,
                    order,
                    h_min,
                    h_max,
                    canyon_width=5e-3,
                    canyon_depth=1.0,
                    canyon_length=3e-2,
                    canyon_choice='v-shape',
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
    ΔL, w, h_canyon = canyon_length, canyon_width, canyon_depth
    canyon_choice_ = canyon_choice

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
        canyon_depth=h_canyon,
        canyon_length=ΔL,
        canyon_choice=canyon_choice_,
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
                    wave_frequency=1.4,
                    sponge_padding=(.225, .175),
                    sponge_function=np.vectorize(lambda x, y: 1),
                    rayleigh_friction_magnitude=5e-2,
                    verbose=True
                ):
    import pickle
    from ppp.FEMDG import FEM
    background_flow = background_flow.upper()
    assert background_flow in ['KELVIN', 'CROSSSHORE'], "Invalid \
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
    ω, r, s = wave_frequency, rayleigh_friction_magnitude, sponge_function
    sponge_padding_ = sponge_padding

    from ppp.File_Management import dir_assurer, file_exist
    dir_assurer("Baroclinic FEM Objects")
    fem_dir = f"Baroclinic FEM Objects/{background_flow}_{mesh_name}_N={order}.pkl"

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

    baroclinic_swes = solver(
        param,
        fem,
        barotropic_sols,
        barotropic_fem,
        barotropic_dir,
        bathymetry_func,
        upper_layer_thickness=param.H_pyc,
        upper_layer_density=1028,
        lower_layer_density=1031.084,
        flux_scheme=scheme_,
        boundary_conditions=boundary_conditions_,
        rotation=True,
        rayleigh_friction=r,
        sponge_function=s,
        sponge_padding=sponge_padding_
    )

    baroclinic_swes.boundary_value_problem(wave_frequency=ω)

    return baroclinic_swes

def main(param, order=3, h_min=1e-3, h_max=5e-3,
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
            del barotropic_sols, baroclinic_swes

if __name__ == "__main__":
    import configure
    param, args = configure.main()
    args.domain = .4

    h_min, h_max = .05, .01 #args.hmin, args.hmax
    order, domain_width = args.order, args.domain
    λ = args.coastal_lengthscale
    LC = args.shelf_width
    LS = λ - LC

    k = param.k * param.L_R  # non-dimensional alongshore wavenumber
    ω = param.ω / param.f  # non-dimensional forcing frequency
    canyon_widths_ = np.linspace(1e-3, 1e-2, 19) #[2::4]
    canyon_widths_ = np.insert(canyon_widths_, 0, 0)
    bbox_barotropic = (-domain_width/2, domain_width/2, 0, .175)
    DLy = .4
    bbox_baroclinic = (-domain_width/2, domain_width/2, λ/2 - DLy, λ/2 + DLy)
    param.bboxes = [bbox_barotropic, bbox_baroclinic]
    for numerical_flux_ in ['Lax-Friedrichs', 'Central']:
        main(param, args.order,
             h_min, h_max,
             coastal_lengthscale=args.coastal_lengthscale,
             canyon_length=args.canyon_length,
             canyon_depth=args.canyon_depth,
             coastal_shelf_width=args.shelf_width,
             canyon_widths=canyon_widths_,
             numerical_flux=numerical_flux_,
             wave_frequency=ω, wavenumber=k)
