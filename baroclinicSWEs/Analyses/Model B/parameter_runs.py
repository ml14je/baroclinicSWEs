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
    canyon_width=15,  # in kilometres
    alpha=.35, # ND
    beta=.5, #ND
    plot_domain=False,
    boundary_conditions=["SPECIFIED", 'SOLID WALL'],
    scheme="Lax-Friedrichs",
    potential_forcing=False,
    θ=0.5,
    rotation=True,
    sponge_padding=(350, 350), #in kilometres
    plot_mesh=True,
    save_all=True,
    verbose=True,
    canyon_kelvin=True,
):
    from solutions import barotropic_flow, baroclinic_flow
    # Canyon parameters defined globally
    param.alpha, param.beta, param.canyon_width = alpha, beta, canyon_width

    if hasattr(order, '__len__'):
        barotropic_order, baroclinic_order = order
    else:
        barotropic_order = order
        baroclinic_order = order

    #Barotropic boundary conditions
    assert boundary_conditions[0].upper() in ["SOLID WALL", "OPEN FLOW",
        "MOVING WALL", "SPECIFIED"], "Invalid choice of boundary conditions."

    #Baroclinic boundary conditions
    assert boundary_conditions[1].upper() in ["SOLID WALL", "OPEN FLOW",
        "MOVING WALL", "SPECIFIED"], "Invalid choice of boundary conditions."

    assert scheme.upper() in [
        "CENTRAL",
        "UPWIND",
        "RIEMANN",
        "LAX-FRIEDRICHS",
        "PENALTY",
        "ALTERNATING",
    ], "Invalid choice of numerical flux."

    # Non-Dimensional Shelf Parameters
    L0, λ = param.L_C / param.L_R, (param.L_C + param.L_S) / param.L_R

    # Non-Dimensional Canyon Parameters
    W = canyon_width * 1e3 / param.L_R
    
    # from ppp.File_Management import dir_assurer
    # folder_name = "Parameter Sweeps"
    # dir_assurer(folder_name)
    x0, xN, y0, yN = param.bboxes[1]
    Lx, Ly = xN - x0, yN - y0
    file_name = f"Domain={Lx:.0f}kmx{Ly:.0f}km_OceanDepth={param.H_D:.0f}m_\
ShelfDepth={param.H_C:.0f}m_\
ShelfWidth={param.L_C*1e-3:.0f}km_SlopeWidth={param.L_S*1e-3:.0f}km"

    ### Topography ###
    from barotropicSWEs.Configuration import topography_sdg_revision as topography
    slope_temp, canyon_temp = topography.coastal_topography(
        param
    )
    slope_topography = lambda y : slope_temp(y * param.L_R) / param.H_D
    canyon_topography = lambda x, y : \
        canyon_temp(x * param.L_R, y * param.L_R) / param.H_D

    print(alpha, beta, canyon_width)
    if alpha < 1e-2 or beta < 1e-2 or canyon_width < 1:
        file_name += '_Slope'
    else:
        file_name += f"_CanyonWidth={canyon_width:.1f}km\
_Alpha={alpha:.2f}_Beta={beta:.2f}"
        
    h_func_dim = lambda x, y: param.H_D * canyon_topography(x, y)

    
    ### Barotropic Kelvin wave (with or without canyon) ###
    if canyon_kelvin:
        h_min = (λ - L0) / 10 if (W < 1e-4 or \
            beta < 1e-2 or \
                alpha < 1e-2) \
                    else min(W / 4, (λ - L0) / 20)

        from barotropicSWEs.MeshGeneration import make_canyon_meshes
        barotropic_mesh = make_canyon_meshes.mesh_generator(
            tuple([xx * 1e3/param.L_R for xx in param.bboxes[0]]),
            param,
            h_min,
            h_max / 3,
            canyon_topography,
            mesh_dir="Barotropic/Meshes",
            plot_mesh=False,
            plot_sdf=False,
            plot_edgefunc=False,
            max_iter=1000,
            mesh_gradation=0.35,
            slope_parameter=28,
            verbose=True,
            save_mesh=save_all,
            model='B'
        )
    
        if plot_mesh:
            P, T, mesh_name = barotropic_mesh
            from barotropicSWEs.MeshGeneration import uniform_box_mesh
            uniform_box_mesh.mesh_plot(
                P,
                T,
                file_name,
                folder="Barotropic/Meshes/Figures",
                h_func=h_func_dim,
                save=True,
                L_R=param.L_R * 1e-3,
                zoom=False,
            )
    
        barotropic_solutions = \
            barotropic_flow(
                barotropic_mesh,
                param,
                barotropic_order,
                h_min,
                h_max,
                slope_topography,
                canyon_topography,
                "Barotropic",
                file_name,
                plot_domain=plot_domain,
                boundary_conditions=boundary_conditions[0],
                scheme=scheme,
                save_all=save_all,
                verbose=verbose,
                )

    else:
        from barotropicSWEs.SWEs import kelvin_solver
    
        u_kelv, v_kelv, p_kelv, ω, k = kelvin_solver(
            slope_topography,
            param,
            coastal_lengthscale=(param.L_C + param.L_S) / param.L_R,
            tidal_amplitude=1,
            forcing_frequency=param.ω / param.f,
            foldername=f"Barotropic/Kelvin Flow/{'_'.join(file_name.split('_')[1:-1])}",
            filename=f"KelvinMode_Domain={Lx}kmx{Ly}km_ω={param.ω:.1e}",
        )
        
        file_name += "_SlopeKelvin"
        
        barotropic_solutions = [u_kelv, v_kelv, p_kelv]
        
        
    ### Baroclinic Response ###
    from baroclinicSWEs.Configuration import sponge_layer
    from baroclinicSWEs.MeshGeneration import make_baroclinic_canyon_meshes

    ## Sponge Layer ##
    sponge_padding = np.array(sponge_padding) * 1e3 / param.L_R
    sponge_func = lambda x, y : sponge_layer.sponge2(
        x, y, param,
        magnitude=1,
        x_padding=sponge_padding[0],
        y_padding=sponge_padding[1],
        xc=0, yc=(param.L_C + param.L_S/2)/param.L_R
        )

    h_min = (λ - L0) / 10 if (W < 1e-4 or \
        beta < 1e-2 or \
            alpha < 1e-2) \
                else min(W / 8, (λ - L0) / 20)
    
    ## Baroclinic Mesh Generation ##
    baroclinic_mesh = make_baroclinic_canyon_meshes.mesh_generator(
        tuple([xx * 1e3/param.L_R for xx in param.bboxes[1]]),
        param,
        h_min,
        h_max,
        canyon_func=canyon_topography,
        mesh_dir='Baroclinic/Meshes',
        plot_mesh=False,
        slope_parameter=25,
        mesh_gradation=.35,
        sponge_function=lambda x, y : sponge_func(.6*x, .3*y),
        verbose=verbose,
    )
    
    if plot_mesh:
        P, T, mesh_name = baroclinic_mesh
        from barotropicSWEs.MeshGeneration import uniform_box_mesh
        uniform_box_mesh.mesh_plot(
            P,
            T,
            file_name,
            folder="Baroclinic/Meshes/Figures",
            h_func=h_func_dim,
            save=True,
            L_R=param.L_R * 1e-3,
            zoom=False,
        )

    ## Barotropic Solution ##
    baroclinic_solutions = baroclinic_flow(param,
                                      baroclinic_order,
                                      baroclinic_mesh,
                                      barotropic_solutions,
                                      file_name,
                                      canyon_topography,
                                      boundary_conditions=boundary_conditions[1],
                                      scheme=scheme,
                                      sponge_padding=sponge_padding,
                                      sponge_function=sponge_func,
                                      rayleigh_friction_magnitude=.05,
                                      verbose=verbose
                                      )
    
    return barotropic_solutions, baroclinic_solutions
            
def parameter_run(
    param,
    canyon_width,
    alpha_values,
    beta_values,
    order=3,
    numerical_flux="Central",
    slope_choice="Cos_Squared",
    h_min=5e-4,
    h_max=1e-2,
    plot_slope_topography=False,
    goal='Solutions'
    ):
    
    assert goal.upper() in ['NORMS', 'SOLUTIONS', 'PERTURBATIONS'], \
        "Invalid choice of goal."
    alpha_save, beta_save = np.array([.5, .35, .2]), np.array([.2, .5, .8])
          
    if goal.upper() == 'SOLUTIONS':
        fine_mesh_ = False
        plot_perturbation_ = False
        plot_solution_ = True
        
        alpha_values = np.array([.5, .35, .2])
        beta_values = np.array([.2, .5, .8])
    
    elif goal.upper() == 'PERTURBATIONS':
        fine_mesh_ = False
        plot_perturbation_ = True
        plot_solution_ = False
        
        alpha_values = np.array([.5, .35, .2])
        beta_values = np.array([.2, .5, .8])

    else:
        fine_mesh_ = True
        plot_perturbation_ = False
        plot_solution_ = False

    ### No Canyon Solution ###
    barotropic_sols, baroclinic_swes = startup(
            param,
            h_min,
            h_max,
            order,
            mesh_name="",
            canyon_width=0,
            alpha=0,
            beta=0,
            plot_domain=False,
            boundary_conditions=["Specified", "Solid Wall"],
            scheme=numerical_flux,
            save_all=True
            )
    
    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):
            param.canyon_depth = param.H_D - (param.H_D - param.H_C) * \
                np.cos(beta * np.pi/2)**2
            param.alpha, param.beta = alpha, beta
            save_all = alpha in alpha_save and beta in beta_save

            barotropic_sols, baroclinic_swes = startup(
                                            param,
                                            h_min,
                                            h_max,
                                            order,
                                            mesh_name="",
                                            canyon_width=canyon_width,
                                            alpha=param.alpha,
                                            beta=param.beta,
                                            plot_domain=False,
                                            boundary_conditions=["Specified", "Solid Wall"],
                                            scheme=numerical_flux,
                                            save_all=save_all
                                            )
            
            raise ValueError
            
            del barotropic_sols, baroclinic_swes

if __name__ == "__main__":
    from barotropicSWEs.Configuration import configure     
    param = configure.main()
    # param.order = 2
    param.bbox_dimensional = (-100, 100, 0, 200)
    bbox_barotropic = (-100, 100, 0, 200)
    bbox_baroclinic = (-750, 750, -650, 850)
    param.bboxes = [bbox_barotropic, bbox_baroclinic]
    alpha_values = np.round(np.linspace(.05, 0.95, 46), 3) #10, 91, 901
    beta_values = np.round(np.linspace(.05, 0.95, 46), 3)

    parameter_run(
        param,
        param.canyon_width,
        alpha_values,
        beta_values,
        order=(3, 2),
        numerical_flux="Central", #,"Lax-Friedrichs"
        goal='SOLUTIONS' #Either plot solutions, plot perturbations or plot norms
    )
