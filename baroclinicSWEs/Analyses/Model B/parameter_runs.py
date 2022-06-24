#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Sun Aug 22 12:27:53 2021

"""
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

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
    domain_extension=(400, 400), #in kilometres
    damping_width=150, #in kilometres
    plot_mesh=False,
    save_all=True,
    verbose=True,
    canyon_kelvin=True,
    animate=False,
):
    from baroclinicSWEs.Analyses import solutions
    # Canyon parameters defined globally
    param.alpha, param.beta, param.canyon_width = alpha, beta, canyon_width

    if hasattr(order, '__len__'):
        barotropic_order, baroclinic_order = order
    else:
        barotropic_order = order
        baroclinic_order = order

    #Barotropic boundary conditions
    assert boundary_conditions[0].upper() in ["SOLID WALL", "OPEN FLOW",
        "MOVING WALL", "SPECIFIED", "REFLECTING"], "Invalid choice of boundary conditions."

    #Baroclinic boundary conditions
    assert boundary_conditions[1].upper() in ["SOLID WALL", "OPEN FLOW",
        "MOVING WALL", "SPECIFIED", "REFLECTING"], "Invalid choice of boundary conditions."

    assert scheme.upper() in [
        "CENTRAL",
        "UPWIND",
        "RIEMANN",
        "LAX-FRIEDRICHS",
        "PENALTY",
        "ALTERNATING",
    ], "Invalid choice of numericalfrom baroclinicSWEs.Analyses import postprocess flux."

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
        
    param.slope_topography = slope_topography
    param.canyon_topography = canyon_topography

    if alpha < 1e-2 or beta < 1e-2 or canyon_width < 1:
        file_name += '_Slope'
    else:
        file_name += f"_CanyonWidth={canyon_width:.1f}km\
_alpha={alpha:.2f}_beta={beta:.2f}"

    file_name += f"_domainpadding={param.domain_padding:.0f}km_\
dampingwidth={param.damping_width:.0f}km"
        
    h_func_dim = lambda x, y: param.H_D * canyon_topography(x, y)

    
    ### Barotropic Kelvin wave (with or without canyon) ###
    print(canyon_kelvin)
    
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
            verbose=verbose,
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
                save=save_all,
                L_R=param.L_R * 1e-3,
                zoom=False,
            )
    
        barotropic_solutions = \
            solutions.barotropic_flow(
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
        param.k = k / param.L_R
        
        file_name += "_SlopeKelvin"
        
        class barotropic_solutions:
            u = u_kelv
            v = v_kelv
            p = p_kelv

    ### Baroclinic Response ###
    from baroclinicSWEs.Configuration import sponge_layer
    from baroclinicSWEs.MeshGeneration import make_baroclinic_canyon_meshes
    
    factor = 4 if param.canyon_width <= 10 else 8

    h_min = (λ - L0) / 10 if (W < 1e-4 or \
        beta < 1e-2 or \
            alpha < 1e-2) \
                else min(W / factor, (λ - L0) / 10)
    domain_padding = domain_extension[0]
    

    ## Baroclinic Mesh Generation ##
    baroclinic_mesh = make_baroclinic_canyon_meshes.mesh_generator(
        tuple([xx * 1e3/param.L_R for xx in param.bboxes[1]]),
        param,
        h_min,
        h_max,
        canyon_func=canyon_topography,
        mesh_dir='Baroclinic/Meshes',
        plot_mesh=False,
        slope_parameter=30,
        mesh_gradation=.2,
        sponge_function=lambda x, y : sponge_layer.lavelle_x(
            .95*x, param,
            magnitude=1,
            x_padding=domain_padding*1e3/param.L_R, #domain_extension[0]*1e3/param.L_R,
            D=param.damping_width,
            xc=0),
        model='B',
        save_mesh=save_all,
        verbose=verbose,
    )
    P, T, mesh_name = baroclinic_mesh

    if plot_mesh:
        P, T, mesh_name = baroclinic_mesh
        from barotropicSWEs.MeshGeneration import uniform_box_mesh
        uniform_box_mesh.mesh_plot(
            P,
            T,
            file_name,
            folder="Baroclinic/Meshes/Figures",
            h_func=h_func_dim,
            save=save_all,
            L_R=param.L_R * 1e-3,
            zoom=False,
        )

    ## Baroclinic Solution ##
    baroclinic_solutions = solutions.baroclinic_flow(
        param,
        baroclinic_order,
        baroclinic_mesh,
        barotropic_solutions,
        file_name,
        canyon_topography,
        boundary_conditions=boundary_conditions[1],
        scheme=scheme,
        domain_extension=domain_extension,
        damping_width=damping_width,
        rayleigh_friction_magnitude=.5,
        animate=animate,
        verbose=verbose,
        save_all=save_all
        )
    
    return barotropic_solutions, baroclinic_solutions

def parameter_run(
    param,
    canyon_width,
    alpha_values,
    beta_values,
    order=3,
    numerical_flux="Central",
    h_min=5e-4,
    h_max=1e-2,
    plot_slope_topography=False,
    goal='Fluxes',
    canyon_kelvin=True,
    plot_transects=False
    ):
    from ppp.File_Management import dir_assurer, file_exist
    import pickle
    from baroclinicSWEs.Analyses import postprocess

    # domain_padding, damping_width= 400, 150 # in km
    assert goal.upper() in ['FLUXES', 'PLOT'], "Incorrect option. Must choose\
 goal in ['FLUXES', 'PLOT']."

    BC = 'SOLID WALL'
    
    alpha_save = [.2, .5, .8]
    beta_save = [.2, .5, .8]
    if goal.upper() == 'PLOT':
        alpha_values = alpha_save
        beta_values = beta_save
        
    try:
        len(beta_values)
        
    except TypeError:
        beta_values = [beta_values]

    try:
        len(alpha_values)

    except TypeError:
        alpha_values = [alpha_values]
    
    data_name = f"CanyonWidth={canyon_width:.0f}km_Order={order}"
    if not canyon_kelvin:
        data_name += "_unperturbed_barotropic_flow"

    
    folder_name = "Energies"
    dir_assurer(folder_name)
    
    if not file_exist(f"{folder_name}/{data_name}.pkl"):
        energies = {}
        
     
    else:
        with open(f"{folder_name}/{data_name}.pkl", "rb") as inp:
            energies = pickle.load(inp)
            
    ### No Canyon Solution ###
    try:
        energies['slope']
    
    # raise ValueError
    except KeyError:
        print("Slope solution", flush=True)
        barotropic_sols, baroclinic_sols = startup(
                param,
                h_min,
                h_max,
                order,
                mesh_name="",
                canyon_width=0,
                alpha=0,
                beta=0,
                plot_domain=False,
                boundary_conditions=["Specified", BC],
                scheme=numerical_flux,
                save_all=True,
                domain_extension=(param.domain_padding,
                                  param.domain_padding), #in kilometres
                damping_width=param.damping_width,
                animate=False,
                verbose=False
                )
        
        Jx_R, Jx_L, Jy_D, Jy_C, D_total = postprocess.post_process(
            param,
            barotropic_sols,
            baroclinic_sols,
            # show_baroclinic_sln=True,
            # show_barotropic_sln=True,
            # show_dissipation=True,
            # show_fluxes=True
            )
        energies['slope'] = Jx_R, Jx_L, Jy_D, Jy_C, D_total
    
    param.canyon_width = canyon_width
    print(f"\n\nCanyon width: {param.canyon_width:.1f} km", flush=True)
    for j, beta in enumerate(beta_values):
        for i, alpha in enumerate(alpha_values):
            print(f"\talpha: {alpha:.2f}, beta: {beta:.2f}", flush=True)
            param.canyon_depth = param.H_D - (param.H_D - param.H_C) * \
                np.cos(beta * np.pi/2)**2
            param.canyon_length = (alpha * param.L_C + beta * param.L_S) * 1e-3
            param.alpha, param.beta = alpha, beta
            save_all = alpha in alpha_save and beta in beta_save        
            
            if goal.upper() == 'FLUXES':
                # Slope solutions
                try:
                    energies[(param.alpha, param.beta)]
                    
                except KeyError:
                    # print(param.canyon_width, param.alpha, param.beta)
                    barotropic_solution, baroclinic_solution = startup(
                        param,
                        5e-4,
                        1e-2,
                        order,
                        mesh_name="",
                        canyon_width=param.canyon_width,  # in kilometres
                        alpha=param.alpha, # ND
                        beta=param.beta, #ND
                        boundary_conditions=["SPECIFIED", BC],
                        scheme="Central",
                        domain_extension=(param.domain_padding,
                                          param.domain_padding), #in kilometres
                        damping_width=param.damping_width,
                        save_all=save_all,
                        canyon_kelvin=canyon_kelvin,
                        verbose=False
                    )
                    
                    Jx_R, Jx_L, Jy_D, Jy_C, D_total = postprocess.post_process(
                        param,
                        barotropic_solution,
                        baroclinic_solution)
                    
                    energies[(alpha, beta)] = Jx_R, Jx_L, Jy_D, Jy_C, D_total
                
                    del barotropic_solution, baroclinic_solution
                    
                    current_time = time.perf_counter()
                    
                    if current_time - param.start_time > param.save_duration:
                        #Try to re-load data before re-saving
                        if not file_exist(f"{folder_name}/{data_name}.pkl"):
                            energies2 = {}
    
                        else:
                            with open(f"{folder_name}/{data_name}.pkl", "rb") as inp:
                                energies2 = pickle.load(inp)
                        
                        energies = {**energies, **energies2}
                        with open(f"{folder_name}/{data_name}.pkl", "wb") as outp:
                            pickle.dump(energies, outp, pickle.HIGHEST_PROTOCOL)
                            
                        param.start_time = current_time #start loop again
                    
            else:
                barotropic_sols, baroclinic_sols = startup(
                    param,
                    5e-4,
                    1e-2,
                    order,
                    mesh_name="",
                    canyon_width=param.canyon_width,  # in kilometres
                    alpha=param.alpha, # ND
                    beta=param.beta, #ND
                    boundary_conditions=["SPECIFIED", BC],
                    scheme="Central",
                    domain_extension=(param.domain_padding,
                                      param.domain_padding), #in kilometres
                    damping_width=param.damping_width,
                    save_all=False,
                    animate=False
                )
                from baroclinicSWEs.Analyses import postprocess
                postprocess.post_process(
                    param,
                    barotropic_sols,
                    baroclinic_sols,
                    )
        
        if plot_transects:
            try:    
                from ppp.Plots import plot_setup
                import matplotlib.pyplot as pt
                values = [np.array([energies[(alpha, beta)][i] for alpha in \
                                    alpha_values]) for i in range(5)]
                    
                fig, ax = plot_setup(
                    "$\\alpha$", 'Energy Flux (W/m)',
                    title=f"Canyon Width: {param.canyon_width:.0f} km, $\\beta$={param.beta:.2f}",
                    scale=.7)
                labels = ['Rightward', 'Leftward', 'Oceanward', 'Shoreward', 'Total']
                
                alpha_temp = np.insert(alpha_values, 0, 0)
                for i in range(5):
                    values_temp = np.insert(values[i], 0, energies["slope"][i])
                    ax.plot(alpha_temp, values_temp/200e3, 'x-', label=labels[i])
            
                ax.legend(fontsize=16)
                pt.show()
                
            except KeyError:
                continue


if __name__ == "__main__":
    from barotropicSWEs.Configuration import configure
    param = configure.main()
    param.start_time = time.perf_counter()
    param.save_duration = 60 * 60 # saves every time duration (1 hr)
    param.bbox_dimensional = (-150, 150, 0, 300)
    bbox_barotropic = (-100, 100, 0, 200)
    alpha_values = np.round(np.linspace(.01, .1, 10), 3) #change here
    beta_values = np.round(np.linspace(.01, .1, 10), 3) #change here
    # param.beta = 0
    beta_values = beta_values[beta_values >= param.beta]
    # param.order = 3 # 3
    # param.canyon_width = 5
    param.domain_padding = 350
    param.damping_width = 150
    X0 = 150
    L_ext = param.domain_padding + param.damping_width
    bbox_baroclinic = (-X0-L_ext, X0+L_ext, 100 - X0-L_ext, 100 + X0+L_ext)
    param.bboxes = [bbox_barotropic, bbox_baroclinic]
    
    if param.nbr_workers == 1: #Serial processing
        parameter_run(
            param,
            param.canyon_width,
            alpha_values,
            beta_values,
            order=param.order,
            numerical_flux="Central",
            goal='Fluxes',
            plot_transects=False,
            canyon_kelvin=False, #change here
            )
    
    else: #Parallel processing
        from multiprocessing import Pool
        from functools import partial
    
        pool = Pool(processes=param.nbr_workers)
        iter_func = partial(parameter_run,
                            param,
                            param.canyon_width,
                            alpha_values,
                            order=param.order,
                            numerical_flux="Central",
                            goal='Fluxes'
                            )
        pool.map(
            iter_func, 
            beta_values
            )
