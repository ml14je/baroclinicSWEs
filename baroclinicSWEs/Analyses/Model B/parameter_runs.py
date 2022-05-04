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
    sponge_padding=(300, 600), #in kilometres
    plot_mesh=True,
    save_all=True,
    verbose=True,
    canyon_kelvin=True,
):
    
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
                param.bboxes[0], #barotropic domain in km's
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

def barotropic_flow(bbox,
                    mesh,
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
    
    if file_exist(sln_dir):
        with open(sln_dir, "rb") as inp:
            global_solutions = dill.load(inp)

    else:
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

        class global_solutions:
            u = lambda x, y, t : \
                funcs[0](x, y) * np.exp(-1j * ω * t) * \
                    (x > x0) * (x < xN) * (y > y0) * (y < yN) + \
                kelvin_solutions[0](x, y, t) * \
                    (1 - (x > x0) * (x < xN) * (y > y0) * (y < yN))
                    
            v = lambda x, y, t : \
                funcs[1](x, y) * np.exp(-1j * ω * t) * \
                    (x > x0) * (x < xN) * (y > y0) * (y < yN) + \
                kelvin_solutions[1](x, y, t) * \
                    (1 - (x > x0) * (x < xN) * (y > y0) * (y < yN))
                    
            p = lambda x, y, t : \
                funcs[2](x, y) * np.exp(-1j * ω * t) * \
                    (x > x0) * (x < xN) * (y > y0) * (y < yN) + \
                kelvin_solutions[2](x, y, t) * \
                    (1 - (x > x0) * (x < xN) * (y > y0) * (y < yN))
        
        with open(sln_dir, "wb") as outp:
            if verbose:
                print("Saving Barotropic solution function")
                
            dill.dump(global_solutions, outp)
            
    # LR = 1e3 / param.L_R
    # xg, yg = np.linspace(-400 * LR, 400 * LR, 1001), np.linspace(0, 500 * LR, 1001)
    # Xg, Yg = np.meshgrid(xg, yg)
           
    # u = global_solutions.u(Xg, Yg, 0)
    # v = global_solutions.v(Xg, Yg, 0)
    # p = global_solutions.p(Xg, Yg, 0) * param.c ** 2 * param.ρ_ref * 1e-3

    # assert not np.all(u - v == 0), \
    #     "Error in function uniqueness!"

    # import matplotlib.pyplot as pt
    # from ppp.Plots import plot_setup, add_colorbar
    
    # for phase in np.exp(-1j * np.linspace(0, 2*np.pi, 21)):
    #     fig, ax = plot_setup(
    #         x_label="Along-shore (km)", y_label="Cross-shore (km)"
    #     )
    #     max_p = np.max(np.abs(p))
    #     c = ax.contourf(
    #         (p * phase).real,
    #         cmap="seismic",
    #         alpha=0.5,
    #         extent=(-400, 400, 0, 500),
    #         origin="lower",
    #         levels=np.linspace(-max_p, max_p, 21),
    #     )
        
    #     cbar = add_colorbar(c)
    #     cbar.ax.tick_params(labelsize=16)
    #     ax.set_aspect('equal')
    #     pt.show()
    # raise ValueError

        
    return global_solutions

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
                    sponge_padding=(.225, .175),
                    sponge_function=np.vectorize(lambda x, y: 1),
                    rayleigh_friction_magnitude=5e-2,
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
    file_name = barotropic_name + \
            f"_rho1={param.ρ_min:.0f}_rho2={param.ρ_max:.0f}_\
rho={param.ρ_ref:.0f}_h1={param.H_pyc:.0f}m"
    folder_name = "Baroclinic"
    
    sln_dir = f"{folder_name}/Functions/{file_name}_N={order}.pkl"
    dir_assurer(f"{folder_name}/Functions")
    
    if file_exist(sln_dir):
        with open(sln_dir, "rb") as inp:
            baroclinic_solutions = dill.load(inp)
            
    else:
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
        ω, r, s = param.ω/param.f, rayleigh_friction_magnitude, sponge_function
        sponge_padding_ = sponge_padding
    
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
            bathymetry_func,
            flux_scheme=scheme_,
            boundary_conditions=boundary_conditions_,
            rotation=True,
            rayleigh_friction=r,
            sponge_function=s,
            sponge_padding=sponge_padding_
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
        
        class baroclinic_solutions:
            u = lambda x, y, t : \
            solutions[0](x, y) * np.exp(-1j * ω * t)
            v = lambda x, y, t : \
            solutions[1](x, y) * np.exp(-1j * ω * t)
            p = lambda x, y, t : \
            solutions[2](x, y) * np.exp(-1j * ω * t)
            swes = baroclinic_swes
            
        with open(sln_dir, "wb") as outp:
            if verbose:
                print("Saving Barotropic solution function")
                
            dill.dump(baroclinic_solutions, outp)
            
        


    
    
    if animate:
        Nx, Ny = 1000, 1000
        
        bbox_temp = np.array(param.bboxes[1]) * 1e3 /param.L_R
        # bbox_temp[0] = - bbox_temp[3]/2
        # bbox_temp[1] = + bbox_temp[3]/2
        bbox_temp[2] = 0 # y0 = 0 (coastline)
        x0, xN, y0, yN = bbox_temp
        
        xg, yg = np.linspace(x0, xN, Nx+1), np.linspace(y0, yN, Ny+1)
        Xg, Yg = np.meshgrid(xg, yg)

        bathymetry = baroclinic_solutions.swes.h_func(Xg, Yg)
        Z1 = baroclinic_solutions.swes.Z1_func(bathymetry)
        # print(self.barotropic_sols)
        u = baroclinic_solutions.u(Xg, Yg, 0)
        v = baroclinic_solutions.v(Xg, Yg, 0)
        p = baroclinic_solutions.p(Xg, Yg, 0)
        LR = 1e-3 * param.L_R


        from baroclinicSWEs.Baroclinic import animate_solutions
        animate_solutions(np.array([param.c * u * Z1,
                                    param.c * v * Z1,
                                    param.ρ_ref * \
                                        (param.c**2) * p * Z1]),
                          (LR * Xg, LR * Yg),
                          wave_frequency=1.4,
                          bbox=[LR * L for L in bbox_temp],
                          padding=(0, 0),
                          file_name=barotropic_name + \
                              f"_rho1={param.ρ_min:.0f}_rho2={param.ρ_max:.0f}_\
rho={param.ρ_ref:.0f}_h1={param.H_pyc:.0f}m",
                          folder_dir="Baroclinic/Baroclinic Animation",
                          mode=1
                          )

    return baroclinic_solutions
            
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
            param.canyon_depth = param.H_D - (param.H_D - param.H_C) * np.cos(beta * np.pi/2)**2
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
                                            canyon_depth=param.canyon_depth,
                                            plot_domain=False,
                                            boundary_conditions=["Specified", "Solid Wall"],
                                            scheme=numerical_flux,
                                            save_all=save_all
                                            )
            
            del barotropic_sols, baroclinic_swes

if __name__ == "__main__":
    from barotropicSWEs.Configuration import configure     
    param = configure.main()
    param.order = 3
    param.bbox_dimensional = (-100, 100, 0, 200)
    bbox_barotropic = (-100, 100, 0, 200)
    bbox_baroclinic = (-400, 400, -300, 500)
    param.bboxes = [bbox_barotropic, bbox_baroclinic]
    alpha_values = np.round(np.linspace(.05, 0.95, 46), 3) #10, 91, 901
    beta_values = np.round(np.linspace(.05, 0.95, 46), 3)

    parameter_run(
        param,
        param.canyon_width,
        alpha_values,
        beta_values,
        order=(1, 1),
        numerical_flux="Central", #,"Lax-Friedrichs"
        goal='SOLUTIONS' #Either plot solutions, plot perturbations or plot norms
    )
