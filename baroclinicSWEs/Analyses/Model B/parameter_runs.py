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
    domain_extension=(400, 400), #in kilometres
    damping_width=150, #in kilometres
    plot_mesh=True,
    save_all=True,
    verbose=True,
    canyon_kelvin=True,
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
        
    param.slope_topography = slope_topography
    param.canyon_topography = canyon_topography

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
        
        barotropic_solutions = [u_kelv, v_kelv, p_kelv]
        
        
    ### Baroclinic Response ###
    from baroclinicSWEs.Configuration import sponge_layer
    from baroclinicSWEs.MeshGeneration import make_baroclinic_canyon_meshes

    h_min = (λ - L0) / 10 if (W < 1e-4 or \
        beta < 1e-2 or \
            alpha < 1e-2) \
                else min(W / 8, (λ - L0) / 20)
                
    # x = np.linspace(param.bboxes[1][0], param.bboxes[1][1], 1001)*1e3/param.L_R
    # y = np.linspace(param.bboxes[1][2], param.bboxes[1][3], 1001)*1e3/param.L_R
    # X, Y = np.meshgrid(x, y)
    # R = sponge_layer.lavelle_x(
    #     X, param,
    #     magnitude=1,
    #     x_padding=domain_extension[0]*1e3/param.L_R,
    #     D=damping_width,
    #     xc=0)
    # import matplotlib.pyplot as pt
    # pt.imshow(R,
    #           cmap='seismic',
    #           aspect='equal',
    #           extent=param.bboxes[1])
    # pt.show()
    # raise ValueError

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
        sponge_function=lambda x, y : sponge_layer.lavelle_x(
            .6*x, param,
            magnitude=1,
            x_padding=domain_extension[0]*1e3/param.L_R,
            D=damping_width,
            xc=0),
        model='B',
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
        verbose=verbose
        )
    
    return barotropic_solutions, baroclinic_solutions

def post_process(param, barotropic_slns, baroclinic_slns,
                 NN=1000,
                 show_barotropic_sln=False,
                 show_baroclinic_sln=False,
                 show_fluxes=False,
                 show_dissipation=False
                 ):
    from ppp.Plots import plot_setup, add_colorbar
    import matplotlib.pyplot as pt

    bbox_barotropic, bbox_baroclinic = param.bboxes
    x0 = np.linspace(bbox_barotropic[0], bbox_barotropic[1], NN+1) * 1e3/param.L_R
    y0 = np.linspace(bbox_barotropic[2], bbox_barotropic[3], NN+1) * 1e3/param.L_R
    dx0, dy0 = param.L_R * (x0[-1] - x0[0])/NN, param.L_R * (y0[-1] - y0[0])/NN
    Xg0, Yg0 = np.meshgrid(x0, y0)
    
    # x1 = np.linspace(bbox_baroclinic[0], bbox_baroclinic[1], NN+1) * 1e3/param.L_R
    # y1 = np.linspace(bbox_baroclinic[2], bbox_baroclinic[3], NN+1) * 1e3/param.L_R
    # dx1, dy1 = param.L_R * (x1[-1] - x1[0])/NN, param.L_R * (y1[-1] - y1[0])/NN
    # Xg1, Yg1 = np.meshgrid(x1, y1)
    
    
    ### Barotropic domain ###
    bathymetry = param.canyon_topography(Xg0, Yg0) * param.H_D
    Z1 = baroclinic_slns.Z1(bathymetry)/np.sqrt(param.g)
    # print(np.max(np.abs((Z1 - (param.H_pyc/bathymetry) * np.sqrt(param.reduced_gravity/param.g)))))
    
    # Baroclinic quantities in BL
    p1 = param.ρ_ref * (param.c**2) * Z1 * baroclinic_slns.p(Xg0, Yg0, 0) # * np.exp(-1j * k * Xg)
    u1 = param.c * Z1 * baroclinic_slns.u(Xg0, Yg0, 0)
    v1 = param.c * Z1 * baroclinic_slns.v(Xg0, Yg0, 0)
    

    # Barotropic quantities in BL
    p0 = param.ρ_ref * (param.c**2) * barotropic_slns.p(Xg0, Yg0, 0)
    u0 = param.c * barotropic_slns.u(Xg0, Yg0, 0)
    v0 = param.c * barotropic_slns.v(Xg0, Yg0, 0)
    
    if show_barotropic_sln:
        for phase in np.arange(5) * 2*np.pi/5:
            fig, ax = plot_setup('Along-shore (km)',
                         'Cross-shore (km)')

            P = 1e-3 * (p0 * np.exp(1j*phase)).real
            c = ax.contourf(P,
                            cmap='seismic',
                            extent=[bbox_barotropic[0], bbox_barotropic[1],
                                    bbox_barotropic[2], bbox_barotropic[3]],
                            alpha=0.5,
                            vmin=-np.max(np.abs(p0)) * 1e-3, vmax=np.max(np.abs(p1)) * 1e-3,
                            levels=np.linspace(-np.max(np.abs(p0)*1e-3),
                                               +np.max(np.abs(p0)*1e-3),
                                               21)
                            )
            cbar = add_colorbar(c, ax=ax)
            cbar.ax.tick_params(labelsize=16)
        
            U, V = (u0 * np.exp(1j*phase)).real, (v0 * np.exp(1j*phase)).real
            X, Y = Xg0 * param.L_R *1e-3, Yg0 * param.L_R *1e-3
            Q = ax.quiver(
                   X[::40, ::40],
                   Y[::40, ::40],
                   U[::40, ::40], V[::40, ::40],
                   width=0.002,
                   scale=1,
                   )
        
            ax.quiverkey(
                Q,
                .75, .03, #x and y position of key
                .05, #unit
                r"$5\,\rm{cm/s}$",
                labelpos="W",
                coordinates="figure",
                fontproperties={"weight": "bold", "size": 18},
            )
            ax.set_aspect('equal')
            fig.tight_layout()
            pt.show()

    if show_baroclinic_sln:
        for phase in np.arange(5) * 2*np.pi/5:
            fig, ax = plot_setup('Along-shore (km)',
                         'Cross-shore (km)')

            P = (p1 * np.exp(1j*phase)).real
            c = ax.contourf(P,
                            cmap='seismic',
                            extent=[bbox_barotropic[0], bbox_barotropic[1],
                                    bbox_barotropic[2], bbox_barotropic[3]],
                            alpha=0.5,
                            vmin=-np.max(np.abs(p1)), vmax=np.max(np.abs(p1)),
                            levels=np.linspace(-np.max(np.abs(p1)),
                                               np.max(np.abs(p1)),
                                               21)
                            )
            cbar = add_colorbar(c, ax=ax)
            cbar.ax.tick_params(labelsize=16)
        
            U, V = (u1 * np.exp(1j*phase)).real, (v1 * np.exp(1j*phase)).real
            X, Y = Xg0 * param.L_R *1e-3, Yg0 * param.L_R *1e-3
            Q = ax.quiver(
                   X[::40, ::40],
                   Y[::40, ::40],
                   U[::40, ::40], V[::40, ::40],
                   width=0.002,
                   scale=1,
                   )
        
            ax.quiverkey(
                Q,
                .75, .03, #x and y position of key
                .05, #unit
                r"$5\,\rm{cm/s}$",
                labelpos="W",
                coordinates="figure",
                fontproperties={"weight": "bold", "size": 18},
            )
            ax.set_aspect('equal')
            fig.tight_layout()
            pt.show()
            
    from barotropicSWEs.Configuration import topography
    h = bathymetry
    h1 = param.H_pyc ; h2 = h - h1
    hx, hy = topography.grad_function(h, dy0, dx0)
    w0 = - (hx * u0 + hy * v0) # barotropic vertical velocity in BL
    D = .5 * (p1 * w0.conjugate()).real #barotropic dissipation
    Jx = .5 * (h * h2/h1) * (p1 * u1.conjugate()).real   # along-shore baroclinic energy flux
    Jy = .5 * (h * h2/h1) * (p1 * v1.conjugate()).real   # cross-shore baroclinic energy flux

    if show_fluxes:
        J = np.sqrt(Jx ** 2 + Jy**2)
        fig, ax = plot_setup('Along-shore (km)',
                              'Cross-shore (km)')
        J_max = np.max(J)
        c = ax.imshow(J,
                      cmap='YlOrBr', aspect='equal',
                      extent=bbox_barotropic,
                      origin='lower',
                      vmin=0, vmax=J_max
                      )
        cbar = fig.colorbar(c, ax=ax)
        cbar.ax.set_ylabel('Energy Flux ($\\rm{W/m}$)', rotation=270,
                            fontsize=16, labelpad=20)
        
        Q = ax.quiver(
               X[::40, ::40],
               Y[::40, ::40],
               1e-3*Jx[::40, ::40], 1e-3*Jy[::40, ::40],
               width=0.002,
               scale=1,
               )
        ax.quiverkey(
            Q,
            .85, .03,
            .4,
            r"$400\,\rm{W/m}$",
            labelpos="W",
            coordinates="figure",
            fontproperties={"weight": "bold", "size": 18},
        )
        ax.set_aspect('equal')
        fig.tight_layout()
        pt.show()


    if show_dissipation:
        fig, ax = plot_setup('Along-shore (km)',
                              'Cross-shore (km)')
        D_max = 1e3 * np.max(D)
        c = ax.imshow(D*1e3,
                      cmap='seismic', aspect='equal',
                      extent=bbox_barotropic,
                      origin='lower',
                      vmin=-D_max, vmax=D_max
                      )
        cbar = fig.colorbar(c, ax=ax)
        cbar.ax.set_ylabel('Energy Dissipation ($\\rm{mW/m^2}$)', rotation=270,
                            fontsize=16, labelpad=20)
        pt.show()

    D_total = dx0 * dy0 * np.sum(
        ((D[1:] + D[:-1])[:, 1:] + (D[1:] + D[:-1])[:, :-1])/4)
    Jx_R = .5 * dy0 * np.sum(Jx[1:, 0] + Jx[:-1, 0])
    Jx_L = .5 * dy0 * np.sum(Jx[1:, -1] + Jx[:-1, -1])
    Jy_D = .5 * dx0 * np.sum(Jy[-1, 1:] + Jy[-1, :-1])
    Jy_C = .5 * dx0 * np.sum(Jy[0, 1:] + Jy[0, :-1])
    print(f"\tOffshore: {Jy_D:.1f} W\n\tOnshore: {Jy_C:.1f} W\
\n\tRightward: {Jx_R:.1f} W\n\tLeftward: {Jx_L:.1f} W")

    Jx_total = Jx_L - Jx_R
    Jy_total = Jy_D - Jy_C
    print(f"Jx = {Jx_total:.1f} W/m, Jy = {Jy_total:.1f} W", flush=True)
    J_total = Jx_total + Jy_total
    print(f"Total Energy Flux in Box: {J_total:.2f} W", flush=True)
    print(f"Total Energy Dissipation in Box: {D_total:.2f} W", flush=True)

    return J_total, D_total

        
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
    ):
    domain_padding, damping_width= 400, 150 # in km

    ### No Canyon Solution ###
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
            boundary_conditions=["Specified", "Solid Wall"],
            scheme=numerical_flux,
            save_all=True,
            domain_extension=(domain_padding,
                              domain_padding), #in kilometres
            damping_width=damping_width
            )
    
    J, D = post_process(param,
                        barotropic_sols,
                        baroclinic_sols)
    
    param.canyon_width = canyon_width
    print(f"Canyon width: {param.canyon_width:.1f} km", flush=True)
    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):
            print(f"\talpha: {alpha:.2f}, beta: {beta:.2f}", flush=True)
            param.canyon_depth = param.H_D - (param.H_D - param.H_C) * \
                np.cos(beta * np.pi/2)**2
            param.canyon_length = (alpha * param.L_C + beta * param.L_S) * 1e-3
            param.alpha, param.beta = alpha, beta
            # save_all = alpha in alpha_save and beta in beta_save

            # Slope solutions
            barotropic_solution, baroclinic_solution = startup(
                param,
                5e-4,
                1e-2,
                order,
                mesh_name="",
                canyon_width=param.canyon_width,  # in kilometres
                alpha=param.alpha, # ND
                beta=param.beta, #ND
                boundary_conditions=["SPECIFIED", 'SOLID WALL'],
                scheme="Central",
                domain_extension=(domain_padding,
                                  domain_padding), #in kilometres
                damping_width=damping_width
            )
            
            J, D = post_process(param,
                                barotropic_solution,
                                baroclinic_solution)
            
            del barotropic_solution, baroclinic_solution

if __name__ == "__main__":
    from barotropicSWEs.Configuration import configure
    param = configure.main()
    param.bbox_dimensional = (-150, 150, 0, 300)
    bbox_barotropic = (-100, 100, 0, 200)
    bbox_baroclinic = (-700, 700, -500, 700)
    param.bboxes = [bbox_barotropic, bbox_baroclinic]
    alpha_values = np.array([.2, .8])
    beta_values = np.round(np.linspace(.05, 0.95, 19), 3)

    parameter_run(
        param,
        param.canyon_width,
        alpha_values,
        beta_values,
        order=param.order,
        numerical_flux="Central",
        )
