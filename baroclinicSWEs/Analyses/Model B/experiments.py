#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Thu May  5 12:03:12 2022

"""
import numpy as np

def barotropic_farfield(param,
                        h_min=5e-4,
                        h_max=1e-2,
                        order=1,
                        scheme='Central',
                        mesh_name="",
                        canyon_width=20,  # in kilometres
                        alpha=.35, # ratio of canyon on shelf
                        beta=.5, # ratio of canyon along slope
                        animate=True,
                        save_all=True,
                        ):

    import time
    from baroclinicSWEs.Analyses import solutions
    from baroclinicSWEs.Baroclinic import plot_solutions
    param.alpha, param.beta, param.canyon_width = alpha, beta, canyon_width
    print(f"Canyon parameters: alpha={alpha:.2f}, beta={beta:.2f}, \
width={canyon_width:.1f}km")
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

    ### Topography ###
    from barotropicSWEs.Configuration import topography_sdg_revision as topography
    slope_temp, canyon_temp = topography.coastal_topography(
        param
    )
    slope_topography = lambda y : slope_temp(y * param.L_R) / param.H_D
    canyon_topography = lambda x, y : \
        canyon_temp(x * param.L_R, y * param.L_R) / param.H_D
    
        
    h_func_dim = lambda x, y: param.H_D * canyon_topography(x, y)
    sizes = [100, 200, 300, 400]
    orders = [1, 2]#, 3]
    from ppp.File_Management import dir_assurer
    dir_assurer("Experiments/BarotropicDomain")
    
    bbox_check = np.array([-15, 15, 1e-3, 100])
    from ppp.Plots import plot_setup, save_plot
    for j, order in enumerate(orders):
        print(f'\torder={order}')
        fig_u, ax_u = plot_setup("$\\mathcal{C}$ (km)", "Along-shore velocity (cm/s)",
                                 scale=.65)
        fig_v, ax_v = plot_setup("$\\mathcal{C}$ (km)", "Cross-shore velocity (cm/s)",
                                 scale=.65)
        for i, size in enumerate(sizes):
            print(f'Domain width: {size}')
            bbox_computational = (-size/2, size/2, 0, size)
            param.bboxes = [bbox_computational, bbox_check]
            x0, xN, y0, yN = bbox_computational
            Lx, Ly = xN - x0, yN - y0
            file_name = f"Domain={Lx:.0f}kmx{Ly:.0f}km_OceanDepth={param.H_D:.0f}m_\
ShelfDepth={param.H_C:.0f}m_\
ShelfWidth={param.L_C*1e-3:.0f}km_SlopeWidth={param.L_S*1e-3:.0f}km"
        
            if alpha < 1e-2 or beta < 1e-2 or canyon_width < 1:
                file_name += "_Slope"
            else:
                file_name += f"_CanyonWidth={canyon_width:.1f}km\
_alpha={alpha:.2f}_beta={beta:.2f}"
    
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
            
            if save_all:
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
                    order,
                    h_min,
                    h_max,
                    slope_topography,
                    canyon_topography,
                    "Barotropic",
                    file_name,
                    plot_domain=False,
                    boundary_conditions='SOLID WALL',
                    scheme=scheme,
                    save_all=save_all,
                    verbose=True,
                    )
                
            # times[i, j] = time.perf_counter() - start
    
            bbox_temp = np.array(bbox_check) * 1e3/param.L_R
            x0, xN, y0, yN = bbox_temp
            d1 = (x0, np.linspace(y0, yN, 1001)[:-1])
            d2 = (np.linspace(x0, xN, 301), yN)
            d3 = (xN, np.linspace(yN, y0, 1001)[1:])
            
            u = np.concatenate([
                barotropic_solutions.u(d1[0], d1[1], 0),
                barotropic_solutions.u(d2[0], d2[1], 0),
                barotropic_solutions.u(d3[0], d3[1], 0)
                ]) * param.c * 1e2
            
            v = np.concatenate([
                barotropic_solutions.v(d1[0], d1[1], 0),
                barotropic_solutions.v(d2[0], d2[1], 0),
                barotropic_solutions.v(d3[0], d3[1], 0)
                ]) * param.c * 1e2
            
            import matplotlib.pyplot as pt
            d = np.linspace(0, 230, 2301)
            
            ax_u.plot(d, u.real, label=f'$\\Omega_{{{i+1}}}$')
            ax_v.plot(d, v.real, label=f'$\\Omega_{{{i+1}}}$')

        save_plot(fig_u, ax_u, file_name=f"CanyonWidth={canyon_width:.1f}km\
_alpha={alpha:.2f}_beta={beta:.2f}_order={order}_Alongshore",
    folder_name="Experiments/BarotropicDomain", my_loc=1)
        save_plot(fig_v, ax_v, file_name=f"CanyonWidth={canyon_width:.1f}km\
_alpha={alpha:.2f}_beta={beta:.2f}_order={order}_Crossshore",
    folder_name="Experiments/BarotropicDomain", my_loc=1)

    
    save = True
    if save:
        bbox_temp = np.array(bbox_computational) * 1e3 / param.L_R
        Nx, Ny = 2001, 1001
        x0, xN, y0, yN = bbox_temp
        xg, yg = np.linspace(x0, xN, Nx+1), np.linspace(y0, yN, Ny+1)
        Xg, Yg = np.meshgrid(xg, yg)
        #should be most accurate solution
        u = barotropic_solutions.u(Xg, Yg, 0)
        v = barotropic_solutions.v(Xg, Yg, 0)
        p = barotropic_solutions.p(Xg, Yg, 0)
    
        import matplotlib.pyplot as pt
        for phase in [.25]:
            solution_figure = plot_solutions(
                np.array([param.c * u * np.exp(2j * np.pi * phase),
                          param.c * v * np.exp(2j * np.pi * phase),
                          1e-3 * param.ρ_ref * (param.c ** 2) * p * \
                              np.exp(2j * np.pi * phase)]),
                              (param.L_R * 1e-3 * Xg, param.L_R * 1e-3 * Yg),
                              wave_frequency=1.4,
                              bbox=bbox_computational,
                              file_name=f"CanyonWidth={canyon_width:.1f}km\
_alpha={alpha:.2f}_beta={beta:.2f}_order={order}_InnerDomain={domain_size}_phase={phase:.3f}",
                              folder_dir="Experiments/BarotropicDomain",
                              mode=0, animate=False, x_pos=.78, y_pos=.04
                              )
                
            from matplotlib import patches
            
            for i, L in enumerate(sizes):
                x1, x2, y1, y2 = -L/2, L/2, 0, L
                rect = patches.Rectangle((x1, y1),
                                          x2-x1, y2-y1,
                                          linewidth=3,
                                          edgecolor='black', facecolor='none')
                solution_figure.ax.add_patch(rect)
                solution_figure.ax.text(x2, y2, f'$\\Omega_{{{i+1}}}$',
                                        ha='right', va='top',
                                        bbox=dict(boxstyle='square,pad=5',
                                                  fc='none', ec='none'),
                                        size=25)
                
            x1, x2, y1, y2 = bbox_check
            rect = patches.Rectangle((x1, y1),
                                      x2-x1, y2-y1,
                                      linewidth=3,
                                      edgecolor='fuchsia', facecolor='none')
            solution_figure.ax.text(0, y2/2, '$\\mathcal{C}$',
                                    size=25, color='fuchsia',
                                    ha='center', va='center')
            solution_figure.ax.add_patch(rect)
            save_plot(solution_figure.fig, solution_figure.ax,
                      file_name=f"CanyonWidth={canyon_width:.1f}km\
_alpha={alpha:.2f}_beta={beta:.2f}_order={order}_Schematic",
                              folder_name="Experiments/BarotropicDomain",
                      )
            
                    
        

def sponge_layer(param, order=3):
    bbox_barotropic = (-100, 100, 0, 200)
    bbox_baroclinic = (-750, 750, -650, 850)
    pass

def domain_size(param, order=3):
    pass

if __name__ == '__main__':
    from barotropicSWEs.Configuration import configure     
    param = configure.main()
    # param.order = 2
    # param.bbox_dimensional = (-100, 100, 0, 200)
    # bbox_barotropic = (-100, 100, 0, 200)
    # bbox_baroclinic = (-750, 750, -650, 850)
    
    barotropic_farfield(param, order=1)