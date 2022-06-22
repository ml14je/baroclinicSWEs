#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7
Created on Sun Jan 10 11:43:06 2022

"""
import numpy as np

def mesh_generator(
    bbox,
    param,
    h_min,
    h_max,
    canyon_func,
    mesh_name="",
    mesh_dir="Meshes",
    plot_mesh=True,
    save_mesh=True,
    sponge_function=np.vectorize(lambda x, y: 0),
    slope_parameter=.3,
    mesh_gradation=.2,
    verbose=True,
    model="A"
):
    """
    Generates a 2D mesh of a box defined by the boundary box, bbox. The mesh
    considers the bathymetric gradient, ensuring a certain minimum number of 
    nodes. However, the user is recommended to provide both a minimum and a
    maximum edge size, h_min and h_max respectively.
    param.

    Parameters
    ----------
    bbox : tuple
        Boundary box of domain, given in Rossby radii.
    param : param class
        A class who attributes define the physical configuration.
    h_min : float
        Minimum edge size of discretised mesh.
    h_max : float
        Maximum edge size of discretised mesh.
    mesh_name : str, optional
        Filename of mesh. The default is "".
    canyon_width : float, optional
        Non-dimensional average width of submarine canyon. The default is 1e-3.
    coastal_shelf_width : TYPE, optional
        Non-dimensional width of the coastal shelf in the absence of a submarine
        canyon, given in Rossby radii. The default is 0.02.
    coastal_lengthscale : float, optional
        Non-dimensional width of the coastal shelf and the continental margin
        in the absence of a submarine margin, given in Rossby radii. The default is 0.03.
    mesh_gradation : float, optional
        The tolerance for element edges to differ. The default is 0.2 (20 %).
    slope_parameter : float, option
        Number of nodes along a bathymetric gradient. The default is 6.
    verbose : bool, optional
        Print progress. The default is True.

    Returns
    -------
    The points and the connectivity matrix of the discretised mesh.

    """

    x0, xN, y0, yN = bbox
    sponge_function_ = sponge_function
    Lx, Ly = xN - x0, yN - y0

    param.domain_size = (Lx, Ly)

    if mesh_name == "":
        mesh_name = f"DomainSize={Lx*param.L_R*1e-3:.0f}x{Ly*param.L_R*1e-3:.0f}km_\
h=[{h_min:.2e},{h_max:.2e}]_slp={slope_parameter:.1f}_"
        if model.upper() == 'A':
            if abs(param.canyon_depth - param.H_C) < 1 or \
                param.canyon_width < 5e-1 or \
                    param.canyon_length < 5:
                        mesh_name += "Slope"
                
            else:
                mesh_name += f"CanyonWidth={param.canyon_width:.1f}km_\
CanyonLength={param.canyon_length:.1f}km_\
CanyonDepth={param.canyon_depth:.1f}"
                
        else:
            if param.alpha < 1e-2 or \
                param.canyon_width < 5e-1 or \
                    param.beta < 1e-2:
                        mesh_name += "Slope"
            else:
                mesh_name += f"CanyonWidth={param.canyon_width:.1f}km_\
alpha={param.alpha:.2f}_beta={param.beta:.2f}_\
domainpadding={param.domain_padding:.0f}km_\
dampingwidth={param.damping_width:.0f}km\
"

    from ppp.File_Management import dir_assurer

    dir_assurer(mesh_dir)

    h_func_dim = lambda x, y: param.H_D * canyon_func(x, y)

    from baroclinicSWEs.MeshGeneration import box_mesh
    P, T = box_mesh.main(
        bbox,
        h_min,
        h_max,
        h_func=h_func_dim,
        sponge_function=sponge_function_,
        edgefuncs=["Sloping"],
        folder=mesh_dir,
        file_name=mesh_name,
        plot_mesh=plot_mesh,
        save_mesh=save_mesh,
        verbose=False,
        plot_sdf=False,
        plot_boundary=False,
        max_iter=50,
        plot_edgefunc=False,
        slp=slope_parameter,
        fl=0,
        wl=100,
        mesh_gradation=mesh_gradation
    )

    return P, T, mesh_name

def main(bbox, param, h_min, h_max, coastal_lengthscale=.03,
         coastal_shelf_width=.02, canyon_intrusion=.015,
     canyon_widths=[0, 1e-3, 5e-3, 1e-2], mesh_name="", h_func=None):
    λ, LC = coastal_lengthscale, coastal_shelf_width
    ΔL = canyon_intrusion
    name_ = mesh_name
    for w in canyon_widths:
        h_func2 = lambda x, y: h_func(x, y, canyon_width=w, canyon_intrusion=ΔL,
                              coastal_shelf_width=LC, coastal_lengthscale=λ)
        mesh_generator(
            bbox,
            param,
            h_min,
            h_max,
            canyon_func=h_func2,
            mesh_name=name_,
            canyon_width=w,
            coastal_lengthscale=λ,
            coastal_shelf_width=LC,
            canyon_intrusion=ΔL,
            verbose=False,
        )

if __name__ == "__main__":
    import configure
    param, args = configure.main()
    args.domain = .4
    # args.order=1

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
    main(bbox_baroclinic, 
         param,
         h_min, h_max,
         coastal_lengthscale=λ,
         canyon_intrusion=.03,
         coastal_shelf_width=LC,
         canyon_widths=w_vals[:5],
         h_func=canyon_func1)
