#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Sun Jan  10 11:46:40 2022
"""

import numpy as np


def create_shoreline_geometry(x0, xN, y0, yN, name="poly"):
    import geopandas as gpd
    import pandas as pd

    box = _bbox(x0, y0, xN, yN)

    gpd.GeoDataFrame(
        pd.DataFrame(["p1"], columns=["geom"]),
        crs={"init": "epsg:4326"},
        geometry=[box],
    ).to_file(name)


def _bbox(x0, y0, xN, yN):
    from shapely.geometry import Polygon

    arr = np.array([[x0, y0], [x0, yN], [xN, yN], [xN, y0]])
    return Polygon(arr)


def main(
    bbox,
    h_min,
    h_max,
    edgefuncs="Uniform",
    sponge_function=np.vectorize(lambda x, y: 0),
    folder="",
    file_name="",
    plot_mesh=False,
    save_mesh=True,
    h_func=None,
    verbose=False,
    plot_sdf=True,
    plot_boundary=True,
    max_iter=50,
    canyon_kwargs=None,
    plot_edgefunc=False,
    HD=4000,
    wl=100,
    slp=10,
    fl=0,
    mesh_gradation=.2
):
    from ppp.File_Management import file_exist, dir_assurer

    file_name = "Mesh" if file_name == "" else file_name

    x0, xN, y0, yN = bbox
    LR = np.sqrt(9.81 * HD) * 1e4 #Rossby radius of deformation - horizontal lengthscale
    if not file_exist(f"{folder}/{file_name}.npz"):
        print(f"{folder}/{file_name}.npz")
        from oceanmesh import (
            Shoreline,
            #delete_boundary_faces,
            #delete_faces_connected_to_one_face,
            fix_mesh,
            distance_sizing_function,
            generate_mesh,
            laplacian2,
            make_mesh_boundaries_traversable,
            feature_sizing_function,
            #signed_distance_function,
            create_bbox,
            enforce_mesh_gradation,
            wavelength_sizing_function,
            compute_minimum,
            Region,
            bathymetric_gradient_sizing_function,
            DEM,
        )
        EPSG = 4326  # EPSG code for WGS84 which is what you want to mesh in
        # Specify and extent to read in and a minimum mesh size in the unit of the projection

        extent = Region(extent=bbox, crs=EPSG)
        if type(edgefuncs) == str:
            edgefuncs = [edgefuncs.upper()]

        edgefuncs = [edgefunc.upper() for edgefunc in edgefuncs]
        x0, xN, y0, yN = bbox
        shp_folder = "Box_Shp"
        dir_assurer(f"{folder}/{shp_folder}")
        fname = f"{folder}/{shp_folder}/box_shape_({x0},{y0})x({xN},{yN}).shp"
        eps = h_max
        create_shoreline_geometry(x0, xN, y0 - eps, y0, name=fname)
        extent_ = Region((x0, xN, y0 - eps / 2, yN), crs=EPSG)
        points_fixed = np.array([[x0, y0], [x0, yN], [xN, yN], [xN, y0]])
        if h_func is None: #By default, it chooses the canyon topography
            from topography import canyon_func1

            if canyon_kwargs is not None and type(canyon_kwargs) == dict:
                #h_min = min(canyon_kwargs['w']/20, h_min) #You force to hav at least 10 grid points over the width of canyon
                h_func = lambda x, y: HD * canyon_func1(x, y, **canyon_kwargs)

            else:
                h_func = lambda x, y: HD * canyon_func1(x, y)

        dem = DEM(h_func, bbox=extent.bbox, resolution=h_min)  # creates a DEM object for own function, h_func
        smoothing = False
        print("creating shoreline")
        if len(edgefuncs) == 1 and edgefuncs[0] == "UNIFORM":
            shore = Shoreline(fname, extent_.bbox, h_max,
                              smooth_shoreline=smoothing)

        else:
            shore = Shoreline(fname, extent_.bbox, h_min,
                              smooth_shoreline=smoothing)

        if verbose:
            print("created shoreline")

        domain = create_bbox(extent.bbox)  # signed_distance_function(shore)

        if verbose:
            print("created a signed distance function")

        edge_lengths = []

        if "UNIFORM" in edgefuncs:
            edge_lengths.append(
                distance_sizing_function(shore, max_edge_length=h_max, rate=0)
            )

        if "DISTANCING" in edgefuncs:
            edge_lengths.append(distance_sizing_function(shore,
                                                         max_edge_length=h_max))

        if "FEATURES" in edgefuncs:
            edge_lengths.append(
                feature_sizing_function(shore, domain, max_edge_length=h_max)
            )

        if "WAVELENGTH" in edgefuncs:
            edge_lengths.append(wavelength_sizing_function(dem, wl=wl*LR))

        if "SLOPING" in edgefuncs:
            h_min *= 50
            edge_lengths.append(bathymetric_gradient_sizing_function(
                dem, slope_parameter=slp, filter_quotient=fl,
                min_edge_length=h_min, max_edge_length=h_max))

        if len(edge_lengths) > 1:
            edge_length = compute_minimum(edge_lengths)

        else:
            edge_length = edge_lengths[0]
            
        x,y = dem.create_grid()
        edge_length.values[:] = sponge_function(x, y) * h_max + (1-sponge_function(x, y)) * edge_length.values
        edge_length.build_interpolant()
        
        

        # Enforce gradation - becomes very unnatural otherwise!
        # Becomes distorted if dx != dy in grid
        edge_length = enforce_mesh_gradation(edge_length,
                                              gradation=mesh_gradation)

        if plot_sdf:  # plot signed distance function
            from ppp.Plots import plot_setup, save_plot

            N = 101
            X, Y = np.linspace(bbox[0], bbox[1], N), np.linspace(bbox[2], bbox[3], N)
            X, Y = np.meshgrid(X, Y)
            X, Y = X.flatten("F"), Y.flatten("F")
            P = np.array([X, Y]).T
            dsf = domain.domain(P).reshape((N, N))
            fig, ax = plot_setup("$x$", "$y$")
            c = ax.matshow(
                dsf.T,
                extent=[x0, xN, y0, yN],
                aspect="auto",
                cmap="seismic",
                origin="lower",
                vmin=-1,
                vmax=1,
            )
            fig.colorbar(c, ax=ax)
            save_plot(fig, ax, "Signed_Distance_Function", folder_name=folder)

        if plot_edgefunc:
            from ppp.Plots import plot_setup, save_plot
            import matplotlib.pyplot as pt

            N = 1001
            X, Y = np.linspace(bbox[0], bbox[1], N), np.linspace(bbox[2], bbox[3], N)
            X, Y = np.meshgrid(X, Y)
            X, Y = X.flatten("F"), Y.flatten("F")
            P = np.array([X, Y]).T
            esf = edge_length.eval(P).reshape((N, N))
            fig, ax = plot_setup("$x$", "$y$")
            c = ax.matshow(
                np.log10(esf.T),
                extent=[x0, xN, y0, yN],
                aspect="auto",
                cmap="seismic",
                origin="lower",
                vmin=-4,
                vmax=1,
            )
            fig.colorbar(c, ax=ax)
            # pt.show()
            save_plot(fig, ax, "Edge_Sizing_Function", folder_name=folder)

        points, cells = generate_mesh(
            domain, edge_length, max_iter=max_iter, pfix=points_fixed, lock_boundary=True,
        )
        # mesh_plot(points, cells, '1 Original Mesh', folder, f_points=points_fixed)

        # Makes sure the vertices of each triangle are arranged in a counter-clockwise order
        points, cells, jx = fix_mesh(points, cells)

        # remove degenerate mesh faces and other common problems in the mesh
        points, cells = make_mesh_boundaries_traversable(points, cells)
        #        mesh_plot(points, cells, '2 Degenerate Elements Removed', folder, f_points=points_fixed)

        # Removes faces connected by a single face
        #        points, cells = delete_faces_connected_to_one_face(points, cells)
        #        mesh_plot(points, cells, '3 Deleted Faces Connected to One Face', folder, f_points=points_fixed)

        # remove low quality boundary elements less than 15%
        #        points, cells = delete_boundary_faces(points, cells, min_qual=0.15)
        #        mesh_plot(points, cells, '4 Deleted Boundary Faces', folder, f_points=points_fixed)

        # apply a Laplacian smoother
        if edgefuncs != "UNIFORM":
            points, cells = laplacian2(points, cells)  # Final poost-processed mesh

        if save_mesh:
            from ppp.Numpy_Data import save_arrays
            save_arrays(file_name, [points, cells], folder_name=folder)

    else:
        from ppp.Numpy_Data import load_arrays

        points, cells = load_arrays(file_name, folder_name=folder)

    if file_exist(f"{folder}/{file_name}.png"):
        save_mesh = False
        
    else:
        save_mesh = True

    if plot_mesh:
        for zoom_ in [True, False]:
            mesh_plot(points, cells, file_name, folder=folder, f_points=None,
                      h_func=h_func, save=save_mesh, zoom=zoom_)

    return points, cells


def mesh_plot(points, cells, name, folder="", f_points=None, h_func=None,
              save=True, L_R=2000, zoom=True):
    from ppp.Plots import plot_setup

    fig, ax = plot_setup("Along-shore (km)", "Cross-shore (km)")
    X, Y = L_R*points.T

    if h_func:
        x0, y0, xN, yN = np.min(X), np.min(Y), np.max(X), np.max(Y)
        x, y = np.linspace(x0, xN, 1001), np.linspace(y0, yN, 1001)
        xg, yg = np.meshgrid(x, y)
        c = ax.matshow(
            h_func(xg/L_R, yg/L_R),
            cmap="Blues",
            vmin=0,
            extent=[x0, xN, y0, yN],
            aspect="auto",
            origin="lower",
            alpha=.7
        )
        fig.colorbar(c, ax=ax)
        
    if f_points is not None:
        X, Y = L_R*f_points.T
        ax.plot(X, Y, "go", markersize=2)
    

    ax.plot(X, Y, "gx", markersize=1)
    ax.triplot(X, Y, cells, linewidth=0.2, color="red")

    if zoom:
        ax.set_xlim([-100, 100])
        ax.set_ylim([0, 250])
        name += '_zoom'

    if save:
        from ppp.Plots import save_plot
        save_plot(fig, ax, name, folder_name=folder)
        
    else:
        import matplotlib.pyplot as pt
        pt.show()

if __name__ == "__main__":
    bbox = (-0.025, 0.025, 0, 0.05)
    hmin, hmax = 1e-3, 1e-2
    for edgefuncs_ in [
        ["Sloping", "Features"],
        "Uniform",
        "Distancing",
        "Features",
        "Wavelength",
        "Sloping",
    ]:
        main(
            bbox,
            hmin,
            hmax,
            folder=edgefuncs_,
            file_name=f"{bbox[1]-bbox[0]:.1f}x{bbox[3]-bbox[2]:.1f}_hmin={hmin:.2e}",
            edgefuncs=edgefuncs_,
            max_iter=1000,
            canyon_kwargs={'w': 1e-3},
        )
