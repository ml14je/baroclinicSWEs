#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Tue Dec  7 17:11:28 2021

"""
import numpy as np

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Project description')

    parser.add_argument(
        '--order',
        type=int,
        default=2,
        help='Order of local polynomial test function in DG-FEM numerics.')
    
    parser.add_argument(
        '--domain',
        type=float,
        default=.1,
        help='Domain lengthscale')

    parser.add_argument(
        '--HD',
        type=float,
        default=4000,
        help='Depth of ocean beyond continental shelf.')

    parser.add_argument(
        '--HC',
        type=float,
        default=200,
        help='Depth of ocean along continental shelf.')
    
    parser.add_argument(
        '--upper_thickness',
        type=float,
        default=100,
        help='Depth of fixed upper layer in two-layer mode.')
    
    parser.add_argument(
        '--min_density',
        type=float,
        default=1025.862,
        help='Minimum density. Density of upper layer in two-layer model.')
    
    parser.add_argument(
        '--max_density',
        type=float,
        default=1048.841035,
        help='Maximum density. Density of lower layer in two-layer model.')

    parser.add_argument(
        '--hmin',
        type=float,
        default=5e-4,
        help='Minimum edge size of mesh.')

    parser.add_argument(
        '--hmax',
        type=float,
        default=5e-2,
        help='Maximum edge size of mesh.')

    parser.add_argument(
        '--coastal_lengthscale',
        type=float,
        default=0.03,
        help='Coastal lengthscale (shelf plus slope) in Rossby radii.')
    
    parser.add_argument(
        '--shelf_width',
        type=float,
        default=0.02,
        help='Shelf width in Rossby radii.')
    
    parser.add_argument(
        '--canyon_width',
        type=float,
        default=5e-3,
        help='Canyon width in Rossby radii.')
    
    parser.add_argument(
        '--canyon_intrusion',
        type=float,
        default=.015,
        help='Canyon instrusion in Rossby radii.')

    parser.add_argument(
        '--wave_frequency',
        type=float,
        default=1.4e-4,
        help='Forcing wave frequency. Default is semi-diurnal.')

    parser.add_argument(
        '--coriolis',
        type=float,
        default=1e-4,
        help='Local Coriolis coefficient. Default is typical of mid-latitude.')

    args = parser.parse_args()



    from ChannelWaves1D.config_param import configure

    param = configure()
    param.bbox = (-args.domain/2, args.domain/2, 0, args.domain)
    param.H_D = args.HD
    param.H_C = args.HC
    param.H_pyc = args.upper_thickness
    param.ρ_max, param.ρ_min = args.max_density, args.min_density
    param.ρ_ref = param.ρ_max
    param.c = np.sqrt(param.g * param.H_D)
    param.f, param.ω = args.coriolis, args.wave_frequency
    param.L_R = param.c/abs(param.f)
    param.Ly = 2 * param.L_R
    
    return param, args