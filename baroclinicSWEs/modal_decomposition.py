#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Fri Dec  3 14:12:23 2021

"""
import numpy as np
from scipy.sparse import csr_matrix as sp

class MultiLayerModes(object):
    def __init__(self, bathymetry, layer_thicknesses, layer_densities,
                 max_density=1031.084, max_depth=None, g=9.81,
                 normalisation="Constant"):

        self.max_depth = np.max(bathymetry) if max_depth is None \
            else max_depth
        self.min_depth = np.min(bathymetry)
        self.bathymetry = bathymetry
        self.g = g #Gravitational acceleration
        
        self.layer_thicknesses = layer_thicknesses if \
            hasattr(layer_thicknesses, "__len__") else \
            np.array([layer_thicknesses])
            
        self.layer_densities = layer_densities if \
            hasattr(layer_densities, "__len__") else \
            np.array([layer_densities])

        self.fixed_depth = np.sum(self.layer_thicknesses)
        self.bottom_layer = self.bathymetry - self.fixed_depth
        self.ρ_vals = np.append(self.layer_densities, max_density)
        self.min_density, self.max_density = self.ρ_vals[[0, -1]]
        self.reduced_gravity = self.g * (self.max_density - self.min_density)/\
                                self.max_density
        self.normalisation = normalisation.upper()
        
        assert np.all(np.diff(self.ρ_vals) >= 0), "Density stratification at \
 rest is not stable"

        assert len(self.layer_thicknesses) == len(self.layer_densities), \
            "Number of layer thinknesses must match the number of layer\
 densities"
            
        assert self.fixed_depth < self.min_depth, "Fixed upper layers must be\
 less than the minimum fluid depth"
 
        assert self.normalisation in ['CONSTANT', 'ANTI-SYMMETRIC'], \
            "Choice of normalisation must be either 'constant' or \
'anti-symmetric'"
            
        self.M = len(self.layer_densities) + 1 #Number of layers in total
        
        if self.M == 2:
            self.two_layer_approximations()
        
    def set_matrices(self):
        self.L = sp(np.tril(np.ones((self.M, self.M))))
        self.D = sp(np.eye(self.M)-np.eye(self.M, k=-1))
        self.R = sp(np.diag(self.ρ_vals))
        self.R_inv = sp(np.diag(1/self.ρ_vals))
        self.Δ = sp(np.diag(self.g * (self.D @ self.ρ_vals)))
        self.Δ_inv = sp(np.diag(1/(self.g * (self.D @ self.ρ_vals))))
        
    def modal_decomposition(self, h_bottom):
        assert h_bottom >= 0, "Fixed upper-layer depths are greater than\
 fluid depth, giving an unphyiscal negative value of bottom-layer fluid depth"
        h_vals = np.append(self.layer_thicknesses, h_bottom)
        B = self.g * np.fmin(self.ρ_vals[:, None], self.ρ_vals[None:,])
        S = sp((h_vals/self.ρ_vals)[None,:]*B)  
        
        from scipy.linalg import eig
        wave_speeds_sqrd, vertical_structures = eig(S.todense())
        wave_speeds, vertical_structures = eig_filter(
            wave_speeds_sqrd,
            vertical_structures
            )
        
        return wave_speeds, vertical_structures
    
    def global_modal_decomposition(self, N=101, plot_decompositions=True):
        bathymetry = np.linspace(self.min_depth, self.max_depth, N)
        Δh = (self.max_depth - self.min_depth)/(N-1)
        bottom_layer_depths = bathymetry - self.fixed_depth
        
        self.wave_speeds_sqrd = np.zeros((N, self.M))
        self.vertical_structures = np.zeros((N, self.M, self.M))
        
        for i, bottom_layer_depth in enumerate(bottom_layer_depths):
            self.wave_speeds_sqrd[i], self.vertical_structures[i] = \
                self.modal_decomposition(bottom_layer_depth)

        self.wave_speeds_sqrd_derivative_wrt_h = \
                np.zeros(self.wave_speeds_sqrd.shape)
                
        self.wave_speeds_sqrd_derivative_wrt_hh = \
                np.zeros(self.wave_speeds_sqrd.shape)
                
        # Second-order finite-differences approximation of 
        # wavespeed squared derivative with respect to fluid depth
        self.wave_speeds_sqrd_derivative_wrt_h[1:-1] = \
            (self.wave_speeds_sqrd[2:] -self.wave_speeds_sqrd[:-2])/(2 * Δh)
        self.wave_speeds_sqrd_derivative_wrt_h[0] = \
            (-3 * self.wave_speeds_sqrd[0] + \
             4 * self.wave_speeds_sqrd[1] - \
                 self.wave_speeds_sqrd[2])/(2 * Δh)
                
        self.wave_speeds_sqrd_derivative_wrt_h[-1] = \
            (3 * self.wave_speeds_sqrd[-1] - \
             4 * self.wave_speeds_sqrd[-2] + \
                 self.wave_speeds_sqrd[-3])/(2 * Δh)
                
        # Second-order finite-differences approximation of 
        # wavespeed squared second derivative with respect to fluid depth
        self.wave_speeds_sqrd_derivative_wrt_hh[1:-1] = \
            ( 1 * self.wave_speeds_sqrd[2:] + \
             -2 * self.wave_speeds_sqrd[1:-1] + \
              1 * self.wave_speeds_sqrd[:-2])/(Δh**2)
                 # 2	−5	4	−1
        self.wave_speeds_sqrd_derivative_wrt_hh[0] = \
            (2 * self.wave_speeds_sqrd[0] - \
             5 * self.wave_speeds_sqrd[1] + \
             4 * self.wave_speeds_sqrd[2] - \
                 self.wave_speeds_sqrd[3] )/(Δh**2)
                
        self.wave_speeds_sqrd_derivative_wrt_hh[-1] = \
            (- 2 * self.wave_speeds_sqrd[-1] + \
               5 * self.wave_speeds_sqrd[-2] - \
               4 * self.wave_speeds_sqrd[-3] + \
                   self.wave_speeds_sqrd[-4] )/(Δh**2)
                
        # Create functions dependent on fluid depth
        from scipy.interpolate import interp1d
        self.wave_speed_sqrd_functions = []
        self.wave_speed_sqrd_der_h_functions = []
        self.wave_speed_sqrd_der_hh_functions = []
        self.vertical_structure_functions = []
        self.normalisations = []

        for i in range(self.M):
            self.wave_speed_sqrd_functions.append(
                interp1d(bathymetry, self.wave_speeds_sqrd[:, i],
                         kind='cubic')
                )
            self.wave_speed_sqrd_der_h_functions.append(
                interp1d(bathymetry, self.wave_speeds_sqrd_derivative_wrt_h[:, i],
                         kind='cubic')
                )
            self.wave_speed_sqrd_der_hh_functions.append(
                interp1d(bathymetry, self.wave_speeds_sqrd_derivative_wrt_hh[:, i],
                         kind='cubic')
                )
            
            if self.normalisation == 'CONSTANT':
                Z_m = np.vectorize(lambda h : 1)
                
            else: # Anti-Symmetric
                Z_m  = np.vectorize(lambda h : np.sqrt(
                    self.wave_speed_sqrd_der_h_functions[-1](h)
                    ))

            structure_funcs = []
            for j in range(self.M):
                structure_funcs.append(interp1d(
                    bathymetry,
                    self.vertical_structures[:, i, j],
                    kind='cubic'))
            func = lambda h : np.array([Z_m(h) * f(h) for f in structure_funcs]).T
            self.vertical_structure_functions.append(func)
            self.normalisations.append(Z_m)

            if plot_decompositions:
                from ppp.Plots import plot_setup
                import matplotlib.pyplot as pt

                fig, ax = plot_setup("Fluid Depth (m)", 'Modal Wavespeed (m/s)',
                                     title=f'Mode {i}')
                bathymetry2 = np.linspace(self.min_depth, self.max_depth, 1001)
                ax.plot(bathymetry2,
                        np.sqrt(
                            self.wave_speed_sqrd_functions[i](bathymetry2)
                            ), 'k-'
                        )
                
                if self.M == 2:
                    ax.plot(bathymetry2,
                            np.sqrt(
                                self.wave_speed_sqrd_functions_apprx[i](bathymetry2)
                                ), 'r:'
                            )
                pt.show()
                
                fig, ax = plot_setup("Fluid Depth (m)", 'Vertical Structure',
                                     title=f'Mode {i}')
                ax.plot(bathymetry2,
                        self.vertical_structure_functions[i](bathymetry2))
                
                
                ax.legend(range(1, self.M+1), loc=1, fontsize=16,
                          title='Layer', title_fontsize=18)
                ax.plot(bathymetry2,
                        self.vertical_structure_functions_apprx[i](bathymetry2),
                        'r-.')
                pt.show()
    
        return self.wave_speed_sqrd_functions, self.vertical_structure_functions
            
    def generate_modal_interaction_coefficients(self, plot_interaction_coefficients=False):
        try:
            wave_speed_sqrd_functions = self.wave_speed_sqrd_functions
            vertical_structure_functions = self.vertical_structure_functions
            
        except AttributeError:
            wave_speed_functions, vertical_structure_functions = \
                self.global_modal_decomposition(N=101, 
                                                plot_decompositions=False)
                
        self.modal_interaction_coefficients = []
    
        
        for m in range(self.M):
            mode_M_coefficients = []
            c_sqrd_m = wave_speed_sqrd_functions[m]
            c_sqrd_m_prime = self.wave_speed_sqrd_der_h_functions[m]
            c_sqrd_m_primeprime = self.wave_speed_sqrd_der_hh_functions[m]

            for n in range(self.M):
                c_sqrd_n = wave_speed_sqrd_functions[n]
                c_sqrd_n_prime = self.wave_speed_sqrd_der_h_functions[n]
                if m == n:
                    if self.normalisation == 'CONSTANT':
                        T = lambda h : -.5 * c_sqrd_m_primeprime(h)/c_sqrd_m_prime(h)

                    else: # Anti-Symmetric
                        T = np.vectorize(lambda h : 0)
    
                else: #if m \neq n
                    if self.normalisation == 'CONSTANT':
                        T = lambda h : c_sqrd_m_prime(h)/(c_sqrd_n(h) - c_sqrd_m(h))
                        
                    else: # Anti-Symmetric
                        T = lambda h : np.sqrt(
                            c_sqrd_m_prime(h) * c_sqrd_n_prime(h)
                            ) / (c_sqrd_n(h) - c_sqrd_m(h))
                
                if plot_interaction_coefficients:
                    bathymetry = np.linspace(200, 4000, 1001)

                    from ppp.Plots import plot_setup
                    import matplotlib.pyplot as pt
                    fig, ax = plot_setup('Fluid Depth (m)', 'Interaction Coefficient',
                                         title=f'$\\mathcal{{T}}_{{{m},{n}}}$')
                    
                    ax.plot(bathymetry, T(bathymetry), 'k-')

                    if self.M == 2:
                        T_approx = self.modal_interaction_coefficients_apprx[m][n]
                        ax.plot(bathymetry, T_approx(bathymetry), 'r:')
                        
                    pt.show()
                    
                mode_M_coefficients.append(T)
            
            self.modal_interaction_coefficients.append(mode_M_coefficients)
            
    def two_layer_approximations(self):
        d = self.fixed_depth
        barotropic_wavespeed_sqrd_apprx = np.vectorize(lambda h : self.g * h)
        barotropic_wavespeed_sqrd_der_h_apprx = np.vectorize(lambda h : self.g)
        barotropic_wavespeed_sqrd_der_hh_apprx = np.vectorize(lambda h : 0)
        baroclinic_wavespeed_sqrd_apprx = np.vectorize(
            lambda h : self.reduced_gravity * d * \
                (h - d)/h
                )
        baroclinic_wavespeed_sqrd_der_h_apprx = np.vectorize(
           lambda h : self.reduced_gravity * (d/h) ** 2
            )
        baroclinic_wavespeed_sqrd_der_hh_apprx = np.vectorize(
            lambda h : -2 * self.reduced_gravity * (d**2)/\
                (h ** 3)
            )
            
        self.wave_speed_sqrd_functions_apprx = [
            barotropic_wavespeed_sqrd_apprx,
            baroclinic_wavespeed_sqrd_apprx
            ]
        self.wave_speed_sqrd_der_h_functions = [
            barotropic_wavespeed_sqrd_der_h_apprx,
            baroclinic_wavespeed_sqrd_der_h_apprx
            ]
        self.wave_speed_sqrd_der_hh_functions = [
            barotropic_wavespeed_sqrd_der_hh_apprx,
            baroclinic_wavespeed_sqrd_der_hh_apprx
            ]
        
        self.modal_interaction_coefficients_apprx = []
        
        if self.normalisation == 'CONSTANT':
            self.approx_normalisations = 2 * [np.vectorize(lambda h : 1)]
            self.modal_interaction_coefficients_apprx = [
                [np.vectorize(lambda h : 0),
                 np.vectorize(lambda h : -1/h)],
                [np.vectorize(lambda h : (self.reduced_gravity/self.g) * \
                              (d**2)/(h**3)),
                 np.vectorize(lambda h : 1/h)]
                    ]
        else: # Anti-Symmetric
            self.approx_normalisations = [
                lambda h : np.sqrt(self.wave_speed_sqrd_der_h_functions[0](h)),
                lambda h : np.sqrt(self.wave_speed_sqrd_der_h_functions[1](h))
                ]
            self.modal_interaction_coefficients_apprx = [
                [np.vectorize(lambda h : 0),
                 np.vectorize(lambda h : -np.sqrt(self.reduced_gravity/self.g) * \
                               (d)/(h**2))],
                [np.vectorize(lambda h : np.sqrt(self.reduced_gravity/self.g) * \
                              (d)/(h**2)),
                 np.vectorize(lambda h : 0)]
                    ]
            
            self.vertical_structure_functions_apprx = [
                lambda h : np.array([
                        self.approx_normalisations[0](h), self.approx_normalisations[0](h)
                    ]).T,
                lambda h : np.array([
                    -self.approx_normalisations[1](h) * (h-d)/d,
                    self.approx_normalisations[1](h)
                    ]).T
                ]
            
        return (self.wave_speed_sqrd_functions_apprx,
                self.modal_interaction_coefficients_apprx,
                self.approx_normalisations,
                self.vertical_structure_functions_apprx)
            
            

def eig_filter(vals, vecs, Z_normalisation=1):
    inds = np.argsort(vals.real)[::-1]
    vals, vecs = vals.real[inds], vecs[:, inds]

    return vals, Z_normalisation*(vecs/vecs[-1]).T

def test(show_topography=False):    
    import configure
    param, args = configure.main()
    
    domain_width = args.domain
    
    bbox = (-domain_width/2, domain_width/2, 0, domain_width)
    x0, xN, y0, yN = bbox
    alongshore = np.linspace(x0, xN, 1001)
    crossshore = np.linspace(y0, yN, 1001)
    
    X, Y = np.meshgrid(alongshore, crossshore)
    
    from barotropicSWEs.topography import canyon_func1
    bathymetry = param.H_D * canyon_func1(X, Y,
                             canyon_width=args.canyon_width,
                             canyon_intrusion=args.canyon_intrusion,
                             coastal_lengthscale=args.coastal_lengthscale,
                             coastal_shelf_width=args.shelf_width,
                             )
    
    if show_topography:
        from ppp.Plots import plot_setup
        import matplotlib.pyplot as pt
        fig, ax = plot_setup('Along-shore (km)', 'Cross-shore (km)')
        c = ax.matshow(
            bathymetry,
            cmap="Blues",
            vmax=param.H_D,
            vmin=0,
            extent=[x0*param.L_R*1e-3, xN*param.L_R*1e-3,
                    y0*param.L_R*1e-3, yN*param.L_R*1e-3],
            aspect="auto",
            origin="lower",
        )
        fig.colorbar(c, ax=ax)
        pt.show()
    
    layer_thicknesses = np.array([100, 50, 20])
    layer_densities = np.array([1028, 1030, 1035])    
    multi_layer_modes = MultiLayerModes(bathymetry, layer_thicknesses[0],
                                        layer_densities[0],
                                        normalisation='Anti-Symmetric')
    
    multi_layer_modes.global_modal_decomposition()
    multi_layer_modes.generate_modal_interaction_coefficients(plot_interaction_coefficients=True)

if __name__ == '__main__':
    test()
    
    
    