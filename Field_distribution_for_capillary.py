# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 11:02:54 2025

@author: nikolay_aprelov


based on: Capillary-Type Microfluidic Sensors Based on Optical Whispering Gallery Mode Resonances
A. Meldrum∗ and F. Marsiglio Department of Physics, University of Alberta, Edmonton, AB, T6G2E1, Canada Microfluidic

"""
# import warnings
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from Numerical_resonances_wavelengths_capillary_and_cylinder import resonance_cylinder, resonance_capillary

def E_capillary(R1, R2, n1, n2, n3, k_0, m, p, r):
    
    '''

    Parameters
    ----------
    R1 : float
        inner radius in mkm.
    R2 : float
        outer radius in mkm.
    n1 : float
        refractive index of the internal environment.
    n2 : float
        refractive index of the capillary tube matherial.
    n3 : float
        refractive index of the external environment.
    k_0 : complex float
        resonance wavenumber
    m : int
        azimuthal number.
    p : int
        radial number.
    r : float
        grid of radii in mkm.

    Returns
    -------
    E: np.array of complex floats
            complex radial distribution of the electric field.
    '''

    J = special.jv
    J_der = special.jvp
    Y = special.yv
    Y_der = special.yvp
    
    H1 = lambda a,b: J(a,b) + 1j*Y(a,b)
    H1_der = lambda a,b: J_der(a,b) + 1j*Y_der(a,b)
    H2 = lambda a,b: J(a,b) - 1j*Y(a,b)
    H2_der = lambda a,b: J_der(a,b) - 1j*Y_der(a,b)

    B = ((n2*J(m, n1*k_0*R1)*H1_der(m, n2*k_0*R1) - n1*J_der(m, n1*k_0*R1)*H1(m, n2*k_0*R1))/
         (-n2*J(m, n1*k_0*R1)*H2_der(m, n2*k_0*R1) + n1*J_der(m, n1*k_0*R1)*H2(m, n2*k_0*R1)))
    A = (B*H2(m, n2*k_0*R1) + H1(m, n2*k_0*R1))/J(m, n1*k_0*R1)
    D = (B*H2(m, n2*k_0*R2) + H1(m, n2*k_0*R2))/H1(m, n3*k_0*R2)
    
    if any([np.isnan(A), np.isnan(B),np.isnan(D),np.isinf(A),np.isinf(B),np.isinf(D)]):
        # warnings.warn('Error in calculations, capillary is too thick. Instead, the field distribution in the cylinder is used.')
        # Будто тут можно вернуть распределение для цилиндра 
        raise ValueError('Error in calculations, capillary is too thick. Use the field distribution of the cylinder instead.')
    
    def make_E_capillary(A,B,D,k_0, x):
        if x <= R1:
            return(A*J(m, n1*k_0*x))
        elif R1 < x < R2:
            return(B*H2(m, n2*k_0*x) + H1(m, n2*k_0*x))
        else:
            return(D*H1(m, n3*k_0*x))
    E = np.array([make_E_capillary(A,B,D,k_0,i) for i in r])
    return(E)


# def E_WGM_resonator(R1, R2, n1, n2, n3, k_0, m, p, r, pol = 'TE'):
#     if R1 == 0:
#         return(E_cylinder(R2,n2, m, p, r, pol))
#     else:
#         return(E_capillary(R1, R2, n1, n2, n3, k_0, m, p, r))




#%%

# R1 = 11
# R2 = 65
# n1 = 1
# n2 = 1.444
# n3 = 1
# m = 355 #117#122#128#               #135#362#590
# p = 25

# wl, k_0 = resonance_capillary(R1, R2, n1, n2, n3, m, p, pol = 'TE')
# Rarray = np.arange(R1-10, R2+10, 1e-3)
# I = np.abs(E_capillary(R1, R2, n1, n2, n3, k_0, m, p, Rarray)**2)
# # I2 = np.abs(E_WGM_resonator(R1, R2, n1_water, n2, n3, m2, p2, Rarray)**2)







# plt.rc('font', size = 20)
# # plt.plot(Rarray, I/np.sum(I), label = f'Microtube, n1 = {n1}, n2 = {n2}, n3 = {n3}, m = {m}, p = {p}')
# plt.plot(Rarray, I/np.sum(I), label = f'Microtube p = {p}')
# # plt.plot(Rarray, I2/np.sum(I2), label = f'Microtube, n1 = {n1}, n2 = {n2}, n3 = {n3}, m = {m2}, p = {p2}')
# plt.axvline(R1, c = 'r', linestyle = ':')
# plt.axvline(R2, c = 'r', linestyle = ':')
# plt.xlabel('Radius, mkm')
# plt.ylabel('Normalized intensity, arb.un')
# # plt.axvline(R2 - 3*2*np.pi*R2*n2/m*p, c = 'purple', linestyle = ':') # В случае толстого капилляра программа сшивает решения для цилиндра и капилляра в этой точке
# plt.legend()
