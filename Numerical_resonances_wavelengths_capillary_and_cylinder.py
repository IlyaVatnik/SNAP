# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:06:46 2025

@author: nikolay_aprelov

based on: Capillary-Type Microfluidic Sensors Based on Optical Whispering Gallery Mode Resonances
A. Meldrum∗ and F. Marsiglio Department of Physics, University of Alberta, Edmonton, AB, T6G2E1, Canada Microfluidic

"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy import special, optimize
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def resonance_cylinder(R, n_mat, n2, m, p, pol = 'TE', dispersion = False, plot_eigen_eq = False, sellmeier_coeffs = None):
    """
    
    Parameters
    ----------
    R : float
        radius of a cylinder.
    n_mat : float
        refractive index of a cylinder material.
    n2 : float
        refractive index of the external environment.
    m : int
        azimuthal number.
    p : int
        radial number.
    pol : str, optional
        'TE' or 'TM' polarization. The default is 'TE'.
    dispersion : boolean, optional
        Consider dispersion, using standart Sellmeier coeffisients for glass. The default is False.
    plot_eigen_eq : boolean, optional
        plot a figure of eigen equation where u can c resonances as dips in graph. The default is False.
    sellmeier_coeffs : list of floats, optional
        list of 6 Sellmeier coeffs. If dispersion == True and sellmeier_coeffs = None:
            sellmeier_coeffs for SiO2 are used. The default is None.

    Raises
    ------
    ValueError
        If the radial number is bigger than number of resonances, the ValueError raises (dunno if it works in a cylinder)

    Returns
    -------
    (res_wl, res_k0) : (float, complex float)
            (resonance wavelength in mkm, resonance wavenumber)

    """

    Y = special.yv
    Y_der = special.yvp
    J = special.jv          # четыре переобозначения для вычисления функций
    J_der = special.jvp
    H1 = lambda a,b: J(a,b) + 1j*Y(a,b)
    H1_der = lambda a,b: J_der(a,b) + 1j*Y_der(a,b)
    
    if dispersion:
        if not sellmeier_coeffs:
            sellmeier_coeffs = SellmeierCoefficientsCalculating('SiO2', 20)
        n1 = lambda wl: RefInd(wl*1e3, sellmeier_coeffs)
    else:
        n1 = lambda wl: n_mat
    
    if pol == 'TE':
        def eigen_eq(k):
            eq_zero = n1(2*np.pi/k)*J_der(m, n1(2*np.pi/k)*k*R)*H1(m, n2*k*R) - n2*J(m, n1(2*np.pi/k)*k*R)*H1_der(m, n2*k*R)
            return(eq_zero)
    elif pol == 'TM':
        def eigen_eq(k):
            eq_zero = 1/n1(2*np.pi/k)*J_der(m, n1(2*np.pi/k)*k*R)*H1(m, n2*k*R) - 1/n2*J(m, n1(2*np.pi/k)*k*R)*H1_der(m, n2*k*R)
            return(eq_zero)
    
    k_min = m/n_mat/R
    k_max = m/n2/R
    roots_brackets = [k_min, k_max]
    
    karray = np.linspace(roots_brackets[0],roots_brackets[1], 10000)
    K = np.array([eigen_eq(i) for i in karray])
    peaks = find_peaks(-np.log10(K))[0]
    
    res = []
    resonances = []
    
    if plot_eigen_eq: 
        plt.figure()
        plt.plot(karray, np.log10(np.abs(K)))
        # plt.plot(karray, np.log10(np.abs(np.real(K))), label = 'real')
        # plt.plot(karray,  np.log10(np.abs(np.imag(K))), label = 'imag')
        
        plt.legend()
        plt.scatter(karray[peaks], np.log10(np.abs(K))[peaks], c = 'r')
        plt.xlabel('k, 1/mkm')
        plt.ylabel('arb. un.')
        plt.axvline(karray[peaks[p-1]]-0.01, c = 'g', linestyle = ':')
        plt.axvline(karray[peaks[p-1]]+0.01, c = 'g', linestyle = ':')
    

    for i in range(len(peaks)):
        X0 = karray[peaks[i]]-0.01
        X1 = karray[peaks[i]]+0.01
        solution = optimize.root_scalar(f = lambda x: np.imag(eigen_eq(x)), bracket=[X0, X1], xtol=1e-18)
        # solution = optimize.root_scalar(eigen_eq, x0 = X0,  x1 = X1, xtol=1e-18)  # другой поиск корней через секущие (иногда выдаёт какой то бред)
        k_0 = solution.root 
        if plot_eigen_eq: 
            plt.axvline(k_0, c = 'purple', linestyle = ':')
        res.append(k_0)
    if p > len(res):
        raise ValueError(f'No resonance with such p: p_max = {len(resonances)} for this configuration')
    res_wl = 2*np.pi/np.real(res[p-1])
    return(res_wl, res[p-1])


def resonance_capillary(R1, R2, n1, n_mat, n3, m, p, pol = 'TE', dispersion = False, plot_eigen_eq = False, sellmeier_coeffs = None):
    """

    Parameters
    ----------
    R1 : float
        inner radius in mkm.
    R2 : float
        outer radius in mkm.
    n1 : float
        refractive index of the internal environment.
    n_mat : float
        refractive index of the capillary tube material.
    n3 : float
        refractive index of the external environment.
    m : int
        azimuthal number.
    p : int
        radial number.
    pol : str, optional
        'TE' or 'TM' polarization. The default is 'TE'.
    dispersion : boolean, optional
        Consider dispersion, using standart Sellmeier coeffisients for glass. The default is False.
    plot_eigen_eq : boolean, optional
        plot a figure of eigen equation where u can c resonances as dips in graph. The default is False.
    sellmeier_coeffs : list of floats, optional
        list of 6 Sellmeier coeffs. If dispersion == True and sellmeier_coeffs = None:
            sellmeier_coeffs for SiO2 are used. The default is None.

    Raises
    ------
    ValueError
        If the radial number is bigger than number of resonances, the ValueError raises

    Returns
    -------
    (res_wl, res_k0) : (float, complex float)
            (resonance wavelength in mkm, resonance wavenumber)

    """
    
    Y = special.yv
    Y_der = special.yvp
    J = special.jv
    # H1 = special.hankel1
    # H2 = special.hankel2
    J_der = special.jvp
    H1 = lambda a,b: J(a,b) + 1j*Y(a,b)
    H1_der = lambda a,b: J_der(a,b) + 1j*Y_der(a,b)
    H2 = lambda a,b: J(a,b) - 1j*Y(a,b)
    H2_der = lambda a,b: J_der(a,b) - 1j*Y_der(a,b)
    
    if dispersion:
        if not sellmeier_coeffs:
            sellmeier_coeffs = SellmeierCoefficientsCalculating('SiO2', 20)
        n2 = lambda wl: RefInd(wl*1e3, sellmeier_coeffs)
    else:
        n2 = lambda wl: n_mat

    if pol == 'TE':
        def eigen_eq(k):
            B = ((n2(2*np.pi/k)*J(m, n1*k*R1)*H1_der(m, n2(2*np.pi/k)*k*R1) - n1*J_der(m, n1*k*R1)*H1(m, n2(2*np.pi/k)*k*R1))/
                  (-n2(2*np.pi/k)*J(m, n1*k*R1)*H2_der(m, n2(2*np.pi/k)*k*R1) + n1*J_der(m, n1*k*R1)*H2(m, n2(2*np.pi/k)*k*R1)))
            if np.isnan(B):
                B = 1
            eq_zero = (n3*H1_der(m, n3*k*R2)*(B*H2(m, n2(2*np.pi/k)*k*R2) + H1(m, n2(2*np.pi/k)*k*R2)) - 
                        n2(2*np.pi/k)*H1(m, n3*k*R2)*(B*H2_der(m, n2(2*np.pi/k)*k*R2) + H1_der(m, n2(2*np.pi/k)*k*R2)))
            return(eq_zero)
    elif pol == 'TM':
        def eigen_eq(k):
            B = ((n1*J(m, n1*k*R1)*H1_der(m, n2(2*np.pi/k)*k*R1) - n2(2*np.pi/k)*J_der(m, n1*k*R1)*H1(m, n2(2*np.pi/k)*k*R1))/
                  (-n1*J(m, n1*k*R1)*H2_der(m, n2(2*np.pi/k)*k*R1) + n2(2*np.pi/k)*J_der(m, n1*k*R1)*H2(m, n2(2*np.pi/k)*k*R1)))
            if np.isnan(B):
                B = 1
            eq_zero = (n2(2*np.pi/k)*H1_der(m, n3*k*R2)*(B*H2(m, n2(2*np.pi/k)*k*R2) + H1(m, n2(2*np.pi/k)*k*R2)) - 
                        n3*H1(m, n3*k*R2)*(B*H2_der(m, n2(2*np.pi/k)*k*R2) + H1_der(m, n2(2*np.pi/k)*k*R2)))
            return(eq_zero)

    k_min = m/n_mat/R2
    k_max = m/n3/R2
    roots_brackets = [k_min, k_max]
    karray = np.linspace(roots_brackets[0],roots_brackets[1], 10000)
    K = np.array([eigen_eq(i) for i in karray])

    peaks = find_peaks(-np.log10(K))[0]    
    resonances = []
    if plot_eigen_eq:
        plt.figure()
        plt.plot(karray, np.log10(np.abs(K)), label = 'abs')
        # plt.plot(karray, np.real(K), label = 'real')
        # plt.plot(karray, np.imag(K), label = 'imag')
        plt.legend()
        peaks = find_peaks(-np.log10(K))[0]
        plt.scatter(karray[peaks], np.log10(K)[peaks], c = 'r')
        if p > len(peaks):
            raise ValueError(f'No resonance with such p: p_max = {len(peaks)} for this configuration')
        plt.axvline(karray[peaks[p-1]]-0.01, c = 'g', linestyle = ':')
        plt.axvline(karray[peaks[p-1]]+0.01, c = 'g', linestyle = ':')
        plt.xlabel('k, 1/mkm')
        plt.ylabel('arb. un.')
    
    for i in range(len(peaks)):
        X0 = karray[peaks[i]]-0.01
        X1 = karray[peaks[i]]+0.01
        # solution = optimize.root_scalar(eigen_eq, bracket=[X0, X1], xtol=1e-18) # В случае возникновения багов надо реализовать этот способ (взять только поиск для мнимой части в диапазоне шумной реальной части)
        solution = optimize.root_scalar(eigen_eq, x0 = X0,  x1 = X1, xtol=1e-18)  # другой поиск корней через секущие (иногда выдаёт какой то бред)
        k_0 = solution.root 
        if plot_eigen_eq:
            plt.axvline(np.real(k_0), c = 'purple', linestyle = ':')
        Q = abs(np.real(k_0)/2/np.imag(k_0))
        resonances.append((k_0, Q))
        if resonances[i-1][1] and resonances[i][1] < 1e3:
            del resonances[i]
            break
    if p > len(resonances):
        raise ValueError(f'No resonance with such p: p_max = {len(resonances)} for this configuration')
    res_wl = 2*np.pi/np.real(resonances[p-1][0])
    if resonances[p-1][1] < 1e4:
        print('Low Q: ', resonances[p-1][1])
    return(res_wl, resonances[p-1][0])
    
def RefInd(w, sellmeier_coeffs): # refractive index versus wavelength, w in nm, standard
    w = w * 1e-3
    t = 1
    # sellmeier_coeffs=Sellmeier_coeffs['SiO2']
    for i in range(0,3):
       t += sellmeier_coeffs[i]*w**2/(w**2-sellmeier_coeffs[i+3]**2)
    return np.sqrt(t)# np.sqrt(t)*(1+(T-T_0)*thermal_responses[medium][0])


def SellmeierCoefficientsCalculating(material, T):
    T = T+273.15
    sellmeier_coeffs = []
    
    if material == 'SiO2': #. D. B. Leviton and B. J. Frey, "Temperature-dependent absolute refractive index measurements of synthetic fused silica," in Optomechanical Technologies for Astronomy, E. Atad-Ettedgui, J. Antebi, and D. Lemke, eds. (2006), Vol. 6273, p. 62732K.
        sellmeier_coeffs.append(1.10127 + T*(-4.94251E-5) + (T**2)*(5.27414E-7) +
        (T**3)*(-1.597E-9) + (T**4)*(1.75949E-12))
        sellmeier_coeffs.append(1.78752E-05 + T*(4.76391E-5) + (T**2)*(-4.49019E-7) +
        (T**3)*(1.44546E-9) + (T**4)*(-1.57223E-12))
        sellmeier_coeffs.append(7.93552E-01 + T*(-1.27815E-3) + (T**2)*(1.84595E-5) +
        (T**3)*(-9.20275E-8) + (T**4)*(1.48829E-10))
        sellmeier_coeffs.append(-8.906E-2 + T*(9.08730E-6) + (T**2)*(-6.53638E-8) +
        (T**3)*(7.77072E-11) + (T**4)*(6.84605E-14))
        sellmeier_coeffs.append(2.97562E-01 + T*(-8.59578E-4) + (T**2)*(6.59069E-6) +
        (T**3)*(-1.09482E-8) + (T**4)*(7.85145E-13))
        sellmeier_coeffs.append(9.34454 + T*(-7.09788E-3) + (T**2)*(1.01968E-4) +
        (T**3)*(-5.0766E-7) + (T**4)*(8.21348E-10))
    else:
        pass
    return sellmeier_coeffs

if __name__ == "__main__":
    R1 = 40
    R2 = 65
    n1 = 1
    n2 = 1.444
    n3 = 1
    m = 355
    p = 1
    polarization = 'TM'
    
    # res, k_0 = resonance_capillary(R1, R2, n1, n2, n3, m, p, pol = polarization, dispersion = True, plot_eigen_eq=False)
    # res_water, k_0_water = resonance_capillary(R1, R2, n1+0.318, n2, n3, m, p, pol = polarization, dispersion = True)
    # res_cyl, k_0_cyl = resonance_cylinder(R2, n2, n3, m, p, polarization, dispersion = True)
    
    # print(f'Resonance wavelength and wavenumber of the empty capillary are {res} mkm, {k_0} 1/mkm')
    # print(f'Resonance wavelength and wavenumber of the water-filled capillary are {res_water} mkm, {k_0_water} 1/mkm')
    # print(f'Resonance wavelength and wavenumber of the cylinder are {res_cyl} mkm, {k_0_cyl} 1/mkm')

    res, k_0 = resonance_capillary(R1, R2, n1, n2, n3, m, p, pol = polarization, dispersion = True, plot_eigen_eq=False)
    res_no_dis, k_0 = resonance_capillary(R1, R2, n1, n2, n3, m, p, pol = polarization, dispersion = False, plot_eigen_eq=False)
    print(f'Resonance wavelength with dispersion {res*1e3} mkm')
    print(f'Resonance wavelength without dispersion {res_no_dis*1e3} mkm')