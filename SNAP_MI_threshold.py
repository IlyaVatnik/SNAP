# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 12:31:27 2025

@author: Илья
"""

__date__='2025.06.19'
__version__='1.3'

# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import airy
from scipy.integrate import quad
from scipy.linalg import eig,eigh
import warnings
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from contextlib import contextmanager

@contextmanager
def suppress_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=np.ComplexWarning)
        yield
        
c = 299792458 * 1e6
chi = 2.5e-10
EPSILON_0 = 8.85e-18



class SNAP_nonlinear_system:
    def __init__(self, params,dim_space=2**12):
        """

        
        Parameters:
        params (dict): Dictionary containing all system parameters:
            
            delta (float): intrinstic losses s^-1 (measured in experiment)
            delta_c (float): taper losses, s^-1
            Gamma (float): internal losses of the resonator, s^-1
            m_val (int): azimuthal number
            CouplingWidth (float): half-width of the taper in the constriction (half-width of the Gaussian function)
       
            Z_taper (float): Taper position along z in microns
            q0 (int): Pump axial mode number (counting from 0)
            
            P_in (float): Desired power threshold
            m_val (int): 
            CouplingWidth (float): 
            z_dr (array): Original z-points for dr
            dr (array): Radius variation values at z_dr points
        """
        # Extract parameters
        try:
            self.C2=params['C2']
            self.ImD=params['ImD']
            self.Gamma = params['Gamma']
        except KeyError:
            self.C2=None
            self.ImD=None
            pass
        
        
        try:
            self.delta_0 = params['delta_0']
            self.delta_c = params['delta_c']
            try:
                self.Gamma = params['Gamma']
            except KeyError:
                print('Gamma is set equal to delta_0')
                self.Gamma = self.delta_0
        except KeyError:
            self.delta_0=None
            self.delta_c=None
            pass
        


        
        self.Z_taper = params['Z_taper']
        self.q0 = params['q0']
        self.mu_max = params['mu_max']
        self.P_max = params['P_max']
        self.m_val = params['m_val']
        self.CouplingWidth = params['CouplingWidth']

        z_dr = params['z_dr']
        dr_original = params['dr']
        self.RadiusFiber = params['RadiusFiber']
        # System constants
        
        self.t0 = 2.338107
        self.number_modes = 1
       
        

        
        
        self.omega_spectrum=None
        self.mode_distribs=None
        
        # Create uniform z grid
        # Interpolate dr to uniform grid
        
        self.dim_space = dim_space
        zleft = np.min(z_dr)
        zright = np.max(z_dr)
        self.SpaceStep = (zright - zleft) / (self.dim_space - 1)
        self.z = np.linspace(zleft, zright, self.dim_space)

        interp_func = interp1d(z_dr, dr_original, kind='linear', 
                              fill_value=0.0, bounds_error=False)
        self.ERV = interp_func(self.z)
        
        
        self.omega = self.Azimuthal_Material_and_geom_DispersionApproximation(np.array([self.m_val]), 1, 1)
        ResonantL = c / self.omega * 2 * np.pi
        self.n = self.n_lambda(ResonantL)
        self.beta = self.omega * self.n / c
        self.coef_disp_mat = (1 + self.dn_omega(self.omega) * self.omega / self.n)
        
        self.I_vec = quad(self.Airy2(1), 0, self.RadiusFiber)[0] * 2 * np.pi / self.NormAiry(1)**2
        self.I_matrix = 2 * np.pi * quad(self.Airy4(1, 1, 1, 1), 0, self.RadiusFiber)[0] / self.NormAiry(1)**4
     
        
        
    def Airy2(self, i):
        def func(x):
            arg = -(2 * self.beta[i-1]**2 / self.RadiusFiber)**(1/3) * (x - self.RadiusFiber) - self.t0
            airy_val = airy(arg)[0]
            return x * airy_val**2
        return func

    def Airy4(self, m1, m2, m3, m4):
        def func(x):
            args = [
                -(2 * self.beta[i-1]**2 / self.RadiusFiber)**(1/3) * (x - self.RadiusFiber) - self.t0
                for i in [m1, m2, m3, m4]
            ]
            airy_vals = [airy(arg)[0] for arg in args]
            return x * airy_vals[0] * airy_vals[1] * airy_vals[2] * airy_vals[3]
        return func

    def calculate_modes(self):
        aa=0
        c = 299792458 * 1e6
        ResOmega = self.omega / (2 * np.pi)
        omega_spectrum_all = np.array([])
        omega_spectrum_az = np.zeros((self.number_modes, 300))
        
        for i in range(self.number_modes):
            if len(self.n) > 1:
                beta = 2 * np.pi * self.n[i] * ResOmega[i] / c
            else:
                beta = 2 * np.pi * self.n * ResOmega[i] / c
            
            VCons = -2 * beta**2 / self.RadiusFiber * self.ERV
            Function_spectrum, E_spectrum = self.Potential_static_spectrum_omega_power(VCons, self.dim_space, self.SpaceStep, 0)
            
            part_omega_spectrum = 1 / self.coef_disp_mat[i] * ResOmega[i] / (2 * beta**2) * E_spectrum
            
            if aa == 0:
                index = np.where(part_omega_spectrum > 0)
                part_omega_spectrum = part_omega_spectrum[part_omega_spectrum > 0]
                Function_spectrum= np.transpose(np.transpose(Function_spectrum)[index])
                
            
            new_spectrum = (-part_omega_spectrum + ResOmega[i]) / 1e12
            index_2 = np.argsort(new_spectrum)
            omega_spectrum_all = np.concatenate((omega_spectrum_all, np.sort(new_spectrum)))
            # omega_spectrum_az[i, :len(part_omega_spectrum)] = new_spectrum
            Function_spectrum= np.transpose(np.transpose(Function_spectrum)[index_2])
            Function_spectrum= np.transpose(np.transpose(Function_spectrum)[index_2])
            self.omega_spectrum, self.mode_distribs = omega_spectrum_all,Function_spectrum
            print('Amount of axial modes is {}'.format(len(self.omega_spectrum)))

            return omega_spectrum_all, omega_spectrum_az, Function_spectrum #  Function_spectrum - распределение амплитуд мод . номер столбца - номер аксиальной моды
    # omega_spectrum_az массив частот аксиальных мод, для каждой азимутальной


    # def calculate_FSR_and_D(self):
    #     fsr = -(self.omega_spectrum[ :- 1] - self.omega_spectrum[1:]) / 2
        
    #     D_in = (self.omega_spectrum - self.omega_spectrum[num] - 
    #             fsr * (range_mode - range_mode[num])) * 2 * np.pi * 1e12
        
    def Azimuthal_Material_and_geom_DispersionApproximation(self, m, p, P):
        t = [-2.33810741, -4.08794944, -5.52055983, -6.78670809, -7.94413359,
             -9.02265085, -10.04017, -11.00852430, -11.93601556, -12.82877675]
        c = 299792458 * 1e6
        a = t[p-1]
        
        A = [0.6961663, 0.4079426, 0.8974794]
        B = [0.0684043, 0.1162414, 9.896161]
        
        s = np.zeros(len(m))
        
        def equation(x_val, m_val):
            n_q = 1 + sum(A[i] * x_val**2 / (x_val**2 - B[i]**2) for i in range(3))
            P_val = 1 if P == 1 else 1 / n_q**2
            
            T = (m_val - a * (m_val/2)**(1/3) + 
                 (3/20) * a**2 * (m_val/2)**(-1/3) + 
                 (a**3 + 10)/1400 * (m_val/2)**(-1) - 
                 a * (479 * a**3 - 40)/504000 * (m_val/2)**(-5/3) - 
                 a**2 * (20231 * a**3 + 55100)/129360000 * (m_val/2)**(-7/3))
            
            lhs = 2 * np.pi * np.sqrt(n_q) * self.RadiusFiber / x_val
            rhs = (T - np.sqrt(n_q) * P_val / np.sqrt(n_q - 1) + 
                  a * (3 - 2 * P_val**2) * P_val * n_q**(3/2) * (m_val/2)**(-2/3) / 
                  (6 * (n_q - 1)**(3/2)) - 
                  n_q * P_val * (P_val - 1) * (P_val**2 * n_q + P_val * n_q - 1) * 
                  (m_val/2)**(-1) / (4 * (n_q - 1)**2))
            
            return lhs - rhs
        
        for i, m_val in enumerate(m):
            try:
                sol = fsolve(lambda x: equation(x, m_val), x0=1.5)
                s[i] = float(sol[0])
            except Exception as e:
                warnings.warn(f"Error solving for m={m_val}: {str(e)}")
                s[i] = np.nan
        
        s = c / s * 2 * np.pi
        return s

    def dn_omega(self, omega):
        c = 299792458 * 1e6
        A = [0.6961663, 0.4079426, 0.8974794]
        B = [0.0684043, 0.1162414, 9.896161]
        
        dn = 0.0
        for i in range(3):
            term = A[i] * B[i]**2 / ((2 * np.pi * c / omega)**2 - B[i]**2)**2
            dn += term
        
        return (2 * np.pi * c)**2 * dn / self.n / omega**3

    def n_lambda(self, lambda_):
        A = [0.6961663, 0.4079426, 0.8974794]
        B = [0.0684043, 0.1162414, 9.896161]
        
        n_q = 1
        for i in range(3):
            n_q += A[i] * lambda_**2 / (lambda_**2 - B[i]**2)
        
        return np.sqrt(n_q)

    def NormAiry(self, i):
        x = np.linspace(0, self.RadiusFiber, 1000)
        arg = -(2 * self.beta[i-1]**2 / self.RadiusFiber)**(1/3) * (x - self.RadiusFiber) - self.t0
        func = airy(arg)[0]
        return np.max(func)
  
    def Potential_static_spectrum_omega_power(self, V, dim, h, aa):
        main_diag = -2 * np.ones(dim)
        off_diag = np.ones(dim - 1)
        
        M = (np.diag(main_diag) + 
              np.diag(off_diag, -1) + 
              np.diag(off_diag, 1)) / h**2 - np.diag(V)
        
        if aa != 0:
            M[0, -1] = 1 / h**2
            M[-1, 0] = 1 / h**2
        
        E, R = eigh(M)
        return R, E



 
 
    def plot_modes_distribs(self,mode_numbers=None):
        if self.mode_distribs is None:
            self.calculate_modes()
        plt.figure()
        plt.plot(self.z, self.ERV*1e3,color='black')
        plt.xlabel('Position, mkm')
        plt.ylabel('ERV, nm')
        plt.twinx(plt.gca())
        # plt.plot(x_ERV,psi_distribs[5]**2)
        if mode_numbers==None:
            for psi in self.mode_distribs.T:
                plt.plot(self.z,abs(psi/np.max(psi))**2)
        else:
            for n in mode_numbers:
                plt.plot(self.z,abs(self.mode_distribs[:,n]/np.max(self.mode_distribs[:,n]))**2)
        plt.ylabel('Intensity')
 
    def calculate_pump_mode_params(self):
        # Calculate integrals
          
        pump_mode_distrib = self.mode_distribs[:,self.q0]
        self.pump_mode_distrib = pump_mode_distrib/np.max(abs(pump_mode_distrib))
        
        # Calculate effective length and volume
        self.L_eff_q0 = (np.abs( self.pump_mode_distrib[0])**2 / 2 + 
                   np.abs( self.pump_mode_distrib[-1])**2 / 2 + 
                   np.sum(np.abs( self.pump_mode_distrib[1:-1])**2)) * self.SpaceStep
        self.V_eff_q0 = self.I_vec * self.L_eff_q0
        
        f = (np.exp(-((self.z - self.Z_taper) / self.CouplingWidth)**2 / 2) / 
             np.sqrt(2 * np.pi) / self.CouplingWidth)
        
        # calculate coupling and q-factor parameters
        if self.C2==None:
            self.C2 = self.delta_c *  self.L_eff_q0 / np.sum( self.pump_mode_distrib**2 * f) / self.SpaceStep
            self.ImD = (self.delta_0+self.delta_c-self.Gamma)  *self.L_eff_q0 / np.sum( self.pump_mode_distrib**2 * f) / self.SpaceStep   # 1. A. Y. Kolesnikova and I. D. Vatnik, "Theory of nonlinear whispering-gallery-mode dynamics in surface nanoscale axial photonics microresonators," Phys. Rev. A 108, 033506 (2023).
        elif self.delta_0==None:
            self.delta_0=(self.ImD-self.C2) / (self.L_eff_q0 / np.sum( self.pump_mode_distrib**2 * f) / self.SpaceStep)+self.Gamma
            self.delta_c=self.C2 / (self.L_eff_q0 / np.sum( self.pump_mode_distrib**2 * f) / self.SpaceStep)
        
            
            
            
            
        
        
        # Calculate nonlinear coefficients
        g0 = (3 * self.omega[0] * chi * self.I_matrix / 
              self.V_eff_q0 / self.coef_disp_mat[0] / 2 / self.n[0]**2)
        self.g0 = g0 * ( self.pump_mode_distrib[0]**4 / 2 + 
                   self.pump_mode_distrib[-1]**4 / 2 + 
                   np.sum( self.pump_mode_distrib[1:-1]**4)) * self.SpaceStep
        
        
        P_threshold_nonlinear_effect = ((self.delta_0+self.delta_c)**3 / self.delta_c * EPSILON_0 * 
                                       self.n[0]**2 * self.V_eff_q0 / self.g0 * 2)
        
        print(f"P_threshold_nonlinear_effect = {P_threshold_nonlinear_effect} W")

        
      
        
    
    def find_min_positive_threshold(self):
        '''
        Calculate minimum positive threshold power for modulation instability
        '''
 
        self.calculate_pump_mode_params()
        # Coupling function
        coupling_function = (np.exp(-((self.z - self.Z_taper) / self.CouplingWidth)**2 / 2) / 
              np.sqrt(2 * np.pi) / self.CouplingWidth)
      
        with suppress_warnings():   
            PP = np.arange(0.01, self.P_max + 0.01, 0.01)
            
    
            while self.mu_max * 2 + 1 > len(self.omega_spectrum):
                self.mu_max -= 1
            
            AA = np.zeros((len(PP), self.mu_max))
            DW = np.zeros((len(PP), self.mu_max))
            P_th = np.zeros((len(PP), self.mu_max))
            
            for m, P in enumerate(PP):
                ResOmega = self.omega_spectrum
                num = self.q0
                range_mode = np.arange(0, len(ResOmega))
                range_mode = range_mode - num
                fsr = -(ResOmega[num - 1] - ResOmega[num+1]) / 2
                
                D_in = (ResOmega - ResOmega[num] - 
                       fsr * (range_mode - range_mode[num])) * 2 * np.pi * 1e12
                
                for mu in range(1, self.mu_max + 1):
                    q_plus = self.q0 + mu
                    q_minus = self.q0 - mu
                    
                    Z1 = self.mode_distribs[:, q_plus]
                    Z1 = Z1 / np.max(np.abs(Z1))
                    Z2 = self.mode_distribs[:,q_minus]
                    Z2 = Z2 / np.max(np.abs(Z2))
                    
                    # Calculate effective lengths
                    L_eff_q_plus = (Z1[0]**2 / 2 + 
                                    Z1[-1]**2 / 2 + 
                                    np.sum(Z1[1:-1]**2)) * self.SpaceStep
                    L_eff_q_minus = (Z2[0]**2 / 2 + 
                                     Z2[-1]**2 / 2 + 
                                     np.sum(Z2[1:-1]**2)) * self.SpaceStep
                    
                    # Calculate nonlinear coefficients
                    g1 = (3 * self.omega * chi * self.I_matrix / 
                          (self.I_vec * L_eff_q_plus) / self.coef_disp_mat / 2 / self.n**2)
                    delta1 = (self.ImD * np.sum(Z1**2 * coupling_function) * self.SpaceStep / 
                              L_eff_q_plus + self.Gamma)
                    delta2 = (self.ImD * np.sum(Z2**2 * coupling_function) * self.SpaceStep / 
                              L_eff_q_minus + self.Gamma)
                    
                    g11 = g1 * ( self.pump_mode_distrib[0]**2 * Z1[0]**2 / 2 + 
                                self.pump_mode_distrib[-1]**2 * Z1[-1]**2 / 2 + 
                                np.sum( self.pump_mode_distrib[1:-1]**2 * Z1[1:-1]**2)) * self.SpaceStep
                    g12 = g1 * ( self.pump_mode_distrib[0]**2 * Z1[0] * Z2[0] / 2 + 
                                self.pump_mode_distrib[-1]**2 * Z1[-1] * Z2[-1] / 2 + 
                                np.sum( self.pump_mode_distrib[1:-1]**2 * Z1[1:-1] * Z2[1:-1])) * self.SpaceStep
                    
                    g2 = (3 * self.omega * chi * self.I_matrix / 
                          (self.I_vec * L_eff_q_minus) / self.coef_disp_mat / 2 / self.n**2)
                    g22 = g2 * ( self.pump_mode_distrib[0]**2 * Z2[0]**2 / 2 + 
                                self.pump_mode_distrib[-1]**2 * Z2[-1]**2 / 2 + 
                                np.sum( self.pump_mode_distrib[1:-1]**2 * Z2[1:-1]**2)) * self.SpaceStep
                    g21 = g2 * ( self.pump_mode_distrib[0]**2 * Z1[0] * Z2[0] / 2 + 
                                self.pump_mode_distrib[-1]**2 * Z1[-1] * Z2[-1] / 2 + 
                                np.sum( self.pump_mode_distrib[1:-1]**2 * Z1[1:-1] * Z2[1:-1])) * self.SpaceStep
                    
                    Disp = (D_in[self.q0 + mu] + D_in[self.q0 - mu]) / 2
                    
                    F = np.sqrt(P * self.delta_c / (self.I_vec *  self.L_eff_q0) / self.n**2 / EPSILON_0)
                    A_vals = np.linspace(np.sqrt(delta1 * delta2) / np.sqrt(g12 * g21), 
                                        F**2 / (self.delta_c+self.delta_0)**2, 100)
                    
                    # Calculate frequency deviations
                    dW5_new = (-(g11 + g22) * A_vals + Disp + 
                              (delta1 + delta2) * np.sqrt((g12 * g21 * A_vals**2 - delta1 * delta2) / delta1 / delta2) / 2)
                    dW6_new = (-(g11 + g22) * A_vals + Disp - 
                              (delta1 + delta2) * np.sqrt((g12 * g21 * A_vals**2 - delta1 * delta2) / delta1 / delta2) / 2)
                    dW1 = -self.g0 * A_vals + np.sqrt(F**2 / A_vals - (self.delta_c+self.delta_0)**2)
                    
                    # Find intersection points
                    for j in range(len(A_vals) - 1):
                        if ((dW1[j] > dW6_new[j] and dW1[j+1] < dW6_new[j+1]) or 
                            (dW1[j] < dW6_new[j] and dW1[j+1] > dW6_new[j+1])):
                            DW[m, mu-1] = (dW1[j] + dW1[j+1]) / 2
                            AA[m, mu-1] = (A_vals[j] + A_vals[j+1]) / 2
                            P_th[m, mu-1] = P
                        elif ((dW1[j] > dW5_new[j] and dW1[j+1] < dW5_new[j+1]) or 
                              (dW1[j] < dW5_new[j] and dW1[j+1] > dW5_new[j+1])):
                            DW[m, mu-1] = (dW1[j] + dW1[j+1]) / 2
                            AA[m, mu-1] = (A_vals[j] + A_vals[j+1]) / 2
                            P_th[m, mu-1] = P
            
        # Find minimum positive threshold
        positive_P_th = P_th[P_th > 0]
        if positive_P_th.size > 0:
            min_P_th = np.min(positive_P_th)
            print(f"P_threshold_MI = {min_P_th} W")
            return min_P_th
        else:
            print("No positive thresholds found")
            return None

if __name__ == "__main__":
    
    h_width=5000 #mkm
    MaxRadVar=0.02 # mkm
    z_dr=np.linspace(-h_width*0.7, h_width*0.7,num=1000)
    dr=np.zeros(len(z_dr))
    dr[np.abs(z_dr) <= h_width/2] = MaxRadVar
    length_of_steepness=200
    mask1 = (z_dr > h_width/2) & (z_dr <= h_width/2 + length_of_steepness)
    dr[mask1] = np.linspace(MaxRadVar, 0, np.sum(mask1))
    mask2= (z_dr <- h_width/2) & (z_dr>= -h_width/2 - length_of_steepness)
    dr[mask2] = np.linspace(0, MaxRadVar, np.sum(mask2))
    cone=0.001
    dr+=cone*(z_dr-np.min(z_dr))*1e-3
    dr+=np.random.random(len(z_dr))*0.0005
    
    plt.figure()
    plt.plot(z_dr,dr)
    
    
    params = {
    # 'delta_0': 4e6, #total losses s^-1
    # 'delta_c': 2e6, # taper coupling, s^-1
    'Gamma': 4e6, # internal losses of the resonator, s^-1
    'Z_taper': 0, #   Taper position along z in microns
    'q0': 0, # Pump axial mode number (counting from 0)
    'mu_max': 3, # maximum detuning that is taken into account
    'P_max': 1, # Desired power threshold
    'm_val': 354, # azimuthal number
    'CouplingWidth': 1, #  half-width of the taper in the constriction (half-width of the Gaussian function)
    'RadiusFiber':62.5, # Fiber radius 
    'z_dr': z_dr,  # grid for ERV in mkm. Note that internal interpolation will be applied!
    'dr': dr  ,         # ERV,
    'C2':33887358691.86023,
    'ImD':33887358691.86023
    }
    
    # Создание и запуск системы
    SNAP = SNAP_nonlinear_system(params)
    SNAP.calculate_modes()
    min_threshold = SNAP.find_min_positive_threshold()

