# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:47:39 2023

@author: Илья
"""


__date__='2025.07.02'
__version__='1.3'

import numpy as np
from numba import njit
import importlib.util
from scipy import sparse
import scipy.linalg as la
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.special import erf


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import os,sys
import time
import pickle

# pathname = os.path.dirname(sys.argv[0])  
pathname=os.path.dirname(__file__)
EPSILON_0=8.85418781762039e-15 # F/mm, dielectric constant
LIGHT_SPEED=3e11 #mm/s, speed of light
SIGMA=0.92*5.6e-8*1e-6 #W/mm**2/K**4 Stephan-Boltzman constant



def make_uneven_grids(x_min,x_max,x_ERV_init,ERV_0_init,mode_resolution_x,thermal_x_step):
    x_ERV=np.arange(min(x_ERV_init)-0.3,max(x_ERV_init)+0.3,mode_resolution_x)
    ERV_0=np.interp(x_ERV, x_ERV_init, ERV_0_init)
    
    x=np.concatenate((np.arange(x_min,x_ERV[0],thermal_x_step),x_ERV[1:-1],np.arange(x_ERV[-1],x_max,thermal_x_step)))
    
    return x, x_ERV, ERV_0

def make_even_grids(x_min,x_max,x_ERV_init,ERV_0_init,ERV_x_step,thermal_x_step):
    x_ERV=np.arange(min(x_ERV_init)-0.3,max(x_ERV_init)+0.3,ERV_x_step)
    ERV_0=np.interp(x_ERV, x_ERV_init, ERV_0_init)
    x=np.arange(x_min,x_max,thermal_x_step)
    return x, x_ERV, ERV_0




@njit(fastmath=True)
def my_tridiagonal_solver(KSI_n_j,N_x,T_bound,a,b,c):
    Temps_next=np.zeros(N_x)
    alpha=np.zeros((N_x-1),np.float32)
    betta=np.zeros((N_x-1),np.float32)
    b0=1
    c0=0
    d0=T_bound

  
    alpha[0]=-1*c0/b0
    betta[0]=d0/b0
    for j in range(0,N_x-2):
        alpha[j+1]= -a[j]/(b[j]+c[j]*alpha[j])
        betta[j+1]=(KSI_n_j[j]-c[j]*betta[j] )/(b[j]+c[j]*alpha[j])
    
    # теперь у нас есть массив альф и бетт, у нас есть рекуентной соотношение на наши тепемпературы

    Temps_next[N_x-1]=T_bound
    
    for j in range(2,N_x+1): #сейчас записываем температуры справа налево
        Temps_next[N_x-j]=alpha[N_x-j]*Temps_next[N_x-j+1]+betta[N_x-j]
    return Temps_next


class SNAP_ThermalModel():
    def __init__(self, x:np.array,T_0:np.array,
                 r_out=62.5e-3, # mm
                 r_in=0, # mm
                 medium_name='fused_silica',
                 T_bound=20,
                 absorption=None): # in Celsius
        '''
        T - in celsius
        
        '''
        
        self.x=x
        self.N_x=np.size(x)
        self.T=np.asarray(T_0,np.float32)
        self.T_bound=T_bound

        
        self.r=r_out
        self.r_in=r_in
        
        self.active_cooling=False
        
        temp=importlib.util.spec_from_file_location("medium_name",pathname+"\\media_params\\"+medium_name+'.py')
        self.medium = importlib.util.module_from_spec(temp)
        temp.loader.exec_module(self.medium)
        
        
        
        self.beta=self.medium.thermal_conductivity/self.medium.specific_heat_capacity/self.medium.density
        self.gamma=self.medium.heat_exchange/self.medium.specific_heat_capacity/self.medium.density*2*r_out/(r_out**2-r_in**2)
        self.delta=SIGMA/self.medium.specific_heat_capacity/self.medium.density*2*r_out/(r_out**2-r_in**2)    
            
        if absorption is None: # if no absorption specified, take the volumetric absorption of the material
            modal_heat_const=EPSILON_0*self.medium.refractive_index*LIGHT_SPEED*self.medium.absorption/2
        else:
            modal_heat_const=EPSILON_0*self.medium.refractive_index*LIGHT_SPEED*absorption/2
        # note that zeta should be muplyplied by Seff, depending on specific mode
        self.zeta=modal_heat_const/self.medium.density/self.medium.specific_heat_capacity/(np.pi*r_out**2) # without Seff!
        self.theta=1/(self.medium.specific_heat_capacity*np.pi*r_out**2*self.medium.density)
        
        self.thermal_optical_coefficient=self.medium.thermal_optical_coefficient
        
        self.delta_0=None
        self.delta_c=None
        
        self.recalculating_modes=False
        self.resonances_dynamics=None
        
        self.laser_pumping=False
        self.times=None
        self.pump_powers=None
        self.follow_resonance=False
        self.secondary_pump_powers=None
        
        self.CO2_positions=None
        self.CO2_powers=None
        self.CO2_laser_effective_source=None
        self.CO2_spot_radius_z=None
        
        '''
        for ERV estimation
        '''
        
        
        
    
    
    def set_active_cooling(self,active_heat_exchange,active_cooling_length):
        '''
        active_heat_exchange - heat exchange coefficient in W/mm**2/K
        2*active_cooling_length is the total active_cooling_length!
        '''
        self.active_cooling=True
        index=np.argmin(abs(self.x-(self.x[0]+active_cooling_length)))
        self.gamma_array=np.ones(len(self.x))*self.gamma
        self.gamma_array[0:index]=self.gamma/self.medium.heat_exchange*active_heat_exchange
        self.gamma_array[-index:]=self.gamma/self.medium.heat_exchange*active_heat_exchange
  
    
    # @njit(parallel=True)
    def estimate_ERV_through_relaxation(self):
        '''
        Fast and correct vectorized version for estimating ERV.
        '''
        DefectsStrength = np.ones(len(self.x), dtype=np.float32)
        ERV_dynamics = np.zeros((len(self.times), len(self.x)), dtype=np.float32)
        
        # Precompute all dt values (times[ii+1] - times[ii] for ii in 0...N-2)
        dt_all = np.diff(self.times)  # length is len(times)-1
        
        for ii in range(len(self.times)-1):  # ii ranges from 0 to len(times)-2
            T = self.T_dynamics[ii+1]  # matches original: T_dynamics[1:] starts at index 1
            
            # Vectorized computation
            t_relaxation = self.medium.viscosity(T) / self.medium.Young_modulus
            exp_factor = np.exp(-dt_all[ii] / t_relaxation)
            
            mask = T > (self.medium.T_annealing - 200)
            DefectsStrength *= np.where(mask, exp_factor, 1.0)
            
            # ERV_dynamics[ii+1] because we're processing step ii but storing in ii+1
            ERV_dynamics[ii+1] = ((1.0 - DefectsStrength) * 
                                 self.medium.maximum_ERV_per_square_mm * 
                                 np.pi * self.r**2)
        
        final_ERV = (1.0 - DefectsStrength) * self.medium.maximum_ERV_per_square_mm * np.pi * self.r**2
        return final_ERV, ERV_dynamics
  
    def estimate_ERV_through_relaxation_old(self):
        '''
        estimating ERV appearing during the whole process of heating
        '''
        

        DefectsStrength=np.ones(len(self.x))
        ERV_dynamics=np.zeros((np.size(self.times),np.size(self.x)),np.float32)
        
        
        for ii,T in enumerate(self.T_dynamics[1:]):
            dt=self.times[ii+1]-self.times[ii]
            DefectsStrength=DefectsStrength*np.array(list(map(lambda T:np.exp(-dt/(self.medium.viscosity(T)/self.medium.Young_modulus)) if T>self.medium.T_annealing-200 else 1,T)))
            ERV_dynamics[ii+1]=(np.ones(len(self.x))-DefectsStrength)*self.medium.maximum_ERV_per_square_mm*np.pi*self.r**2  
        
        final_ERV_array_ThroughRelaxation=(np.ones(len(self.x))-DefectsStrength)*self.medium.maximum_ERV_per_square_mm*np.pi*self.r**2
        return final_ERV_array_ThroughRelaxation,ERV_dynamics

  
    def set_SNAP_parameters(self,x_ERV:np.array,ERV_0:np.array,intristic_losses,lambda_0):
        '''
        x_ERV - is the dense grid for ERV determination
        lambda_0 in nm
        '''
        self.x_ERV=x_ERV
        self.ERV_0=ERV_0
        self.intristic_losses=intristic_losses # in 1/s
        
        dx=self.x_ERV[1]-self.x_ERV[0]
        N=np.size(self.x_ERV)-1
        self.Tmtx=-1/dx**2*sparse.diags([-2*np.ones(N+1),np.ones(N+1)[1:],np.ones(N+1)[1:]],[0,-1,1]).toarray()
                
        self.refractive_index =1.44
        self.k0=2*np.pi*self.refractive_index/(lambda_0*1e-6) # in 1/mm
        
        depth=2e-3 # mm
        self.Seff=depth*2*np.pi*self.r 
        self.lambda_0=lambda_0
        
        
        E,psi_distribs=self.solve_Shrodinger(ERV_0)
        
        self.resonances,self.delta_v=self.get_detunings_from_energies(E)
        
        self.psi_distribs=psi_distribs


        
    def set_external_CO2_laser_parameters(self,times,powers, positions,spot_radius_y,shift_y,spot_radius_z):
        if self.times is None:
            self.times=times
        elif len(self.times)!=len(powers):
            print('error! Times already are set with different array size')
            return
        self.CO2_positions=positions # in mm
        self.CO2_powers=powers # in W
        self.CO2_spot_radius_z=spot_radius_z
        self.CO2_laser_effective_source=1/np.sqrt(2)/self.CO2_spot_radius_z*self.theta*(erf((shift_y+self.r)*np.sqrt(2)/spot_radius_y)-erf((shift_y-self.r)*np.sqrt(2)/spot_radius_y))
        
        
        
        
    def set_pump_parameters(self,times,powers,wavelengths=None,detunings=None,mode_to_pump=None):
        '''
        times in s, powers in W, wavelengths in nm, detuning in nm
        '''
        self.recalculating_modes=True
        self.laser_pumping=True
        self.times=times 
        self.pump_powers=powers
        if wavelengths is None:
            self.follow_resonance=True
            self.pump_detunings=detunings
            self.mode_to_pump=mode_to_pump
            self.pump_wavelengths=np.zeros(len(times))
        else:
            self.pump_wavelengths=wavelengths
            self.follow_resonance=False
            self.pump_detunings=np.zeros(len(times))
        
    def set_secondary_pump_parameters(self,powers,wavelengths=None,detunings=None,mode_to_pump=None):
        '''
        times in s, powers in W, wavelengths in nm, detuning in nm
        '''
        self.secondary_pump_powers=powers
        if wavelengths is None:
            self.secondary_pump_detunings=detunings
            self.secondary_mode_to_pump=mode_to_pump
        else:
            self.secondary_pump_wavelengths=wavelengths
            
            
    
        
    def set_deltas(self,delta_0,delta_c):
        self.delta_0=delta_0
        self.delta_c=delta_c
        
        
    def set_taper_parameters(self,taper_position,taper_C2,taper_ImD,taper_losses):
        self.taper_position=taper_position
        self.taper_C2=taper_C2
        self.taper_ImD=taper_ImD
        self.taper_position_index=np.argmin(abs(self.x-self.taper_position))
        self.taper_losses=taper_losses
        
        
    
    # @jit(nopython=False)
    def solve_temper_model_with_source(self,dt:float,t_max:float,T:np.array,source:np.array):

        '''
        made by Arkady
        refactored by Ilya
        '''
        # dt=t_max/N_t
        t=0
        N_x=self.N_x
        #aj*U_n+1,j+1 +bj*U_n+1,j +cj*U_n+1,j-1=ksi_n - схема кранка-николcон
        """
        a=(-1*Beta*dt)/(2*dx*dx)
        c=a
        b=1+dt*Beta/(dx*dx)"""
        Temps_previous=T # массив температур
 
        
 
        #     heat capacity -120 +4.56*Temp-(7.38*1e-3)*np.power(Temp,2)+(6.59*1e-6)*np.power(Temp,3)-(3.05*1e-9)*np.power(Temp,4)+(5.72*1e-13)*np.power(Temp,5)
        

        
        #пользуемся гран условиями 
        #b0X0+c0X1=d0

      

        h_a=np.diff(self.x[1:])
        h_b=np.diff(self.x[:-1])
        a=(-1*self.beta*dt)/(2*h_a*h_a)
        c=(-1*self.beta*dt)/(2*h_a*h_b)
        b=1+0.5*dt*self.beta*(h_a+h_b)/(h_a*h_a*h_b)
        

        
        while t<t_max: #идем по времени
            t+=dt
            
            KSI_n_j=(1-0.5*self.beta*dt*(h_a+h_b)/(h_a*h_a*h_b))*Temps_previous[1:-1] +(self.beta*dt/(2*h_a*h_a))*Temps_previous[2:] +(self.beta*dt/(2*h_a*h_b))*Temps_previous[:-2] + dt*source[1:-1] \
                        -dt*(self.delta*(Temps_previous[1:-1]+273)**4-self.delta*(self.T_bound+273)**4)
            
            
            if self.active_cooling:
                KSI_n_j-=dt*self.gamma_array[1:-1]*(Temps_previous[1:-1]-self.T_bound )
            else:
                KSI_n_j-=dt*self.gamma*(Temps_previous[1:-1]-self.T_bound )
                
            Temps_previous = my_tridiagonal_solver(KSI_n_j,N_x,self.T_bound,a,b,c)
         # print(times1,times2)
        return Temps_previous
     

            

    def solve_temper_model_with_source_old(self,dt:float,t_max:float,T:np.array,source:np.array):

        '''
        made by Arkady
        refactored by Ilya
        '''
        # dt=t_max/N_t
        t=0
        N_x=self.N_x
        #aj*U_n+1,j+1 +bj*U_n+1,j +cj*U_n+1,j-1=ksi_n - схема кранка-николcон
        """
        a=(-1*Beta*dt)/(2*dx*dx)
        c=a
        b=1+dt*Beta/(dx*dx)"""
        Temps_previous=T # массив температур
        Temps_next = np.empty_like(T) #                               Temps_next=np.zeros(len(T),np.float32)
        
 
        #     heat capacity -120 +4.56*Temp-(7.38*1e-3)*np.power(Temp,2)+(6.59*1e-6)*np.power(Temp,3)-(3.05*1e-9)*np.power(Temp,4)+(5.72*1e-13)*np.power(Temp,5)
        
        alpha=np.zeros((N_x-1),np.float32)
        betta=np.zeros((N_x-1),np.float32)
        
        #пользуемся гран условиями 
        #b0X0+c0X1=d0
        b0=1
        c0=0
        d0=self.T_bound
        alpha[0]=-1*c0/b0
        betta[0]=d0/b0
        n=1

        h_a=np.diff(self.x[1:])
        h_b=np.diff(self.x[:-1])
        
        a=(-1*self.beta*dt)/(2*h_a*h_a)
        c=(-1*self.beta*dt)/(2*h_a*h_b)
        b=1+0.5*dt*self.beta*(h_a+h_b)/(h_a*h_a*h_b)
        
        bm=1
        am=0
        dm=self.T_bound
        
        while t<t_max: #идем по времени
            t+=dt
            KSI_n_j=(1-0.5*self.beta*dt*(h_a+h_b)/(h_a*h_a*h_b))*Temps_previous[1:-1] +(self.beta*dt/(2*h_a*h_a))*Temps_previous[2:] +(self.beta*dt/(2*h_a*h_b))*Temps_previous[:-2] + dt*source[1:-1] \
                        -dt*(self.delta*(Temps_previous[1:-1]+273)**4-self.delta*(self.T_bound+273)**4)
            
            if self.active_cooling:
                KSI_n_j-=dt*self.gamma_array[1:-1]*(Temps_previous[1:-1]-self.T_bound )
            else:
                KSI_n_j-=dt*self.gamma*(Temps_previous[1:-1]-self.T_bound )
        
                   
            for j in range(0,N_x-2):
                alpha[j+1]= -a[j]/(b[j]+c[j]*alpha[j])
                betta[j+1]=(KSI_n_j[j]-c[j]*betta[j] )/(b[j]+c[j]*alpha[j])
            
            # теперь у нас есть массив альф и бетт, у нас есть рекуентной соотношение на наши тепемпературы
            
            U1=(dm-am*betta[N_x-2])/(alpha[N_x-2]*am+bm)
            Temps_next[N_x-1]=U1
            
            for j in range(2,N_x+1): #сейчас записываем температуры справа налево
                Temps_next[N_x-j]=alpha[N_x-j]*Temps_next[N_x-j+1]+betta[N_x-j]
            Temps_previous=Temps_next
            # 
            
            
        # print(times1,times2)
        return Temps_next
 
    
    def solve_Shrodinger(self,ERV,Tmtx=None):
        '''
        calculate eigenvalues and eigenvectors normalized by maximum. 
        return only  eigenvalues<0
        '''

        U=-2*self.k0**2*ERV*(1e-6)/self.r/self.refractive_index
        
        
        Vmtx=np.diag(U)
        if Tmtx is not None:
            Hmtx=Tmtx+Vmtx    
        else:
            Hmtx=self.Tmtx+Vmtx
        (eigvals,eigvecs)=la.eigh(Hmtx,check_finite=False)
        eigvecs=eigvecs.T
        
        indexes=np.where(eigvals<0)
        eigvals,eigvecs=eigvals[indexes],eigvecs[indexes]
        
        eigvecs=(eigvecs.T/(np.max(np.abs(eigvecs),axis=1))).T
        
        return eigvals,eigvecs
    
    def calculate_V_eff(self,x,psi_distrib,Seff):
        L_eff=integrate.trapezoid(psi_distrib**2,x)
        return L_eff*Seff
 
     
    def get_detunings_from_energies(self,E):
        wl = (-E/(2*self.k0**2))*self.lambda_0+self.lambda_0    
        return wl,LIGHT_SPEED*(1/wl-1/self.lambda_0)
    
    def calculate_deltas(self,x,psi_distrib):
        L_eff=integrate.trapezoid(abs(psi_distrib)**2,x)
        index=np.argmin(abs(x-self.taper_position))
        delta_c=psi_distrib[index]**2/L_eff*self.taper_C2
        delta_0=psi_distrib[index]**2/L_eff*(self.taper_ImD-self.taper_C2)+self.intristic_losses
        return delta_0,delta_c

    def solve_full_system(self,dt=1e-3,log=True): # in s
        time1=time.time()
        T_array=np.zeros((np.size(self.times),np.size(self.x)),np.float32)
        T_array[0]=self.T
        N_steps=len(self.times)
        resonances_dynamics=[self.resonances]
      
        time_tic_1=0
        
        # ind1=np.argmin(abs(self.x_ERV[0]-self.x))-1
        # ind2=np.argmin(abs(self.x_ERV[-1]-self.x))+1
        
        for ii,t in enumerate(self.times[:-1]):
            source=np.zeros(len(self.x))
            
            if ii%100==0 and log:
                time_tic_2=time.time()
                time_remaining=(N_steps-ii)/100*(time_tic_2-time_tic_1)
                print('Modeling time={:.3f} s, step {} of {}, time remaining={:.0f} min {:.1f} s'.format(t,ii,N_steps,time_remaining//60,np.mod(time_remaining,60)))
                time_tic_1=time_tic_2
            
            
            if self.recalculating_modes:
                ERV=self.ERV_0+np.interp(self.x_ERV,self.x,self.T-self.T_bound)*self.thermal_optical_coefficient*self.r*1e6
                # ERV=self.ERV_0+(self.T[ind1:ind2]-self.T_bound)*self.thermal_optical_coefficient*self.r*1e6
                E_array,psi_distribs= self.solve_Shrodinger(ERV)
                resonance_wavelengths,delta_v=self.get_detunings_from_energies(E_array)
                resonances_dynamics.append(resonance_wavelengths)
                
            
            if self.laser_pumping:
                if self.follow_resonance is False:
                    laser_wavelength=self.pump_wavelengths[ii]
                    mode_to_pump=np.argmin(abs(laser_wavelength-resonance_wavelengths))
                    self.pump_detunings[ii]=laser_wavelength-resonance_wavelengths[mode_to_pump]
                else:
                    if len(resonance_wavelengths)<1:
                        print('ERROR: no resonances found')
                    mode_to_pump=self.mode_to_pump
                    laser_wavelength=resonance_wavelengths[mode_to_pump]+self.pump_detunings[ii]
                    self.pump_wavelengths[ii]=laser_wavelength
  
                Veff=self.calculate_V_eff(self.x_ERV,psi_distribs[mode_to_pump],self.Seff)
                resonance_wavelength=resonance_wavelengths[mode_to_pump]
                detuning_w=-2*np.pi*(laser_wavelength-resonance_wavelength)*LIGHT_SPEED/resonance_wavelength**2*1e6 # in 1/s
                
                
                if self.delta_0 is None:
                    delta_0,delta_c=self.calculate_deltas(self.x_ERV,psi_distribs[mode_to_pump])
                else:
                    delta_0=self.delta_0
                    delta_c=self.delta_c
                
                F=np.sqrt(4*self.pump_powers[ii]*delta_c/EPSILON_0/self.refractive_index**2/Veff)#
                Amplitude=np.sqrt(F**2/((delta_c+delta_0)**2+detuning_w**2)) 
                  
                intf= interp1d(self.x_ERV,psi_distribs,axis=1,bounds_error=False,fill_value=0)
                psi_distribs=intf(self.x)
                source=self.zeta*self.Seff*np.power(np.abs(Amplitude),2)*np.power(np.abs(psi_distribs[mode_to_pump]),2)
                
           
            # source[ind1:ind2]=self.zeta*self.Seff*np.power(np.abs(Amplitude),2)*np.power(np.abs(psi_distribs[mode_to_pump]),2)
            
                source[self.taper_position_index]+=self.taper_losses*self.pump_powers[ii]*self.theta/(self.x[self.taper_position_index]-self.x[self.taper_position_index-1])
            
            if self.CO2_powers is not None:
              source+=np.exp(-(2*(self.x-self.CO2_positions[ii])/self.CO2_spot_radius_z)**2)*self.CO2_powers[ii]*self.CO2_laser_effective_source
            
            if self.secondary_pump_powers is not None:
                F_2=np.sqrt(4*self.secondary_pump_powers[ii]*delta_c/EPSILON_0/self.refractive_index**2/Veff)#
                Amplitude_2=np.sqrt(F**2/((delta_c+delta_0)**2+detuning_w**2)) 
                secondary_source=self.zeta*self.Seff*np.power(np.abs(Amplitude_2),2)*np.power(np.abs(psi_distribs[self.secondary_mode_to_pump]),2)
                source+=  secondary_source    
                
                
            # print(max(source))
            self.T=self.solve_temper_model_with_source(dt,self.times[ii+1]-self.times[ii],
                                                       np.asarray(self.T,dtype=np.float32),np.asarray(source,dtype=np.float32))
            T_array[ii+1]=self.T
            time_tic_2=time.time()
        if self.laser_pumping:
            self.pump_wavelengths[-1]=laser_wavelength
        self.T_dynamics=T_array
        self.resonances_dynamics=resonances_dynamics
        time2=time.time()
        time_elapsed=time2-time1
        print('Total time of calculation is {} min {:.0f} s'.format(time_elapsed//60, np.mod(time_elapsed,60)))
        return T_array,resonances_dynamics
    
    def plot_pump_dynamics(self):
        plt.figure()
        plt.plot(self.times,self.pump_wavelengths)
        plt.xlabel('Time, s')
        plt.ylabel('Wavelength, nm')
        plt.twinx(plt.gca())
        plt.plot(self.times,self.pump_powers,color='red')
        plt.ylabel('Power, W')
        plt.tight_layout()
        
    
    def plot_modes_distribs(self):
        _,psi_distribs=self.solve_Shrodinger(self.ERV_0)
        plt.figure()
        plt.plot(self.x_ERV, self.ERV_0,color='black')
        plt.xlabel('Position, mm')
        plt.ylabel('ERV, nm')
        plt.twinx(plt.gca())
        # plt.plot(x_ERV,psi_distribs[5]**2)
        for psi in psi_distribs:
            plt.plot(self.x_ERV,abs(psi)**2)
        plt.ylabel('Intensity')
        
        
    def get_ERV_at_T(self,T=0, mode_resolution_x=None):
        if len(T)<1:
            ERV=self.ERV_0
        else:
            ERV=self.ERV_0+np.interp(self.x_ERV,self.x,T-self.T_bound)*self.thermal_optical_coefficient*self.r*1e6
        return ERV
  
    def get_modes_at_T(self,T=0, mode_resolution_x=None):
        ERV=self.get_ERV_at_T(T, mode_resolution_x)
        if mode_resolution_x!=None:
            dense_x=np.arange(self.x_ERV[0],self.x_ERV[-1],mode_resolution_x)
            ERV=np.interp(dense_x,self.x_ERV,ERV)
            dx=dense_x[1]-dense_x[0]
            N=np.size(dense_x)-1
            Tmtx=-1/dx**2*sparse.diags([-2*np.ones(N+1),np.ones(N+1)[1:],np.ones(N+1)[1:]],[0,-1,1]).toarray()
        else:
            Tmtx=None
        
        E,psi_distribs=self.solve_Shrodinger(ERV,Tmtx)
        resonance_wavelengths,_=self.get_detunings_from_energies(E)

        return resonance_wavelengths,(dense_x,psi_distribs)
        
    def get_modes_dynamics(self):
        resonances=[]
        for ii,T in enumerate(self.T_dynamics):
            resonance_wavelengths=self.get_resonances_at_T(T)
            resonances.append(resonance_wavelengths)
        self.resonances_dynamics=resonances
        return resonances

    def get_transmission(self, resonance_dynamics, pump_wavelengths):
        detuning_w=-2*np.pi*(pump_wavelengths-resonance_dynamics)*LIGHT_SPEED/self.lambda_0**2*1e6 # in 1/s
        transmission=1-4*self.delta_0*self.delta_c/((self.delta_0+self.delta_c)**2+detuning_w**2)
        return transmission
    
    
    def get_amplitude(self, resonance_dynamics, pump_wavelengths,normalized=True):
        '''
        amplitude calculated for the linear stationary case
        if "normalized" - normalized to the maximum of the amplitude
        '''
        detuning_w=-2*np.pi*(pump_wavelengths-resonance_dynamics)*LIGHT_SPEED/self.lambda_0**2*1e6 # in 1/s
        if normalized:
            a_2=(self.delta_0+self.delta_c)**2/((self.delta_0+self.delta_c)**2+detuning_w**2)
        else:
            Veff=self.calculate_V_eff(self.x_ERV,self.psi_distribs[self.mode_to_pump],self.Seff)
            F=np.sqrt(4*self.pump_powers*self.delta_c/EPSILON_0/self.refractive_index**2/Veff)#
            a_2=(F**2/((self.delta_c+self.delta_0)**2+detuning_w**2)) 
      
        return a_2
        
    
    def plot_modes_dynamics(self,xaxis='waves',step=20,modes_to_plot=None):
        import matplotlib.cm as cm
        if modes_to_plot is None:
            n = max([len(i) for i in self.resonances_dynamics])
        else:
            n=len(modes_to_plot)
        colormap = cm.rainbow(np.linspace(0, 1, n))
        plt.figure()
        
        for ii,T in enumerate(self.times):
            if ii%step==0:
                color_index=0
                for jj,w in enumerate(self.resonances_dynamics[ii]):
                    # print(w)
                    if modes_to_plot is not None:
                        if jj not in modes_to_plot:
                            continue
                    if xaxis=='waves':
                        plt.scatter(self.pump_wavelengths[ii], w,color=colormap[color_index])
                        
                    elif xaxis=='times':
                        plt.scatter(T, w,color=colormap[color_index])
                    color_index=+1
                        
                           
        if xaxis=='waves':
            plt.xlabel('Pump wavelength, nm')
        elif xaxis=='times':
            plt.xlabel('Time, s')
            
        plt.ylabel('Resonance wavelength, nm')
        if xaxis=='times':
            plt.plot(self.times, self.pump_wavelengths)
   
            
    def plot_T_dynamics(self,step=1):
        fig, ax = plt.subplots()
        plt.xlabel('Coordinate, mm')
        plt.ylabel('Temperature, C')
        T_max=np.max(self.T_dynamics)
        def update(frame):
            ax.clear()
            ax.plot(self.x, self.T_dynamics[frame])
            ax.set_ylim(self.T_bound, T_max)
            ax.set_title(f'Time: {self.times[frame]:.2f} s')
            
            
        ani = FuncAnimation(fig, update, frames=len(self.T_dynamics[::step]), repeat=False)
        
        # Save as GIF
        ani.save("Results.gif", writer=PillowWriter(fps=10))
        plt.show()
        
        
    def plot_T_max_dynamics(self):
        plt.figure()
        T_max=np.max(self.T_dynamics,axis=1)
        plt.plot(self.times, T_max)
        
        plt.xlabel('Time,s')
        plt.ylabel('Temperature at maximum, C')
        
    def plot_T(self,time):
        time_ind=np.argmin(abs(self.times-time))
        plt.figure()
        plt.plot(self.x, self.T_dynamics[time_ind])
        plt.xlabel('Position, mm')
        plt.ylabel('Temperature, C')

    def save_model(self,f_name):
        with open(f_name,'wb') as f:
            del(self.medium)
            pickle.dump(self,f)
            
    def load_results(self,f_name):
        with open(f_name,'rb') as f:
            [self.x,self.times, self.ERV_0,self.pump_powers,self.pump_wavelengths,self.T_dynamics,self.resonances_dynamics]=pickle.load(f)
            
            
def load_model(f_name):
    with open(f_name,'rb') as f:
        model=pickle.load(f)
    return model
        
            
    
    
def get_FSR(resonances,number_of_modes=None):
    if number_of_modes==None:
        number_of_modes=len(resonances)
    freqs=LIGHT_SPEED/1e3/resonances[:number_of_modes] # in GHz
    diff=np.diff(freqs)
    return diff
    
    
def get_Dint(resonances,pump_mode,number_of_modes=None):
    if number_of_modes==None:
        number_of_modes=len(resonances)
    freqs=LIGHT_SPEED/1e3/resonances[:number_of_modes] # in GHz
    diff=np.diff(freqs)
    FSR=(diff[pump_mode]+diff[pump_mode-1])/2
    linspace=np.arange(0,number_of_modes)-pump_mode
    mu_by_FSR=(linspace)*(np.ones(np.shape(freqs)).T*FSR).T
    Dint=freqs-mu_by_FSR-freqs[pump_mode]
    return Dint




def get_synchronism(resonances,pump_mode,number_of_modes=None):
    if number_of_modes==None:
        number_of_modes=len(resonances)
    freqs=LIGHT_SPEED/1e3/resonances # in GHz
    S=[]
    for ii in range(number_of_modes):
        S.append(freqs[pump_mode+ii]+freqs[pump_mode-ii]-2*freqs[pump_mode])
    return np.array(S)



def get_Dint_dynamics(resonances_dynamics,mode_to_pump,number_of_modes):
    Dint_array=[]
    for resonances in resonances_dynamics:
        Dint=get_Dint(resonances,mode_to_pump,number_of_modes)
        Dint_array.append(Dint)
        
    return Dint_array    

def plot_Dint_dynamics(resonances_dynamics,mode_to_pump,number_of_modes):
    Dint_array=get_Dint_dynamics(resonances_dynamics, mode_to_pump, number_of_modes)
    plt.figure()
    plt.plot(Dint_array[0],label='init')
    plt.plot(Dint_array[-1],label='final')
    plt.legend()
    plt.xlabel('Mode number')
    plt.ylabel(r'$D_{int}$, GHz')
    plt.tight_layout()
    
    D_added=Dint_array[-1]-Dint_array[0]
    plt.figure()
    plt.plot(D_added)
    plt.xlabel('Mode number')
    plt.ylabel(r'$D_{int}^{thermal}$, GHz')
    plt.tight_layout()
         

           


    
if __name__=='__main__':
    print()
    # a=SNAP_ThermalModel(0,0,0)
    
    
