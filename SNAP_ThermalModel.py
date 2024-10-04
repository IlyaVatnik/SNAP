# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:47:39 2023

@author: Илья
"""

"""
Physical constants
"""

__date__='04.10.2024'
__version__='0.1'

import numpy as np
import importlib.util
from scipy import sparse
import scipy.linalg as la
import scipy.integrate as integrate
import os,sys
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# pathname = os.path.dirname(sys.argv[0])  
pathname=os.path.dirname(__file__)
EPSILON_0=8.85418781762039e-15 # F/mm, dielectric constant
LIGHT_SPEED=3e11 #mm/s, speed of light
SIGMA=0.92*5.6e-8*1e-6 #W/mm**2/K**4 Stephan-Boltzman constant



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
        self.T=T_0
        self.T_bound=T_bound
        
        self.r=r_out
        self.r_in=r_in
        
        
        
        
        

        temp=importlib.util.spec_from_file_location("medium_name",pathname+"\\media_params\\"+medium_name+'.py')
        medium = importlib.util.module_from_spec(temp)
        temp.loader.exec_module(medium)
        
        self.beta=medium.thermal_conductivity/medium.specific_heat_capacity/medium.density
        self.gamma=medium.heat_exchange/medium.specific_heat_capacity/medium.density*2*r_out/(r_out**2-r_in**2)
        self.delta=SIGMA/medium.specific_heat_capacity/medium.density*2*r_out/(r_out**2-r_in**2)    
            
        if absorption is None:
            modal_heat_const=EPSILON_0*medium.epsilon*LIGHT_SPEED*medium.absorption/2
        else:
            modal_heat_const=EPSILON_0*medium.epsilon*LIGHT_SPEED*absorption/2
        # note that zeta should be muplyplied by Seff
        self.zeta=modal_heat_const/medium.density/medium.specific_heat_capacity/(np.pi*r_out**2) # without Seff!
   
  
        self.theta=1/(medium.specific_heat_capacity*np.pi*r_out**2*medium.density)
        
        self.thermal_optical_coefficient=medium.thermal_optical_coefficient
        
        self.delta_0=None
        self.delta_c=None
        
    def set_SNAP_parameters(self,x_ERV:np.array,ERV_0:np.array,intristic_losses,lambda_0):
        '''
        x_ERV - is the dense grid for ERV determination
        lambda_0 in nm
        '''
        self.x_ERV=x_ERV
        self.ERV_0=ERV_0
        self.intristic_losses=intristic_losses # in 1/s
                
        self.refractive_index =1.44
        self.k0=2*np.pi*self.refractive_index/(lambda_0*1e-6) # in 1/mm
        
        depth=2e-3 # mm
        self.Seff=depth*2*np.pi*self.r
        self.lambda_0=lambda_0
        
        
        E,psi_distribs=self.solve_Shrodinger(x_ERV, ERV_0)
        
        self.resonance_wavelengths,self.delta_v=self.get_detunings_from_energies(E)
        

        

        
        
    def set_pump_parameters(self,times,wavelengths,powers):
        self.times=times
        self.pump_wavelengths=wavelengths
        self.pump_powers=powers
        
        
    def set_deltas(self,delta_0,delta_c):
        self.delta_0=delta_0
        self.delta_c=delta_c
        
        
    def set_taper_parameters(self,taper_position,taper_C2,taper_ImD):
        self.taper_position=taper_position
        self.taper_C2=taper_C2
        self.taper_ImD=taper_ImD
        
        
        
            
    def solve_temper_model_with_source(self,dt:float,t_max:float,T:np.array,source:np.array):
        '''
        made by Arkady
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
        Temps_next=np.zeros(len(T))
 
        #     heat capacity -120 +4.56*Temp-(7.38*1e-3)*np.power(Temp,2)+(6.59*1e-6)*np.power(Temp,3)-(3.05*1e-9)*np.power(Temp,4)+(5.72*1e-13)*np.power(Temp,5)
        
        while t<t_max: #идем по времени
            t+=dt
            alpha=np.zeros((N_x-1))
            betta=np.zeros((N_x-1))
            #пользуемся гран условиями 
            #b0X0+c0X1=d0
            b0=1
            c0=0
            d0=self.T_bound
            alpha[0]=-1*c0/b0
            betta[0]=d0/b0
     
            #теперь определим все коэф alpha, betta
    
            
            for j in range(1,N_x-1):
                #print('j=',j)
                
                h_a=self.x[j+1]-self.x[j]
                h_b=self.x[j]-self.x[j-1]
                
                a=(-1*self.beta*dt)/(2*h_a*h_a)
                c=(-1*self.beta*dt)/(2*h_a*h_b)
                #если у меня есть тепплоотдача, то ее нужно учесть в тут
                b=1+0.5*dt*self.beta*(h_a+h_b)/(h_a*h_a*h_b)
                

                # if np.abs(self.x[j]-self.taper_position)<=h_a:
                #     deltafunction=1/h_a
                # else:
                #     deltafunction=0
                
                KSI_n_j=(1-0.5*self.beta*dt*(h_a+h_b)/(h_a*h_a*h_b))*Temps_previous[j] +(self.beta*dt/(2*h_a*h_a))*Temps_previous[j+1] +(self.beta*dt/(2*h_a*h_b))*Temps_previous[j-1] + dt*source[j] -dt*self.gamma*(Temps_previous[j]-self.T_bound )-dt*(self.delta*(Temps_previous[j]+273)**4-self.delta*(self.T_bound+273)**4)
                alpha[j]= -a/(b+c*alpha[j-1])
                betta[j]=(KSI_n_j-c*betta[j-1] )/(b+c*alpha[j-1])

            # теперь у нас есть массив альф и бетт, у нас есть рекуентной соотношение на наши тепемпературы
                
            #amXm-1+bmXm=dm
            bm=1
            am=0
            dm=self.T_bound
            U1=(dm-am*betta[N_x-2])/(alpha[N_x-2]*am+bm)
            
            Temps_next[N_x-1]=U1
            # for j in range(2,N_x+1): #сейчас записываем температуры справа налево
                # Temps_next[N_x-j]=alpha[N_x-j]*Temps_next[N_x-j+1]+betta[N_x-j]
            '''
            by ChatGPT
            '''
            x_indices = np.arange(2, N_x + 1)
            Temps_next[-x_indices] = alpha[-x_indices] * Temps_next[1-N_x + x_indices] + betta[-x_indices]
        
            Temps_previous=Temps_next
     
        return Temps_next
    
    
    def solve_Shrodinger(self,x,ERV):
        '''
        calculate eigenvalues and eigenvectors normalized by maximum. 
        return only  eigenvalues<0
        '''
        dx=x[1]-x[0]
        N=np.size(x)-1
        U=-2*self.k0**2*ERV*(1e-6)/self.r/self.refractive_index
        Tmtx=-1/dx**2*sparse.diags([-2*np.ones(N+1),np.ones(N+1)[1:],np.ones(N+1)[1:]],[0,-1,1]).toarray()
        Vmtx=np.diag(U)
        Hmtx=Tmtx+Vmtx
        (eigvals,eigvecs)=la.eigh(Hmtx,check_finite=False)
        eigvecs=eigvecs.T
        sorted_indexes=np.argsort(np.real(eigvals))
        eigvals,eigvecs=[eigvals[sorted_indexes],eigvecs[sorted_indexes]] 
        indexes=np.where(eigvals<0)
        eigvals,eigvecs=eigvals[indexes],eigvecs[indexes]
        eigvecs=eigvecs/np.max(eigvecs)
     
        
        return eigvals,eigvecs
    
    def calculate_V_eff(self,x,psi_distrib,Seff):
         L_eff=integrate.trapezoid(psi_distrib**2,x)
         return L_eff*Seff
 
    
 
    def plot_modes(self):
        _,psi_distribs=self.solve_Shrodinger(self.x_ERV, self.ERV_0)
        plt.figure()
        plt.plot(self.x_ERV, self.ERV_0,color='black')
        plt.xlabel('Position, mm')
        plt.ylabel('ERV, nm')
        plt.twinx(plt.gca())
        # plt.plot(x_ERV,psi_distribs[5]**2)
        for psi in psi_distribs:
            plt.plot(self.x_ERV,abs(psi)**2)
        plt.ylabel('Intensity')

     
    def get_detunings_from_energies(self,E):
        wl = (-E/(2*self.k0**2))*self.lambda_0+self.lambda_0    
        return wl,LIGHT_SPEED*(1/wl-1/self.lambda_0)
    
    def calculate_deltas(self,x,psi_distrib):
        L_eff=integrate.trapezoid(abs(psi_distrib)**2,x)
        index=np.argmin(abs(x-self.taper_position))
        delta_c=psi_distrib[index]**2/L_eff*self.taper_C2
        delta_0=psi_distrib[index]**2/L_eff*(self.taper_ImD-self.taper_C2)+self.intristic_losses
        return delta_0,delta_c

    def solve_full_system(self):
        dt=1e-3 # s
        T_array=np.zeros(len(self.times),len(self.x))
        T_array[0]=self.T
        for ii,t in enumerate(self.times[:-1]):
            print('time={%.3f}'.format(t))
            ERV=self.ERV_0+np.interp(self.x_ERV,self.x,self.T-self.T_bound)*self.thermal_optical_coefficient*self.r*1e6
            E_array,psi_distribs= self.solve_Shrodinger(self.x_ERV,ERV)
            
            intf= interp1d(self.x_ERV,psi_distribs,axis=1,bounds_error=False,fill_value=0)
            psi_distribs=intf(self.x)
                    
            laser_wavelength=self.pump_wavelengths[ii]
            resonance_wavelengths,delta_v=self.get_detunings_from_energies(E_array)
            # find the mode that is pumped
            mode_number_to_pump=np.argmin(abs(laser_wavelength-resonance_wavelengths))
            Veff=self.calculate_V_eff(self.x_ERV,psi_distribs[mode_number_to_pump],self.Seff)
            resonance_wavelength=resonance_wavelengths[mode_number_to_pump]
            detuning_v=-2*np.pi*(laser_wavelength-resonance_wavelength)*LIGHT_SPEED/resonance_wavelength**2
            
            if self.delta_0 is None:
                delta_0,delta_c=self.calculate_deltas(self.x_ERV,psi_distribs[mode_number_to_pump])
            
            F=np.sqrt(4*self.pump_powers[ii]*delta_c/EPSILON_0/self.refractive_index**2/Veff)#
            Amplitude=np.sqrt(F**2/((delta_c+delta_0)**2+detuning_v**2)) 
            source=self.zeta*self.Seff*np.power(np.abs(Amplitude),2)*np.power(np.abs(psi_distribs[mode_number_to_pump]),2)
            
            self.T=self.solve_temper_model_with_source(dt,self.times[ii+1],self.T,source)
            T_array[ii+1]=self.T
            
            return T_array
            
         
           
    
if __name__=='__main__':
    print()
    # a=SNAP_ThermalModel(0,0,0)
    
    
