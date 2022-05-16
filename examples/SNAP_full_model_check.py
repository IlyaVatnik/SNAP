# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 21:11:01 2021

Modification 13.04.2021

Time-dependent numerical solution for temperature distribution along the fiber under local heating with WGM mode
Following Gorodecky, p.313
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import scipy.integrate as integrate
from scipy.fftpack import fft,fftfreq
from scipy.signal import find_peaks

import time as time_module
import pickle



'''
Constants
'''
epsilon_0=8.85418781762039e-15 # F/mm, dielectric constant
c=3e11 #mm/s, speed of light
"""
Fused silica parameters
"""
thermal_conductivity=1.38*1e-3 # W/mm/K
heat_exchange=10*1e-6 #W/mm**2/K
sigma=0.92*5.6e-8*1e-6 #W/mm**2/K**4 Stephan-Boltzman constant

r=20.5e-3 #mm, fiber radius

specific_heat_capacity=740 # J/kg/K
density=2.2*1e-3*1e-3 # kg/mm**3
refractive_index=1.45
epsilon=refractive_index**2 #refractive index
absorption_in_silica=3.27e-09 #absorption in silica, 1/mm
# thermal_optical_responce=1.25e9 # Hz/Celcium, detuning of the effective_ra
thermal_optical_responce=0 # Hz/Celcium, detuning of the effective_ra
hi_3=2.5e-16 # mm**2/ V , nonliear responce
# hi_3=0 # mm**2/ V , nonliear responce
wavelength_0=1550e-6 # mm

"""
Sample parameters
"""
L=15 # mm, fiber sample length
T0=20 #Celsium, initial temperature of the sample being in equilibrium 

'''
Properties of the heating into the core
'''

absorption=50 #dB/m , absoprtion in the active core
ESA_parameter=0.15 # Excitated state absorption parameter, from  [Guzman-Chavez AD, Barmenkov YO, Kir’yanov A V. Spectral dependence of the excited-state absorption of erbium in silica fiber within the 1.48–1.59μm range. Appl Phys Lett 2008;92:191111. https://doi.org/10.1063/1.2926671.]
# thermal_expansion_coefficient=0.0107*r*1e3 #  nm/K, for effective radius variation

transmission_from_taper_to_amplifier=0.05  # parts, betwee the taper and amplifier
gain_small_signal=15
 # dB, gain of the amplifier guiding to the core
P_sat=0.08 # W, saturation power for the amplifier

x_slice=2*L/5 # position of the slice

"""
Properties of the input radiation
"""
Pin=5 # W, power launched through the taper

dv=0e6 ## 2*pi * Hz, detuning of the pump from the center of the cold resonance , in radian!
d_dv=1e6
dv_period=5e-4
x_0=L/2 # point where the center of the mode is  and where taper is

'''
Mode properties
'''
Gamma=1
delta_0=50e6 # Hz, spectral width of the resonance due to inner losses
delta_c=5e6 # Hz, spectral width of the resonance due to coupling

mode_width=0.1 #mm
def mode_distrib(x): # WGM mode distribution normilized as max(mode_distrib)=1
    return np.exp(-(x-x_0)**2/mode_width**2)


"""
grid parameters
"""

t_max=1e-6 # s
dv_max=20*(delta_0+delta_c)
N_dv=50

dx=0.05
dt = 1/(delta_c+delta_0)/20 # also should be no less than dx**2/2/beta
# dt=5e-11 # s

dt_large=1e-5 #s 
delta_t_to_save=dt # s

'''
Internal parameters
'''

N=int(L/dx) # number of numerical points along space axis
x = np.linspace(0, L, N+1)
T_0 = np.ones(N+1)*T0 #initial temperature distribution

frequency_0=c/wavelength_0
mu=3*frequency_0*hi_3/8/refractive_index**2*2 # (11.19), p.174 frjm Gorodetsky

Veff=integrate.quad(mode_distrib,-10,10)[0]*2*np.pi*r*2e-3 # effective volume of the WGM
mode_distrib_array=np.array(list(map(mode_distrib,x)))
mode_distrib_sum=np.sum(mode_distrib_array)

n_steps_to_save=(delta_t_to_save//dt)
n_steps_to_make_temperature_derivation=(dt_large//dt)

beta = thermal_conductivity/specific_heat_capacity/density
if dt>dx**2/beta/10:
    print('change dt to meet Heat transfer equation requirement')
    dt=dx**2/beta/10
    
gamma=heat_exchange/specific_heat_capacity/density*2/r
delta=sigma/specific_heat_capacity/density*2/r

modal_heat_const=epsilon_0*epsilon/2/1.5*c*absorption_in_silica
zeta=modal_heat_const/density/specific_heat_capacity/(np.pi*r**2)

theta=1/(specific_heat_capacity*np.pi*r**2*density)

alpha=absorption/4.34/1e3 # absorption in 1/mm for ln case
core_heating_constant=alpha*ESA_parameter*transmission_from_taper_to_amplifier
gain_small_signal_lin=10**(gain_small_signal/10)




def solve_model(Pin,dv,t_max,a=0,T=np.ones(N+1)*T0):
    F=np.sqrt(4*Pin*delta_c/epsilon_0/epsilon/Veff)
    N_t=int(t_max/dt)
    # Ensure that any list/tuple returned from f_ is wrapped as array
    rhs_thermal_array = lambda Pin,a,dv,T,T_averaged_over_mode, t: np.asarray(_rhs_thermal(Pin,a,dv,T,T_averaged_over_mode, t))
    t=0
    time_array=[]
    T_array=[]
    a_array=[]
    T_averaged_dynamics=[]
    time_start=time_module.time()
    T_averaged_over_mode=np.sum(T*mode_distrib_array)/mode_distrib_sum
    
    for n in range(N_t+1):
        
        t+=dt
        dv+=d_dv*np.sin(2*np.pi*t/dv_period)
        
        
        # a=a+dt*_rhs_modal(F,a,T_averaged_over_mode,t,dv)
        a=a+dt/6*Runge_Kutta_step(F,a,T_averaged_over_mode,t,dv)
        # a=_analytical_step_for_WGM_amplitude(F,a,T_averaged_over_mode,dt,dv)
                
 
        
        if abs(a)>1e10:
            print('unstable simulation. Detuning is too large')
            break
        
        if (n%n_steps_to_make_temperature_derivation)==1:
            T+=dt_large*rhs_thermal_array(Pin,a,dv,T,T_averaged_over_mode, t)
            T_averaged_over_mode=np.sum(T*mode_distrib_array)/mode_distrib_sum
        # test.append(heating_from_core(dv,L/2,du_average))
        # test.append(thermal_optical_responce*np.sum((u-T0)*mode_distrib_array/mode_distrib_sum))

        if (n%n_steps_to_save)==0:
            time_array.append(t)
            T_array.append(T)
            T_averaged_dynamics.append(T_averaged_over_mode)
            a_array.append(a)

        if (n%10000)==1:
           
            time=time_module.time()
            time_left=(time-time_start)*(N_t/n-1)
            print('step {} of {}, time left: {:.2f} s, or {:.2f} min'.format(n,N_t,time_left,time_left/60))
    return time_array,np.array(a_array),T,T_averaged_dynamics

def _rhs_modal(F,a,T_averaged_over_mode,t,dv):  # eq. (11.19), p 174 from Gorodetsky
    return 1j*F-a*(delta_c+delta_0)+a*1j*(thermal_optical_responce*(T_averaged_over_mode-T0)+dv)+ 1j*mu*a*abs(a)**2

def Runge_Kutta_step(F,a,T_averaged_over_mode,t,dv): # forward Runge_Kutta_ 4th order 
    k1=_rhs_modal(F,a,T_averaged_over_mode,t,dv)
    k2=_rhs_modal(F,a+k1*dt/2,T_averaged_over_mode,t+dt/2,dv)
    k3=_rhs_modal(F,a+k2*dt/2,T_averaged_over_mode,t+dt/2,dv)
    k4=_rhs_modal(F,a+k3*dt,T_averaged_over_mode,t+dt,dv)
    return k1+2*k2+2*k3+k4

def _analytical_step_for_WGM_amplitude(F,a,T_averaged_over_mode,t,dv):
    temp=1j*(thermal_optical_responce*(T_averaged_over_mode-T0)+dv)-delta_c-delta_0
    return np.exp(temp*t)*(a+1j*F/temp)-1j*F/temp

def _rhs_thermal(Pin,a,dv,T,T_averaged_over_mode, t):
    N = len(T) - 1
    rhs = np.zeros(N+1)
    rhs[0] = 0 #dsdt(t)
#    for i in range(1, N):
#        rhs[i] = (beta/dx**2)*(u[i+1] - 2*u[i] + u[i-1]) + \
#                 f(x[i], t)
    rhs[1:N] = (beta/dx**2)*(T[2:N+1] - 2*T[1:N] + T[0:N-1])+_heating_from_core(Pin,dv,x[1:N],T_averaged_over_mode)*theta + _heating_from_WGM(a,x[1:N], t)*zeta - (T[1:N]-T0)*gamma-(T[1:N]+273)**4*delta+(T0+273)**4*delta
    rhs[N] = (beta/dx**2)*(2*T[N-1]  -
                           2*T[N]) +_heating_from_core(Pin,dv,x[N],T_averaged_over_mode)*theta+ _heating_from_WGM(a,x[N], t)*zeta - (T[N]-T0)*gamma -(T[N]+273)**4*delta+(T0+273)**4*delta #+ 2*dx*dudx(t)
    return rhs




# def dsdt(): # derivative of the u at the left end 
#     return 0


def _heating_from_WGM(a,x, t): # source distribution
        return abs(a)**2*mode_distrib(x)
    
def transmission(dv,T_averaged_over_mode=T0):
    return 1-4*delta_c*delta_0*Gamma**2/((delta_c+delta_0)**2+(dv+(T_averaged_over_mode-T0)*thermal_optical_responce)**2)

def _amplifying_before_core(P):
    return gain_small_signal_lin**(1/(1+P/P_sat))

def _heating_from_core(Pin,dv,x,T_averaged_over_mode): # source distribution
    if np.size(x)>1:   
        output=np.zeros(np.shape(x))
        inds=np.where(x>x_slice)
        output[inds]=np.exp(-alpha*(x[inds]-x_slice))
        PintoCore=Pin*transmission(dv,T_averaged_over_mode)
        return output*core_heating_constant*PintoCore*_amplifying_before_core(PintoCore)
    else:
       return 0

  
def stationary_solution(Pin,dv):
    F=np.sqrt(4*Pin*delta_c/epsilon_0/epsilon/Veff)
    return np.sqrt(F**2/((delta_c+delta_0)**2+dv**2))


def find_spectral_response(Pin,dv_max=40*delta_c,N_dv=50,t_equilibr=t_max,direction='forward'):
    t_max_dv_first=t_equilibr
    t_max_dv=t_equilibr
    if direction=='forward':
        dv_array=np.linspace(-dv_max,dv_max,N_dv)
    if direction=='backward':
        dv_array=np.linspace(dv_max,-dv_max,N_dv)
    a_VS_dv=[]
    T=np.ones(N+1)*T0
    a=0
    for ii,dv in enumerate(dv_array):
        print('step={} of {} in {}'.format(ii,N_dv,direction))
        if ii==0:
            TimeArray,a_array,u,test = solve_model(Pin,dv,t_max_dv_first,a,T)
        else:
            TimeArray,a_array,u,test = solve_model(Pin,dv,t_max_dv,a,T)
        a=a_array[-1]
        a_VS_dv.append(a)
    return dv_array,np.array(a_VS_dv)







#%%

# timeArray,a_array,T,T_averaged_dynamics = solve_model(Pin,dv,t_max=t_max)
# fig=plt.figure(1)
# x=x-L/2
# plt.figure(1)
# # a=(a_array-np.mean(a_array))
# # a=a/np.max(a)*np.exp(1)
# plt.plot(timeArray,abs(a_array))
# plt.xlabel('Time, s')
# plt.ylabel('amplitude in the cavity, V/m')


# a_w=fft(a_array)
# N=len(a_array)
# xf = fftfreq(N, dt)[:N//2]
# plt.figure(2)
# plt.plot(xf, 2.0/N * np.abs(a_w[0:N//2])**2)

#%%
dv_array_forward,a_array_forward=find_spectral_response(Pin,t_equilibr=t_max,dv_max=dv_max,N_dv=N_dv,direction='forward')
dv_array_backward,a_array_backward=find_spectral_response(Pin,t_equilibr=t_max,dv_max=dv_max,N_dv=N_dv,direction='backward')

a_array_backward=abs(a_array_backward)
a_array_forward=abs(a_array_forward)
#%%
plt.figure(3)
plt.clf()
plt.plot(dv_array_forward,a_array_forward,label='forward')
plt.plot(dv_array_backward,a_array_backward,label='backward')
a_array_num=np.array(list(map(lambda dv:stationary_solution(Pin,dv),dv_array_forward)))
plt.plot(dv_array_forward,a_array_num,'.',label='no nonlinearities')
plt.legend()
plt.xlabel('detuning, Hz')
plt.ylabel('Amplitude in the cavity, V/m')
plt.title('Pin={:.3f},heat_from_mode={}, gain={:.2f}, transmission to amplifier={:.3f}'.format(Pin,bool(absorption_in_silica),gain_small_signal,transmission_from_taper_to_amplifier))
plt.savefig('Results\\Pin={:.3f},heat_from_mode={}, gain={:.2f}, transmission to amplifier={:.3f}.png'.format(Pin,bool(absorption_in_silica),gain_small_signal,transmission_from_taper_to_amplifier),dpi=300)
with open('Results\\results.pkl','wb') as f:
    pickle.dump([dv_array_forward,a_array_forward,dv_array_backward,a_array_backward],f)

