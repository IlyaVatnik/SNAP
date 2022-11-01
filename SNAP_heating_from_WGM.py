# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 14:14:19 2022

@author: User
Calculation of shift of the resonanca wavelngth due to heating by WGM
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import scipy.integrate as integrate
import time as time_module
import pickle
from numba import jit
import SNAP_model as sp

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

r=62.5e-3 #mm, fiber radius

specific_heat_capacity=740 # J/kg/K
density=2.2*1e-3*1e-3 # kg/mm**3
refractive_index=1.45
epsilon=refractive_index**2 #refractive index
absorption_in_silica=3.27e-09 #absorption in silica, 1/mm
thermal_optical_responce=1.25e9 # Hz/Celcium, detuning of the effective_ra
thermal_expansion_coefficient=0.0107*r*1e3 #nm/K for effective radius variation

hi_3=2.5e-16 # mm**2/ V
wavelength_0=1550e-6 # mm

k0=2*np.pi*refractive_index/(wavelength_0*1e-3) # in 1/mkm
"""
Sample parameters
"""
L=10 # mm, fiber sample length
T0=20 #Celsium, initial temperature of the sample being in equilibrium 

'''
Properties of the heating into the core
'''

absorption=8 #dB/m , absoprtion in the active core
ESA_parameter=0.15 # Excitated state absorption parameter, from  [Guzman-Chavez AD, Barmenkov YO, Kir’yanov A V. Spectral dependence of the excited-state absorption of erbium in silica fiber within the 1.48–1.59μm range. Appl Phys Lett 2008;92:191111. https://doi.org/10.1063/1.2926671.]


transmission_from_taper_to_amplifier=0.54*0  # parts, betwee the taper and amplifier
gain_small_signal=20
 # dB, gain of the amplifier guiding to the core
P_sat=0.025 # W, saturation power for the amplifier

x_slice=2*L/5 # position of the slice

"""
Properties of the input radiation
"""
Pin=0.01 # W, power launched through the taper

dv=80e6 ## Hz, detuning of the pump from the center of the cold resonance 
d_dv=0e6
dv_period=5e-4 # frequency of pump wavelength oscillating around detuning dv
x_0=L/3 # point where the center of the mode is  and where taper is

'''
Mode properties
'''
Gamma=-1
delta_0=100e6 # Hz*pi, spectral width of the resonance due to inner losses
delta_c=50e6 # Hz*pi, spectral width of the resonance due to coupling
phase=2 # Fano phase, in pi
mode_width=0.2 #mm
nonlinearity=False

def mode_distrib(x): # WGM mode distribution normilized as max(mode_distrib)=1
    return np.exp(-(x-x_0)**2/mode_width**2)


"""
grid parameters
"""

t_max=0.1 # s
dv_max=120*(delta_0+delta_c)
N_dv=150

dx=0.06 #mm
dt_large=5e-4 #s , for thermal step


if nonlinearity:
    dt = 1/(delta_c+delta_0)/10 # also should be no less than dx**2/2/beta
else:
    #dt=5e-5
    dt=dt_large
# dt=5e-11 # s

delta_t_to_save=dt_large*5 # s

'''
Internal parameters
'''

N=int(L/dx) # number of numerical points along space axis
x = np.linspace(0, L, N+1)
T_0 = np.ones(N+1)*T0 #initial temperature distribution

frequency_0=c/wavelength_0
if nonlinearity:
    mu=3*frequency_0*hi_3/8/refractive_index**2*2 # (11.19), p.174 frjm Gorodetsky
else:
    mu=0

#Veff=integrate.quad(mode_distrib,-10,10)[0]*2*np.pi*r*2e-3 # effective volume of the WGM
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
#zeta=modal_heat_const/density/specific_heat_capacity/(np.pi*r**2)
zeta=epsilon_0*1.45*3e8*absorption_in_silica*1e3*6.3e-9/(2*specific_heat_capacity*density*np.pi*62.5**2*1e-15) #?????в каких это должно быть единицах?

theta=1/(specific_heat_capacity*np.pi*r**2*density)

alpha=absorption/4.34/1e3 # absorption in 1/mm for ln case
core_heating_constant=alpha*ESA_parameter*transmission_from_taper_to_amplifier
gain_small_signal_lin=10**(gain_small_signal/10)




def solve_model(Pin,dv,t_max,psi, a_initial=0,T=np.ones(N+1)*T0,is_first=False):
    F=0
    N_t=int(t_max/dt)
    # Ensure that any list/tuple returned from f_ is wrapped as array
    rhs_thermal_array = lambda Pin,A,T,t,dv,is_first: np.asarray(_rhs_thermal(Pin,A,T, t, psi,is_first))
    t=0
    time_array=[]
    a_array=[]
    a=a_initial
    T_averaged_dynamics=[]
    time_start=time_module.time()
    T_averaged_over_mode=np.sum(T*mode_distrib_array)/mode_distrib_sum
    if is_first==False:
        A=np.zeros(psi[:,0].shape,dtype=complex)
    else:
        A=0
    for n in range(N_t+1):
        t+=dt
        if nonlinearity:
            dv+=d_dv*np.sin(2*np.pi*t/dv_period)
            a=a+dt/6*Runge_Kutta_step(F,a,T_averaged_over_mode,t,dv)
        else:
            if is_first==False:
                for i,k in enumerate(psi):
                    F=np.sqrt(4*Pin*delta_c/epsilon_0/epsilon/Veff(k))
                    #print(F)
                    A[i]=_analytical_step_for_WGM_amplitude(F,A[i],t,dv[i])
                    #A[i]=A[i]+dt/6*Runge_Kutta_step(F,A[i],dv[i])
            else: 
                Vef=integrate.quad(mode_distrib,-10,10)[0]*2*np.pi*r*2e-3
                F=np.sqrt(4*Pin*delta_c/epsilon_0/epsilon/Vef)
                A=_analytical_step_for_WGM_amplitude(F,A,dt,0)
                
        if (n%n_steps_to_make_temperature_derivation)==0:
            T+=dt_large*rhs_thermal_array(Pin, A, T, t, psi, is_first)

        if (n%50000)==1:
           
            time=time_module.time()
            time_left=(time-time_start)*(N_t/n-1)
            print('step {} of {}, time left: {:.2f} s, or {:.2f} min'.format(n,N_t,time_left,time_left/60))
    return time_array,a_array,T,T_averaged_dynamics


@jit(nopython=True)
def _rhs_modal(F,a,dv):  # eq. (11.19), p 174 from Gorodetsky
    return 1j*F-a*(delta_c+delta_0)+a*1j*dv

@jit(nopython=True)
def Runge_Kutta_step(F,a,dv): # forward Runge_Kutta_ 4th order 
    k1=_rhs_modal(F,a,dv)
    k2=_rhs_modal(F,a+k1*dt/2,dv)
    k3=_rhs_modal(F,a+k2*dt/2,dv)
    k4=_rhs_modal(F,a+k3*dt,dv)
    return k1+2*k2+2*k3+k4

@jit(nopython=True)
def _analytical_step_for_WGM_amplitude(F,a,t,dv):
    temp=1j*(dv)-delta_c-delta_0
    return np.exp(temp*t)*(a+1j*F/temp)-1j*F/temp

# @jit(nopython=True)
def _rhs_thermal(Pin,a,T,t,psi,is_first=False):
    N = len(T) - 1
    rhs = np.zeros(N+1)
    rhs[0] = 0 #dsdt(t)
    if is_first==False:
        rhs[1:N] = (beta/dx**2)*(T[2:N+1] - 2*T[1:N] + T[0:N-1]) - (T[1:N]-T0)*gamma-(T[1:N]+273)**4*delta+(T0+273)**4*delta +_heating_from_WGM(a, psi[:,1:N], True)*zeta
        rhs[N] = (beta/dx**2)*(2*T[N]  -
                           2*T[N-1]) - (T[N]-T0)*gamma -(T[N]+273)**4*delta+(T0+273)**4*delta #+ _heating_from_WGM(a, psi[:,N],True)*zeta #+ 2*dx*dudx(t)
    else:
        rhs[1:N] = (beta/dx**2)*(T[2:N+1] - 2*T[1:N] + T[0:N-1]) - (T[1:N]-T0)*gamma-(T[1:N]+273)**4*delta+(T0+273)**4*delta +_heating_from_WGM(a, psi[1:N])*zeta
        rhs[N] = (beta/dx**2)*(2*T[N]  -
                           2*T[N-1]) - (T[N]-T0)*gamma -(T[N]+273)**4*delta+(T0+273)**4*delta + _heating_from_WGM(a, psi[N])*zeta 
    return rhs

def Veff (psi):
    #print(np.sum(psi)*dx*2*np.pi*r*2e-3)
    return np.sum(psi)*dx*2*np.pi*r*2e-3 # effective volume of the WGM


# def dsdt(): # derivative of the u at the left end 
#     return 0

# @jit(nopython=True)
# def _heating_from_WGM(a,x): # source distribution
#     return abs(a)**2*mode_distrib(x)

def _heating_from_WGM(a,psi,is_array=False):
    sum_a=0
    if is_array==True:
        for i,_ in enumerate(a):
            # print(a.shape)
            # print(psi.shape)
            try:
                sum_a+=(abs(a[i])**2)*(psi[i,:])**2
                #print(i)
            except:
                sum_a+=(abs(a[i])**2)*(psi)**2
            if np.isnan(sum_a.any()):
                break
    else: sum_a=(abs(a)**2)*abs(psi)**2
    return sum_a

  
def stationary_solution(Pin,dv):
    F=np.sqrt(4*Pin*delta_c/epsilon_0/epsilon/Veff)
    return np.sqrt(F**2/((delta_c+delta_0)**2+dv**2))


def SNAP_spectrogramm(x,ERV,lambda_array,lambda_0=1552.21,res_width=1e-4,R_0=62.5):
    SNAP=sp.SNAP(x,ERV,lambda_array,lambda_0=lambda_0,res_width=1e-4,R_0=62.5)
    SNAP.set_taper_params(absS=np.sqrt(0.8),phaseS=0.0,ReD=0.00,ImD_exc=2e-3,Csquared=0.001)
    return SNAP.plot_spectrogram(plot_ERV=True,scale='log')

def Shrodinger_solution (x,ERV,lambda_array,lambda_0=1552.21,res_width=1e-4,R_0=62.5):
    U=-2*k0**2*ERV*(1e-3)/R_0
    SNAP=sp.SNAP(x,ERV,lambda_array,lambda_0=lambda_0,res_width=1e-4,R_0=62.5)
    Psi_values, Psi_vector=SNAP.solve_Shrodinger(U)
    return Psi_values, Psi_vector

def detuning (E, lambda_0=1550e-6):
    wl = E/(2*k0**2)*lambda_0+lambda_0    
    return c*(1/wl-1/lambda_0)

def psi_normalization(psi):
    #print(p_n.shape)
    for i,l in enumerate(psi):
        psi[i]=(abs(l)**2/(max(abs(psi[0,:])**2)))
    return psi

#%%
#P=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06,0.07,0.08,0.09,0.1]
# P=0.1
# ERV=[]
# ERV_max=[]
#timeArray,a_array,T,T_averaged_dynamics = solve_model(P,dv,t_max=t_max,a_initial=0)
#ERV=((T-T0)*thermal_expansion_coefficient)

# fig=plt.figure(2)
# plt.xlabel('position, mm')
# plt.ylabel('ERV, nm')
# for i in P:
#     timeArray,a_array,T,T_averaged_dynamics = solve_model(i,dv,t_max=t_max,a_initial=0)
#     legend='P='+str(i)+' mW'
#     plt.plot(x/L/2,(T-T0)*thermal_expansion_coefficient, label=legend)
#     ERV_max.append(max((T-T0)*thermal_expansion_coefficient))
#     ERV.append((T-T0)*thermal_expansion_coefficient)
# plt.legend()

# plt.figure(3)
# plt.xlabel('P, mW')
# plt.ylabel('max ERV, nm')
# plt.plot(P, ERV_max,'bo-')



#timeArray,a_array,T,T_averaged_dynamics = solve_model(Pin,dv,t_max=t_max,a_initial=0)
# fig=plt.figure(1)
# plt.plot(x-L/2,T)
# plt.xlabel('position, mm')
# plt.ylabel('Temperature, $^0$C')

# plt.figure(2)
# plt.plot(x/L/2,(T-T0)*thermal_expansion_coefficient)
# plt.xlabel('position, mm')
# plt.ylabel('ERV, nm')
# plt.figure(3)
# plt.plot(timeArray,T_averaged_dynamics)
# plt.xlabel('Time, s')
# plt.ylabel('Mode temperature, $^0C$')

#%%iteration solution process
P=0.1
T=np.ones(N+1)*T0 #temperature along cavity
mode_dis=mode_distrib(x)
#_,_,T,_ = solve_model(P, 0, t_max, mode_dis, a_initial=0,is_first=True)
#print(T) 
#plt.plot(x,mode_dis)
ERV=np.ones(T.shape)*mode_dis
#ERV=((T-T0)*thermal_expansion_coefficient)
#plt.plot(x,ERV) 
lambda_0=1550.0
wave_min,wave_max,res=1549.9,1550.1, 1e-4    
lambda_array=np.arange(wave_min,wave_max,res)
SNAP_spectrogramm(x*1e3,ERV,lambda_array,lambda_0=lambda_0,res_width=1e-5,R_0=62.5)
for i in range(2):
    E, Psi = Shrodinger_solution(x*1e3,ERV,lambda_array,lambda_0=lambda_0,res_width=1e-5,R_0=62.5)
    delta_v=detuning(E)
    psi=psi_normalization(Psi)
    plt.plot(x,psi[0,:])
    _,_,T,_ = solve_model(P, delta_v, 0.03, psi, a_initial=0, is_first=False)#a_initial должна быть нулём на каждом шаге? 
    
    ERV=((T-T0)*thermal_expansion_coefficient)
    print(ERV) #часть значений nan, возможно из-за большого шага по времени 
SNAP_spectrogramm(x*1e3,ERV,lambda_array,lambda_0=lambda_0,res_width=1e-5,R_0=62.5)
plt.plot(x,ERV) 
    

#%%

#print('Time spent={} s'.format(time.time()-time0))