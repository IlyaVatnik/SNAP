# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 21:11:01 2021

Time-dependent numerical solution for temperature distribution along the fiber under local heating with WGM mode
Following Gorodecky, p.313
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import scipy.integrate as integrate
import time


'''
Constants
'''
epsilon_0=8.85418781762039e-15 #F/mm, dielectric constant
epsilon=1.5**2
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
absorption_in_silica=3.27e-09 #absorption in silica, 1/mm

thermal_optical_responce=1.25e9 # Hz/Celcium

"""
Sample parameters
"""
L=15 # mm, fiber sample length
T0=25 #K

'''
Properties of the heating into the core
'''
splice_transmission=0.87
absorption=8 #dB/m
ESA_parameter=0.15 # Excitated state absorption parameter, from  [Guzman-Chavez AD, Barmenkov YO, Kir’yanov A V. Spectral dependence of the excited-state absorption of erbium in silica fiber within the 1.48–1.59μm range. Appl Phys Lett 2008;92:191111. https://doi.org/10.1063/1.2926671.]
thermal_expansion_coefficient=0.0107*r*1e3 #  nm/K, for effective radius variation

losses_at_taper=0.8
gain_small_signal=80
P_sat=0.08

"""
Properties of the resonance and input power
"""
Pin=0.03 # W, power launched through the taper
dw=-10e6 ## 1/s, detuning of the pump from the center of the resonance
delta_c=100e6 #1/s, spectral width of the resonance due to coupling
delta_0=10e6 #1/s, spectral width of the resonance due to ineer losses
x_0=L/2 # point where the center of the mode is  and where taper is

# =============================================================================
# Mode
# =============================================================================
mode_length=0.1 #mm
def mode_distrib(x): # normilized as max(mode_distrib)=1
    return np.exp(-(x-x_0)**2/mode_length**2)


"""
grid parameters
"""
dx=0.05
# dt = 1/delta_c/6 # also should be no less than dx**2/2/beta
dt=1e-4

'''
Internal parameters
'''

N=int(L/dx) # number of numerical points along space axis
x = np.linspace(0, L, N+1)
u = np.zeros(N+1) # temperature
U_0 = np.ones(N+1)*T0 #initial temperature distribution



Veff=integrate.quad(mode_distrib,-10,10)[0]*2*np.pi*r*1.5e-3
mode_distrib_array=np.array(list(map(mode_distrib,x)))
mode_distrib_sum=np.sum(mode_distrib_array)

beta = thermal_conductivity/specific_heat_capacity/density
if dt>dx**2/2/beta/5:
    print('change dt')
    dt=dx**2/2/beta/5
    
gamma=heat_exchange/specific_heat_capacity/density*2/r
delta=sigma/specific_heat_capacity/density*2/r

modal_heat_const=epsilon_0*epsilon/2/1.5*c*absorption_in_silica
gamma1=modal_heat_const/density/specific_heat_capacity

omicron=1/(specific_heat_capacity*np.pi*r**2*density)
alpha=absorption/4.34/1e3 # absorption in 1/mm for ln case

F=np.sqrt(4*Pin*delta_c/epsilon_0/epsilon/Veff)
core_heating_constant=alpha*omicron*ESA_parameter*splice_transmission*losses_at_taper

def ode_FE(dw,T_max,a=0,u=np.ones(N+1)*T0):
    N_t=int(T_max/dt)
    Indexes_to_save=[N_t]
    # Ensure that any list/tuple returned from f_ is wrapped as array
    rhs_thermal_array = lambda a,u,du_average, t: np.asarray(rhs_thermal(a,u,du_average, t))
    t=0
    u_array=[]
    a_array=[]
    TimeArray=np.linspace(0,dt*N_t,N_t+1)
    for n in range(N_t+1):
        t=t+dt
        # a=a+dt*rhs_modal(a,u,t,dw)
        du_average=np.sum((u-T0)*mode_distrib_array)/mode_distrib_sum
        a=analytical_step_for_WGM_amplitude(a,du_average,dt,dw)
        
        if abs(a)>1e10:
            print('unstable simulation. Detuning is too large')
            TimeArray=np.linspace(0,dt*n,n)
            break
        a_array.append(abs(a))
        # if (n%100)==0:
        u = u + dt*rhs_thermal_array(a,u,du_average, t)
        # test.append(thermal_optical_responce*np.sum((u-T0)*mode_distrib_array/mode_distrib_sum))
        if n in Indexes_to_save:
            u_array.append(u)
        if (n%10000)==0:
            print('step ', n,' of ', N_t)

    
    return TimeArray,a_array,u

def rhs_modal(a,du_average,t,dw):
    return 1j*F-a*(delta_c+delta_0)+a*1j*(thermal_optical_responce*du_average+dw)    

def analytical_step_for_WGM_amplitude(a,du_average,t,dw):
    temp=1j*(thermal_optical_responce*du_average+dw)-delta_c-delta_0
    return np.exp(temp*t)*(a+1j*F/temp)-1j*F/temp

def rhs_thermal(a,u,du_average, t):
    N = len(u) - 1
    rhs = np.zeros(N+1)
    rhs[0] = 0 #dsdt(t)
#    for i in range(1, N):
#        rhs[i] = (beta/dx**2)*(u[i+1] - 2*u[i] + u[i-1]) + \
#                 f(x[i], t)
    rhs[1:N] = (beta/dx**2)*(u[2:N+1] - 2*u[1:N] + u[0:N-1])+heating_from_core(x,dw,du_average) + heating_from_WGM(a,x[1:N], t)*gamma1 - (u[1:N]-T0)*gamma-(u[1:N]+273)**4*delta+(T0+273)**4*delta
    rhs[N] = (beta/dx**2)*(2*u[N-1]  -
                           2*u[N]) +heating_from_core(x,dw,du_average)+ heating_from_WGM(a,x[N], t)*gamma1 - (u[N]-T0)*gamma -(u[N]+273)**4*delta+(T0+273)**4*delta #+ 2*dx*dudx(t)
    return rhs


# def dsdt(): # derivative of the u at the left end 
#     return 0


def heating_from_WGM(a,x, t): # source distribution
        return abs(a)**2*mode_distrib(x)
    
def transmission(dw,u_average=0):
    return (1-4*delta_c*delta_0/(delta_c**2+delta_0**2)+(dw+(u_average-T0)*thermal_optical_responce)**2)

def amplifying_before_core(P):
    return np.exp(gain_small_signal/(1+P/P_sat))

def heating_from_core(dw,x,du_average): # source distribution
    if np.size(x)>1:
        output=np.zeros(np.shape(x))
        inds=np.where(x>x_0)
        output[inds]=np.exp(-alpha*(x[inds]-x_0))
        return output*core_heating_constant*Pin*transmission(dw,du_average)*amplifying_before_core(Pin)
    else:
        return 0


def find_spectral_responce(direction='forward'):
    T_max_first=1e-2
    T_max=2e8/delta_c
    dw_max=5*delta_c
    N_dw=40
    if direction=='forward':
        dw_array=np.linspace(-dw_max,dw_max,N_dw)
    if direction=='backward':
        dw_array=np.linspace(dw_max,-dw_max,N_dw)
    a_VS_dw=[]
    u=np.ones(N+1)*T0
    a=0
    for ii,dw in enumerate(dw_array):
        print('step={} of {} in {}'.format(ii,N_dw,direction))
        if ii==0:
            TimeArray,a_array,u = ode_FE(dw,T_max_first,a,u)
        else:
            TimeArray,a_array,u = ode_FE(dw,T_max,a,u)
        a=a_array[-1]
        a_VS_dw.append(a)
    return dw_array,np.array(a_VS_dw)

  
def stationary_solution(dw):
    return np.sqrt(F**2/((delta_c+delta_0)**2+dw**2))

time0=time.time()

# TimeArray,a_array,u = ode_FE(dw,T_max=0.2)
# fig=plt.figure(1)
# x=x-L/2
# plt.plot(x,u)
# plt.figure(2)
# plt.plot(TimeArray,a_array)

# u,Uarray, TimeArray,a_array,test = ode_FE(dw,T_max)
# plt.figure(2)
# plt.plot(TimeArray,a_array)
# plt.figure(22)
# plt.plot(TimeArray,test)

dw_array_forward,a_array_forward=find_spectral_responce()
dw_array_backward,a_array_backward=find_spectral_responce('backward')
plt.figure(3)
plt.plot(dw_array_forward,a_array_forward,label='forward')
plt.plot(dw_array_backward,a_array_backward,label='backward')
a_array_num=np.array(list(map(stationary_solution,dw_array_forward)))
plt.plot(dw_array_forward,a_array_num,'.',label='no nonlinearities')
plt.legend()
plt.xlabel('detuning, Hz')
plt.ylabel('Amplitude in the cavity')

# fig=plt.figure(1)
# for ind,t in enumerate(Times):
#     p=plt.plot(x,Uarray[ind,:],label=str(t))
#     ax=plt.gca()

# plt.legend()
# plt.xlabel('Distance, mm')
# plt.ylabel('Temperature, C')
# plt.savefig('Distributions for P='+str(Pin)+ '%, T='+str(Times)+'.png')

print(time.time()-time0)
