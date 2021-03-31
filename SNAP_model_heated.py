# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 21:11:01 2021

Modification 31.03.2021

Time-dependent numerical solution for temperature distribution along the fiber under local heating with WGM mode
Following Gorodecky, p.313
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import scipy.integrate as integrate
import time
import dill


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
thermal_optical_responce=1.25e9 # Hz/Celcium, detuning of the effective_ra

"""
Sample parameters
"""
L=15 # mm, fiber sample length
T0=20 #K, initial temperature of the sample being in equilibrium 

'''
Properties of the heating into the core
'''

absorption=8 #dB/m , absoprtion in the active core
ESA_parameter=0.15 # Excitated state absorption parameter, from  [Guzman-Chavez AD, Barmenkov YO, Kir’yanov A V. Spectral dependence of the excited-state absorption of erbium in silica fiber within the 1.48–1.59μm range. Appl Phys Lett 2008;92:191111. https://doi.org/10.1063/1.2926671.]
# thermal_expansion_coefficient=0.0107*r*1e3 #  nm/K, for effective radius variation

non_resonant_transmission_at_taper_and_splice_with_active_core=0.0
 # parts
gain_small_signal=15 # dB
P_sat=0.08 # W

"""
Properties of the resonance and input power
"""
Pin=0.01 # W, power launched through the taper
dw=+50e6 ## Hz, detuning of the pump from the center of the resonance
delta_c=100e6 # Hz, spectral width of the resonance due to coupling
delta_0=100e6 # Hz, spectral width of the resonance due to ineer losses
x_0=L/2 # point where the center of the mode is  and where taper is

# =============================================================================
# Mode distribution
# =============================================================================
mode_length=0.2 #mm
def mode_distrib(x): # normilized as max(mode_distrib)=1
    return np.exp(-(x-x_0)**2/mode_length**2)


"""
grid parameters
"""
dx=0.05
# dt = 1/delta_c/6 # also should be no less than dx**2/2/beta
dt=1e-3
T_max=0.3

'''
Internal parameters
'''

N=int(L/dx) # number of numerical points along space axis
x = np.linspace(0, L, N+1)
u = np.zeros(N+1) # temperature
U_0 = np.ones(N+1)*T0 #initial temperature distribution



Veff=integrate.quad(mode_distrib,-10,10)[0]*2*np.pi*r*2e-3
mode_distrib_array=np.array(list(map(mode_distrib,x)))
mode_distrib_sum=np.sum(mode_distrib_array)

beta = thermal_conductivity/specific_heat_capacity/density
if dt>dx**2/beta/10:
    print('change dt to meet Heat transfer equation requirement')
    dt=dx**2/beta/10
    
gamma=heat_exchange/specific_heat_capacity/density*2/r
delta=sigma/specific_heat_capacity/density*2/r

modal_heat_const=epsilon_0*epsilon/2/1.5*c*absorption_in_silica
dzeta=modal_heat_const/density/specific_heat_capacity/(np.pi*r**2)

theta=1/(specific_heat_capacity*np.pi*r**2*density)

alpha=absorption/4.34/1e3 # absorption in 1/mm for ln case

F=np.sqrt(4*Pin*delta_c/epsilon_0/epsilon/Veff)
core_heating_constant=alpha*ESA_parameter*non_resonant_transmission_at_taper_and_splice_with_active_core
gain_small_signal_lin=10**(gain_small_signal/10)


def ode_FE(dw,T_max,a=0,u=np.ones(N+1)*T0):
    N_t=int(T_max/dt)
    Indexes_to_save=[N_t]
    # Ensure that any list/tuple returned from f_ is wrapped as array
    rhs_thermal_array = lambda a,u,du_average, t: np.asarray(rhs_thermal(a,u,du_average, t))
    t=0
    u_array=[]
    a_array=[]
    test=[]
    TimeArray=np.linspace(0,dt*N_t,N_t+1)
    for n in range(N_t+1):
        t=t+dt
        # a=a+dt*rhs_modal(a,u,t,dw)
        du_average=np.sum((u-T0)*mode_distrib_array)/mode_distrib_sum
        a=analytical_step_for_WGM_amplitude(a,du_average,dt,dw)
        test.append(du_average)
        if abs(a)>1e10:
            print('unstable simulation. Detuning is too large')
            TimeArray=np.linspace(0,dt*n,n)
            break
        a_array.append(abs(a))
        # if (n%100)==0:
        u = u + dt*rhs_thermal_array(a,u,du_average, t)
        # test.append(heating_from_core(dw,L/2,du_average))
        # test.append(thermal_optical_responce*np.sum((u-T0)*mode_distrib_array/mode_distrib_sum))
        if n in Indexes_to_save:
            u_array.append(u)
        if (n%10000)==0:
            print('step ', n,' of ', N_t)
    return TimeArray,a_array,u,test

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
    rhs[1:N] = (beta/dx**2)*(u[2:N+1] - 2*u[1:N] + u[0:N-1])+heating_from_core(dw,x[1:N],du_average)*theta + heating_from_WGM(a,x[1:N], t)*dzeta - (u[1:N]-T0)*gamma-(u[1:N]+273)**4*delta+(T0+273)**4*delta
    rhs[N] = (beta/dx**2)*(2*u[N-1]  -
                           2*u[N]) +heating_from_core(dw,x[N],du_average)*theta+ heating_from_WGM(a,x[N], t)*dzeta - (u[N]-T0)*gamma -(u[N]+273)**4*delta+(T0+273)**4*delta #+ 2*dx*dudx(t)
    return rhs


# def dsdt(): # derivative of the u at the left end 
#     return 0


def heating_from_WGM(a,x, t): # source distribution
        return abs(a)**2*mode_distrib(x)
    
def transmission(dw,u_average=0):
    return 1-4*delta_c*delta_0/((delta_c+delta_0)**2+(dw+(u_average-T0)*thermal_optical_responce)**2)

def amplifying_before_core(P):
    return gain_small_signal_lin**(1/(1+P/P_sat))

def heating_from_core(dw,x,du_average): # source distribution
    if np.size(x)>1:   
        output=np.zeros(np.shape(x))
        inds=np.where(x>x_0)
        output[inds]=np.exp(-alpha*(x[inds]-x_0))
        PintoCore=Pin*transmission(dw,du_average)
        return output*core_heating_constant*PintoCore*amplifying_before_core(PintoCore)
    else:
        return 0

  
def stationary_solution(dw):
    return np.sqrt(F**2/((delta_c+delta_0)**2+dw**2))


def find_spectral_responce(direction='forward'):
    T_max_dw_first=T_max
    T_max_dw=T_max
    dw_max=40*delta_c
    N_dw=50
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
            TimeArray,a_array,u,test = ode_FE(dw,T_max_dw_first,a,u)
        else:
            TimeArray,a_array,u,test = ode_FE(dw,T_max_dw,a,u)
        a=a_array[-1]
        a_VS_dw.append(a)
    return dw_array,np.array(a_VS_dw)




time0=time.time()


#%%
# TimeArray,a_array,u,test = ode_FE(dw,T_max)
# fig=plt.figure(1)
# x=x-L/2
# plt.plot(x,u)
# plt.xlabel('position, mm')
# plt.ylabel('Temperature, $^0$C')
# plt.figure(2)
# plt.plot(TimeArray,a_array)
# plt.xlabel('Time, s')
# plt.ylabel('amplitude in the cavity, V/m')
# plt.figure(3)
# plt.plot(TimeArray,test)
# plt.xlabel('Time, s')
# plt.ylabel('Mode temperature, $^0C$')


#%%
dw_array_forward,a_array_forward=find_spectral_responce()
dw_array_backward,a_array_backward=find_spectral_responce('backward')
plt.figure(3)
plt.plot(dw_array_forward,a_array_forward,label='forward')
plt.plot(dw_array_backward,a_array_backward,label='backward')
a_array_num=np.array(list(map(stationary_solution,dw_array_forward)))
plt.plot(dw_array_forward,a_array_num,'.',label='no nonlinearities')
plt.legend()
plt.xlabel('detuning, Hz')
plt.ylabel('Amplitude in the cavity, V/m')

# fig=plt.figure(1)
# for ind,t in enumerate(Times):
#     p=plt.plot(x,Uarray[ind,:],label=str(t))
#     ax=plt.gca()

# plt.legend()
# plt.xlabel('Distance, mm')
# plt.ylabel('Temperature, C')
# plt.savefig('Distributions for P='+str(Pin)+ '%, T='+str(Times)+'.png')

# #%%
# plt.figure(4)
# N_dw=30
# dw_max=4*delta_c
# dw_array=np.linspace(dw_max,-dw_max,N_dw)
# plt.plot(dw_array,list(map(lambda w:transmission(w,T0),dw_array)))

# plt.figure(5)
# P_array=np.linspace(0,0.1,20)
# plt.plot(P_array,list(map(amplifying_before_core,P_array)))
print('Time spent={} s'.format(time.time()-time0))
dill.dump_session('Session Pin={} gain={}.pkl'.format(Pin,gain_small_signal))
