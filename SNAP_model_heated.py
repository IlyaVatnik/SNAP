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



'''
Constants
'''
epsilon_0=8.85418781762039e-15 #F/mm, dielectric constant
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

epsilon=1.5**2 #refractive index
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

transmission_from_taper_to_amplifier=0.2 # parts, betwee the taper and amplifier
gain_small_signal=15 # dB, gain of the amplifier guiding to the core
P_sat=0.08 # W, saturation power for the amplifier

"""
Properties of the input radiation
"""
Pin=0.01 # W, power launched through the taper
dv=-100e6 ## Hz, detuning of the pump from the center of the cold resonance 
d_dv=1*300e6
dv_period=5e-4
x_0=L/2 # point where the center of the mode is  and where taper is

'''
Mode properties
'''
delta_c=100e6 # Hz, spectral width of the resonance due to coupling
delta_0=100e6 # Hz, spectral width of the resonance due to inner losses


'''
Mode distribution
'''
mode_width=0.2 #mm
def mode_distrib(x): # WGM mode distribution normilized as max(mode_distrib)=1
    return np.exp(-(x-x_0)**2/mode_width**2)


"""
grid parameters
"""
dx=0.05
# dt = 1/delta_c/6 # also should be no less than dx**2/2/beta
dt=1e-5 # s
T_max=0.3 # s

'''
Internal parameters
'''

N=int(L/dx) # number of numerical points along space axis
x = np.linspace(0, L, N+1)
u = np.zeros(N+1) # temperature
U_0 = np.ones(N+1)*T0 #initial temperature distribution



Veff=integrate.quad(mode_distrib,-10,10)[0]*2*np.pi*r*2e-3 # effective volume of the WGM
mode_distrib_array=np.array(list(map(mode_distrib,x)))
mode_distrib_sum=np.sum(mode_distrib_array)

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


def solve_model(Pin,dv,T_max,a=0,u=np.ones(N+1)*T0):
    F=np.sqrt(4*Pin*delta_c/epsilon_0/epsilon/Veff)
    N_t=int(T_max/dt)
    Indexes_to_save=[N_t]
    # Ensure that any list/tuple returned from f_ is wrapped as array
    rhs_thermal_array = lambda Pin,a,dv,u,du_average, t: np.asarray(_rhs_thermal(Pin,a,dv,u,du_average, t))
    t=0
    u_array=[]
    a_array=[]
    test=[]
    TimeArray=np.linspace(0,dt*N_t,N_t+1)
    for n in range(N_t+1):
        t=t+dt
        dv+=d_dv*np.sin(2*np.pi*t/dv_period)
        # a=a+dt*_rhs_modal(F,a,u,t,dv)
        du_average=np.sum((u-T0)*mode_distrib_array)/mode_distrib_sum
        a=_analytical_step_for_WGM_amplitude(F,a,du_average,dt,dv)
        test.append(du_average)
        if abs(a)>1e10:
            print('unstable simulation. Detuning is too large')
            TimeArray=np.linspace(0,dt*n,n)
            break
        a_array.append(abs(a))
        # if (n%100)==0:
        u = u + dt*rhs_thermal_array(Pin,a,dv,u,du_average, t)
        # test.append(heating_from_core(dv,L/2,du_average))
        # test.append(thermal_optical_responce*np.sum((u-T0)*mode_distrib_array/mode_distrib_sum))
        if n in Indexes_to_save:
            u_array.append(u)
        if (n%10000)==0:
            print('step ', n,' of ', N_t)
    return TimeArray,a_array,u,test

def _rhs_modal(F,a,du_average,t,dv):
    return 1j*F-a*(delta_c+delta_0)+a*1j*(thermal_optical_responce*du_average+dv)    

def _analytical_step_for_WGM_amplitude(F,a,du_average,t,dv):
    temp=1j*(thermal_optical_responce*du_average+dv)-delta_c-delta_0
    return np.exp(temp*t)*(a+1j*F/temp)-1j*F/temp

def _rhs_thermal(Pin,a,dv,u,du_average, t):
    N = len(u) - 1
    rhs = np.zeros(N+1)
    rhs[0] = 0 #dsdt(t)
#    for i in range(1, N):
#        rhs[i] = (beta/dx**2)*(u[i+1] - 2*u[i] + u[i-1]) + \
#                 f(x[i], t)
    rhs[1:N] = (beta/dx**2)*(u[2:N+1] - 2*u[1:N] + u[0:N-1])+_heating_from_core(Pin,dv,x[1:N],du_average)*theta + _heating_from_WGM(a,x[1:N], t)*zeta - (u[1:N]-T0)*gamma-(u[1:N]+273)**4*delta+(T0+273)**4*delta
    rhs[N] = (beta/dx**2)*(2*u[N-1]  -
                           2*u[N]) +_heating_from_core(Pin,dv,x[N],du_average)*theta+ _heating_from_WGM(a,x[N], t)*zeta - (u[N]-T0)*gamma -(u[N]+273)**4*delta+(T0+273)**4*delta #+ 2*dx*dudx(t)
    return rhs


# def dsdt(): # derivative of the u at the left end 
#     return 0


def _heating_from_WGM(a,x, t): # source distribution
        return abs(a)**2*mode_distrib(x)
    
def transmission(dv,u_average=0):
    return 1-4*delta_c*delta_0/((delta_c+delta_0)**2+(dv+(u_average-T0)*thermal_optical_responce)**2)

def _amplifying_before_core(P):
    return gain_small_signal_lin**(1/(1+P/P_sat))

def _heating_from_core(Pin,dv,x,du_average): # source distribution
    if np.size(x)>1:   
        output=np.zeros(np.shape(x))
        inds=np.where(x>x_0)
        output[inds]=np.exp(-alpha*(x[inds]-x_0))
        PintoCore=Pin*transmission(dv,du_average)
        return output*core_heating_constant*PintoCore*_amplifying_before_core(PintoCore)
    else:
        return 0

  
def stationary_solution(Pin,dv):
    F=np.sqrt(4*Pin*delta_c/epsilon_0/epsilon/Veff)
    return np.sqrt(F**2/((delta_c+delta_0)**2+dv**2))


def find_spectral_response(Pin,dv_max=40*delta_c,N_dv=50,T_equilibr=T_max,direction='forward'):
    T_max_dv_first=T_equilibr
    T_max_dv=T_equilibr
    if direction=='forward':
        dv_array=np.linspace(-dv_max,dv_max,N_dv)
    if direction=='backward':
        dv_array=np.linspace(dv_max,-dv_max,N_dv)
    a_VS_dv=[]
    u=np.ones(N+1)*T0
    a=0
    for ii,dv in enumerate(dv_array):
        print('step={} of {} in {}'.format(ii,N_dv,direction))
        if ii==0:
            TimeArray,a_array,u,test = solve_model(Pin,dv,T_max_dv_first,a,u)
        else:
            TimeArray,a_array,u,test = solve_model(Pin,dv,T_max_dv,a,u)
        a=a_array[-1]
        a_VS_dv.append(a)
    return dv_array,np.array(a_VS_dv)




time0=time.time()


#%%
TimeArray,a_array,u,test = solve_model(Pin,dv,T_max=3)
fig=plt.figure(1)
x=x-L/2
plt.plot(x,u)
plt.xlabel('position, mm')
plt.ylabel('Temperature, $^0$C')
plt.figure(2)
plt.plot(TimeArray,a_array)
plt.xlabel('Time, s')
plt.ylabel('amplitude in the cavity, V/m')
plt.figure(3)
plt.plot(TimeArray,test)
plt.xlabel('Time, s')
plt.ylabel('Mode temperature, $^0C$')


#%%

# for Pin in np.linspace(1e-3,4e-2,6):
# dv_array_forward,a_array_forward=find_spectral_response(Pin,T_equilibr=1,dv_max=20*delta_c,direction='forward')
# dv_array_backward,a_array_backward=find_spectral_response(Pin,T_equilibr=1,dv_max=20*delta_c,direction='backward')
# plt.figure(3)
# plt.clf()
# plt.plot(dv_array_forward,a_array_forward,label='forward')
# plt.plot(dv_array_backward,a_array_backward,label='backward')
# a_array_num=np.array(list(map(lambda dv:stationary_solution(Pin,dv),dv_array_forward)))
# plt.plot(dv_array_forward,a_array_num,'.',label='no nonlinearities')
# plt.legend()
# plt.xlabel('detuning, Hz')
# plt.ylabel('Amplitude in the cavity, V/m')
# plt.title('Pin={:.3f}, gain={:.2f}, transmission to amplifier={:.3f}'.format(Pin,gain_small_signal,transmission_from_taper_to_amplifier))
# plt.savefig('Results\\Pin={:.3f}, gain={:.2f}, transmission to amplifier={:.3f}.png'.format(Pin,gain_small_signal,transmission_from_taper_to_amplifier),dpi=300)

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
# N_dv=30
# dv_max=4*delta_c
# dv_array=np.linspace(dv_max,-dv_max,N_dv)
# plt.plot(dv_array,list(map(lambda w:transmission(w,T0),dv_array)))

# plt.figure(5)
# P_array=np.linspace(0,0.1,20)
# plt.plot(P_array,list(map(amplifying_before_core,P_array)))
print('Time spent={} s'.format(time.time()-time0))

