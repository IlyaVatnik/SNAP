# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 21:11:01 2021

@author: t-vatniki
Time-dependent numerical solution for temperature distribution along the fiber under local heating with WGM mode
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import scipy.integrate as integrate


'''
Constant
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
L=3 # mm, fiber sample length
T0=25 #K


"""
Properties of the resonance and input power
"""
Pin=0.08 # W, power launched through the taper
dw=-1e7 ## 1/s, detuning of the pump from the center of the resonance
delta_c=10e6 #1/s, spectral width of the resonance
x_0=L/2 # point where mode is and where taper is

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
dt = 1/delta_c/5 # also should be no less than dx**2/2/beta
T_max=0.01 #s


N=int(L/dx) # number of numerical points along space axis
x = np.linspace(0, L, N+1)
u = np.zeros(N+1) # temperature
U_0 = np.ones(N+1)*T0 #initial temperature distribution



    
"""
Internal parameters
"""
Veff=integrate.quad(mode_distrib,-10,10)[0]*2*np.pi*r*1.5e-3
mode_distrib_array=np.array(list(map(mode_distrib,x)))
mode_distrib_sum=np.sum(mode_distrib_array)

beta = thermal_conductivity/specific_heat_capacity/density
gamma=heat_exchange/specific_heat_capacity/density*2/r
delta=sigma/specific_heat_capacity/density*2/r

modal_heat_const=epsilon_0*epsilon/2/1.5*c*absorption_in_silica
gamma1=modal_heat_const/density/specific_heat_capacity

F=np.sqrt(4*Pin*delta_c/epsilon_0/epsilon/Veff)


def ode_FE(dw,T_max,a=0,u=np.ones(N+1)*T0):
    N_t=int(T_max/dt)
    Indexes_to_save=[N_t]
    # Ensure that any list/tuple returned from f_ is wrapped as array
    rhs_thermal_array = lambda a,u, t: np.asarray(rhs_thermal(a,u, t))
    t=0
    Uarray=[]
    a_array=np.zeros((N_t+1,1))
    TimeArray=np.linspace(0,T_max,N_t+1)
    for n in range(N_t+1):
        t=t+dt
        a=a+dt*rhs_modal(a,u,t,dw)
        u = u + dt*rhs_thermal_array(a,u, t)
        a_array[n]=abs(a)
        if n in Indexes_to_save:
            Uarray.append(u)
        if (n%10000)==0:
            print('step ', n,' of ', N_t)
    return u, np.array(Uarray), TimeArray,a_array

def rhs_modal(a,u,t,dw):
    return 1j*F-a*delta_c+a*1j*(thermal_optical_responce*np.sum((u-T0)*mode_distrib_array)/mode_distrib_sum+dw)    

def rhs_thermal(a,u, t):
    N = len(u) - 1
    rhs = np.zeros(N+1)
    rhs[0] = 0 #dsdt(t)
#    for i in range(1, N):
#        rhs[i] = (beta/dx**2)*(u[i+1] - 2*u[i] + u[i-1]) + \
#                 f(x[i], t)
    rhs[1:N] = (beta/dx**2)*(u[2:N+1] - 2*u[1:N] + u[0:N-1]) + f_modal(a,x[1:N], t)*gamma1 - (u[1:N]-T0)*gamma-(u[1:N]+273)**4*delta+(T0+273)**4*delta
    rhs[N] = (beta/dx**2)*(2*u[N-1]  -
                           2*u[N]) + f_modal(a,x[N], t)*gamma1 - (u[N]-T0)*gamma -(u[N]+273)**4*delta+(T0+273)**4*delta #+ 2*dx*dudx(t)
    return rhs


# def dsdt(): # derivative of the u at the left end 
#     return 0


def f_modal(a,x, t): # source distribution
        return abs(a)**2*mode_distrib(x)

def f_core():
    return 0
# def dudx(t): # derivative of the u at the right end over x
#     return 0#Q0/np.pi/r**2/thermal_conductivity


def find_spectral_responce(direction='forward'):
    T_max=100/delta_c
    dw_max=3.5*delta_c
    N_dw=30
    if direction=='forward':
        dw_array=np.linspace(-dw_max,dw_max,N_dw)
    if direction=='backward':
        dw_array=np.linspace(dw_max,-dw_max,N_dw)
    a_VS_dw=[]
    u=np.ones(N+1)*T0
    a=0
    for ii,dw in enumerate(dw_array):
        print('step={} of {}'.format(ii,N_dw))
        u,Uarray, TimeArray,a_array = ode_FE(dw,T_max,a,u)
        a=a_array[-1]
        a_VS_dw.append(a)
    return dw_array,np.array(a_VS_dw)

  
def stationary_solution(dw):
    return np.sqrt(F**2/(delta_c**2+dw**2))


# u_final,Uarray, TimeArray,a_array = ode_FE(U_0,dw,T_max)
# fig=plt.figure(1)
# x=x-L/2
# plt.plot(x,Uarray[0])

u,Uarray, TimeArray,a_array = ode_FE(dw,T_max)
plt.figure(2)
plt.plot(TimeArray,a_array)

# dw_array_forward,a_array_forward=find_spectral_responce()
# dw_array_backward,a_array_backward=find_spectral_responce('backward')
# plt.figure(3)
# plt.plot(dw_array_forward,a_array_forward,label='forward')
# plt.plot(dw_array_backward,a_array_backward,label='backward')
# a_array_num=np.array(list(map(stationary_solution,dw_array_forward)))
# plt.plot(dw_array_forward,a_array_num,label='no nonlinearities')
# plt.legend()
# plt.xlabel('detuning, Hz')
# plt.ylabel('Amplitude in the cavity')

# fig=plt.figure(1)
# for ind,t in enumerate(Times):
#     p=plt.plot(x,Uarray[ind,:],label=str(t))
#     ax=plt.gca()

# plt.legend()
# plt.xlabel('Distance, mm')
# plt.ylabel('Temperature, C')
# plt.savefig('Distributions for P='+str(Pin)+ '%, T='+str(Times)+'.png')

