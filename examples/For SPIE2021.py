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
import matplotlib.animation as animation #для гифки 
import scipy.integrate as integrate
import time
import scipy as sc
from scipy.signal import chirp, find_peaks, peak_widths
from celluloid import Camera
import pickle

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
T0=20 #Celsium, initial temperature of the sample being in equilibrium 

'''
Properties of the heating into the core
'''

absorption=25 #dB/m , absoprtion in the active core
ESA_parameter=0.15 # Excitated state absorption parameter, from  [Guzman-Chavez AD, Barmenkov YO, Kir’yanov A V. Spectral dependence of the excited-state absorption of erbium in silica fiber within the 1.48–1.59μm range. Appl Phys Lett 2008;92:191111. https://doi.org/10.1063/1.2926671.]
# thermal_expansion_coefficient=0.0107*r*1e3 #  nm/K, for effective radius variation

transmission_from_taper_to_amplifier=0.3  # parts, betwee the taper and amplifier
gain_small_signal=20 #amplifiyng after first diod
gss2=10 #amplifiyng after second diod
 # dB, gain of the amplifier guiding to the core
P_sat=0.08 # W, saturation power for the amplifier

x_slice=2*L/5 # position of the slice

"""
Properties of the input radiation
"""
Pin=0.08 # W, power launched through the taper

dv=-1e8 ## Hz, detuning of the pump from the center of the cold resonance 
d_dv=70e6
dv_period=5e-4
x_0=L/2 # point where the center of the mode is  and where taper is

'''
Mode properties
'''
Gamma=1
delta_0=150e6 # Hz, spectral width of the resonance due to inner losses
delta_c=100e6 # Hz, spectral width of the resonance due to coupling

mode_width=0.2 #mm
def mode_distrib(x): # WGM mode distribution normilized as max(mode_distrib)=1
    return np.exp(-(x-x_0)**2/mode_width**2)


"""
grid parameters
"""

t_max=0.8 # s
dv_max=100*(delta_0+delta_c)
N_dv=200

dx=0.05
# dt = 1/delta_c/6 # also should be no less than dx**2/2/beta
dt=1e-3 # s


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
gss2_lin = 10**(gss2/10)


def solve_model(Pin,dv,t_max,a=0,T=np.ones(N+1)*T0):
    F=np.sqrt(4*Pin*delta_c/epsilon_0/epsilon/Veff)
    N_t=int(t_max/dt)
    Indexes_to_save=[N_t]
    # Ensure that any list/tuple returned from f_ is wrapped as array
    rhs_thermal_array = lambda Pin,a,dv,T,T_averaged_over_mode, t: np.asarray(_rhs_thermal(Pin,a,dv,T,T_averaged_over_mode, t))
    t=0
    T_array=[]
    a_array=[]
    test=[]
    TimeArray=np.linspace(0,dt*N_t,N_t+1)
    for n in range(N_t+1):
        t+=dt
        dv+=d_dv*np.sin(2*np.pi*t/dv_period)
        # a=a+dt*_rhs_modal(F,a,u,t,dv)
        T_averaged_over_mode=np.sum(T*mode_distrib_array)/mode_distrib_sum
        a=_analytical_step_for_WGM_amplitude(F,a,T_averaged_over_mode,dt,dv)
        test.append(T_averaged_over_mode)
        if abs(a)>1e10:
            print('unstable simulation. Detuning is too large')
            TimeArray=np.linspace(0,dt*n,n)
            break
        a_array.append(abs(a))
        # if (n%100)==0:
        T+=dt*rhs_thermal_array(Pin,a,dv,T,T_averaged_over_mode, t)
        # test.append(heating_from_core(dv,L/2,du_average))
        # test.append(thermal_optical_responce*np.sum((u-T0)*mode_distrib_array/mode_distrib_sum))
        if n in Indexes_to_save:
            T_array.append(T)
        if (n%10000)==0:
            print('step ', n,' of ', N_t)
    return TimeArray,a_array,T,T_averaged_over_mode

def _rhs_modal(F,a,T_averaged_over_mode,t,dv):
    return 1j*F-a*(delta_c+delta_0)+a*1j*(thermal_optical_responce*(T_averaged_over_mode-T0)+dv)    

def _analytical_step_for_WGM_amplitude(F,a,T_averaged_over_mode,t,dv):
    temp=1j*(thermal_optical_responce*(T_averaged_over_mode-T0)+dv)-delta_c-delta_0 #there is not Kerr nonlinearity 
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
    #ph_term=4*delta_c*(delta_c+delta_0)*np.cos(np.pi/3)/((delta_c+delta_0)**2+(dv+(T_averaged_over_mode-T0)*thermal_optical_responce)**2)
    return np.abs(np.exp(1j*np.pi)*0.01-0.04*2*delta_c/((delta_c+delta_0)-1j*(dv+(T_averaged_over_mode-T0)*thermal_optical_responce)))
    #return 1*0.0011-4*0.002*delta_c*delta_0*Gamma**2/((delta_c+delta_0)**2+(dv+(T_averaged_over_mode-T0)*thermal_optical_responce)**2)
    #return 1-ph_term+4*delta_c*delta_c*Gamma**2/((delta_c+delta_0)**2+(dv+(T_averaged_over_mode-T0)*thermal_optical_responce)**2)

def _amplifying_before_core(P,P_s=P_sat,gain_lin=gain_small_signal_lin):
    return gain_lin**(1/(1+P/P_s))

def _amplifying_before_core2(P):
    return gss2_lin**(1/(1+P/P_sat))

def _heating_from_core(Pin,dv,x,T_averaged_over_mode): # source distribution
    if np.size(x)>1:   
        output=np.zeros(np.shape(x))
        inds=np.where(x>x_slice)
        output[inds]=np.exp(-alpha*(x[inds]-x_slice))
        PintoCore=Pin*transmission(dv,T_averaged_over_mode)
        return output*core_heating_constant*PintoCore*_amplifying_before_core(PintoCore)#*_amplifying_before_core2(PintoCore) #amplifying_before_core2 is the amplification from second diode
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
    T_list=[]
    T_aver_list=[]
    a=0
    for ii,dv in enumerate(dv_array):
        print('step={} of {} in {}'.format(ii,N_dv,direction))
        TimeArray,a_array,T,T_averaged_over_mode = solve_model(Pin,dv,t_max_dv,a,T)
        a=a_array[-1]
        a_VS_dv.append(a)
        #U.append(test)
        T_list.append(T)
        print(np.sum(T*mode_distrib_array)/mode_distrib_sum-T_averaged_over_mode)
        T_aver_list.append(T_averaged_over_mode)
    return dv_array,np.array(a_VS_dv),T_list,T_aver_list




time0=time.time()


#%%

#timeArray,a_array,T,test = solve_model(Pin,dv,t_max=3)
#fig=plt.figure(1)
#x=x-L/2
#plt.plot(x,T)
#plt.xlabel('position, mm')
#plt.ylabel('Temperature, $^0$C')
# plt.figure(2)
# plt.plot(timeArray,a_array)
# plt.xlabel('Time, s')
# plt.ylabel('amplitude in the cavity, V/m')
# plt.figure(3)
# plt.plot(timeArray,test)
# plt.xlabel('Time, s')
# plt.ylabel('Mode temperature, $^0C$')


#%%

# for Pin in np.linspace(1e-3,4e-2,6):
# dv_array_forward,a_array_forward, T_forward,T_aver_list_forward=find_spectral_response(Pin,t_equilibr=t_max,dv_max=dv_max,N_dv=N_dv,direction='forward')
dv_array_backward,a_array_backward, T_back,T_aver_list_backward=find_spectral_response(Pin,t_equilibr=t_max,dv_max=dv_max,N_dv=N_dv,direction='backward')
core_heating_constant=0.0
dv_array_backward,a_array_backward, T_back_no_core,T_aver_list_backward_no_core=find_spectral_response(Pin,t_equilibr=t_max,dv_max=dv_max,N_dv=N_dv,direction='backward')

with open('results.pkl','wb') as file:
    pickle.dump((dv_array_backward,a_array_backward,T_back,T_aver_list_backward,T_back_no_core,T_aver_list_backward_no_core),file)

# peaks,_ = find_peaks(a_array_backward, 0.5)
# width = peak_widths(a_array_backward,peaks)
#%%
# plt.figure(3)
# plt.clf()
# plt.plot(dv_array_forward,a_array_forward,label='forward')
# plt.plot(dv_array_backward,a_array_backward,label='backward')
# a_array_num=np.array(list(map(lambda dv:stationary_solution(Pin,dv),dv_array_forward)))
# plt.plot(dv_array_forward,a_array_num,'.',label='no nonlinearities')
# plt.legend()
# plt.xlabel('Detuning, Hz')
# plt.ylabel('Amplitude in the cavity, V/m')
# plt.title('Pin={:.3f},heat_from_mode={}, gain={:.2f}, transmission to amplifier={:.3f}, absorbtion={:.0f}, peak width={:.0f}'.format(Pin,bool(absorption_in_silica),gain_small_signal,transmission_from_taper_to_amplifier, absorption,width[0][0]))
# # plt.savefig('Results\\Pin={:.3f},heat_from_mode={}, gain={:.2f}, transmission to amplifier={:.3f}.png'.format(Pin,bool(absorption_in_silica),gain_small_signal,transmission_from_taper_to_amplifier),dpi=300)
# for_save=np.array([dv_array_backward,a_array_backward,a_array_forward,a_array_num])
# for_save=for_save.T
# np.savetxt('a_array_p10mW_fano_diodon.txt', for_save)



# plt.figure(4)
# plt.xlabel('Detuning, Hz')
# plt.ylabel('Transmission')
# plt.plot(dv_array_forward,transmission(dv_array_forward,))
# tr_shape=np.array([dv_array_backward,transmission(dv_array_backward)])
# tr_shape=tr_shape.T
# np.savetxt('fano_shape1.txt', tr_shape)

# fig=plt.figure(1)
# for ind,t in enumerate(Times):
#     p=plt.plot(x,Uarray[ind,:],label=str(t))
#     ax=plt.gca()

# plt.legend()
# plt.xlabel('Distance, mm')
# plt.ylabel('Temperature, C')
# plt.savefig('Distributions for P='+str(Pin)+ '%, T='+str(Times)+'.png')


#%%
# from matplotlib.ticker import EngFormatter
# formatter1 = EngFormatter()
with open('results.pkl','rb') as file:
    dv_array_backward,a_array_backward,T_back,T_aver_list_backward,T_back_no_core,T_aver_list_backward_no_core=pickle.load(file)


fig,axs = plt.subplots(2,1)
camera = Camera(fig)

axs[0].set_xlabel('Detuning, GHz')
axs[0].set_ylabel('Transmission')
axs[1].set_xlabel('Detuning, GHz')
axs[1].set_ylabel('Transmission')
axs[0].set_title('with core injection')
axs[1].set_title('no core injection')
plt.tight_layout()
    
Taveraged=[]
Tmax=[]
for ii,dv in enumerate(dv_array_backward):
    # plt.xlim(-1e10, 1e10)

    # axs[0].plot(dv_array_backward, transmission(dv_array_backward),color='blue')
    axs[0].plot(dv_array_backward/1e9, transmission(dv_array_backward,T_aver_list_backward[ii]),color='blue', label='standard')# color='blue')
    axs[0].axvline(dv/1e9, 0.00, 1, color='red')
    axs[1].plot(dv_array_backward/1e9, transmission(dv_array_backward,T_aver_list_backward_no_core[ii]),color='blue', label='with core heating')# color='blue')
    axs[1].axvline(dv/1e9, 0.00, 1, color='red')
    Taveraged.append(np.sum(T_back[ii]*mode_distrib_array/mode_distrib_sum))
    Tmax.append(max(T_back[ii]))
    #plt.title('{}, {},{}'.format(T_forward[ii],ii,dv))
    # plt.axvline(dv_array_forward[np.argmax(transmission(dv_array_forward,T_forward[i]))], 0.04, 0.05, color='red')


    
    camera.snap()
 
animation = camera.animate()
animation.save('peak.gif', writer = 'imagemagick')

#%%
'''
speed up the gif
'''
import imageio

gif_original = 'peak.gif'
gif_speed_up = 'peak_speeded.gif'

gif = imageio.mimread(gif_original)

imageio.mimsave(gif_speed_up, gif, fps=40)

#%%
plt.figure(10)
plt.clf()
plt.plot(dv_array_backward,Taveraged, label='av')
plt.plot(dv_array_backward,Tmax, label='max')
plt.plot(dv_array_backward,T_aver_list_backward, label='aver')
plt.legend()



#%%

# plt.figure(4)
# delta_c=10e6
# delta_0=100e6
# N_dv=100
# dv_max=30*(delta_c+delta_0)
# dv_array=np.linspace(dv_max,-dv_max,N_dv)
# plt.plot(dv_array,list(map(lambda w:transmission(w,T0),dv_array)))

# plt.figure(5)
# plt.plot(dv_array,list(map(lambda w:max(_heating_from_core(Pin, w, x, T_averaged_over_mode=T0)),dv_array)))

print('Time spent={} s'.format(time.time()-time0))

#%%
gain_small_signal=18
gain_small_signal_lin=10**(gain_small_signal/10)
P_sat=0.025 # in W
P_in_array=np.linspace(1e-5,10,50000)
actual_gain=[_amplifying_before_core(P,P_sat,gain_small_signal_lin) for P in P_in_array]
plt.figure(15)
plt.clf()
plt.plot(10*np.log10(P_in_array*actual_gain/0.001),10*np.log10(actual_gain))
plt.xlabel(' Output power, dBm')
plt.ylabel(' gain, dB')

plt.figure(16)
plt.clf()
plt.plot(10*np.log10(P_in_array/0.001),10*np.log10(actual_gain))
plt.xlabel(' Input power, dBm')
plt.ylabel(' gain, dB')
