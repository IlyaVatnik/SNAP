# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 14:14:19 2022

@author: User
Calculation of shift of the resonanca wavelngth due to heating by WGM
"""
__date__='2023.03.30'
__version__='2023.03.30'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import scipy.integrate as integrate
import time as time_module
import pickle
from numba import jit
import SNAP_model as sp
from scipy import sparse
import scipy.linalg as la
from scipy import interpolate 
from scipy.optimize import curve_fit
import pandas as pd
import bottleneck as bn


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
#r=49.5e-3
specific_heat_capacity=740 # J/kg/K
density=2.2*1e-3*1e-3 # kg/mm**3
refractive_index=1.47
epsilon=refractive_index**2 #refractive index
#absorption_in_silica=1000*3.27e-09 #absorption in silica, 1/mm
absorption_in_silica=10*1.035e-6 #absorption in silica, 1/mm


thermal_optical_responce=1.25e9 # Hz/Celcium, detuning of the effective_ra
thermal_expansion_coefficient=0.0107*r*1e3 #nm/K for effective radius variation

hi_3=2.5e-16 # mm**2/ V

lambda_0=1540.428e-6#1550e-6 # mm
#1540.55688
k0=2*np.pi*refractive_index/(lambda_0) # in 1/mm

"""
Sample parameters
"""
L=5 # mm, fiber sample length
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
#Pin=0.01 # W, power launched through the taper

dv=80e6 ## Hz, detuning of the pump from the center of the cold resonance 
d_dv=0e6
dv_period=5e-4 # frequency of pump wavelength oscillating around detuning dv
x_0=L/2 # point where the center of the mode is  and where taper is

'''
Mode properties
'''
Gamma=-1
delta_0=1713e6 # Hz*pi, spectral width of the resonance due to inner losses
delta_c=227e6 # Hz*pi, spectral width of the resonance due to coupling
phase=2 # Fano phase, in pi
ERV_width=0.2 #mm
nonlinearity=False


'''
Taper and resonator parameters 
'''
res_width=delta_0*lambda_0**2/c # 1e-4 #ширина потерь, надо сделать в нанометрах, должна наверно завитесть от координаты вдоль оси резонатора 
res_width_norm=8*np.pi**2*refractive_index**2/(lambda_0*1e6)**3*res_width*1e6 #из Сумецкого 
taper_absS=0.9 # 1/mm
taper_phaseS=0
taper_ReD=0.0002/(1e3) # 1/mm
taper_ImD_exc=1e-4/(1e3) # 1/mm
taper_Csquared=1e-2/(1e3) # 1/mm
transmission=None

taper_position=-0.055
P=0.063 # W18  dBm
#P=0.04 # 16 dBm
Psc=0.00*P #W мощность рассеянного излучения 5% от накачки
exp_shift=0#100*125e6#3*125e6

"""
calculation parametrs
"""
t_max=0.1 # s
dv_max=120*(delta_0+delta_c)
N_dv=150

dx=0.01 #mm
dt_large=1e-5 #s , for thermal step
delta_t_to_save=dt_large*5 # s


"""
Equation coef
"""



gamma=heat_exchange/specific_heat_capacity/density*2/r
delta=sigma/specific_heat_capacity/density*2/r

#modal_heat_const=epsilon_0*epsilon/2/1.5*c*10e-7#absorption_in_silica
modal_heat_const=epsilon_0*epsilon*c*absorption_in_silica/2
Seff=640e-6 #mm^2
zeta=modal_heat_const/density/specific_heat_capacity/(np.pi*r**2)*Seff

#zeta=epsilon_0*1.45*c*absorption_in_silica*Seff/(2*specific_heat_capacity*density*np.pi*r**2) #?????в mm

theta=1/(specific_heat_capacity*np.pi*r**2*density)

alpha=absorption/4.34/1e3 # absorption in 1/mm for ln case
core_heating_constant=alpha*ESA_parameter*transmission_from_taper_to_amplifier
gain_small_signal_lin=10**(gain_small_signal/10)




def min_imag_D():
    taper_ReS=taper_absS*np.cos(taper_phaseS*np.pi)
    return taper_Csquared*(1-taper_ReS)/(1-taper_absS**2)

def D():
    return taper_ReD+1j*(taper_ImD_exc+min_imag_D())
    
def complex_D_exc():
    return taper_ReD+1j*taper_ImD_exc
    
def S():
    return taper_absS*np.exp(1j*taper_phaseS*np.pi)

need_to_update_transmission=True
"""
grid parameters
"""

def ERV_distrib(x): # WGM mode distribution normilized as max(mode_distrib)=1
    return np.exp(-(x-x_0)**2/ERV_width**2)
#
def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
#

def ERV_from_exp ():
    #res scan without heating_resaved_ERV
    # Processed_spectrogram_final_resaved_ERV
    with open('Initial_ERV.pkl','rb') as f:
        Data=pickle.load(f)
    x0=Data['positions']*1e-3
    N=int((x0[-1]-x0[0])/dx)
    x=np.linspace(x0[0],x0[-1],N)
    #ERVs=Data['ERVs']
    ERV_pd=pd.DataFrame(Data['ERVs']).fillna(method='ffill')
    ERV_0=ERV_pd.to_numpy()
    ERV_0 =ERV_0-ERV_0[0]
    a0=float(max(ERV_0))
    f = interpolate.interp1d(x0, ERV_0[:,0],
                                kind='cubic',
                                fill_value="extrapolate")
    ERV=f(x)
   
    def Gauss(x,  x0, sigma,a=a0):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


    mean = sum(ERV * x) / sum(ERV)
    sigma = np.sqrt(sum(ERV * (x - mean)**2) / sum(ERV))
    popt,pcov = curve_fit(Gauss, x, ERV, p0=[ mean,sigma])
    #[0]=a0
    ERV= Gauss(x, *popt)
    
    N=int((4)/dx)
    x=np.linspace(-2,2,N)
    print(np.size(x))
    ERV_fin=np.ones(np.size(x))*ERV[0]
    #ERV_fin[0:165]=ERV[0]
    ERV_fin[165:240]=ERV
    #ERV_fin[240:-1]=ERV[-1]
    #print(ERV_fin)
    plt.xlabel('Координата вдоль оси резонатора, мм')
    plt.ylabel('Эффективная вариация радиуса, нм')
    plt.plot(x0+0.1,ERV_0)
    plt.plot(x,ERV_fin)
    return ERV_fin,x  



if nonlinearity:
    dt = 1/(delta_c+delta_0)/10 # also should be no less than dx**2/2/beta
else:
    dt=1e-5
    #dt=dt_large
# dt=5e-11 # s



'''
Internal parameters
'''

# N=int(L/dx) # number of numerical points along space axis
# x = np.linspace(0, L, N+1)
# T_0 = np.ones(N+1)*T0 #initial temperature distribution


from_experiment=False

if from_experiment: 
    ERV_0,x = ERV_from_exp()
    N=int((x[-1]-x[0])/dx)-1
    T_0 = np.ones(N+1)*T0
else:    

    DATA = np.load('good_ERV.npy')
    ERV_0=DATA[:,1]
    x=DATA[:,0]*0.001
    N=np.size(x)-1
    T_0 = np.ones(N+1)*T0 #initial temperature distribution
    print("good ERV!")
    
if nonlinearity:
    mu=3*c/lambda_0*hi_3/8/refractive_index**2*2 # (11.19), p.174 frjm Gorodetsky
else:
    mu=0

#Veff=integrate.quad(mode_distrib,-10,10)[0]*2*np.pi*r*2e-3 # effective volume of the WGM


n_steps_to_save=(delta_t_to_save//dt)
n_steps_to_make_temperature_derivation=(dt_large//dt)

beta = thermal_conductivity/specific_heat_capacity/density
if dt>dx**2/beta/10:
    print('change dt to meet Heat transfer equation requirement')
    dt=dx**2/beta/10
    





def solve_model(Pin,dv,t_max,psi_distribs, A_array_initial=None,T=np.ones(N+1)*T0):
    F=0
    N_t=int(t_max/dt)
    # Ensure that any list/tuple returned from f_ is wrapped as array
  
    
    
    rhs_thermal_array = lambda Pin,A,T,t,dv: np.asarray(_rhs_thermal(Pin,A,T, t, psi_distribs))
    t=0
    time_array=[]
    T_averaged_dynamics=[]
    time_start=time_module.time()
    if A_array_initial is None:
        A_array=np.zeros(psi_distribs.shape[0],dtype=complex)
    else:
        A_array=A_array_initial
  
    for n in range(N_t+1):
        t+=dt
        if nonlinearity:
            dv+=d_dv*np.sin(2*np.pi*t/dv_period)
            
        else:
            for i,k in enumerate(psi_distribs):
                F=np.sqrt(4*Pin*delta_c/epsilon_0/epsilon/Veff(k))
                A_array[i]=_analytical_step_for_WGM_amplitude(F,A_array[i],t,dv[i])
                #A[i]=A[i]+dt/6*Runge_Kutta_step(F,A[i],dv[i])
         
                
        if (n%n_steps_to_make_temperature_derivation)==0:
            T+=dt_large*rhs_thermal_array(Pin, A_array, T, t, psi_distribs)

        if (n%50000)==1:
           
            time=time_module.time()
            time_left=(time-time_start)*(N_t/n-1)
            print('step {} of {}, time left: {:.2f} s, or {:.2f} min'.format(n,N_t,time_left,time_left/60))
    item_a=np.max((beta/dx**2)*(T[2:N+1] - 2*T[1:N] + T[0:N-1]) )
    item_b=np.max( - (T[1:N]-T0)*gamma)
    item_c=np.max(-(T[1:N]+273)**4*delta+(T0+273)**4*delta)
    item_d=np.max(_heating_from_WGM(A_array, psi_distribs[:,1:N])*zeta)
    item_f=np.max(Psc*theta*delta_function(x, taper_position)[1:N])    
    return time_array,A_array,T,T_averaged_dynamics,item_a,item_b,item_c,item_d, item_f


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
def _rhs_thermal(Pin,a,T,t,psi): #массив температур в каждой точке вдоль оси 
    N = len(T) - 1
    rhs = np.zeros(N+1)
    rhs[0] = 0 #dsdt(t)
    """
    item_a=np.max((beta/dx**2)*(T[2:N+1] - 2*T[1:N] + T[0:N-1]) )
    item_b=np.max( - (T[1:N]-T0)*gamma)
    item_c=np.max(-(T[1:N]+273)**4*delta+(T0+273)**4*delta)
    item_d=np.max(_heating_from_WGM(A_array, psi[:,1:N])*zeta)
    item_f=np.max(Psc*theta*delta_function(x, taper_position)[1:N])"""
    #rhs[1:N] = (beta/dx**2)*(T[2:N+1] - 2*T[1:N] + T[0:N-1]) - (T[1:N]-T0)*gamma-(T[1:N]+273)**4*delta+(T0+273)**4*delta +_heating_from_WGM(A_array, psi[:,1:N])*zeta+Psc*theta*delta_function(x, taper_position)[1:N]
    rhs[1:N] = (beta/dx**2)*(T[2:N+1] - 2*T[1:N] + T[0:N-1]) +_heating_from_WGM(A_array, psi[:,1:N])*zeta+Psc*theta*delta_function(x, taper_position)[1:N]
    rhs[N] = 0#rhs[N-1]#(beta/dx**2)*(2*T[N] - 2*T[N-1]) - (T[N]-T0)*gamma -(T[N]+273)**4*delta+(T0+273)**4*delta + _heating_from_WGM2(a, psi[:,N])*zeta #+ 2*dx*dudx(t) rhs[N-1]#
    #return rhs,item_a,item_b,item_c,item_d, item_f
    return rhs

def Veff (psi): #??????? это правильно? 
    #print(np.sum(psi)*dx*2*np.pi*r*2e-3)
    return np.sum(psi)*dx*2*np.pi*r*2e-3 # effective volume of the WGM

def delta_function(x, x0):
    delta_f=np.zeros(x.size)
    x0=np.argwhere(abs(x)-abs(x0)<0.001)
    delta_f[x0]=1
    return delta_f

def _heating_from_WGM(a,psi):
    sum_a=0
    for i,_ in enumerate(a):
        # print(a.shape)
        # print(psi.shape)

        sum_a+=(abs(a[i])**2)*abs(psi[i,:])
                #print(i)
        if np.isnan(sum_a.any()):
            print('break')
            break
        # except:
        #     sum_a+=(abs(a[i])**2)*psi[N]**2

    return sum_a

def _heating_from_WGM2(a,psi):
    sum_a=0
    for i,_ in enumerate(a):
        # print(a.shape)
        # print(psi.shape)

        sum_a+=(abs(a[i])**2)*psi[i]
                #print(i)
        if np.isnan(sum_a.any()):
            print('break')
            break
        # except:
        #     sum_a+=(abs(a[i])**2)*psi[N]**2

    return sum_a

  
def stationary_solution(Pin,dv):
    F=np.sqrt(4*Pin*delta_c/epsilon_0/epsilon/Veff)
    return np.sqrt(F**2/((delta_c+delta_0)**2+dv**2))


def SNAP_spectrogram(x,ERV,lambda_array, name='SNAP_objekt', res_width=1e-4,R_0=62.5):
    x=x*1e3
    SNAP=sp.SNAP(x,ERV,lambda_array,lambda_0=lambda_0*1e6,res_width=1e-4,R_0=62.5)
    SNAP.set_taper_params(absS=np.sqrt(0.8),phaseS=0.0,ReD=0.00,ImD_exc=2e-3,Csquared=0.001)
    Spectrogram=SNAP.plot_spectrogram(plot_ERV=False,scale='log')
    SNAP.save(name)
    return Spectrogram


def get_detunings_from_energies(E):
    wl = (-E/(2*k0**2))*lambda_0+lambda_0    
    return wl,c*(1/wl-1/lambda_0)

def psi_normalization(psi):
    #print(p_n.shape)
    # for i in range(np.shape(psi)[1]):
    #     psi[:,i]=psi[:,i]/np.max(psi[:,i])
    for i,l in enumerate(psi):
        psi[i]=(abs(l)**2/(max(abs(psi[i,:])**2)))
    return psi

def solve_Shrodinger(x,ERV):
    U=-2*k0**2*ERV*(1e-6)/(r)/refractive_index
    Tmtx=-1/dx**2*sparse.diags([-2*np.ones(N+1),np.ones(N+1)[1:],np.ones(N+1)[1:]],[0,-1,1]).toarray()
    Vmtx=np.diag(U)
    Hmtx=Tmtx+Vmtx
    (eigvals,eigvecs)=la.eigh(Hmtx,check_finite=False)
    sorted_indexes=np.argsort(np.real(eigvals))
    eigvals,eigvecs=[eigvals[sorted_indexes],eigvecs[sorted_indexes]]
    #eigvecs=eigvecs/np.sqrt(dx)  # to get normalization for integral (psi**2 dx) =1
    #eigvecs=eigvecs/np.max(eigvecs)
    #eigvecs=psi_normalization(eigvecs)
    return eigvals,eigvecs

def GreenFunctionXX(eigvals,eigvecs,wavelength):
    E=-2*k0**2*(wavelength*1e-6-lambda_0)/(lambda_0)
    return bn.nansum(eigvecs**2/(E-eigvals+1j*res_width_norm),1) 

def derive_transmission(lambdas,ERV, x, show_progress=False):
    taper_D=D()
    taper_S=S()
    T=np.zeros((len(lambdas),len(x)))
    #U=-2*k0**2*ERV/62.5/refractive_index
    eigvals,eigvecs=solve_Shrodinger(x,ERV)
    for ii,wavelength in enumerate(lambdas):
        if ii%50==0 and show_progress: print('Deriving T for wl={}, {} of {}'.format(wavelength,ii,len(lambdas)))
        G=GreenFunctionXX(eigvals,eigvecs,wavelength)
        ComplexTransmission=(taper_S-1j*taper_Csquared*G/(1+taper_D*G))  ## 
        T[ii,:]=abs(ComplexTransmission)**2 
    need_to_update_transmission=False
    transmission=T
    if np.amax(T)>1:
        print('Some error in the algorimth! Transmission became larger than 1')
    return x, lambdas,transmission    

def plot_spectrogram(lambdas,transmission, x, ERV, scale='log',ERV_axis=True,plot_ERV=False,amplitude=False):
    wave_max=max(lambdas)
    def _convert_ax_Wavelength_to_Radius(ax_Wavelengths):
        """
        Update second axis according with first axis.
        """
        y1, y2 = ax_Wavelengths.get_ylim()
        nY1=(y1-lambda_0)/wave_max*r*1e3
        nY2=(y2-lambda_0)/wave_max*r*1e3
        ax_Radius.set_ylim(nY1, nY2)
    if need_to_update_transmission:
        _,_,transmission=derive_transmission(lambdas, x, ERV,)
    fig=plt.figure()
    plt.clf()
    ax_Wavelengths = fig.subplots()
    if amplitude:
        temp=np.sqrt(transmission)
    else:
        temp=transmission
    if scale=='log':
        temp=10*np.log10(temp)
    try:
        im = ax_Wavelengths.pcolorfast(x,lambdas,temp,cmap='jet')
    except:
        im = ax_Wavelengths.pcolor(x,lambdas,temp, cmap='jet')
    if scale=='lin':
        plt.colorbar(im,ax=ax_Wavelengths,pad=0.12,label='Transmission')
    elif scale=='log':
        plt.colorbar(im,ax=ax_Wavelengths,pad=0.12,label='Transmission,dB')                
    ax_Wavelengths.set_xlabel(r'Position, $\mu$m')
    ax_Wavelengths.set_ylabel('Wavelength, nm')
    if ERV_axis:
        ax_Radius = ax_Wavelengths.twinx()
        ax_Wavelengths.callbacks.connect("ylim_changed", _convert_ax_Wavelength_to_Radius)
        ax_Radius.set_ylabel('Variation, nm')
    plt.title('simulation')
    if plot_ERV:
        ax_Radius.plot(x,ERV)
        plt.gca().set_xlim((x[0],x[-1]))
    plt.tight_layout()
    return fig

#%%iteration solution process

T=np.ones(N+1)*T0 #temperature along cavity
#ERV=np.array(list(map(ERV_distrib,x))) #в нм, начальная модификация 
#ERV=np.zeros(np.size(T))
#ERV_0=ERV
#ERV=((T-T0)*thermal_expansion_coefficient)
#plt.plot(x,ERV)

lambda_pump=1540.65e-6 
wave_min,wave_max,res=1540.4,1540.85, 1e-5  #1549.9,1550.1, 1e-5    
lambda_array=np.arange(wave_min,wave_max,res)
#plot_spectrogram(lambda_array, transmission, x, ERV_0)
SNAP_spectrogram(x,ERV_0,lambda_array,'without pump.pkl',res_width=1e-5,R_0=62.5)
t_max=0.05
#E, Psi = Shrodinger_solution(x,ERV,lambda_array,res_width=1e-5,R_0=62.5)
E, Psi = solve_Shrodinger(x, ERV_0)
A_array=np.zeros(len(E),dtype=complex)
M=10
T_max=np.zeros((M))
numb=np.array(range(0,M))
fig, ax = plt.subplots()
ax.set_title("Items in heat transfer eq.")
ax.set_ylabel("Bananas, K / sec")
ax.set_xlabel("step")
i_A,i_B,i_C,i_D,i_F=np.ones((M)),np.ones((M)),np.ones((M)),np.ones((M)),np.ones((M))
for i in range(M):
    Psi=psi_normalization(Psi)
    resonance_wavelengths,delta_v=get_detunings_from_energies(E)
    #в эксперименте была отстройка 50 pkm, то есть 6,25 GHz 
    
    delta_v=delta_v-delta_v[-1]+exp_shift#c/lambda_pump#delta_v[-1]#c/lambda_pump
    #print("delta_v=",delta_v)
    _,A_array,T,_,item_a,item_b,item_c,item_d, item_f = solve_model(P, delta_v, t_max, Psi, A_array_initial=A_array) #a_initial должна быть нулём на каждом шаге? 
    T_max[i]=np.max(T)
    i_A[i]=item_a
    i_B[i]=item_b
    i_C[i]=item_c
    i_D[i]=item_d
    i_F[i]=item_f
    #plt.figure(9)
    #plt.plot(x,abs(Psi[-1,:]))
    ERV=ERV_0+((T-T0)*thermal_expansion_coefficient)
    E, Psi = solve_Shrodinger(x, ERV)
   
    if i==M-1:
        print("done")
                
        
        
ax.plot(numb,i_A,label="Thermal conductivity")
ax.plot(numb,i_B,label="surface conductance")
ax.plot(numb,i_C,label="Radiation cooling")
ax.plot(numb,i_D,label="Mode heating")
ax.plot(numb,i_F,label="Taper heating")
ax.legend()    
    
fig, ax2 = plt.subplots()
        
ax2.set_xlabel('i')
ax2.set_ylabel('Максимальная температура. C')
ax2.plot(numb,T_max)

    #print(ERV) #часть значений nan, возможно из-за большого шага по времени 
SNAP_spectrogram(x,ERV,lambda_array,'with pump.pkl', res_width=1e-5,R_0=62.5)
fig, ax1 = plt.subplots()
ax1.set_ylabel("Эффективная вариация радиуса, нм")
ax1.set_xlabel("Координата вдоль оси резонатора, мм")
ax1.plot(x,ERV_0,label='ERV_0')
ax1.plot(x,ERV,label="ERV")
ax.legend()   
ERV_file=pd.DataFrame([[x,'x'], [ERV,'ERV'], [ERV_0, 'ERV_0']])#{'x':x,'ERV':ERV,'ERV_0':ERV_0}
ERV_totxt=[[x],[ERV],[ERV_0]]
np.save('ERV_values',ERV_totxt)
ERV_file.to_csv('erv.txt', sep='\t')   

#%%

#print('Time spent={} s'.format(time.time()-time0))