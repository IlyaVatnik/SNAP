
"""
Created on Fri Oct 11 15:29:55 2019

@author: Ilya
V.5
26.11.2019
Time-dependent numerical solution for temperature distribution along the fiber under local heating with _moving_ laser beam
This considers relaxation times as well
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import json
import os


ResultsDirName='Results\\' # folder to save results
if not os.path.exists(ResultsDirName):
    os.mkdir(ResultsDirName)
    
'''
Important laser parameters
'''
P=25 # power of the C02 laser in percents
IsMovingBeam=False
T_laser_on=0.051 #s,  time the beam is standing at the x_0 before sweeping
T_laser_off=1 #s
NumberOfShots=1


    

"""
Physical constants
"""
k=1.380e-23 ##J/K
R=8.31 #J/mol/K
sigma=0.92*5.6e-8*1e-6 #W/mm**2/K**4 Stephan-Boltzman constant

"""
Fused silica parameters
"""
thermal_conductivity=1.38*1e-3 # W/mm/K
heat_exchange=10*1e-6 #W/mm**2/K
specific_heat_capacity=740 # J/kg/K
density=2.2*1e-3*1e-3 # kg/mm**3
T_annealing=1140 # Celcium degree, temperature to make silica glass soft
T_melting=2000 # Celcium degree, temperature to make silica glass melt

Q_H=500e3 #J/mol, this is the activation energy at low temperatures
G=31e9 #Pa, Young module
t_relaxation_at_annealing_temperature=100 #s
 
def t_relaxation(T): #[1] Ojovan, M. I., “Viscosity and Glass Transition in Amorphous Oxides,” Adv. Condens. Matter Phys. 2008, 1–23 (2008)., eq (3),(4)
    eta_by_G=np.exp(Q_H/R*(1/T-1/T_annealing))*t_relaxation_at_annealing_temperature
    return eta_by_G

AmountOfRelaxedDefects_To_EVR=1/4e-4*(0.91-0.895)/2 #nm per part , derived from experimental data from our work 


"""
Sample parameters
"""
r=62.5e-3 #mm, fiber radius
L=15 # mm, fiber sample length
T0=28 #K


"""
Properties of the CO2 laser beam
"""

laser_initial_spot_radius=3.5/2 #mm, out of the laser
distance_propagated=350 #mm, by the beam before the lense
divergence=4e-3/2 #rad
laser_spot_radius=laser_initial_spot_radius+divergence*distance_propagated
laser_focused_spot_width=0.05 # mm, radius of the waist at the focus
x_beam_0=L/2 # position of the laser beam at the zero time
Laser_power_absorbed=P*0.01*32*r/laser_spot_radius # W, part of the power that is thought to be absrorbed by the fiber



x_beam_position=np.zeros(NumberOfShots+1)+x_beam_0

L_modification=1 # mm, length of the modificaition




""" 
grid parameters
"""

dx=laser_focused_spot_width/6 # mm, grid step
N=int(L/dx) # number of numerical points along space axis
x = np.linspace(0, L, N+1)
u = np.zeros(N+1) # temperature
U_0 = np.ones(N+1)*T0 #initial temperature distribution
beta = thermal_conductivity/specific_heat_capacity/density
dt = dx**2/3/beta # should be no less than dx**2/2/beta

"""
Internal parameters
"""

Total_exposition_time=(T_laser_on+T_laser_off )*NumberOfShots  
gamma=heat_exchange/specific_heat_capacity/density*2/r
delta=sigma/specific_heat_capacity/density*2/r


f0=Laser_power_absorbed/(np.sqrt(np.pi)*laser_focused_spot_width)/(specific_heat_capacity*density*np.pi*r**2) #checked
print('Total_exposition_time=',Total_exposition_time,' s')
T_array_to_plot=np.arange(0,Total_exposition_time,T_laser_on) # in sec, Time stamps to save distributions, 
Indexes_to_save= (T_array_to_plot/dt).astype(int)
N_t=max(Indexes_to_save)
TimeArray=np.arange(0,N_t+1)*dt



def ode_FE(rhs, U_0, dt,T_array_to_plot):
    # Ensure that any list/tuple returned from f_ is wrapped as array
    rhs_ = lambda u, t: np.asarray(rhs(u, t))
    u = U_0
    t=0
    Uarray=[]
    MaxTemperatureList=[]
    AnnealedAreaArray=[]
    Exposition=np.zeros(len(x))
    DefectsStrength=np.ones(len(x))
    for n in range(N_t+1):
        t=t+dt
        u = u + dt*rhs_(u, t)
        MaxTemperatureList.append(np.max(u))
        if T_melting>MaxTemperatureList[-1]>T_annealing:        
            edges = np.argpartition(abs(u-T_annealing), 2)
            AnnealedAreaArray.append(abs(edges[1]-edges[0])*dx)
            Exposition[edges[1]:edges[0]]+=dt*(u[edges[1]:edges[0]]-T_annealing)
            DefectsStrength=DefectsStrength*np.array(list(map(lambda T:np.exp(-dt/t_relaxation(T)) if T>900 else 1,u)))
        elif MaxTemperatureList[-1]>T_melting:
            raise RuntimeError('Error for t={:f} s. Temperature is too high. Fiber has been melt'.format(t))
        else:   
            AnnealedAreaArray.append(0)
        if n in Indexes_to_save:
            Uarray.append(u)
        if (n%10000)==0:
            print('{},{},{}'.format(max((u[1:N]-T0)*gamma),max((u[1:N]+273)**4*delta-(T0+273)**4*delta),max((beta/dx**2)*(u[2:N+1] - 2*u[1:N] + u[0:N-1]))))         
            print('step ', n,' of ', N_t)
    ERVarrayThroughRelaxation=(np.ones(len(x))-DefectsStrength)*AmountOfRelaxedDefects_To_EVR
    return u, np.array(Uarray), MaxTemperatureList,AnnealedAreaArray,Exposition,ERVarrayThroughRelaxation

def rhs(u, t):
    N = len(u) - 1
    rhs = np.zeros(N+1)
    rhs[0] = dsdt(t)
    rhs[1:N] = (beta/dx**2)*(u[2:N+1] - 2*u[1:N] + u[0:N-1]) + beam_distribution(x[1:N], t) - (u[1:N]-T0)*gamma-(u[1:N]+273)**4*delta+(T0+273)**4*delta
    rhs[N] = (beta/dx**2)*(2*u[N-1] + 2*dx*dudx(t) -
                           2*u[N]) + beam_distribution(x[N], t) - (u[N]-T0)*gamma -(u[N]+273)**4*delta+(T0+273)**4*delta
    return rhs


def dsdt(t): # derivative of the u at the left end 
    return 0

def dudx(t): # derivative of the u at the right end over x
    return 0#Q0/np.pi/r**2/thermal_conductivity



def beam_distribution(x, t): # source distribution
    if t%(T_laser_off+T_laser_on)<T_laser_on:
        x_0=x_beam_position[int(t//(T_laser_off+T_laser_on))]
        return f0*np.exp(-((x-x_0)/laser_focused_spot_width)**2)
    else:
        return 0


"""
Numerical simulation
"""
u_final,Uarray,MaxTemperatureArray,AnnealedAreaArray,Exposition,ERVarrayThroughRelaxation = ode_FE(rhs, U_0, dt, T_array_to_plot)

fig=plt.figure(1)
for ind,t in enumerate(T_array_to_plot):
    p=plt.plot(x,Uarray[ind,:],label=str(t))
    ax=plt.gca()
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between(x, 0, 1, where=Uarray[ind,:] > T_annealing,
                facecolor=p[0].get_color(), alpha=0.1, transform=trans)
plt.legend()
plt.xlabel('Distance, mm')
plt.ylabel('Temperature, C')
plt.savefig(ResultsDirName+'Distributions for P='+str(P)+ '%, T='+str(Total_exposition_time)+'.png',dpi=300)

fig, ax1 = plt.subplots()
ax1.plot(TimeArray,MaxTemperatureArray)
ax1.set_xlabel('Time,s')
ax1.set_ylabel('Temperature at maximum, Celsium degrees')
ax1.axhline(T_annealing,alpha=0.05)

ax2=ax1.twinx()

ax2.plot(TimeArray,AnnealedAreaArray,color='red')
ax2.set_ylabel('Width of the annealed area, mm',color='red')

plt.savefig(ResultsDirName+'Temperature and Annealed area for P='+str(P)+'%, T='+str(Total_exposition_time)+'.png',dpi=300)

fig3=plt.figure(3)
plt.plot(x,Exposition)
if any(Exposition>0):
    xmin=-0.1+x_beam_0
    xmax=0.1+x_beam_0+L_modification
    plt.xlim([xmin,xmax])
plt.xlabel('Distance, mm')
plt.ylabel('Exposition, s*K')
plt.savefig(ResultsDirName+'Exposition for P='+str(P)+ '%, T='+str(Total_exposition_time)+'.png',dpi=300)
np.savetxt(ResultsDirName+'Exposition.txt',np.stack((x,Exposition),axis=-1))


fig5,ax_Radius=plt.subplots()
ax_Radius.plot(x,ERVarrayThroughRelaxation)
if any(ERVarrayThroughRelaxation>0):
     plt.xlim([xmin,xmax])
plt.xlabel('Distance, mm')
plt.ylabel('ERV derived from relaxation times,nm')
def radius_to_wavelength(x):
    return x*1550/62.5e3
def wavelength_to_radius(x):
    return x/1550*62.5e3
ax_Wavelengths = ax_Radius.secondary_yaxis('right',functions=(radius_to_wavelength,wavelength_to_radius))
ax_Wavelengths.set_ylabel('Wavelength shift, nm')
plt.tight_layout()
plt.savefig(ResultsDirName+'Derived ERV for P='+str(P)+ '%, T='+str(Total_exposition_time)+'.png',dpi=300)
plt.figure(5)

Variables={
        'P':P,
        'laser_spot_radius':laser_spot_radius,
        'laser_spot_radius':laser_spot_radius,
        'laser_focused_spot_width':laser_focused_spot_width,
        'L_modification':L_modification,
        'dx':dx,
        'dt':dt}

with open(ResultsDirName+'variables.txt', 'w') as json_file:
    json.dump(Variables, json_file)
#with open(ResultsDirName+'variables.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#    pickle.dump([P,laser_spot_radius,laser_focused_spot_width,L_modification,dx_modification], f)
#


