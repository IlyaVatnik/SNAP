# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 12:29:29 2025

@author: Илья
"""


"""
Created on Fri Oct 11 15:29:55 2019

@author: Ilya
V.6
13.10.2022
Time-dependent numerical solution for temperature distribution along the fiber under local heating with _moving_ laser beam
This considers relaxation times as well
Using different inner and outer radia for a cappilar
"""

from SNAP import SNAP_ThermalModel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import json
import os

L=200


t_max=150
t_step=1
x_beam_0=0 # position of the laser beam at the zero time

thermal_x_step=0.1
mode_resolution_x=thermal_x_step

x_ERV_init=np.arange(-0.5,0.5,mode_resolution_x)
ERV_0_init=3.2*(1-(x_ERV_init/L)**2)*0


'''
make noneven x grid
by adding few points to the 
'''
x,x_ERV,ERV_0=SNAP_ThermalModel.make_uneven_grids(-L, L, x_ERV_init,ERV_0_init,mode_resolution_x,thermal_x_step)
#%%


T_0=np.ones(len(x))*20
model=SNAP_ThermalModel.SNAP_ThermalModel(x,T_0,T_bound=T_0[0],r_out=3, r_in=2, absorption=0,medium_name='sapphire')

laser_spot_radius,z_R=SNAP_ThermalModel.CO2_laser_spot_after_propogating(distance_propagated=700)



for pump_in_percents in [1,5,3,6,12,15]:

    
    Power=36*0.91*pump_in_percents*0.01
    times=np.arange(0,t_max,t_step)
    powers=Power*np.ones(len(times))
    positions=np.ones(len(times))*x_beam_0
    model.set_external_CO2_laser_parameters(times, powers, positions, laser_spot_radius,0, laser_spot_radius)
    model.set_SNAP_parameters(x_ERV, ERV_0, intristic_losses=0, lambda_0=1550)
    model.set_taper_parameters( taper_C2=0, taper_ImD=0,taper_position=0, taper_losses=0)
    
    model.solve_full_system(dt=0.5e-3)
    model.save_model('data\\Power={} %.model'.format(pump_in_percents))
#%%

def read_file(name):
    f=open(name,'r')
    X=f.readlines()
    [P,T]=[float(X[i].split()[0]) for i in range(len(X))],[float(X[i].split()[1]) for i in range(len(X))]
    P.sort()
    T.sort()
    f.close()
    X=[P,T]
    return X



X=read_file('Experiment.txt')
Powers=X[0]
Temps=X[1]



plt.figure(1)
plt.plot(Powers,Temps,label='эксперимент')


file_list=os.listdir('data')
file_list.sort(key=lambda x: float(x.split(' %')[0].split('=')[1]))
for file in file_list:
    model=SNAP_ThermalModel.load_model('data\\'+file)
    power_in_percents=float(file.split(' %')[0].split('=')[1])
    Power=power_in_percents*36*0.91*0.01
    plt.plot(power_in_percents,np.max(model.T),'o',label=file)
    

plt.xlabel('Мощность на контроллере лазера, %')
plt.ylabel('Температура внутри трубки, градусы')
plt.legend()



