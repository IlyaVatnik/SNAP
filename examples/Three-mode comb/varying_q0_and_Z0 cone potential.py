# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 19:25:42 2025

@author: Илья
"""

from SNAP import SNAP_MI_threshold
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


Z0_array=np.arange(0,200,2.5)


h_width=3000 #mkm
MaxRadVar=0.0100 # mkm
z_dr=np.linspace(-h_width*0.7, h_width*0.7,num=1000)
dr=np.zeros(len(z_dr))
dr[np.abs(z_dr) <= h_width/2] = MaxRadVar
length_of_steepness=200 
mask1 = (z_dr > h_width/2) & (z_dr <= h_width/2 + length_of_steepness)
dr[mask1] = np.linspace(MaxRadVar, 0, np.sum(mask1))
mask2= (z_dr <- h_width/2) & (z_dr>= -h_width/2 - length_of_steepness)
dr[mask2] = np.linspace(0, MaxRadVar, np.sum(mask2))
cone=0.001
dr+=cone*(z_dr-np.min(z_dr))*1e-3
dr+=np.random.random(len(z_dr))*0.0005


params = {
# 'delta_0': 4e6, #total losses s^-1
# 'delta_c': 2e6, # taper coupling, s^-1
'Gamma': 3e6, # internal losses of the resonator, s^-1
'Z_taper': 0, #   Taper position along z in microns
'q0': 0, # Pump axial mode number (counting from 0)
'mu_max': 4, # maximum detuning that is taken into account
'P_max': 1, # Desired power threshold
'm_val': 354, # azimuthal number
'CouplingWidth': 1, #  half-width of the taper in the constriction (half-width of the Gaussian function)
'RadiusFiber':62.5, # Fiber radius 
'z_dr': z_dr,  # grid for ERV in mkm. Note that internal interpolation will be applied!
'dr': dr  ,         # ERV, in mkm
'C2':25000*1e6, # mkm/s
'ImD':25000*1e6 # mkm/s
}

file='results test.data'

plt.figure()
plt.plot(z_dr,dr)

SNAP = SNAP_MI_threshold.SNAP_nonlinear_system(params,dim_space=2**12)
SNAP.calculate_modes()
SNAP.calculate_pump_mode_params()
C2=SNAP.C2
ImD=SNAP.ImD
q0_array=np.arange(15,SNAP.axial_number_of_modes)
thresholds_array=np.ones((len(q0_array),len(Z0_array)))*np.nan
mu_array=np.ones((len(q0_array),len(Z0_array)))*np.nan
#%%
k=0
time_tic_1=time.time()
N_steps=len(q0_array)*len(Z0_array)
for ii,q0 in enumerate(q0_array):
    SNAP.q0=q0
    for jj,Z0 in enumerate(Z0_array):
        
        
        SNAP.Z_taper=Z0
        SNAP.delta_0=None
        min_threshold,mu,_ = SNAP.find_min_positive_threshold()
        thresholds_array[ii,jj]=min_threshold
        mu_array[ii,jj]=mu
        k+=1
        time_tic_2=time.time()
        time_elapsed=(time_tic_2-time_tic_1)
        time_remaining=(N_steps-k+1)*time_elapsed/k
        print(f'Scanning at q0={q0} Z0={Z0}, step {k} of {N_steps}, time elapsed for step {time_elapsed/k:.1f} s, time remaining={time_remaining//60:.0f} min {np.mod(time_remaining,60):.1f} s')

    with open(file,'wb') as f:
        pickle.dump([params,q0_array,Z0_array,thresholds_array,mu_array],f)    
    
#%%


