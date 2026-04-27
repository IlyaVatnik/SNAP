# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 19:25:42 2025

@author: Илья
"""

from SNAP import SNAP_MI_threshold
import numpy as np
import matplotlib.pyplot as plt
import pickle

q0_array=np.arange(1,50,1)
Z0_array=np.arange(0,100,5)
thresholds_array=np.zeros((len(q0_array),len(Z0_array)))

h_width=3000 #mkm
MaxRadVar=0.0060 # mkm
z_dr=np.linspace(-h_width*0.7, h_width*0.7,num=1000)
dr=np.zeros(len(z_dr))
dr[np.abs(z_dr) <= h_width/2] = MaxRadVar
length_of_steepness=200
mask1 = (z_dr > h_width/2) & (z_dr <= h_width/2 + length_of_steepness)
dr[mask1] = np.linspace(MaxRadVar, 0, np.sum(mask1))
mask2= (z_dr <- h_width/2) & (z_dr>= -h_width/2 - length_of_steepness)
dr[mask2] = np.linspace(0, MaxRadVar, np.sum(mask2))
cone=0.0002
dr+=cone*(z_dr-np.min(z_dr))*1e-3
dr+=np.random.random(len(z_dr))*0.0005


params = {
# 'delta_0': 4e6, #total losses s^-1
# 'delta_c': 2e6, # taper coupling, s^-1
'Gamma': 4e6, # internal losses of the resonator, s^-1
'Z_taper': 0, #   Taper position along z in microns
'q0': 0, # Pump axial mode number (counting from 0)
'mu_max': 3, # maximum detuning that is taken into account
'P_max': 1, # Desired power threshold
'm_val': 354, # azimuthal number
'CouplingWidth': 1, #  half-width of the taper in the constriction (half-width of the Gaussian function)
'RadiusFiber':62.5, # Fiber radius 
'z_dr': z_dr,  # grid for ERV in mkm. Note that internal interpolation will be applied!
'dr': dr  ,         # ERV, in mkm
'C2':33887358691.86023,
'ImD':33887358691.86023
}

file='results.data'

plt.figure()
plt.plot(z_dr,dr)

SNAP = SNAP_MI_threshold.SNAP_nonlinear_system(params,dim_space=2**12)
SNAP.calculate_modes()
SNAP.calculate_pump_mode_params()
C2=SNAP.C2
ImD=SNAP.ImD


#%%

for ii,q0 in enumerate(q0_array):
    SNAP.q0=q0
    for jj,Z0 in enumerate(Z0_array):
        print(Z0,q0)
        SNAP.Z_taper=Z0
        SNAP.delta_0=None
        min_threshold = SNAP.find_min_positive_threshold()[0]
        thresholds_array[ii,jj]=min_threshold
    with open(file,'wb') as f:
        pickle.dump([params,q0_array,Z0_array,thresholds_array],f)    
    
#%%

SNAP.plot_modes_distribs([-1])
