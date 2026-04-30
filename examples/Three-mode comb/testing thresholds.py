from SNAP import SNAP_model
from SNAP import SNAP_MI_threshold
import numpy as np
import matplotlib.pyplot as plt



#%%


#%%
 
h_width=3000 #mkm
MaxRadVar=0.0100 # mkm
z_dr=np.linspace(-h_width*0.7, h_width*0.7,num=1000)
dr=np.zeros(len(z_dr))
dr[np.abs(z_dr) <= h_width/2] = MaxRadVar
length_of_steepness=0 
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
'Z_taper': 20, #   Taper position along z in microns
'q0': 2, # Pump axial mode number (counting from 0)
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

# Создание и запуск системы
SNAP = SNAP_MI_threshold.SNAP_nonlinear_system(params,dim_space=2**10)
SNAP.calculate_modes()
min_threshold,mu_min,P_nl = SNAP.find_min_positive_threshold()
#%%
SNAP.plot_modes_distribs([20])
#%%
plt.figure()
plt.plot(SNAP.z,SNAP.mode_distribs[:,0]**2)
# plt.plot(SNAP_1.x, SNAP_1.mode_distribs[0])

#%%
R_0=62.5
lambda_0=1550
wave_min,wave_max,res=lambda_0-0.01,lambda_0*(1+MaxRadVar/R_0*1.2), 1e-4

lambda_array=np.arange(wave_min,wave_max,res)

SNAP_1=SNAP_model.SNAP(z_dr,dr*1e3,lambda_array,lambda_0=lambda_0,res_width=1e-4,R_0=R_0)
SNAP_1.set_taper_params(absS=np.sqrt(0.8),phaseS=0.0,ReD=0.00,ImD_exc=0e-3,Csquared=0.00001)
fig=SNAP_1.plot_spectrogram(plot_ERV=True,scale='log')
SNAP_1.find_modes(plot_at_spectrogram=False)
#%%
