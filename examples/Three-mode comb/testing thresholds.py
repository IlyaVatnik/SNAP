from SNAP import SNAP_model
from SNAP import SNAP_MI_threshold
import numpy as np
import matplotlib.pyplot as plt

 
lambda_0=1552.21
R_0=100

A=15*1e-3
sigma=160

pump_mode=2

def ERV(x):
    if abs(x)<=sigma:
        return A*(1-(x/sigma)**2)
       # return A*np.exp(-(x**2/2/sigma**2))
    else:
       return 0
#            return ERV(5)-1/2*(x)**2
z_dr=np.linspace(-1000, 1000,num=2**8)
dr=np.array(list(map(ERV,z_dr)))
# plt.figure()
# plt.plot(z_dr,dr)


#%%

wave_min,wave_max,res=lambda_0-0.01,lambda_0*(1+A/R_0*1.2), 1e-4

lambda_array=np.arange(wave_min,wave_max,res)

SNAP_1=SNAP_model.SNAP(z_dr,dr*1e3,lambda_array,lambda_0=lambda_0,res_width=1e-4,R_0=R_0)
SNAP_1.set_taper_params(absS=np.sqrt(0.8),phaseS=0.0,ReD=0.00,ImD_exc=0e-3,Csquared=0.00001)
fig=SNAP_1.plot_spectrogram(plot_ERV=True,scale='log')
SNAP_1.find_modes(plot_at_spectrogram=False)
#%%
Dint=SNAP_1.get_Dint(pump_mode)
plt.figure()
plt.plot(Dint)

#%%

params = {
# 'delta_0': 4e6, #total losses s^-1
# 'delta_c': 2e6, # taper coupling, s^-1
'Gamma': 4e6, # internal losses of the resonator, s^-1
'Z_taper': 30, #   Taper position along z in microns
'q0': pump_mode, # Pump axial mode number (counting from 0, that is the lowest wavelength!)
'mu_max': 5, # maximum detuning that is taken into account
'P_max': 5, # Desi5red power threshold
'm_val': 400, # azimuthal number
'CouplingWidth': 1, #  half-width of the taper in the constriction (half-width of the Gaussian function)
'RadiusFiber':R_0, # Fiber radius 
'z_dr': z_dr,  # grid for ERV in mkm. Note that internal interpolation will be applied!
'dr': dr*1.45  ,         # ERV,
'C2':33887358690.86023,
'ImD':33887358690.86023
}

# Создание и запуск системы
SNAP = SNAP_MI_threshold.SNAP_nonlinear_system(params,dim_space=2**10)
SNAP.calculate_modes()
print(SNAP.lambda_0)
min_threshold = SNAP.find_min_positive_threshold()
#%%
plt.figure()
plt.plot(SNAP.z,SNAP.mode_distribs[:,-1])
plt.plot(SNAP_1.x, SNAP_1.mode_distribs[0])
