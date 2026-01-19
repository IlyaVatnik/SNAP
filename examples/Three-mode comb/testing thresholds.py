from SNAP import SNAP_model
from SNAP import SNAP_MI_threshold
import numpy as np
import matplotlib.pyplot as plt

 
A=10*1e-3
sigma=160
p=1
def ERV(x):
    if abs(x)<=sigma:
       return A*(1-(x/sigma)**2)
    # return A*np.exp(-(x**2/2/sigma**2)**p)
    else:
       return 0
#            return ERV(5)-1/2*(x)**2
z_dr=np.linspace(-1000, 1000,num=1000)
dr=np.array(list(map(ERV,z_dr)))
plt.figure()
plt.plot(z_dr,dr)


params = {
# 'delta_0': 4e6, #total losses s^-1
# 'delta_c': 2e6, # taper coupling, s^-1
'Gamma': 4e6, # internal losses of the resonator, s^-1
'Z_taper': 30, #   Taper position along z in microns
'q0': 1, # Pump axial mode number (counting from 0)
'mu_max': 10, # maximum detuning that is taken into account
'P_max': 5, # Desi5red power threshold
'm_val': 354, # azimuthal number
'CouplingWidth': 1, #  half-width of the taper in the constriction (half-width of the Gaussian function)
'RadiusFiber':100, # Fiber radius 
'z_dr': z_dr,  # grid for ERV in mkm. Note that internal interpolation will be applied!
'dr': dr  ,         # ERV,
'C2':33887358690.86023,
'ImD':33887358690.86023
}

# Создание и запуск системы
SNAP = SNAP_MI_threshold.SNAP_nonlinear_system(params)
SNAP.calculate_modes()
min_threshold = SNAP.find_min_positive_threshold()
