# -*- coding: utf-8 -*-
"""
Data from [1] Sumetsky, M., “Theory of SNAP devices: basic equations and comparison with the experiment,” Opt. Express 20(20), 22537 (2012).
@author: Ilya
"""

import numpy as np
import matplotlib.pyplot as plt
from SNAP import SNAP_model




#%% Fig 2 WRONG! 
# C2_array=2*10.0**(-1*np.arange(1,6,1))

# N=200
# lambda_0=1500.0
# wave_min,wave_max,res=lambda_0-0.001,lambda_0+0.001, 2e-5

# x=np.linspace(-30,30,N)
# lambda_array=np.arange(wave_min,wave_max,res)

# psi_function=np.zeros(len(x))
# z_0=12
# for ii,z in enumerate(x):
#     psi_function[ii]=(2/np.pi)**(0.25)*z_0**(-0.5)*np.exp(-(z/z_0)**2)

# SNAP=SNAP_model.SNAP(x,None,lambda_array,lambda_0=lambda_0,res_width=1e-4,R_0=38/2,n=1.5)

    
# SNAP.set_taper_params(absS=np.sqrt(0.99999),phaseS=-0.00,ReD=0.0,ImD_exc=0.0e-2,Csquared=0.00)

# fig, axs = plt.subplots(5, 1, sharex=True, sharey=True)
# for ii,C2 in enumerate(C2_array):  
#     SNAP.set_taper_params(Csquared=C2)
#     SNAP.set_taper_params(ImD_exc=-SNAP.min_imag_D()+C2/2)
#     print(SNAP.D(),SNAP.taper_Csquared)
#     x,l,T=SNAP.derive_transmission_test(psi_function)
#     a=axs[4-ii].pcolorfast(x,l,T,label=str(C2),cmap='jet')
# plt.legend()
# plt.colorbar(a,pad=0.12)
# print(np.min(T))

#%% Fig 5b
N=200
lambda_0=1548.45
wave_min,wave_max,res=lambda_0,1549.8, 3e-4
lambda_array=np.arange(wave_min,wave_max,res)
x=np.linspace(0,250,N)
ERV=np.zeros(len(x))
ERV_0=11
hi=10.9*2.5
z_0=120
epsilon=0.22

for ii,z in enumerate(x):
    ERV[ii]=ERV_0*(np.exp(-(z-z_0)**2/hi**2)+epsilon/(1+((z-z_0)**2/hi**2)))

SNAP=SNAP_model.SNAP(x,ERV,lambda_array,lambda_0=lambda_0,res_width=1e-4,R_0=19,n=1.45)
S=0.879-0.084*1j
C2=0.026
D=0.022+0.02*1j

SNAP.set_taper_params(absS=np.abs(S),phaseS=np.angle(S)/np.pi,ReD=np.real(D),Csquared=C2)
SNAP.set_taper_params(ImD_exc=-SNAP.min_imag_D()+np.imag(D))
SNAP.plot_ERV()
fig=SNAP.plot_spectrogram(plot_ERV=True,amplitude=True)
im=fig.axes[0].get_images()
im[0].set_clim(0.4,0.9)
print(SNAP.D(),SNAP.S())

