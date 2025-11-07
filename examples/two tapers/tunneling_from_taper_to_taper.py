'''
january 2022
For the megagrant report 2021
'''

import numpy as np
import matplotlib.pyplot as plt
from SNAP import SNAP_model

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