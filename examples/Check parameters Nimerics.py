# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 15:49:49 2020

@author: Ilya
"""

import numpy as np
from SNAP import SNAP_model,SNAP_experiment
import matplotlib.pyplot as plt

ERV_params= [65.48169695155937, 1.7464943342294084, -0.0001155838785117577]
taper_params= [0.84406748696, -0.5, -1e-2, 0, 1e-3]
#(absS, phaseS, ReD, ImD_exc, C2)": 
N=500
x_num=np.linspace(-200,200,N)
x_center=0
# ERV_array=SNAP_experiment.ERV_gauss(x_num,x_center,ERV_params)
ERV_array=np.array([10 if abs(x)<50 else 0 for x in x_num])
lambda_0=1549.98
wave_min,wave_max,res=lambda_0,lambda_0+0.5, 2e-4
lambda_array=np.arange(wave_min,wave_max,res)

    
S=SNAP_model.SNAP(x_num,ERV_array,lambda_array,wave_min, R_0=20)
S.set_taper_params(*taper_params)
S.ERV_params=ERV_params
S.plot_spectrogram(plot_ERV=True)
waves,spectrum=S.get_spectrum(0)

# plt.figure(5)
# for t in 10.0**(-np.arange(3,7,1)):
#     S.set_taper_params(Csquared=t)
#     waves,spectrum=S.get_spectrum(0)
#     plt.plot(waves,spectrum,label=str(t))
# plt.legend()


# plt.figure(6)
# for t in 10.0**(-np.arange(3,7,1)):
#     S.set_taper_params(ReD=t)
#     waves,spectrum=S.get_spectrum(0)
#     plt.plot(waves,spectrum,label=str(t))
# plt.legend()

# plt.figure(7)
# for t in np.linspace(-1,1,8):
#     S.set_taper_params(phaseS=t)
#     waves,spectrum=S.get_spectrum(0)
#     plt.plot(waves,spectrum,label=str(t))
# plt.legend()