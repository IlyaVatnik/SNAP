# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:57:44 2020

@author: Ilya
"""
import numpy as np
import matplotlib.pyplot as plt
from SNAP import SNAP_model

C2_array=np.arange(0.01,0.05,0.001)

N=200
lambda_0=1552.21
wave_min,wave_max,res=1552.2,1552.6, 2e-4

x=np.linspace(-250,250,N)
lambda_array=np.arange(wave_min,wave_max,res)

A=3.274
sigma=123.5934
p=1.1406
def ERV(x):
    # if abs(x)<=200:
#            return (x)**2
    return A*np.exp(-(x**2/2/sigma**2)**p)
    # else:
        # return 0
#            return ERV(5)-1/2*(x)**2
ERV=np.array(list(map(ERV,x)))
SNAP=SNAP_model.SNAP(x,ERV,lambda_array,lambda_0=lambda_0,res_width=1e-5,R_0=38/2)
SNAP.set_taper_params(absS=np.sqrt(0.8),phaseS=-0.05,ReD=0.025,ImD_exc=0.9e-2,Csquared=0.03)
# SNAP.plot_spectrogram(plot_ERV=False,scale='log')

minTrans_list=[]
for C2 in C2_array:
    print(C2)
    SNAP.set_taper_params(Csquared=C2)
    w,T=SNAP.get_spectrum(0)
    minTrans_list.append(min(np.log(T)))
plt.figure()
plt.plot(C2_array,minTrans_list)
    