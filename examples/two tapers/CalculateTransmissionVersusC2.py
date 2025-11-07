# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:57:44 2020

@author: Ilya
"""
import numpy as np
import matplotlib.pyplot as plt
from SNAP import SNAP_model

C2_array=np.arange(0.0001,0.08,0.001)
x_0=-100

N=200
lambda_0=1552.21
wave_min,wave_max,res=1552.46,1552.5, 3e-4

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
SNAP=SNAP_model.SNAP(x,ERV,lambda_array,lambda_0=lambda_0,res_width=1e-14,R_0=38/2)
SNAP.set_taper_params(absS=np.sqrt(0.8),phaseS=0.00,ReD=0.01,ImD_exc=0.1e-3,Csquared=min(C2_array))
SNAP.plot_spectrogram(plot_ERV=False,scale='log')
# SNAP.plot_spectrum(x_0,scale='lin')
SNAP.plot_spectrum(x_0,scale='log')

minTrans_list=[]
for C2 in C2_array:
    print(C2)
    SNAP.set_taper_params(Csquared=C2)
    w,T=SNAP.get_spectrum(x_0)
    minTrans_list.append(min(np.log10(T)))
plt.figure()
plt.plot(C2_array,minTrans_list)
plt.axvline(SNAP.critical_Csquared(),color='red',label='Sumetsky')
plt.axvline(SNAP.critical_Csquared_1(x_0),color='green', label='My_1')
plt.axvline(SNAP.critical_Csquared_2(),color='black',label='My_2')
plt.legend()

print(SNAP.critical_Csquared(),SNAP.critical_Csquared_1(x_0),SNAP.critical_Csquared_2())


    