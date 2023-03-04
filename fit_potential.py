# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:02:34 2023

@author: Илья
"""
import numpy as np
from SNAP import SNAP_model
import matplotlib.pyplot as plt
import scipy.optimize as sciopt
exp_modes=[1540.5505,1540.510, 1540.4680,1540.43725]
lambda_0=1540.4256


A=10
sigma=50
x_0=0



N=800
x_min=-300
x_max=300

wave_min,wave_max,res=lambda_0-1e-2,lambda_0+A/62.5e3*lambda_0*1.5, 1e-3


A_1=A
A_2=-A/5
sigma_1=sigma
sigma_2=sigma/5


p0=[A_1,sigma_1,A_2,sigma_2]
# bounds=((0,30),(10,200),(0,30),(10,200))
# upper_b=(30,200,30,200)

def ERV(x,A_1,sigma_1,A_2,sigma_2):
    # if abs(x)<=200:
#            return (x)**2
    return A_1*np.exp(-((x-x_0)**2/2/sigma_1**2))+A_2*np.exp(-((x-x_0)**2/2/sigma_2**2))

# def ERV(x,A,sigma):
#     # if abs(x)<=200:
# #            return (x)**2
#     return A/(1+(x-x_0)**2/sigma**2)


ERV=np.vectorize(ERV)

x=np.linspace(x_min,x_max,N)
lambda_array=np.arange(wave_min,wave_max,res)


    
# SNAP=SNAP_model.SNAP(x,ERV(x,A,sigma),lambda_array,lambda_0=lambda_0,res_width=1e-4,R_0=62.5)
# modes=SNAP.find_modes()
# fig=SNAP.plot_spectrogram(plot_ERV=True,scale='log')

def cost_function( params):

        def closest_argmin(A, B): # from https://stackoverflow.com/questions/45349561/find-nearest-indices-for-one-array-against-all-values-in-another-array-python
            L = B.size
            sidx_B = B.argsort()
            sorted_B = B[sidx_B]
            sorted_idx = np.searchsorted(sorted_B, A)
            sorted_idx[sorted_idx==L] = L-1
            mask = (sorted_idx > 0) & \
            ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
            return sidx_B[sorted_idx-mask]     

        def measure(exp, theory):
            closest_indexes=closest_argmin(exp, theory)
            return sum((exp-theory[closest_indexes])**2)/len(theory)
        
        SNAP=SNAP_model.SNAP(x,ERV(x,*params),lambda_array,lambda_0=lambda_0,res_width=1e-4,R_0=62.5)
        num_modes,num_mode_distribs=SNAP.find_modes()
        
        return measure(exp_modes,num_modes)
    
res=sciopt.minimize(cost_function,p0,method='Nelder-Mead',options={'maxiter':1000},tol=1e-12)
print(res['message'])
SNAP=SNAP_model.SNAP(x,ERV(x,*res['x']),lambda_array,lambda_0=lambda_0,res_width=1e-4,R_0=62.5, taper_ImD_exc=1e-3)

fig=SNAP.plot_spectrogram(plot_ERV=True,scale='log')
modes,m_distribs=SNAP.find_modes(plot_at_spectrogram=True)

print(modes)
for mode in exp_modes:
    fig.axes[0].axhline(mode,color='yellow')
print(*res['x'])


plt.figure()
plt.plot(m_distribs**2)
