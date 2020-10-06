# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:30:03 2020

@author: Ilya
"""

import numpy as np
import matplotlib.pyplot as plt
from SNAP_model import SNAP_model,SNAP_fittingExp as SNAP_exp
import pickle
from scipy import interpolate
from scipy.optimize import minimize as sp_minimize
import scipy.signal
from  scipy.ndimage import center_of_mass

FolderPath=''
DataFile='Processed_spectrogram_one_peak_cropped.pkl'
ERV_File='Processed_spectrogram_one_peak_cropped_ERV.txt'
    

x_exp,wavelengths,exp_data=SNAP_exp.load_exp_data(FolderPath+DataFile)
x_ERV,ERV_est,lambda_0=SNAP_exp.load_ERV_estimation_data(FolderPath+ERV_File)
width_est=(x_ERV[-1]-x_ERV[0])/6
fig_exp=SNAP_exp.plot_exp_data(x_exp,wavelengths,exp_data,lambda_0)

###################
N=150
x_num=np.linspace(min(x_exp),max(x_exp),N)
(absS,phaseS,ReD,ImD_exc,C2)=(0.9,0.05,+1e-3,4e-4,1e-4)
taper_params=(absS,phaseS,ReD,ImD_exc,C2)
x_center_0=x_ERV[np.argmax(ERV_est)]
init_ERV_params=[width_est,np.nanmax(ERV_est)*1.2]
ERV=SNAP_exp.ERV_gauss(x_num,x_center_0,init_ERV_params)

lambda_0=1552.32
######################

SNAP=SNAP_model.SNAP(x_num,ERV,wavelengths,lambda_0)
SNAP.set_taper_params(*taper_params)
SNAP.plot_ERV() 
fig_num=SNAP.plot_spectrogram()
fig_num.axes[0].set_xlim((x_exp[0],x_exp[-1]))

################
x_0=x_center_0
waves,T=SNAP.get_spectrum(x_0)
plt.figure(45)
plt.plot(waves,T)
ind_exp=np.argmin(abs(x_exp-x_0))
plt.plot(wavelengths,exp_data[:,ind_exp])
#
##########################
ERV_f=SNAP_exp.ERV_gauss
ERV_params_bounds=((-np.inf,np.inf),(np.nanmax(ERV_est)*0.7,np.nanmax(ERV_est)*1.3))
# res,ERV_params=SNAP_exp.optimize_ERV_shape(ERV_f,init_ERV_params,x_center_0,x_num,wavelengths,lambda_0,
#                                     taper_params,[x_exp,exp_data],max_iter=10)
res,x_0_ERV=SNAP_exp.optimize_ERV_position(ERV_f,x_center_0,x_num,init_ERV_params,
                                           wavelengths,lambda_0,taper_params,[x_exp,exp_data],max_iter=5)
# ############################################

ERV_array=SNAP_exp.ERV_gauss(x_num,x_center_0,ERV_params)
print(ERV_params)
SNAP=SNAP_model.SNAP(x_num,ERV_array,wavelengths,lambda_0)
SNAP.set_taper_params(*taper_params)
SNAP.plot_spectrogram(plot_ERV=True)
fig_num.axes[0].set_xlim((x_exp[0],x_exp[-1]))
plt.show()

###########
x_0=x_center_0
waves,T=SNAP.get_spectrum(x_0)
plt.figure(46)
plt.plot(waves,T)
ind_exp=np.argmin(abs(x_exp-x_0))
plt.plot(wavelengths,exp_data[:,ind_exp])


SNAP_exp.find_exp_modes(wavelengths,lambda_0,exp_data)
s=np.nanmean(exp_data,axis=0)
plt.figure()
plt.plot(x_exp,s)
pl.tfigure()
plt.pcolor(x_exp,wavelengths,exp_data)
