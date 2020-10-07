# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:30:03 2020

@author: Ilya
"""

import numpy as np
import matplotlib.pyplot as plt
from SNAP import SNAP_model
from SNAP import SNAP_experiment
import pickle
from scipy import interpolate
from scipy.optimize import minimize as sp_minimize
import scipy.signal
from  scipy.ndimage import center_of_mass

FolderPath=''
DataFile='Processed_spectrogram_one_peak_cropped.pkl'
ERV_File='Processed_spectrogram_one_peak_cropped_ERV.txt'
    

SNAP_exp=SNAP_experiment.SNAP()
SNAP_exp.load_data(FolderPath+DataFile)
x_ERV,ERV_est,lambda_0=SNAP_exp.load_ERV_estimation(FolderPath+ERV_File)
width_est=(x_ERV[-1]-x_ERV[0])/6
fig_exp=SNAP_exp.plot_data()

###################
N=150
x_num=np.linspace(min(SNAP_exp.x),max(SNAP_exp.x),N)
(absS,phaseS,ReD,ImD_exc,C2)=(0.9,0.05,+1e-3,4e-4,1e-4)
taper_params=(absS,phaseS,ReD,ImD_exc,C2)

x_center=SNAP_exp.find_center()
# x_center=7971
init_ERV_params=[width_est,np.nanmax(ERV_est)*1.2,0]
ERV=SNAP_experiment.ERV_gauss(x_num,x_center,init_ERV_params)

lambda_0=1552.32
######################

SNAP_num=SNAP_model.SNAP(x_num,ERV,SNAP_exp.wavelengths,lambda_0)
SNAP_num.set_taper_params(*taper_params)
SNAP_num.plot_ERV() 
fig_num=SNAP_num.plot_spectrogram()
fig_num.axes[0].set_xlim((SNAP_exp.x[0],SNAP_exp.x[-1]))

################
x_0=x_center
waves,T=SNAP_num.get_spectrum(x_0)
plt.figure(45)
plt.plot(waves,T)
ind_exp=np.argmin(abs(SNAP_exp.x-x_0))
plt.plot(SNAP_exp.wavelengths,SNAP_exp.transmission[:,ind_exp])
#
##########################
ERV_f=SNAP_experiment.ERV_gauss
ERV_params_bounds=((-np.inf,np.inf),(np.nanmax(ERV_est)*0.7,np.nanmax(ERV_est)*1.3))
res,ERV_params=SNAP_experiment.optimize_ERV_shape(ERV_f,init_ERV_params,x_center,x_num,lambda_0,
                                    taper_params,SNAP_exp,max_iter=10)
# ############################################

ERV_array=SNAP_experiment.ERV_gauss(x_num,x_center,ERV_params)
print(ERV_params)
SNAP_num=SNAP_model.SNAP(x_num,ERV_array,SNAP_exp.wavelengths,lambda_0)
SNAP_num.set_taper_params(*taper_params)
SNAP_num.plot_spectrogram(plot_ERV=True)
fig_num.axes[0].set_xlim((SNAP_exp.x[0],SNAP_exp.x[-1]))
plt.show()

###########
x_0=x_center
waves,T=SNAP_num.get_spectrum(x_0)
plt.figure(46)
plt.plot(waves,T)
ind_exp=np.argmin(abs(SNAP_exp.x-x_0))
plt.plot(SNAP_exp.wavelengths,SNAP_exp.transmission[:,ind_exp])



