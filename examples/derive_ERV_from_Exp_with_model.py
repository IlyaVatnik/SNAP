# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:30:03 2020

@author: Ilya
"""

import numpy as np
import matplotlib.pyplot as plt
from SNAP import SNAP_model
from SNAP import SNAP_experiment




FolderPath=''
DataFile='Processed_spectrogram 1527_cropped_cropped.pkl'
Initial=1
lambda_0=1527.78
x_center=507
max_iter=15

SNAP_exp=SNAP_experiment.SNAP()
SNAP_exp.load_data(FolderPath+DataFile)
SNAP_exp.plot_spectrogram()
x_ERV,ERV_est,_=SNAP_exp.extract_ERV(0.3)
width_est=(x_ERV[-1]-x_ERV[0])/7
ERV_max_est=np.nanmax(ERV_est)
print(SNAP_exp.find_modes())
if Initial:
# ###################
    N=200
    x_num=np.linspace(min(SNAP_exp.x),max(SNAP_exp.x),N)
    (absS,phaseS,ReD,ImD_exc,C2)=(0.78,-0.5,+2e-3,4e-4,5e-4)
    taper_params=(absS,phaseS,ReD,ImD_exc,C2)
    
    ERV_params=[width_est,ERV_max_est,0]
    ERV=SNAP_experiment.ERV_gauss(x_num,x_center,ERV_params)
    SNAP_num=SNAP_model.SNAP(x_num,ERV,SNAP_exp.wavelengths,lambda_0)
    SNAP_num.set_taper_params(*taper_params)
# # ######################
else:
    SNAP_num=SNAP_model.SNAP.loader()
    x_num=SNAP_num.x
    ERV_params=SNAP_num.ERV_params
    taper_params=SNAP_num.get_taper_params()    
    lambda_0=SNAP_num.lambda_0

SNAP_num.plot_ERV() 
fig_num=SNAP_num.plot_spectrogram()
fig_num.axes[0].set_xlim((SNAP_exp.x[0],SNAP_exp.x[-1]))
print(SNAP_num.find_modes())

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
res,ERV_params=SNAP_experiment.optimize_ERV_shape_by_modes(ERV_f,ERV_params,x_center,x_num,lambda_0,
                                                  taper_params,SNAP_exp,max_iter=max_iter)
# ############################################

ERV_array=SNAP_experiment.ERV_gauss(x_num,x_center,ERV_params)
print(ERV_params)
SNAP_num=SNAP_model.SNAP(x_num,ERV_array,SNAP_exp.wavelengths,lambda_0)
SNAP_num.set_taper_params(*taper_params)
SNAP_num.ERV_params=ERV_params
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

SNAP_num.save()




