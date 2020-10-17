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
DataFile='Processed_spectrogram_cropped.pkl'
Initial=0
    

SNAP_exp=SNAP_experiment.SNAP()
SNAP_exp.load_data(FolderPath+DataFile)
fig_exp=SNAP_exp.plot_spectrogram()
x_center=7310

taper_params_bounds=((0,1),(0,1),(-4e-4,5e-4),(0,1e-2),(0,1e-2))

###################
if Initial:
    N=100
    x_num=np.linspace(min(SNAP_exp.x),max(SNAP_exp.x),N)
    (absS,phaseS,ReD,ImD_exc,C2)=(0.6,0.8,+1e-5,4e-4,1e-3)
    taper_params=(absS,phaseS,ReD,ImD_exc,C2)
    
    ERV_params=[5.70936663e+01,  5.89798060e+00, -1.14857411e-04]
    ERV=SNAP_experiment.ERV_gauss(x_num,x_center,ERV_params)
    lambda_0=1550.775
    
    SNAP_num=SNAP_model.SNAP(x_num,ERV,SNAP_exp.wavelengths,lambda_0)
    SNAP_num.set_taper_params(*taper_params)

else:
    SNAP_num=SNAP_model.SNAP.loader()


################
x_0=x_center
waves,T=SNAP_num.get_spectrum(x_0)
plt.figure(45)
plt.plot(waves,T)
ind_exp=np.argmin(abs(SNAP_exp.x-x_0))
plt.plot(SNAP_exp.wavelengths,SNAP_exp.transmission[:,ind_exp])
###########



res,taper_params=SNAP_experiment.optimize_taper_params(SNAP_num.x,SNAP_num.ERV,SNAP_exp.wavelengths,SNAP_num.lambda_0,
                                                       SNAP_num.get_taper_params(),SNAP_exp,taper_params_bounds,max_iter=30)


###########
x_0=x_center
SNAP_num.set_taper_params(*taper_params)
SNAP_num.derive_transmission()
waves,T=SNAP_num.get_spectrum(x_0)
plt.figure(46)
plt.plot(waves,T)
ind_exp=np.argmin(abs(SNAP_exp.x-x_0))
plt.plot(SNAP_exp.wavelengths,SNAP_exp.transmission[:,ind_exp])
SNAP_num.save()

fig=SNAP_num.plot_spectrogram()
fig.axes[0].set_xlim((min(SNAP_exp.x),max(SNAP_exp.x)))
fig.axes[0].set_ylim((min(SNAP_exp.wavelengths),max(SNAP_exp.wavelengths)))

print('absS,phaseS,ReD,ImD_exc,C2=',SNAP_num.get_taper_params())


