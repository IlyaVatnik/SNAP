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
DataFile='Processed_spectrogram_one_mode.pkl'

    

SNAP_exp=SNAP_experiment.SNAP()
SNAP_exp.load_data(FolderPath+DataFile)
fig_exp=SNAP_exp.plot_data()

###################
N=300
x_num=np.linspace(min(SNAP_exp.x),max(SNAP_exp.x),N)
(absS,phaseS,ReD,ImD_exc,C2)=(0.9,0.05,+1e-2,4e-4,1e-4)
taper_params=(absS,phaseS,ReD,ImD_exc,C2)

x_center=7971
ERV_params=[5.15689300e+01,  6.10237464e+00, 0*-9.58333333e-04]
ERV=SNAP_experiment.ERV_gauss(x_num,x_center,ERV_params)

lambda_0=1552.32
######################

SNAP_num=SNAP_model.SNAP(x_num,ERV,SNAP_exp.wavelengths,lambda_0)
SNAP_num.set_taper_params(*taper_params)
SNAP_num.plot_ERV() 
fig_num=SNAP_num.plot_spectrogram(plot_ERV=True)
fig_num.axes[0].set_xlim((SNAP_exp.x[0],SNAP_exp.x[-1]))

res,taper_params=SNAP_experiment.optimize_taper_params(x_num,ERV,SNAP_exp.wavelengths,lambda_0,
                                                       taper_params,SNAP_exp,max_iter=5)

################
x_0=x_center
waves,T=SNAP_num.get_spectrum(x_0)
plt.figure(45)
plt.plot(waves,T)
ind_exp=np.argmin(abs(SNAP_exp.x-x_0))
plt.plot(SNAP_exp.wavelengths,SNAP_exp.transmission[:,ind_exp])
#

###########
x_0=x_center
SNAP_num.set_taper_params(*taper_params)
SNAP_num.derive_transmission()
waves,T=SNAP_num.get_spectrum(x_0)
plt.figure(46)
plt.plot(waves,T)
ind_exp=np.argmin(abs(SNAP_exp.x-x_0))
plt.plot(SNAP_exp.wavelengths,SNAP_exp.transmission[:,ind_exp])


SNAP_num.plot_spectrogram()
print(SNAP_num.get_taper_params())

