# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 23:29:57 2020

@author: t-vatniki
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from SNAP import SNAP_model
from SNAP import SNAP_experiment

FolderPath=''
DataFile='Processed_spectrogram 1527_cropped_cropped.pkl'
Total_iterations_from_ERV_to_taper=5
max_iter=30
x_center=507
taper_params=(0.78,-0.5,+2e-3,4e-4,5e-4) # initial  (absS,phaseS,ReD,ImD_exc,C2)
taper_params_bounds=((0.5,1),(-1,1),(-3e-3,3e-3),(0,1e-1),(0,1e-1)) #bounds for  (absS,phaseS,ReD,ImD_exc,C2)

SNAP_exp=SNAP_experiment.SNAP()
SNAP_exp.load_data(FolderPath+DataFile)
SNAP_exp.plot_spectrogram()
plt.savefig('Experimental spectrogram.png')
lambda_0=min(SNAP_exp.wavelengths)

x_ERV,ERV_est,_=SNAP_exp.extract_ERV(0.2)
width_est=(x_ERV[-1]-x_ERV[0])/6
ERV_max_est=np.nanmax(ERV_est)
ERV_init_params=[width_est,ERV_max_est,0]


N=100
x_num=np.linspace(min(SNAP_exp.x),max(SNAP_exp.x),N)
ERV=SNAP_experiment.ERV_gauss(x_num,x_center,ERV_init_params)
SNAP_num=SNAP_model.SNAP(x_num,ERV,SNAP_exp.wavelengths,lambda_0)
SNAP_num.set_taper_params(*taper_params)

ERV_f=SNAP_experiment.ERV_gauss

res_ERV,ERV_params=SNAP_experiment.optimize_ERV_shape_by_modes(ERV_f,ERV_init_params,x_center,x_num,lambda_0,
                                                  taper_params,SNAP_exp,max_iter=max_iter)
ERV_array=SNAP_experiment.ERV_gauss(x_num,x_center,ERV_params)
res_taper,taper_params=SNAP_experiment.optimize_taper_params(x_num,ERV_array,SNAP_exp.wavelengths,SNAP_num.lambda_0,
                                                       taper_params,SNAP_exp,taper_params_bounds,max_iter=max_iter)

    
for i in range(Total_iterations_from_ERV_to_taper):

    res_ERV,ERV_params=SNAP_experiment.optimize_ERV_shape_by_whole_transmission(ERV_f,ERV_params,x_center,x_num,lambda_0,
                                                      taper_params,SNAP_exp,max_iter=max_iter)
    ERV_array=SNAP_experiment.ERV_gauss(x_num,x_center,ERV_params)
    res_taper,taper_params=SNAP_experiment.optimize_taper_params(x_num,ERV_array,SNAP_exp.wavelengths,SNAP_num.lambda_0,
                                                       taper_params,SNAP_exp,taper_params_bounds,max_iter=max_iter)


ERV_array=SNAP_experiment.ERV_gauss(x_num,x_center,ERV_params)
SNAP_num=SNAP_model.SNAP(x_num,ERV_array,SNAP_exp.wavelengths,lambda_0)
SNAP_num.set_taper_params(*taper_params)
SNAP_num.ERV_params=ERV_params
SNAP_num.plot_spectrogram(plot_ERV=True)
plt.show()
plt.savefig('Final numerical spectrogram.png')

x_0=x_center-ERV_params[0]/2
waves,T=SNAP_num.get_spectrum(x_0)
plt.figure(45)
plt.clf()
plt.plot(waves,T)
ind_exp=np.argmin(abs(SNAP_exp.x-x_0))
plt.plot(SNAP_exp.wavelengths,SNAP_exp.transmission[:,ind_exp])
plt.xlabel('Wavelength,nm')
plt.ylabel('Transmission')
plt.savefig('Single spectrum comparison at x={}.png'.format(x_0))
#
print('ERV_params (width_est,ERV_max,K0)=',ERV_params)
print('absS,phaseS,ReD,ImD_exc,C2=',taper_params)
Dict={}
Dict['ERV_params (width, height, background)']=ERV_params.tolist()
Dict['taper_params (absS, phaseS, ReD, ImD_exc, C2)']=taper_params.tolist()
Dict['minimum ImD,Critical coupling C2']=[SNAP_num.critical_Csquared(),SNAP_num.min_imag_D()]
with open('Results.txt','w') as f:
    json.dump(Dict,f)
