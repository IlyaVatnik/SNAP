# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:30:03 2020

@author: Ilya Vatnik

v.1 
"""

import numpy as np
import matplotlib.pyplot as plt
from . import SNAP_model
import pickle
import bottleneck as bn
from scipy import interpolate
from scipy.optimize import minimize as sp_minimize

R_0=62.5
Cmap='jet'

def load_exp_data(file_name):
    number_of_axis={'X':0,'Y':1,'Z':2,'W':3,'p':4}

    print('loading data for analyzer from ',file_name)
    f=open(file_name,'rb')
    D=(pickle.load(f))
    f.close()
    axis=D['axis']
    Positions=np.array(D['Positions'])
    wavelengths,exp_data=D['Wavelengths'],D['Signal']
    exp_data=10**((exp_data-np.max(exp_data))/10)
    x=Positions[:,number_of_axis[axis]]*2.5
    return x,wavelengths,exp_data
    
def load_ERV_estimation_data(file_name):
    global R_0
    A=np.loadtxt(file_name)
    x_ERV=A[:,0]*2.5
    Waves=A[:,1]
    lambda_0=np.nanmin(Waves)
    ERV=(Waves-lambda_0)/np.nanmean(Waves)*R_0*1e3

    if (max(np.diff(x_ERV))-min(np.diff(x_ERV)))>0:
        f = interpolate.interp1d(x_ERV, ERV)
        x_ERV=np.linspace(min(x_ERV),max(x_ERV),len(x_ERV))
        ERV=f(x_ERV)
    return x_ERV,ERV,lambda_0

def plot_exp_data(x,w,signal,lambda_0):
    w_0=np.mean(w)
    def _convert_ax_Wavelength_to_Radius(ax_Wavelengths):
        """
        Update second axis according with first axis.
        """
        y1, y2 = ax_Wavelengths.get_ylim()
        nY1=(y1-lambda_0)/w_0*R_0*1e3
        nY2=(y2-lambda_0)/w_0*R_0*1e3
        ax_Radius.set_ylim(nY1, nY2)

    fig=plt.figure(10)
    plt.clf()
    ax_Wavelengths = fig.subplots()
    ax_Radius = ax_Wavelengths.twinx()
    ax_Wavelengths.callbacks.connect("ylim_changed", _convert_ax_Wavelength_to_Radius)
    try:
        im = ax_Wavelengths.pcolorfast(x,w,signal,cmap=Cmap)
    except:
        im = ax_Wavelengths.contourf(x,w,signal,cmap=Cmap)
    plt.colorbar(im,ax=ax_Radius,pad=0.12)
    ax_Wavelengths.set_xlabel(r'Position, $\mu$m')
    ax_Wavelengths.set_ylabel('Wavelength, nm')
    ax_Radius.set_ylabel('Variation, nm')
    plt.title('experiment')
    plt.tight_layout()
    return fig
    
def difference_on_taper_at_distinct_x(taper_params,*details):
    (absS,phaseS,ReD,ImD_exc,C)=taper_params
    SNAP=details[0]
    x_0=details[1]
    exp_data=details[2]
    SNAP.set_taperParams(absS,phaseS,ReD,ImD_exc,C)
    lambdas,num_data=SNAP.get_spectrum(x_0)
    return np.sum(abs(exp_data-num_data))

def difference_on_taper(taper_params,*details):
    (absS,phaseS,ReD,ImD_exc,C)=taper_params
    x,ERV,lambdas,lambda_0,x_exp,exp_data=details
    SNAP=SNAP_model.SNAP(x,ERV,lambdas,lambda_0)
    SNAP.set_taperParams(absS,phaseS,ReD,ImD_exc,C)
    x,lambdas,num_data=SNAP.derive_transmission()
    f = interpolate.interp2d(x, lambdas, num_data, kind='cubic')
    return np.sum(abs(exp_data-f(x_exp,lambdas)))


def optimize_taper_params_at_x(x_0,x,ERV,wavelengths,lambda_0,init_taper_params,exp_data):
    SNAP=SNAP_model(x,ERV,wavelengths,lambda_0)
    bounds=((0,1),(0,1),(0.0001,0.05),(0,5e-2),(1e-5,5e-2))
    i_x_exp=np.argmin(abs(x-x_0))
    options={}
    options['maxiter']=5
    (absS,phaseS,ReD,ImD_exc,C)=init_taper_params # use current taper parameters as initial guess
    res=sp_minimize(difference_on_taper_at_distinct_x,[absS,phaseS,ReD,ImD_exc,C],args=(SNAP,x_0,exp_data[:,i_x_exp]),bounds=bounds,options=options)
    plt.figure(20)
    plt.plot(wavelengths,exp_data[:,i_x_exp])
    lambdas,transmission=SNAP.get_spectrum(x_0)
    plt.plot(lambdas,transmission)
    plt.title('X={}'.format(x[i_x_exp]))
    return res

def optimize_taper_params(x,ERV,wavelengths,lambda_0,init_taper_params,x_exp,exp_data,bounds):
    options={}
    options['maxiter']=1  
    [absS,phaseS,ReD,ImD_exc,C]=init_taper_params # use current taper parameters as initial guess
    res=sp_minimize(difference_on_taper,[absS,phaseS,ReD,ImD_exc,C],args=(x,ERV,wavelengths,lambda_0,x_exp,exp_data),bounds=bounds,options=options)
    taper_params=res.x
    return res, taper_params
    


def ERV_gauss(ERV_params,*details):
    x_0=ERV_params[0]
    sigma=ERV_params[1]
    A=ERV_params[2]
    x=details[0]
    return np.array(list(map(lambda x:np.exp(-(x-x_0)**2/2/sigma**2)*A,x)))

def difference_on_ERV(ERV_params,*details):
    x,wavelengths,lambda_0,taper_params,x_exp,exp_data=details
    ERV=ERV_gauss(ERV_params,*details)
    SNAP=SNAP_model.SNAP(x,ERV,wavelengths,lambda_0)
    SNAP.set_taperParams(*taper_params)
    SNAP.plot_spectrogram()
    x,lambdas,num_data=SNAP.derive_transmission()
    f = interpolate.interp2d(x, lambdas, num_data, kind='cubic')
    return bn.nansum(abs(exp_data-bn.nanmean(exp_data)*(f(x_exp,lambdas)-bn.nanmean(num_data))))

def optimize_ERV(x,initial_ERV_params,wavelengths,lambda_0,taper_params,x_exp,exp_data,bounds=None):
    options={}
    options['maxiter']=10  
    [absS,phaseS,ReD,ImD_exc,C]=taper_params # use current taper parameters as initial guess
    res=sp_minimize(difference_on_ERV,initial_ERV_params,args=(x,wavelengths,lambda_0,taper_params,x_exp,exp_data),bounds=bounds,options=options)
    ERV_params=res.x
    return res, ERV_params


