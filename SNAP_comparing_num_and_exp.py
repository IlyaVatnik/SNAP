# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 02:45:03 2021

@author: t-vatniki
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import bottleneck as bn
from scipy import interpolate
from scipy.optimize import minimize as sp_minimize
import scipy.signal
from  scipy.ndimage import center_of_mass
import SNAP_model
import SNAP_experiment


def _difference_between_exp_and_num(x_exp,exp_data,x_num,num_data,lambdas):
    f = interpolate.interp2d(x_num, lambdas, num_data, kind='quintic')
#    print(np.shape(exp_data),np.shape(f(x_exp,lambdas)))
#    return signal.correlate(exp_data,np.reshape(f(x_exp,lambdas),-1))
    t=np.sum((exp_data-(f(x_exp,lambdas)))**2)    
    return t



def _difference_for_ERV_position(x_0_ERV,*details):
    ERV_f,ERV_params,x,wavelengths,lambda_0,taper_params,x_exp,signal_exp=details
    ERV_array=np.squeeze(ERV_f(x,x_0_ERV,ERV_params))
    SNAP=SNAP_model.SNAP(x,ERV_array,wavelengths,lambda_0)
    # print(ERV_array)
    SNAP.set_taper_params(*taper_params)
    x_num,lambdas,num_data=SNAP.derive_transmission()
    x_center_num=x_num[int(center_of_mass(num_data)[1])]
    x_center_exp=x_exp[int(center_of_mass(signal_exp)[1])]
    t=abs(x_center_exp-x_center_num)
    print('num={},exp={},difference in mass centers is {}'.format(x_center_num,x_center_exp,t))
    return t
        
        
    # return difference_between_exp_and_num(x_exp,signal_exp,x,num_data,lambdas)





def optimize_taper_params(x,ERV,wavelengths,lambda_0,init_taper_params,SNAP_exp,bounds=None,max_iter=5):
    def _difference_on_taper(taper_params,*details):
        (absS,phaseS,ReD,ImD_exc,C)=taper_params
        x,ERV,wavelengths,lambda_0,x_exp,signal_exp=details
        SNAP=SNAP_model.SNAP(x,ERV,wavelengths,lambda_0)
        SNAP.set_taper_params(absS,phaseS,ReD,ImD_exc,C)
        x,wavelengths,num_data=SNAP.derive_transmission()
        t=_difference_between_exp_and_num(x_exp,signal_exp,x,num_data,wavelengths)
        print('taper opt. delta_t is {}'.format(t))
        return t
    
    if SNAP_exp.transmission_scale=='log':
        SNAP_exp.convert_to_lin_transmission()
    x_exp,signal_exp=SNAP_exp.x,SNAP_exp.transmission
    options={}
    options['maxiter']=max_iter 
    [absS,phaseS,ReD,ImD_exc,C]=init_taper_params # use current taper parameters as initial guess
    res=sp_minimize(_difference_on_taper,[absS,phaseS,ReD,ImD_exc,C],args=(x,ERV,wavelengths,lambda_0,x_exp,signal_exp),bounds=bounds,options=options)
    taper_params=res.x
    return res, taper_params
   

def optimize_ERV_shape_by_modes(ERV_f,initial_ERV_params,x_0_ERV,x,lambda_0,
                       taper_params,SNAP_exp,bounds=None,max_iter=20):
    
    def _difference_for_ERV_shape(ERV_params,*details):
        ERV_f,x_0_ERV,x,wavelengths,lambda_0,taper_params,SNAP_exp=details
        exp_modes=SNAP_exp.find_modes()
        ERV_array=ERV_f(x,x_0_ERV,ERV_params)
        SNAP=SNAP_model.SNAP(x,ERV_array,wavelengths,lambda_0)
        SNAP.set_taper_params(*taper_params)
        x_num,wavelengths,num_data=SNAP.derive_transmission()
        num_modes=SNAP.find_modes()
        N_num=len(num_modes)
        N_exp=len(exp_modes)
        if N_num>N_exp:
            exp_modes=np.sort(np.append(exp_modes,lambda_0*np.ones((N_num-N_exp,1))))
        elif N_exp>N_num:
            num_modes=np.sort(np.append(num_modes,lambda_0*np.ones((N_exp-N_num,1))))
        t=np.sum(abs(num_modes-exp_modes))
        print('mode positions difference is {}'.format(t))
        return t

    if SNAP_exp.transmission_scale=='log':
        SNAP_exp.convert_to_lin_transmission()
    wavelengths=SNAP_exp.wavelengths
    options={}
    options['maxiter']=max_iter  
    [absS,phaseS,ReD,ImD_exc,C]=taper_params # use current taper parameters as initial guess
    res=sp_minimize(_difference_for_ERV_shape,initial_ERV_params,args=(ERV_f,x_0_ERV,x,wavelengths,lambda_0,taper_params,SNAP_exp),
                    bounds=bounds,options=options,method='Nelder-Mead')
    ERV_params=res.x
    return res, ERV_params


def optimize_ERV_shape_by_whole_transmission(ERV_f,initial_ERV_params,x_0_ERV,x,lambda_0,
                       taper_params,SNAP_exp,bounds=None,max_iter=20):
    
    def _difference_for_ERV_shape(ERV_params,*details):
        ERV_f,x_0_ERV,x,wavelengths,lambda_0,taper_params,SNAP_exp=details
        ERV_array=ERV_f(x,x_0_ERV,ERV_params)
        SNAP=SNAP_model.SNAP(x,ERV_array,wavelengths,lambda_0)
        SNAP.set_taper_params(*taper_params)
        x_num,wavelengths,num_data=SNAP.derive_transmission()
        t=_difference_between_exp_and_num(SNAP_exp.x,SNAP_exp.transmission,x_num,num_data,wavelengths)
        print('ERV opt, delta_t is {}'.format(t))
        return t
    

    if SNAP_exp.transmission_scale=='log':
        SNAP_exp.convert_to_lin_transmission()
    wavelengths=SNAP_exp.wavelengths
    options={}
    options['maxiter']=max_iter  
    [absS,phaseS,ReD,ImD_exc,C]=taper_params # use current taper parameters as initial guess
    res=sp_minimize(_difference_for_ERV_shape,initial_ERV_params,args=(ERV_f,x_0_ERV,x,wavelengths,lambda_0,taper_params,SNAP_exp),
                    bounds=bounds,options=options,method='Nelder-Mead')
    ERV_params=res.x
    return res, ERV_params

# def optimize_ERV_position(ERV_f,initial_x_0_ERV,x,ERV_params,
#                           wavelengths,lambda_0,taper_params,exp_data,
#                           bounds=None,max_iter=20):
#     x_exp,signal_exp=exp_data[0],exp_data[1]
#     options={}
#     options['maxiter']=max_iter  
#     [absS,phaseS,ReD,ImD_exc,C]=taper_params # use current taper parameters as initial guess
#     res=sp_minimize(_difference_for_ERV_position,initial_x_0_ERV,
#                     args=(ERV_f,ERV_params,x,wavelengths,lambda_0,taper_params,x_exp,signal_exp),
#                     bounds=bounds,options=options,method='Nelder-Mead')
#     x_0_ERV_res=res.x
#     return res, x_0_ERV_res



def ERV_gauss(x,x_0_ERV,ERV_params):
    sigma=ERV_params[0]
    A=ERV_params[1]
    K=ERV_params[2]
    x_0=x_0_ERV
    return np.array(list(map(lambda x:K+np.exp(-(x-x_0)**2/2/sigma**2)*A,x)))
