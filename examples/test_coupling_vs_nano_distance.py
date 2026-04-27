# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:00:02 2026

@author: Илья
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar as minimize_scalar

wave=1.55
def T(z):
    return 1-np.exp(-2*z/wave)



T_z0=1

T_z0_delta_z=T_z0*0.9
delta_z=2


def func(z):
    return T(z)/T_z0-T(z-delta_z)/T_z0_delta_z
    
res=minimize_scalar(func)
print(res.x)

#%%
delta_z=0.5
z0_array=np.arange(0.1,10,0.01)
T1=T(z0_array-delta_z)
T2=T(z0_array)*0.9

'''
Решаем численно выражение

(1-exp(-(z-delta_z)/lambda)) / (1-exp(-(z)/lambda)) =0.9, при изменении расстояния между тейпером и ми крорезонатором на 1 мкм пропускание упало на 10 % 
Все приблизительно
'''
plt.figure()
plt.plot(z0_array,T1)
plt.plot(z0_array,T2)





    