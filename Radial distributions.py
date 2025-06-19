# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:48:02 2019

@author: Ilya
"""

# -*- coding: utf-8 -*-
"""
Based on 
1. Y. A. Demchenko and M. L. Gorodetsky, "Analytical estimates of eigenfrequencies, dispersion, and field distribution in whispering gallery resonators," J. Opt. Soc. Am. B 30, 3056 (2013).
(23)



"""


date='01.01.2024'
version='0.3'


import os
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import special


m=364 ## Asimuthal Number of WGM 
p=1
R_0=62.5 # micron
n=1.445



r_min=R_0*0.8
r_max=R_0*1.1
step=R_0*0.00001 # Number of points


T_mp=special.jn_zeros(m,p)[p-1]

def E(x,R,pol='TE'): #phase not considered
    if pol=='TE':
        P=1
    elif pol=='TM':
        P=1/n**2
    k_0=m/R/n
    gamma=np.sqrt(n**2-1)*k_0
    R_eff=R+P/gamma
    
   
    if x<R:
        return special.jn(m,x/R_eff*T_mp)
    else:
        return 1/P *special.jn(m,R/R_eff*T_mp)*np.exp(-gamma*(x-R))

F = np.vectorize(E)

Rarray=np.arange(r_min,r_max,step)
F_TM_Array=abs(F(Rarray,pol='TM', R=R_0))**2
F_TE_Array=abs(F(Rarray,pol='TE',R=R_0))**2
#

plt.figure(1)    
plt.clf()
plt.plot(Rarray,F_TM_Array,label='TM')
plt.plot(Rarray,F_TE_Array,label='TE')
plt.vlines(R_0,0,np.max(F_TE_Array),color='black')
plt.xlabel('radius, um')
plt.ylabel('Intensity,arb.u.')
plt.legend()

#R_TM_Max=Rarray[(np.argmax(F_TM_Array))]
#R_TE_Max=Rarray[(np.argmax(F_TE_Array))]
#print('Positions of maximum of TM and TE WGMS are ', R_TM_Max,' and ',R_TE_Max)
#print('X_TM_Max/X_TE_Max= ',R_TM_Max/R_TE_Max)
#




