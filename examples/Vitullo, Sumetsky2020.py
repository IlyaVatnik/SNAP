# -*- coding: utf-8 -*-
"""
Data from [1] Vitullo, D. L. P., Zaki, S., Jones, D. E., Sumetsky, M. and Brodsky, M., “Coupling between waveguides and microresonators: the local approach,” Opt. Express 28(18), 25908 (2020).

@author: Ilya
"""

import numpy as np
import matplotlib.pyplot as plt
from SNAP import SNAP_model

N=300
lambda_0=1552.21
wave_min,wave_max,res=1552.2,1552.6, 2e-4

x=np.linspace(-250,250,N)
lambda_array=np.arange(wave_min,wave_max,res)

A=3.2744
sigma=123.5934
p=1.1406
def ERV(x):
    return A*np.exp(-(x**2/2/sigma**2)**p)

ERV=np.array(list(map(ERV,x)))

SNAP=SNAP_model.SNAP(x,ERV,lambda_array,lambda_0=lambda_0,res_width=1e-4,R_0=38/2)
SNAP.set_taper_params(absS=np.sqrt(0.9),phaseS=-0.05,ReD=0.025,ImD_exc=0.009,Csquared=0.025)
SNAP.plot_spectrogram(plot_ERV=False,scale='log')
plt.gcf().axes[0].set_ylim((1552.46,1552.5))
plt.xlim((-150,150))
SNAP.plot_spectrum(0,scale='log')

green_function_squared_at_0=SNAP._calculate_GreenFunction_squared_at_point(x0=0)
print(min(green_function_squared_at_0),max(green_function_squared_at_0))
SNAP.print_taper_params()
