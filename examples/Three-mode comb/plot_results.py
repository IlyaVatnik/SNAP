# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 19:42:14 2025

@author: Илья
"""

from SNAP import SNAP_MI_threshold
import numpy as np
import matplotlib.pyplot as plt
import pickle

q0=45

z0=30

file='results.data'
with open(file,'rb') as f:
    D=pickle.load(f)
params,q0_array,Z0_array,thresholds_array=D[0],D[1],D[2],D[3]
Q0, Z0 = np.meshgrid(q0_array, Z0_array)
index_z0=np.argmin(abs(Z0_array-z0))

plt.figure(figsize=(10, 6))
plt.contourf(Q0, Z0, thresholds_array.T, levels=20, cmap='viridis')
plt.colorbar(label='Threshold, W')
plt.xlabel('Axial pump number,q0')
plt.ylabel('Pump position Z0, mkm')
plt.title('Thresholds as a function of q0 and Z0')
plt.grid(True)
plt.show()

#%%
plt.figure(2)
plt.title(f'for Z0={Z0_array[index_z0]} mkm')
plt.plot(q0_array, thresholds_array[:,index_z0],'o')
plt.xlabel('q0')
plt.ylabel('Threshold , W')
plt.show()

plt.figure(3)
plt.plot(Z0_array, thresholds_array[q0,:],'o')
plt.xlabel('Z0,mkm')
plt.ylabel('Threshold for q0={}, W'.format(q0))
plt.show()

plt.figure(4)
plt.plot(params['z_dr'],params['dr']*1e3)
plt.xlabel('Position, mkm')
plt.ylabel('Effective radius variation, nm')
plt.show()
