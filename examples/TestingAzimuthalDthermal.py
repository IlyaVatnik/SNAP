# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:07:22 2024

@author: Илья
"""
from SNAP import QuantumNumbersStructure
import matplotlib.pyplot as plt
import numpy as np
wave_min = 1500
wave_max = 1600
n = 1.45
R = 62514
p_max = 5
medium='SiO2'
shape='cylinder'
T_0=20

material_dispersion=True
resonances=QuantumNumbersStructure.Resonances(wave_min, wave_max, n, R, p_max, material_dispersion,shape,medium, temperature=T_0)

# resonances.plot_all(-3, 3, 'both')
# print(SellmeierCoefficientsCalculating('SiO2', 293))
# resonances.plot_int_dispersion(polarization='TM',p=1)
freqs,Dint=(resonances.get_int_dispersion(polarization='TM',p=1))
figure = plt.figure()
plt.plot(freqs*1e-12, Dint*1e-6)
plt.xlabel('Frequency, THz')
plt.ylabel('Dispersion $D_{int}$, MHz')
plt.tight_layout()

plt.figure()
for temperature in T_0+np.arange(0,20,2):
    resonances2=QuantumNumbersStructure.Resonances(wave_min, wave_max, n, R, p_max, material_dispersion,shape,medium, temperature=temperature)
    freqs,Dint2=(resonances2.get_int_dispersion(polarization='TM',p=1))
    D_thermal=Dint2-Dint[:len(Dint2)]

    plt.plot(D_thermal*1e-6, label=temperature)
plt.xlabel('Mode number')
plt.ylabel('Dispersion $D_{thermal}$, MHz')
plt.legend()
plt.tight_layout()
# plt.plot


