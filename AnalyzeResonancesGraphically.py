# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:50:41 2020

@author: Ilya
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.fftpack import rfft, irfft, fftfreq
from scipy.signal import find_peaks
from SNAP.QuantumNumbersStructure import Resonances
import pickle
#from ComputingAzimuthalAndRadialModes import Resonances


MinimumPeakDepth=25  ## For peak searching 
MinimumPeakDistance=2000 ## For peak searching 
threshold=0.001


Wavelength_min=1530
Wavelength_max=1565

FileName1='Polarization 1.pkl'
FileName2='Polarization 2.pkl'

def get_experimental_data():
    

    
    FilterLowFreqEdge=0.00
    FilterHighFreqEdge=0.01
    def FFTFilter(y_array):
        W=fftfreq(y_array.size)
        f_array = rfft(y_array)
        Indexes=[i for  i,w  in enumerate(W) if all([abs(w)>FilterLowFreqEdge,abs(w)<FilterHighFreqEdge])]
        f_array[Indexes] = 0
    #        f_array[] = 0
        return irfft(f_array)
    with open(FileName1,'rb') as f:
        Temp=pickle.load(f)
    Wavelengths_TE=Temp[:,0]
    index_TE_min=np.argmin(abs(Wavelengths_TE-Wavelength_min))
    index_TE_max=np.argmin(abs(Wavelengths_TE-Wavelength_max))
    Signals_TE=FFTFilter(Temp[index_TE_min:index_TE_max,1])
    Wavelengths_TE=Wavelengths_TE[index_TE_min:index_TE_max]
    with open(FileName2,'rb') as f:
        Temp=pickle.load(f)
    Wavelengths_TM=Temp[:,0]
    index_TM_min=np.argmin(abs(Wavelengths_TM-Wavelength_min))
    index_TM_max=np.argmin(abs(Wavelengths_TM-Wavelength_max))
    Signals_TM=FFTFilter(Temp[index_TM_min:index_TM_max,1])
    Wavelengths_TM=Wavelengths_TM[index_TM_min:index_TM_max]
    Resonances_TE_indexes,_=find_peaks(-Signals_TE,height=MinimumPeakDepth,distance=MinimumPeakDistance,threshold=threshold)
    Resonances_TE_exp=Wavelengths_TE[Resonances_TE_indexes]
    return Wavelengths_TE,Signals_TE,Wavelengths_TM,Signals_TM,Resonances_TE_indexes,Resonances_TE_exp

Wavelengths_TE,Signals_TE,Wavelengths_TM,Signals_TM,Resonances_TE_indexes,Resonances_TE_exp=get_experimental_data()

fig, axs = plt.subplots(2, 1, sharex=True,figsize=(15, 12))
plt.subplots_adjust(left=0.1, bottom=0.3)
axs[0].plot(Wavelengths_TE,Signals_TE)
axs[0].plot(Wavelengths_TM,Signals_TM)
axs[0].plot(Wavelengths_TE[Resonances_TE_indexes],Signals_TE[Resonances_TE_indexes],'.')
axs[0].set_title('N=%d' % len(Resonances_TE_indexes))
plt.sca(axs[1])
R0 = 62.5e3
n0 = 1.44443
p0=3
delta_n = 1e-5
delta_R = 1e-3
resonances=Resonances(Wavelength_min,Wavelength_max,n0,R0,p0,dispersion=True)
tempdict=resonances.__dict__
resonances.plot_all(0,1,'both')
plt.xlim([Wavelength_min,Wavelength_max])
axs[1].set_title('N=%d,n=%f,R=%f,p_max=%d' % (resonances.N_of_resonances['Total'],n0,R0,p0))
#

#s = n0 * np.sin(2 * np.pi * R0 * t)
#l, = axs[1].plot(t, s, lw=2)
#axs[1].margins(x=0)

axcolor = 'lightgoldenrodyellow'
ax_n = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_R = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_p = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)

s_n = Slider(ax_n, 'n', 1.44, 1.45, valinit=n0, valstep=delta_n)
s_R = Slider(ax_R, 'R', 62e3, 63e3, valinit=R0,valstep=delta_R)
s_p = Slider(ax_p, 'p', 1, 5, valinit=p0,valstep=1)
#axs[1].set_title('N=%d,n=%f,R=%f,p_max=%d' % (Resonances_th.N_of_resonances,best_res['x'][0],best_res['x'][1],p_best))

def update(val):
    n = s_n.val
    R = s_R.val
    p=s_p.val
    axs[1].clear()
    resonances=Resonances(Wavelength_min,Wavelength_max,n,R,p,dispersion=True)    
    plt.sca(axs[1])
    resonances.plot_all(0,1,'both')
    axs[1].set_title('N=%d,n=%f,R=%f,p_max=%d' % (resonances.N_of_resonances['Total'],n,R,p))
#    plt.xlim([Wavelength_min,Wavelength_max])
#    plt.show()


s_n.on_changed(update)
s_R.on_changed(update)
s_p.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    s_n.reset()
    s_R.reset()
    s_p.reset()
button.on_clicked(reset)

#rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
#radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
#
#
#def colorfunc(label):
#    l.set_color(label)
#    fig.canvas.draw_idle()
#radio.on_clicked(colorfunc)

plt.show()