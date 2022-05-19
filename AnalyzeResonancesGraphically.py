# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:50:41 2020

@author: Ilya
"""

__date__='2022.05.19'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.fftpack import rfft, irfft, fftfreq
from scipy.signal import find_peaks
from SNAP.QuantumNumbersStructure import Resonances
import pickle
#from ComputingAzimuthalAndRadialModes import Resonances


MinimumPeakDepth=0.8  ## For peak searching 
MinimumPeakDistance=100 ## For peak searching 
threshold=0.001

dispersion=True
simplified=False


Wavelength_min=None
Wavelength_max=None

FileName1="G:\!Projects\!SNAP system\Modifications\Bending\\2022.02.25 loop spectra\Processed_spectrogram_at_spot_at_2.0.pkl"
FileName2=None


def measure(exp,theory):
    closest_indexes=closest_argmin(exp,theory)
    return sum((exp-theory[closest_indexes])**2)

def closest_argmin(A, B): # from https://stackoverflow.com/questions/45349561/find-nearest-indices-for-one-array-against-all-values-in-another-array-python
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
    return sidx_B[sorted_idx-mask]     

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
if Wavelength_min is None:
    Wavelength_min=min(Wavelengths_TE)
if Wavelength_max is None:
    Wavelength_max=max(Wavelengths_TE)
index_TE_min=np.argmin(abs(Wavelengths_TE-Wavelength_min))
index_TE_max=np.argmin(abs(Wavelengths_TE-Wavelength_max))
Signals_TE=FFTFilter(Temp[index_TE_min:index_TE_max,1])
Wavelengths_TE=Wavelengths_TE[index_TE_min:index_TE_max]
resonances_indexes_TE,_=find_peaks(abs(Signals_TE-np.nanmean(Signals_TE)),height=MinimumPeakDepth,distance=MinimumPeakDistance)
exp_resonances_TE=Wavelengths_TE[resonances_indexes_TE]
if FileName2 is not None:
    with open(FileName2,'rb') as f:
        Temp=pickle.load(f)
    Wavelengths_TM=Temp[:,0]
    index_TM_min=np.argmin(abs(Wavelengths_TM-Wavelength_min))
    index_TM_max=np.argmin(abs(Wavelengths_TM-Wavelength_max))
    Signals_TM=FFTFilter(Temp[index_TM_min:index_TM_max,1])
    Wavelengths_TM=Wavelengths_TM[index_TM_min:index_TM_max]
    resonances_indexes_TM,_=find_peaks(abs(Signals_TM-np.nanmean(Signals_TM)),height=MinimumPeakDepth,distance=MinimumPeakDistance)
    exp_resonances_TM=Wavelengths_TM[resonances_indexes_TM]
    exp_resonances=np.hstack(exp_resonances_TE,exp_resonances_TM)
else:
    Signals_TM=None
    Wavelengths_TM=None
    exp_resonances=exp_resonances_TE



fig, axs = plt.subplots(1, 1)#, figsize=(15, 12))
plt.subplots_adjust(left=0.1, bottom=0.3)
axs.plot(Wavelengths_TE,Signals_TE)
axs.plot(Wavelengths_TE[resonances_indexes_TE],Signals_TE[resonances_indexes_TE],'.')
if Wavelengths_TM is not None:
    axs.plot(Wavelengths_TM,Signals_TM)
    axs.plot(Wavelengths_TM[resonances_indexes_TM],Signals_TM[resonances_indexes_TM],'.')
# plt.sca(axs[1])


R0 = 62.5e3
n0 = 1.44443
p0=3
delta_n = 1e-5
delta_R = 1
resonances=Resonances(Wavelength_min,Wavelength_max,n0,R0,p0,dispersion=dispersion, simplified=simplified)
th_resonances,labels=resonances.create_unstructured_list('both')
cost_function=measure(exp_resonances,th_resonances)
tempdict=resonances.__dict__
resonances.plot_all(0,0.9,'both')
plt.xlim([Wavelength_min,Wavelength_max])
axs.set_title('N=%d,n=%f,R=%f,p_max=%d,cost=%f' % (resonances.N_of_resonances['Total'],n0,R0,p0,cost_function))
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




resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
plt.sca(axs)



def update(val):
    n = s_n.val
    R = s_R.val
    p=s_p.val
    axs.clear()
    axs.plot(Wavelengths_TE,Signals_TE)
    axs.plot(Wavelengths_TE[resonances_indexes_TE],Signals_TE[resonances_indexes_TE],'.')
    if Wavelengths_TM is not None:
        axs.plot(Wavelengths_TM,Signals_TM)
        axs.plot(Wavelengths_TM[resonances_indexes_TM],Signals_TM[resonances_indexes_TM],'.')
    resonances=Resonances(Wavelength_min,Wavelength_max,n,R,p,dispersion=dispersion, simplified=simplified)
    th_resonances,labels=resonances.create_unstructured_list('both')
    cost_function=measure(exp_resonances,th_resonances)
    # plt.sca(axs[1])
    resonances.plot_all(0,0.9,'both')
    axs.set_title('N=%d,n=%f,R=%f,p_max=%d, cost=%f' % (resonances.N_of_resonances['Total'],n,R,p,cost_function))
#    plt.xlim([Wavelength_min,Wavelength_max])
#    plt.show()


s_n.on_changed(update)
s_R.on_changed(update)
s_p.on_changed(update)

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