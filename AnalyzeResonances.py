import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, fftfreq
import scipy.optimize as sciopt
from ComputingAzimuthalAndRadialModesByGorodetsky import Resonances
from scipy.signal import find_peaks
import pickle 

MinimumPeakDepth=5  ## For peak searching 
MinimumPeakDistance=30 ## For peak searching 
threshold=0.001


Wavelength_min=1545
Wavelength_max=1557
p_max_guess=[1,2,3,4,5]

# FileName1='1.txt'
FileName='Polarization 1.pkl'
# FileName='Polarization 2.pkl'

     

def closest_argmin(A, B): # from https://stackoverflow.com/questions/45349561/find-nearest-indices-for-one-array-against-all-values-in-another-array-python
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
    return sidx_B[sorted_idx-mask]
    
def func_to_minimize(param,*args): # try one and another polarization
    cost=10    
    n,R=param
    p_max=args[3]
    exp_resonances=args[2]
    wave_min=args[0]
    wave_max=args[1]
    resonances=Resonances(wave_min,wave_max,n,R,p_max)
    th_resonances,labels=resonances.create_unstructured_list('TE')
    if len(th_resonances)>len(exp_resonances):
        closest_indexes=closest_argmin(exp_resonances,th_resonances)
        cost=sum(abs(exp_resonances-th_resonances[closest_indexes]))
    else:
        closest_indexes=closest_argmin(th_resonances,exp_resonances)
        cost=sum(abs(th_resonances-exp_resonances[closest_indexes]))
    th_resonances,labels=resonances.create_unstructured_list('TM')
    if len(th_resonances)>len(exp_resonances):
        closest_indexes=closest_argmin(exp_resonances,th_resonances)
        cost=sum(abs(exp_resonances-th_resonances[closest_indexes]))
    else:
        closest_indexes=closest_argmin(th_resonances,exp_resonances)
        cost=sum(abs(th_resonances-exp_resonances[closest_indexes]))
    return cost
 
    
def func_to_minimize_number_of_resonances(param,*args):
    n,R,p_max=param
    exp_resonances=args[2]
    wave_min=args[0]
    wave_max=args[1]
    resonances=Resonances(wave_min,wave_max,n,R,p_max)
    return abs(len(exp_resonances)-resonances.N_of_resonances)


FilterLowFreqEdge=0.00
FilterHighFreqEdge=0.01
def FFTFilter(y_array):
    W=fftfreq(y_array.size)
    f_array = rfft(y_array)
    Indexes=[i for  i,w  in enumerate(W) if all([abs(w)>FilterLowFreqEdge,abs(w)<FilterHighFreqEdge])]
    f_array[Indexes] = 0
#        f_array[] = 0
    return irfft(f_array)
# Temp=np.loadtxt(FileName1)
with open(FileName,'rb') as f:
    Temp=pickle.load(f)
    
Wavelengths_TE=Temp[:,0]
index_TE_min=np.argmin(abs(Wavelengths_TE-Wavelength_min))
index_TE_max=np.argmin(abs(Wavelengths_TE-Wavelength_max))
Signals_TE=FFTFilter(Temp[index_TE_min:index_TE_max,1])
Wavelengths_TE=Wavelengths_TE[index_TE_min:index_TE_max]
Resonances_indexes,_=find_peaks(-Signals_TE,height=MinimumPeakDepth,distance=MinimumPeakDistance,threshold=threshold)
Resonances_exp=Wavelengths_TE[Resonances_indexes]


fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(Wavelengths_TE,Signals_TE)
axs[0].plot(Resonances_exp,Signals_TE[Resonances_indexes],'.')
axs[0].set_title('N=%d' % len(Resonances_exp))


#print(func_to_minimize((1.45,62.4e3,1),Wavelength_min,Wavelength_max,Resonances_exp))
func_min=3
best_res=None
for p in p_max_guess:
    # res=sciopt.minimize(func_to_minimize,((1.45,62.5e3)),bounds=((1.4,1.55),(61.5e3,63.5e3)),
    #                 args=(Wavelength_min,Wavelength_max,Resonances_exp,p),
    #                 method='Nelder-Mead',options={'maxiter':1000})
    res=sciopt.least_squares(func_to_minimize,((1.45,62.3e3)),bounds=((1.4,1.55),(61.5e3,63.5e3)),
                    args=(Wavelength_min,Wavelength_max,Resonances_exp,p))
    
    print(res)
    if res['fun']<func_min:
        func_min=res['fun']
        best_res=res
        p_best=p



Resonances_th=Resonances(Wavelength_min,Wavelength_max,best_res['x'][0],best_res['x'][1],p_best)
plt.sca(axs[1])
Resonances_th.plot_all(-1,1,'both')
axs[1].set_title('N=%d,n=%f,R=%f,p_max=%d' % (Resonances_th.N_of_resonances,best_res['x'][0],best_res['x'][1],p_best))
plt.xlabel('Wavelength,nm')

