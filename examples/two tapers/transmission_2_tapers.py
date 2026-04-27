# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:51:13 2024

@author: Илья

following 1. Y. Yang and M. Sumetsky, "In situ observation of slow and tunnelling light at the cutoff wavelength of an optical fiber," Opt. Lett. 45, 762 (2020).,
formula 2,3
"""
__version__='2'
__date__='2025.04.11'

import numpy as np
import matplotlib.pyplot as plt



K=2**(3/2)*3.141*1.5/(1.55**(3/2)) ## mkm ** (-3/2) 

Q=1e8

D=1j*0.008 # mkm-1
# C=0.11 # 
C=(2*abs(D))**(1/2)
S_12=0


L_0=1000 # mkm
spectrum_range=1000 # pm


lam_c =  1.545678365559927
gamma=lam_c/Q # mkm

# d_l=1e-6 # mkm
# print(4*K**2*d_l)
# print(8*K*D**2*np.sqrt(d_l))
# print(D**4*L_0**2)

# print(K**2*L_0**2*d_l)

def G(x,L,D):
    # try:
    return 2*1j*beta(x)*np.exp(1j*beta(x)*np.abs(L))/((2*1j*beta(x)+D)**2-D**2*np.exp(2*1j*beta(x)*L))
    # except:
        # return 0
    
def beta(x):
    '''
    x is detuning \lambda_c - \lambda in mkm
    
    beta in 1/mkm
    '''
    return K*(x+gamma*1j)**(1/2)

def transmission(x,L=L_0,D=D,C=C):
    '''
    x in mkm
    '''
    return abs(S_12- 1j*C**2*G(x,L,D))**2

def get_losses_over_spectrum(detunings,spectrum,L,D,C):
    t=transmission(detunings, L,D,C)
    losses=np.sum(spectrum*t)/np.sum(spectrum)
    return losses

detunings=np.linspace(1,2500,30000)*1e-6 # mkm
plt.figure()
# plt.plot(-detunings*1e3, 10*np.log10(transmission(detunings,D=D,C=C)))
plt.plot(-detunings*1e3, transmission(detunings,D=D,C=C))
plt.xlabel('Detuning, nm')
plt.ylabel('Transmission, dB')
plt.tight_layout()
print(10*np.log10(transmission(1e-12)))


#%%
D_array=np.linspace(1e-7,5e-1,500)*1j
transmission_array=np.zeros(len(D_array)) 
losses_array=np.zeros(len(D_array)) 


detunings=np.linspace(1,spectrum_range,100)*1e-6 
spectrum=np.ones(len(detunings))
for ii,D_0 in enumerate(D_array):
    C_0=(2*abs(D_0))**(1/2)
    transmission_array[ii]=10*np.log10(transmission(1e-12,D=D_0,C=C_0))
    losses_array[ii]=10*np.log10(get_losses_over_spectrum(detunings,spectrum,L_0,D=D_0,C=C_0))

plt.figure()
plt.plot(np.imag(D_array),transmission_array,color='b')
plt.ylabel(r'$|S_{12}(\delta \lambda \approx 0)|^2$, dB')
plt.xlabel(r'Im(D), mkm$^{-1}$')

plt.figure()
plt.plot(np.imag(D_array),losses_array,color='b')
plt.xlabel(r'$C^2, \mu m^{-1}$')
plt.ylabel(r'Losses $ <|S_{12}|^2 >_{\lambda}$ averaged over spectrum, dB')
plt.ylim((-35,0))

print('min losses are {:.4f} dB at ImD={:.4f}'.format(np.max(losses_array),D_array[np.argmax(losses_array)]))


        
    
#%%
# lengths=np.arange(-600,600,4)
# transmission_map=np.zeros((len(lengths),len(detunings)))
# for ii,length in enumerate(lengths):
#     transmission_map[ii]=transmission(detunings,L=length)
# transmission_map=transmission_map.T
#     #%%
# X,Y=np.meshgrid(lengths,detunings)
# plt.figure()
# im=plt.gca().pcolorfast(X,Y,10*np.log10(transmission_map[:-1,:-1]),vmax=10,vmin=-100,colormap='jet')
# plt.colorbar(im)

#%%
# plt.figure()
# plt.plot(detunings, 10*np.log10(transmission_map[10]))
    
