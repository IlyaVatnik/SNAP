from SNAP import SNAP_model
import numpy as np
import matplotlib.pyplot as plt

N=3000
lambda_0=1500
R_0=62.5# mkm

h_width=3000 #mkm

MaxRadVar=10 # nm
z_dr=np.linspace(-h_width*0.7, h_width*0.7,num=N)
dr=np.zeros(len(z_dr))
dr[np.abs(z_dr) <= h_width/2] = MaxRadVar
length_of_steepness=200 
mask1 = (z_dr > h_width/2) & (z_dr <= h_width/2 + length_of_steepness)
dr[mask1] = np.linspace(MaxRadVar, 0, np.sum(mask1))
mask2= (z_dr <- h_width/2) & (z_dr>= -h_width/2 - length_of_steepness)
dr[mask2] = np.linspace(0, MaxRadVar, np.sum(mask2))
cone=0.0001
dr+=cone*(z_dr-np.min(z_dr))*1e-3
dr+=np.random.random(len(z_dr))*MaxRadVar*0.05


wave_min,wave_max,res=lambda_0-0.01,lambda_0*(1+MaxRadVar*1e-3/R_0*1.1), 1e-3


lambda_array=np.arange(wave_min,wave_max,res)


model=SNAP_model.SNAP(z_dr,dr,lambda_array,lambda_0=lambda_0,res_width=1e-4,R_0=R_0)
model.set_taper_params(absS=np.sqrt(0.8),phaseS=0.0,ReD=0.00,ImD_exc=0e-3,Csquared=0.00001)
# fig=model.plot_spectrogram(plot_ERV=False,scale='log')
model.find_modes()
print(model.mode_number)
model.calculate_mode_lengths()
print(model.mode_lengths[30])
# plt.xlim((-150,150))
model.plot_ERV()
# SNAP.plot_spectrum(0,scale='log')


# plt.xlim((1552.46,1552.5))
# print(SNAP.find_modes())
# print(SNAP.critical_Csquared())
#%%
m_distrib=model.mode_distribs[1]
plt.figure()
plt.plot(z_dr,m_distrib**2)



    

