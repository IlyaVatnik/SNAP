# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:33:09 2024

@author: Илья
"""# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 23:54:24 2024

@author: t-vatniki
"""

from SNAP import SNAP_ThermalModel
import numpy as np
import matplotlib.pyplot as plt
import time




refractive_index=1.44
lambda_0=1550


Pin=0.05 # W


r_0=62.5e-3 # mm

absorption_in_water=1400 *1e-3 # 1/mm #https://refractiveindex.info/?shelf=main&book=H2O&page=Kedenburg
refractive_index_water=1.3
water_layer_width=0.3*1e-6 # mm
absorption_in_water_layer=absorption_in_water/((refractive_index**2-1)*r_0/2/refractive_index_water**2/water_layer_width) # 1/mm  [Городецкий М.Л. Оптические микрорезонаторы с гигантской добротностью. 2012. стр. 290]
Q_absorption=2*np.pi/(lambda_0*1e-6*absorption_in_water_layer)*refractive_index
intristic_losses=2*np.pi*(3e8/(lambda_0*1e-9))/(2*Q_absorption)

active_heat_exchange=10e-6
active_cooling_length=3

thermal_x_step=0.03
mode_resolution_x=2.5e-3
l=1.5
x_ERV_init=np.arange(-l-0.5,l+0.5,mode_resolution_x)
ERV_0_init=np.array([10  if abs(x)<l else 0 for x in x_ERV_init] )
# ERV_0_init[(x_ERV_init<-l) | (x_ERV_init>l)]=0


taper_C2=25000*1e6 # mm/s
taper_ImD=25000*1e6 # mm/s
taper_position=0.04 # mm

taper_losses=0.05
'''
make noneven x grid
by adding few points to the 
'''
x,x_ERV,ERV_0=SNAP_ThermalModel.make_uneven_grids(-8, 8, x_ERV_init,ERV_0_init,mode_resolution_x,thermal_x_step)
#%%


T_0=np.ones(len(x))*20
model=SNAP_ThermalModel.SNAP_ThermalModel(x,T_0,T_bound=T_0[0],r_out=r_0, absorption=absorption_in_water_layer) 
model.set_SNAP_parameters(x_ERV, ERV_0, intristic_losses=intristic_losses, lambda_0=lambda_0)
model.set_taper_parameters( taper_C2=taper_C2, taper_ImD=taper_ImD,taper_position=taper_position, taper_losses=taper_losses)
freqs=3e8/(model.resonances)

# model.plot_modes_distribs()

# _,psi_distribs=model.solve_Shrodinger(x_ERV, ERV_0)
# delta_0,delta_c=model.calculate_deltas(x_ERV, psi_distribs[mode_number])
# V_eff=model.calculate_V_eff(x_ERV,psi_distribs[mode_number],model.Seff)
# print('{:.2e},{:.2e},{:.2e}'.format(delta_0,delta_c,V_eff))
# print(model.resonance_wavelengths)



    #%%
T=1 # whole time of experiment
tuning_range=0.64e-3 # nm, 80 MHz
tuning_time=1/888 # s
# tuning_range=4e-3
# tuning_time=1e-4
adjusting_range=100e-3 # nm

t_step=1e-5 

tuning_speed=tuning_range/tuning_time # nm/s

t_first_step=t_step
t_last_step=t_step

pump_wavelengths_half_cycle=np.arange(model.resonances[0]-tuning_range/2,model.resonances[0]+tuning_range/2,tuning_speed*t_step)
pump_wavelengths_half_cycle_reverse=pump_wavelengths_half_cycle[::-1]
N_in_half_cycle=len(pump_wavelengths_half_cycle)
N_cycles=int(T/(2*tuning_time))
pump_wavelengths=np.zeros(N_cycles*2*N_in_half_cycle)
for i in range(N_cycles):
    pump_wavelengths[2*i*N_in_half_cycle:(2*i+1)*N_in_half_cycle]=pump_wavelengths_half_cycle
    pump_wavelengths[(2*i+1)*N_in_half_cycle:(2*i+2)*N_in_half_cycle]=pump_wavelengths_half_cycle_reverse
pump_wavelengths+=np.linspace(0, adjusting_range,len(pump_wavelengths))
times=np.arange(0,len(pump_wavelengths))*t_step+t_first_step
# plt.figure()
# plt.plot(times,pump_wavelengths)

pump_powers=np.ones(len(times))*Pin

model.set_pump_parameters(times,pump_powers,wavelengths=pump_wavelengths)
# model.set_active_cooling(active_heat_exchange=active_heat_exchange,active_cooling_length=active_cooling_length)
# delta_0=(235)*1e6 # +- 1.2 Hz*pi, spectral width of the resonance due to inner losses
# delta_c=(60)*1e6 # +- 16.5 Hz*pi, spectral width of the resonance due to coupling
# model.set_deltas(delta_0, delta_c)
model.plot_pump_dynamics()
model.plot_modes_distribs()

#%%
model.solve_full_system(dt=t_step)
model.save_model('scanning rectangular potential.model'.format(Pin,tuning_time,active_heat_exchange,active_cooling_length))

