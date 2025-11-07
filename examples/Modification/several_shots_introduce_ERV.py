# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 16:45:28 2025

@author: Илья
"""

'''
test the focusing on 200 mkm fiber
'''


from SNAP import SNAP_ThermalModel
import numpy as np
import matplotlib.pyplot as plt
import time


r_out=0.0625 # mm
r_in=0 # mm

P_laser=20 * (0.01*36) # W
t_on=0.100 # s
t_off=1 #s 
N_shots=60




thermal_x_step=0.002
mode_resolution_x=thermal_x_step

x_ERV_init=[-5,5]
ERV_0_init=[0,0]
L=50

'''
make noneven x grid
by adding few points to the 
'''
x,x_ERV,ERV_0=SNAP_ThermalModel.make_uneven_grids(-L, L, x_ERV_init,ERV_0_init,mode_resolution_x,thermal_x_step)
T_0=np.ones(len(x))*20

laser_initial_spot_radius=3.5/2 #mm, waist right out of the laser. Beam specification measured at 1/e^2
distance_propagated=400 #mm, by the beam before the lense, 
lens_focus=22 # mm


z_R_0=np.pi*laser_initial_spot_radius**2/10.6e-3 
divergence=10.6e-3/3.1415/laser_initial_spot_radius #rad
laser_spot_radius_y=laser_initial_spot_radius*np.sqrt(1+(distance_propagated/z_R_0)**2)



alpha=lens_focus/np.sqrt((distance_propagated-lens_focus)**2+z_R_0**2)
z_R=alpha**2*z_R_0
laser_spot_radius_z=alpha*laser_initial_spot_radius # mm, radius of the waist at the focus


model=SNAP_ThermalModel.SNAP_ThermalModel(x,T_0,T_bound=T_0[0],r_out=r_out, r_in=r_in, absorption=0,medium_name='fused_silica')
### 
    
    
t_step=0.002 # s
times=np.arange(0,(N_shots*(t_on+t_off)),t_step)
period=t_on+t_off
laser_powers = np.where((times % period) < t_on, P_laser, 0)
CO2_beam_positions=np.zeros(len(times))
laser_spot_radius_y_shift=0

model.set_external_CO2_laser_parameters(times, laser_powers, CO2_beam_positions, laser_spot_radius_y,laser_spot_radius_y_shift, laser_spot_radius_z)
model.set_SNAP_parameters(x_ERV, ERV_0, intristic_losses=0, lambda_0=1550)
model.set_taper_parameters( taper_C2=0, taper_ImD=0,taper_position=0, taper_losses=0)

model.solve_full_system(dt=0.5e-3)


    
#%%      
ERV,ERV_dynamics=model.estimate_ERV_through_relaxation()
# ERV,ERV_dynamics=model.estimate_ERV_through_relaxation_old()
#%%
fig, (ax1, ax2,ax3) = plt.subplots(nrows=3, sharex=True,figsize=(10,7))
ax1.plot(model.times,laser_powers)
ax1.set_ylabel('laser power, W')
ax2.plot(model.times,np.max(model.T_dynamics,1),color='red')
ax2.set_ylabel('Temperature at max, C')
ax3.plot(model.times,ERV_dynamics[:,len(model.x)//2],color='green')
ax3.set_ylabel('ERV introduced, nm')
ax3.set_xlabel('Time, s')

plt.figure()
plt.plot(np.arange(N_shots),ERV_dynamics[::int(len(times)/N_shots)+1,len(model.x)//2])
plt.xlabel('Shot number')
plt.ylabel('ERV introduced, nm')

