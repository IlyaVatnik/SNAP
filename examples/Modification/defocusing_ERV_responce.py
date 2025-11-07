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




thermal_x_step=0.003
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
laser_spot_radius=laser_initial_spot_radius*np.sqrt(1+(distance_propagated/z_R_0)**2)




alpha=lens_focus/np.sqrt((distance_propagated-lens_focus)**2+z_R_0**2)
z_R=alpha**2*z_R_0
laser_focused_spot_width_0=alpha*laser_initial_spot_radius # mm, radius of the waist at the focus

def beam_radius(defocusing): # defocusing in mm
    return laser_focused_spot_width_0*np.sqrt(1+(defocusing/z_R)**2)



### 
defocusings=np.linspace(-1.5,0,8)


    
t_step=0.002 # s
times=np.arange(0,(N_shots*(t_on+t_off)),t_step)
period=t_on+t_off
laser_powers = np.where((times % period) < t_on, P_laser, 0)
CO2_beam_positions=np.zeros(len(times))


ERV_vs_defocusing=np.zeros((len(defocusings),N_shots))
for ii,defocusing in enumerate(defocusings):
    print(ii,defocusing)
    model=SNAP_ThermalModel.SNAP_ThermalModel(x,T_0,T_bound=T_0[0],r_out=r_out, r_in=r_in, absorption=0,medium_name='fused_silica')
    model.set_external_CO2_laser_parameters(times, laser_powers, CO2_beam_positions, laser_spot_radius, beam_radius(defocusing))
    model.set_SNAP_parameters(x_ERV, ERV_0, intristic_losses=0, lambda_0=1550)
    model.set_taper_parameters( taper_C2=0, taper_ImD=0,taper_position=0, taper_losses=0)
    
    model.solve_full_system(dt=1e-3,log=False)
    ERV,ERV_dynamics=model.estimate_ERV_through_relaxation()
    ERV_vs_defocusing[ii]=ERV_dynamics[::int(len(times)/N_shots)+1,len(model.x)//2]
    del model
    
    
#%%      

#%%
plt.figure()
plt.plot(defocusings,beam_radius(defocusings))
plt.tight_layout()

plt.figure()
for ii,defocusing in enumerate(defocusings):
    plt.plot(np.arange(N_shots),ERV_vs_defocusing[ii], label='{:.3f}'.format(defocusing))
    plt.xlabel('Shot number')
    plt.ylabel('ERV introduced, nm')
    plt.legend()
    


