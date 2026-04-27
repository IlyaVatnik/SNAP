

from SNAP import SNAP_ThermalModel
import numpy as np
import matplotlib.pyplot as plt
import os

L=200 ## mm

r_out=0.062
r_in=0.0

distance_propagated=400

t_max=200
t_step=0.5
x_beam_0=0 # position of the laser beam at the zero time
velocity_beam=0.1 # mm/s

thermal_x_step=0.1
mode_resolution_x=thermal_x_step

x_ERV_init=np.arange(-0.5,0.5,mode_resolution_x)
ERV_0_init=x_ERV_init*0


'''
make noneven x grid 
by adding few points to the 
'''
x,x_ERV,ERV_0=SNAP_ThermalModel.make_uneven_grids(-L, L, x_ERV_init,ERV_0_init,mode_resolution_x,thermal_x_step)

if not os.path.exists('data'):
    os.makedirs('data')
#%%


T_0=np.ones(len(x))*20
model=SNAP_ThermalModel.SNAP_ThermalModel(x,T_0,T_bound=T_0[0],r_out=r_out, r_in=r_in, absorption=0,medium_name='fused_silica')

laser_spot_radius_y,laser_spot_radius_z,z_R_y,z_R_z=SNAP_ThermalModel.CO2_laser_spot_after_cylindrical_lens(distance_propagated=distance_propagated)

times=np.arange(0,t_max,t_step)
positions=x_beam_0+velocity_beam*times



for pump_in_percents in [15]:
    print(pump_in_percents)
    Power=36*0.91*pump_in_percents*0.01
    powers=Power*np.ones(len(times))
    model.set_external_CO2_laser_parameters(times, powers, positions, laser_spot_radius_y,0, laser_spot_radius_z)



    model.set_SNAP_parameters(x_ERV, ERV_0, intristic_losses=0, lambda_0=1550)
    model.set_taper_parameters( taper_C2=0, taper_ImD=0,taper_position=0, taper_losses=0)
    
    model.solve_full_system(dt=0.2e-3)
    model.save_model('data\\Power={} %.model'.format(pump_in_percents))

model.plot_T_dynamics()

#%%

# file_list=os.listdir('data')
# file_list.sort(key=lambda x: float(x.split(' %')[0].split('=')[1]))
# for file in file_list:
#     model=SNAP_ThermalModel.load_model('data\\'+file)
#     power_in_percents=float(file.split(' %')[0].split('=')[1])
    
#     Power=power_in_percents*36*0.91*0.01
#     plt.plot(power_in_percents,np.max(model.T),'o',label=file)
    
#     plt.figure()
#     plt.plot(model.x,model.T)
    

# plt.xlabel('Мощность на контроллере лазера, %')
# plt.ylabel('Температура капилляра, градусы')
# plt.legend()
