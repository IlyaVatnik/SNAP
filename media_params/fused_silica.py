# silica
import numpy as np
specific_heat_capacity=740 # J/kg/K
density=2.2*1e-3*1e-3 # kg/mm**3
thermal_conductivity=1.38*1e-3 # W/mm/K
refractive_index=1.444
epsilon=refractive_index**2 #refractive index
        #absorption_in_silica=1000*3.27e-09 #absorption in silica, 1/mm
absorption_in_silica=10*1.035e-6 #absorption in silica, 1/mm

heat_exchange=10*1e-6 #W/mm**2/K
        # thermal_expansion_coefficient=0.0107*r*1e3 #nm/K for effective radius variation
absorption=3.27e-09 #1/mm

thermal_optical_coefficient=10*1e-6 #1/K for effective radius variation


'''
Viscosity ,v1
Ojovan, M. I., “Viscosity and Glass Transition in Amorphous Oxides,” Adv. Condens. Matter Phys. 2008, 1–23 (2008)., eq (3),(4)
'''
Q_H=500e3 #J/mol, this is the activation energy at low temperatures#[1] 
Young_modulus=31e9 #Pa, Young module
t_relaxation_at_annealing_temperature=100 #s
T_annealing=1140 # Celcium degree, temperature to make silica glass soft
maximum_ERV_per_square_mm=18/3.1415/(0.0625)**2 # maximum varation 18 nm obtained for 62.5 mkm fiber
R=8.31 #J/mol/K

def viscosity_v1(T):
    '''
    T in C
    '''
    return np.exp(Q_H/R*(1/(T+273)-1/(T_annealing+273)))*t_relaxation_at_annealing_temperature*Young_modulus


'''
 R. H. Doremus, "Viscosity of silica," J. Appl. Phys. 92, 7619–7629 (2002)., eq 2, for 1000-1400 celcium
'''
def viscosity(T): # R. H. Doremus, "Viscosity of silica," J. Appl. Phys. 92, 7619–7629 (2002)., eq 2, for 1000-1400 celcium
    return 3.8*1e-13*np.exp(712000/R/(T))

# 3.8共10兲⫺13 exp共712 000/RT兲, #
    
    