import numpy as np
import matplotlib.pyplot as plt
from SNAP import QuantumNumbersStructure
thermo_optical_coefficient=8.6*1e-6 # for SiO2
thermal_expansion=0.5*1e-6
get_resonance=QuantumNumbersStructure.lambda_m_p_cylinder
m=355
p=1
polarization='TE'
R_0=62.5e3
n=1.45
t1=get_resonance(m,p,polarization,n,R_0,dispersion=True,medium='SiO2')
t2=get_resonance(m,p,polarization,n,R_0,dispersion=True,medium='SiO2',temperature=23)
print(t1,t2,t1-t2)


T_min=20
T_max=55
T_array=np.arange(T_min,T_max,0.1)
resonance_full_formula=np.zeros((len(T_array)))
resonance_simple=np.zeros(len(T_array))

def R(T):
    return R_0*(1+thermal_expansion*(T-T_min))

for i,T in enumerate(T_array):
    resonance_full_formula[i]=get_resonance(m,p,polarization,n,R(T),dispersion=True,medium='SiO2',temperature=T)
    resonance_simple[i]=2*np.pi*R(T)*n*(1+thermo_optical_coefficient*(T-T_min))/m

plt.figure(1)
plt.plot(T_array-T_min,resonance_simple-resonance_simple[0],label=r'Simple formula  $ \frac {2 \pi R(T) n(T)}{m} $')
plt.plot(T_array-T_min,resonance_full_formula-resonance_full_formula[0],label='General formula for resonances at cylinder')
plt.title('')
plt.xlabel('Temperature difference, $^0$C')
plt.ylabel('Resonance shift, nm')

plt.legend()
plt.tight_layout()

plt.figure(2)
plt.plot(T_array-T_min,resonance_full_formula-resonance_simple-resonance_full_formula[0]+resonance_simple[0])
plt.xlabel('Temperature, $^0$C')
plt.ylabel('Resonance shift difference, nm')
plt.tight_layout()