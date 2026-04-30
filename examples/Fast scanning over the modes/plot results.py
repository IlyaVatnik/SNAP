# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 12:51:35 2024

@author: Илья
"""

from SNAP import SNAP_ThermalModel

import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from matplotlib.ticker import EngFormatter
formatter1 = EngFormatter(places=1, sep="\N{THIN SPACE}")  # U+2009


# file=r"F:\!Projects\!SNAP system\Heating and thermal effects\2025.06 Numerics fast scanning\data\fast tuning 5.00e-02 W 1.00e-03 tuning time 0.00064 tuning range.model"
# file=r"F:\!Projects\!SNAP system\Heating and thermal effects\2025.06 Numerics fast scanning\data\fast tuning 5.00e-02 W 1.00e-03 active_cooling 0.0002 length 3.model"
# files=[r"F:\!Projects\!SNAP system\Heating and thermal effects\2025.06 Numerics fast scanning\data\fast tuning 5.00e-02 W 1.00e-03 active_cooling 0.01 length 3.model",
# r"F:\!Projects\!SNAP system\Heating and thermal effects\2025.06 Numerics fast scanning\data\fast tuning 5.00e-02 W 1.00e-03 active_cooling 0.01 length 7.model",
# r"F:\!Projects\!SNAP system\Heating and thermal effects\2025.06 Numerics fast scanning\data\fast tuning 5.00e-02 W 1.00e-03 active_cooling 0.01 length 8.model"]
files=[r"F:\!Projects\!SNAP system\Heating and thermal effects\2025.06 Numerics fast scanning\data resonator 62 mkm\fast tuning 5.00e-02 W 1.00e-04 active_cooling 0.01 length 3.model"]
tuning_time=1e-3
for file in files:
    model=SNAP_ThermalModel.load_model(file)
    all_res=model.resonances_dynamics
    res_dynamics=[]
    for i in range(len(all_res)):
        res_dynamics.append(all_res[i][0])
        
    res_dynamics=np.array(res_dynamics)
    
    fig,ax=plt.subplots(3,1,sharex=True,figsize=(10,8))
    ax[0].plot(model.times,res_dynamics,label='resonance')
    ax[0].set_title('P={:.1e} W,rate={:.1e} Hz, radius {} mm, delta_0 {:.1e}, delta_c {:.1e}'.format(model.pump_powers[0],1/tuning_time,
                                                                                           model.r,model.delta_0, model.delta_c))
    
    
    ax[0].set_ylabel('Wavelength, nm')
    ax[0].plot(model.times,model.pump_wavelengths,color='r',label='pump')
    ax[0].xaxis.set_major_formatter(formatter1)
    ax[0].legend()
    plt.tight_layout()
    
    transmission=model.get_transmission(res_dynamics,model.pump_wavelengths)
    ax[1].plot(model.times,model.pump_wavelengths,color='r')
    ax[1].xaxis.set_major_formatter(formatter1)
    ax[1].set_xlabel('Time, s')
    ax[1].set_ylabel('Pump wavelength, nm')
    ax_1_2=ax[1].twinx()
    ax_1_2.plot(model.times,transmission)
    plt.ylabel('Transmission')
    plt.tight_layout()
    
    amplitude=model.get_amplitude(res_dynamics,model.pump_wavelengths,normalized=True)
    ax[2].plot(model.times,amplitude,color='g')
    ax[2].set_xlabel('Time, s')
    ax[2].set_ylabel('Normalized amplitude')
    plt.tight_layout()
    
    
    
    print('P={:.1e} W,rate={:.1e} Hz, radius {} mm, delta_0 {:.1e}, delta_c {:.1e}'.format(model.pump_powers[0],1/tuning_time,
                                                                                           model.r,model.delta_0, model.delta_c))

