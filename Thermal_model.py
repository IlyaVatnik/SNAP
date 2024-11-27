

__date__='21.11.2024'
__version__='0.1'

import numpy as np
import importlib.util
from scipy import sparse
import scipy.linalg as la
import scipy.integrate as integrate
from numba import jit
from scipy.interpolate import interp1d
from scipy.special import erf


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import os,sys
import time
import pickle

# pathname = os.path.dirname(sys.argv[0])  
pathname=os.path.dirname(__file__)
SIGMA=0.92*5.6e-8*1e-6 #W/mm**2/K**4 Stephan-Boltzman constant



def make_uneven_grids(x_min,x_max,x_ERV_init,ERV_0_init,mode_resolution_x,thermal_x_step):
    x_ERV=np.arange(min(x_ERV_init)-0.3,max(x_ERV_init)+0.3,mode_resolution_x)
    ERV_0=np.interp(x_ERV, x_ERV_init, ERV_0_init)
    
    x=np.concatenate((np.arange(x_min,x_ERV[0],thermal_x_step),x_ERV[1:-1],np.arange(x_ERV[-1],x_max,thermal_x_step)))
    
    return x, x_ERV, ERV_0

def make_even_grids(x_min,x_max,x_ERV_init,ERV_0_init,ERV_x_step,thermal_x_step):
    x_ERV=np.arange(min(x_ERV_init)-0.3,max(x_ERV_init)+0.3,ERV_x_step)
    ERV_0=np.interp(x_ERV, x_ERV_init, ERV_0_init)
    x=np.arange(x_min,x_max,thermal_x_step)
    return x, x_ERV, ERV_0

class ThermalModel():
    def __init__(self, x:np.array,T_0:np.array,
                 r_out=1, # mm
                 r_in=0.5, # mm
                 medium_name='corundum',
                 T_bound=20): # in Celsius
        '''
        T - in celsius
        
        '''
        
        self.x=x
        self.N_x=np.size(x)
        self.T=np.asarray(T_0,np.float32)
        self.T_bound=T_bound

        
        self.r_out=r_out
        self.r_in=r_in
        
        self.active_cooling=False
        
        temp=importlib.util.spec_from_file_location("medium_name",pathname+"\\media_params\\"+medium_name+'.py')
        medium = importlib.util.module_from_spec(temp)
        temp.loader.exec_module(medium)
        
        self.medium_heat_exchange=medium.heat_exchange
        
        self.beta=medium.thermal_conductivity/medium.specific_heat_capacity/medium.density
        self.gamma=medium.heat_exchange/medium.specific_heat_capacity/medium.density*2*r_out/(r_out**2-r_in**2)
        self.delta=SIGMA/medium.specific_heat_capacity/medium.density*2*r_out/(r_out**2-r_in**2)    
    
        # note that zeta should be muplyplied by Seff, depending on specific mode
        
        self.theta=1/(medium.specific_heat_capacity*medium.density*(r_out**2-r_in**2))   
        
        

        
    
    
    def set_active_cooling(self,active_heat_exchange,length):
        '''
        active_heat_exchange - heat exchange coefficient in W/mm**2/K
        '''
        self.active_cooling=True
        index=np.argmin(abs(self.x-(self.x[0]+length)))
        self.gamma_array=np.ones(len(self.x))*self.gamma
        self.gamma_array[0:index]=self.gamma/self.medium_heat_exchange*active_heat_exchange
        self.gamma_array[-index:]=self.gamma/self.medium_heat_exchange*active_heat_exchange
  
         
    def set_CO2_laser_params(self,times,powers,laser_spot_radius,laser_focused_spot_width,CO2_beam_positions):
        self.times=times 
        self.CO2_powers=powers
        self.CO2_beam_positions=CO2_beam_positions
        self.laser_focused_spot_width=laser_focused_spot_width
        self.CO2_coeff=np.sqrt(2)*erf(np.sqrt(2)*self.r_out/laser_spot_radius)/(np.pi**(3/2)*laser_focused_spot_width)*self.theta
        
    def beam_distribution(self,x_0): # source distribution
        return np.exp(-((self.x-x_0)/self.laser_focused_spot_width)**2)

        
    # @jit(nopython=False)
    def solve_temper_model_with_source(self,dt:float,t_max:float,T:np.array,source:np.array):

        '''
        made by Arkady
        refactored by Ilya
        '''
        # dt=t_max/N_t
        t=0
        N_x=self.N_x
        #aj*U_n+1,j+1 +bj*U_n+1,j +cj*U_n+1,j-1=ksi_n - схема кранка-николcон
        """
        a=(-1*Beta*dt)/(2*dx*dx)
        c=a
        b=1+dt*Beta/(dx*dx)"""
        Temps_previous=T # массив температур
        Temps_next=np.zeros(len(T),np.float32)
 
        #     heat capacity -120 +4.56*Temp-(7.38*1e-3)*np.power(Temp,2)+(6.59*1e-6)*np.power(Temp,3)-(3.05*1e-9)*np.power(Temp,4)+(5.72*1e-13)*np.power(Temp,5)
        
        alpha=np.zeros((N_x-1),np.float32)
        betta=np.zeros((N_x-1),np.float32)
        
        #пользуемся гран условиями 
        #b0X0+c0X1=d0
        b0=1
        c0=0
        d0=self.T_bound
        alpha[0]=-1*c0/b0
        betta[0]=d0/b0
        n=1

        h_a=np.diff(self.x[1:])
        h_b=np.diff(self.x[:-1])
        
        a=(-1*self.beta*dt)/(2*h_a*h_a)
        c=(-1*self.beta*dt)/(2*h_a*h_b)
        b=1+0.5*dt*self.beta*(h_a+h_b)/(h_a*h_a*h_b)
        
        bm=1
        am=0
        dm=self.T_bound
        
        while t<t_max: #идем по времени
            t+=dt
            
            KSI_n_j=(1-0.5*self.beta*dt*(h_a+h_b)/(h_a*h_a*h_b))*Temps_previous[1:-1] +(self.beta*dt/(2*h_a*h_a))*Temps_previous[2:] +(self.beta*dt/(2*h_a*h_b))*Temps_previous[:-2] + dt*source[1:-1] \
                        -dt*(self.delta*(Temps_previous[1:-1]+273)**4-self.delta*(self.T_bound+273)**4)
            if self.active_cooling:
                KSI_n_j-=dt*self.gamma_array[1:-1]*(Temps_previous[1:-1]-self.T_bound )
            else:
                KSI_n_j-=dt*self.gamma*(Temps_previous[1:-1]-self.T_bound )
           
            for j in range(0,N_x-2):
                alpha[j+1]= -a[j]/(b[j]+c[j]*alpha[j])
                betta[j+1]=(KSI_n_j[j]-c[j]*betta[j] )/(b[j]+c[j]*alpha[j])
            
  
            # теперь у нас есть массив альф и бетт, у нас есть рекуентной соотношение на наши тепемпературы
            
            U1=(dm-am*betta[N_x-2])/(alpha[N_x-2]*am+bm)
            Temps_next[N_x-1]=U1
            
            for j in range(2,N_x+1): #сейчас записываем температуры справа налево
                Temps_next[N_x-j]=alpha[N_x-j]*Temps_next[N_x-j+1]+betta[N_x-j]
            Temps_previous=Temps_next
        # print(times1,times2)
        return Temps_next
        
    
 
    def solve_full_system(self,dt=1e-3,log=True): # in s
        N_log=2 #logging each N_log steps
        time1=time.time()
        T_array=np.zeros((np.size(self.times),np.size(self.x)),np.float32)
        T_array[0]=self.T
        N_steps=len(self.times)
          
        time_tic_1=0
        
        # ind1=np.argmin(abs(self.x_ERV[0]-self.x))-1
        # ind2=np.argmin(abs(self.x_ERV[-1]-self.x))+1
        source=np.zeros(len(self.x))
        for ii,t in enumerate(self.times[:-1]):
            
            if ii%N_log==0 and log:
                time_tic_2=time.time()
                time_remaining=(N_steps-ii)/N_log*(time_tic_2-time_tic_1)
                print('Modeling time={:.3f} s, step {} of {}, time remaining={:.0f} min {:.1f} s'.format(t,ii,N_steps,time_remaining//60,np.mod(time_remaining,60)))
                time_tic_1=time_tic_2
        
            
       
            source*=0
            source=self.CO2_powers[ii]*self.beam_distribution(self.CO2_beam_positions[ii])*self.CO2_coeff
            
            # source[ind1:ind2]=self.zeta*self.Seff*np.power(np.abs(Amplitude),2)*np.power(np.abs(psi_distribs[mode_to_pump]),2)
            
            
            # print(max(source))
            self.T=self.solve_temper_model_with_source(dt,self.times[ii+1]-self.times[ii],
                                                       np.asarray(self.T,dtype=np.float32),np.asarray(source,dtype=np.float32))
            T_array[ii+1]=self.T
            time_tic_2=time.time()
            
   
        self.T_dynamics=T_array

        time2=time.time()
        time_elapsed=time2-time1
        if log:
            print('Total time of calculation is {} min {:.0f} s'.format(time_elapsed//60, np.mod(time_elapsed,60)))
        return T_array
   
    def plot_T_dynamics(self,step=1):
        fig, ax = plt.subplots()
        plt.xlabel('Coordinate, mm')
        plt.ylabel('Temperature, C')
        T_max=np.max(self.T_dynamics)
        def update(frame):
            ax.clear()
            ax.plot(self.x, self.T_dynamics[frame])
            ax.set_ylim(self.T_bound, T_max)
            ax.set_title(f'Time: {self.times[frame]:.2f} s')
            
            
        ani = FuncAnimation(fig, update, frames=len(self.T_dynamics[::step]), repeat=False)
        
        # Save as GIF
        ani.save("Results.gif", writer=PillowWriter(fps=10))
        plt.tight_layout()
        plt.show()
        
        
    def plot_T_max_dynamics(self):
        plt.figure()
        T_max=np.max(self.T_dynamics,axis=1)
        plt.plot(self.times, T_max)
        
        plt.xlabel('Time,s')
        plt.ylabel('Temperature at maximum, C')
        
    def plot_T(self,time):
        time_ind=np.argmin(abs(self.times-time))
        plt.figure()
        plt.plot(self.x, self.T_dynamics[time_ind])
        plt.xlabel('Position, mm')
        plt.ylabel('Temperature, C')

    def save_model(self,f_name):
        with open(f_name,'wb') as f:
            pickle.dump(self,f)
            
    def load_results(self,f_name):
        with open(f_name,'rb') as f:
            [self.x,self.times, self.ERV_0,self.pump_powers,self.pump_wavelengths,self.T_dynamics,self.resonances_dynamics]=pickle.load(f)
            
            
def load_model(f_name):
    with open(f_name,'rb') as f:
        model=pickle.load(f)
    return model
        
            
    
   


    
if __name__=='__main__':
    print()
    # a=SNAP_ThermalModel(0,0,0)
    
    
