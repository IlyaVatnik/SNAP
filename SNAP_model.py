'''
Created on Fri Sep 25 

@author: Ilya Vatnik

v.2

After papers 
1. Sumetsky, M. Theory of SNAP devices: basic equations and comparison with the experiment. Opt. Express 20, 22537 (2012).
2. Vitullo, D. L. P., Zaki, S., Jones, D. E., Sumetsky, M. & Brodsky, M. Coupling between waveguides and microresonators: the local approach. Opt. Express 28, 25908 (2020).
'''

import numpy as np
import bottleneck as bn
from scipy import sparse
import scipy.linalg as la
import matplotlib.pyplot as plt
import pickle
import scipy.signal


# class Taper():
    

class SNAP():
    
# =============================================================================
#     initialization functions
# =============================================================================
    @classmethod
    def loader(SNAP,f_name='SNAP object'):
        with open(f_name,'rb') as f:
            return pickle.load(f)
        

    
    def __init__(self,x=None,ERV=None,Wavelengths=None,lambda_0=1550,taper_absS=0.9,taper_phaseS=0,taper_ReD=0.0002,taper_ImD_exc=1e-4,taper_Csquared=1e-2,
                 res_width=1e-4,R_0=62.5,n=1.45): # Note that x is in microns!
        
        self.res_width=res_width    # nm, resonance linewidth corresponding to inner losses of resonator
        self.R_0=R_0                  ## mkm, Fiber radius
        self.refractive_index=n                 ## Cladding refractive index
        
        self.lambda_0=lambda_0  # nm, resonance wavelength for the undisturbed cylinder
        self.k0=2*np.pi*self.refractive_index/(self.lambda_0*1e-3) # in 1/mkm
        self.res_width_norm=8*np.pi**2*self.refractive_index**2/(self.lambda_0*1e-3)**3*self.res_width*1e-3
        
        if x is not None:
            self.ERV=ERV # nm
            self.x=x  # mkm
            self.N=len(x)
            self.U=-2*self.k0**2*self.ERV*(1e-3)/self.R_0/self.refractive_index
        else:
            self.ERV=None # nm
            self.x=None # mkm
            self.N=None
            self.U=None

        
        self.transmission=None
        self.lambdas=Wavelengths
        self.need_to_update_transmission=True
        
        self.taper_absS=taper_absS  
        self.taper_phaseS=taper_phaseS  # * pi, this parameter is in parts of one pi
        self.taper_Csquared=taper_Csquared # 1/mkm
        self.taper_ReD=taper_ReD # 1/mkm
        self.taper_ImD_exc=taper_ImD_exc # 1/mkm
        
        
        self.mode_distribs=None
        self.mode_wavelengths=None
        
        self.Cmap='jet'
        
        self.ERV_params=None
        self.fig=None
        
        
    def set_fiber_params(self,res_width=None,R_0=None,n=None):
        if res_width is not None:
            self.res_width=res_width    # in nm, resonance width corresponding to inner losses of resonator
        if R_0 is not None:
            self.R_0=R_0                  ## Fiber radius, in um
        if n is not None:
            self.refractive_index=n                 ## Cladding refractive index
        self.need_to_update_transmission=True
            
    def get_fiber_params(self,**a):
        return self.res_width,self.R_0,self.refractive_index
            

    
    def set_taper_params(self,absS=None,phaseS=None,ReD=None,ImD_exc=None,Csquared=None):
        if absS is not None:
            if absS>1:
                print('abs(S_0) cannot be larger then 1! S_0 is kept to {}'.format(self.taper_absS))
            else:
                self.taper_absS=absS
                self.need_to_update_transmission=True
                
        if phaseS is not None:
            self.taper_phaseS=phaseS
            self.need_to_update_transmission=True
        
        if Csquared is not None:
            self.taper_Csquared=Csquared
            self.need_to_update_transmission=True
        if ReD is not None:
            self.taper_ReD=ReD
            self.need_to_update_transmission=True
        if ImD_exc is not None:
            self.taper_ImD_exc=ImD_exc
            self.need_to_update_transmission=True    
        
          
    def get_taper_params(self):
        return self.taper_absS,self.taper_phaseS,self.taper_ReD,self.taper_ImD_exc,self.taper_Csquared
    
# =============================================================================
# core deriving functions    
# =============================================================================
    def min_imag_D(self):
        taper_ReS=self.taper_absS*np.cos(self.taper_phaseS*np.pi)
        return self.taper_Csquared*(1-taper_ReS)/(1-self.taper_absS**2)
    
    def critical_Csquared(self):
        taper_ReS=self.taper_absS*np.cos(self.taper_phaseS*np.pi)
        taper_ImD=self.taper_ImD_exc+self.min_imag_D()
        return self.taper_absS**2*taper_ImD/taper_ReS
    
    def critical_Csquared_1(self,x_0,number_of_level=0):
        U=-2*self.k0**2*self.ERV*(1e-3)/self.R_0
        _,eigvecs=self.solve_Shrodinger(U)
        i=np.argmin(abs(self.x-x_0))
        num=np.imag(self.S()*self.complex_D_exc())+self.res_width_norm/eigvecs[number_of_level,i]**2
        
        taper_ReS=self.taper_absS*np.cos(self.taper_phaseS*np.pi)
        denum=1-taper_ReS*(1-taper_ReS)/(1-self.taper_absS**2)
        return num/denum
    
    
    def critical_Csquared_2(self):
        taper_ReS=self.taper_absS*np.cos(self.taper_phaseS*np.pi)
        return self.taper_ImD_exc*self.taper_absS**2*(1-self.taper_absS**2)/(taper_ReS-self.taper_absS**2)
        
    
    
    def D(self):
        return self.taper_ReD+1j*(self.taper_ImD_exc+self.min_imag_D())
    
    def complex_D_exc(self):
        return self.taper_ReD+1j*(self.taper_ImD_exc)
    
    def S(self):
        return self.taper_absS*np.exp(1j*self.taper_phaseS*np.pi)
        
        
    def solve_Shrodinger(self):
        dx=self.x[1]-self.x[0]
        Tmtx=-1/dx**2*sparse.diags([-2*np.ones(self.N),np.ones(self.N)[1:],np.ones(self.N)[1:]],[0,-1,1]).toarray()
        Vmtx=np.diag(self.U)
        Hmtx=Tmtx+Vmtx
        (eigvals,eigvecs)=la.eigh(Hmtx,check_finite=False)
        sorted_indexes=np.argsort(np.real(eigvals))
        eigvals,eigvecs=[eigvals[sorted_indexes],eigvecs.T[sorted_indexes]]
        eigvecs=eigvecs/np.sqrt(dx)  # to get normalization for integral (psi**2 dx) =1
        return eigvals,eigvecs
    
    def find_modes(self,plot_at_spectrogram=False):
        eigvals,eigvectors=self.solve_Shrodinger()
        wavelengths=self.lambda_0-eigvals*self.lambda_0/(2*self.k0**2)
        indexes=np.where(wavelengths>self.lambda_0)
        self.mode_wavelengths=wavelengths[indexes]
        self.mode_distribs=eigvectors[indexes]
        if plot_at_spectrogram:
            for mode in self.mode_wavelengths:
                self.fig.axes[0].axhline(mode, color='black')
        return self.mode_wavelengths,self.mode_distribs.T
    
    
    def GreenFunctionXX(self,eigvals,eigvecs,wavelength):
        E=-2*self.k0**2*(wavelength-self.lambda_0)/self.lambda_0
        return bn.nansum(eigvecs**2/(E-eigvals+1j*self.res_width_norm),1) 
    

    
    def GreenFunction(self,eigvals,eigvecs,wavelength,x1,x2):
        ind_1=np.argmin(abs(self.x-x1))
        ind_2=np.argmin(abs(self.x-x2))
        
        E=-2*self.k0**2*(wavelength-self.lambda_0)/self.lambda_0
        return bn.nansum(eigvecs[:,ind_1]*eigvecs[:,ind_2]/(E-eigvals+1j*self.res_width_norm),1) 
        

    

    def derive_transmission(self,show_progress=False):
        taper_D=self.D()
        taper_S=self.S()
        T=np.zeros((len(self.lambdas),len(self.x)))
        eigvals,eigvecs=self.solve_Shrodinger()
        for ii,wavelength in enumerate(self.lambdas):
            if ii%50==0 and show_progress: print('Deriving T for wl={}, {} of {}'.format(wavelength,ii,len(self.lambdas)))
            G=self.GreenFunctionXX(eigvals,eigvecs,wavelength)
            ComplexTransmission=(taper_S-1j*self.taper_Csquared*G/(1+taper_D*G))  ## 
            T[ii,:]=abs(ComplexTransmission)**2 
        self.need_to_update_transmission=False
        self.transmission=T
        if np.amax(T)>1:
            print('Some error in the algorimth! Transmission became larger than 1')
        return self.x, self.lambdas,self.transmission
    
    def GreenFunctionForTwoTapers(self,eigvals,eigvecs,wavelength,x_0):
        ind_1=np.argmin(abs(self.x-x_0))
        E=-2*self.k0**2*(wavelength-self.lambda_0)/self.lambda_0
        return bn.nansum(eigvecs[:,ind_1]*eigvecs/(E-eigvals+1j*self.res_width_norm),1) 
    
    
    def derive_transmission_2_tapers(self,x_0,show_progress=False,**kwargs):
        '''
        derive transmission between two tapers with equal parameters. Following eq (3) from Crespo-Ballesteros M, Yang Y, Toropov N, Sumetsky M. Four-port SNAP microresonator device. Opt Lett 2019;44:3498. https://doi.org/10.1364/OL.44.003498.
        z_0 is the position of the input taper
        '''
        
        taper_D=self.D()
        taper_S=self.S()
        T=np.zeros((len(self.lambdas),len(self.x)))
        U=-2*self.k0**2*self.ERV*(1e-3)/self.R_0/self.refractive_index
        eigvals,eigvecs=self.solve_Shrodinger(U)
        ind_0=np.argmin(abs(self.x-x_0))
        for ii,wavelength in enumerate(self.lambdas):
            if ii%50==0 and show_progress: print('Deriving T for wl={}, {} of {}'.format(wavelength,ii,len(self.lambdas)))
            E=-2*self.k0**2*(wavelength-self.lambda_0)/self.lambda_0
            G=bn.nansum((eigvecs[:,ind_0])*(eigvecs)/(E-eigvals+1j*self.res_width_norm+(eigvecs[:,ind_0]**2+eigvecs**2)*taper_D),1)
            ComplexTransmission=(taper_S-1j*self.taper_Csquared*G)  ## 
            T[ii,:]=abs(ComplexTransmission)**2 
        self.need_to_update_transmission=False
        self.transmission=T
        if np.amax(T)>1:
            print('Some error in the algorimth! Transmission became larger than 1')
        return self.x, self.lambdas,self.transmission
        
    
# =============================================================================
#     These functions are for debugging
# =============================================================================
    def _derive_transmission_test(self,psi_function):
        taper_D=self.D()
        taper_S=self.S()
        T=np.zeros((len(self.lambdas),len(self.x)))
        for ii,wavelength in enumerate(self.lambdas):
            E=-2*self.k0**2*(wavelength-self.lambda_0)/self.lambda_0
            G=psi_function**2/(E+1j*self.res_width_norm) 
            ComplexTransmission=(taper_S-1j*self.taper_Csquared*G/(1+taper_D*G))  ## 
            T[ii,:]=abs(ComplexTransmission)
        self.transmission=T
        if np.amax(T)>1:
            print('Some error in the algorimth! Transmission became larger than 1')
        return self.x, self.lambdas,self.transmission   
    
    def _calculate_GreenFunction_squared_at_point(self,x0):
        index_x=np.argmin(abs(self.x-x0))
        G2=np.zeros(len(self.lambdas))
        U=-2*self.k0**2*self.ERV*(1e-3)/self.R_0
        eigvals,eigvecs=self.solve_Shrodinger(U)
        for ii,wavelength in enumerate(self.lambdas):
            G2[ii]=abs(self.GreenFunctionXX(eigvals,eigvecs,wavelength)[index_x])**2
   
        return G2
##################################  

# =============================================================================
#   processing      
# =============================================================================
    def find_modes_old(self,prominence_factor=3):
        if self.need_to_update_transmission:
            self.derive_transmission()
        T_shrinked=np.nanmean(abs(self.transmission-np.nanmean(self.transmission,axis=0)),axis=1)
        mode_indexes,_=scipy.signal.find_peaks(T_shrinked,prominence=np.std(T_shrinked)*prominence_factor)
        temp=np.sort(self.lambdas[mode_indexes])
        self.mode_wavelengths=np.array([x for x in temp if x>self.lambda_0])
        return self.mode_wavelengths
    

    
    def get_spectrum(self,x):
        if self.need_to_update_transmission:
            self.derive_transmission()
        i=np.argmin(abs(self.x-x))
        return self.lambdas,self.transmission[:,i]


# =============================================================================
#  plotting, printing etc 
# =============================================================================

    
     
    def plot_spectrum(self,x,scale='lin'):
        w,l=self.get_spectrum(x)
        fig=plt.figure(3)
        if scale=='lin':
            plt.plot(w,l)
            plt.ylabel('Transmission')
        elif scale=='log':
            plt.plot(w,10*np.log10(l))
            plt.ylabel('Transmission, dB')
        plt.xlabel('Wavelength,nm')     
        return fig
    
    def plot_spectrogram(self,scale='lin',ERV_axis=True,plot_ERV=False,amplitude=False):
        wave_max=max(self.lambdas)
        
        def _convert_ax_Wavelength_to_Radius(ax_Wavelengths):
            """
            Update second axis according with first axis.
            """
            y1, y2 = ax_Wavelengths.get_ylim()
            nY1=(y1-self.lambda_0)/wave_max*self.R_0*1e3
            nY2=(y2-self.lambda_0)/wave_max*self.R_0*1e3
            ax_Radius.set_ylim(nY1, nY2)
            
        def _forward(x):
            return (x-self.lambda_0)/wave_max*self.R_0*self.refractive_index*1e3

        def _backward(x):
            return self.lambda_0 + wave_max*x/self.R_0/self.refractive_index/1e3
    
    
        if self.need_to_update_transmission:
            self.derive_transmission()
        fig=plt.figure()
        plt.clf()
        ax_Wavelengths = fig.subplots()
        if amplitude:
            temp=np.sqrt(self.transmission)
        else:
            temp=self.transmission
        if scale=='log':
            temp=10*np.log10(temp)
        try:
            im = ax_Wavelengths.pcolorfast(self.x,self.lambdas,temp,cmap=self.Cmap)
        except:
            im = ax_Wavelengths.pcolor(self.x,self.lambdas,temp, cmap=self.Cmap)
        if scale=='lin':
            plt.colorbar(im,ax=ax_Wavelengths,pad=0.12,label='Transmission')
        elif scale=='log':
            plt.colorbar(im,ax=ax_Wavelengths,pad=0.12,label='Transmission,dB')                
        ax_Wavelengths.set_xlabel(r'Position, $\mu$m')
        ax_Wavelengths.set_ylabel('Wavelength, nm')
        if ERV_axis:
            # ax_Radius = ax_Wavelengths.secondary_yaxis('right', functions=(_forward,_backward))
            ax_Radius = ax_Wavelengths.twinx()
            ax_Wavelengths.callbacks.connect("ylim_changed", _convert_ax_Wavelength_to_Radius)
            ax_Radius.set_ylabel('Variation, nm')
        plt.title('simulation')
        if plot_ERV:
            ax_Radius.plot(self.x,self.ERV)
            _convert_ax_Wavelength_to_Radius(ax_Wavelengths)
            # plt.gca().set_xlim((self.x[0],self.x[-1]))
        plt.tight_layout()
        self.fig=fig
        return fig
    
    
    def plot_ERV(self):
        fig=plt.figure(2)  
        plt.clf() 
        plt.plot(self.x,self.ERV)
        plt.title('ERV')
        plt.xlabel(r'Position, $\mu$m')
        plt.ylabel('Variation, nm')
        return fig
        
    def save(self,f_name='SNAP object'):
        with open(f_name,'wb') as f:
            return pickle.dump(self,f)
        
           
    def print_taper_params(self):
        print('absS={},phaseS={}*pi,ReD={},ImD_exc={},Csquared={}'.format(*self.get_taper_params()))
        
        
        
def load_model(file_name):
    with open(file_name,'rb') as f:
        return pickle.load(f)



        
if __name__=='__main__':

    
    N=800
    lambda_0=1552.21
    wave_min,wave_max,res=1552.24,1552.25, 1e-4
    
    x=np.linspace(-10000,10000,N)
    lambda_array=np.arange(wave_min,wave_max,res)
    
    A=4
    sigma=50
    p=1.1406
    def ERV(x):
        # if abs(x)<=200:
#            return (x)**2
        return A*np.exp(-(x**2/2/sigma**2)**p)-0.5*A*np.exp(-(x**2/2/(sigma*1.2)**2)**p)
        # else:
            # return 0
#            return ERV(5)-1/2*(x)**2
    ERV=np.array(list(map(ERV,x)))
    
    SNAP=SNAP(x,ERV,lambda_array,lambda_0=lambda_0,res_width=1e-4,R_0=62.5)
    SNAP.set_taper_params(absS=np.sqrt(0.8),phaseS=0.0,ReD=0.00,ImD_exc=2e-3,Csquared=0.001)
    fig=SNAP.plot_spectrogram(plot_ERV=True,scale='log')
    # plt.xlim((-150,150))
    # SNAP.plot_ERV()
    SNAP.plot_spectrum(0,scale='log')
    # plt.xlim((1552.46,1552.5))
    # print(SNAP.find_modes())
    # print(SNAP.critical_Csquared())
    # modes,m_distribs=SNAP.find_modes(plot_at_spectrogram=True)
    # plt.figure()
    # plt.plot(x,m_distribs**2)
    # SNAP.save()
    
    
        
    
    


