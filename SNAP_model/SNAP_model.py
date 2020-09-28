'''
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



class SNAP():

    
    def __init__(self,x=None,ERV=None,Wavelengths=None,lambda_0=1550,taper_absS=0.9,taper_phaseS=0,taper_ReD=0.0002,taper_ImD_exc=1e-4,taper_Csquared=1e-2,
                 res_width=1e-7,R_0=62.5,n=1.45): # Note that x is in microns!
        if x is not None:
            self.ERV=ERV
            self.x=x
            self.dx=x[1]-x[0]
            self.N=len(x)
        
        self.res_width=res_width    # in nm, resonance width corresponding to inner losses of resonator
        self.R_0=R_0                  ## Fiber radius, in um 
        self.n=n                 ## Cladding refractive index
        
        self.lambda_0=lambda_0
        self.k0=2*np.pi*self.n/(self.lambda_0*1e-3) # in 1/um
        self.Res_Width=(8*np.pi**2*self.n**2/(self.lambda_0*1e-3)**3)*self.res_width
        
        
        self.Transmission=None
        self.lambdas=Wavelengths
        self.needToUpdateTransmission=True
        
        self.taper_absS=taper_absS
        self.taper_phaseS=taper_phaseS
        self.taper_Csquared=taper_Csquared
        self.taper_ReD=taper_ReD
        self.taper_ImD_exc=taper_ImD_exc
        
        
    def set_fiberParams(self,res_width=None,R_0=None,n=None):
        if res_width is not None:
            self.res_width=res_width    # in nm, resonance width corresponding to inner losses of resonator
        if R_0 is not None:
            self.R_0=R_0                  ## Fiber radius, in um
        if n is not None:
            self.n=n                 ## Cladding refractive index
        self.needToUpdateTransmission=True
            
    def get_fiberParams(self,**a):
        return self.res_width,self.R_0,self.n
            
    def min_imag_D(self):
        taper_ReS=self.taper_absS*np.cos(self.taper_phaseS)
        return self.taper_Csquared*(1-taper_ReS)/(1-self.taper_absS**2)
    
    def critical_Csquared(self):
        taper_ReS=self.taper_absS*np.cos(self.taper_phaseS)
        taper_ImD=self.taper_ImD_exc+self.min_imag_D()
        return abs(self.taper_absS)**2*taper_ImD/taper_ReS
    
    def set_taperParams(self,absS=None,phaseS=None,ReD=None,ImD_exc=None,Csquared=None):
        if absS is not None:
            if absS>=1:
                print('abs(S_0) cannot be larger then 1! S_0 is kept to {}'.format(self.taper_absS))
            else:
                self.taper_absS=absS
                self.needToUpdateTransmission=True
                
        if phaseS is not None:
            self.taper_phaseS=phaseS
            self.needToUpdateTransmission=True
        
        if Csquared is not None:
            self.taper_Csquared=Csquared
            self.needToUpdateTransmission=True
        if ReD is not None:
            self.taper_ReD=ReD
            self.needToUpdateTransmission=True
        if ImD_exc is not None:
            self.taper_ImD_exc=ImD_exc
            self.needToUpdateTransmission=True    
        
               

            
    def get_taperParams(self):
        return self.taper_absS,self.taper_phaseS,self.taper_ReD,self.taper_ImD_exc,self.taper_Csquared
        
    def solve_Shrodinger(self,U):
        Tmtx=-1/self.dx**2*sparse.diags([-2*np.ones(self.N)+U,np.ones(self.N)[1:],np.ones(self.N)[1:]],[0,-1,1]).toarray()
        Vmtx=np.diag(U)
        Hmtx=Tmtx+Vmtx
#        Hmtx[0,-1]=-1/self.dx**2
#        Hmtx[-1,0]=-1/self.dx**2
        (eigvals,eigvecs)=la.eigh(Hmtx,check_finite=False)
        sorted_indexes=np.argsort(np.real(eigvals))
        eigvals,eigvecs=[eigvals[sorted_indexes],eigvecs[sorted_indexes]]
        return eigvals,eigvecs
    
#    def find_localized_modes(self):
#        eigvals,eigvecs=self.solve_Shrodinger()
#        x_central=np.sum(self.x*self.U)/np.sum(self.U)
#        width_U=np.sqrt(np.sum((self.x-x_central)**2 * (self.U-np.mean(self.U))**2)/np.sum((self.U-np.mean(self.U))**2))
##        print(width_U)
#        widths=np.array([np.sqrt(np.sum((self.x-x_central)**2 * (wave-np.mean(wave))**2)/np.sum((wave-np.mean(wave))**2)) for wave in abs(eigvecs)])
##        print(widths)
#        indexes=np.argwhere(widths<width_U)
##        print(indexes)
#        eigvals,eigvecs,widths=eigvals[indexes],eigvecs[indexes],widths[indexes]
#        sorted_indexes=np.argsort(np.real(eigvals))
#        eigvals,eigvecs,widths=eigvals[sorted_indexes],eigvecs[sorted_indexes],widths[sorted_indexes]
#        return eigvals,eigvecs,widths
    
    def GreenFunction(self,eigvals,eigvecs,wavelength):
        E=-2*self.k0**2*(wavelength-self.lambda_0-1j*self.res_width*1e3)/self.lambda_0
        return bn.nansum(eigvecs*np.conjugate(eigvecs)/(E-eigvals+1j*self.Res_Width),1)
#        return bn.nansum(eigvecs*np.conjugate(eigvecs)/eigvals,1)
        
    
    def derive_transmission(self):
        taper_D=self.taper_ReD+1j*(self.taper_ImD_exc+self.min_imag_D())
        taper_S=self.taper_absS*np.exp(1j*self.taper_phaseS*np.pi)
        T=np.zeros((len(self.lambdas),len(self.x)))
        for ii,wavelength in enumerate(self.lambdas):
            if ii%50==0: print('{} of {}'.format(ii,len(self.lambdas)))
            U=-2*self.k0**2*self.ERV*(1e-3)/self.R_0
            eigvals,eigvecs=self.solve_Shrodinger(U)
            G=self.GreenFunction(eigvals,eigvecs,wavelength)
            ComplexTransmission=(taper_S-1j*self.taper_Csquared*G/(1+taper_D*G))  ## 
            T[ii,:]=abs(ComplexTransmission)**2
        
        self.needToUpdateTransmission=False
        self.Transmission=T
        
        return self.x, self.lambdas,self.Transmission
    
    def get_spectrum(self,x):
        if self.needToUpdateTransmission:
            self.derive_transmission()
        i=np.argmin(abs(self.x-x))
        return self.lambdas,self.Transmission[:,i]
    
    def plot_spectrum(self,x):
        w,l=self.get_spectrum(x)
        plt.figure(3)
        plt.plot(w,l)
        plt.xlabel('Wavelength,nm')
        plt.ylabel('Transmission')
    
    def plot_spectrogram(self):
        wave_max=max(self.lambdas)
        def _convert_ax_Wavelength_to_Radius(ax_Wavelengths):
            """
            Update second axis according with first axis.
            """
            y1, y2 = ax_Wavelengths.get_ylim()
            nY1=(y1-self.lambda_0)/wave_max*self.R_0*1e3
            nY2=(y2-self.lambda_0)/wave_max*self.R_0*1e3
            ax_Radius.set_ylim(nY1, nY2)
        if self.needToUpdateTransmission:
            self.derive_transmission()
        fig2=plt.figure(1)
        plt.clf()
        ax_Wavelengths = fig2.subplots()
        ax_Radius = ax_Wavelengths.twinx()
        ax_Wavelengths.callbacks.connect("ylim_changed", _convert_ax_Wavelength_to_Radius)
        try:
            im = ax_Wavelengths.pcolorfast(self.x,self.lambdas,self.Transmission)
        except:
            im = ax_Wavelengths.pcolor(self.x,self.lambdas,self.Transmission)
        plt.colorbar(im,ax=ax_Radius,pad=0.12)
        ax_Wavelengths.set_xlabel(r'Position, $\mu$m')
        ax_Wavelengths.set_ylabel('Wavelength, nm')
        ax_Radius.set_ylabel('Variation, nm')
        plt.title('Simulation')
        plt.tight_layout()

    
    def plotERV(self):
        plt.figure(2)  
        plt.clf() 
        plt.plot(self.x,self.ERV)
        plt.title('ERV')
        plt.xlabel(r'Position, $\mu$m')
        plt.ylabel('Variation, nm')
        
    def save(self,path='\\',file_name='Numerical_model_SNAP.pkl'):
        file=open(path+file_name,'wb')
        pickle.dump([self.x,self.ERV,self.lambda_0,self.lambdas,self.Transmission,self.get_taperParams(),self.get_fiberParams()],file)
        file.close()
        
    def load(self,path='\\',file_name='Numerical_model_SNAP.pkl'):
        file=open(path+file_name,'rb')
        self.x,self.ERV,self.lambda_0,self.lambdas,self.Transmission,taper_params,fiber_params=pickle.load(file)
        absS,phaseS,ReD,ImD_exc,C2=taper_params
        self.set_taperParams(absS,phaseS,ReD,ImD_exc,C2)
        r,R,n=fiber_params
        self.set_fiberParams(r,R,n)
        file.close()
        
        
if __name__=='__main__':

    
    N=200
    wave_min,wave_max,res=1549.98,1550.07, 3e-4
    
    x=np.linspace(-600,600,N)
    lambda_array=np.arange(wave_min,wave_max,res)
    
    def ERV(x):
        if abs(x)<=200:
#            return (x)**2
            return 2
        else:
            return 0
#            return ERV(5)-1/2*(x)**2
    ERV=np.array(list(map(ERV,x)))
    
    SNAP=SNAP(x,ERV,lambda_array,lambda_0=1550)
    SNAP.set_taperParams(absS=0.7,phaseS=0.05,ReD=0.0002,ImD_exc=0,Csquared=1e-4)
    SNAP.plot_spectrogram()
    SNAP.plotERV()
    SNAP.plot_spectrum(0)
    print(SNAP.get_taperParams())
#    SNAP.load()
    
        
    
    


