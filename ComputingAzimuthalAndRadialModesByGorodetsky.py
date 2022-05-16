########
# Calculating spectra of WGM for different azimuthal and radial numbers 
#
#Using Demchenko, Y. A. and Gorodetsky, M. L., “Analytical estimates of eigenfrequencies, dispersion, and field distribution in whispering gallery resonators,” J. Opt. Soc. Am. B 30(11), 3056 (2013).
#
#See formula A3 for lambda_m_p
########

__version__='2'
__date__='2022.05.16'

import numpy as np
from scipy import special
import matplotlib.pyplot as plt

def ref_index(w):
    return np.sqrt(0.6961663*w**2/(w**2-0.0684043**2) +0.4079426*w**2/(w**2-0.1162414**2) +0.8974794*w**2/(w**2-9.896161**2)+1)

def airy_zero(p):
    a, ap, ai, aip = special.ai_zeros(p)
    return (a[-1])

def T(m,p):
    a=airy_zero(p)
    T = m-a*(m/2)**(1/3)+3/20*a**2*(m/2)**(-1/3) \
        + (a**3+10)/1400*(m/2)**(-1)-a*(479*a**3-40)/504000*(m/2)**(-5/3)
    return T

def lambda_m_p(m,p,polarization,n,R): # following formula A3 from Demchenko and Gorodetsky
    if polarization=='TE':
        P=1
    elif polarization=='TM':
        P=1/n**2
    temp=T(m,p)-n*P/np.sqrt(n**2-1)+airy_zero(p)*(3-2*P**2)*P*n**3*(m/2)**(-2/3)/6/(n**2-1)**(3/2) \
         - n**2*P*(P-1)*(P**2*n**2+P*n**2-1)*(m/2)**(-1)/4/(n**2-1)**2
    return 2*np.pi*n*R/temp

def lambda_m_p_with_dispersion(m,p,polarization,n,R): # following formula A3 from Demchenko and Gorodetsky
    if polarization=='TE':
        P=1
    elif polarization=='TM':
        P=1/n**2
    temp=T(m,p)-n*P/np.sqrt(n**2-1)+airy_zero(p)*(3-2*P**2)*P*n**3*(m/2)**(-2/3)/6/(n**2-1)**(3/2) \
         - n**2*P*(P-1)*(P**2*n**2+P*n**2-1)*(m/2)**(-1)/4/(n**2-1)**2
    return 2*np.pi*n*R/temp


class Resonances():
    #########################
    ### Structure is as follows:
    ### {Polarization_dict-> list(p_number)-> np.array(m number)}
    
    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
    pmax=10
    dispersion=False
    
    def __init__(self,wave_min,wave_max,n,R,p_max=10,dispersion=False):
        m0=np.floor(2*np.pi*n*R/wave_max)
        self.pmax=p_max
        self.Structured={'TE':[],'TM':[]}
        self.dispersion=dispersion
        N=0
        for Pol in ['TE','TM']:
            p=1
            if Pol=='TE':
                m=int(np.floor(m0*( 1 + airy_zero(p)*(2*m0**2)**(-1/3)+ n/(m0*(n**2-1)**0.5))))-3
            else:
                m=int(np.floor(m0*( 1 + airy_zero(p)*(2*m0**2)**(-1/3)+ 1/n/(m0*(n**2-1)**0.5))))-3
            if not self.dispersion:
                wave=lambda_m_p(m,p,Pol,n,R)
            while wave>wave_min and p<self.pmax+1: 
                resonance_temp_list=[]
                resonance_m_list=[]
                while wave>wave_min: 
                    if wave<wave_max:
                        resonance_temp_list.append(wave)
                        resonance_m_list.append(m)
                        N+=1
                    m+=1
                    wave=lambda_m_p(m,p,Pol,n,R)
                Temp=np.column_stack((np.array(resonance_temp_list),np.array(resonance_m_list)))
                self.Structured[Pol].append(Temp)
                p+=1
                if Pol=='TE':
                    m=np.floor(m0*( 1 + airy_zero(p)*(2*m0**2)**(-1/3)+ n/(m0*(n**2-1)**0.5)))-3
                else:
                    m=np.floor(m0*( 1 + airy_zero(p)*(2*m0**2)**(-1/3)+ 1/n/(m0*(n**2-1)**0.5)))-3
                wave=lambda_m_p(m,p,Pol,n,R)
        self.N_of_resonances=N
                
    def create_unstructured_list(self,Polarizations_to_account):  
        if Polarizations_to_account=='both':
            Polarizations=['TE','TM']
        elif Polarizations_to_account=='TE':
            Polarizations=['TE']
        elif Polarizations_to_account=='TM':
            Polarizations=['TM']
        list_of_resonances=[]
        list_of_labels=[]
        for Pol in Polarizations:
            for p,L in enumerate(self.Structured[Pol]):
                for wave,m in L:
                    list_of_resonances.append(wave)
                    list_of_labels.append(Pol+','+str(int(m))+','+str(p+1))
                    
        labels=[x for _,x in sorted(zip(list_of_resonances,list_of_labels))]#, key=lambda pair: pair[0])]
        resonances=sorted(list_of_resonances)
        return np.array(resonances),labels
    
                       
    def find_max_distance(self):
        res,_=self.create_unstructured_list(Polarizations)
        return np.max(np.diff(res))
    
    def plot_all(self,y_min,y_max,Polarizations_to_account):
#        plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
        resonances,labels=self.create_unstructured_list(Polarizations_to_account)
        for i,wave in enumerate(resonances):
            if labels[i].split(',')[0]=='TM':
                color='blue'
            else:
                color='red'
            plt.axvline(wave,ymin=y_min,ymax=y_max,color=color)
            y=y_min+(y_max-y_min)/self.pmax*float(labels[i].split(',')[2])
            plt.annotate(labels[i],(wave,y))
            

            
            
if __name__=='__main__':   
    resonances=Resonances(1552,1558,n=1.45,R=62.59e3,p_max=4)
    tempdict=resonances.__dict__
    plt.figure(2)
    resonances.plot_all(0,1,'both')
    plt.xlim([1552,1558])


