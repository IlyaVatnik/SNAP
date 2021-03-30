import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


class raw_signal():
    def __init__(self,f_name,first_column=2):
        Temp=np.genfromtxt(f_name,skip_header=9)
        
        self.lambdas=np.array(Temp[:,0])
        self.N=len(self.lambdas)
        self.JonesMatrixes=[]
        for i in range(len(self.lambdas)):
            a=Temp[i,first_column]*np.exp(1j*Temp[i,first_column+4])
            b=Temp[i,first_column+1]*np.exp(1j*Temp[i,first_column+5])
            c=Temp[i,first_column+2]*np.exp(1j*Temp[i,first_column+6])
            d=Temp[i,first_column+3]*np.exp(1j*Temp[i,first_column+7])
            self.JonesMatrixes.append(np.array([[a,b],[c,d]]))
        
        
    def get_IL(self):
        IL=np.zeros(self.N)
        for i in range(self.N):
            IL[i]=np.sum(abs(self.JonesMatrixes[i])**2)/2
        return IL
    
    def get_principal_IL(self):
        IL_1=np.zeros(self.N)
        IL_2=np.zeros(self.N)
        for i in range(self.N):
            IL_1[i],IL_2[i]=abs(la.eigvals(self.JonesMatrixes[i]))**2
        return IL_1,IL_2
    
    def plot_IL(self, scale='log'):
        IL=self.get_IL()
        if scale=='log':
            plt.plot(self.lambdas, 10*np.log10(IL),label='IL')
            plt.xlabel('Wavelength, nm')
            plt.ylabel('Insertion losses, dB')
            
    def plot_principal_IL(self, scale='log'):
        IL_1,IL_2=self.get_principal_IL()
        if scale=='log':
            plt.plot(self.lambdas, 10*np.log10(IL_1),label='IL_1')
            plt.plot(self.lambdas, 10*np.log10(IL_2),label='IL_2')
            plt.ylabel('Insertion losses, dB')
        plt.xlabel('Wavelength, nm')
        plt.legend()
        

            
# =============================================================================
#       testing 
# =============================================================================
def get_IL_from_file(f_name):
    Temp=np.genfromtxt(f_name,skip_header=9)
    return Temp[:,0],Temp[:,2]

def get_IL_considering_taper(f_name_in,f_name_out):
    
    signal_in_contact=raw_signal(f_name_in)
    signal_out_contact=raw_signal(f_name_out)
            
    IL_1=np.zeros(signal_in_contact.N)
    IL_2=np.zeros(signal_in_contact.N)
    
    for i in range(signal_in_contact.N):
        M=np.dot(la.inv(signal_out_contact.JonesMatrixes[i]),signal_in_contact.JonesMatrixes[i])
        results=la.eig(M)
        IL_1[i],IL_2[i]=abs(results[0])**2
    
    plt.figure()
    plt.plot(signal_in_contact.lambdas,IL_1,label='IL_1')
    plt.plot(signal_in_contact.lambdas,IL_2,label='IL_2')
    # plt.plot(10*np.log10(IL_my),label='my')
    plt.legend()
    plt.title(str(f_name_in))
    
    # plt.figure()
    # plt.plot(signal_in_contact.lambdas,10*np.log10((IL_1+IL_2)/2),label='sum of two principle IL')
    # signal_in_contact.plot_IL()
    # plt.plot(signal_in_contact.lambdas,10*np.log10(IL_1),label='IL_1')
    # plt.plot(signal_in_contact.lambdas,10*np.log10(IL_2),label='IL_2')
    # plt.legend()

# =============================================================================
# example
# =============================================================================
if __name__=='__main__':
    f_name_in='In contact.txt'
    f_name_out='Out of contact.txt'
    signal_in=raw_signal(f_name_in,first_column=9)
    signal_out=raw_signal(f_name_out,first_column=9)
    
    IL_1=np.zeros(signal_in.N)
    IL_2=np.zeros(signal_in.N)

    for i in range(signal_in.N):
        M=np.dot(la.inv(signal_out.JonesMatrixes[i]),signal_in.JonesMatrixes[i])
        results=la.eig(M)
        IL_1[i],IL_2[i]=abs(results[0])**2

    plt.figure()
    plt.plot(signal_in.lambdas,IL_1,label='IL_1')
    plt.plot(signal_in.lambdas,IL_2,label='IL_2')
# plt.plot(10*np.log10(IL_my),label='my')
    plt.legend()

    plt.figure()
    plt.plot(signal_in.lambdas,10*np.log10((IL_1+IL_2)/2),label='sum of two principle IL')
    signal_in.plot_IL()
    plt.plot(signal_in.lambdas,10*np.log10(IL_1),label='IL_1')
    plt.plot(signal_in.lambdas,10*np.log10(IL_2),label='IL_2')
    plt.legend()
        

    
    
    