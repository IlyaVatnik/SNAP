# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 17:29:23 2023

@author: Илья
"""
import numpy as np
from scipy import sparse
import scipy.linalg as la
import matplotlib.pyplot as plt
x=np.linspace(-1,1,1000)
N=len(x)
a=1
U=x**2*a

dx=x[1]-x[0]


Tmtx=-1/dx**2*sparse.diags([-2*np.ones(N),np.ones(N)[1:],np.ones(N)[1:]],[0,-1,1]).toarray()
Vmtx=np.diag(U)
Hmtx=Tmtx+Vmtx
(eigvals,eigvecs)=la.eigh(Hmtx,check_finite=False)


sorted_indexes=np.argsort(np.real(eigvals))
eigvals,eigvecs=[eigvals[sorted_indexes],eigvecs.T[sorted_indexes]]
eigvecs=eigvecs/np.sqrt(dx)  # to get normalization for integral (psi**2 dx) =1

plt.plot(eigvecs[0])