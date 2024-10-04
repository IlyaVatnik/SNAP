'''
https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/04_PartialDifferentialEquations/04_04_Diffusion_Implicit.html
'''
import sys
# sys.path.insert(0, '../modules')

# Euler step functions
# from steppers import euler_step
import numpy as np
from scipy.sparse import diags
# Function to compute an error in L2 norm
# from norms import l2_diff

# Function to compute d^2 / dx^2 with second-order 
# accurate centered scheme
# from matrices import d2_mat_dirichlet
# from pde_module import rhs_heat_centered, exact_solution
import matplotlib.pyplot as plt
import numpy as np

# Function for the RHS of the heat equation and
# exact solution of our example problem



def d2_mat_dirichlet(nx, dx):
    """
    Constructs the centered second-order accurate second-order derivative for
    Dirichlet boundary conditions.

    Parameters
    ----------
    nx : integer
        number of grid points
    dx : float
        grid spacing

    Returns
    -------
    d2mat : numpy.ndarray
        matrix to compute the centered second-order accurate first-order deri-
        vative with Dirichlet boundary conditions on both side of the interval
    """
    # We construct a sequence of main diagonal elements,
    diagonals = [[1.], [-2.], [1.]]
    # and a sequence of positions of the diagonal entries relative to the main
    # diagonal.
    offsets = [-1, 0, 1]

    # Call to the diags routine; note that diags return a representation of the
    # array; to explicitly obtain its ndarray realisation, the call to .toarray()
    # is needed. Note how the matrix has dimensions (nx-2)*(nx-2).
    d2mat = diags(diagonals, offsets, shape=(nx-2,nx-2)).toarray()

    # Return the final array divided by the grid spacing **2.
    return d2mat / dx**2


specific_heat_capacity=740 # J/kg/K
density=2.2*1e-3*1e-3 # kg/mm**3

thermal_conductivity=1.38*1e-3 # W/mm/K
heat_exchange=10*1e-6 #W/mm**2/K

r=62.5e-3
P=0.00063
fraction=0.05

# Physical parameters
alpha = thermal_conductivity/   specific_heat_capacity/ density              # Heat transfer coefficient
dzita=1/specific_heat_capacity/density/np.pi/r**2



lx = 6                        # Size of computational domain
ti = 0.0                       # Initial time
tf = 50                 # Final time

# Grid parameters
nx = 500                      # number of grid points
dx = lx / (nx-1)               # grid spacing
x = np.linspace(0., lx, nx)    # coordinates of grid points

# Solution parameters
T0 =     np.zeros(len(x))          # initial condition
source =np.zeros(len(x)) # 2*np.sin(np.pi*x)          # heat source term
source[nx//2:nx//2+100]=P*fraction*dzita/dx

fourier = 55              # Fourier number
dt = fourier*dx**2/alpha  # time step

nt = int((tf-ti)/dt)      # number of time steps

# d^2 / dx^2 matrix with Dirichlet boundary conditions
D2 = d2_mat_dirichlet(nx, dx)     

# I+A matrix
M = np.eye(nx-2) + alpha*dt*D2

# I-0.5*A matrix + inverse
A = np.eye(nx-2) - 0.5*alpha*dt*D2
Ainv = np.linalg.inv(A)

# I+0.5*A matrix
B = np.eye(nx-2) + 0.5*alpha*dt*D2

# (I+0.5A)^{-1} * (I-0.5*A)
C = np.dot(Ainv, B)

T = np.empty((nt+1, nx)) # Allocate storage for the solution    
T[0] = T0.copy()         # Set the initial condition


scn = np.dot(Ainv, source[1:-1]*dt)
for i in range(nt):
    T[i+1, 1:-1] = np.dot(C, T[i, 1:-1]) + scn

# Set boundary values
T[-1,0] = 0
T[-1,-1] = 0


plt.figure()
plt.plot(x,T[-1,:])
