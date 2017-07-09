''' Implementing the 1D Diffuion Equation
    Governing Equation :
        u_t = v * u_xx (where v = viscocity)
    After discretizing u_xx, we get u_xx = (u_i+1 - 2*u_i + u_i-1)/dx**2 + O(dx**4)

    Discretized Equation:
        u_i(n+1) = u_i(n) + (v*dt/dx**2)*(u_i+1(n) - 2*u_i(n) + u_i-1(n))

    Initial and Boundary conditions : at t=0, u=2 in the interval 0.5≤x≤1 and u=1 everywhere else.
'''

import numpy as np
import matplotlib.pyplot as plt

nx = 41     #  number of grid points lying between 0 and 2
nt = 20     # number of time steps
dx = 2/(nx-1)   # distance between any pair of adjacent grid points sigma = 0.2
nu = 0.3    # viscocity value
sigma = 0.2
dt = (sigma * (dx**2))/nu       # amount of time each time step covers

u = np.ones(nx)
# 1 everywhere in the function
# nx elements long

u[int(0.5/dx):int(1/dx+1)] = 2      # u=2 between 0.5 and 1 as per our initial conditions

un = np.ones(nx)        # placeholder array

for n in range(nt):
    un = u.copy()
    for i in range(1,nx-1):
        u[i] = un[i] + (nu*(dt/(dx**2)))*(un[i+1] - 2*(un[i]) + un[i-1])

print(u)
plt.plot(u)
plt.show()
