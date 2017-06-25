''' Implementing non-linear convection
    Governing Equation:

        u_t + u*u_x = 0

    Instead of the constant factor c in the second term as in the case of linear convection, we now use the non-linear term u.
    We define an evenly spaced grid of points within a spatial domain
    that is 2 units of length wide, i.e., x_i∈(0,2). We'll define a variable nx,
    which will be the number of grid points we want and dx will be the distance
    between any pair of adjacent grid points.
    '''

import numpy as np
import matplotlib.pyplot as plt

nx = 41     # number of grid points lying between 0 and 2
dx = 2/(nx - 1)     # distance between any pair of adjacent grid points
nt = 25     # number of time steps
dt = 0.025      # amount of time each time step covers
c = 1       # We assume wave speed to be 1

''' Initial conditions: the initial velocity u_0 is given as u=2 in the interval
    0.5≤x≤1 and u=1 everywhere else in (0,2)
'''

u = np.ones(nx)
# 1 everywhere in the function
# nx elements long

u[int(0.5/dx):int(1/dx+1)] = 2      # u=2 between 0.5 and 1 as per our initial conditions
print(u)

# plotting:  Hat function:

#plt.plot(np.linspace(0,2,nx), u)
#plt.show()

''' implementing the discretization of the non-linear convection equation using a finite difference scheme.'''

un = np.ones(nx)     # initializing a temporary variable un

for n in range(nt):    # 0 to nt -> will run nt times
    un = u.copy()    # copy the existing values of u into un
    for i in range(1,nx):    # 1 to nx    : taking initial conditions into consideration
        u[i] = un[i] - un[i] * (dt/dx) * (un[i] - un[i-1])     # applying the operation to every element in u.

plt.plot(np.linspace(0,2,nx), u)
plt.show()
