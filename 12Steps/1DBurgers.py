'''
    Implementing the Burgers' Equation in 1D :

        ∂u/∂t + u*∂u/∂x = ν*∂2u/∂x2

    Discretized Equation :

        u_i(n+1) = u_i(n) - u_i(n)*dt/dx*(u_i(n)-u_i-1(n)) + nu*(dt/dx**2)*(u_i+1(n) - 2*u_i(n) + u_i-1(n))

    Initial conditions:

        u = − 2ν/ϕ * ∂ϕ/∂x + 4
        ϕ = exp(−(x−4t)**2/4ν(t+1))+exp(−(x−4t−2π)**2/4ν(t+1))

    Boundary conditions :
        u(0) = u(2π)        ---> Periodic Boundary condition

    We plot the analytical solution and compare it with the computational solution using the finite difference equation.
'''

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from sympy import init_printing
init_printing(use_latex = True)

x, nu, t = sp.symbols('x nu t')
phi = (sp.exp(-(x-4*t)**2/(4*nu*(t+1)))) + (sp.exp(-(x - 4*t - 2*np.pi)**2/(4*nu*(t+1))))   # Auxiliary function

# Now to evaluate ∂ϕ/∂x

phiprime = phi.diff(x)
#print(phiprime)

from sympy.utilities.lambdify import lambdify

u = -2*nu*(phiprime/phi) + 4
#print(u)

ufunc = lambdify((t,x,nu),u)
#print(ufunc(1,4,3)) ---> TEST

# variable declaration :

nx = 101    # Nunber of grid points
nt = 100    # Number of time steps
dx = (2*np.pi)/(nx -1)    # distance between any pair of adjacent grid points
nu = 0.07   # viscocity
dt = dx * nu    # nu = dt/dx

x = np.linspace(0,2*np.pi,nx)   # 0 to 2π, nx elements long
un = np.empty(nx)   # array that will be filled later
t = 0
u = np.asarray([ufunc(t,x0,nu) for x0 in x])    # Initial Condition. "Saw-tooth function"

# Periodic Boundary conditions :
# With periodic boundary conditions, when a point gets to the right-hand side of the frame, it wraps around back to the front of the frame.

for n in range(nt):
    un = u.copy()
    for i in range(1, nx-1):
        u[i] = un[i] - un[i]*(dt/dx)*(un[i]-un[i-1]) + nu*(dt/dx**2)*(un[i+1] - 2*un[i] + un[i-1])
        u[0] = un[0] - un[0]*(dt/dx)*(un[0]-un[-2]) + nu*(dt/dx**2)*(un[1] - 2*un[0] + un[-2])
        u[-1] = u[0]

u_analytical = np.asarray([ufunc(nt*dt, xi, nu) for xi in x])

plt.figure(figsize=(11,7), dpi=100)
plt.plot(x,u,marker='o', lw =2, label = 'Computational')
plt.plot(x,u_analytical,label='Analytical')
plt.xlim([0,2*np.pi])
plt.ylim([0,10])
plt.legend()
plt.suptitle("Comaprison of Analytical and Computational solutions to the 1D Burgers' Equation")
plt.show()
