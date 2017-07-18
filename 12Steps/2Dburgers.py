'''
    Implementing Burgers' Equation in 2D

    Governing Equation :

        ∂u/∂t + u*∂u/∂x + v*∂u/∂y = ν*(∂2u/∂x2 + ∂2u/∂y2)

        ∂v/∂t + u*∂v/∂x + v*∂v/∂y = ν*(∂2v/∂x2 + ∂2v/∂y2)

    Discretized Equation :

        u_i,j(n+1) = u_i,j(n+1) - (dt/dx)*(u_i,j(n))*(u_i,j(n) - u_i-1,j(n)) - (dt/dy)*v_i,j(n)*(u_i,j(n) - u_i,j-1(n)) + ((nu*dt)/dx**2)*
                     (u_i+1,j(n) - 2*u_i,j(n) + u_i-1,j(n)) + ((nu*dt)/dy**2)*(u_i,j+1(n) - 2*u_i,j(n) + u_i,j+1(n))

        v_i,j(n+1) = v_i,j(n+1) - (dt/dx)*(u_i,j(n))*(v_i,j(n) - v_i-1,j(n)) - (dt/dy)*v_i,j(n)*(v_i,j(n) - v_i,j-1(n)) + ((nu*dt)/dx**2)*
                         (v_i+1,j(n) - 2*v_i,j(n) + v_i-1,j(n)) + ((nu*dt)/dy**2)*(v_i,j+1(n) - 2*v_i,j(n) + v_i,j+1(n))

    Initial Conditions :

        u(x,y) = 2 for 0.5≤x,y≤1 and 1 everywhere else

    Boundary Conditions :

        u = 1 for x = 0,2 and y = 0,2
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Variable Declaration :

nx = 41     # Number of grid points in the x direction
ny = 41     # wNumber of grid points in the y direction
nt = 120    # Number of time steps
c = 1       # Number of time steps
dx = 2/(nx - 1)     # Distance between two adjacent points in the x direction
dy = 2/(ny - 1)     # Distance between two adjacent points in the y direction
sigma = 0.0009
nu = 0.01
dt = (sigma*dx*dy)/nu

x = np.linspace(0,2,nx)
y = np.linspace(0,2,ny)

u = np.ones((ny,nx))
v = np.ones((ny,nx))
un = np.ones((ny,nx))
vn = np.ones((ny,nx))
comb = np.ones((ny,nx))

# Set initial conditions: (Hat function):

u[int(0.5/dy):int(1/dy + 1), int(0.5/dx):int(1/dx + 1)] = 2
v[int(0.5/dy):int(1/dy + 1), int(0.5/dx):int(1/dx + 1)] = 2

# plotting initial condition :

'''
fig = plt.figure(figsize=(11,7), dpi=100)
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x,y)
ax.plot_surface(X,Y,u[:],cmap=cm.viridis,cstride=1,rstride=1)
ax.plot_surface(X,Y,v[:],cmap=cm.viridis,cstride=1,rstride=1)
plt.suptitle("Hat Funciton Plot")
plt.show()
'''

for n in range(nt+1):
    un = u.copy()
    vn = v.copy()

    u[1:-1,1:-1] = (un[1:-1,1:-1] - (dt/dx)*(un[1:-1,1:-1])*(un[1:-1,1:-1] - un[1:-1,0:-2]) - (dt/dy)*(vn[1:-1,1:-1])*(un[1:-1,1:-1] - un[0:-2,1:-1]) + nu*(dt/dx**2)*(un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,0:-2]) + nu*(dt/dy**2)*(un[2:,1:-1] - 2*un[1:-1,1:-1] + un[0:-2,1:-1]))
    v[1:-1,1:-1] = (vn[1:-1,1:-1] - (dt/dx)*(un[1:-1,1:-1])*(vn[1:-1,1:-1] - vn[1:-1,0:-2]) - (dt/dy)*(vn[1:-1,1:-1])*(vn[1:-1,1:-1] - vn[0:-2,1:-1]) + nu*(dt/dx**2)*(vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,0:-2]) + nu*(dt/dy**2)*(vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[0:-2,1:-1]))

    u[0,:] = 1
    u[-1,:] = 1
    u[:,0] = 1
    u[:,-1] = 1

    v[0,:] = 1
    v[-1,:] = 1
    v[:,0] = 1
    v[:,-1] = 1

fig = plt.figure(figsize=(11,7),dpi=100)
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x,y)
ax.plot_surface(X,Y,u,cmap=cm.viridis,rstride=1,cstride=1)
ax.plot_surface(X,Y,v,cmap=cm.viridis,rstride=1,cstride=1)
plt.suptitle("Implementation of the 2D Burgers' Equation")
plt.show()
