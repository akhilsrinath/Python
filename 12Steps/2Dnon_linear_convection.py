'''
    Implementing non linear convection :
    Governing Equations :

        ∂u/∂t + u*∂u/∂x + v*∂u/∂y = 0
        ∂v/∂t + u*∂v/∂x + v*∂v/∂y = 0

    Discretized Equations:

        u_i,j(n+1) = u_i,j(n) - u_i,j*(dt/dx)*(u_i,j(n) - u_i-1,j(n)) - v_i,j(n)*(dt/dx)*(u_i,j(n) - u_i,j-1(n))
        v_i,j(n+1) = v_i,j(n) - u_i,j*(dt/dx)*(v_i,j(n) - v_i-1,j(n)) - v_i,j(n)*(dt/dx)*(v_i,j(n) - v_i,j-1(n))

    Initial Conditions :

        u,v = 2 for x,y∈(0.5,1)×(0.5,1) and 1, everywhere else

    Boundary Conditions :

        u = 1, v = 1 for x = 0,2 and y = 0,2

'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Variable Declaration :

nx = 101    # Number of grid points in the x direction
ny = 101    # Number of grid points in the y direction
nt = 80     # Number of time steps
c = 1       # Wave speed
dx = 2/(nx - 1)         # distance between any two adjacent grid points in the x direction
dy = 2/(ny - 1)         # distance between any two adjacent grid points in the y direction
sigma = 0.2
dt = sigma*dx

x = np.linspace(0,2,nx)
y = np.linspace(0,2,ny)

u = np.ones((ny,nx))
v = np.ones((ny,nx))
un = np.ones((ny,nx))
vn = np.ones((ny,nx))

# Set initial condition : (Hat function)

u[int(.5/dy):int(1/dy + 1), int(0.5/dx):int(1/dx + 1)] = 2
v[int(.5/dy):int(1/dy + 1), int(0.5/dx):int(1/dx + 1)] = 2

#fig = plt.figure(figsize = (11,7), dpi = 100)
#ax = plt.gca(projection = '3d')
#X, Y = np.meshgrid(x, y)
#ax.plot_surface(X, Y, u, cmap = cm.viridis, rstride=2, cstride=2)
#plt.show()

for i in range(nt + 1):
    un = u.copy()
    vn = v.copy()
    u[1:, 1:] = (un[1:,1:] - un[1:,1:] * c * (dt/dx) * (un[1:,1:] - un[1:,:-1]) - (vn[1:,1:] * c * (dt/dy) * (un[1:,1:] - un[:-1,1:])))
    v[1:, 1:] = (vn[1:,1:] - un[1:,1:] * c * (dt/dx) * (vn[1:,1:] - vn[1:,:-1]) - (vn[1:,1:] * c * (dt/dy) * (vn[1:,1:] - vn[:-1,1:])))

    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

fig = plt.figure(figsize=(11,7), dpi = 100)
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u,cmap=cm.viridis,rstride=2, cstride=2)
plt.show()
