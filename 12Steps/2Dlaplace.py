''' Implementation of the 2D Laplace Equation :

    Governing Equation :

        ∂2p/∂x2 + ∂2p/∂y2 = 0

    Discretized Equation :

        (p_i+1,j(n) - 2*p_i,j(n) + p_i-1,j(n))/dx**2 + (p_i,j+1(n) - 2*p_i,j(n) + p_i,j-1(n))/dy**2 = 0

    Rearranging and solving for p_i,j(n):

        p_i,j(n) = ((p_i+1,j(n) + p_i-1,j(n))*dy**2 + (p_i,j+1(n) + p_i,j-1(n))*dx**2)/2*(dx**2 + dy**2)

    Initial conditions :

        p = 0 everywhere

    Boundary Conditions :

        p = 0 at x = 0
        p = y at x = 2
        ∂p/∂y = 0 at y = 0,1

    Under these conditions, there is an analytical solution to the Laplace Equation :

        p(x,y)=(x/4) − 4∑n=1,odd to ∞ (1/((nπ)**2*sinh2nπ)*sinhnπxcosnπy

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot_2D(x,y,p):
    fig = plt.figure(figsize=(11,7), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x,y)
    surf = ax.plot_surface(X,Y,p[:],cmap=cm.viridis,cstride=1,rstride=1,linewidth=0,antialiased=False)
    ax.set_xlim(0,2)
    ax.set_ylim(0,1)
    ax.view_init(30,225)
    plt.suptitle("Implementation of the 2D Laplace Equation")
    plt.show()

def laplace_2d(p,y,dx,dy,l1norm_target):
    l1norm = 1
    pn = np.empty_like(p)
    while l1norm > l1norm_target:
        pn = p.copy()
        p[1:-1,1:-1] = ((dy**2 * (pn[1:-1,2:] + pn[1:-1,0:-2]) + dx**2 * (pn[2:,1:-1] + pn[0:-2,1:-1]))/(2*(dx**2 + dy**2)))

        p[:,0] = 0      # p = 0 at x = 0
        p[:,-1] = y     # p = y at x = 2 (last column)
        p[0,:] = p[1,:]     # dp/dy = 0 at y = 0
        p[-1,:] = p[-2,:]   # dp/dy = 0 at y = 1
        l1norm = (np.sum(np.abs(p[:]) - np.abs(pn[:]))/np.sum(np.abs(pn[:])))

    return p

# Variable Declarations :

nx = 31
ny = 31
c = 1
dx = 2/(nx - 1)
dy = 2/(nx - 1)

# Initial Conditions :

p = np.zeros((ny,nx))
x = np.linspace(0,2,nx)
y = np.linspace(0,1,ny)
p = laplace_2d(p,y,dx,dy,1e-4)
# Boundary Conditions:

p[:,0] = 0      # p = 0 at x = 0
p[:,-1] = y     # p = y at x = 2 (last column)
p[0,:] = p[1,:]     # dp/dy = 0 at y = 0
p[-1,:] = p[-2,:]   # dp/dy = 0 at y = 1

print(plot_2D(x,y,p))
