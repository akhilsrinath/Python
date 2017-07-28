''' The Poisson Equation is obtained by adding a source term to the RHS of the Laplace Equation.
    Governing Equation :

        ∂2p/∂x2 + ∂2p/∂y2 = b

    'b' is a source term - a finite value inside the field that affects the solution

    Discretized Equation :

        p_i,j(n) = ((p_i+1,j(n) + p_i-1,j(n))*dy**2 + (p_i,j+1(n) + p_i,j-1(n))*dx**2)/2*(dx**2 + dy**2)

    Initial Conditions :

        p = 0 everywhere in (0,2)X(0,1)

    Boundary Conditions :

        p = 0 @ x = 0,2 and y = 0,1

    The source term 'b' consists of two initial spikes inside the domain, as follows:

        b_i,j = 100 at i = nx/4, j = ny/4
        b_i,j = -100 at i = nx*3/4, j = ny*3/4
        b_i,j = 0 everywhere else

'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# Variable Declarations :

nx = 50
ny = 50
nt = 100
c = 1
xmin = 0
xmax = 2
ymin = 0
ymax = 1

dx = (xmax - xmin)/(nx - 1)
dy = (ymax - ymin)/(ny - 1)

# initialization :

p = np.zeros((ny,nx))
pd = np.zeros((ny,nx))
b = np.zeros((ny,nx))
x = np.linspace(xmin,xmax,nx)
y = np.linspace(ymin,ymax,ny)

# Source term

b[int(ny/4),int(nx/4)] = 100
b[int(3*ny/4),int(3*nx/4)] = -100

for it in range(nt):
    pd = p.copy()

    p[1:-1,1:-1] = (((pd[1:-1,2:] + p[1:-1,:-2])*dy**2 + (pd[2:,1:-1] + pd[:-2,1:-1])*dx**2 - (b[1:-1,1:-1])*(dx**2)*(dy**2))/(2*(dx**2 + dy**2)))

    p[0,:] = 0
    p[ny-1,:] = 0
    p[:,0] = 0
    p[:,nx-1] = 0

def plot_2D(x,y,p):
    fig = plt.figure(figsize=(11,7), dpi=100)
    ax = fig.gca(projection='3d')
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,p[:],rstride=1,cstride=1,cmap=cm.viridis,linewidth=0,antialiased=False)
    ax.view_init(30,225)
    plt.suptitle("Implementation of the 2D Poisson Equation")
    plt.show()

plot_2D(x,y,p)
