''' Implementing 2D linear convection :
    Governing Equation :

        ∂u/∂t + c*∂u/∂x + c*∂u/∂y = 0

    This is the exact same form as with 1-D Linear Convection, except that we
    now have two spatial dimensions to account for as we step forward in time.
    The time step will be discretized as a forward difference and both spatial steps
    will be discretized as backward differences.
    We use i to track our x values and y to track our y values.

    Discretized Equation :

        (u_i,j(n+1) - u_i,j(n))/Δt + c*(u_i,j(n) - u_i-1,j(n))/Δx + c*(u_i,j(n) - u_i,j-1(n)/Δy) = 0

    Soving for u_i,j(n+1):

        u_i,j(n+1) = u_i,j(n) - c*(Δt/Δx)*(u_i,j(n)-u_i-1,j(n)) - c*(Δt/Δy)*(u_i,j(n)-u_i,j-1(n))

    Initial Conditions :

        u(x,y) = 2 for 0.5≤x,y≤1 and 1 everywhere else

    Boundary Conditions:

        u = 1 for x = 0,2 and y = 0,2

'''

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Variable Declaration :
nx = 81     # Number of grid points in the x direction
ny = 81     # Number of grid points in the y direction
nt = 100    # Number of time steps
c = 1   # Wave speed
dx = 2/(nx-1)    # distance between any two adjacent grid points in the x direction
dy = 2/(ny-1)     # distance between any two adjacent grid points in the y direction
sigma = 0.2
dt = sigma*dx

x = np.linspace(0,2,nx)     # 0 to 2, nx elements long, x direction
y = np.linspace(0,2,ny)     # 0 to 2, ny elements long, y direction

'''u = np.ones((ny,nx))        # nx X ny array of 1's
un = np.ones((ny,nx))       # "---"----"

# Initial Conditions (Hat function):
u[int(.5/dy):int(1/dy+1), int(.5/dx):int(1/dx+1)] = 2

# Plot Initial Condition :
fig = plt.figure(figsize=(11,7), dpi=100)  # figsize - (w,h) tuple in inches. Can be changed to produce different sized images
ax = fig.gca(projection='3d')       # gets current axes
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X,Y, u[:], cmap=cm.viridis)
plt.show()
'''

u = np.ones((ny, nx))
u[int(.5/dy):int(1/dy + 1), int(.5/dx):int(1/dx + 1)] = 2

for n in range(nt+1):   # Loop across Number of time steps
    un = u.copy()
    row, col = u.shape
    for j in range(1, row):
        for i in range(1,col):
            u[j,i] = (un[j,i] - c*(dt/dx)*(un[j,i] - un[j,i-1]) - c*(dt/dy)*(un[j,i] - un[j-1,i]))
            u[0,:] = 1  # first row
            u[-1,:] = 1     # last row
            u[:,0] = 1      # first column
            u[:,-1] = 1     # last column

fig = plt.figure(figsize=(11,7), dpi = 100)
ax = fig.gca(projection = '3d')
X, Y = np.meshgrid(x, y)
surf2 = ax.plot_surface(X,Y,u[:], cmap=cm.viridis)
plt.suptitle("2D Linear Convection")
plt.show()
