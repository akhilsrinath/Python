#  integrate a funciton between two limits a and b.

import numpy as np

def integrate(f, a, b, N):
    x = np.linspace(a+(b-a)/(2*N), b-(b-a)/(2*N), N)    # a number line with N points between a and b stored in a vector x
    fx = f(x)
    area = np.sum(fx)*(b-a)/N    # total area of all the rectangles combined
    return area

print(integrate(np.sin, 0, np.pi/2, 100))
