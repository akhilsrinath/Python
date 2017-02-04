# implementation of the gaussian equation

import numpy as np
import matplotlib.pyplot as plt
import math

x = np.linspace(-3,3,100)
def gaussian(x,m,s):
    return 1/(math.sqrt(2*np.pi)*s) * np.exp(-0.5 * ((x-m)/s)**2)

print(gaussian(x,0,2))
plt.plot(gaussian(x,0,2))
plt.show()
