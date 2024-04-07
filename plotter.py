import numpy as np
import matplotlib.pylab as plt

from simulator import FieldSimulator
from simulator import MtxType as MT


size : int = 20

mtx = FieldSimulator.get_radial_gradient_noise(size)

for row in range(size):
    for col in range(size):
        print(mtx[row][col])
        
figure = plt.figure()
axis = figure.add_subplot(1, 1, 1)
axis.set_aspect('equal')

plt.imshow(mtx, interpolation = 'nearest', cmap = plt.cm.plasma)
plt.colorbar()
plt.show()