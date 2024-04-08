import numpy as np
import matplotlib.pylab as plt

from typing import List

from simulator import FieldSimulator
from simulator import MtxType as MT


size : int = 20
mu : float = 10.
sigma : float = 2.
strength : int = 100
noise : int = 1
density : int = 7

simulator = FieldSimulator(size, mu, sigma, strength, noise, density, MT.RAD_GRAD_NOISE)

field : List[List[float]] = simulator.get_field()
probe : List[List[float]] = simulator.get_probe()

for row in range(size):
    for col in range(size):
        print(field[row][col])
        
figure = plt.figure()

axes = figure.add_subplot(1, 2, 1)
axes.set_aspect('equal')
plt.imshow(field, interpolation = 'nearest', cmap = plt.cm.plasma)

axes = figure.add_subplot(1, 2, 2)
axes.set_aspect('equal')
plt.imshow(probe, interpolation = 'nearest', cmap = plt.cm.plasma)

plt.show()