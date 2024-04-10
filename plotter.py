import numpy as np
import matplotlib.pylab as plt

from typing import List

from simulator import Simulator
from simulator import MtxType as MT

from interpolator import Interpolator
from interpolator import InterpolationType as IT


size : int = 20
mu : float = 10.
sigma : float = 2.
strength : int = 12
noise : int = 2
density : int = 10
sparsity : int = 2

simulator = Simulator(size, mu, sigma, strength, noise, density, sparsity, MT.RAD_GRAD_NOISE)

field : List[List[int]] = simulator.get_field()
probe : List[List[int]] = simulator.get_probe()

for row in range(size):
    for col in range(size):
        print(field[row][col])

interpolator = Interpolator(probe, IT.NNM)
interpolated : List[List[int]] = interpolator.get_interpolated()
        
figure = plt.figure()

axes = figure.add_subplot(1, 3, 1)
axes.set_aspect('equal')
plt.imshow(field, interpolation = 'nearest', cmap = plt.cm.plasma)

axes = figure.add_subplot(1, 3, 2)
axes.set_aspect('equal')
plt.imshow(probe, interpolation = 'nearest', cmap = plt.cm.plasma)

axes = figure.add_subplot(1, 3, 3)
axes.set_aspect('equal')
plt.imshow(interpolated, interpolation = 'nearest', cmap = plt.cm.plasma)

plt.show()