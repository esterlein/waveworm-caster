import numpy as np
import matplotlib.pylab as plt

from field_caster import FieldCaster
from field_caster import MtxType as MT


mtx = FieldCaster(20, 10, 2, MT.RAD_GRAD_CLEAN)

for row in range(mtx.m_size):
    for col in range(mtx.m_size):
        print(mtx.m_sim_mtx[row][col])
        
figure = plt.figure()
axis = figure.add_subplot(1, 1, 1)
axis.set_aspect('equal')

plt.imshow(mtx.m_sim_mtx, interpolation = 'nearest', cmap = plt.cm.ocean)
plt.colorbar()
plt.show()