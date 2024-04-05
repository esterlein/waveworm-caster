from field_caster import FieldCaster
from field_caster import MtxType as MT


mtx = FieldCaster(20, 10, 2, MT.GAUSS_NORM)

for row in range(mtx.m_size):
    for col in range(mtx.m_size):
        print(mtx.m_sim_mtx[row][col])