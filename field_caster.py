import math
import numpy as np

from enum import Enum


class MtxType(Enum):
    GAUSS_NORM = 1
    GAUSS_OFFS = 2
    
    
class FieldCaster:
    
    m_sim_mtx = [[]]
    
    m_size = 20
    m_mu = 10
    m_sigma = 2
    
    def __init__(self, size, mu, sigma, type = MtxType.GAUSS_NORM):
        self.m_size = size
        self.m_mu = mu
        self.m_sigma = sigma
        
        match type:
            case MtxType.GAUSS_NORM:
                self.init_mtx_gaussian_normal()
            case MtxType.GAUSS_OFFS:
                self.init_mtx_gaussian_offset()
    
    def init_mtx_gaussian_normal(self):
        self.m_sim_mtx = np.random.normal(self.m_mu, self.m_sigma, (self.m_size, self.m_size))
        return
    
    def init_mtx_gaussian_offset(self):
        self.m_sim_mtx = np.random.normal(self.m_mu, self.m_sigma, (self.m_size, self.m_size))
        for col in range(self.m_size):
            for row in range(self.m_size):
                self.m_sim_mtx[row][col] = math.trunc(self.m_sim_mtx[row][col]) + self.m_mu
        return