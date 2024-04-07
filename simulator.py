import math
import random
import numpy as np

from enum import Enum
from typing import List


class MtxType(Enum):
    GAUSS_NORM = 1
    GAUSS_OFFS = 2
    RAD_GRAD_CLEAN = 3
    RAD_GRAD_NOISE = 4
    
    
class FieldSimulator:
    
    m_sim_mtx = List[List[float]]
    
    m_size : int = 20
    
    m_mu : float = 10.
    m_sigma : float = 2.
    
    
    def __init__(self, size, mu, sigma, type = MtxType.GAUSS_NORM):
        self.m_size = size
        self.m_mu = mu
        self.m_sigma = sigma
        
        match type:
            case MtxType.GAUSS_NORM:
                self.get_mtx_gaussian_normal(self.m_sim_mtx)
            case MtxType.GAUSS_OFFS:
                self.get_mtx_gaussian_offset(self.m_sim_mtx)
            case MtxType.RAD_GRAD_CLEAN:
                self.get_radial_gradient_clean(self.m_sim_mtx)
            case MtxType.RAD_GRAD_NOISE:
                self.get_radial_gradient_noise(self.m_sim_mtx)
    
    
    def get_gaussian_normal(self, size : int, mu : float, sigma : float) -> List[List[float]]:
        mtx = np.random.normal(mu, sigma, (size, size))
        return mtx
    
    
    def get_gaussian_offset(self, size : int, mu : float, sigma : float) -> List[List[float]]:
        mtx = np.random.normal(mu, sigma, (size, size))
        
        for col in range(size):
            for row in range(size):
                mtx[row][col] = math.trunc(mtx[row][col]) + mu
        
        return mtx
    
    
    def get_radial_gradient_clean(self, size : int) -> List[List[float]]:
        mtx = np.zeros((size, size), np.float64)
        
        center_x = size / 2
        center_y = center_x
        
        for x in range(size):
            for y in range(size):
                mtx[x][y] = math.sqrt(pow(abs(center_x - x), 2) + pow(abs(center_y - y), 2))
        
        return mtx
    
    
    def get_radial_gradient_noise(self, size : int) -> List[List[float]]:
        mtx = self.get_radial_gradient_clean(size)
        
        for x in range(size):
            for y in range(size):
                mtx[x][y] += random.random()
        
        return mtx
        
        
                